# ros2 run path_planner executor --ros-args -p goal_x:=1.0 -p goal_y:=-.5

from .imports import *

def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_pose_xytheta(tf: TransformStamped):
    t = tf.transform.translation
    r = tf.transform.rotation
    return t.x, t.y, yaw_from_quat(r)

class NavigationExecutor(Node):
    def __init__(self):
        super().__init__('executor')
        self.robot_radius_px = None   

        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 0.0)

        self.cmd_topic = '/cmd_vel'
        self.target_frame = 'map'
        self.base_frame = 'base_link'
        self.goal_tol_xy = 0.15
        self.v_max = 0.15
        self.w_max = 0.8
        self.control_rate = 20.0
        
        self.rx, self.ry, self.rth = 0., 0., 0.
        self.old_rx, self.old_ry = 100., 100.
        self.stuck_count = 0
        self.prev_stuck = False

        self.tf_buffer: Buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.planner: Planner = AStar(self, occ_threshold=50, connect8=True)
        self.current_path: Path | None = None
        self.goal_index: int = 0
        self.occ_map: OccupancyGrid | None = None
        self.scan_ranges = None
        self.twist = None
        self.need_replan: bool = False
        self.goal_reached = False
        self.vis_topic = '/visualization_marker_array'
        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)
        self.vis_pub = self.create_publisher(MarkerArray, self.vis_topic, 10)
        self.path_pub = self.create_publisher(
            Path,
            '/planned_path',
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=1
            )
        )

        self.map_sub = self.create_subscription(
            OccupancyGrid, '/map', self.occ_callback,
            QoSProfile(
                reliability=ReliabilityPolicy.RELIABLE,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                history=HistoryPolicy.KEEP_LAST,
                depth=5
            )
        )
        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)
            
        self.cmd_sub = self.create_subscription(
            Twist,
            '/cmd_vel',
            self.cmd_callback,
            qos_profile=qos_profile_sensor_data
        )
        self.robot_pose_cached = None



        self.replan_count = 0
        self.replan_stuck_limit = 5
        self.escape_active = False
        self._last_progress_pose = None
        self._last_goal_index = 0
        self._progress_move_eps = 0.05  # meters
        self.escape_active = False
        self.backoff_until = None
        self.backoff_linear = 0.0
        self.backoff_angular = 0.0

        # recovery state
        self.recovery_mode = 'none'          # 'none' | 'backoff' | 'reorient'
        self.recovery_deadline = None
        self.reorient_deadline = None
        self.reorient_target_yaw = None

        # knobs
        self.backoff_sec = 1.5               
        self.backoff_linear = -0.06          
        self.backoff_angular = 0.0

        self.reorient_max_sec = 2.5          # max time allowed to re-orient
        self.reorient_yaw_tol = math.radians(12.0)  # done when |yaw error| < 12°
        self.reorient_w_max = 0.8            # max turn rate during reorientation

        self.stop_dist = 0.16


        self.blocked_streak = 0
        self.blocked_streak_thresh = 3      # how many consecutive frames before acting
        self.lookahead_skip = 4             # how many waypoints to probe ahead for a free one
        self.replan_cooldown = 0.7          # seconds
        self.next_allowed_replan = self.get_clock().now()

        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
    def _angle_normalize(self, a):
        return (a + math.pi) % (2.0 * math.pi) - math.pi

    def _avg_heading_to_next_k(self, k=3):
        """Average **direction** (unit vector) from current pose to next k waypoints."""
        if self.current_path is None or not self.current_path.poses:
            return None
        i0 = max(self.goal_index, 0)
        i1 = min(len(self.current_path.poses), i0 + k)

        sx, sy = 0.0, 0.0
        count = 0
        for i in range(i0, i1):
            wp = self.current_path.poses[i].pose.position
            dx = wp.x - self.rx
            dy = wp.y - self.ry
            n = math.hypot(dx, dy)
            if n > 1e-6:
                sx += dx / n
                sy += dy / n
                count += 1

        if count == 0:
            return None
        # average unit vector -> heading
        return math.atan2(sy, sx)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.has_scan_received = True
        
    def cmd_callback(self, msg):
        self.twist = msg

    def _is_wp_free(self, wp):
        mc = self.planner.world_to_map(wp.x, wp.y)
        return (mc is not None) and self.planner.is_free(mc[0], mc[1])

    def _first_free_ahead(self, start_idx, max_ahead):
        poses = self.current_path.poses
        for j in range(start_idx+1, min(len(poses), start_idx+1+max_ahead)):
            if self._is_wp_free(poses[j].pose.position):
                return j
        return None

    def _micro_detour_target(self, i):
        """Try small lateral offsets around path segment (left/right) and forward step."""
        poses = self.current_path.poses
        if i+1 >= len(poses): return None
        p0 = poses[i].pose.position
        p1 = poses[i+1].pose.position
        dx, dy = (p1.x - p0.x), (p1.y - p0.y)
        L = math.hypot(dx, dy) or 1.0
        nx, ny = -dy/L, dx/L  # perpendicular
        # try a few candidate offsets
        candidates = [
            (p0.x + 0.15*nx, p0.y + 0.15*ny),
            (p0.x - 0.15*nx, p0.y - 0.15*ny),
            (p0.x + 0.25*nx, p0.y + 0.25*ny),
            (p0.x - 0.25*nx, p0.y - 0.25*ny),
            (p0.x + 0.20*dx/L, p0.y + 0.20*dy/L),  # tiny forward nudge
        ]
        for (cx, cy) in candidates:
            mc = self.planner.world_to_map(cx, cy)
            if (mc is not None) and self.planner.is_free(mc[0], mc[1]):
                return cx, cy
        return None

    def pub_path_vis(self, path):
        # goal position (green sphere)
        marker_array = MarkerArray() 
        gx = self.get_parameter('goal_x').get_parameter_value().double_value
        gy = self.get_parameter('goal_y').get_parameter_value().double_value
        
        print(gx, gy)

        goal = Marker()
        goal.header.frame_id = 'map'
        goal.header.stamp = self.get_clock().now().to_msg()
        goal.type = goal.SPHERE
        goal.id = 100000  # unique id
        goal.action = goal.ADD
        goal.scale.x = 0.10
        goal.scale.y = 0.10
        goal.scale.z = 0.1
        
        goal.color.r = 0.0
        goal.color.g = 1.0
        goal.color.b = 0.0
        goal.color.a = 1.0
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        marker_array.markers.append(goal)

        if path is None: 
            self.vis_pub.publish(marker_array)
            return       

        i = 0
        for node in path.poses:
            marker = Marker()
            marker.header.frame_id = 'map'
            marker.header.stamp = self.get_clock().now().to_msg()
            
            marker.type = marker.CUBE
            marker.id = i
            marker.action = marker.ADD
            
            marker.scale.x = 0.05
            marker.scale.y = 0.05
            marker.scale.z = 0.05
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.pose.position.x = node.pose.position.x
            marker.pose.position.y = node.pose.position.y
            
            marker_array.markers.append(marker)
            i += 1        
        
        self.vis_pub.publish(marker_array)

    def occ_callback(self, msg: OccupancyGrid):

        if self.goal_reached:
            return
        res = msg.info.resolution

        first = self.occ_map is None
        self.robot_radius_px = 4
        morpher_class = Morpher(msg, self.robot_radius_px)
        msg_dilated = morpher_class.dilate()
        self.occ_map = msg_dilated 
        w, h = msg.info.width, msg.info.height
        raw_map = [list(msg.data[i*w:(i+1)*w]) for i in range(h)]

        dw, dh = msg_dilated.info.width, msg_dilated.info.height
        dilated_map = [list(msg_dilated.data[i*dw:(i+1)*dw]) for i in range(dh)]

        save_map_morphology(raw_map, dilated_map)


        if first:
            self.planner.set_map(msg)
        else:
            self.planner.update_map(self.occ_map)


        if first:
            self.get_logger().info(
                f"Map received: {msg.info.width}x{msg.info.height} @ {msg.info.resolution} m "
                f"origin=({msg.info.origin.position.x:.2f},{msg.info.origin.position.y:.2f}) "
                f"frame={msg.header.frame_id}"
            )

        if self.current_path:
            for i in range(self.goal_index, len(self.current_path.poses)):
                p = self.current_path.poses[i].pose.position
                map_coords = self.planner.world_to_map(p.x, p.y)
                if map_coords is None or not self.planner.is_free(map_coords[0], map_coords[1]):
                    print("\033[31mwarning: next waypoints not free -> replan\033[0m")
                    self.need_replan = True
                    break

    def get_robot_pose(self):
        t_req = rclpy.time.Time()
        try:
            for bf in (self.base_frame, 'base_footprint', 'base_link'):
                if self.tf_buffer.can_transform(self.target_frame, bf, rclpy.time.Time(),
                                                timeout=rclpy.duration.Duration(seconds=0.1)):
                    tf = self.tf_buffer.lookup_transform(self.target_frame, bf, rclpy.time.Time())
                    return transform_pose_xytheta(tf)
            self.get_logger().warn(f"TF not ready: {self.target_frame} -> {self.base_frame}", throttle_duration_sec=1.0)
            print("self.target_frame, self.base_frame, t_req", self.target_frame, self.base_frame, t_req)
                # todo check what robot actually publishes: ros2 topic list | grep tf
                # does it publish base_link or base_footprint?
            print("stored in buffer", self.tf_buffer.all_frames_as_string())
            return None
        except Exception as e:
            self.get_logger().warn(f"TF lookup failed: {e}", throttle_duration_sec=1.0)
            return None


    def _pose_cb(self, msg: PoseWithCovarianceStamped):
        p = msg.pose.pose
        self.robot_pose_cached = (p.position.x, p.position.y, yaw_from_quat(p.orientation))

    def control_loop(self):

        ############ recovery ###################
        now = self.get_clock().now()

        if self.recovery_mode == 'backoff':
            if now < self.recovery_deadline:
                cmd = Twist()
                cmd.linear.x  = self.backoff_linear
                cmd.angular.z = self.backoff_angular
                self.cmd_pub.publish(cmd)
                return
            else:
                self.stop_robot()

                # compute desired yaw toward average of next 3 nodes
                tgt = self._avg_heading_to_next_k(k=3)
                if tgt is None:
                    self.recovery_mode = 'none'
                    return

                self.reorient_target_yaw = tgt
                self.reorient_deadline = now + rclpy.duration.Duration(seconds=self.reorient_max_sec)
                self.recovery_mode = 'reorient'

        if self.recovery_mode == 'reorient':
            rx, ry, rth = self.rx, self.ry, self.rth
            dth = self._angle_normalize(self.reorient_target_yaw - rth)

            if abs(dth) <= self.reorient_yaw_tol or now >= self.reorient_deadline:
                self.stop_robot()
                self.recovery_mode = 'none'
                self.need_replan = True  # try a fresh plan after turning
                return
            else:
                cmd = Twist()
                kP = 1.0
                cmd.angular.z = max(-self.reorient_w_max, min(self.reorient_w_max, kP * dth))
                cmd.linear.x = 0.0
                self.cmd_pub.publish(cmd)
                return
        #################################################################################


        if self.occ_map is None:
            print("if self.occ_map is None: true")
            return
        now = self.get_clock().now()
        if self.backoff_until is not None:
            if now < self.backoff_until:
                cmd = Twist()
                cmd.linear.x  = self.backoff_linear
                cmd.angular.z = self.backoff_angular
                self.cmd_pub.publish(cmd)   
                return                      # skip planning while backing up
            else:
                self.backoff_until = None
                self.stop_robot()
        if self.scan_ranges is None:
            return
            
        pose = self.get_robot_pose() or self.robot_pose_cached
        
        if pose is None:
            print("pose is none")
            return
            
        self.rx, self.ry, self.rth = pose


        angle_idx = 20

        # laser scan check
        obstacle_distance_front = min(
            min(self.scan_ranges[0:angle_idx]),
            min(self.scan_ranges[-angle_idx:])
        )
        
        sm = self.planner.world_to_map(self.rx, self.ry)
        if not self.planner.is_free(*sm):
        	print("currently inside walls")
        if obstacle_distance_front < self.stop_dist or not self.planner.is_free(*sm):
            self.recovery_mode = 'backoff'
            self.recovery_deadline = self.get_clock().now() + rclpy.duration.Duration(seconds=self.backoff_sec)
            self.need_replan = True
            self.replan_count += 1

            cmd = Twist()
            cmd.linear.x  = self.backoff_linear
            cmd.angular.z = self.backoff_angular
            self.cmd_pub.publish(cmd)
            return

            
        if self.twist is not None:
            if self.twist.linear.x > 0.1 and math.hypot(self.old_rx - self.rx, self.old_ry - self.ry) < 0.00001:
                print("possibly stuck")
                self.stuck_count += 1
                self.prev_stuck = True
            else:
                self.prev_stuck = False
        self.old_rx = self.rx
        self.old_ry = self.ry
        
        if self.prev_stuck:
            if self.stuck_count > 3:
                self.stop_robot()
                twist = Twist()
                twist.linear.x = -0.05                                
                self.cmd_pub.publish(twist)
                time.sleep(2)
                self.stop_robot()

                self.need_replan = True
        else:
            self.stuck_count = 0

        gx = self.get_parameter('goal_x').get_parameter_value().double_value
        gy = self.get_parameter('goal_y').get_parameter_value().double_value

        if self.current_path is None or self.goal_index >= len(self.current_path.poses) or self.need_replan:
            path = self.planner.plan((self.rx, self.ry), (gx, gy))
            self.pub_path_vis(path)
            #print("neuer path", path)
            if path is None or not path.poses:
                self.get_logger().warn("Planner failed: no path (start/goal blocked or outside map)")
                self.stop_robot()
                self.need_replan = False
                return
            self.current_path = path
            self.goal_index = 0
            self.need_replan = False
            self.path_pub.publish(self.current_path)
            self.get_logger().info(f"Planned path with {len(path.poses)} poses")


        if math.hypot(self.rx - gx, self.ry - gy) <= self.goal_tol_xy:
            """### am ende goal angle matchen###
            desired_yaw = math.atan2(gy-ry, gx-rx)
            dth = (desired_yaw - rth + math.pi) % (2*math.pi) - math.pi
            if abs(dth) > math.radians(10):  
                cmd = Twist()
                cmd.angular.z = max(min(dth, self.w_max), -self.w_max)
                self.cmd_pub.publish(cmd); return
            #######################################"""
            if not self.goal_reached:
                self.goal_reached = True
                self.current_path = None
                self.stop_robot()
                self.get_logger().info("Goal reached. Stopping.") 
            return
            
        poses = self.current_path.poses
        sm = self.planner.world_to_map(self.rx, self.ry)
        gm = self.planner.world_to_map(gx, gy)
        self.get_logger().info(
            f"sm={sm} gm={gm} "
            f"start_free={(sm and self.planner.is_free(*sm))} "
            f"goal_free={(gm and self.planner.is_free(*gm))}",
            throttle_duration_sec=1.0
        )

        if not poses or self.goal_index >= len(poses):
            self.stop_robot()
            self.current_path = None
            return

        next_wp = poses[self.goal_index].pose.position
        map_coords = self.planner.world_to_map(next_wp.x, next_wp.y)
        if map_coords is None or not self.planner.is_free(map_coords[0], map_coords[1]):
            self.need_replan = True
            return

        dx, dy = next_wp.x - self.rx, next_wp.y - self.ry
        dist_wp = math.hypot(dx, dy)

        if dist_wp <= self.goal_tol_xy:
            self.goal_index += 1
            return

        desired_yaw = math.atan2(dy, dx)
        dth = (desired_yaw - self.rth + math.pi) % (2 * math.pi) - math.pi
        heading_scale = max(0.0, math.cos(dth))
        v_cmd = min(self.v_max, dist_wp) * heading_scale
        w_cmd = max(min(dth, self.w_max), -self.w_max)
        #print(v_cmd, w_cmd)
        cmd = Twist()
        cmd.linear.x = max(min(v_cmd, self.v_max), -self.v_max)
        cmd.angular.z = w_cmd
        self.get_logger().info(
            f"wp={self.goal_index} dist={dist_wp:.2f} dth={dth:.2f} v={v_cmd:.2f} w={w_cmd:.2f}",
            throttle_duration_sec=1.0
        )
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())
def save_map_morphology(raw_map, dilated_map, filename=None):
    import os, time, errno
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    def to_img(grid):
        g = np.asarray(grid, dtype=np.int16)

        unknown = (g == -1) | (g == 255)
        free    = (g == 0) | ((g >= 0) & (g < 50))
        occ     = (g >= 50) | (g == 254) | (g == 253)

        img = np.empty(g.shape, dtype=np.uint8)
        img[unknown] = 128
        img[free]    = 255
        img[occ & ~unknown] = 0
        return img

    if filename is None:
        filename = "map_morphology_latest.png"

    script_dir = os.path.dirname(os.path.abspath(__file__))
    out_path_primary = os.path.join(script_dir, filename)

    fallback_dir = os.path.join(os.path.expanduser("~"), ".ros")
    os.makedirs(fallback_dir, exist_ok=True)
    out_path_fallback = os.path.join(fallback_dir, filename)

    def _save(to_path):
        os.makedirs(os.path.dirname(to_path), exist_ok=True)
        fig, axs = plt.subplots(1, 2, figsize=(8, 4))
        for ax, (title, m) in zip(
            axs,
            [("Raw map", raw_map), ("Dilated map", dilated_map)]
        ):
            ax.imshow(to_img(m), cmap="gray", origin="lower")
            ax.set_title(title)
            ax.axis("off")
        plt.tight_layout()
        plt.savefig(to_path, dpi=300, bbox_inches="tight")
        plt.close(fig)
        print(f"Map morphology saved to: {to_path}")
        return to_path

    try:
        return _save(out_path_primary)
    except Exception as e:
        print(f"[save_map_morphology] Primary path failed ({out_path_primary}): {e}")
        return _save(out_path_fallback)




def main():
    rclpy.init()
    node = NavigationExecutor()
    rclpy.spin(node)
    rclpy.shutdown()

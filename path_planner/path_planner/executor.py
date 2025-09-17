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
        self.occ_map_full_dilated: OccupancyGrid | None = None
        self.occ_map_plan: OccupancyGrid | None = None
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

        self.create_subscription(PoseWithCovarianceStamped, '/slam_toolbox/pose', self._pose_cb, 10)


        self.replan_count = 0
        self.replan_stuck_limit = 5
        self.escape_active = False
        self._last_progress_pose = None
        self._last_goal_index = 0
        self._progress_move_eps = 0.05  # meters
        self.escape_active = False



        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges

        
    def cmd_callback(self, msg):
        self.twist = msg


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

        radius_px = 5  # max(1, int(math.ceil(self.robot_radius / res)))

        first = self.occ_map is None
        morpher_class = Morpher(msg, radius_px)
        msg_dilated = morpher_class.dilate()

        self.occ_map = msg_dilated 
        self.robot_radius_px = 4


        #  remove ONLY dilation around current start
        msg_for_plan = OccupancyGrid()
        msg_for_plan.header = msg_dilated.header
        msg_for_plan.info = msg_dilated.info
        msg_for_plan.data = list(msg_dilated.data)

        pose = self.get_robot_pose() or self.robot_pose_cached
        if pose is not None:
            rx, ry, _ = pose

            def world_to_map_raw(wx, wy, info):
                mx = int((wx - info.origin.position.x) / info.resolution)
                my = int((wy - info.origin.position.y) / info.resolution)
                if 0 <= mx < info.width and 0 <= my < info.height:
                    return mx, my
                return None

            sm = world_to_map_raw(rx, ry, msg.info)
            if sm is not None:
                mx, my = sm
                w, h = msg.info.width, msg.info.height
                occ_thr = getattr(self.planner, 'occ_threshold', 50)

                def idx(x, y): return y * w + x

                raw_occ = msg.data[idx(mx, my)]
                dil_occ = msg_dilated.data[idx(mx, my)]

                # if raw is free and dilation blocked the start, carve a small bubble
                if (raw_occ >= 0 and raw_occ < occ_thr) and (dil_occ >= occ_thr): # cells that are free in raw map but got filled only by dilation
                    bubble_r = max(1, int(self.robot_radius_px or 4)) # define bounding box of size 2 * bubble_r + 1, fallback: 4
                    r2 = bubble_r * bubble_r
                    ymin = max(0, my - bubble_r)
                    ymax = min(h, my + bubble_r + 1)
                    xmin = max(0, mx - bubble_r)
                    xmax = min(w, mx + bubble_r + 1)

                    for yy in range(ymin, ymax):
                        dy = yy - my
                        for xx in range(xmin, xmax):
                            dx = xx - mx
                            if dx*dx + dy*dy <= r2:
                                # Clear ONLY dilation: raw free & dilated occupied
                                if (msg.data[idx(xx, yy)] >= 0 and msg.data[idx(xx, yy)] < occ_thr) and \
                                (msg_for_plan.data[idx(xx, yy)] >= occ_thr):
                                    msg_for_plan.data[idx(xx, yy)] = 0

        
        self.occ_map_full_dilated = msg_dilated
        self.occ_map_plan = msg_for_plan
        self.occ_map = self.occ_map_full_dilated
        self.robot_radius_px = radius_px

        if first:
            self.planner.set_map(self.occ_map_full_dilated)
        else:
            self.planner.update_map(self.occ_map_full_dilated)

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
                    self.replan_count += 1
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



    # add to NavigationExecutor class
    def _dist2(self, a, b):
        ax, ay = a; bx, by = b
        dx = ax - bx; dy = ay - by
        return dx*dx + dy*dy

    def _nearest_free_cell(self, mx, my, max_r=20):
        grid = self.occ_map_plan if self.occ_map_plan is not None else self.planner.map
        if grid is None:
            return None

        w, h = grid.info.width, grid.info.height
        res = grid.info.resolution
        def idx(x, y): return y * w + x

        def is_free(x, y):
            if not (0 <= x < w and 0 <= y < h): 
                return False
            v = grid.data[idx(x, y)]
            return (v == -1) or (0 <= v < self.planner.occ_threshold)

        # Require the escape target to be at least this far away in meters
        min_world_dist = 0.15  # 15 cm
        def far_enough(x, y):
            wx, wy = self.planner.map_to_world(x, y)
            rx, ry, _ = (self.get_robot_pose() or self.robot_pose_cached or (None, None, None))
            if rx is None:
                return True  # fallback if pose missing
            return math.hypot(wx - rx, wy - ry) >= min_world_dist

        # Do NOT return the current cell; find something actually away from the robot
        for r in range(1, max_r + 1):
            # vertical segments of the ring
            for yy in range(my - r, my + r + 1):
                for xx in (mx - r, mx + r):
                    if is_free(xx, yy) and far_enough(xx, yy):
                        return (xx, yy)
            # horizontal segments of the ring
            for xx in range(mx - r + 1, mx + r):
                for yy in (my - r, my + r):
                    if is_free(xx, yy) and far_enough(xx, yy):
                        return (xx, yy)
        return None


    def _start_escape_to_cell(self, cell):
        # Keep planner & checks on planning map during escape
        if self.occ_map_plan is not None:
            self.planner.update_map(self.occ_map_plan)
        wx, wy = self.planner.map_to_world(cell[0], cell[1])
        p = Path()
        p.header.frame_id = self.planner.frame_id
        p.header.stamp = self.get_clock().now().to_msg()
        ps = PoseStamped()
        ps.header = p.header
        ps.pose.position.x = wx
        ps.pose.position.y = wy
        ps.pose.orientation.w = 1.0
        p.poses.append(ps)
        self.current_path = p
        self.goal_index = 0
        self.path_pub.publish(self.current_path)
        self.escape_active = True
        self.get_logger().info(f"Escape: driving to nearest free cell at ({wx:.2f},{wy:.2f})")





    def control_loop(self):
        if self.occ_map is None:
            print("if self.occ_map is None: true")
            return
            
        if self.scan_ranges is None:
            return
            
        pose = self.get_robot_pose() or self.robot_pose_cached
        
        if pose is None:
            print("pose is none")
            return
            
        self.rx, self.ry, self.rth = pose

        if self._last_progress_pose is None:
            self._last_progress_pose = (rx, ry)
            self._last_goal_index = self.goal_index
        else:
            moved = self._dist2((rx, ry), self._last_progress_pose) > (self._progress_move_eps ** 2)
            advanced = self.goal_index > self._last_goal_index
            if moved or advanced:
                self.replan_count = 0
                self._last_progress_pose = (rx, ry)
                self._last_goal_index = self.goal_index

        # reach escape target?
        if self.current_path is None: print("debug: ", self.current_path)
        if self.escape_active and math.hypot(rx - self.current_path.poses[-1].pose.position.x,
                                            ry - self.current_path.poses[-1].pose.position.y) <= self.goal_tol_xy:
            self.escape_active = False
            self.current_path = None
            self.need_replan = True
            return


        angle_idx = 20

        # laser scan check
        obstacle_distance_front = min(
            min(self.scan_ranges[0:angle_idx]),
            min(self.scan_ranges[-angle_idx:])
        )

        if obstacle_distance_front < .16 and not self.escape_active:

        self.stop_robot()
        sm = self.world_to_map(self.rx, self.ry)
        if !self.is_free(*sm):
        	print("currently inside walls")
        if obstacle_distance_front < .16 or !self.is_free(*sm):
            print("\033[31mwarning: laser scan closer than 15.5cm\033[0m")

            print("\033[31mwarning: laser scan closer than 15.5cm\033[0m")
            self.need_replan = True
            self.replan_count += 1

            
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



        if (self.current_path is None or self.goal_index >= len(self.current_path.poses) or self.need_replan) and not self.escape_active:


            # stuck in replanning loop
            if self.replan_count > self.replan_stuck_limit and not self.escape_active:
                print("\033[31mstuck in replan loop: drive to next free cell and replan\033[0m")
                sm = self.planner.world_to_map(self.rx, self.ry)
                nf = self._nearest_free_cell(sm[0], sm[1], max_r=12) if sm is not None else None
                if nf is not None:
                    self._start_escape_to_cell(nf)
                    self.need_replan = False
                    self.replan_count = 0
                    return  # drive escape first
                else:
                    self.get_logger().warn("Escape: no nearby free cell found")



            # temporarily use planning map (dilation cleared near start)
            if hasattr(self, "occ_map_plan") and self.occ_map_plan is not None:
                self.planner.update_map(self.occ_map_plan)

            path = self.planner.plan((self.rx, self.ry), (gx, gy))

            # restore full dilation for driving
            if hasattr(self, "occ_map_full_dilated") and self.occ_map_full_dilated is not None:
                self.planner.update_map(self.occ_map_full_dilated)


            self.pub_path_vis(path)
            if path is None or not path.poses:
                self.get_logger().warn("Planner failed: no path (start/goal blocked or outside map)")
                self.stop_robot()
                # --- increment replan_count on hard plan failure so escape logic can trigger ---
                if not self.escape_active:
                    self.replan_count += 1
                    # if we keep failing immediately, try escape right away
                    if self.replan_count > self.replan_stuck_limit:
                        sm = self.planner.world_to_map(rx, ry)
                        nf = self._nearest_free_cell(sm[0], sm[1], max_r=12) if sm is not None else None
                        if nf is not None:
                            self._start_escape_to_cell(nf)
                            self.need_replan = False
                            self.replan_count = 0
                            return
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
                self.get_logger().info("\033[92mGoal reached. Stopping.\033[0m")

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
            if not self.escape_active:
                self.need_replan = True
                self.replan_count += 1

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


def main():
    rclpy.init()
    node = NavigationExecutor()
    rclpy.spin(node)
    rclpy.shutdown()

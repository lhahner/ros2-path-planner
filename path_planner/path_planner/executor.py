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
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.has_scan_received = True
        
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

        first = self.occ_map is None
        morpher_class = Morpher(msg, radius_px)
        msg_dilated = morpher_class.dilate()
        self.occ_map = msg_dilated 
        self.robot_radius_px = 4


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


        angle_idx = 20

        # laser scan check
        obstacle_distance_front = min(
            min(self.scan_ranges[0:angle_idx]),
            min(self.scan_ranges[-angle_idx:])
        )
        
        sm = self.world_to_map(self.rx, self.ry)
        if !self.is_free(*sm):
        	print("currently inside walls")
        if obstacle_distance_front < .16 or !self.is_free(*sm):
            print("\033[31mwarning: laser scan closer than 15.5cm\033[0m")
            self.stop_robot()
            twist = Twist()
            rand = np.random.rand(1)
            if rand < 0.33:
                twist.linear.x = -0.05
            elif rand < 0.66:
                twist.angular.z = -0.2
            else:
                twist.angular.z = 0.2            
            self.cmd_pub.publish(twist)
            time.sleep(2)
            self.stop_robot()

            self.need_replan = True
            
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

def main():
    rclpy.init()
    node = NavigationExecutor()
    rclpy.spin(node)
    rclpy.shutdown()

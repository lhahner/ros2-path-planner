

# example call: 
# ros2 run nav_exec navigation_executor   --ros-args   -p target_frame:=odom   -p base_frame:=base_footprint   -p test_path_type:=zigzag   -p path_scale:=.8   -p path_resolution:=0.03
# see more options for test_path_type in class PathLibrary



import math
from typing import Optional
from geometry_msgs.msg import PoseStamped
import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from nav_msgs.msg import Path
import tf2_ros
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)

def transform_pose_xytheta(tf: TransformStamped):
    t = tf.transform.translation
    r = tf.transform.rotation
    return t.x, t.y, yaw_from_quat(r)
class PathLibrary:
    def __init__(self, node: Node):
        self.node = node

    def _add_pose(self, path: Path, x: float, y: float):
        ps = PoseStamped()
        ps.header.frame_id = path.header.frame_id
        ps.header.stamp = path.header.stamp
        ps.pose.position.x = x
        ps.pose.position.y = y
        ps.pose.orientation.w = 1.0
        path.poses.append(ps)

    def build(self, frame_id: str, shape: str, scale: float, res: float, ox: float, oy: float) -> Path:
        path = Path()
        path.header.frame_id = frame_id
        path.header.stamp = self.node.get_clock().now().to_msg()
        res = max(1e-4, res)

        if shape == 'line':
            L = max(0.5, 1.0 * scale)
            steps = max(2, int(L / res))
            for i in range(steps + 1):
                self._add_pose(path, ox + (i / steps) * L, oy)

        elif shape == 'square':
            S = max(0.5, 1.0 * scale)
            n = max(1, int(S / res))
            for i in range(n + 1): self._add_pose(path, ox + i*(S/n), oy)
            for i in range(1, n + 1): self._add_pose(path, ox + S, oy + i*(S/n))
            for i in range(1, n + 1): self._add_pose(path, ox + S - i*(S/n), oy + S)
            for i in range(1, n): self._add_pose(path, ox, oy + S - i*(S/n))

        elif shape == 'circle':
            R = max(0.25, 0.5 * scale)
            th = 0.0
            while th <= 2*math.pi + 1e-6:
                self._add_pose(path, ox + R*math.cos(th), oy + R*math.sin(th))
                th += res  # res as rad step

        elif shape == 'figure8':
            a = max(0.25, 0.5 * scale)
            b = max(0.25, 0.35 * scale)
            t = 0.0
            while t <= 2*math.pi + 1e-6:
                self._add_pose(path, ox + a*math.sin(t), oy + b*math.sin(2*t))
                t += res  # param step

        elif shape == 's_curve':
            L = max(0.8, 1.2 * scale)
            A = 0.4 * scale
            k = 2.0 * math.pi / max(0.8, 1.0 * scale)
            n = max(2, int(L / res))
            for i in range(n + 1):
                x = ox + (i / n) * L
                y = oy + A * math.sin(k * (x - ox))
                self._add_pose(path, x, y)

        elif shape == 'zigzag':
            seg_len = max(0.2, 0.3 * scale)
            amp = 0.4 * scale
            segments = 8
            x, y, up = ox, oy, True
            pts_per = max(1, int(seg_len / res))
            self._add_pose(path, x, y)
            for _ in range(segments):
                xn = x + seg_len
                yn = oy + (amp if up else -amp)
                for i in range(1, pts_per + 1):
                    xi = x + (i/pts_per)*(xn - x)
                    yi = y + (i/pts_per)*(yn - y)
                    self._add_pose(path, xi, yi)
                x, y, up = xn, yn, not up

        elif shape == 'lemniscate':
            a = max(0.3, 0.5 * scale)
            t = -math.pi/4
            t_end = math.pi/4
            while t <= t_end + 1e-6:
                r = (2*(a**2)*math.cos(2*t))**0.5
                self._add_pose(path, ox + r*math.cos(t), oy + r*math.sin(t))
                t += res
            t = 3*math.pi/4
            t_end = 5*math.pi/4
            while t <= t_end + 1e-6:
                r = (2*(a**2)*math.cos(2*t))**0.5
                self._add_pose(path, ox + r*math.cos(t), oy + r*math.sin(t))
                t += res

        else:
            L = max(0.5, 1.0 * scale)
            steps = max(2, int(L / res))
            for i in range(steps + 1):
                self._add_pose(path, ox + (i / steps) * L, oy)

        return path



class NavigationExecutor(Node):
    def __init__(self):
        super().__init__('navigation_executor')
        # self.declare_parameter('my_parameter', 'world') of the constructor creates 
        # a parameter with the name my_parameter and a default value of world
        self.declare_parameter('path_topic', '/planned_path')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('target_frame', 'odom')       # or 'map'
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lookahead', 0.5)             
        self.declare_parameter('goal_tolerance_xy', 0.15)    
        self.declare_parameter('goal_tolerance_yaw', 0.25)   # rad
        self.declare_parameter('v_max', 0.1)                # m/s
        self.declare_parameter('w_max', 0.75)                 # rad/s
        self.declare_parameter('k_v', 1.0)                   # scale lin
        self.declare_parameter('k_w', 2.0)                   # scale ang
        self.declare_parameter('control_rate', 20.0)         # Hz
        ############ path test declarations#####

        self.declare_parameter('test_path_type', 'figure8')
        self.declare_parameter('path_scale', 1.0)
        self.declare_parameter('path_resolution', 0.05)
        self.declare_parameter('origin_x', 0.0)
        self.declare_parameter('origin_y', 0.0)
        ##########################################
        self.path_topic = self.get_parameter('path_topic').get_parameter_value().string_value
        
        self.cmd_topic = self.get_parameter('cmd_topic').get_parameter_value().string_value
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.lookahead = self.get_parameter('lookahead').get_parameter_value().double_value
        self.goal_tol_xy = self.get_parameter('goal_tolerance_xy').get_parameter_value().double_value
        self.goal_tol_yaw = self.get_parameter('goal_tolerance_yaw').get_parameter_value().double_value
        self.v_max = self.get_parameter('v_max').get_parameter_value().double_value
        self.w_max = self.get_parameter('w_max').get_parameter_value().double_value
        self.k_v = self.get_parameter('k_v').get_parameter_value().double_value
        self.k_w = self.get_parameter('k_w').get_parameter_value().double_value
        self.control_rate = self.get_parameter('control_rate').get_parameter_value().double_value

        self.tf_buffer: Buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)
        #############path test init parameters#####################
        self.test_path_type = self.get_parameter('test_path_type').get_parameter_value().string_value
        self.path_scale = self.get_parameter('path_scale').get_parameter_value().double_value
        self.path_resolution = self.get_parameter('path_resolution').get_parameter_value().double_value
        self.origin_x = self.get_parameter('origin_x').get_parameter_value().double_value
        self.origin_y = self.get_parameter('origin_y').get_parameter_value().double_value



        ###########################################################
       
        lib = PathLibrary(self)
        self.current_path = lib.build(
            frame_id=self.target_frame,
            shape=self.test_path_type,
            scale=self.path_scale,
            res=self.path_resolution,
            ox=self.origin_x,
            oy=self.origin_y
        )


        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        #self.current_path: Optional[Path] = None
        self.goal_index: int = 0
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)


    def path_cb(self, msg: Path):

        if not msg.poses:
            self.get_logger().warn("Received empty path")
            return
        if msg.header.frame_id != self.target_frame:
            self.get_logger().warn(
                f'Path frame "{msg.header.frame_id}" != target_frame "{self.target_frame}"'
            )
        self.get_logger().info(
            f"Received path with {len(msg.poses)} poses (frame: {msg.header.frame_id})"
        )
        if not msg.poses:
            return
        if msg.header.frame_id != self.target_frame:
            self.get_logger().warn(f'Path frame "{msg.header.frame_id}" != target_frame "{self.target_frame}"')
        self.current_path = msg
        self.goal_index = 0

    def get_robot_pose(self):
        try:
            tf = self.tf_buffer.lookup_transform(self.target_frame, self.base_frame, rclpy.time.Time())
            return transform_pose_xytheta(tf)
        except Exception:
            return None

    def find_lookahead_index(self, rx, ry):
        if self.current_path is None:
            return None
        poses = self.current_path.poses
        n = len(poses)
        # advance to nearest segment
        start = self.goal_index
        best_i = start
        best_d = float('inf')
        for i in range(start, n):
            p = poses[i].pose.position
            d = math.hypot(p.x - rx, p.y - ry)
            if d < best_d:
                best_d = d
                best_i = i
        # find lookahead
        for i in range(best_i, n):
            p = poses[i].pose.position
            if math.hypot(p.x - rx, p.y - ry) >= self.lookahead:
                return i
        return n - 1

    def control_loop(self):
        if self.current_path is None:
            return

        pose = self.get_robot_pose()
        if pose is None:
            return
        rx, ry, rth = pose

        poses = self.current_path.poses
        if not poses:
            return

        if self.goal_index >= len(poses):
            self.stop_robot()
            self.current_path = None
            return

        cur_wp = poses[self.goal_index].pose
        tx, ty = cur_wp.position.x, cur_wp.position.y

        # distance to current waypoint
        dx, dy = tx - rx, ty - ry
        dist_wp = math.hypot(dx, dy)

        # if close enough to current waypoint → advance
        if dist_wp <= self.goal_tol_xy:
            self.get_logger().info(
            f"[DEBUG] Reached waypoint {self.goal_index} (tol={self.goal_tol_xy:.3f}), advancing"
        )
            self.goal_index += 1
            if self.goal_index >= len(poses):
                self.stop_robot()
                self.current_path = None
                print("finished path")
            return  # let next timer tick pick up the next waypoint

       
        desired_yaw = math.atan2(dy, dx)
        dth = (desired_yaw - rth + math.pi) % (2*math.pi) - math.pi

        # heading alignment gating
        heading_scale = max(0.0, math.cos(dth))  # 1 when aligned, 0 when 90° off
        v_cmd = self.k_v * min(self.v_max, dist_wp) * heading_scale
        w_cmd = self.k_w * dth

      
        v_cmd = max(min(v_cmd, self.v_max), -self.v_max)
        w_cmd = max(min(w_cmd, self.w_max), -self.w_max)
        """self.get_logger().info(
            f"[DEBUG] heading_scale={heading_scale:.3f} "
            f"v_cmd={v_cmd:.3f} w_cmd={w_cmd:.3f}"
            )"""
        cmd = Twist()
        cmd.linear.x = v_cmd
        cmd.angular.z = w_cmd
        self.cmd_pub.publish(cmd)

    def stop_robot(self):
        #self.get_logger().info("Publishing final stop command (goal reached): v=0.0, w=0.0")
        self.cmd_pub.publish(Twist())

def main():
    rclpy.init()
    node = NavigationExecutor()
    rclpy.spin(node)
  

    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()



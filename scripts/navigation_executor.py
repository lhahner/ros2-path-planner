import math
from typing import Optional

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

class NavigationExecutor(Node):
    def __init__(self):
        super().__init__('navigation_executor')
        # self.declare_parameter('my_parameter', 'world') of the constructor creates 
        # a parameter with the name my_parameter and a default value of world
        self.declare_parameter('path_topic', '/planned_path')
        self.declare_parameter('cmd_topic', '/cmd_vel')
        self.declare_parameter('target_frame', 'odom')       # or 'map'
        self.declare_parameter('base_frame', 'base_link')
        self.declare_parameter('lookahead', 0.5)             # m
        self.declare_parameter('goal_tolerance_xy', 0.15)    # m
        self.declare_parameter('goal_tolerance_yaw', 0.25)   # rad
        self.declare_parameter('v_max', 0.25)                # m/s
        self.declare_parameter('w_max', 1.0)                 # rad/s
        self.declare_parameter('k_v', 1.0)                   # scale lin
        self.declare_parameter('k_w', 2.0)                   # scale ang
        self.declare_parameter('control_rate', 20.0)         # Hz

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

        self.path_sub = self.create_subscription(
            Path,                 # message type
            self.path_topic,      # topic name (string)
            self.path_cb,         # callback when a message arrives
            1                    # queue size
        )

        self.cmd_pub = self.create_publisher(Twist, self.cmd_topic, 10)

        self.current_path: Optional[Path] = None
        self.goal_index: int = 0
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)
        print("finished init")

    def path_cb(self, msg: Path):
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
        idx = self.find_lookahead_index(rx, ry)
        if idx is None:
            return

        goal = self.current_path.poses[-1].pose
        gx, gy = goal.position.x, goal.position.y
        # goal.orientation is a quaternion (x, y, z, w)
        gth = yaw_from_quat(goal.orientation) # yaw angle (heading) in radians

        dxg, dyg = gx - rx, gy - ry
        dist_goal = math.hypot(dxg, dyg)
        if dist_goal <= self.goal_tol_xy:
            dth = (gth - rth + math.pi) % (2*math.pi) - math.pi # shortest angle between 
            #the robot’s facing direction and the goal’s desired facing direction
            if abs(dth) <= self.goal_tol_yaw:
                self.stop_robot()
                self.current_path = None
                return

        target = self.current_path.poses[idx].pose.position
        dx, dy = target.x - rx, target.y - ry
        # transform to robot frame
        c, s = math.cos(-rth), math.sin(-rth)
        x_r = c*dx - s*dy
        y_r = s*dx + c*dy

        if x_r <= 1e-3 and abs(y_r) < 1e-3:
            cmd = Twist()
            self.cmd_pub.publish(cmd)
            return

        curvature = 2.0 * y_r / (self.lookahead**2 + 1e-6)
        v = self.k_v * min(self.v_max, dist_goal)
        w = self.k_w * v * curvature
        v = max(min(v, self.v_max), -self.v_max)
        w = max(min(w, self.w_max), -self.w_max)

        cmd = Twist()
        cmd.linear.x = v
        cmd.angular.z = w
        print(cmd)
        self.cmd_pub.publish(cmd)

        self.goal_index = max(self.goal_index, idx)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

def main():
    rclpy.init()
    node = NavigationExecutor()
    rclpy.spin(node)
    node.stop_robot()
    node.destroy_node()
    rclpy.shutdown()

import math, rclpy
from rclpy.node import Node
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Path
import tf2_ros
from tf2_ros import Buffer, TransformListener
from geometry_msgs.msg import TransformStamped

def yaw_from_quat(q):
    s = 2.0*(q.w*q.z + q.x*q.y); c = 1.0 - 2.0*(q.y*q.y + q.z*q.z)
    return math.atan2(s, c)

class StraightPathPublisher(Node):
    def __init__(self):
        super().__init__('straight_path_publisher')
        self.buf = Buffer(); self.lst = TransformListener(self.buf, self, spin_thread=True)
        self.pub = self.create_publisher(Path, '/planned_path', 1)
        self.timer = self.create_timer(1.0, self.tick)
        self.target_frame = 'odom'
        self.base_frame = 'base_footprint'

        self.len_m, self.step = 1.0, 0.05

    def tick(self):
        try:
            tf: TransformStamped = self.buf.lookup_transform(self.target_frame, self.base_frame, rclpy.time.Time())
        except Exception:
            return
        x = tf.transform.translation.x; y = tf.transform.translation.y
        th = yaw_from_quat(tf.transform.rotation)

        path = Path(); path.header.frame_id = self.target_frame
        path.header.stamp = self.get_clock().now().to_msg()
        for i in range(int(self.len_m/self.step)+1):
            dx = i*self.step*math.cos(th); dy = i*self.step*math.sin(th)
            ps = PoseStamped(); ps.header.frame_id = self.target_frame; ps.header.stamp = path.header.stamp
            ps.pose.position.x = x+dx; ps.pose.position.y = y+dy; ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        self.pub.publish(path)
        self.pub.publish(path)
        self.get_logger().info(f"Published path with {len(path.poses)} points.")


def main():
    rclpy.init(); n = StraightPathPublisher(); rclpy.spin(n); rclpy.shutdown()

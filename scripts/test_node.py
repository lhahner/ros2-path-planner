from geometry_msgs.msg import Twist
import rclpy
from rclpy.node import Node
from rclpy.qos import qos_profile_sensor_data
from rclpy.qos import QoSProfile
from sensor_msgs.msg import LaserScan
from nav_msgs.msg import OccupancyGrid
from nav_msgs.msg import Odometry

import matplotlib.pyplot as plt
import numpy as np


class Turtlebot3ObstacleDetection(Node):

    def __init__(self):
        super().__init__('turtlebot3_obstacle_detection')
        # print('TurtleBot3 Obstacle Detection - Auto Move Enabled')
        # print('----------------------------------------------')
        # print('stop angle: -90 ~ 90 deg')
        # print('stop distance: 0.5 m')
        # print('----------------------------------------------')

        self.scan_ranges = []
        self.has_scan_received = False

        self.stop_distance = 0.5
        self.tele_twist = Twist()
        self.tele_twist.linear.x = 0.0
        self.tele_twist.angular.z = 0.0
        
        self.occ_map = None
        self.pos = None

        qos = QoSProfile(depth=10)

        self.cmd_vel_pub = self.create_publisher(Twist, 'cmd_vel', qos)

        self.scan_sub = self.create_subscription(
            LaserScan,
            'scan',
            self.scan_callback,
            qos_profile=qos_profile_sensor_data)

        self.cmd_vel_raw_sub = self.create_subscription(
            Twist,
            'cmd_vel_raw',
            self.cmd_vel_raw_callback,
            qos_profile=qos_profile_sensor_data)

        self.map_sub = self.create_subscription(
            OccupancyGrid,
            'map',
            self.occ_callback,
            qos_profile=qos_profile_sensor_data
        )
        
        self.tf_sub = self.create_subscription(
            Odometry,
            'odom',
            self.tf_callback,
            qos_profile=qos_profile_sensor_data
        )
        
        self.timer = self.create_timer(0.1, self.timer_callback)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.has_scan_received = True

    def cmd_vel_raw_callback(self, msg):
        self.tele_twist = msg
        
    def occ_callback(self, msg):
        self.occ_map = msg
        
    def tf_callback(self, msg):
        self.pos = msg

    def timer_callback(self):
        if self.has_scan_received:
            self.detect_obstacle()

    def detect_obstacle(self):
        ls_len = len(self.scan_ranges)
        left_range = int(ls_len / 4)  		# 90 degrees ?
        back_range = int(ls_len / 2)            # 180 degress ? 
        right_range = int(ls_len * 3 / 4) 	# 270 degrees ?

        obstacle_distance_front = min(
            min(self.scan_ranges[0:left_range]),
            min(self.scan_ranges[right_range:])
        )
        
        obstacle_distance_left = min(self.scan_ranges[back_range:])
        obstacle_distance_right = min(self.scan_ranges[0:back_range])
                
        twist = Twist()
        if obstacle_distance_front < self.stop_distance:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            #self.get_logger().info('Obstacle detected! Stopping and turning.')
        elif obstacle_distance_front >= self.stop_distance:
            twist.linear.x = 0.0
            twist.angular.z = 0.0
            #self.get_logger().info('Moving.', throttle_duration_sec=2)
            
        if obstacle_distance_left < 0.2:
            self.get_logger().info('obstacle left.')
            
        if obstacle_distance_right < 0.2:
            self.get_logger().info('obstacle right.')
            
            
        
        self.cmd_vel_pub.publish(twist)
        
        if self.occ_map is not None:            
            om = self.occ_map
            
            om_height, om_width = om.info.height, om.info.width
            print(om_height, om_width, om_height*om_width, len(om.data))
            
            #om = np.reshape(om.data, (om_height, om_width))
            
            #plt.imshow(om)
            #plt.show()        
            
            print(self.pos.pose.pose.position)
            print(self.pos.pose.pose.orientation)
            
            o = self.pos.pose.pose.orientation
            
            x, y, z, w = o.x, o.y, o.z, o.w
            
            a = 2 * (w * z + x * y);
            b = 1 - 2 * (y * y + z * z);
            yaw = np.arctan2(a, b)
            print((yaw+np.pi) * (180/np.pi))
            
            return


def main(args=None):
    rclpy.init(args=args)
    turtlebot3_obstacle_detection = Turtlebot3ObstacleDetection()
    rclpy.spin(turtlebot3_obstacle_detection)

    turtlebot3_obstacle_detection.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()

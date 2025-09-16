# ros2 run path_planner A_star_test --ros-args   -p goal_x:=1.0   -p goal_y:=-.5   -p target_frame:=map   -p base_frame:=base_link

import math
import rclpy
import time
from geometry_msgs.msg import PoseStamped, Twist, TransformStamped
from nav_msgs.msg import Path, OccupancyGrid
from rclpy.node import Node
from rclpy.qos import QoSProfile, ReliabilityPolicy, DurabilityPolicy, HistoryPolicy
from tf2_ros import Buffer, TransformListener
from abc import ABC, abstractmethod
import heapq
from geometry_msgs.msg import PoseWithCovarianceStamped
from visualization_msgs.msg import Marker, MarkerArray
import cv2 as cv
import numpy as np
from sensor_msgs.msg import LaserScan
from rclpy.qos import qos_profile_sensor_data
#import matplotlib.pyplot as plt


class Morpher:
    def __init__(self, occupancy_grid, robot_width):
        self.occupancy_grid = occupancy_grid
        self.robot_width = robot_width

    def _grid_to_np(self, msg):
        h, w = msg.info.height, msg.info.width
        a = np.asarray(msg.data, dtype=np.int8).reshape((h, w))
        return a

    def _np_to_grid(self, arr, tmpl):
        out = OccupancyGrid()
        out.header = tmpl.header
        out.info = tmpl.info
        out.data = arr.astype(np.int8).flatten().tolist()
        return out

    def _to_cv_mask(self, a, occ_thresh=75, unknown_as_occ=True):
        if unknown_as_occ:
            a = np.where(a < 0, 100, a)
        mask = (a >= occ_thresh).astype(np.uint8) * 255
        return mask

    def _mask_to_occ(self, mask, unknown_fill=-1):
        # map back to 0/100 and keep unknown as unknown_fill
        occ = (mask > 0).astype(np.int8) * 100
        if unknown_fill != -1:
            return occ
        # restore unknown where original was unknown
        a_orig = self._grid_to_np(self.occupancy_grid)
        unknown = (a_orig < 0)
        occ[unknown] = -1
        return occ

    def dilate(self):
        assert self.occupancy_grid is not None, "grid is empty"
        a = self._grid_to_np(self.occupancy_grid)
        print(a.shape)
        mask = self._to_cv_mask(a, occ_thresh=50, unknown_as_occ=False)
        kernel = np.ones((self.robot_width, self.robot_width), np.uint8)
        dil = cv.dilate(mask, kernel, iterations=1)
        occ_arr = self._mask_to_occ(dil, unknown_fill=-1)
             
        #plt.imshow(occ_arr)
        #plt.savefig("test.png")
       # plt.close()
        return self._np_to_grid(occ_arr, self.occupancy_grid)

class GridCommon:
    def __init__(self, node, occ_threshold=50, connect8=True):
        self.node = node
        self.map: OccupancyGrid | None = None
        self.width = 0
        self.height = 0
        self.res = 0.0
        self.ox = 0.0
        self.oy = 0.0
        self.frame_id = 'map'
        self.occ_threshold = occ_threshold
        self.connect8 = connect8

    def _is_free_value(self, v: int) -> bool:
        return (v == -1) or (0 <= v < self.occ_threshold)

    def set_map(self, grid: OccupancyGrid):
        self.map = grid
        self.width = grid.info.width
        self.height = grid.info.height
        self.res = grid.info.resolution
        self.ox = grid.info.origin.position.x
        self.oy = grid.info.origin.position.y
        self.frame_id = grid.header.frame_id

    def world_to_map(self, x: float, y: float):
        mx = int((x - self.ox) / self.res)
        my = int((y - self.oy) / self.res)
        if 0 <= mx < self.width and 0 <= my < self.height:
            return (mx, my)
        return None

    def map_to_world(self, mx: int, my: int):
        return (self.ox + (mx + 0.5) * self.res, self.oy + (my + 0.5) * self.res)

    def is_free(self, mx: int, my: int):
        idx = my * self.width + mx
        v = self.map.data[idx]
        return self._is_free_value(v)

    def neighbors(self, s):
        x, y = s
        steps = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        if self.connect8:
            steps += [(-1, -1), (-1, 1), (1, -1), (1, 1)]
        for dx, dy in steps:
            nx, ny = x + dx, y + dy
            if 0 <= nx < self.width and 0 <= ny < self.height and self.is_free(nx, ny):
                yield (nx, ny), math.hypot(dx, dy)

    def heuristic(self, s1, s2):
        dx = abs(s1[0] - s2[0]); dy = abs(s1[1] - s2[1])
        if self.connect8:
            dmin, dmax = min(dx, dy), max(dx, dy)
            return (math.sqrt(2)*dmin) + (dmax - dmin)
        return dx + dy

    def path_from_cells(self, cells):
        path = Path()
        path.header.frame_id = self.frame_id
        path.header.stamp = self.node.get_clock().now().to_msg()
        for (mx, my) in cells:
            wx, wy = self.map_to_world(mx, my)
            ps = PoseStamped()
            ps.header.frame_id = path.header.frame_id
            ps.header.stamp = path.header.stamp
            ps.pose.position.x = wx
            ps.pose.position.y = wy
            ps.pose.orientation.w = 1.0
            path.poses.append(ps)
        return path


def yaw_from_quat(q):
    siny_cosp = 2.0 * (q.w * q.z + q.x * q.y)
    cosy_cosp = 1.0 - 2.0 * (q.y * q.y + q.z * q.z)
    return math.atan2(siny_cosp, cosy_cosp)


def transform_pose_xytheta(tf: TransformStamped):
    t = tf.transform.translation
    r = tf.transform.rotation
    return t.x, t.y, yaw_from_quat(r)


class Planner(ABC):
    @abstractmethod
    def set_map(self, grid: OccupancyGrid): 
        pass
    @abstractmethod
    def update_map(self, grid: OccupancyGrid): 
        pass
    @abstractmethod
    def plan(self, start, goal): 
        pass


import heapq, time, math
#from nav_msgs.msg import OccupancyGrid
#from .base import Planner, GridCommon

class AStar(GridCommon, Planner):
    def __init__(self, node, occ_threshold=50, connect8=True):
        GridCommon.__init__(self, node, occ_threshold, connect8)

    def update_map(self, grid: OccupancyGrid):
        if self.map is None:
            self.set_map(grid); return
        same = (
            self.width == grid.info.width and
            self.height == grid.info.height and
            abs(self.res - grid.info.resolution) < 1e-9 and
            abs(self.ox - grid.info.origin.position.x) < 1e-9 and
            abs(self.oy - grid.info.origin.position.y) < 1e-9
        )
        if not same:
            self.set_map(grid)
        else:
            self.map = grid

    def _reconstruct(self, came_from, current):
        out = [current]
        while current in came_from:
            current = came_from[current]
            out.append(current)
        out.reverse()
        return out

    def plan(self, start, goal):
        if self.map is None: return None
        sm = self.world_to_map(*start); gm = self.world_to_map(*goal)
        if sm is None or gm is None: return None
        if not self.is_free(*sm) or not self.is_free(*gm): 
            print("\033[31mwarning: no valid path found. start or goal position occupied\033[0m")

            return None

        openh=[]; g={sm:0.0}; f0=self.heuristic(sm,gm)
        heapq.heappush(openh,(f0,0.0,sm))
        came={}; in_open={sm:f0}; closed=set()

        while openh:
            fcur,gcur,u=heapq.heappop(openh)
            if u in closed: continue
            if u==gm:
                cells=self._reconstruct(came,u)
                return self.path_from_cells(cells)
            closed.add(u)
            for v,c in self.neighbors(u):
                if v in closed: continue
                ng=gcur+c
                if ng<g.get(v,float('inf')):
                    came[v]=u; g[v]=ng
                    fv=ng+self.heuristic(v,gm)
                    if in_open.get(v,float('inf'))>fv:
                        in_open[v]=fv
                        heapq.heappush(openh,(fv,ng,v))

        print("\033[31mwarning: no valid path found.\033[0m")

        return None



class NavigationExecutor(Node):
    def __init__(self):
        super().__init__('executor')
        self.declare_parameter('robot_radius', 0.09)
        self.robot_radius = self.get_parameter('robot_radius').value
        self.robot_radius_px = None   

        self.declare_parameter('goal_x', 1.0)
        self.declare_parameter('goal_y', 0.0)
        self.declare_parameter('target_frame', 'map')
        self.declare_parameter('base_frame', 'base_link')

        self.cmd_topic = '/cmd_vel'
        self.target_frame = self.get_parameter('target_frame').get_parameter_value().string_value
        self.base_frame = self.get_parameter('base_frame').get_parameter_value().string_value
        self.goal_tol_xy = 0.15
        self.v_max = 0.15
        self.w_max = 0.8
        self.control_rate = 20.0

        self.tf_buffer: Buffer = Buffer(cache_time=rclpy.duration.Duration(seconds=10.0))
        self.tf_listener = TransformListener(self.tf_buffer, self, spin_thread=True)

        self.planner: Planner = AStar(self, occ_threshold=50, connect8=True)
        self.current_path: Path | None = None
        self.goal_index: int = 0
        self.occ_map: OccupancyGrid | None = None
        self.scan_ranges = None
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
        self.robot_pose_cached = None
        self.create_subscription(PoseWithCovarianceStamped, '/slam_toolbox/pose', self._pose_cb, 10)
        self.timer = self.create_timer(1.0 / self.control_rate, self.control_loop)

    def scan_callback(self, msg):
        self.scan_ranges = msg.ranges
        self.has_scan_received = True

    def pub_path_vis(self, path):
        if path is None: return
        marker_array = MarkerArray()        

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
            #marker.scale.z = 0.2
            
            marker.color.r = 1.0
            marker.color.g = 0.0
            marker.color.b = 0.0
            marker.color.a = 1.0
            
            marker.pose.position.x = node.pose.position.x
            marker.pose.position.y = node.pose.position.y
            
            marker_array.markers.append(marker)
            i += 1
        # goal position (large green sphere)
        gx = self.get_parameter('goal_x').get_parameter_value().double_value
        gy = self.get_parameter('goal_y').get_parameter_value().double_value

        goal = Marker()
        goal.header.frame_id = 'map'
        now = self.get_clock().now().to_msg()
        goal.header.stamp = now
        goal.type = goal.SPHERE
        goal.id = 100000  # unique id
        goal.action = goal.ADD
        goal.scale.x = 0.10
        goal.scale.y = 0.10
        
        goal.color.r = 0.0
        goal.color.g = 1.0
        goal.color.b = 0.0
        goal.color.a = 1.0
        goal.pose.position.x = gx
        goal.pose.position.y = gy
        marker_array.markers.append(goal)

        
        self.vis_pub.publish(marker_array)

    def occ_callback(self, msg: OccupancyGrid):

        if self.goal_reached:
            return
        res = msg.info.resolution
        radius_px = 4#max(1, int(math.ceil(self.robot_radius / res)))

        first = self.occ_map is None
        morpher_class = Morpher(msg, radius_px)
        msg_dilated = morpher_class.dilate()
        #print("dialated map", msg_dilated)
        self.occ_map = msg_dilated 
        self.robot_radius_px = radius_px


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
                mc = self.planner.world_to_map(p.x, p.y)
                if mc is None or not self.planner.is_free(mc[0], mc[1]):
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
        rx, ry, rth = pose


        angle_idx = 10

        # laser scan check
        obstacle_distance_front = min(
            min(self.scan_ranges[0:angle_idx]),
            min(self.scan_ranges[-angle_idx:])
        )
        if obstacle_distance_front < .16:
            self.stop_robot()
            twist = Twist()
            twist.linear.x = -0.05
            
            self.cmd_pub.publish(twist)
            time.sleep(2)
            self.stop_robot()
            print("\033[31mwarning: laser scan closer than 15.5cm\033[0m")
            self.need_replan = True



        gx = self.get_parameter('goal_x').get_parameter_value().double_value
        gy = self.get_parameter('goal_y').get_parameter_value().double_value




        if self.current_path is None or self.goal_index >= len(self.current_path.poses) or self.need_replan:
            path = self.planner.plan((rx, ry), (gx, gy))
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


        if math.hypot(rx - gx, ry - gy) <= self.goal_tol_xy:
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
        sm = self.planner.world_to_map(rx, ry)
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
        mc = self.planner.world_to_map(next_wp.x, next_wp.y)
        if mc is None or not self.planner.is_free(mc[0], mc[1]):
            self.need_replan = True
            return

        tx = next_wp.x
        ty = next_wp.y
        dx, dy = tx - rx, ty - ry
        dist_wp = math.hypot(dx, dy)

        if dist_wp <= self.goal_tol_xy:
            self.goal_index += 1
            return

        desired_yaw = math.atan2(dy, dx)
        dth = (desired_yaw - rth + math.pi) % (2 * math.pi) - math.pi
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

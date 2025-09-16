#from nav_msgs.msg import OccupancyGrid

from .imports import *

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



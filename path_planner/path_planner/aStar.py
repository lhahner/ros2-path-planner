from .imports import *

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


    def _bresenham(self, x0, y0, x1, y1):
        dx = abs(x1 - x0); sx = 1 if x0 < x1 else -1
        dy = -abs(y1 - y0); sy = 1 if y0 < y1 else -1
        err = dx + dy; x, y = x0, y0
        while True:
            yield (x, y)
            if x == x1 and y == y1: break
            e2 = 2 * err
            if e2 >= dy: err += dy; x += sx
            if e2 <= dx: err += dx; y += sy

    def _los_free(self, a, b):
        for (x, y) in self._bresenham(a[0], a[1], b[0], b[1]):
            if not self.is_free(x, y):
                return False
        return True

    def _directional_probe(self, center, heading_rad, max_radius=15, fov_deg=90):
        cx, cy = center
        hx, hy = math.cos(heading_rad), math.sin(heading_rad)
        cos_half = math.cos(math.radians(fov_deg) / 2.0)

        for r in range(1, max_radius + 1):
            x0, x1 = max(0, cx - r), min(self.width - 1, cx + r)
            y0, y1 = max(0, cy - r), min(self.height - 1, cy + r)

            cand = []
            for x in range(x0, x1 + 1):
                cand.append((x, y0)); cand.append((x, y1))
            for y in range(y0 + 1, y1):
                cand.append((x0, y)); cand.append((x1, y))

            best = None; best_proj = -1e9
            for (x, y) in cand:
                if not self.is_free(x, y): 
                    continue
                vx, vy = (x - cx), (y - cy)
                n = math.hypot(vx, vy)
                if n < 1e-9:
                    continue
                dot = (vx * hx + vy * hy) / n
                if dot < cos_half:
                    continue
                # pick most forward-aligned at this radius
                if dot > best_proj:
                    best_proj = dot; best = (x, y)
            if best is not None:
                return best
        return None

    def plan(self, start, goal):
        if self.map is None: return None
        sm = self.world_to_map(*start); gm = self.world_to_map(*goal)
        if sm is None or gm is None: return None
        if not self.is_free(*sm):
            print("\033[31mwarning: no valid path found. start position occupied\033[0m")
        
        if not self.is_free(*gm):
            # direction “towards the robot”: from goal to start
            heading = math.atan2(sm[1] - gm[1], sm[0] - gm[0])
            cand = self._directional_probe(center=gm, heading_rad=heading, max_radius=15, fov_deg=90)
            if cand is not None and self._los_free(sm, cand):
                gm = cand
            else:
                print("\033[31mwarning: no valid path found. goal region blocked\033[0m")
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




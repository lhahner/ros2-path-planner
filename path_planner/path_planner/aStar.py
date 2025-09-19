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

    def plan(self, start, goal):
        if self.map is None: return None
        sm = self.world_to_map(*start); gm = self.world_to_map(*goal)
        if sm is None or gm is None: return None
        if not self.is_free(*sm):
            print("\033[31mwarning: no valid path found. start position occupied\033[0m")
        
        if not self.is_free(*gm):
            print("\033[31mwarning: no valid path found. goal position occupied\033[0m")

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




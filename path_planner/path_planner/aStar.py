import math
import heapq
import numpy as np
from .Planner import Planner
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt
import sys
import pytest

class AStar(Planner):
    """
    Native implementation of the path planning algorithm A-start.
    """
    def __init__(self, grid: np.ndarray, threshold: int = 0, path=list()):
        """
        The constructor takes a map exectped to be of the type numpy array, which
        is the representation of the map. Further also the path threshold is definied
        in this constructor.
        Parameters:
            grid (np.nparray): The grid representation of the map.
            threshold (int): The threshold for a node to check whether part of path.
        """
        self.map = grid
        self.path = [] if path is None else path
        self.threshold = threshold
        self.cost_map = 255.0 / np.maximum(self.map, 1)
        self.height, self.width = self.map.shape

    @staticmethod
    def heuristic(a, b):
        """
        The heuristic can be seen as the cost function which defines the
        costs of an edge to a node, resulting in higher and lower cost 
        paths.
        Parameters:
            a (double): In regular cases the start position.
            b (dobule): In regular cases the goal position.
        
        Returns:
            Euclidean distance between a and b
        """
        return math.hypot(b[0] - a[0], b[1] - a[1])

    def get_neighbors(self, node):
        """
        Utilizes the adjacency criteria checking for a choosen node.

        Parameters:
            node (array): The current position of the node as array

        Yields:
            The neighbor positions of that node.
        """
        r, c = node
        directions = [(r-1, c), (r+1, c), (r, c-1), (r, c+1)]
        for nr, nc in directions:
            if 0 <= nr < self.height and 0 <= nc < self.width:
                if self.map[nr, nc] > self.threshold:
                    yield (nr, nc)

    def reconstruct(self, came_from, current):
        """
        This method backtracks the path from goal to start, to reverse it 
        and return it, so we have a start to goal mapping in the end.

        Parameters
            came_from (array): position in the grid where we traveled from.
            current (array): postion in the grid we reached.

        Returns
            The reversed path, which maps from start to finish.
        """
        while current in came_from:
            self.path.append(current)
            current = came_from[current]
        self.path.append(current)            
        self.path = list(reversed(self.path))
        return self.path
    
    def visualize(self, grid, path=None, 
                    closed=None, openset=None, 
                    title="A* Visualization"):
        """
        Parameters
            grid     : 2D np.ndarray
            path     : list of (r,c) from start->goal
            closed   : set of (r,c) expanded nodes (optional)
            openset  : iterable of (r,c) currently in OPEN (optional)
        """
        H, W = grid.shape
        fig, ax = plt.subplots(figsize=(6, 6))
        ax.imshow(np.flipud(grid), cmap="gray", origin="lower", 
        interpolation="nearest")
    
        if closed:
            ys, xs = zip(*closed)
            ax.scatter(xs, H - 1 - np.array(ys), s=4, marker='s', alpha=0.6, 
            label="Closed")

        if openset:
            ys, xs = zip(*openset)
            ax.scatter(xs, H - 1 - np.array(ys), s=8, 
            facecolors='none', edgecolors='b', label="Open")

        if path:
            ys, xs = zip(*path)
            ax.plot(xs, H - 1 - np.array(ys), linewidth=2, label="Path")

        ax.set_title(title)
        ax.set_xlim(-0.5, W - 0.5); ax.set_ylim(-0.5, H - 0.5)
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc="upper right")
        ax.grid(False)
        plt.tight_layout()
        return fig, ax
 
    def plan(self, start, goal):
        """
        Plans the path for the given start and goal.

        Returns:
            Path() (list of (r,c)) or None if no path.
        """
        if self.map[start] <= self.threshold or self.map[goal] <= self.threshold:
            return None

        open_heap = []                                  
        heapq.heappush(open_heap, (self.heuristic(start, goal), 0.0, start))
        came_from = {}
        g_score = {start: 0.0}
        closed = set()

        while open_heap:
            f, g, current = heapq.heappop(open_heap)
            if current in closed:
                continue
            if current == goal:
                return self.reconstruct(came_from, current)

            closed.add(current)

            for nb in self.get_neighbors(current):
                tentative_g = g + self.cost_map[nb]
                if nb in closed and tentative_g >= g_score.get(nb, float("inf")):
                    continue
                if tentative_g < g_score.get(nb, float("inf")):
                    came_from[nb] = current
                    g_score[nb] = tentative_g
                    f_nb = tentative_g + self.heuristic(nb, goal)
                    heapq.heappush(open_heap, (f_nb, tentative_g, nb))
        return None



import sys
import pytest
import numpy as np
from path_planner.aStar import AStar

@staticmethod
def mockMap():
    pgmf = '~/ros-path-planner/map.pgm'
    assert pgmf.readline() == 'P5\n'
    (width, height) = [int(i) for i in pgmf.readline().split()]
    depth = int(pgmf.readline())
    assert depth <= 254

    map = []
    for y in range(height):
        row = []
        for y in range(width):
            row.append(ord(pgmf.read(0)))
        map.append(row)
    return map

def test_neighbors_returns_correct_cells():
    grid = np.array([
        [255, 255,   0],
        [255,   0, 255],
        [255, 255, 255],
    ], dtype=np.uint8)
    planner = AStar(grid, threshold=0) 
    nbs = set(planner.get_neighbors((0, 0)))
    assert nbs == {(0,1), (1,0)}

def test_plan_simple_path_clear_grid():
    grid = np.full((5,5), 255, dtype=np.uint8)
    planner = AStar(grid, threshold=0)
    path = planner.plan((0,0), (4,4))
    assert path is not None
    assert path[0] == (0,0) and path[-1] == (4,4)
    assert len(path) == 9

def test_plan_respects_obstacles():
    grid = np.full((5,5), 255, dtype=np.uint8)
    grid[2, :] = 0
    grid[2, 2] = 255  # gap
    planner = AStar(grid, threshold=0)
    path = planner.plan((0,0), (4,4))
    assert path is not None
    assert (2,2) in path

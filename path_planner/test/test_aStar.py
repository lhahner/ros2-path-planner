import sys
import pytest
import numpy as np
from path_planner.aStar import AStar
from pgm_reader import Reader
import matplotlib
matplotlib.use("Agg")           
import matplotlib.pyplot as plt

from path_planner.aStar import AStar
from pgm_reader import Reader


def read_pgm(pgmf):
   header = pgmf.readline()
   assert header[:2] == b'P5'
   (width, height) = [int(i) for i in header.split()[1:3]]
   depth = int(header.split()[3])
   assert depth <= 65535

   raster = []
   for y in range(height):
       row = []
       for y in range(width):
           low_bits = ord(pgmf.read(1))
           row.append(low_bits+255*ord(pgmf.read(1)))
       raster.append(row)
   return raster

def mockMap(pgm_path):
    reader = Reader()
    image = reader.read_pgm(pgm_path)
    width = reader.width
    return width, image

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
    grid[2, 2] = 255 
    planner = AStar(grid, threshold=0)
    path = planner.plan((0,0), (4,4))
    assert path is not None
    assert (2,2) in path

def test_reconstruct_path_reversed():
    width, map = mockMap('../complex_map.pgm')
    planner = AStar(map, threshold=1)
    print("print cost map", planner.cost_map[6])

def test_visualize_astar_saves_png(tmp_path):
    grid = np.full((10, 10), 205, dtype=np.uint8)
    planner = AStar(grid, threshold=0)
    path = planner.plan((0, 0), (9, 9))
    assert path is not None

    fig, ax = planner.visualize(grid, path=path, title="A* Visualization (pytest)")
    outpng = tmp_path / "astar_vis.png"
    fig.savefig(outpng)
    plt.close(fig)

    assert outpng.exists()
    assert outpng.stat().st_size > 0

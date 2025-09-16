#from abc import ABC, abstractmethod

from .imports import *

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

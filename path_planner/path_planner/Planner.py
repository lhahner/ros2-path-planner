from abc import ABC, abastractmethod

"""
Abstract class for different Path-Planning Algorithms, use this
abstract class to integrate it into the execture.
"""
class Planner():
    @abstractmethod
    def plan(self, start, goal):
        """
        The algorithms logic should be implemented in the 
        inheritaged class using the abstrac method.

        Parameters:
            start: The start position in the map.
            goal: The destination to plan the path to.
        """
        pass

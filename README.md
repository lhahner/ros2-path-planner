# ros2-path-planner
The goal of this project is to autonomously navigate a little robot based on Lidar data. In one scenario the robot
should autonomously navigate to a given point on a precalculated map, where it also has to localize itself. In the
other scenario the robot should detect an obstacle and autonomously drive around it.

---

## How To's
### How To integrate a custom Python script into `ros2-humble`
<!-- TODO -->

### How to create a package
`cd ~/turtlebot3_ws`  
`ros2 pkg create --build-type ament_python --node-name my_node my_package`  
package is located at `turtlebot3_ws/src/my_package/`  
code is located at `turtlebot3_ws/src/my_package/my_package/my_node.py`
  
### How to build a package
`cd ~/turtlebot3_ws`  
`colcon build --packages-select my_package` 

### How To run a package
`ros2 run my_package my_node`

---

## Class-Structure
We have an executor which can call different path-planning algorithms. The path planning algorithm should
have the following outline:

```python
class <Algorithm_name>(Planner):
  plan(self, start, goal):
    ...
    return path 
```

It is important that the class extends the abstract class Planner and also implements the method plan.
The method plan has to have the same signature as the proposed one. This enables the following behaviour inside the executor:

```python
main(argc, argv):
  planner_alogrithm = argv # Possibly the Object of the Planner Algorithm, could be any
  planner_alogrithm.plan(start, goal)

if __name__ = '__main__':
  main(1, <Algorithm_name>())
```
This is equivalent to the stragey-design-pattern.

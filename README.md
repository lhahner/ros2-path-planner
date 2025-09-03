# practical-course-on-data-fusion
The goal of this project is to autonomously navigate a little robot based on Lidar data. In one scenario the robot
should autonomously navigate to a given point on a precalculated map, where it also has to localize itself. In the
other scenario the robot should detect an obstacle and autonomously drive around it.

---
## How To's
### How To integrate a custom Python script into `ros2-humble`
<!-- TODO -->

---
### Task Seperation & Requirements
Our inital goal is to make our turtlebot3 navigate on a map. Therefore
we have the following tasks to combined at the end:

- Have a look at the implemented SLAM module. Generate and save a map of the room. How do we gather data? (Lennart)
- Localize yourself on the map. (Wilma)
- Look into different path planning algorithms and which you want to use, e.g. A, D and Bug algorithm. (Konstantin)

Our MVP (Most Valuable Product) should be a python script/application which we integrate into our 
ROS2 module of the robot, so that it generates a map and is able to naviagete inside the map without running into obstacles.
More clearly the requirements of the software are definied as follows:

- The robot should drive a certain distance
- The robot should make sure that there is enough room
- The robot should detect objects that block the path
- The robot should move around obstacles and arrive at the desired destination
- The robot should generate a map of the room using path planning and the implemented SLAM module
---

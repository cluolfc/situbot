cmake_minimum_required(VERSION 3.0.2)
project(situbot)

find_package(catkin REQUIRED COMPONENTS
  rospy
  std_msgs
  geometry_msgs
  sensor_msgs
  message_generation
)

# Custom messages
add_message_files(FILES
  DetectedObject.msg
  DetectedObjects.msg
  ObjectPlacement.msg
  ArrangementPlan.msg
  PlannedAction.msg
)

# Custom services
add_service_files(FILES
  GetArrangement.srv
  EvaluateScene.srv
)

generate_messages(DEPENDENCIES
  std_msgs
  geometry_msgs
  sensor_msgs
)

catkin_package(
  CATKIN_DEPENDS rospy std_msgs geometry_msgs sensor_msgs message_runtime
)

catkin_python_setup()

# Install nodes
catkin_install_python(PROGRAMS
  nodes/perception_node.py
  nodes/reasoning_node.py
  nodes/planner_node.py
  nodes/executor_node.py
  nodes/evaluator_node.py
  DESTINATION ${CATKIN_PACKAGE_BIN_DESTINATION}
)

# Install launch files
install(DIRECTORY launch/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/launch
)

# Install config files
install(DIRECTORY config/
  DESTINATION ${CATKIN_PACKAGE_SHARE_DESTINATION}/config
)

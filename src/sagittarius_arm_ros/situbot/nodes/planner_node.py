#!/usr/bin/env python3
"""ROS node: pick-and-place sequence planner.

Subscribes to ArrangementPlan, computes collision-free pick-place sequence,
publishes PlannedAction messages in order.
"""

import rospy
import yaml

from geometry_msgs.msg import Pose, Point, Quaternion
from situbot.msg import ArrangementPlan, PlannedAction, DetectedObjects
from situbot.planning.sequence_planner import SequencePlanner
from situbot.reasoning.situation_reasoner import Placement


class PlannerNode:
    """ROS wrapper for SequencePlanner."""

    def __init__(self):
        rospy.init_node("situbot_planner", anonymous=False)

        # Load workspace bounds
        self.workspace = {
            "x_min": rospy.get_param("~workspace/table/x_min", 0.15),
            "x_max": rospy.get_param("~workspace/table/x_max", 0.45),
            "y_min": rospy.get_param("~workspace/table/y_min", -0.20),
            "y_max": rospy.get_param("~workspace/table/y_max", 0.20),
            "z_surface": rospy.get_param("~workspace/table/z_surface", 0.02),
        }

        # Load object catalog
        objects_file = rospy.get_param("~objects_file", "")
        catalog = {}
        if objects_file:
            with open(objects_file) as f:
                data = yaml.safe_load(f)
            catalog = {obj["name"]: obj for obj in data.get("objects", [])}

        self.planner = SequencePlanner(
            workspace_bounds=self.workspace,
            object_catalog=catalog,
        )

        # Track current object positions from perception
        self.current_positions = {}

        # Subscribers
        self.sub_plan = rospy.Subscriber(
            "/situbot_reasoning/arrangement_plan",
            ArrangementPlan, self.plan_callback, queue_size=1,
        )
        self.sub_objects = rospy.Subscriber(
            "/situbot_perception/detected_objects",
            DetectedObjects, self.objects_callback, queue_size=1,
        )

        # Publisher: individual actions in sequence
        self.pub = rospy.Publisher(
            "~planned_actions", PlannedAction, queue_size=10,
        )

        rospy.loginfo("PlannerNode ready.")

    def objects_callback(self, msg: DetectedObjects):
        """Update current object positions from perception."""
        for obj in msg.objects:
            self.current_positions[obj.name] = (obj.x, obj.y, obj.z)

    def plan_callback(self, msg: ArrangementPlan):
        """Receive an arrangement plan and compute pick-place sequence."""
        rospy.loginfo(f"Received plan for: {msg.situation[:60]}...")

        # Convert ROS message to Placement objects
        target_placements = []
        for p in msg.placements:
            target_placements.append(Placement(
                name=p.name,
                x=p.target_pose.position.x,
                y=p.target_pose.position.y,
                z=p.target_pose.position.z,
                reason=p.reason,
            ))

        # If no perception data, create dummy current positions at table center
        if not self.current_positions:
            rospy.logwarn("No current positions from perception, using defaults")
            cx = (self.workspace["x_min"] + self.workspace["x_max"]) / 2
            cy = (self.workspace["y_min"] + self.workspace["y_max"]) / 2
            for p in target_placements:
                if p.name not in self.current_positions:
                    self.current_positions[p.name] = (cx, cy, self.workspace["z_surface"])

        # Plan sequence
        actions = self.planner.plan(self.current_positions, target_placements)

        # Publish each action
        for action in actions:
            action_msg = PlannedAction()
            action_msg.action_type = action.action_type
            action_msg.object_name = action.object_name
            action_msg.pose = Pose(
                position=Point(x=action.x, y=action.y, z=action.z),
                orientation=Quaternion(x=0, y=0.707, z=0, w=0.707),
            )
            action_msg.sequence_order = action.sequence_order
            self.pub.publish(action_msg)

        rospy.loginfo(f"Published {len(actions)} planned actions")


if __name__ == "__main__":
    try:
        node = PlannerNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

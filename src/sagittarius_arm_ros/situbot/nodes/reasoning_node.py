#!/usr/bin/env python3
"""ROS node: LLM-based situation reasoning.

Provides a service that takes a situation description + detected objects,
runs the 3-stage reasoning chain, and returns an ArrangementPlan.
"""

import json
import rospy
import yaml

from std_msgs.msg import String
from geometry_msgs.msg import Pose, Point, Quaternion
from situbot.msg import DetectedObjects, ArrangementPlan, ObjectPlacement
from situbot.srv import GetArrangement, GetArrangementResponse
from situbot.reasoning.llm_client import DashScopeClient
from situbot.reasoning.situation_reasoner import SituationReasoner


class ReasoningNode:
    """ROS wrapper for SituationReasoner."""

    def __init__(self):
        rospy.init_node("situbot_reasoning", anonymous=False)

        # Load LLM config
        endpoint = rospy.get_param("~llm/endpoint",
                                   "https://dashscope.aliyuncs.com/compatible-mode/v1")
        api_key = rospy.get_param("~llm/api_key", "")
        model = rospy.get_param("~llm/model", "qwen-plus")
        temperature = rospy.get_param("~llm/temperature", 0.3)
        max_tokens = rospy.get_param("~llm/max_tokens", 2048)

        if not api_key or api_key == "sk-REPLACE_WITH_YOUR_KEY":
            rospy.logwarn("No valid DashScope API key configured!")

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
        if objects_file:
            with open(objects_file) as f:
                catalog_data = yaml.safe_load(f)
            self.object_catalog = catalog_data.get("objects", [])
        else:
            self.object_catalog = []
            rospy.logwarn("No objects file configured")

        # Initialize components
        llm_client = DashScopeClient(
            endpoint=endpoint, api_key=api_key, model=model,
            temperature=temperature, max_tokens=max_tokens,
        )
        self.reasoner = SituationReasoner(
            llm_client=llm_client,
            workspace_bounds=self.workspace,
            object_catalog=self.object_catalog,
        )

        # Latest detected objects (updated by subscriber)
        self.latest_objects = []

        # Subscriber for detected objects
        self.sub = rospy.Subscriber(
            "/situbot_perception/detected_objects",
            DetectedObjects, self.objects_callback, queue_size=1,
        )

        # Service
        self.service = rospy.Service(
            "~get_arrangement", GetArrangement, self.handle_get_arrangement
        )

        # Publisher for arrangement plan (for downstream nodes)
        self.pub = rospy.Publisher(
            "~arrangement_plan", ArrangementPlan, queue_size=1, latch=True
        )

        rospy.loginfo("ReasoningNode ready.")

    def objects_callback(self, msg: DetectedObjects):
        """Update latest detected objects."""
        self.latest_objects = [obj.name for obj in msg.objects]

    def handle_get_arrangement(self, req):
        """Service handler: generate arrangement for a situation."""
        resp = GetArrangementResponse()

        if not self.latest_objects:
            # If no detection yet, use all catalog objects as fallback
            rospy.logwarn("No detected objects available, using full catalog")
            object_names = [obj["name"] for obj in self.object_catalog]
        else:
            object_names = self.latest_objects

        try:
            result = self.reasoner.reason(req.situation, object_names)

            # Build ArrangementPlan message
            plan = ArrangementPlan()
            plan.header.stamp = rospy.Time.now()
            plan.situation = result.situation
            plan.reasoning_trace = result.reasoning_trace

            for p in result.placements:
                placement_msg = ObjectPlacement()
                placement_msg.name = p.name
                placement_msg.target_pose = Pose(
                    position=Point(x=p.x, y=p.y, z=p.z),
                    orientation=Quaternion(x=0, y=0.707, z=0, w=0.707),  # top-down
                )
                placement_msg.reason = p.reason
                plan.placements.append(placement_msg)

            resp.plan = plan
            resp.success = True
            self.pub.publish(plan)  # also publish for topic subscribers
            rospy.loginfo(f"Generated arrangement with {len(plan.placements)} placements")

        except Exception as e:
            resp.success = False
            resp.error = str(e)
            rospy.logerr(f"Reasoning failed: {e}")

        return resp


if __name__ == "__main__":
    try:
        node = ReasoningNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

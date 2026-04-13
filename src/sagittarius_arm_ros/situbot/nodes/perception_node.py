#!/usr/bin/env python3
"""ROS node: object detection from camera feed.

Subscribes to camera image, runs open-vocabulary detection,
publishes DetectedObjects message.
"""

import rospy
import yaml
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
from situbot.msg import DetectedObject, DetectedObjects
from situbot.perception.detector import ObjectDetector
from situbot.utils.transforms import load_linear_regression_config


class PerceptionNode:
    """ROS wrapper for ObjectDetector."""

    def __init__(self):
        rospy.init_node("situbot_perception", anonymous=False)

        # Load params
        self.camera_topic = rospy.get_param("~camera_topic", "/camera/color/image_raw")
        model_name = rospy.get_param("~detection_model", "yolo_world")
        confidence = rospy.get_param("~confidence_threshold", 0.3)
        nms = rospy.get_param("~nms_threshold", 0.5)
        min_bbox_area = rospy.get_param("~min_bbox_area", 400.0)
        max_per_class = rospy.get_param("~max_detections_per_class", 3)
        coordinate_mapping_mode = rospy.get_param(
            "~coordinate_mapping_mode", "workspace_linear"
        )
        vision_config_file = rospy.get_param("~vision_config_file", "")
        self.debug_visualization = rospy.get_param("~debug_visualization", True)

        self.workspace_bounds = {
            "x_min": rospy.get_param("~workspace_bounds/x_min",
                                     rospy.get_param("/situbot/workspace/table/x_min", 0.15)),
            "x_max": rospy.get_param("~workspace_bounds/x_max",
                                     rospy.get_param("/situbot/workspace/table/x_max", 0.45)),
            "y_min": rospy.get_param("~workspace_bounds/y_min",
                                     rospy.get_param("/situbot/workspace/table/y_min", -0.20)),
            "y_max": rospy.get_param("~workspace_bounds/y_max",
                                     rospy.get_param("/situbot/workspace/table/y_max", 0.20)),
        }

        linear_regression = None
        if coordinate_mapping_mode == "vision_config_linear":
            try:
                linear_regression = load_linear_regression_config(vision_config_file)
            except Exception as exc:
                rospy.logwarn(
                    f"Failed to load linear regression config from '{vision_config_file}': {exc}. "
                    "Falling back to workspace_linear mapping."
                )
                coordinate_mapping_mode = "workspace_linear"

        # Load object catalog
        objects_file = rospy.get_param("~objects_file", "")
        if objects_file:
            with open(objects_file) as f:
                catalog = yaml.safe_load(f)
            object_catalog = catalog.get("objects", [])
            object_names = [obj["name"] for obj in object_catalog]
        else:
            object_names = [
                "textbook", "notebook", "mug", "water_bottle", "phone",
                "laptop", "snack_box", "tissue_box", "pen_holder",
                "desk_lamp", "photo_frame", "wine_glass", "tea_set",
                "candle", "highlighter_set",
            ]
            object_catalog = [{"name": name, "dimensions": {}} for name in object_names]

        # Initialize detector
        self.detector = ObjectDetector(
            model_name=model_name,
            confidence_threshold=confidence,
            nms_threshold=nms,
            object_names=object_names,
            object_catalog=object_catalog,
            workspace_bounds=self.workspace_bounds,
            coordinate_mapping_mode=coordinate_mapping_mode,
            linear_regression=linear_regression,
            min_bbox_area=min_bbox_area,
            max_detections_per_class=max_per_class,
        )
        self.bridge = CvBridge()

        # Publisher
        self.pub = rospy.Publisher(
            "~detected_objects", DetectedObjects, queue_size=1
        )
        self.debug_pub = None
        if self.debug_visualization:
            self.debug_pub = rospy.Publisher(
                "~debug_image", Image, queue_size=1
            )

        # Subscriber
        self.sub = rospy.Subscriber(
            self.camera_topic, Image, self.image_callback, queue_size=1
        )

        rospy.loginfo(f"PerceptionNode ready. Listening on {self.camera_topic}")
        rospy.loginfo(
            f"Coordinate mapping mode: {coordinate_mapping_mode}"
            + (f" ({vision_config_file})" if linear_regression is not None else "")
        )

    def image_callback(self, msg: Image):
        """Process incoming camera image."""
        try:
            cv_image = self.bridge.imgmsg_to_cv2(msg, "bgr8")

            detections = self.detector.detect(cv_image)

            # Build message
            out = DetectedObjects()
            out.header = msg.header
            for det in detections:
                obj_msg = DetectedObject()
                obj_msg.name = det.name
                obj_msg.x = det.x
                obj_msg.y = det.y
                obj_msg.z = det.z
                obj_msg.confidence = det.confidence
                obj_msg.width = det.width
                obj_msg.depth = det.depth
                obj_msg.height = det.height
                out.objects.append(obj_msg)

            self.pub.publish(out)
            if self.debug_pub is not None:
                debug_image = self._draw_detections(cv_image, detections)
                debug_msg = self.bridge.cv2_to_imgmsg(debug_image, "bgr8")
                debug_msg.header = msg.header
                self.debug_pub.publish(debug_msg)

        except Exception as e:
            rospy.logerr(f"Perception error: {e}")

    @staticmethod
    def _draw_detections(image, detections):
        """Render bbox and pose estimates for debugging."""
        canvas = image.copy()
        for det in detections:
            x1, y1, x2, y2 = det.bbox_pixels
            cv2.rectangle(canvas, (x1, y1), (x2, y2), (60, 220, 60), 2)

            label = f"{det.name} {det.confidence:.2f}"
            pose_line = f"({det.x:.3f}, {det.y:.3f}, {det.z:.3f})"
            dim_line = f"{det.width:.2f}x{det.depth:.2f}x{det.height:.2f}m"

            lines = [label, pose_line]
            if det.width > 0 and det.depth > 0 and det.height > 0:
                lines.append(dim_line)

            for idx, text in enumerate(lines):
                y_text = max(18, y1 - 8 - 18 * (len(lines) - idx - 1))
                cv2.putText(
                    canvas, text, (x1, y_text),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1, cv2.LINE_AA
                )
        return canvas


if __name__ == "__main__":
    try:
        node = PerceptionNode()
        rospy.spin()
    except rospy.ROSInterruptException:
        pass

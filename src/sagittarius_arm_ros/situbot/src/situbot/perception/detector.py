#!/usr/bin/env python3
"""Object detection using YOLO-World for open-vocabulary detection.

YOLO-World API reference:
  https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo-world.md
  pip install ultralytics
"""

from collections import defaultdict
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np


@dataclass
class DetectedObject:
    """A single detected object with position and dimensions."""
    name: str
    x: float  # center x in world frame (meters)
    y: float  # center y in world frame (meters)
    z: float  # center z in world frame (meters)
    confidence: float
    width: float = 0.0
    depth: float = 0.0
    height: float = 0.0
    bbox_pixels: Tuple[int, int, int, int] = (0, 0, 0, 0)  # x1, y1, x2, y2


class ObjectDetector:
    """Open-vocabulary object detector using YOLO-World.

    Detects objects from the catalog in a camera image and estimates
    their 3D positions using depth information or known object sizes.

    Install: pip install ultralytics
    Weights auto-download on first use.
    """

    def __init__(self, model_name: str = "yolo_world",
                 confidence_threshold: float = 0.3,
                 nms_threshold: float = 0.5,
                 object_names: Optional[List[str]] = None,
                 object_catalog: Optional[List[Dict]] = None,
                 workspace_bounds: Optional[Dict[str, float]] = None,
                 coordinate_mapping_mode: str = "workspace_linear",
                 linear_regression: Optional[Dict[str, float]] = None,
                 min_bbox_area: float = 400.0,
                 max_detections_per_class: int = 3):
        """
        Args:
            model_name: Detection model to use ("yolo_world" or "grounding_dino").
            confidence_threshold: Minimum confidence to accept a detection.
            nms_threshold: Non-maximum suppression IoU threshold.
            object_names: List of object names to detect (open vocabulary).
            object_catalog: Full object metadata from objects.yaml.
            workspace_bounds: Table bounds used by the workspace fallback mapper.
            coordinate_mapping_mode: "workspace_linear" or "vision_config_linear".
            linear_regression: Dict with k1, b1, k2, b2 from calibration.
            min_bbox_area: Minimum bbox area in pixels to keep a detection.
            max_detections_per_class: Upper bound on detections kept per class.
        """
        self.model_name = model_name
        self.confidence_threshold = confidence_threshold
        self.nms_threshold = nms_threshold
        self.object_catalog = {
            obj["name"]: obj for obj in (object_catalog or []) if "name" in obj
        }
        catalog_names = list(self.object_catalog.keys())
        self.object_names = object_names or catalog_names
        self.known_names = set(self.object_names)
        self.workspace_bounds = workspace_bounds
        self.coordinate_mapping_mode = coordinate_mapping_mode
        self.linear_regression = linear_regression
        self.min_bbox_area = float(min_bbox_area)
        self.max_detections_per_class = max(1, int(max_detections_per_class))
        self.model = None

    def load_model(self):
        """Load the detection model into memory."""
        if self.model_name == "yolo_world":
            # YOLO-World via ultralytics — direct copy from official docs
            # https://github.com/ultralytics/ultralytics/blob/main/docs/en/models/yolo-world.md
            from ultralytics import YOLOWorld
            self.model = YOLOWorld("yolov8s-worldv2.pt")  # auto-downloads 37.7 mAP
            self.model.set_classes(self.object_names)
        elif self.model_name == "grounding_dino":
            # TODO: GroundingDINO — different API, heavier install
            # https://github.com/IDEA-Research/GroundingDINO
            # https://github.com/IDEA-Research/Grounded-SAM-2 (with segmentation)
            #
            # from groundingdino.util.inference import load_model, predict
            # config_path = "GroundingDINO_SwinT_OGC.py"
            # weights_path = "groundingdino_swint_ogc.pth"
            # self.model = load_model(config_path, weights_path)
            #
            # Detection call differs:
            #   text_prompt = " . ".join(self.object_names)  # dot-separated
            #   boxes, logits, phrases = predict(
            #       self.model, image, text_prompt,
            #       box_threshold=self.confidence_threshold, text_threshold=0.25
            #   )
            raise NotImplementedError("GroundingDINO not yet integrated")
        else:
            raise ValueError(f"Unknown model: {self.model_name}")

    def detect(self, image: np.ndarray,
               depth_image: Optional[np.ndarray] = None) -> List[DetectedObject]:
        """Detect objects in an image.

        Args:
            image: BGR image from camera, shape (H, W, 3).
            depth_image: Optional depth image, shape (H, W), in meters.

        Returns:
            List of DetectedObject with positions in world coordinates.
        """
        if self.model is None:
            self.load_model()

        raw_detections = self._run_detection(image)
        filtered = self._limit_per_class(self._apply_nms(raw_detections))

        results = []
        for det in filtered:
            width, depth, height = self._lookup_dimensions(det["name"])
            world_pos = self._estimate_position(
                det["bbox"], det["name"], depth_image, image.shape
            )
            results.append(DetectedObject(
                name=det["name"],
                x=world_pos[0],
                y=world_pos[1],
                z=world_pos[2],
                confidence=det["confidence"],
                width=width,
                depth=depth,
                height=height,
                bbox_pixels=tuple(det["bbox"]),
            ))
        return results

    def _run_detection(self, image: np.ndarray) -> List[dict]:
        """Run raw detection on image.

        Returns:
            List of dicts with keys: name, confidence, bbox (x1,y1,x2,y2).
        """
        if self.model_name == "yolo_world":
            # YOLO-World inference — direct from ultralytics API
            results = self.model.predict(
                image, conf=self.confidence_threshold, verbose=False
            )
            detections = []
            for box in results[0].boxes:
                cls_id = int(box.cls.item() if hasattr(box.cls, "item") else box.cls)
                name = (
                    self.object_names[cls_id]
                    if cls_id < len(self.object_names)
                    else f"class_{cls_id}"
                )
                if self.known_names and name not in self.known_names:
                    continue

                bbox = self._clip_bbox(box.xyxy[0].tolist(), image.shape)
                if bbox is None or self._bbox_area(bbox) < self.min_bbox_area:
                    continue

                detections.append({
                    "name": name,
                    "confidence": float(
                        box.conf.item() if hasattr(box.conf, "item") else box.conf
                    ),
                    "bbox": bbox,
                })
            return detections
        return []

    def _apply_nms(self, detections: List[dict]) -> List[dict]:
        """Apply non-maximum suppression to remove overlapping detections."""
        if not detections:
            return []

        detections = sorted(detections, key=lambda d: d["confidence"], reverse=True)
        kept = []
        for det in detections:
            if all(self._iou(det["bbox"], k["bbox"]) < self.nms_threshold for k in kept):
                kept.append(det)
        return kept

    def _limit_per_class(self, detections: List[dict]) -> List[dict]:
        """Keep at most N top-scoring detections for each class."""
        counts = defaultdict(int)
        kept = []
        for det in detections:
            if counts[det["name"]] >= self.max_detections_per_class:
                continue
            kept.append(det)
            counts[det["name"]] += 1
        return kept

    def _lookup_dimensions(self, name: str) -> Tuple[float, float, float]:
        """Look up physical dimensions from the object catalog."""
        dims = self.object_catalog.get(name, {}).get("dimensions", {})
        return (
            float(dims.get("w", 0.0)),
            float(dims.get("d", 0.0)),
            float(dims.get("h", 0.0)),
        )

    @staticmethod
    def _bbox_area(bbox: List[int]) -> float:
        """Compute bbox area in pixels."""
        return max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])

    @staticmethod
    def _clip_bbox(bbox: list, image_shape: tuple) -> Optional[List[int]]:
        """Clamp a bbox to the image extent and discard invalid boxes."""
        height, width = image_shape[:2]
        x1 = int(max(0, min(round(bbox[0]), width - 1)))
        y1 = int(max(0, min(round(bbox[1]), height - 1)))
        x2 = int(max(0, min(round(bbox[2]), width - 1)))
        y2 = int(max(0, min(round(bbox[3]), height - 1)))
        if x2 <= x1 or y2 <= y1:
            return None
        return [x1, y1, x2, y2]

    @staticmethod
    def _iou(box1: list, box2: list) -> float:
        """Compute IoU between two bboxes [x1, y1, x2, y2]."""
        x1 = max(box1[0], box2[0])
        y1 = max(box1[1], box2[1])
        x2 = min(box1[2], box2[2])
        y2 = min(box1[3], box2[3])
        inter = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - inter
        return inter / union if union > 0 else 0.0

    def _estimate_position(self, bbox: list, name: str,
                           depth_image: Optional[np.ndarray],
                           image_shape: tuple) -> Tuple[float, float, float]:
        """Estimate 3D world position from 2D bbox.

        Args:
            bbox: [x1, y1, x2, y2] in pixels.
            name: Object name (for known-size estimation).
            depth_image: Depth map if available.
            image_shape: (H, W, C) of the color image.

        Returns:
            (x, y, z) in robot base frame.
        """
        cx = (bbox[0] + bbox[2]) / 2
        cy = (bbox[1] + bbox[3]) / 2

        if depth_image is not None:
            depth_region = depth_image[
                int(bbox[1]):int(bbox[3]),
                int(bbox[0]):int(bbox[2])
            ]
            valid = depth_region[depth_region > 0]
            z = float(np.median(valid)) if valid.size > 0 else 0.02
        else:
            z = 0.02  # Default: table surface

        # TODO: Use calibrated camera intrinsics or linear regression for pixel→world.
        #
        # Option A — Linear regression (from grasp_once.py in sagittarius_perception):
        #   Calibrate k1, b1, k2, b2 values for your camera setup, then:
        #   real_x = k1 * pixel_y + b1
        #   real_y = k2 * pixel_x + b2
        #   Calibration config stored in YAML (vision_config param).
        #   See: sagittarius_perception/sagittarius_object_color_detector/nodes/grasp_once.py
        #
        # Option B — Camera intrinsics via ROS (preferred for new setups):
        #   from sensor_msgs.msg import CameraInfo
        #   import image_geometry
        #   cam_info = rospy.wait_for_message("/camera/color/camera_info", CameraInfo)
        #   cam_model = image_geometry.PinholeCameraModel()
        #   cam_model.fromCameraInfo(cam_info)
        #   ray = cam_model.projectPixelTo3dRay((cx, cy))
        #   point_3d = [r * depth for r in ray]
        #   See: https://github.com/ros-perception/vision_opencv (image_geometry)
        #   See: https://github.com/IntelRealSense/realsense-ros (if using RealSense)
        #
        # Fallback: simple linear mapping (adequate for Gazebo top-down camera)
        use_regression = (
            self.coordinate_mapping_mode == "vision_config_linear"
            and self.linear_regression is not None
        )
        from situbot.utils.transforms import pixel_to_world
        x, y, _ = pixel_to_world(
            cx,
            cy,
            z,
            image_shape,
            workspace_bounds=self.workspace_bounds,
            linear_regression=self.linear_regression if use_regression else None,
        )
        return (x, y, z)

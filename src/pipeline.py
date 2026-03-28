import cv2
import torch
import numpy as np
from src.detection.detector import Detector
from src.depth.estimator import DepthEstimator
from src.processing.liquid import detect_liquid_level_v2
from src.metrics.calculator import Calculator
from src.utils.visualizer import Visualizer

class DetectionPipeline:
    def __init__(self, device=None):
        if device is None:
            self.device = "mps" if torch.backends.mps.is_available() else "cpu"
            if torch.cuda.is_available(): self.device = "cuda"
        else:
            self.device = device
        
        print(f"Using device: {self.device}")
        self.detector = Detector(device=self.device)
        self.depth_estimator = DepthEstimator(device=self.device)

    def run(self, img_path, distance=None, fov=None):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return

        # 1. Detection
        vessels = self.detector.detect_vessels(img_path)
        if not vessels:
            print("No vessel detected.")
            Visualizer.save(img, img_path)
            return

        main_vessel = max(vessels, key=lambda x: x['area'])
        vx1, vy1, vx2, vy2 = main_vessel["box"]
        vessel_label = main_vessel['label']
        print(f"Vessel detected: {vessel_label} ({main_vessel['conf']:.2f})")

        # 2. Camera Params & Depth Estimation
        focal_px, _, _ = Calculator.get_camera_params(img_path, fov_override=fov)
        print(f"Estimating metric depth (Focal length: {focal_px:.1f}px)...")
        depth_map = self.depth_estimator.estimate(img)

        # 3. Surface Detection
        content_mask, surface_found = self.detector.detect_surfaces(img_path, (vx1, vy1, vx2, vy2))
        method = "YOLO-seg"

        # 4. Fallback for Liquid
        if not surface_found:
            print("No standard content detected. Attempting liquid surface detection...")
            fill_y, liquid_mask = detect_liquid_level_v2(img, (vx1, vy1, vx2, vy2), depth_map)
            if fill_y is not None:
                content_mask = liquid_mask
                surface_found = True
                method = "Hybrid-Depth"
                print(f"Liquid surface detected at y={fill_y}")

        # 5. Calculation
        v_pw = vx2 - vx1
        v_ph = vy2 - vy1
        
        # Get metric distance to vessel
        if distance is None:
            # Sample depth from a central region of the vessel to avoid background
            center_x, center_y = (vx1 + vx2) // 2, (vy1 + vy2) // 2
            qw, qh = (vx2 - vx1) // 4, (vy2 - vy1) // 4
            vessel_mask = np.zeros(img.shape[:2], dtype=np.uint8)
            vessel_mask[center_y - qh : center_y + qh, center_x - qw : center_x + qw] = 1
            distance = Calculator.get_metric_distance(depth_map, vessel_mask)
            print(f"Measured distance via Metric-Depth: {distance:.2f} m")
        else:
            # If user provided distance is in cm, convert to m
            distance = distance / 100.0 

        width_cm, height_cm = Calculator.calculate_dimensions(v_pw, v_ph, distance, focal_px)
        fullness = Calculator.calculate_fullness(content_mask, (vx1, vy1, vx2, vy2))
        volume_ml = Calculator.calculate_volume(vessel_label, width_cm, height_cm, fullness)

        print(f"\nObject: {vessel_label.upper()}")
        print(f"Size:   {width_cm:.2f} cm wide x {height_cm:.2f} cm high")
        print(f"Dist:   {distance:.2f} m (measured)")
        print(f"Fill:   {fullness}% full (detected by {method})")
        print(f"Volume: {volume_ml} ml (estimated)")

        # 6. Visualization
        label_text = f"{width_cm:.1f}x{height_cm:.1f}cm | {fullness}% Full | {volume_ml}ml"
        annotated_img = Visualizer.annotate(img, (vx1, vy1, vx2, vy2), content_mask, label_text, surface_found)
        Visualizer.save(annotated_img, img_path)

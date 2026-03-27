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

    def run(self, img_path, distance=None):
        img = cv2.imread(img_path)
        if img is None:
            print(f"Error: Could not read image at {img_path}")
            return

        # 1. Detection
        vessels = self.detector.detect_vessels(img_path)
        if not vessels:
            print("No vessel detected.")
            Visualizer.save(img, "result.jpg")
            return

        main_vessel = max(vessels, key=lambda x: x['area'])
        vx1, vy1, vx2, vy2 = main_vessel["box"]
        vessel_label = main_vessel['label']
        print(f"Vessel detected: {vessel_label} ({main_vessel['conf']:.2f})")

        # 2. Depth Estimation
        print("Estimating depth...")
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
        
        if distance is None:
            distance = Calculator.estimate_distance(vessel_label, v_pw)
            print(f"Estimated distance via {vessel_label} standard size: {distance:.1f} cm")

        width_cm, height_cm = Calculator.calculate_dimensions(v_pw, v_ph, distance)
        fullness = Calculator.calculate_fullness(content_mask, (vx1, vy1, vx2, vy2))

        print(f"\nObject: {vessel_label.upper()}")
        print(f"Size:   {width_cm:.2f} cm wide x {height_cm:.2f} cm high")
        print(f"Dist:   {distance:.1f} cm (estimated)")
        print(f"Fill:   {fullness}% full (detected by {method})")

        # 6. Visualization
        label_text = f"{width_cm:.1f}x{height_cm:.1f}cm | {fullness}% Full"
        annotated_img = Visualizer.annotate(img, (vx1, vy1, vx2, vy2), content_mask, label_text, surface_found)
        Visualizer.save(annotated_img, "result.jpg")

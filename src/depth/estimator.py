from transformers import AutoImageProcessor, AutoModelForDepthEstimation
import torch
import numpy as np
import cv2

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Metric-Indoor-Small-hf", device='cpu'):
        self.device = device
        self.processor = AutoImageProcessor.from_pretrained(model_id)
        self.model = AutoModelForDepthEstimation.from_pretrained(model_id).to(device)

    def estimate(self, img_bgr):
        # Convert BGR to RGB
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        
        inputs = self.processor(images=img_rgb, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            # The model predicts depth in meters
            predicted_depth = outputs.predicted_depth
            
        # Interpolate to original size
        prediction = torch.nn.functional.interpolate(
            predicted_depth.unsqueeze(1),
            size=img_bgr.shape[:2],
            mode="bicubic",
            align_corners=False,
        )
        
        return prediction.squeeze().cpu().numpy()

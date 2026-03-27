from ultralytics import YOLO
import cv2
import numpy as np

class Detector:
    VESSEL_CLASSES = ['bowl', 'cup', 'wine glass', 'bottle', 'vase']
    FOOD_CLASSES = ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']

    def __init__(self, vessel_model_path='models/yolov10n.pt', surface_model_path='models/yolo26n-seg.pt', device='cpu'):
        self.vessel_model = YOLO(vessel_model_path)
        self.surface_model = YOLO(surface_model_path)
        self.device = device

    def detect_vessels(self, img_path, conf=0.2):
        results = self.vessel_model(img_path, device=self.device, conf=conf)
        vessels = []
        for r in results:
            for box in r.boxes:
                label = self.vessel_model.names[int(box.cls[0])]
                if label in self.VESSEL_CLASSES:
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                    vessels.append({
                        "label": label,
                        "box": (x1, y1, x2, y2),
                        "conf": float(box.conf[0]),
                        "area": (x2 - x1) * (y2 - y1)
                    })
        return vessels

    def detect_surfaces(self, img_path, vessel_box, conf=0.1):
        vx1, vy1, vx2, vy2 = vessel_box
        results = self.surface_model(img_path, device=self.device, conf=conf)
        
        img = cv2.imread(img_path)
        h, w = img.shape[:2]
        content_mask = np.zeros((h, w), dtype=np.uint8)
        surface_found = False
        
        for r in results:
            if r.masks is None: continue
            for i, mask_obj in enumerate(r.masks.data):
                label = self.surface_model.names[int(r.boxes[i].cls[0])]
                m = cv2.resize(mask_obj.cpu().numpy(), (w, h))
                m = (m > 0.5).astype(np.uint8)
                
                m_ys, m_xs = np.where(m > 0)
                if len(m_xs) == 0: continue
                
                in_box_ratio = np.mean((m_xs >= vx1) & (m_xs <= vx2) & (m_ys >= vy1) & (m_ys <= vy2))
                
                if in_box_ratio > 0.5:
                    if label in self.FOOD_CLASSES or label not in self.VESSEL_CLASSES:
                        content_mask = np.maximum(content_mask, m)
                        surface_found = True
        
        return content_mask, surface_found

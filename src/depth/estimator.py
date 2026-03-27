from transformers import pipeline
from PIL import Image
import numpy as np
import cv2

class DepthEstimator:
    def __init__(self, model_id="depth-anything/Depth-Anything-V2-Small-hf", device='cpu'):
        self.pipe = pipeline("depth-estimation", model=model_id, device=device)

    def estimate(self, img_bgr):
        pil_img = Image.fromarray(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB))
        output = self.pipe(pil_img)
        return np.array(output["depth"])

import cv2
import numpy as np
import scipy.signal

def detect_liquid_level_v2(img, vessel_box, depth_map):
    vx1, vy1, vx2, vy2 = vessel_box
    v_height = vy2 - vy1
    v_width = vx2 - vx1
    
    if v_height <= 0 or v_width <= 0:
        return None, None

    # Depth Gradient Analysis
    depth_roi = depth_map[vy1:vy2, vx1:vx2].astype(float)
    depth_profile = np.mean(depth_roi, axis=1)
    depth_profile_smooth = scipy.signal.savgol_filter(depth_profile, min(11, len(depth_profile)), 3)
    depth_grad = np.abs(np.gradient(depth_profile_smooth))
    
    # Horizontal Edge Analysis
    img_roi = img[vy1:vy2, vx1:vx2]
    gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    edge_profile = np.sum(edges, axis=1)
    
    # Combined Scoring
    start_idx = int(v_height * 0.1)
    end_idx = int(v_height * 0.9)
    if end_idx <= start_idx: return None, None
        
    norm_depth_grad = (depth_grad - depth_grad.min()) / (depth_grad.max() - depth_grad.min() + 1e-6)
    norm_edge_profile = (edge_profile - edge_profile.min()) / (edge_profile.max() - edge_profile.min() + 1e-6)
    combined_score = (norm_depth_grad * 0.4) + (norm_edge_profile * 0.6)
    
    best_relative_y = np.argmax(combined_score[start_idx:end_idx]) + start_idx
    best_absolute_y = vy1 + best_relative_y
    
    liquid_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    liquid_mask[best_absolute_y:vy2, vx1:vx2] = 1
    
    return best_absolute_y, liquid_mask

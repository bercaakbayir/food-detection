import cv2
import argparse
import numpy as np
import torch
from ultralytics import YOLO
from PIL import Image
from transformers import pipeline
import scipy.signal


def calculate_fullness_sampling(content_mask, vessel_mask):
    """
    Calculates fullness based on the area ratio of the content vs the vessel's internal area.
    """
    if content_mask is None or vessel_mask is None:
        return 0
    
    content_area = np.sum(content_mask)
    vessel_area = np.sum(vessel_mask)
    
    if vessel_area == 0:
        return 0
        
    fullness = (content_area / vessel_area) * 100
    return int(np.clip(fullness, 0, 100))


def detect_liquid_level(img, vessel_box, depth_map):
    """
    Detects the liquid level (meniscus) inside a vessel using depth gradients 
    and horizontal edge detection.
    """
    vx1, vy1, vx2, vy2 = vessel_box
    v_height = vy2 - vy1
    v_width = vx2 - vx1
    
    if v_height <= 0 or v_width <= 0:
        return None, None

    # 1. Depth Gradient Analysis
    # Crop depth map to vessel ROI
    depth_roi = depth_map[vy1:vy2, vx1:vx2].astype(float)
    
    # Calculate vertical profile (mean depth per row)
    depth_profile = np.mean(depth_roi, axis=1)
    
    # Smooth profile to reduce noise
    depth_profile_smooth = scipy.signal.savgol_filter(depth_profile, min(11, len(depth_profile)), 3)
    
    # Calculate vertical gradient
    depth_grad = np.abs(np.gradient(depth_profile_smooth))
    
    # 2. Horizontal Edge Analysis
    img_roi = img[vy1:vy2, vx1:vx2]
    gray_roi = cv2.cvtColor(img_roi, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray_roi, 50, 150)
    
    # Project edges horizontally to find lines
    edge_profile = np.sum(edges, axis=1)
    
    # 3. Combine Scores
    # We look for a peak in both depth gradient and edge profile
    # Focus on the middle 80% to avoid top/bottom rim artifacts
    start_idx = int(v_height * 0.1)
    end_idx = int(v_height * 0.9)
    
    if end_idx <= start_idx:
        return None, None
        
    # Normalize and combine
    norm_depth_grad = (depth_grad - depth_grad.min()) / (depth_grad.max() - depth_grad.min() + 1e-6)
    norm_edge_profile = (edge_profile - edge_profile.min()) / (edge_profile.max() - edge_profile.min() + 1e-6)
    
    combined_score = (norm_depth_grad * 0.4) + (norm_edge_profile * 0.6)
    
    # Find peak in the restricted range
    best_relative_y = np.argmax(combined_score[start_idx:end_idx]) + start_idx
    best_absolute_y = vy1 + best_relative_y
    
    # Create a mask for the liquid (from bottom of glass to detected level)
    liquid_mask = np.zeros(img.shape[:2], dtype=np.uint8)
    # Fill from the detected level to the approximate bottom of the vessel
    # We use a simple rectangle for the mask within the vessel box
    liquid_mask[best_absolute_y:vy2, vx1:vx2] = 1
    
    return best_absolute_y, liquid_mask


def run_pipeline(path, distance):
    # Focal Length Calibration (Default 800)
    FOCAL_LENGTH = 800
    
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    if torch.cuda.is_available(): device = "cuda"
    print(f"Using device: {device}")

    print("Loading YOLO models...")
    # Vessel detection model
    vessel_model = YOLO('yolov10n.pt')
    # Surface segmentation model
    surface_model = YOLO('yolo26n-seg.pt')
    
    # Load Depth Estimation model
    print("Loading Depth Estimation model (Depth-Anything-V2-Small)...")
    depth_estimator = pipeline("depth-estimation", model="depth-anything/Depth-Anything-V2-Small-hf", device=device)
    
    # Classes considered as vessels
    VESSEL_CLASSES = ['bowl', 'cup', 'wine glass', 'bottle', 'vase']
    # Classes considered as food/drink surface (standard COCO food classes)
    FOOD_CLASSES = ['banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza', 'donut', 'cake']
    
    # 1. Detect Vessels with YOLOv10
    vessel_results = vessel_model(path, device=device, conf=0.2)
    img = cv2.imread(path)
    img_height, img_width = img.shape[:2]

    vessels = []
    for r in vessel_results:
        for box in r.boxes:
            label = vessel_model.names[int(box.cls[0])]
            if label in VESSEL_CLASSES:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy().astype(int)
                conf = float(box.conf[0])
                vessels.append({
                    "label": label,
                    "box": (x1, y1, x2, y2),
                    "conf": conf,
                    "area": (x2 - x1) * (y2 - y1)
                })

    if not vessels:
        print("No vessel detected with YOLOv10.")
        cv2.imwrite("result.jpg", img)
        return

    # Select largest vessel
    main_vessel = max(vessels, key=lambda x: x['area'])
    vx1, vy1, vx2, vy2 = main_vessel["box"]
    
    print(f"Vessel detected: {main_vessel['label']} ({main_vessel['conf']:.2f})")

    # 2. Depth Estimation (needed for both distance and liquid detection)
    print("Estimating depth...")
    pil_img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    depth_output = depth_estimator(pil_img)
    depth_map = np.array(depth_output["depth"])

    # 3. Detect Surface with YOLO26-seg (YOLOv8-seg)
    surface_results = surface_model(path, device=device, conf=0.1) # Lower threshold for surfaces
    
    content_mask = np.zeros((img_height, img_width), dtype=np.uint8)
    surface_found = False
    vessel_mask_obj = None
    using_fallback = False
    
    for r in surface_results:
        if r.masks is None:
            continue
            
        for i, mask_obj in enumerate(r.masks.data):
            label = surface_model.names[int(r.boxes[i].cls[0])]
            
            # Binary mask for current object
            m = mask_obj.cpu().numpy()
            m = cv2.resize(m, (img_width, img_height))
            m = (m > 0.5).astype(np.uint8)
            
            # Calculate intersection with vessel box (from YOLOv10)
            m_ys, m_xs = np.where(m > 0)
            if len(m_xs) == 0: continue
            
            in_box_ratio = np.mean((m_xs >= vx1) & (m_xs <= vx2) & (m_ys >= vy1) & (m_ys <= vy2))
            
            if in_box_ratio > 0.5:
                if label in FOOD_CLASSES:
                    content_mask = np.maximum(content_mask, m)
                    surface_found = True
                    print(f"Surface content detected (class: {label})")
                elif label in VESSEL_CLASSES and vessel_mask_obj is None:
                    # Keep track of the vessel mask from YOLO26-seg for area calculation
                    vessel_mask_obj = m
                elif label not in VESSEL_CLASSES:
                    # Any other non-vessel object inside the vessel
                    content_mask = np.maximum(content_mask, m)
                    surface_found = True
                    print(f"Surface content detected (class: {label} - inside vessel)")

    # 3b. Fallback for Transparent Liquids (Water Detection)
    if not surface_found:
        print("No standard content detected. Attempting liquid surface detection...")
        fill_y, liquid_mask = detect_liquid_level(img, (vx1, vy1, vx2, vy2), depth_map)
        if fill_y is not None:
            content_mask = liquid_mask
            surface_found = True
            using_fallback = True
            print(f"Liquid surface detected at y={fill_y}")

    # 4. Automatic Distance Estimation
    print("Calculating distance...")
    
    # Calculate pixel dimensions 
    vessel_pixel_width = vx2 - vx1
    vessel_pixel_height = vy2 - vy1

    # Get depth at the center of the vessel
    center_x = (vx1 + vx2) // 2
    center_y = (vy1 + vy2) // 2
    
    # Ensure coordinates are within image bounds
    center_x = np.clip(center_x, 0, img_width - 1)
    center_y = np.clip(center_y, 0, img_height - 1)
    
    # rel_depth = depth_map[center_y, center_x] 
    
    if distance is None:
        # Calibrate based on the vessel type
        standard_sizes = {
            'bowl': 18,      # cm width
            'cup': 8,        # cm width
            'wine glass': 7, # cm width
            'bottle': 7,     # cm width
            'vase': 12       # cm width
        }
        
        standard_width = standard_sizes.get(main_vessel['label'], 10)
        # distance = (standard_width * focal_length) / pixel_width
        est_distance = (standard_width * FOCAL_LENGTH) / vessel_pixel_width
        print(f"Estimated distance via {main_vessel['label']} standard size: {est_distance:.1f} cm")
        calc_distance = est_distance
    else:
        calc_distance = distance

    # 4. Size Calculation
    actual_width_cm = (vessel_pixel_width * calc_distance) / FOCAL_LENGTH
    actual_height_cm = (vessel_pixel_height * calc_distance) / FOCAL_LENGTH

    # 5. Fullness Calculation
    fullness = 0
    if surface_found:
        # Vertical Extent Heuristic:
        # Use the highest point of any detected content as the "fill level"
        ys, xs = np.where(content_mask > 0)
        highest_content_y = np.min(ys)
        
        # Calculate height from vessel bottom (vy2) to highest content point
        content_v_height = vy2 - highest_content_y
        vessel_v_height = vy2 - vy1
        
        if vessel_v_height > 0:
            fullness = int(np.clip((content_v_height / vessel_v_height) * 100, 0, 100))
        
        print(f"Fullness estimated via vertical extent: {fullness}%")

    print(f"\nObject: {main_vessel['label'].upper()}")
    print(f"Size:   {actual_width_cm:.2f} cm wide x {actual_height_cm:.2f} cm high")
    print(f"Dist:   {calc_distance:.1f} cm (estimated)")
    method = "Hybrid-Depth" if using_fallback else "YOLO26-seg"
    print(f"Fill:   {fullness}% full (detected by {method})")

    # 5. Visualization
    annotated_img = img.copy()
    cv2.rectangle(annotated_img, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
    
    if surface_found:
        overlay = annotated_img.copy()
        overlay[content_mask > 0] = (0, 255, 0)
        cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0, annotated_img)
        
    label_text = f"{actual_width_cm:.1f}x{actual_height_cm:.1f}cm | {fullness}% Full"
    cv2.putText(annotated_img, label_text, (vx1, vy1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    output_path = "result.jpg"
    cv2.imwrite(output_path, annotated_img)
    print(f"Result saved to {output_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True)
    parser.add_argument("--distance", type=float, default=None)
    args = parser.parse_args()
    run_pipeline(args.path, args.distance)


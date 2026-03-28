from PIL import Image, ExifTags
import numpy as np

class Calculator:
    # Fallback focal length if EXIF is missing (standard mobile FOV ~65 deg)
    DEFAULT_FOV = 65

    @staticmethod
    def get_camera_params(image_path, fov_override=None):
        """
        Extracts focal length and image dimensions from EXIF data.
        Returns: focal_length_px, width, height
        """
        try:
            img = Image.open(image_path)
            w, h = img.size
            
            if fov_override:
                f_px = (w / 2) / np.tan(np.deg2rad(fov_override / 2))
                return f_px, w, h
                
            exif = img._getexif()
            if exif:
                # 37386 = FocalLength, 41989 = FocalLengthIn35mmFilm
                f_mm = exif.get(37386) or exif.get(41989)
                if isinstance(f_mm, tuple): f_mm = f_mm[0] / f_mm[1]
                
                # If we have 35mm equivalent
                if 41989 in exif:
                    f_35mm = exif[41989]
                    f_px = (f_35mm * w) / 36.0
                elif f_mm:
                    # Generic heuristic if we don't have sensor width: assume 1/2.3" sensor (~6.17mm)
                    f_px = (f_mm * w) / 6.17

            if f_px is None:
                # Fallback to FOV-based focal length
                f_px = (w / 2) / np.tan(np.deg2rad(Calculator.DEFAULT_FOV / 2))
                
            return f_px, w, h
        except Exception as e:
            print(f"Warning: EXIF parsing failed ({e}). Using default FOV.")
            return 800, 640, 480 # Safe defaults

    @staticmethod
    def get_metric_distance(depth_map, mask):
        """
        Extracts the median distance (in meters) to the object from the metric depth map.
        """
        object_depths = depth_map[mask > 0]
        if len(object_depths) == 0:
            return 1.0 # Fallback 1 meter
        return float(np.median(object_depths))

    @staticmethod
    def calculate_dimensions(vessel_pixel_width, vessel_pixel_height, distance_m, focal_px):
        """
        Calculates physical dimensions in cm using metric depth and focal length.
        W_cm = (W_px * Dist_cm) / Focal_px
        """
        dist_cm = distance_m * 100.0
        width = (vessel_pixel_width * dist_cm) / focal_px
        height = (vessel_pixel_height * dist_cm) / focal_px
        return width, height

    @staticmethod
    def calculate_fullness(content_mask, vessel_box):
        vx1, vy1, vx2, vy2 = vessel_box
        ys, _ = np.where(content_mask > 0)
        if len(ys) == 0: return 0
        
        highest_content_y = np.min(ys)
        content_v_height = vy2 - highest_content_y
        vessel_v_height = vy2 - vy1
        
        if vessel_v_height <= 0: return 0
        return int(np.clip((content_v_height / vessel_v_height) * 100, 0, 100))

    @staticmethod
    def calculate_volume(vessel_label, width_cm, height_cm, fullness_pct):
        """Estimates volume in milliliters (ml). 1 cm^3 = 1 ml"""
        radius = width_cm / 2.0
        fullness_factor = fullness_pct / 100.0
        
        if vessel_label in ['cup', 'wine glass', 'bottle', 'vase']:
            total_volume = np.pi * (radius**2) * height_cm
            estimated_volume = total_volume * fullness_factor
        elif vessel_label == 'bowl':
            total_volume = (2.0/3.0) * np.pi * (radius**2) * height_cm
            estimated_volume = total_volume * fullness_factor
        else:
            total_volume = width_cm * width_cm * height_cm * 0.6
            estimated_volume = total_volume * fullness_factor
            
        return int(max(0.0, float(estimated_volume)))

import numpy as np

class Calculator:
    FOCAL_LENGTH = 800
    STANDARD_SIZES = {
        'bowl': 18,
        'cup': 8,
        'wine glass': 7,
        'bottle': 7,
        'vase': 12
    }

    @staticmethod
    def estimate_distance(vessel_label, vessel_pixel_width):
        standard_width = Calculator.STANDARD_SIZES.get(vessel_label, 10)
        return (standard_width * Calculator.FOCAL_LENGTH) / vessel_pixel_width

    @staticmethod
    def calculate_dimensions(vessel_pixel_width, vessel_pixel_height, distance):
        width = (vessel_pixel_width * distance) / Calculator.FOCAL_LENGTH
        height = (vessel_pixel_height * distance) / Calculator.FOCAL_LENGTH
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
        """
        Estimates volume in milliliters (ml) based on vessel geometry and fullness.
        1 cm^3 = 1 ml
        """
        radius = width_cm / 2.0
        fullness_factor = fullness_pct / 100.0
        
        if vessel_label in ['cup', 'wine glass', 'bottle', 'vase']:
            # Modeling as a cylinder: V = pi * r^2 * h
            total_volume = np.pi * (radius**2) * height_cm
            estimated_volume = total_volume * fullness_factor
        elif vessel_label == 'bowl':
            # Modeling as a hemi-ellipsoid: V = (2/3) * pi * rx * ry * rz
            # Assuming rx=ry=radius, rz=height
            total_volume = (2.0/3.0) * np.pi * (radius**2) * height_cm
            # For a hemisphere-like shape, fullness isn't linear with height, 
            # but for simplicity and lack of exact profile, we use linear fill of volume.
            estimated_volume = total_volume * fullness_factor
        else:
            # Default: Simple box-based approximation
            total_volume = width_cm * width_cm * height_cm * 0.6 # Assuming 60% of bounding box
            estimated_volume = total_volume * fullness_factor
            
        return int(max(0.0, float(estimated_volume)))

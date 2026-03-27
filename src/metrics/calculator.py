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

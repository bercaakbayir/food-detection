import cv2

class Visualizer:
    @staticmethod
    def annotate(img, vessel_box, content_mask, label_text, surface_found):
        vx1, vy1, vx2, vy2 = vessel_box
        annotated_img = img.copy()
        cv2.rectangle(annotated_img, (vx1, vy1), (vx2, vy2), (255, 0, 0), 2)
        
        if surface_found:
            overlay = annotated_img.copy()
            # Green overlay for content
            overlay[content_mask > 0] = (0, 255, 0)
            cv2.addWeighted(overlay, 0.4, annotated_img, 0.6, 0, annotated_img)
            
        cv2.putText(annotated_img, label_text, (vx1, vy1 - 10), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        return annotated_img

    @staticmethod
    def save(img, image_name):
        import os
        # results/<image-name>/result.jpg
        base_name = os.path.splitext(os.path.basename(image_name))[0]
        output_dir = os.path.join("results", base_name)
        os.makedirs(output_dir, exist_ok=True)
        
        output_path = os.path.join(output_dir, "result.jpg")
        cv2.imwrite(output_path, img)
        print(f"Result saved to {output_path}")

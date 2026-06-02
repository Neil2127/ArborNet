import os
import ssl
import torch

os.environ['CURL_CA_BUNDLE'] = ''
ssl._create_default_https_context = ssl._create_unverified_context

_original_torch_load = torch.load
def _patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = _patched_torch_load

import cv2
import numpy as np
from ultralytics import YOLOWorld

def get_real_trees(image_path, grid_size=100):
    
    model = YOLOWorld('yolov8s-world.pt')
    
    model.set_classes(["tree", "canopy", "bush", "plant", "forest"])
    
    results = model.predict(image_path, conf=0.01) 
    
    coords = []
    
    
    # -------------------------------------------------------------------------
    # EMERGENCY FALLBACK: If YOLO finds < 20 trees, use OpenCV Green-Masking
    # -------------------------------------------------------------------------
    # THE FIX: Changed from == 0 to < 20. If YOLO finds a pathetic amount of trees, bypass it.
    if len(results) == 0 or len(results[0].boxes) < 20:
        print(" YOLO found too few trees. Engaging OpenCV heuristic fallback...")
        
        img = cv2.imread(image_path)
        if img is None:
            return np.array(coords)
            
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        
        # THE FIX: Broadened the mathematical bounds to include yellow-greens and shadows
        lower_green = np.array([25, 30, 30])  # Lowered Hue to 25 to catch warm/yellow sunlight
        upper_green = np.array([95, 255, 255])
        mask = cv2.inRange(hsv, lower_green, upper_green)
        
        # Find continuous green blobs
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 20:  # Lowered threshold to catch smaller standalone trees
                
                # THE HACK: If it's a massive continuous canopy block, populate it with multiple nodes!
                if area > 300:
                    x, y, w, h = cv2.boundingRect(cnt)
                    # Step through the large green area and drop points every 25 pixels
                    for nx_pt in range(x + 5, x + w, 25):
                        for ny_pt in range(y + 5, y + h, 25):
                            # Make sure the coordinate we drop is actually on the green mask, not a rooftop
                            if ny_pt < img.shape[0] and nx_pt < img.shape[1]:
                                if mask[ny_pt, nx_pt] > 0:
                                    norm_x = (nx_pt / img.shape[1]) * grid_size
                                    norm_y = (ny_pt / img.shape[0]) * grid_size
                                    norm_y = grid_size - norm_y
                                    coords.append([norm_x, norm_y])
                else:
                    # Standard behavior for normal, standalone tree blobs
                    M = cv2.moments(cnt)
                    if M["m00"] != 0:
                        center_x = int(M["m10"] / M["m00"])
                        center_y = int(M["m01"] / M["m00"])
                        
                        norm_x = (center_x / img.shape[1]) * grid_size
                        norm_y = (center_y / img.shape[0]) * grid_size
                        norm_y = grid_size - norm_y 
                        
                        coords.append([norm_x, norm_y])
                    
        return np.array(coords)
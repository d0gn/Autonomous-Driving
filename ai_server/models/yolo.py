import torch
import cv2
import numpy as np
from pathlib import Path

class YOLODetector:
    def __init__(self, weights_path='yolov5s.pt', conf_thres=0.25, img_size=416, device='cpu'):
        self.device = device
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
        self.model.to(self.device)
        self.model.conf = conf_thres
        self.model.iou = 0.45
        self.model.imgsz = img_size

    def detect_image(self, image_path):
        
        img = cv2.imread(image_path)
        if img is None:
            raise ValueError(f"Image not found at: {image_path}")

        results = self.model(img)
        annotated_img = results.render()[0]  # numpy 배열
        return results, annotated_img

    def detect_array(self, img_array):
        
        results = self.model(img_array)
        annotated_img = results.render()[0]
        return results, annotated_img

    def save_result(self, annotated_img, save_path='output.jpg'):
        cv2.imwrite(save_path, annotated_img)
        print(f"[INFO] Saved result image to: {save_path}")

    def extract_detections(self, results):
        
        detections = []
        for *xyxy, conf, cls in results.xyxy[0]:
            detections.append({
                'bbox': [float(x.item()) for x in xyxy],
                'confidence': float(conf.item()),
                'class_id': int(cls.item()),
                'class_name': self.model.names[int(cls.item())]
            })
        return detections

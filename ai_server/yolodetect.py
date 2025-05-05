
# --- 필요한 라이브러리 임포트 ---
# 표준 라이브러리
import os
import sys
import time
# import argparse # 서버 실행 시 필수는 아니지만, 필요시 사용
from pathlib import Path
#aaa
# 서드파티 라이브러리
from flask import Flask, request, jsonify
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
from PIL import Image 
# import glob
# import base64 

# PyTorch 및 관련 라이브러리
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
# import torch.optim # 추론 서버이므로 최적화는 필요 없음
from torchvision import transforms
# 별도 파일(예: yolo_detector.py)로 분리하여 임포트하는 것이 더 깔끔합니다.
# 여기서는 요청에 따라 서버 코드 파일 안에 포함시킵니다.
class YOLODetector:
    def __init__(self, weights_path='yolov5s.pt', conf_thres=0.25, img_size=640, device='cpu'): # img_size 기본값 640으로 변경
        self.device = device
        print(f"💡 YOLODetector 사용 장치: {self.device}")

        # 가중치 파일 존재 확인 (torch.hub 자동 다운로드 기능을 사용할 경우 생략 가능)
        if not os.path.exists(weights_path):
             print(f"🚨 경고: YOLO 가중치 파일이 로컬에 없습니다: {weights_path}")
             print("torch.hub에서 표준 모델 이름으로 다운로드 시도합니다.")
             # 로컬에 없으면 파일 이름만 사용하여 torch.hub 자동 다운로드 시도
             weights_path = os.path.basename(weights_path)
             if not weights_path.endswith('.pt'): # .pt 확장자가 없으면 표준 모델 이름으로 간주
                  # 예를 들어 'yolov5s' 같은 이름
                 pass # torch.hub가 알아서 다운로드할 것이라고 가정
             else: # .pt 확장자가 있는데 로컬에 없으면 문제가 있을 수 있음
                  print(f"🚨 오류: YOLO 가중치 파일({weights_path})이 로컬에 없으며 표준 모델 이름이 아닐 수 있습니다.")
                  # 오류 발생 또는 모델 로딩 실패로 이어질 수 있습니다.

        try:
            # torch.hub를 사용하여 모델 로딩
            # force_reload=True 로 설정하면 매번 모델을 새로 다운로드
            self.model = torch.hub.load('ultralytics/yolov5', 'custom', path=weights_path, force_reload=False)
            self.model.to(self.device)
            self.model.eval() # 추론 모드 설정 (매우 중요)
            self.model.conf = conf_thres # 객체 검출 신뢰도 임계값
            self.model.iou = 0.45 # Non-Maximum Suppression (NMS) IoU 임계값
            # YOLOv5 모델의 입력 이미지 사이즈 설정
            self.model.imgsz = img_size
            print(f"✅ YOLO 모델 로딩 완료: {weights_path}, img_size={self.model.imgsz}")
        except Exception as e:
             print(f"❌ YOLO 모델 로딩 실패: {e}")
             self.model = None # 로딩 실패 시 모델을 None으로 설정하여 이후 호출에서 오류 방지
             # 실제 운영 환경에서는 여기서 예외를 다시 발생시키거나 서버를 종료하는 것을 고려해야 합니다.


    def detect_array(self, img_array):
        
        if self.model is None:
            print("🚨 YOLO 모델이 로딩되지 않았습니다. 객체 검출 건너뜜.")
            return None, None

        if img_array is None or img_array.size == 0:
             print("🚨 detect_array: 유효하지 않은 입력 이미지입니다.")
             return None, None

        try:
            results = self.model(img_array)
            annotated_img = results.render()[0] # 배치 중 첫 번째 이미지 결과
            return results, annotated_img
        except Exception as e:
            print(f"❌ YOLO detect_array 중 오류 발생: {e}")
            return None, None

    def extract_detections(self, results):
        detections = []
        if results is not None and hasattr(results, 'xyxy') and len(results.xyxy) > 0:
            # results.xyxy[0]는 배치 중 첫 번째 이미지의 검출 결과 [x1, y1, x2, y2, confidence, class_id]
            for *xyxy, conf, cls in results.xyxy[0]:
                detections.append({
                    'bbox': [float(x.item()) for x in xyxy], # 바운딩 박스 좌표 [x1, y1, x2, y2]
                    'confidence': float(conf.item()),       # 신뢰도
                    'class_id': int(cls.item()),            # 클래스 ID
                    'class_name': self.model.names[int(cls.item())] if self.model and hasattr(self.model, 'names') else f'class_{int(cls.item())}' # 클래스 이름
                })
        return detections
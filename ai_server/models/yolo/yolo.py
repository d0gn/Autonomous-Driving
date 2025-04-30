import sys
import os
import cv2

# yolov5-master 경로 추가
yolo_path = os.path.join(os.path.dirname(__file__), 'yolov5-master')
sys.path.append(yolo_path)

# lineDetection 경로 추가
line_path = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'lineDetection'))
sys.path.append(line_path)

# YOLOv5 객체 감지 실행 함수 및 차선 감지 클래스 import
from detect import run
from lineDetection import LaneDetector

# 차선 감지 + 객체 감지 수행 함수
def detect_and_show(img_path):
    frame = cv2.imread(img_path)
    if frame is None:
        print(f"❌ 이미지를 찾을 수 없습니다: {img_path}")
        return

    detector = LaneDetector()
    lane_image = detector.detect_lane(frame)

    # YOLO로 넘기기 전에 임시 이미지 저장
    temp_path = os.path.join(os.path.dirname(__file__), 'temp_lane.jpg')
    cv2.imwrite(temp_path, lane_image)

    run(
        weights=os.path.join(yolo_path, 'runs/train/self_driving_car_yolov5_result3/weights/best.pt'),
        source=temp_path,
        imgsz=(416, 416),
        conf_thres=0.5,
        save_txt=False,
        save_conf=False,
        nosave=False
    )

    # 가장 최신 감지 결과 폴더 경로 찾기
    detect_dir = os.path.join(yolo_path, 'runs', 'detect')
    result_folder = sorted(os.listdir(detect_dir), reverse=True)[0]
    result_path = os.path.join(detect_dir, result_folder, os.path.basename(temp_path))

    if os.path.exists(result_path):
        result_img = cv2.imread(result_path)
        cv2.imshow("Final Result", result_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print(f"❌ 결과 이미지가 존재하지 않음: {result_path}")

# test5.jpg 경로 지정 후 실행
if __name__ == "__main__":
    img_path = os.path.join(os.path.dirname(__file__), 'yolov5-master', 'test5.jpg')
    detect_and_show(img_path)

import sys
import os
import cv2

# YOLOv5 코드 경로 추가
sys.path.append(os.path.join(os.path.dirname(__file__), 'yolov5-master'))

# YOLOv5 객체 감지용 함수
from detect import run

# 차선 감지 클래스
from lineDetection import LaneDetector

def detect_from_laned_images():
    # 차선 감지기 초기화
    detector = LaneDetector(input_path='test_images', output_path='test_images_output')

    # 입력 이미지 목록 가져오기
    input_images = os.listdir(detector.input_path)
    input_images = [img for img in input_images if img.lower().endswith(('.jpg', '.png'))]

    for filename in input_images:
        input_img_path = os.path.join(detector.input_path, filename)
        img = cv2.imread(input_img_path)
        if img is None:
            print(f"❌ 이미지 로드 실패: {input_img_path}")
            continue

        # 차선 감지 수행
        lane_img = detector.lane_finding_pipeline(img)

        # 차선 감지된 이미지를 임시 파일로 저장
        temp_output_path = os.path.join('test_images_output', f'lane_{filename}')
        cv2.imwrite(temp_output_path, lane_img)

        # YOLOv5 객체 감지 수행
        run(
            weights='yolov5-master/runs/train/self_driving_car_yolov5_result3/weights/best.pt',
            source=temp_output_path,
            imgsz=(416, 416),
            conf_thres=0.5,
            save_txt=False,
            save_conf=False,
            nosave=False
        )

        # YOLO 결과 디렉토리에서 결과 이미지 불러오기
        detect_dir = os.path.join('yolov5-master', 'runs', 'detect')
        result_folder = sorted(os.listdir(detect_dir), reverse=True)[0]
        result_img_path = os.path.join(detect_dir, result_folder, f'lane_{filename}')

        if os.path.exists(result_img_path):
            result_img = cv2.imread(result_img_path)
            if result_img is not None:
                cv2.imshow("Final Result", result_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            else:
                print("❌ 결과 이미지를 불러올 수 없습니다.")
        else:
            print(f"❌ 결과 이미지가 존재하지 않음: {result_img_path}")

# 실행
if __name__ == "__main__":
    detect_from_laned_images()

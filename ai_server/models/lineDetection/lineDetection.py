import os
import cv2
import numpy as np
import matplotlib.image as mpimg
import re

class LaneDetector:
    def __init__(self, input_path='test_images/', output_path='test_images_output/'):
        self.input_path = input_path
        self.output_path = output_path

    def grayscale(self, img):
        return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

    def gaussian_blur(self, img, kernel_size):
        return cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)

    def canny(self, img, low_threshold, high_threshold):
        return cv2.Canny(img, low_threshold, high_threshold)

    def region_of_interest(self, img, vertices):
        mask = np.zeros_like(img)
        if len(img.shape) > 2:
            ignore_mask_color = (255,) * img.shape[2]
        else:
            ignore_mask_color = 255
        cv2.fillPoly(mask, vertices, ignore_mask_color)
        return cv2.bitwise_and(img, mask)

    def hough_lines(self, img, rho, theta, threshold, min_line_len, max_line_gap):
        lines = cv2.HoughLinesP(img, rho, theta, threshold, np.array([]),
                                minLineLength=min_line_len, maxLineGap=max_line_gap)
        line_img = np.zeros((img.shape[0], img.shape[1], 3), dtype=np.uint8)
        self.draw_lines(line_img, lines)
        return line_img

    def draw_lines(self, img, lines, thickness=3):
        left_lines = []
        right_lines = []

        for line in lines:
            for x1, y1, x2, y2 in line:
                slope = (y2 - y1) / (x2 - x1 + 1e-6)
                if -0.8 < slope < -0.2:
                    left_lines.append(line)
                elif 0.2 < slope < 0.8:
                    right_lines.append(line)

        if left_lines:
            LL_avg = np.mean(left_lines, axis=0)[0]
            x1, y1, x2, y2 = LL_avg
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            intercept = y1 - slope * x1
            y1, y2 = 539, 320
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            cv2.line(img, (x1, y1), (x2, y2), [255, 255, 0], thickness)

        if right_lines:
            RL_avg = np.mean(right_lines, axis=0)[0]
            x1, y1, x2, y2 = RL_avg
            slope = (y2 - y1) / (x2 - x1 + 1e-6)
            intercept = y1 - slope * x1
            y1, y2 = 539, 320
            x1 = int((y1 - intercept) / slope)
            x2 = int((y2 - intercept) / slope)
            cv2.line(img, (x1, y1), (x2, y2), [0, 0, 255], thickness)

    def weighted_img(self, img, initial_img, α=0.5, β=1.0, λ=0.):
        return cv2.addWeighted(img, α, initial_img, β, λ)

    def lane_finding_pipeline(self, img):
        gray = self.grayscale(img)
        blur = self.gaussian_blur(gray, kernel_size=5)
        edges = self.canny(blur, 70, 140)
        cv2.imshow("Canny Edges", edges)  # 디버깅용: Canny 결과 확인
        cv2.waitKey(0)
        vertices = np.array([[(0, 539), (425, 340), (510, 320), (959, 539)]], dtype=np.int32)
        masked = self.region_of_interest(edges, vertices)
        lines = self.hough_lines(masked, 1, np.pi / 180, 12, 10, 2)

        if lines is None:
            print("❌ HoughLinesP가 선을 찾지 못했습니다.")
            return img  # 기본 이미지 반환

        output = self.weighted_img(lines, img)
        return output

    def detect_lane(self, img):
        return self.lane_finding_pipeline(img)

    def read_and_save_images(self):
        images = os.listdir(self.input_path)
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)

        for filename in images:
            # 파일 확장자를 검사하고 PNG도 처리 가능하게 수정
            image_path = os.path.join(self.input_path, filename)
            if filename.lower().endswith(('png', 'jpg', 'jpeg')):
                image = mpimg.imread(image_path)
                result = self.lane_finding_pipeline(image)

                # 파일 확장자에 맞게 저장
                save_name = re.sub(r'\.(jpg|jpeg|png)$', '', filename) + '_output.png'
                mpimg.imsave(os.path.join(self.output_path, save_name), result)

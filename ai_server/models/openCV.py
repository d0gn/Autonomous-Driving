import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        pass

    def detect_lane(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, 50, 150)
        height, width = frame.shape[:2]
        roi = np.array([[(0, height), (width // 2 - 50, height // 2 + 50),
                         (width // 2 + 50, height // 2 + 50), (width, height)]], dtype=np.int32)
        mask = np.zeros_like(edges)
        cv2.fillPoly(mask, roi, 255)
        masked_edges = cv2.bitwise_and(edges, mask)
        lines = cv2.HoughLinesP(masked_edges, 1, np.pi / 180, threshold=50, minLineLength=50, maxLineGap=150)
        line_image = np.zeros_like(frame)
        if lines is not None:
            for line in lines:
                x1, y1, x2, y2 = line[0]
                cv2.line(line_image, (x1, y1), (x2, y2), (0, 255, 0), 5)
        combined = cv2.addWeighted(frame, 0.8, line_image, 1, 1)
        return combined

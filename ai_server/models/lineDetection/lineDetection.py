import cv2
import numpy as np

class LaneDetector:
    def __init__(self):
        self.low_threshold = 50
        self.high_threshold = 150
        self.rho = 1
        self.theta = np.pi / 180
        self.threshold = 50
        self.min_line_length = 100
        self.max_line_gap = 50

    def region_of_interest(self, img):
        height, width = img.shape[:2]
        polygons = np.array([[(0, height), (width, height), (width, int(height*0.6)), (0, int(height*0.6))]])
        mask = np.zeros_like(img)
        cv2.fillPoly(mask, polygons, 255)
        return cv2.bitwise_and(img, mask)

    def average_slope_intercept(self, lines, img_shape):
        left, right = [], []
        if lines is None:
            return None, None
        for line in lines:
            x1, y1, x2, y2 = line[0]
            if x2 - x1 == 0: continue
            slope = (y2 - y1) / (x2 - x1)
            intercept = y1 - slope * x1
            if slope < -0.3: left.append((slope, intercept))
            elif slope > 0.3: right.append((slope, intercept))
        return (self.make_line_from_avg(left, img_shape) if left else None,
                self.make_line_from_avg(right, img_shape) if right else None)

    def make_line_from_avg(self, lines, img_shape):
        slope, intercept = np.mean(lines, axis=0)
        y1, y2 = img_shape[0], int(img_shape[0] * 0.6)
        x1 = int((y1 - intercept) / slope)
        x2 = int((y2 - intercept) / slope)
        return np.array([x1, y1, x2, y2])

    def draw_curve(self, img, line):
        x1, y1, x2, y2 = line
        points = np.array([
            [x1, y1],
            [int((2*x1 + x2)/3), int((2*y1 + y2)/3)],
            [int((x1 + 2*x2)/3), int((y1 + 2*y2)/3)],
            [x2, y2]
        ])
        poly = np.polyfit(points[:, 1], points[:, 0], 2)
        plot_y = np.linspace(y2, y1, 50, dtype=int)
        plot_x = poly[0]*plot_y**2 + poly[1]*plot_y + poly[2]
        pts = np.array([np.transpose(np.vstack([plot_x, plot_y]))], dtype=np.int32)
        cv2.polylines(img, pts, isClosed=False, color=(0, 255, 0), thickness=5)

    def detect_lane(self, frame):
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blur = cv2.GaussianBlur(gray, (5, 5), 0)
        edges = cv2.Canny(blur, self.low_threshold, self.high_threshold)
        cropped = self.region_of_interest(edges)
        lines = cv2.HoughLinesP(cropped, self.rho, self.theta, self.threshold,
                                np.array([]), self.min_line_length, self.max_line_gap)
        left, right = self.average_slope_intercept(lines, frame.shape)
        line_img = np.zeros_like(frame)
        if left is not None: self.draw_curve(line_img, left)
        if right is not None: self.draw_curve(line_img, right)
        return cv2.addWeighted(frame, 0.8, line_img, 1, 1)

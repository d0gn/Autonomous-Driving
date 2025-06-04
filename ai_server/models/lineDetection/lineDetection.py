import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

class LaneDetector:
    def __init__(self, cal_dir='camera_cal/cal_pickle.p'):
        # 카메라 보정 정보가 저장될 경로 설정
        self.cal_dir = cal_dir
        # 곡선 방정식 계수를 저장할 리스트 초기화 (평균화에 사용)
        self.left_a, self.left_b, self.left_c = [], [], []
        self.right_a, self.right_b, self.right_c = [], [], []

    def calibrate_camera(self):
        # 카메라 보정을 위한 체스보드의 3D 기준 좌표 생성
        obj_pts = np.zeros((6 * 9, 3), np.float32)
        obj_pts[:, :2] = np.mgrid[0:9, 0:6].T.reshape(-1, 2)

        objpoints = []  # 3D 좌표
        imgpoints = []  # 2D 이미지 좌표

        # 카메라 보정용 이미지 불러오기
        images = glob.glob('camera_cal/*.jpg')
        for fname in images:
            img = cv2.imread(fname)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            ret, corners = cv2.findChessboardCorners(gray, (9, 6), None)
            if ret:
                objpoints.append(obj_pts)
                imgpoints.append(corners)

        img_size = (img.shape[1], img.shape[0])
        # 카메라 보정 실행
        ret, mtx, dist, _, _ = cv2.calibrateCamera(objpoints, imgpoints, img_size, None, None)
        # 보정 결과를 파일로 저장
        pickle.dump({'mtx': mtx, 'dist': dist}, open(self.cal_dir, 'wb'))

    def undistort(self, img):
        # 보정 정보 로딩 후 왜곡 제거
        with open(self.cal_dir, mode='rb') as f:
            data = pickle.load(f)
        mtx, dist = data['mtx'], data['dist']
        return cv2.undistort(img, mtx, dist, None, mtx)

    def pipeline(self, img, s_thresh=(100, 255), sx_thresh=(15, 255)):
        # 이미지 전처리: HLS 변환, Sobel 엣지 검출, 채널 임계값 이진화
        img = self.undistort(img)
        hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
        l_channel = hls[:, :, 1]
        s_channel = hls[:, :, 2]

        # Sobel X 방향 경계 검출
        sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1)
        scaled_sobel = np.uint8(255 * np.absolute(sobelx) / np.max(np.absolute(sobelx)))

        # X 방향 gradient 이진화
        sxbinary = np.zeros_like(scaled_sobel)
        sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1

        # S 채널 이진화
        s_binary = np.zeros_like(s_channel)
        s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1

        # 두 이진화 결과를 합쳐 최종 이진 이미지 생성
        combined_binary = np.zeros_like(sxbinary)
        combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
        return combined_binary

    def perspective_warp(self, img, dst_size=(640, 480),
                         src=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)]),
                         dst=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)])):
        # 원근 변환 수행 (탑뷰로 변환)
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src, dst = src * img_size, dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, dst_size)

    def inv_perspective_warp(self, img, dst_size=(640, 480),
                              src=np.float32([(0, 0), (1, 0), (0, 1), (1, 1)]),
                              dst=np.float32([(0.43, 0.65), (0.58, 0.65), (0.1, 1), (1, 1)])):
        # 원근 변환을 역으로 수행 (탑뷰 → 원래 시점)
        img_size = np.float32([(img.shape[1], img.shape[0])])
        src, dst = src * img_size, dst * np.float32(dst_size)
        M = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(img, M, dst_size)

    def get_hist(self, img):
        # 하단 절반에서의 수평 히스토그램 생성 (차선 위치 추정용)
        return np.sum(img[img.shape[0] // 2:, :], axis=0)

    def sliding_window(self, img, nwindows=9, margin=150, minpix=1):
        # 슬라이딩 윈도우를 이용한 차선 검출 및 2차 곡선 피팅
        histogram = self.get_hist(img)
        midpoint = int(histogram.shape[0] / 2)
        leftx_base = np.argmax(histogram[:midpoint])
        rightx_base = np.argmax(histogram[midpoint:]) + midpoint

        window_height = int(img.shape[0] / nwindows)
        nonzero = img.nonzero()
        nonzeroy, nonzerox = np.array(nonzero[0]), np.array(nonzero[1])

        leftx_current, rightx_current = leftx_base, rightx_base
        left_lane_inds, right_lane_inds = [], []

        # 윈도우를 이동하면서 차선 픽셀 수집
        for window in range(nwindows):
            win_y_low = img.shape[0] - (window + 1) * window_height
            win_y_high = img.shape[0] - window * window_height
            win_xleft_low = leftx_current - margin
            win_xleft_high = leftx_current + margin
            win_xright_low = rightx_current - margin
            win_xright_high = rightx_current + margin

            good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                              (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
            good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                               (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]

            left_lane_inds.append(good_left_inds)
            right_lane_inds.append(good_right_inds)

            # 충분한 픽셀이 있으면 중심 좌표 업데이트
            if len(good_left_inds) > minpix:
                leftx_current = int(np.mean(nonzerox[good_left_inds]))
            if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))

        # 픽셀 위치 추출 및 2차 함수 피팅
        leftx = nonzerox[np.concatenate(left_lane_inds)]
        lefty = nonzeroy[np.concatenate(left_lane_inds)]
        rightx = nonzerox[np.concatenate(right_lane_inds)]
        righty = nonzeroy[np.concatenate(right_lane_inds)]

        left_fit = np.polyfit(lefty, leftx, 2)
        right_fit = np.polyfit(righty, rightx, 2)

        # 최근 10개 프레임의 평균 계수로 smoothing
        self.left_a.append(left_fit[0])
        self.left_b.append(left_fit[1])
        self.left_c.append(left_fit[2])
        self.right_a.append(right_fit[0])
        self.right_b.append(right_fit[1])
        self.right_c.append(right_fit[2])

        left_avg = [np.mean(self.left_a[-10:]), np.mean(self.left_b[-10:]), np.mean(self.left_c[-10:])]
        right_avg = [np.mean(self.right_a[-10:]), np.mean(self.right_b[-10:]), np.mean(self.right_c[-10:])]

        # 각 차선의 좌표 계산
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        left_fitx = left_avg[0] * ploty ** 2 + left_avg[1] * ploty + left_avg[2]
        right_fitx = right_avg[0] * ploty ** 2 + right_avg[1] * ploty + right_avg[2]

        return (left_fitx, right_fitx), (left_avg, right_avg), ploty

    def get_curve(self, img, leftx, rightx):
        # 곡률 반경과 차량의 차선 중심으로부터의 위치 계산
        ploty = np.linspace(0, img.shape[0] - 1, img.shape[0])
        y_eval = np.max(ploty)
        ym_per_pix = 30.5 / 720  # y방향 단위(m/pixel)
        xm_per_pix = 3.7 / 720   # x방향 단위(m/pixel)

        left_fit_cr = np.polyfit(ploty * ym_per_pix, leftx * xm_per_pix, 2)
        right_fit_cr = np.polyfit(ploty * ym_per_pix, rightx * xm_per_pix, 2)

        left_curverad = ((1 + (2 * left_fit_cr[0] * y_eval * ym_per_pix + left_fit_cr[1]) ** 2) ** 1.5) / abs(
            2 * left_fit_cr[0])
        right_curverad = ((1 + (2 * right_fit_cr[0] * y_eval * ym_per_pix + right_fit_cr[1]) ** 2) ** 1.5) / abs(
            2 * right_fit_cr[0])

        # 차량과 차선 중심 간 거리 계산
        car_pos = img.shape[1] / 2
        l_fit_x_int = left_fit_cr[0] * img.shape[0] ** 2 + left_fit_cr[1] * img.shape[0] + left_fit_cr[2]
        r_fit_x_int = right_fit_cr[0] * img.shape[0] ** 2 + right_fit_cr[1] * img.shape[0] + right_fit_cr[2]
        lane_center_position = (r_fit_x_int + l_fit_x_int) / 2
        center = (car_pos - lane_center_position) * xm_per_pix / 10
        return (left_curverad, right_curverad, center)

    def draw_lanes(self, img, left_fit, right_fit, ploty):
        # 차선 영역을 시각적으로 이미지에 그려줌
        color_img = np.zeros_like(img)

        # 왼쪽과 오른쪽 곡선 좌표 생성
        left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
        right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
        points = np.hstack((left, right))

        # 차선 영역을 채워서 색칠
        cv2.fillPoly(color_img, np.int32([points]), (0, 200, 255))

        # 중앙 선 위 좌표를 점으로 그려서 출력
        mid_fit = (left_fit + right_fit) / 2
        mid_points = np.array([np.transpose(np.vstack([mid_fit, ploty]))], dtype=np.int32)
        for point in mid_points[0]:
            cv2.circle(color_img, (int(point[0]), int(point[1])), 1, (0, 0, 0), -1)
        print("중심좌표:", point)

        # 다시 원래 시점으로 역 원근 변환
        inv_perspective = self.inv_perspective_warp(color_img)
        return cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)

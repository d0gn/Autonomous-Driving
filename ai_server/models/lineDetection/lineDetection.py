import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt
import pickle

#카메라 왜곡을 보정하는 함수
def undistort_img():
    obj_pts = np.zeros((6*9,3), np.float32)
    obj_pts[:,:2] = np.mgrid[0:9, 0:6].T.reshape(-1,2)

    objpoints = []
    imgpoints = []

    images = glob.glob('camera_cal/*.jpg')

    for indx, fname in enumerate(images):
        img = cv2.imread(fname)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        ret, corners = cv2.findChessboardCorners(gray, (9,6), None)

        if ret == True:
            objpoints.append(obj_pts)
            imgpoints.append(corners)
    img_size = (img.shape[1], img.shape[0])

    ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, img_size, None,None)

    dst = cv2.undistort(img, mtx, dist, None, mtx)
    dist_pickle = {}
    dist_pickle['mtx'] = mtx
    dist_pickle['dist'] = dist
    pickle.dump( dist_pickle, open('camera_cal/cal_pickle.p', 'wb') )

# 저장된 보정 파라미터를 불러와 이미지 왜곡을 보정하는 함수
def undistort(img, cal_dir='camera_cal/cal_pickle.p'):
    with open(cal_dir, mode='rb') as f:
        file = pickle.load(f)
    mtx = file['mtx']
    dist = file['dist']
    dst = cv2.undistort(img, mtx, dist, None, mtx)
    
    return dst

# 이미지 전처리 파이프라인 (왜곡 보정 + HLS 변환 및 색상/경계선 임계값 처리)
def pipeline(img, s_thresh=(100, 255), sx_thresh=(15, 255)):
    img = undistort(img)
    img = np.copy(img)
    
    # HLS 색 공간으로 변환 후 V 채널 분리
    hls = cv2.cvtColor(img, cv2.COLOR_RGB2HLS).astype(float)
    l_channel = hls[:,:,1]
    s_channel = hls[:,:,2]
    h_channel = hls[:,:,0]
    
    # Sobel x 진행
    sobelx = cv2.Sobel(l_channel, cv2.CV_64F, 1, 1) 
    abs_sobelx = np.absolute(sobelx)
    scaled_sobel = np.uint8(255*abs_sobelx/np.max(abs_sobelx))
    
     # x 방향 기울기 임계값 적용
    sxbinary = np.zeros_like(scaled_sobel)
    sxbinary[(scaled_sobel >= sx_thresh[0]) & (scaled_sobel <= sx_thresh[1])] = 1
    
    # 색상 채널 임계값 적용
    s_binary = np.zeros_like(s_channel)
    s_binary[(s_channel >= s_thresh[0]) & (s_channel <= s_thresh[1])] = 1
    
    color_binary = np.dstack((np.zeros_like(sxbinary), sxbinary, s_binary)) * 255
    
    combined_binary = np.zeros_like(sxbinary)
    combined_binary[(s_binary == 1) | (sxbinary == 1)] = 1
    return combined_binary

# Bird-Eye-View 위에서 내려다본 시점으로 변환하는 함수
def perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)]),
                     dst=np.float32([(0,0), (1, 0), (0,1), (1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped

# Bird-Eye-View된 이미지를 원본시점으로 복원
def inv_perspective_warp(img, 
                     dst_size=(1280,720),
                     src=np.float32([(0,0), (1, 0), (0,1), (1,1)]),
                     dst=np.float32([(0.43,0.65),(0.58,0.65),(0.1,1),(1,1)])):
    img_size = np.float32([(img.shape[1],img.shape[0])])
    src = src* img_size
    # 출발점(src)과 목적지(dst) 점을 기반으로 원근 변환 행렬 계산
    dst = dst * np.float32(dst_size)
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(img, M, dst_size)
    return warped


# 차선위치 탐색을 위한 이미지 하단 밝기합산을 통한 히스토그램생성
def get_hist(img):
    hist = np.sum(img[img.shape[0]//2:,:], axis=0)
    return hist

left_a, left_b, left_c = [],[],[]
right_a, right_b, right_c = [],[],[]

# 슬라이딩 윈도우를 이용하여 차선 후보 픽셀들을 탐색하고 2차 다항식으로 차선 곡선을 피팅하는 함수
def sliding_window(img, nwindows=9, margin=150, minpix = 1, draw_windows=True):
    global left_a, left_b, left_c,right_a, right_b, right_c 
    left_fit_= np.empty(3)
    right_fit_ = np.empty(3)
    out_img = np.dstack((img, img, img))*255

    histogram = get_hist(img)
    # 좌우 절반에서 각각 히스토그램 최대값(차선의 시작점) 찾기
    midpoint = int(histogram.shape[0]/2)
    leftx_base = np.argmax(histogram[:midpoint])
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint
    
    # 윈도우 높이 설정
    window_height = int(img.shape[0]/nwindows)
    # 이미지 내 모든 0이 아닌 픽셀 좌표 찾기
    nonzero = img.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])
    
    leftx_current = leftx_base
    rightx_current = rightx_base
    
    left_lane_inds = []
    right_lane_inds = []

    # 윈도우 단위로 슬라이딩 하며 차선 픽셀 찾기
    for window in range(nwindows):
        win_y_low = img.shape[0] - (window+1)*window_height
        win_y_high = img.shape[0] - window*window_height
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # 윈도우 시각화용 사각형 그리기
        if draw_windows == True:
            cv2.rectangle(out_img,(win_xleft_low,win_y_low),(win_xleft_high,win_y_high),
            (100,255,255), 3) 
            cv2.rectangle(out_img,(win_xright_low,win_y_low),(win_xright_high,win_y_high),
            (100,255,255), 3) 
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xleft_low) &  (nonzerox < win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & 
        (nonzerox >= win_xright_low) &  (nonzerox < win_xright_high)).nonzero()[0]
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        if len(good_right_inds) > minpix:        
            rightx_current = int(np.mean(nonzerox[good_right_inds]))

    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 좌우 차선 픽셀 좌표 추출
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds] 
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds] 

    # 2차 다항식으로 좌우 차선 곡선 근사
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)
    
    left_a.append(left_fit[0])
    left_b.append(left_fit[1])
    left_c.append(left_fit[2])
    
    right_a.append(right_fit[0])
    right_b.append(right_fit[1])
    right_c.append(right_fit[2])
    
    # 최근 10개 다항식 계수 평균으로 부드럽게 처리
    left_fit_[0] = np.mean(left_a[-10:])
    left_fit_[1] = np.mean(left_b[-10:])
    left_fit_[2] = np.mean(left_c[-10:])
    
    right_fit_[0] = np.mean(right_a[-10:])
    right_fit_[1] = np.mean(right_b[-10:])
    right_fit_[2] = np.mean(right_c[-10:])
    
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0] )
    left_fitx = left_fit_[0]*ploty**2 + left_fit_[1]*ploty + left_fit_[2]
    right_fitx = right_fit_[0]*ploty**2 + right_fit_[1]*ploty + right_fit_[2]

    out_img[nonzeroy[left_lane_inds], nonzerox[left_lane_inds]] = [255, 0, 100]
    out_img[nonzeroy[right_lane_inds], nonzerox[right_lane_inds]] = [0, 100, 255]
    
    return out_img, (left_fitx, right_fitx), (left_fit_, right_fit_), ploty

# 차선 곡선의 반경과 차량의 차선 중앙에서의 위치를 계산하는 함수
def get_curve(img, leftx, rightx):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    y_eval = np.max(ploty)
    ym_per_pix = 30.5/720 # y 방향 픽셀 당 미터 단위 거리
    xm_per_pix = 3.7/720 # x 방향 픽셀 당 미터 단위 거리

    left_fit_cr = np.polyfit(ploty*ym_per_pix, leftx*xm_per_pix, 2)
    right_fit_cr = np.polyfit(ploty*ym_per_pix, rightx*xm_per_pix, 2)
    left_curverad = ((1 + (2*left_fit_cr[0]*y_eval*ym_per_pix + left_fit_cr[1])**2)**1.5) / np.absolute(2*left_fit_cr[0])
    right_curverad = ((1 + (2*right_fit_cr[0]*y_eval*ym_per_pix + right_fit_cr[1])**2)**1.5) / np.absolute(2*right_fit_cr[0])

    car_pos = img.shape[1]/2
    l_fit_x_int = left_fit_cr[0]*img.shape[0]**2 + left_fit_cr[1]*img.shape[0] + left_fit_cr[2]
    r_fit_x_int = right_fit_cr[0]*img.shape[0]**2 + right_fit_cr[1]*img.shape[0] + right_fit_cr[2]
    lane_center_position = (r_fit_x_int + l_fit_x_int) /2
    center = (car_pos - lane_center_position) * xm_per_pix / 10
    return (left_curverad, right_curverad, center)

# 검출된 차선 다항식 결과를 원본 이미지 위에 색칠해서 시각화하는 함수
def draw_lanes(img, left_fit, right_fit):
    ploty = np.linspace(0, img.shape[0]-1, img.shape[0])
    color_img = np.zeros_like(img)
    
    left = np.array([np.transpose(np.vstack([left_fit, ploty]))])
    right = np.array([np.flipud(np.transpose(np.vstack([right_fit, ploty])))])
    points = np.hstack((left, right))
    
    cv2.fillPoly(color_img, np.int32([points]), (0, 200, 255))
    inv_perspective = inv_perspective_warp(color_img)
    inv_perspective = cv2.addWeighted(img, 1, inv_perspective, 0.7, 0)
    return inv_perspective

def visualize_result(original_img, binary_img, warped_img, out_img, curves, lanes, ploty, curverad):
    """
    전체 파이프라인 단계별 결과를 시각화하는 함수
    """
    left_curverad, right_curverad, center_offset = curverad
    
    fig, axs = plt.subplots(2,3, figsize=(18,10))
    
    axs[0,0].imshow(original_img)
    axs[0,0].set_title('Original Image')
    axs[0,0].axis('off')

    axs[0,1].imshow(binary_img, cmap='gray')
    axs[0,1].set_title('Binary Thresholded Image')
    axs[0,1].axis('off')

    axs[0,2].imshow(warped_img, cmap='gray')
    axs[0,2].set_title('Bird-Eye View (Warped)')
    axs[0,2].axis('off')

    axs[1,0].imshow(out_img)
    axs[1,0].set_title('Sliding Window Search')
    axs[1,0].axis('off')

    img_with_lanes = draw_lanes(original_img, curves[0], curves[1])
    axs[1,1].imshow(img_with_lanes)
    axs[1,1].set_title('Lane Overlay on Original Image')
    axs[1,1].axis('off')

    info_text = f'왼쪽 차선 곡률 반경 : {left_curverad:.2f} m\n오른쪽 차선 곡률 반경 : {right_curverad:.2f} m\nCenter Offset: {center_offset:.2f} m'
    axs[1,2].text(0.1, 0.5, info_text, fontsize=14)
    axs[1,2].axis('off')
    axs[1,2].set_title('Curvature and Position Info')

    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    undistort_img()  # 보정 파라미터 저장 (한 번만 수행해도 됨)
    
    img = cv2.imread('test_images/test6.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    binary_img = pipeline(img)
    warped_img = perspective_warp(binary_img, dst_size=(1280,720))

    out_img, curves, lanes, ploty = sliding_window(warped_img)
    curverad = get_curve(img, curves[0], curves[1])

    # 좌표값 프린트
    print("=== 좌측 차선 x 좌표 샘플 ===")
    print(curves[0][1])  # 왼쪽 차선 x 좌표 일부 출력
    print("=== 우측 차선 x 좌표 샘플 ===")
    print(curves[1][1])  # 오른쪽 차선 x 좌표 일부 출력
    print("=== y 좌표 샘플 ===")
    print(ploty[1])      # y 좌표 일부 출력

    print("\n=== 좌측 차선 2차 다항식 계수 ===")
    print(lanes[0])
    print("=== 우측 차선 2차 다항식 계수 ===")
    print(lanes[1])

    visualize_result(img, binary_img, warped_img, out_img, curves, lanes, ploty, curverad)

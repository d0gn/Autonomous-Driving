#디헤이징 테스트 코드 삭제요망
import torch
import torch.nn as nn
import torchvision
import glob
import cv2
import numpy as np
from PIL import Image
import os
import sys
from pathlib import Path
script_dir = Path(Path(__file__).parent).parent
model_dir = script_dir / 'models' 
if not model_dir.exists():
    print(f"디렉토리 존재 x: {model_dir}")
else:
    sys.path.append(str(model_dir))
    print(f"'{model_dir}' 경로추가")
import net
#이미지 디헤이징
def dehaze_image(image):
    data_hazy = (np.asarray(image) / 255.0)
    data_hazy = torch.from_numpy(data_hazy).float()
    data_hazy = data_hazy.permute(2, 0, 1).unsqueeze(0).cuda()

    dehaze_net = net.dehaze_net().cuda()
    dehaze_net.load_state_dict(torch.load('./checkpoints/dehazer.pth'))

    clean_image = dehaze_net(data_hazy)
    return clean_image.squeeze(0).permute(1, 2, 0).cpu().detach().numpy()
#비디오처리
def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frame_count = 0
    saved_frames = 0

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        
        if frame_count % 10 == 0:
            clean_image = dehaze_image(frame)
            clean_image = (clean_image * 255).astype(np.uint8)
            output_path = f"results/video/cleanedarc_frame_{saved_frames}.png"
            cv2.imwrite(output_path, clean_image)
            saved_frames += 1
            print(f"Processed frame {frame_count}, saved as {output_path}")

        frame_count += 1

    cap.release()

if __name__ == '__main__':
    video_list = glob.glob("test_videos/262215_medium.mp4")  

    for video in video_list:
        process_video(video)
        print(video, "완료")

#삭제요망
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
from pathlib import Path
import argparse
import time
import dataloader
script_dir = Path(Path(__file__).parent).parent
model_dir = script_dir / 'models' 
if not model_dir.exists():
    print(f"디렉토리 존재 x: {model_dir}")
else:
    sys.path.append(str(model_dir))
    print(f"'{model_dir}' 경로추가")
import net
import numpy as np
from torchvision import transforms
from PIL import Image
import glob

#이미지를 디헤이징
def dehaze_image(image_path):

	data_hazy = Image.open(image_path)
	data_hazy = (np.asarray(data_hazy)/255.0)

	data_hazy = torch.from_numpy(data_hazy).float()
	data_hazy = data_hazy.permute(2,0,1)
	data_hazy = data_hazy.cuda().unsqueeze(0)

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.load_state_dict(torch.load('./checkpoints/dehazer.pth'))

	clean_image = dehaze_net(data_hazy)
	torchvision.utils.save_image(torch.cat((data_hazy, clean_image),0), "./results/" + image_path.split("/")[-1])
	

if __name__ == '__main__':

	test_list = glob.glob("test_images/*")

	for image in test_list:

		dehaze_image(image)
		print(image, "완료")

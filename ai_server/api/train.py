#업그레이드 전 디헤이징 훈련에 사용된코드
#경우에 따라 필요할수있음
import torch
import torch.nn as nn
import torchvision
import torch.backends.cudnn as cudnn
import torch.optim
import os
import sys
import argparse
import time
import dataloader
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
import numpy as np
from torchvision import transforms


def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def train(config):

	dehaze_net = net.dehaze_net().cuda()
	dehaze_net.apply(weights_init)

	train_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path)		
	val_dataset = dataloader.dehazing_loader(config.orig_images_path,
											 config.hazy_images_path, mode="val")		
	train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=config.train_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config.val_batch_size, shuffle=True, num_workers=config.num_workers, pin_memory=True)

	criterion = nn.MSELoss().cuda()
	optimizer = torch.optim.Adam(dehaze_net.parameters(), lr=config.lr, weight_decay=config.weight_decay)
	
	dehaze_net.train()

	for epoch in range(config.num_epochs):
		for iteration, (img_orig, img_haze) in enumerate(train_loader):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()

			clean_image = dehaze_net(img_haze)

			loss = criterion(clean_image, img_orig)

			optimizer.zero_grad()
			loss.backward()
			torch.nn.utils.clip_grad_norm(dehaze_net.parameters(),config.grad_clip_norm)
			optimizer.step()

			if ((iteration+1) % config.display_iter) == 0:
				print("로스 ", iteration+1, ":", loss.item())
			if ((iteration+1) % config.snapshot_iter) == 0:
				
				torch.save(dehaze_net.state_dict(), config.snapshots_folder + "epoch" + str(epoch) + '.pth') 		

		# 검증
		for iter_val, (img_orig, img_haze) in enumerate(val_loader):

			img_orig = img_orig.cuda()
			img_haze = img_haze.cuda()

			clean_image = dehaze_net(img_haze)

			torchvision.utils.save_image(torch.cat((img_haze, clean_image, img_orig),0), config.sample_output_folder+str(iter_val+1)+".jpg")

		torch.save(dehaze_net.state_dict(), config.snapshots_folder + "dehazer.pth") 

		







if __name__ == "__main__":

	parser = argparse.ArgumentParser()

	# Input Parameters
	parser.add_argument('--orig_images_path', type=str, default="data/images/")
	parser.add_argument('--hazy_images_path', type=str, default="data/data/")
	parser.add_argument('--lr', type=float, default=0.0001)
	parser.add_argument('--weight_decay', type=float, default=0.0001)
	parser.add_argument('--grad_clip_norm', type=float, default=0.1)
	parser.add_argument('--num_epochs', type=int, default=10)
	parser.add_argument('--train_batch_size', type=int, default=8)
	parser.add_argument('--val_batch_size', type=int, default=8)
	parser.add_argument('--num_workers', type=int, default=4)
	parser.add_argument('--display_iter', type=int, default=10)
	parser.add_argument('--snapshot_iter', type=int, default=200)
	parser.add_argument('--snapshots_folder', type=str, default="snapshots/")
	parser.add_argument('--sample_output_folder', type=str, default="samples/")

	config = parser.parse_args()

	if not os.path.exists(config.snapshots_folder):
		os.mkdir(config.snapshots_folder)
	if not os.path.exists(config.sample_output_folder):
		os.mkdir(config.sample_output_folder)

	train(config)








	

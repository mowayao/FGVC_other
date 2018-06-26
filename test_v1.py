from torchnet.logger import VisdomPlotLogger, VisdomLogger
import torchnet as tnt
from dataset import FGV5Data, FGV5Data_for_test
from config import Config as cfg
import os
from net import DenseNet, InceptionResNetV2, Xception, InceptionV4, SEResNeXt, SENet
import json
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms
from utils import *
import torch
import torch.nn.functional as F
import argparse
import cv2
import PIL
from PIL import Image
import pandas as pd
parser = argparse.ArgumentParser(description='FGVC5')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size(default: 32)')
parser.add_argument('--img_size', type=int, default=224,
                    help='target image size (default: 224)')

parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')


args = parser.parse_args()
subm = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')

torch.manual_seed(2014)
#train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

test_img_lists = os.listdir(os.path.join(cfg.data_root, cfg.test_dir))
test_img_lists = list(filter(lambda x: "jpg" in x, test_img_lists))
test_img_lists = list(map(lambda x: os.path.join(cfg.data_root, cfg.test_dir, x), test_img_lists))
print("the number of testing images:", len(test_img_lists))
test_transforms = transforms.Compose([
	transforms.Resize(size=(args.img_size+20, args.img_size+20)),
	transforms.RandomCrop(size=(args.img_size, args.img_size)),
	transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(cfg.mean, cfg.std)
])

class HFLIP():
	def __call__(self, img):
		"""
		Args:
			img (PIL Image): Image to be flipped.
		Returns:
			PIL Image: Randomly flipped image.
		"""
		return img.transpose(Image.FLIP_LEFT_RIGHT)
test_transforms_flip = transforms.Compose([
	transforms.Resize(size=(args.img_size+20, args.img_size+20)),
	transforms.RandomCrop(size=(args.img_size, args.img_size)),
	HFLIP(),
	#transforms.RandomHorizontalFlip(),
	transforms.ToTensor(),
	transforms.Normalize(cfg.mean, cfg.std)
])
#dataset_flip = FGV5Data_for_test(test_img_lists, test_transforms_flip)
dataset = FGV5Data_for_test(test_img_lists, test_transforms)


#dataloaders = [DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False),
#				DataLoader(dataset_flip, num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False),
#			   ]
print ("net init....")
net = DenseNet(201, cfg.num_classes).cuda()
#net = SEResNeXt(cfg.num_classes).cuda()
#net = InceptionResNetV2(cfg.num_classes)
##net = InceptionV4(cfg.num_classes).cuda()
#net = SENet(cfg.num_classes).cuda()
checkpoints = list(filter(lambda x: net.name in x, os.listdir("../checkpoints")))
checkpoints = list(map(lambda x:os.path.join("../checkpoints", x), checkpoints))


filtered_checkpoints = []
tmp = []

print("filtering checkpoints")
for i, checkpoint in enumerate(checkpoints):
	print(i)
	state = torch.load(checkpoint)
	if 'best_loss' in state:
		loss = state['best_loss']
	else:
		loss = np.inf
	if 'best_acc' in state:
		acc = state['best_acc']
	else:
		acc = -np.inf
	if loss == None:
		loss = np.inf
	if acc == None:
		acc = -np.inf
	flag = True
	for tl, ta, idx in tmp:
		 if tl == loss and ta == acc:
			 flag = False
			 break
	if flag == True:
		tmp.append((loss, acc, i))
tmp = sorted(tmp, key=lambda x:x[0])
for t in tmp:
	###set acc threshold
	if t[0] > 0.47:
		filtered_checkpoints.append(checkpoints[t[-1]])
	if len(filtered_checkpoints) >= 8:
		break
final_dict = dict()


def eval():

	#print()
	#net.load_state_dict(torch.load("../checkpoints/best_model_DenseNet_201_2018-04-18 21:55:18.417103_512_loss.pth")['state_dict'])
	net.eval()
	print ("net init done")

	pred_dict = dict()

	#num_data = len(dataloaders)
	repeat_times = 12

	#test_dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)
	test_dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)

	for t in range(repeat_times):
		print("current:", t)
		#test_dataloader = dataloaders[k]
		for (data, ids) in test_dataloader:
			data = Variable(data).cuda()
			preds = net(data)
			preds = F.softmax(preds, dim=1).cpu().data.numpy()
			for _ in range(preds.shape[0]):
				ID = ids[_]
				pred = preds[_]# * ratios[t]
				if ID not in pred_dict:
					pred_dict[ID] = pred / repeat_times
				else:
					pred_dict[ID] += pred / repeat_times
	for ID in pred_dict:
		if ID not in final_dict:
			final_dict[ID] = pred_dict[ID] / len(checkpoints)
		else:
			final_dict[ID] += pred_dict[ID] / len(checkpoints)


for checkpoint in filtered_checkpoints:
	print(checkpoint)
	state = torch.load(checkpoint)
	if 'best_loss' in state:
		if state['best_loss'] is None:
			continue
		print(state['best_loss'])

		if state['best_loss'] > 0.50 or state['best_loss'] < 0.47:
			continue
	if 'best_acc' in state:
		print(state['best_acc'])
		if state['best_acc'] is None or state['best_acc'] < 85.5:
			continue
	try:
		net.load_state_dict(state['state_dict'])
	except:
		continue
	net.cuda()
	net.eval()
	#eval()

import pickle
for ID in final_dict:
	pred = final_dict[ID]
	#pickle.dump(pred, file=open("den_pred", 'w'))
	#pred = pred_dict[ID]
	subm['predicted'][ID] = np.argmax(pred) + 1

'''
for (data, ids) in test_dataloader:
	data = Variable(data).cuda()
	preds = net(data)
	preds = F.softmax(preds, dim=1).cpu().data.numpy()
	for _ in range(preds.shape[0]):
		subm['predicted'][ids[_]] = np.argmax(preds[_]) + 1
'''
#print("writing file")
#subm.to_csv("result.csv")
'''
print("writing file")
import datetime
cur_t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
subm.to_csv(net.name+"_"+cur_t+"result.csv")

with open(net.name+"_"+cur_t,'wb') as f:
	pickle.dump(final_dict, f)
print("done")
#print("done")
'''
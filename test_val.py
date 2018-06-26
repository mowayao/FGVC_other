from torchnet.logger import VisdomPlotLogger, VisdomLogger
import torchnet as tnt
from dataset import FGV5Data, FGV5Data_for_test
from config import Config as cfg
import os
from net import DenseNet, InceptionResNetV2, Xception, InceptionV4, SEResNeXt, SENet, FBResNet, DPN, ResNext
import json
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms
from utils import *
import shutil
import torch
import torch.nn.functional as F
import argparse
import cv2
import PIL
from PIL import Image
import torch.nn as nn
import pandas as pd
parser = argparse.ArgumentParser(description='FGVC5')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size(default: 32)')
parser.add_argument('--img_size', type=int, default=224,
                    help='target image size (default: 224)')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--test_net', type=str, default="fbresnet",
                    help='net for testing')

args = parser.parse_args()
subm = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')

torch.manual_seed(2014)
#train_data = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True)

val_img_lists = os.listdir(os.path.join(cfg.data_root, cfg.val_dir))
val_img_lists = list(map(lambda x: os.path.join(cfg.data_root, cfg.val_dir, x), val_img_lists))
val_transforms = transforms.Compose([
	transforms.Resize(size=(args.img_size, args.img_size)),
	transforms.ToTensor(),
	transforms.Normalize(cfg.mean, cfg.std)
	#transforms.CenterCrop(size=(args.img_size, args.img_size))
	#transforms.FiveCrop(size=(args.img_size, args.img_size)),
	#transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(cfg.mean, cfg.std)(transforms.ToTensor()(crop)) for crop in crops]))
])
val_dataset = FGV5Data(list(val_img_lists), val_transforms)
val_data = DataLoader(val_dataset, batch_size=32, shuffle=False, drop_last=False, num_workers=args.num_workers)


print ("net init....")

net_dict = {
		'inceptionresnet': InceptionResNetV2,
		'seresnext': SEResNeXt,
		'densenet': DenseNet,
		'dpn': DPN,
		'senet': SENet,
		'resnext': ResNext,
		'xception': Xception,
		'inceptionv4': InceptionV4,
		'fbresnet': FBResNet
}

net = net_dict[args.test_net](cfg.num_classes)
net.cuda().eval()
checkpoints = list(filter(lambda x: net.name in x, os.listdir("../good_checkpoints")))
checkpoints = list(map(lambda x:os.path.join("../good_checkpoints", x), checkpoints))


filtered_checkpoints = []
tmp = []

print("filtering checkpoints")
for i, checkpoint in enumerate(checkpoints):
	print(checkpoint)
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
	#if t[0] > 0.47:
	filtered_checkpoints.append(checkpoints[t[-1]])


val_meter_loss = tnt.meter.AverageValueMeter()
val_meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
test_criterion = nn.CrossEntropyLoss().cuda()

def val():

	#print()
	#net.load_state_dict(torch.load("../checkpoints/best_model_DenseNet_201_2018-04-18 21:55:18.417103_512_loss.pth")['state_dict'])

	print ("net init done")

	#test_dataloader = DataLoader(dataset, num_workers=8, batch_size=args.batch_size, shuffle=False, drop_last=False)
	test_dataloader = DataLoader(val_dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False)


	for (data, label) in test_dataloader:
		data = Variable(data).cuda()
		label = Variable(label).cuda()
		pred = net(data)
		loss = test_criterion(pred, label)
		val_meter_loss.add(loss.cpu().data.numpy()[0])
		val_meter_accuracy.add(pred.cpu().data, label.cpu().data)
		#preds = F.softmax(preds, dim=1).cpu().data.numpy()
	loss = val_meter_loss.value()[0]
	acc = val_meter_accuracy.value()[0]
	return loss, acc
print(len(filtered_checkpoints))
for i, checkpoint in enumerate(filtered_checkpoints):
	print(checkpoint)
	state = torch.load(checkpoint)
	try:
		net.load_state_dict(state['state_dict'])
	except:
		print("fail!!!!!!!!")
		print(checkpoint)
		continue
	print("load successful")
	net.eval()
	loss, acc = val()

	print(loss, acc)
	state['best_loss'] = loss
	state['best_acc'] = acc
	if loss > 0.51:
		os.remove(checkpoint)
	#torch.save(state, checkpoint)
	#eval()
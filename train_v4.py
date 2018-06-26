from torchnet.logger import VisdomPlotLogger, VisdomLogger
import torchnet as tnt
from dataset import FGV5Data, FGV5Data_new
from config import Config as cfg
import os
from net import DenseNet, ResNet, DPN, InceptionResNetV2, ResNext, FBResNet, \
	SENet, InceptionV4, Xception, SEResNeXt, SENet_154, DenseNet161
from loss import FocalLoss, RingLoss
from torch.nn import CrossEntropyLoss, DataParallel
from torch.utils.data.sampler import  WeightedRandomSampler
from torch.optim import Adam, SGD, Rprop, RMSprop
from torch.optim.lr_scheduler import StepLR, ReduceLROnPlateau, MultiStepLR, LambdaLR
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms
import torch.nn.functional as F
from torch.backends import cudnn
import torch
import argparse
import datetime
from sklearn.model_selection import train_test_split
import numpy as np
import random
from utils import *
from sklearn.utils import class_weight

parser = argparse.ArgumentParser(description='FGVC5')
parser.add_argument('--batch_size', type=int, default=32,
                    help='input batch size(default: 32)')
parser.add_argument('--test_batch_size', type=int, default=32,
                    help='input batch size for test(default: 32)')
parser.add_argument('--epochs', type=int, default=15,
                    help='epochs for train (default: 15)')
parser.add_argument('--img_size', type=int, default=224,
                    help='target image size (default: 224)')
parser.add_argument('--decay_step', type=int, default=10,
                    help='decay interval (default: 10 epochs)')
parser.add_argument('--num_iters', type=int, default=100000,
                    help='number of iterations (default: 100000)')
parser.add_argument('--lr', type=float, default=1e-5,
                    help='learning rate')
parser.add_argument('--train_net', type=str, default="densenet",
                    help='net for training')
parser.add_argument('--resume', type=bool, default=False,
                    help='resume')
parser.add_argument('--min_lr', type=float, default=5e-7,
                    help='minimum learning rate')
parser.add_argument('--gpu_id', type=int, default=0,
                    help='gpu id')
parser.add_argument('--num_workers', type=int, default=4,
                    help='number of workers')
parser.add_argument('--wd', type=float, default=5e-4,
                    help='weight decay')
parser.add_argument('--log_interval', type=int, default=20,
                    help='log interval')
parser.add_argument('--seed', type=int, default=1023, metavar='S',
                    help='random seed (default: 1024)')
parser.add_argument('--use_visdom', type=bool, default=True,
                    help='visdom')
parser.add_argument('--visdom_port', type=int, default=7777,
                    help='visdom port')
parser.add_argument('--multi_gpu', type=bool, default=False,
                    help='multi gpus')

args = parser.parse_args()
#cudnn.benchmark = True
net_dict = {
		#'resnet': ResNet(152, cfg.num_classes),
		#'dpn': DPN,
		'densenet': DenseNet,
		'senet': SENet,
		'dense161': DenseNet161,
		'inceptionresnet': InceptionResNetV2,
		'resnext': ResNext,
		'resnet': ResNet,
		'seresnext': SEResNeXt,
		'xception': Xception,
		'inceptionv4': InceptionV4,
		'fbresnet': FBResNet,
		'senet154': SENet_154
}
cur_time = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
torch.manual_seed(args.seed)
train_meter_loss = tnt.meter.AverageValueMeter()
train_meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
train_confusion_meter = tnt.meter.ConfusionMeter(cfg.num_classes, normalized=True)

val_meter_loss = tnt.meter.AverageValueMeter()
val_meter_accuracy = tnt.meter.ClassErrorMeter(accuracy=True)
val_confusion_meter = tnt.meter.ConfusionMeter(cfg.num_classes, normalized=True)

def reset_meters(type = "train"):
	if type == "train":
		train_meter_accuracy.reset()
		train_meter_loss.reset()
		train_confusion_meter.reset()
	else:
		val_meter_accuracy.reset()
		val_meter_loss.reset()
		val_confusion_meter.reset()




#cnt_class_weights = float(len(train_img_lists)) / (cfg.num_classes * label_cnt)


train_transforms_warm = transforms.Compose(
	[
		transforms.Resize(size=(args.img_size+20, args.img_size+20)),

		transforms.RandomCrop(size=(args.img_size, args.img_size)),
		transforms.RandomHorizontalFlip(),
		#transforms.RandomRotation((-10, 10)),
		transforms.ColorJitter(0.3, 0.3, 0.3),
	    transforms.ToTensor(),
		transforms.Normalize(cfg.mean, cfg.std)
	]
)
train_transforms = transforms.Compose(
	[
		transforms.Resize(size=(args.img_size+20, args.img_size+20)),
		#transforms.RandomRotation((-10, 10)),
		transforms.RandomCrop(size=(args.img_size, args.img_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ColorJitter(0.3, 0.3, 0.3),
	    transforms.ToTensor(),
		transforms.Normalize(cfg.mean, cfg.std)
	]
)
train_transforms_no_color_aug = transforms.Compose(
	[
		transforms.Resize(size=(args.img_size+20, args.img_size+20)),
		transforms.RandomHorizontalFlip(),
		#transforms.RandomRotation((-6, 6)),
		transforms.RandomCrop(size=(args.img_size, args.img_size)),
	    transforms.ToTensor(),
		transforms.Normalize(cfg.mean, cfg.std)
	]
)
val_transforms = transforms.Compose([
	transforms.Resize(size=(args.img_size, args.img_size)),
	#transforms.CenterCrop(size=(args.img_size, args.img_size)),
	transforms.ToTensor(),
	transforms.Normalize(cfg.mean, cfg.std)
	#@transforms.FiveCrop(size=(args.img_size, args.img_size)),
	#transforms.Lambda(lambda crops: torch.stack([transforms.Normalize(cfg.mean, cfg.std)(transforms.ToTensor()(crop)) for crop in crops])),
])

semi_dir = "semi_data"
semi_train_dir = os.path.join(cfg.data_root, semi_dir, "train")
semi_val_dir = os.path.join(cfg.data_root, semi_dir, "val")


import pandas as pd

train_csv = "semi_trainset.csv"
train_info = pd.read_csv(os.path.join(cfg.data_root, semi_dir, train_csv))
train_image_ids = train_info.image_id.values
train_label_ids = train_info.label_id.values-1

val_csv = "semi_valset.csv"
val_info = pd.read_csv(os.path.join(cfg.data_root, semi_dir, val_csv))
val_image_ids = val_info.image_id.values
val_label_ids = val_info.label_id.values-1

train_img_lists = map(lambda x: os.path.join(semi_train_dir, x), train_image_ids)

train_dataset = FGV5Data_new(list(train_img_lists), train_label_ids, train_transforms)
#train_dataset_no_aug = FGV5Data_new(list(train_img_lists), train_label_ids, train_transforms_no_color_aug)
train_dataset_warm = FGV5Data_new(list(train_img_lists), train_label_ids, train_transforms_warm)


val_img_lists = map(lambda x: os.path.join(semi_val_dir, x), val_image_ids)
val_dataset = FGV5Data_new(list(val_img_lists), val_label_ids, val_transforms)
val_data = DataLoader(val_dataset, batch_size=10, shuffle=False, drop_last=False, num_workers=args.num_workers)

class_weight = class_weight.compute_class_weight('balanced',
                                             np.unique(train_label_ids),
												 train_label_ids)
train_img_weights = [class_weight[label] for label in train_label_ids]



print("train data size:{}".format(len(train_dataset)))
print("val data size:{}".format(len(val_dataset)))
print ("net init....")


net = net_dict[args.train_net](cfg.num_classes)

if args.multi_gpu == True:
	par_net = DataParallel(net)


if args.use_visdom == True:
	train_cls_loss_logger = VisdomPlotLogger(
		'line', env=net.name, port=args.visdom_port, opts={'title': '{} Train Classification Loss-{}'.format(net.name, cur_time)}
	)
	train_cls_acc_logger = VisdomPlotLogger(
		'line', env=net.name, port=args.visdom_port, opts={'title': '{} Train Classification Accuracy-{}'.format(net.name, cur_time)}
	)
	train_confusion_logger = VisdomLogger('heatmap', env=net.name, port=args.visdom_port, opts={
		'title': '{} Train Confusion matrixmh, time: {}'.format(net.name, cur_time),
		'columnnames': list(
			range(cfg.num_classes)),
		'rownames': list(
			range(cfg.num_classes))})
	val_cls_loss_logger = VisdomPlotLogger(
		'line', env=net.name, port=args.visdom_port, opts={'title': '{} Validation Classification Loss-{}'.format(net.name, cur_time)}
	)
	val_cls_acc_logger = VisdomPlotLogger(
		'line', env=net.name, port=args.visdom_port, opts={'title': '{} Validation Classification Accuracy-{}'.format(net.name, cur_time)}
	)
	val_confusion_logger = VisdomLogger('heatmap', env=net.name, port=args.visdom_port, opts={
		'title': '{} Val Confusion matrixmh, time: {}'.format(net.name, cur_time),
		'columnnames': list(
			range(cfg.num_classes)),
		'rownames': list(
			range(cfg.num_classes))})

print("number of weights:", len(list(net.parameters())))


net.cuda()
#par_net = DataParallel(net)

print ("net init done")
#print(net.parameters())
criterion = CrossEntropyLoss().cuda()
#ring_loss = RingLoss().cuda()
#criterion = FocalLoss(cfg.num_classes).cuda()# CrossEntropyLoss()
test_criterion = CrossEntropyLoss().cuda()

lr_scheduler = [1e-2, 1e-5]
train_sampler = WeightedRandomSampler(train_img_weights, len(train_dataset), replacement=True)
train_data_1 = DataLoader(train_dataset, batch_size=args.batch_size, sampler=train_sampler,
	                        num_workers=args.num_workers, drop_last=False)
train_data_2 = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
	                        num_workers=args.num_workers, drop_last=False)
###TODO:from smaller to bigger?
def adjust_learning_rate(optimizer, decay_rate=.9):
    for param_group in optimizer.param_groups:
        param_group['lr'] = param_group['lr'] * decay_rate
def train():
	global train_data, train_dataset, state
	#date_time = datetime.datetime.now()
	#params_filename1 = "best_model_{}_{}_{}_acc.pth".format(net.name, datetime.datetime.now(), args.img_size)
	params_filename_loss = "best_model_semi_{}_{}_{}_loss.pth".format(net.name, cur_time, args.img_size)
	params_filename_acc = "best_model_semi_{}_{}_{}_acc.pth".format(net.name, cur_time, args.img_size)

	best_loss = np.inf
	best_acc = -np.inf
	num_iterations = int(len(train_dataset)/ args.batch_size)
	reset_meters("train")

	iter_cnt = 0
	print ("start training(warm up)")
	lr = 0.001
	print ("warm up done")
	optimizer = SGD(net.parameters(), lr=lr, momentum=0.9, nesterov=True, weight_decay=5e-4)
	#if args.resume == True:
		#optimizer.load_state_dict(state['optimizer'])
	#for param_group in optimizer.param_groups:
	##	print("init lr:", lr)
	#print("init lr:", lr)
	#optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
	patience = 0
	exp_lr_scheduler = StepLR(optimizer, step_size=5, gamma=0.1)
	for epoch in range(1, args.epochs+1):
		#scheduler.step(epoch)
		net.train()
		#if e#poch % 3 == 0:
			#train_data = train_data_2
		#else:
		if epoch <= 16:
			exp_lr_scheduler.step()
		train_data = train_data_1
		reset_meters("train")
		reset_meters("val")
		if patience == 2:
			patience = 0
			net.load_state_dict(torch.load(os.path.join('../checkpoints', net.name, params_filename_loss))['state_dict'])
		#	lr /= 10
			#optimizer = Adam(net.parameters(), lr=lr, weight_decay=1e-4)
			#optimizer = SGD(net.parameters(), momentum=0.9, lr=lr, weight_decay=1e-5)
		#	optimizer = SGD(net.parameters(), momentum=0.9, nesterov=True, lr=lr, weight_decay=1e-4)
		#	print(f'[+] set lr={lr}')
		#if epoch == args.epoch:
			#train_data = DataLoader(train_dataset_no_aug, batch_size=args.batch_size, shuffle=True,
			#						num_workers=args.num_workers, drop_last=True)

		for iteration, (data, label) in enumerate(train_data):
			optimizer.zero_grad()

			data = Variable(data).cuda()
			label = Variable(label).cuda()
			if args.multi_gpu == True:
				pred = par_net(data)
			else:
				pred = net(data)
			loss = criterion(pred, label) #+ ring_loss(pred)
			loss.backward()
			optimizer.step()

			loss_val = loss.cpu().data.numpy()[0]
			try:
				train_meter_loss.add(loss_val)
				train_confusion_meter.add(pred.cpu().data, label.cpu().data)
			except:
				reset_meters("train")
				train_meter_loss.add(loss_val)
				train_confusion_meter.add(pred.cpu().data, label.cpu().data)
			train_meter_accuracy.add(pred.cpu().data, label.cpu().data)
			print ("Epoch: {}\t Iterations: {}/{}\t loss:{} \t".format(epoch, iteration + 1, num_iterations, loss_val))
			if (iteration + 1) % args.log_interval == 0 and args.use_visdom:
				train_cls_loss_logger.log(iter_cnt, train_meter_loss.value()[0])
				train_cls_acc_logger.log(iter_cnt, train_meter_accuracy.value()[0])
				train_confusion_logger.log(train_confusion_meter.value())
				iter_cnt += 1
				reset_meters("train")

		val_loss, val_acc, confusion_data = validation()

		print ("testing loss:{} \t testing accuracy:{}".format(val_loss, val_acc))
		if args.use_visdom:
			val_cls_loss_logger.log(epoch, val_loss)
			val_cls_acc_logger.log(epoch, val_acc)
			val_confusion_logger.log(confusion_data)

		if best_loss > val_loss:
			if not os.path.exists(os.path.join("../checkpoints", net.name)):
				os.mkdir(os.path.join("../checkpoints", net.name))
			best_loss = val_loss
			patience = 0
			state = {
				'state_dict': net.state_dict(),
				'best_loss': best_loss,
				'best_acc': best_acc,
				'optimizer': optimizer.state_dict()
			}
			torch.save(state, os.path.join('../checkpoints', net.name, params_filename_loss))
		else:
			patience += 1
		if best_acc < val_acc:
			if not os.path.exists(os.path.join("../checkpoints", net.name)):
				os.mkdir(os.path.join("../checkpoints", net.name))
			best_acc = val_acc
			state = {
				'state_dict': net.state_dict(),
				'best_loss': best_loss,
				'best_acc': best_acc,
				'optimizer': optimizer.state_dict(),
				'epoch': epoch
			}
			torch.save(state, os.path.join('../checkpoints', net.name, params_filename_acc))
		if val_loss < 0.25 and val_acc > 92:
			state = {
				'state_dict': net.state_dict(),
				'best_loss': best_loss,
				'best_acc': best_acc,
				'optimizer': optimizer.state_dict(),
				'epoch': epoch
			}
			filename = "model_semi_{}_{}_{}_{}.pth".format(net.name, cur_time, args.img_size, epoch)
			torch.save(state, os.path.join('../checkpoints', net.name, filename))
		#adjust_learning_rate(optimizer, decay_rate=0.8)

def validation():
	global val_data
	reset_meters("val")
	print ("testing......")
	net.eval()
	for iteration, (data, label) in enumerate(val_data):
		label = Variable(label).cuda()
		data = Variable(data).cuda()
		#bs, ncrops, c, h, w = data.size()
		if args.multi_gpu == True:# fuse batch size and ncrops
			pred = par_net(data)
		else:
			pred = net(data)

		#pred = result.view(bs, ncrops, -1).mean(1)  # avg over crops
		loss = test_criterion(pred, label)
		val_meter_loss.add(loss.cpu().data.numpy()[0])
		val_meter_accuracy.add(pred.cpu().data, label.cpu().data)
		val_confusion_meter.add(pred.cpu().data, label.cpu().data)
	loss = val_meter_loss.value()[0]
	acc = val_meter_accuracy.value()[0]
	confusion_data = val_confusion_meter.value()
	net.train()
	return loss, acc, confusion_data
def main():
	#reset_meters("train")
	#reset_meters("test")
	train()

if __name__ == "__main__":
	main()


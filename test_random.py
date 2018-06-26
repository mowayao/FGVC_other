
from dataset import FGV5Data_for_test
from config import Config as cfg
import os
from net import DenseNet, InceptionResNetV2, Xception, InceptionV4, SEResNeXt, SENet, FBResNet, ResNext
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torchvision.transforms import transforms
from utils import *
import torch
from torch.nn import DataParallel
import torch.nn.functional as F
import argparse
from tqdm import tqdm
import pandas as pd
parser = argparse.ArgumentParser(description='FGVC5')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size(default: 1)')

parser.add_argument('--num_workers', type=int, default=16,
                    help='number of workers')
parser.add_argument('--topk', type=int, default=10,
                    help='top k models')
parser.add_argument('--img_size', type=int, default=224,
                    help='img size')
args = parser.parse_args()

torch.manual_seed(2014)
test_img_lists = os.listdir(os.path.join(cfg.data_root, cfg.test_dir))
test_img_lists = list(filter(lambda x: "jpg" in x, test_img_lists))
test_img_lists = list(map(lambda x: os.path.join(cfg.data_root, cfg.test_dir, x), test_img_lists))
print("the number of testing images:", len(test_img_lists))


nets = [
	DenseNet(cfg.num_classes),
	#InceptionResNetV2(cfg.num_classes),
	#FBResNet(cfg.num_classes),
	#SEResNeXt(cfg.num_classes),
	#ResNext(cfg.num_classes),
	#SENet(cfg.num_classes),
	#InceptionV4(cfg.num_classes)
]
eps = 1e-6
def eval_res(net, repeat_times, test_dataloader_1, test_dataloader_2, test_dataloader_3, final_dict_v1, final_dict_v2, final_dict_v3, num_checkpoints):
	net.eval()
	#par_net = DataParallel(net)
	print ("net init done")
	for t in range(repeat_times):
		print("current:", t)
		#test_dataloader = dataloaders[k]
		for (data, ids) in test_dataloader_1:
			data = Variable(data).cuda()
			preds = net(data)
			probs = F.softmax(preds, dim=1).cpu().data.numpy()
			preds = preds.cpu().data.numpy()
			for _ in range(preds.shape[0]):
				ID = ids[_]
				pred = preds[_]# * ratios[t]
				prob = probs[_]
				if ID not in final_dict_v1:
					final_dict_v1[ID] = prob / num_checkpoints / (repeat_times+2)
				else:
					final_dict_v1[ID] += prob / num_checkpoints / (repeat_times+2)
				if ID not in final_dict_v2:
					final_dict_v2[ID] = np.log(prob + eps) / num_checkpoints / (repeat_times+2)
				else:
					final_dict_v2[ID] += np.log(prob + eps) / num_checkpoints / (repeat_times+2)

				if ID not in final_dict_v3:
					final_dict_v3[ID] = pred / num_checkpoints / (repeat_times+2)
				else:
					final_dict_v3[ID] += pred / num_checkpoints / (repeat_times+2)
	for (data, ids) in test_dataloader_2:
		data = Variable(data).cuda()
		preds = net(data)
		probs = F.softmax(preds, dim=1).cpu().data.numpy()
		preds = preds.cpu().data.numpy()
		for _ in range(preds.shape[0]):
			ID = ids[_]
			pred = preds[_]# * ratios[t]
			prob = probs[_]
			if ID not in final_dict_v1:
				final_dict_v1[ID] = prob / num_checkpoints / (repeat_times+2)
			else:
				final_dict_v1[ID] += prob / num_checkpoints / (repeat_times+2)
			if ID not in final_dict_v2:
				final_dict_v2[ID] = np.log(prob + eps) / num_checkpoints / (repeat_times+2)
			else:
				final_dict_v2[ID] += np.log(prob + eps) / num_checkpoints / (repeat_times+2)

			if ID not in final_dict_v3:
				final_dict_v3[ID] = pred / num_checkpoints / (repeat_times+2)
			else:
				final_dict_v3[ID] += pred / num_checkpoints / (repeat_times+2)
	for (data, ids) in test_dataloader_3:
		data = Variable(data).cuda()
		preds = net(data)
		probs = F.softmax(preds, dim=1).cpu().data.numpy()
		preds = preds.cpu().data.numpy()
		for _ in range(preds.shape[0]):
			ID = ids[_]
			pred = preds[_]# * ratios[t]
			prob = probs[_]
			if ID not in final_dict_v1:
				final_dict_v1[ID] = prob / num_checkpoints / (repeat_times+2)
			else:
				final_dict_v1[ID] += prob / num_checkpoints / (repeat_times+2)
			if ID not in final_dict_v2:
				final_dict_v2[ID] = np.log(prob + eps) / num_checkpoints / (repeat_times+2)
			else:
				final_dict_v2[ID] += np.log(prob + eps) / num_checkpoints / (repeat_times+2)

			if ID not in final_dict_v3:
				final_dict_v3[ID] = pred / num_checkpoints / (repeat_times+2)
			else:
				final_dict_v3[ID] += pred / num_checkpoints / (repeat_times+2)
for net in nets:
	net.cuda().eval()
	subm_v1 = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')
	subm_v2 = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')

	checkpoint_dir = os.path.join("../checkpoints", net.name)
	checkpoints = list(filter(lambda x: net.name in x and str(args.img_size) in x and "semi" in x, os.listdir(checkpoint_dir)))
	checkpoints = list(map(lambda x: os.path.join(checkpoint_dir, x), checkpoints))

	filtered_checkpoints = []
	tmp = []

	print("filtering checkpoints")
	for i, checkpoint in enumerate(checkpoints):
		state = torch.load(checkpoint)
		loss = state['best_loss']
		acc = state['best_acc']
		flag = True
		for t in tmp:
			if loss == t[0] and acc == t[1]:
				flag = False
				break
		#if loss > 0.5 or loss < 0.48:
			#continue
		if flag:
			tmp.append((loss, acc, i))
	tmp = sorted(tmp, key=lambda x:x[0])
	for t in tmp[:args.topk]:
		print(t[0], t[1])
		filtered_checkpoints.append(checkpoints[t[-1]])
	final_dict_v1 = dict()
	final_dict_v2 = dict()
	final_dict_v3 = dict()

	print("img size:", args.img_size)
	test_transforms_1 = transforms.Compose([
		transforms.Resize(size=(args.img_size + 20, args.img_size + 20)),
		transforms.RandomCrop(size=(args.img_size, args.img_size)),
		transforms.RandomHorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(cfg.mean, cfg.std)
	])
	test_transforms_2 = transforms.Compose([
		transforms.Resize(size=(args.img_size, args.img_size)),
		transforms.ToTensor(),
		transforms.Normalize(cfg.mean, cfg.std)
	])


	test_transforms_3 = transforms.Compose([
		transforms.Resize(size=(args.img_size, args.img_size)),
		HorizontalFlip(),
		transforms.ToTensor(),
		transforms.Normalize(cfg.mean, cfg.std)
	])
	dataset_1 = FGV5Data_for_test(test_img_lists, test_transforms_1)
	dataset_2 = FGV5Data_for_test(test_img_lists, test_transforms_2)
	dataset_3 = FGV5Data_for_test(test_img_lists, test_transforms_3)
	test_dataloader_1 = DataLoader(dataset_1, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False)
	test_dataloader_2 = DataLoader(dataset_2, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
								   drop_last=False)
	test_dataloader_3 = DataLoader(dataset_3, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False,
								   drop_last=False)
	for checkpoint in filtered_checkpoints:
		print(checkpoint)
		state = torch.load(checkpoint)
		print(state['best_loss'], state['best_acc'])
		net.load_state_dict(state['state_dict'])

		eval_res(net, 10, test_dataloader_1, test_dataloader_2, test_dataloader_3, final_dict_v1, final_dict_v2, final_dict_v3, len(filtered_checkpoints))

	import pickle


	if not os.path.exists("../results/" + net.name):
		os.mkdir("../results/" + net.name)
	print("writing file")
	import datetime

	cur_t = datetime.datetime.now().strftime("%Y-%m-%d_%H:%M:%S")
	#subm_v1.to_csv(os.path.join("../results/" + net.name, cur_t + "_result_v1.csv"))
	#subm_v2.to_csv(os.path.join("../results/" + net.name, cur_t + "_result_v2.csv"))
	net_res_path = os.path.join("../results/", net.name)
	if not os.path.exists(net_res_path):
		os.mkdir(net_res_path)

	with open(os.path.join(net_res_path, cur_t + "_random_v2_top_{}_size_{}".format(args.topk, args.img_size)), 'wb') as f:
		pickle.dump(final_dict_v2, f)
	with open(os.path.join(net_res_path, cur_t + "_random_v1_top_{}_size_{}".format(args.topk, args.img_size)), 'wb') as f:
		pickle.dump(final_dict_v1, f)
	with open(os.path.join(net_res_path, cur_t + "_random_v3_top_{}_size_{}".format(args.topk, args.img_size)), 'wb') as f:
		pickle.dump(final_dict_v3, f)
	print("done")



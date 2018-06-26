
from dataset import FGV5Data_for_test
from config import Config as cfg
import os
import pickle
from net import DenseNet, InceptionResNetV2, Xception, InceptionV4, SEResNeXt, SENet, FBResNet, ResNext
from torch.utils.data import DataLoader
from torch.autograd import Variable
from torch.nn import DataParallel
from torchvision.transforms import transforms
from utils import *
import torch
import torch.nn.functional as F
import argparse
from tqdm import  tqdm
import pandas as pd
parser = argparse.ArgumentParser(description='FGVC5')

parser.add_argument('--num_workers', type=int, default=16,
                    help='number of workers')
parser.add_argument('--batch_size', type=int, default=1,
                    help='input batch size(default: 1)')

args = parser.parse_args()

torch.manual_seed(2014)
test_img_lists = os.listdir(os.path.join(cfg.data_root, cfg.test_dir))
test_img_lists = list(filter(lambda x: "jpg" in x, test_img_lists))
test_img_lists = list(map(lambda x: os.path.join(cfg.data_root, cfg.test_dir, x), test_img_lists))
print("the number of testing images:", len(test_img_lists))
def eval_res(net, test_dataloader, final_dict_v1, final_dict_v2, final_dict_v3, num_checkpoints):
	net.cuda().eval()
	par_net = DataParallel(net)
	print ("net init done")

	for (data, ids) in tqdm(test_dataloader):

		bs, ncrops, c, h, w = data.size()
		result = par_net(Variable(data.view(-1, c, h, w)).cuda())  # fuse batch size and ncrops
		preds = result.view(bs, ncrops, -1).mean(1).cpu().data.numpy()  # avg over crops
		probs = F.softmax(result, dim=1).view(bs, ncrops, -1).mean(1).cpu().data.numpy()
		for _ in range(preds.shape[0]):
			ID = ids[_]
			prob = probs[_]
			pred = preds[_]
			if ID not in final_dict_v1:##naive
				final_dict_v1[ID] = prob / num_checkpoints
			else:
				final_dict_v1[ID] += prob / num_checkpoints
			if ID not in final_dict_v2:##mul => log add
				final_dict_v2[ID] = np.log(prob + eps) / num_checkpoints
			else:
				final_dict_v2[ID] += np.log(prob + eps) / num_checkpoints

			if ID not in final_dict_v3: ##fusion
				final_dict_v3[ID] = pred / num_checkpoints
			else:
				final_dict_v3[ID] += pred / num_checkpoints







nets = [
	#DenseNet(201, cfg.num_classes),
	#InceptionResNetV2(cfg.num_classes),
	#FBResNet(cfg.num_classes),
	SEResNeXt(cfg.num_classes),
	#ResNext(cfg.num_classes),
	#SENet(cfg.num_classes),
	#InceptionV4(cfg.num_classes)
]
eps = 1e-6
for net in nets:

	#subm_v1 = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')
	#subm_v2 = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')
	checkpoint_dir = os.path.join("../checkpoints", net.name)
	checkpoints = list(filter(lambda x: net.name in x, os.listdir(checkpoint_dir)))
	checkpoints = list(map(lambda x:os.path.join(checkpoint_dir, x), checkpoints))

	filtered_checkpoints = []
	tmp = []

	print("filtering checkpoints")
	for i, checkpoint in enumerate(checkpoints):
		state = torch.load(checkpoint)
		loss = state['best_loss']
		acc = state['best_acc']

		tmp.append((loss, acc, i))
	tmp = sorted(tmp, key=lambda x:x[0])
	for t in tmp:
		filtered_checkpoints.append(checkpoints[t[-1]])
		if len(filtered_checkpoints) >= 10:
			break

	final_dict_v1 = dict()
	final_dict_v2 = dict()
	final_dict_v3 = dict()
	if "ception" in net.name:
		cfg.img_size = 299
	else:
		cfg.img_size = 224
	print("img size:", cfg.img_size)
	test_transforms = transforms.Compose([
		transforms.Resize(size=(cfg.img_size + 20, cfg.img_size + 20)),
		transforms.TenCrop(size=(cfg.img_size, cfg.img_size)),
		transforms.Lambda(lambda crops: torch.stack(
			[transforms.Normalize(cfg.mean, cfg.std)(transforms.ToTensor()(crop)) for crop in crops]))
	])

	dataset = FGV5Data_for_test(test_img_lists, test_transforms)
	test_dataloader = DataLoader(dataset, num_workers=args.num_workers, batch_size=args.batch_size, shuffle=False, drop_last=False)
	for checkpoint in filtered_checkpoints:
		print(checkpoint)
		state = torch.load(checkpoint)
		net.load_state_dict(state['state_dict'])
		print("load successful")
		eval_res(net, test_dataloader, final_dict_v1, final_dict_v2, final_dict_v3, len(filtered_checkpoints))



	#for ID in final_dict_v1:
		#pred = final_dict_v1[ID]
		#subm_v1['predicted'][ID] = np.argmax(pred) + 1
	#for ID in final_dict_v2:
		#pred = final_dict_v2[ID]
		#subm_v2['predicted'][ID] = np.argmax(pred) + 1

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

	with open(os.path.join(net_res_path, cur_t + "_standard_v2"), 'wb') as f:
		pickle.dump(final_dict_v2, f)
	with open(os.path.join(net_res_path, cur_t + "_standard_v1"), 'wb') as f:
		pickle.dump(final_dict_v1, f)
	with open(os.path.join(net_res_path, cur_t + "_standard_v3"), 'wb') as f:
		pickle.dump(final_dict_v3, f)
	print("done")

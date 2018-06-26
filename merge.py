import pickle
import pandas as pd
import numpy as np
import os
import argparse
import pandas as pd

parser = argparse.ArgumentParser(description='FGVC5')

parser.add_argument('--type', type=str, default="standard",
                    )
parser.add_argument('--v', type=str, default="1",
                    )
parser.add_argument('--topk', type=str, default="6",
                    )
parser.add_argument('--all', type=bool, default=False,
                    )
args = parser.parse_args()
subm = pd.read_csv("sample_submission_randomlabel.csv", index_col='id')

res_dir = "../results"

nets = [
		"DenseNet_201",
		"InceptionResNetV2",
		#"ResNext",
		"SEResNeXt",
		"FBResNet",
		#"InceptionV4",
		#"SENET"
		]

res_dict = {}

if args.all:

	target_ids = ["standard"+ "_v" + args.v, "random" + "_v" + args.v]
	for target_id in target_ids:
		print(target_id)
		for net in nets:
			net_dir = os.path.join(res_dir, net)
			all_files = os.listdir(net_dir)
			target_file = list(filter(lambda x: target_id in x, all_files))[0]
			target_file = os.path.join(net_dir, target_file)
			with open(target_file, 'rb') as f:
				DICT = pickle.load(f)
				for key in DICT:
					if key not in res_dict:
						res_dict[key] = DICT[key]
					else:
						res_dict[key] += DICT[key]
else:
	target_id = args.type + "_v" + args.v

	for net in nets:
		net_dir = os.path.join(res_dir, net)
		all_files = os.listdir(net_dir)
		if net == "FBResNet":
			topk = "top_"+ "3"
		else:
			topk = "top_" + args.topk
		target_file = list(filter(lambda x: target_id in x and topk in x and ("5-16" in x or "5-18" in x), all_files))[0]
		target_file = os.path.join(net_dir, target_file)

		with open(target_file, 'rb') as f:
			DICT = pickle.load(f)
			for key in DICT:
				if key not in res_dict:
					res_dict[key] = DICT[key]
				else:
					res_dict[key] += DICT[key]

for key in res_dict:
	subm['predicted'][key] = np.argmax(res_dict[key]) + 1
print("writing file")
print("merge_result_v{}.csv".format(args.v))
subm.to_csv("merge_result_v{}.csv".format(args.v))
print("writing done")
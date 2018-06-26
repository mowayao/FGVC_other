from torch.utils.data import Dataset
import cv2
import PIL.Image as Image
from pathlib import Path
class FGV5Data(Dataset):
	def __init__(self, img_lists, transforms):
		super(FGV5Data, self).__init__()
		self.img_lists = img_lists
		self.transforms = transforms
		self.label_list = None
	def get_label(self, path):
		img_idx = int(Path(path).name.split('.')[0].split('_')[-1])
		return img_idx - 1


	def get_label_list(self):
		if self.label_list == None:
			self.label_list = list(map(lambda x: self.get_label(x), self.img_lists))
		return self.label_list
	def __getitem__(self, item):

		img_path = self.img_lists[item]
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		label = self.get_label(img_path)
		if self.transforms != None:
			img = self.transforms(img)
		return img, label

	def __len__(self):
		return len(self.img_lists)
class FGV5Data_new(Dataset):
	def __init__(self, img_lists, label_lists, transforms):
		super(FGV5Data_new, self).__init__()
		self.img_lists = img_lists
		self.transforms = transforms
		self.label_list = label_lists

	def __getitem__(self, item):

		img_path = self.img_lists[item]
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		label = self.label_list[item]
		if self.transforms != None:
			img = self.transforms(img)
		return img, label

	def __len__(self):
		return len(self.img_lists)

class FGV5Data_for_test(Dataset):
	def __init__(self, img_lists, transforms):
		super(FGV5Data_for_test, self).__init__()
		self.img_lists = list(filter(lambda x: "jpg" in x, img_lists))
		self.transforms = transforms
	def __getitem__(self, item):

		img_path = self.img_lists[item]
		img = cv2.imread(img_path)
		img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
		img = Image.fromarray(img)
		img = self.transforms(img)
		ID = self.img_lists[item].split('/')[-1].split('.')[0]
		return img, int(ID)

	def __len__(self):
		return len(self.img_lists)
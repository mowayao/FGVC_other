import torchvision.models as torch_model
import torch.nn as nn
import torch.nn.functional as F
import pretrainedmodels as model_zoo
from config import Config as cfg

class DenseNet(nn.Module):
	def __init__(self, num_classes):
		super(DenseNet, self).__init__()
		#try:
		#	densenet = getattr(torch_model, "densenet{}".format(depth))(pretrained=True)
		#except:
		#	print ("need right depth!")
		self.name = "DenseNet_{}".format(201)
		densenet = torch_model.densenet201(pretrained=True)
		self.features = densenet.features
		#self.fc = nn.Sequential(
		#	nn.Dropout(0.1),
		#	nn.Linear(densenet.classifier.in_features, num_classes)
		#)
		self.fc = nn.Linear(densenet.classifier.in_features, num_classes)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		x = self.fc(x)
		return x
class DenseNet161(nn.Module):
	def __init__(self, num_classes):
		super(DenseNet161, self).__init__()
		self.name = "DenseNet_{}".format(161)
		densenet = torch_model.densenet161()
		self.features = densenet.features
		self.fc = nn.Linear(densenet.classifier.in_features, num_classes)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		x = self.fc(x)
		return x
class ResNet(nn.Module):
	def __init__(self, num_classes):
		super(ResNet, self).__init__()
		self.name = "ResNet_{}".format(152)
		resnet = torch_model.resnet152()
		self.features = nn.Sequential(
			resnet.conv1,
			resnet.bn1,
			resnet.relu,
			resnet.maxpool,
			resnet.layer1,
			resnet.layer2,
			resnet.layer3,
			resnet.layer4
		)
		self.fc = nn.Linear(2048, num_classes)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		x = self.fc(x)
		return x
class DPN(nn.Module):
	def __init__(self, num_classes):
		super(DPN, self).__init__()
		self.name = "DPN"
		pretrained = model_zoo.dpn131()
		cfg.mean = pretrained.mean
		cfg.std = pretrained.std
		self.features = pretrained.features
		self.fc = nn.Linear(2688, num_classes)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)



class NASNet(nn.Module):
	def __init__(self):
		super(NASNet, self).__init__()
		self.name = "NASNET"
		self.model = model_zoo.inceptionv4()
		#self.fc = nn.Linear(self.model.)
	def forward(self, x):
		pass


class InceptionV4(nn.Module):
	def __init__(self, num_classes):
		super(InceptionV4, self).__init__()
		self.name = "InceptionV4"
		self.pretrained_model = model_zoo.inceptionv4()

		cfg.mean = self.pretrained_model.mean
		cfg.std = self.pretrained_model.std
		self.fc = nn.Linear(1536, num_classes)
		nn.init.xavier_uniform(self.fc.weight, gain=2)
	def forward(self, x):
		x = self.pretrained_model.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)

class Xception(nn.Module):
	def __init__(self, num_classes):
		super(Xception, self).__init__()
		self.name = "Xception"
		self.pretrained_model = model_zoo.xception()

		cfg.mean = self.pretrained_model.mean
		cfg.std = self.pretrained_model.std
		self.fc = nn.Linear(2048, num_classes)
	def forward(self, x):
		x = self.pretrained_model.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)

class SENet_154(nn.Module):
	def __init__(self, num_classes):
		super(SENet_154, self).__init__()
		self.name = "SENET_154"
		self.model = model_zoo.senet154()
		self.fc = nn.Linear(2048, num_classes)

		nn.init.xavier_uniform(self.fc.weight, gain=2)
	def forward(self, x):
		x = self.model.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)
class SENet(nn.Module):
	def __init__(self, num_classes):
		super(SENet, self).__init__()
		self.name = "SENET"
		pretrained = model_zoo.se_resnet152()

		self.features = nn.Sequential(
			pretrained.layer0,
			pretrained.layer1,
			pretrained.layer2,
			pretrained.layer3,
			pretrained.layer4
		)
		self.fc = nn.Linear(2048, num_classes)
		#nn.init.xavier_uniform(self.fc.weight, gain=2)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)
class InceptionResNetV2(nn.Module):
	def __init__(self, num_classes):
		super(InceptionResNetV2, self).__init__()
		self.name = "InceptionResNetV2"
		self.model = model_zoo.inceptionresnetv2()
		cfg.mean = self.model.mean
		cfg.std = self.model.std
		self.fc = nn.Linear(1536, num_classes)

	def forward(self, x):
		x = self.model.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)
class ResNext(nn.Module):
	def __init__(self, num_classes):
		super(ResNext, self).__init__()
		self.name = "ResNext"
		pretrained = model_zoo.resnext101_64x4d()
		cfg.mean = pretrained.mean
		cfg.std = pretrained.std
		self.features = pretrained.features
		self.fc = nn.Linear(2048, num_classes)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)


class SEResNeXt(nn.Module):
	def __init__(self, num_classes):
		super(SEResNeXt, self).__init__()
		self.name = "SEResNeXt"
		self.model = model_zoo.se_resnext101_32x4d()
		#pretrained.features()
		self.fc = nn.Linear(2048, num_classes)
	def forward(self, x):
		x = self.model.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)

class FBResNet(nn.Module):
	def __init__(self, num_classes):
		super(FBResNet, self).__init__()
		self.name = "FBResNet"
		pretrained = model_zoo.fbresnet152()
		cfg.mean = pretrained.mean
		cfg.std = pretrained.std
		self.features = nn.Sequential(
			pretrained.conv1,
			pretrained.bn1,
			pretrained.relu,
			pretrained.maxpool,
			pretrained.layer1,
			pretrained.layer2,
			pretrained.layer3,
			pretrained.layer4
		)
		self.fc = nn.Linear(2048, num_classes)
		nn.init.kaiming_uniform(self.fc.weight)
		nn.init.xavier_uniform(self.fc.weight, gain=2)
	def forward(self, x):
		x = self.features(x)
		b, c, h, w = x.size()
		x = F.avg_pool2d(x, kernel_size=(h, w))
		x = x.view(b, -1)
		return self.fc(x)

'''
from torch.autograd import Variable
import torch

net = ResNext(128)
print(net.__annotations__)
a = Variable(torch.ones((1, 3, 256, 256)))
print(net(a).size())
'''
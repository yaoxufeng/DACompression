# coding:utf-8

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms


class DANNet(nn.Module):
	def __init__(self, num_class, arch):
		super(DANNet, self).__init__()
		self.arch = arch
		self.num_class = num_class
		if self.arch == "vgg16":
			self.base_model = models.vgg16(pretrained=True)  # use vgg16 arch and pretrained params
			self.base_model.classifier = nn.Sequential(*list(self.base_model.classifier.children())[:-3])  # drop the last fc layer
			self.hidden_size = 4096
			self.input_size = 224
			self.input_mean = [0.485, 0.456, 0.406]
			self.input_std = [0.229, 0.224, 0.225]
		elif self.arch == "resnet50":
			self.base_model = models.resnet50(pretrained=True)  # user resnet50 arch and pretrained params
			self.base_model = nn.Sequential(*list(self.base_model.children())[:-1])  # drop the last fc layer
			self.hidden_size = 2048
			self.input_size = 224
			self.input_mean = [0.485, 0.456, 0.406]
			self.input_std = [0.229, 0.224, 0.225]
		else:
			raise ValueError("Unknown model architecture {}".format(self.arch))
		
		self.base_model
		self.fc = nn.Linear(self.hidden_size, num_class)
		self.prelu = nn.PReLU()
		
	def forward(self, source, target):
		b = source.size()[0]  # get the length of the source data
		x_total = torch.cat([source, target], dim=0)
		x = self.base_model(x_total)
		x = x.view(x.size()[0], -1)
		feature = self.prelu(x)
		cls = self.fc(feature)
		source_feature = feature[:b, :]
		target_feature = feature[b:, ]
		source_cls = cls[:b]
		target_cls = cls[b:]

		return source_feature, target_feature, source_cls, target_cls
	
	def train_augmentation(self):
		return transforms.Compose([
			transforms.Resize(256),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.input_mean, std=self.input_std)
		])
	
	def train_multi_augmentation(self):
		return transforms.Compose([
			transforms.Resize(256),
			transforms.RandomResizedCrop(224),
			transforms.RandomHorizontalFlip(),
			transforms.RandomRotation([-30, 30]),
			transforms.ColorJitter(),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.input_mean, std=self.input_std)
		])
	
	def test_augmentation(self):
		return transforms.Compose([
			transforms.Resize(256),
			transforms.CenterCrop(224),
			transforms.ToTensor(),
			transforms.Normalize(mean=self.input_mean, std=self.input_std)
		])


if __name__ == "__main__":
	source = torch.randn(16, 3, 224, 224)
	target = torch.randn(16, 3, 224, 224)
	model_danet = DANNet(num_class=31, arch="resnet50")
	source_f, target_f, source_cls, target_cls = model_danet(source, target)
	print("source_feature size is {}, target_feature size is {}, "
	      "source_cls size is {}, target_cls size is {}".format(source_f.size(), target_f.size(), source_cls.size(), target_cls.size()))
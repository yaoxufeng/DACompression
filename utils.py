# coding:utf-8

import torch
import numpy as np
from torchvision import transforms


# convert tensor function
def convert_tensor(*tensors, device='cpu'):
	return [t.to(device) for t in tensors]


# convert multi transform data
def convert_batch_data(batch_data):
	return batch_data.view(batch_data.size(0)*batch_data.size(1), batch_data.size(2), batch_data.size(3), batch_data.size(4))


# convert multi transform label
def convert_batch_label(batch_label):
	return batch_label.view(batch_label.size(0)*batch_label.size(1))


# correct num calculation in one batch
def acc_cal(pred, target, top_n=None):
	_, pred = torch.max(pred.data, 1)
	correct = pred.eq(target.data.view_as(target)).float().cpu().sum()
	
	return correct


def l2_norm(x):
	norm = x.float().norm(p=2, dim=1, keepdim=True)
	x_normalized = x.div(norm.expand_as(x))
	
	return x_normalized


# ========================= mixmatch  ==========================
# sharpen function for mixmatch
def sharpen(x, t=0.5):
	"""
	@https://github.com/gan3sh500/mixmatch-pytorch/blob/master/mixmatch_utils.py
	:param x: avg prob for mutli data augmentation result
	:param T: parameter to control the degree of sharpen, larger T, more sharper
	:return: sharpened x
	"""
	temp = x ** (1/t)
	temp_sum = torch.sum(temp, 1)
	
	return 1. * temp / temp_sum.view(temp.size(0), 1)


# combine and shuffle source and target dataset
def shuffle_cat(source, source_label, target, target_label):
	data = torch.cat([source, target], dim=0)
	label = torch.cat([source_label, target_label.long()], dim=0)
	shuffle_idx = torch.randperm(data.size(0))
	
	return data[shuffle_idx], label[shuffle_idx]
	

# mixupmatch
def mixup_match(source_data, source_label, target_data, target_label, alpha=0.75, w_index=[], use_cuda=True):
	"""
	:param source_data: x with labeled data
	:param source_label: x with label
	:param target_data: w with shuffled and combined labeled and unlabeled data
	:param target_label: w with label and pseu label
	:param alpha: alpha parameter, default 0.75 according to paper for a sharper result
	:param w_index: w_index for selecting w_data to mixup
	:param use_cuda:
	:return: mixed_data, mixed_label, lam, w_index
	"""
	
	# randomly generate lam
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
		lam = max(lam, 1 - lam)
	else:
		lam = 1.
	
	if len(w_index) > 0:
		mixed_x = lam * source_data + (1 - lam) * target_data[w_index][-source_data.size(0):]
		y_a, y_b = source_label, target_label[w_index][-source_data.size(0):]
		return mixed_x, y_a, y_b, lam
		
	else:
		batch_size = target_data.size(0)
		if use_cuda:
			w_index = torch.randperm(batch_size).cuda()
		else:
			w_index = torch.randperm(batch_size)
		
		mixed_x = lam * source_data + (1 - lam) * target_data[w_index][:source_data.size(0)]
		y_a, y_b = source_label, target_label[w_index][:source_data.size(0)]
		
		return mixed_x, y_a, y_b, lam, w_index
	

# transform index to onehot label
def convert_onehot(x, num_class, use_cuda=True):
	if use_cuda:
		return torch.zeros(x.size(0), num_class).cuda().scatter_(1, x.view(-1, 1), 1)
	else:
		return torch.zeros(x.size(0), num_class).scatter_(1, x.view(-1, 1), 1)


# ========================= mix up  ==========================
def mixup(x, y, alpha=1.0, use_cuda=True):
	'''
	a classical data augmentation method for reducing inductive bias
	@https://github.com/hongyi-zhang/mixup
	:param x: source data
	:param y: source target
	:param alpha: alpha parameters
	:param use_cuda: whether to use cuda
	:return: mixed_x, y_a, y_b lam
	'''
	if alpha > 0.:
		lam = np.random.beta(alpha, alpha)
	else:
		lam = 1.
	batch_size = x.size()[0]
	if use_cuda:
		index = torch.randperm(batch_size).cuda()
	else:
		index = torch.randperm(batch_size)
	
	mixed_x = lam * x + (1 - lam) * x[index, :]
	y_a, y_b = y, y[index]
	return mixed_x, y_a, y_b, lam


def mixup_criterion(y_a, y_b, lam):
	return lambda criterion, pred: lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)


def mixup_acc_cal(pred, target_a, target_b, lam):
	_, pred = torch.max(pred.data, 1)
	correct = lam * pred.eq(target_a.data).cpu().sum().item() + (1 - lam) * pred.eq(target_b.data).cpu().sum().item()
	return correct







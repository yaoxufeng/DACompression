# coding:utf-8

"""
train file
"""

import os
import glob
import argparse
import random
import numpy as np
import math
import logging
import shutil
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torch.optim as optim
from torch.utils.data import DataLoader
from tqdm import trange, tqdm
from tensorboardX import SummaryWriter
from datasets import OfficeDataSet
from opts import parser
from models import DANNet
from loss import mmd_rbf_noaccelerate, consistency_loss, CenterLoss
from utils import *

args = parser.parse_args()
best_prec1 = 0  # init best_precision


def set_seed():
	random.seed(args.seed)
	np.random.seed(args.seed)
	torch.manual_seed(args.seed)
	torch.cuda.manual_seed_all(args.seed)
	

def train(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=0.5, k=1):
	"""
	:param source_train_loader: training set of source dataloader
	:param target_train_loader: training set of target dataloader (random crop)
	:param model: train model
	:param criterion: loss function
	:param optimizer: optimizer function
	:param epoch: speicific epoch (for showing the epoch information only)
	:param gamma: gamma parameter for loss
	:return: training {losses, mmdlosses, softmaxloss, train_acc}
	"""
	
	model.train()  # switch train mode
	losses = 0  # init losses
	softmaxloss = 0  # init softmax loss
	centerlosses = 0  # init center loss
	mmdlosses = 0  # init mmdlosses
	train_acc = 0  # inti train_acc
	iter_source = iter(source_train_loader)
	iter_target = iter(target_train_loader)
	
	# iterate whole source data
	for i in tqdm(range(1, len(source_train_loader))):
		source_data, source_label = iter_source.next()
		source_data = convert_batch_data(source_data)
		source_label = convert_batch_label(source_label)
		if i <= len(target_train_loader):
			target_data, target_label = iter_target.next()
			target_data = convert_batch_data(target_data)
			target_label = convert_batch_label(target_label)
		source_data, source_label, target_data, target_label = convert_tensor(source_data, source_label, target_data, target_label, device='cuda')
		source_feature, target_feature, source_cls, target_cls = model(source_data, target_data)
		
		# normalize target_feature and transform it to the same size with source_feature
		#target_feature = target_feature.view(k, source_feature.size(0), -1)
		#target_feature = torch.mean(target_feature, 0)

		target_feature = target_feature[:source_feature.size(0)]
		optimizer.zero_grad()
		mmd_loss = mmd_rbf_noaccelerate(source_feature, target_feature)
		softmax_loss = criterion(source_cls, source_label)
		correct = acc_cal(source_cls, source_label)

		loss = softmax_loss + gamma * mmd_loss
		losses += loss.item()
		mmdlosses += mmd_loss.item()
		softmaxloss += softmax_loss.item()
		train_acc += correct.item()
		loss.backward()
		optimizer.step()
	
	losses /= 1. * len(source_train_loader)  # avg loss
	mmdlosses /= 1. * len(source_train_loader)  # avg mmdloss
	softmaxloss /= 1. * len(source_train_loader)  # avg softmax loss
	train_acc /= 1. * len(source_train_loader.dataset)  # avg train_acc
	print("training Epoch: {}, loss: {}, softmaxloss: {}, mmdloss: {}, train_acc: {}".format(epoch, losses, softmaxloss, mmdlosses, train_acc))

	return losses, mmdlosses, softmaxloss, train_acc


def train_ada(source_train_loader, target_train_loader, model, criterion, center_loss, optimizer, optimizer_centloss, epoch, gamma=0.5, k=1):
	'''
	:param source_train_loader:
	:param target_train_loader:
	:param model:
	:param criterion:
	:param center_loss:
	:param optimizer:
	:param optimizer_centloss:
	:param epoch:
	:param gamma:
	:return:
	'''
	
	model.train()  # switch train mode
	losses = 0  # init losses
	softmaxloss = 0  # init softmax loss
	centerlosses_source = 0  # init source center loss
	centerlosses_target = 0  # init target center loss
	mmdlosses = 0  # init mmdlosses
	l2losses = 0
	train_acc = 0  # init train_acc
	iter_source = iter(source_train_loader)
	iter_target = iter(target_train_loader)
	
	# iterate whole source data
	for i in tqdm(range(1, len(source_train_loader))):
		source_data, source_label = iter_source.next()
		source_data = convert_batch_data(source_data)
		source_label = convert_batch_label(source_label)
		if i <= len(target_train_loader):
			target_data, target_label = iter_target.next()
			target_data = convert_batch_data(target_data)
			target_label = convert_batch_label(target_label)
		source_data, source_label, target_data, target_label = convert_tensor(source_data, source_label, target_data,
		                                                                      target_label, device='cuda')
		source_feature, target_feature, source_cls, target_cls = model(source_data, target_data)
		# target_feature = target_feature.view(k, source_feature.size(0), -1)
		# target_feature = torch.mean(target_feature, 0)
		# target_pseu = target_cls.view(k, source_cls.size(0), -1)
		# target_pseu = torch.mean(target_pseu, 0)
		# _, target_pseu = torch.max(target_pseu, 1)
		_, target_pseu = torch.max(target_cls, 1)
		optimizer.zero_grad()
		optimizer_centloss.zero_grad()
		mmd_loss = mmd_rbf_noaccelerate(source_feature, target_feature[:source_feature.size(0)])
		softmax_loss = criterion(source_cls, source_label)
		centerloss_source = center_loss(source_label, source_feature)
		centerloss_target = center_loss(target_pseu[:source_label.size(0)], target_feature[:source_feature.size(0)])
		l2loss = consistency_loss(sharpen(F.softmax(target_cls)), k=k)
		correct = acc_cal(source_cls, source_label)
		
		loss = softmax_loss + 0.1 * gamma * (centerloss_source + centerloss_target) + 10 * l2loss
		losses += loss.item()
		mmdlosses += mmd_loss.item()
		centerlosses_source += centerloss_source.item()
		centerlosses_target += centerloss_target.item()
		softmaxloss += softmax_loss.item()
		l2losses += l2loss.item()
		train_acc += correct.item()
		loss.backward()
		optimizer.step()
		optimizer_centloss.step()
	
	losses /= 1. * len(source_train_loader)  # avg loss
	mmdlosses /= 1. * len(source_train_loader)  # avg mmdloss
	softmaxloss /= 1. * len(source_train_loader)  # avg softmax loss
	centerlosses_source /= 1. * len(source_train_loader)
	centerlosses_target /= 1. * len(source_train_loader)
	l2losses /= 1. * len(source_train_loader)
	train_acc /= 1. * len(source_train_loader.dataset)  # avg train_acc
	print("training Epoch: {}, loss: {}, softmaxloss: {}, mmdloss: {}, "
	      "centerloss_source: {}, centerloss_target: {}, l2loss: {}, train_acc: {}".format(epoch,
	                                                                           losses,softmaxloss,
	                                                                           mmdlosses,
	                                                                           centerlosses_source,
                                                                               centerlosses_target,
                                                                                l2losses,
                                                                               train_acc))
	
	return losses, mmdlosses, softmaxloss, train_acc
	

def train_consistency_regu(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=0.5, k=2):
	'''
	:param source_train_loader:
	:param target_train_loader:
	:param model:
	:param criterion:
	:param optimizer:
	:param epoch:
	:param gamma:
	:param k:
	:return:
	'''
	model.train()  # switch train mode
	losses = 0  # init losses
	softmaxloss = 0  # init softmax loss
	l2_losses = 0  # inti target l2 loss
	mmdlosses = 0  # init mmdlosses
	train_acc = 0  # init train accuracy
	iter_source = iter(source_train_loader)
	iter_target = iter(target_train_loader)
	
	# iterate whole source data
	for i in tqdm(range(1, len(source_train_loader))):
		source_data, source_label = iter_source.next()
		source_data = convert_batch_data(source_data)
		source_label = convert_batch_label(source_label)
		if i <= len(target_train_loader):
			target_data, target_label = iter_target.next()
			target_data = convert_batch_data(target_data)
			target_label = convert_batch_label(target_label)
		source_data, source_label, target_data, target_label = convert_tensor(source_data, source_label, target_data,
		                                                                      target_label, device='cuda')
		
		source_feature, target_feature, source_cls, target_cls = model(source_data, target_data)
		target_feature = target_feature.view(k, source_feature.size(0), -1)
		target_feature = torch.mean(target_feature, 0)
		optimizer.zero_grad()
		mmd_loss = mmd_rbf_noaccelerate(source_feature, target_feature)
		l2loss = consistency_loss(target_cls, k=k)
		softmax_loss = criterion(source_cls, source_label)
		correct = acc_cal(source_cls, source_label)
		if epoch < 10:
			loss = softmax_loss + gamma * mmd_loss
		else:
			loss = softmax_loss + gamma * mmd_loss + 5 * l2loss
		losses += loss.item()
		mmdlosses += mmd_loss.item()
		softmaxloss += softmax_loss.item()
		l2_losses += l2loss.item()
		train_acc += correct
		loss.backward()
		optimizer.step()
	
	losses /= 1. * len(source_train_loader)  # avg loss
	mmdlosses /= 1. * len(source_train_loader)  # avg mmdloss
	softmaxloss /= 1. * len(source_train_loader)  # avg softmax loss
	l2_losses /= 1. * len(source_train_loader)  # avg l2loss
	train_acc /= 1. * len(source_train_loader.dataset)  # avg train_acc
	print("training Epoch: {}, loss: {}, softmaxloss: {}, l2loss: {},  mmdloss: {}, train_acc: {}".format(epoch, losses,
	                                                                                                      softmaxloss,
	                                                                                                      l2_losses,
	                                                                                                      mmdlosses,
	                                                                                                      train_acc))
	
	return losses, softmaxloss, l2_losses, mmdlosses, train_acc


def train_mixup(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=0.5):
	"""
	train with mixup
	:param source_train_loader: training set of source dataloader
	:param target_train_loader: training set of target dataloader (random crop)
	:param model: train model
	:param criterion: loss function
	:param optimizer: optimizer function
	:param epoch: speicific epoch (for showing the epoch information only)
	:param gamma: gamma parameter for loss
	:return: training {losses, mmdlosses, softmaxloss, train_acc}
	"""
	
	model.train()  # switch train mode
	losses = 0  # init losses
	softmaxloss = 0  # init softmax loss
	mmdlosses = 0  # init mmdlosses
	train_acc = 0  # init train accuracy
	iter_source = iter(source_train_loader)
	iter_target = iter(target_train_loader)
	
	# iterate whole source data
	for i in tqdm(range(1, len(source_train_loader))):
		source_data, source_label = iter_source.next()
		source_data = convert_batch_data(source_data)
		source_label = convert_batch_label(source_label)
		if i <= len(target_train_loader):
			target_data, target_label = iter_target.next()
			target_data = convert_batch_data(target_data)
			target_label = convert_batch_label(target_label)
		source_data, source_label, target_data, target_label = convert_tensor(source_data, source_label, target_data, target_label, device='cuda')

		source_data, source_label_a, source_label_b, lam = mixup(source_data, source_label)
		source_feature, target_feature, source_cls, target_cls = model(source_data, target_data)
		optimizer.zero_grad()
		mmd_loss = mmd_rbf_noaccelerate(source_feature, target_feature)
		loss_func = mixup_criterion(source_label_a, source_label_b, lam)
		softmax_loss = loss_func(criterion, source_cls)
		correct = mixup_acc_cal(source_cls, source_label_a, source_label_b, lam)
		loss = softmax_loss + gamma * mmd_loss
		losses += loss.item()
		mmdlosses += mmd_loss.item()
		softmaxloss += softmax_loss.item()
		train_acc += correct
		loss.backward()
		optimizer.step()
	
	losses /= 1. * len(source_train_loader)  # avg loss
	mmdlosses /= 1. * len(source_train_loader)  # avg mmdloss
	softmaxloss /= 1. * len(source_train_loader)  # avg softmax loss
	train_acc /= 1. * len(source_train_loader.dataset)  # avg train_acc
	print("training Epoch: {}, loss: {}, softmaxloss: {}, mmdloss: {}, train_acc: {}".format(epoch, losses, softmaxloss, mmdlosses, train_acc))

	return losses, mmdlosses, softmaxloss, train_acc
	

def train_mixmatch(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=0.5, k=2, T=0.5):
	"""
	train with mixmatch
	:param source_train_loader: training set of source dataloader
	:param target_train_loader: training set of target dataloader (random crop)
	:param model: train model
	:param criterion: loss function
	:param optimizer: optimizer function
	:param epoch: specific epoch (for showing the epoch information only)
	:param gamma: gamma parameter for loss
	:param k: num of data augmentation
	:param T: parameters to control the degree of sharp of the avg prob
	:return: training {losses, softmaxloss, mmdlosses, train_acc}
	"""
	model.train()  # switch train mode
	losses = 0  # init losses
	softmaxloss = 0  # init softmax loss
	l2_losses = 0  # inti target loss
	mmdlosses = 0  # init mmdlosses
	train_acc = 0  # init train accuracy
	iter_source = iter(source_train_loader)
	iter_target = iter(target_train_loader)
	
	# iterate whole source data
	for i in tqdm(range(1, len(source_train_loader))):
		source_data, source_label = iter_source.next()
		source_data = convert_batch_data(source_data)
		source_label = convert_batch_label(source_label)
		if i <= len(target_train_loader):
			target_data, target_label = iter_target.next()
			target_data = convert_batch_data(target_data)
			target_label = convert_batch_label(target_label)
		source_data, source_label, target_data, target_label = convert_tensor(source_data, source_label, target_data,
		                                                                      target_label, device='cuda')

		source_feature, target_feature, source_cls, target_cls = model(source_data, target_data)
		target_feature = target_feature.view(k, source_feature.size(0), -1)
		target_feature = torch.mean(target_feature, 0)
		target_cls_pseu = target_cls.view(k, source_cls.size(0), -1)
		target_cls_pseu = torch.mean(target_cls_pseu, 0)
		target_cls_pseu = sharpen(F.softmax(target_cls_pseu), t=T)
		target_cls_pseu = target_cls_pseu.repeat(k, 1)
		_, target_cls_pseu = torch.max(target_cls_pseu, 1)

		# generate w indicates for shuffled and combined labeled and unlabeled data
		data_combine, label_combine = shuffle_cat(source_data, source_label, target_data, target_cls_pseu)
		datax, datax_label_a, datax_label_b, lamx, w_index = mixup_match(source_data, source_label,
		                                                                 data_combine, label_combine)
		datau, datau_label_a, datau_label_b, lamu = mixup_match(target_data, target_cls_pseu, data_combine,
		                                                        label_combine, w_index=w_index)
		
		source_feature_x, target_feature_x, source_x_cls, target_x_cls = model(datax, datax)
		source_feature_u, target_feature_u, source_u_cls, target_u_cls = model(datau, datau)
		
		optimizer.zero_grad()
		mmd_loss = mmd_rbf_noaccelerate(source_feature, target_feature)
		loss_func_x = mixup_criterion(datax_label_a, datax_label_b, lamx)
		loss_func_u = mixup_criterion(convert_onehot(datau_label_a, 31), convert_onehot(datau_label_b, 31), lamu)
		softmax_loss = loss_func_x(criterion, source_x_cls)
		l2loss = loss_func_u(consistency_loss, source_u_cls)
		correct = mixup_acc_cal(source_x_cls, datax_label_a, datax_label_b, lamx)
		loss = softmax_loss + gamma * mmd_loss
		losses += loss.item()
		mmdlosses += mmd_loss.item()
		softmaxloss += softmax_loss.item()
		l2_losses += l2loss.item()
		train_acc += correct
		loss.backward()
		optimizer.step()
	
	losses /= 1. * len(source_train_loader)  # avg loss
	mmdlosses /= 1. * len(source_train_loader)  # avg mmdloss
	softmaxloss /= 1. * len(source_train_loader)  # avg softmax loss
	l2_losses /= 1. * len(source_train_loader)  # avg l2loss
	train_acc /= 1. * len(source_train_loader.dataset)  # avg train_acc
	print("training Epoch: {}, loss: {}, softmaxloss: {}, l2loss: {},  mmdloss: {}, train_acc: {}".format(epoch, losses, softmaxloss, l2_losses, mmdlosses, train_acc))
	
	return losses, softmaxloss, l2_losses, mmdlosses, train_acc


def validate(val_loader, model, criterion):
	"""
	:param val_loader: validation set of target dataloader (center crop)
	:param model: train model
	:param criterion: loss function
	:param gamma: gamma parameter for loss
	:return:
	"""
	
	model.eval()  # switch eval mode
	losses = 0  # init losses
	accuracy = 0  # init acc
	for i, (inputs, target) in tqdm(enumerate(val_loader)):
		inputs, target = convert_tensor(inputs, target, device='cuda')
		inputs = convert_batch_data(inputs)
		target = convert_batch_label(target)
		s_feature, t_feature, s_cls, t_cls = model(inputs, inputs)
		loss = criterion(t_cls, target)
		correct = acc_cal(t_cls, target)
		losses += loss.item()
		accuracy += correct.item()
		
	losses /= 1. * len(val_loader)
	accuracy /= 1. * len(val_loader.dataset)
	
	return accuracy, losses


def save_checkpoint(state, is_best, filename="checkpoint.pth.tar"):
	filename = args.checkpoint_dir + filename
	torch.save(state, filename)
	if is_best:
		best_name = args.checkpoint_dir + 'best.pth.tar'
		shutil.copyfile(filename, best_name)
		

def main():
	if args.dataset == "Office31":
		num_class = 31
	elif args.dataset == "ImageCLEF":
		num_class = 12
	else:
		raise ValueError("Unknown dataset {}".format(args.dataset))
	
	if args.arch == "vgg16":
		model = DANNet(arch="vgg16", num_class=num_class)
		center_loss = CenterLoss(num_class, 128).to('cuda')
	elif args.arch == "resnet50":
		model = DANNet(arch="resnet50", num_class=num_class)
		center_loss = CenterLoss(num_class, 128).to('cuda')
	else:
		raise ValueError("Unknown model architecture {}".format(args.arch))
	
	if len(args.gpu_ids) > 1:
		model = nn.DataParallel(model, device_ids=args.gpu_ids).cuda()
	else:
		model.cuda()
	
	# load source dataset default multi gpu
	if len(args.gpu_ids) > 1:
		transformer_train = model.module.train_augmentation()
		transformer_train_multi = model.module.train_multi_augmentation()
		transformer_test = model.module.test_augmentation()
	else:
		transformer_train = model.train_augmentation()
		transformer_train_multi = model.train_multi_augmentation()
		transformer_test = model.test_augmentation()
	source_train_loader = DataLoader(
		OfficeDataSet(data_path=args.train_path, transformer=transformer_train, k=1),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=True,
		drop_last=True,
		pin_memory=False
		)
	
	# load target dataset for training, its label is not used for training default multi gpu
	target_train_loader = DataLoader(
		OfficeDataSet(data_path=args.test_path, transformer=transformer_train, k=args.k),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=True,
		drop_last=True,
		pin_memory=False
		)
	
	# load target dataset for testing, its label is used for testing until the whole training process ends.
	target_test_loader = DataLoader(
		OfficeDataSet(data_path=args.test_path, transformer=transformer_test, k=1),
		batch_size=args.batch_size,
		num_workers=args.workers,
		shuffle=False,
		pin_memory=False
		)
	
	# set seed
	set_seed()
	
	# set writer to load some training information
	writer = SummaryWriter(args.tensorboard_file)
	logging.basicConfig(level=logging.INFO, filename=args.log_file, filemode='w')
	
	# set loss function
	criterion = nn.CrossEntropyLoss()
	
	# set optimizer
	if args.optimizer == "SGD":
		optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
		optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)
	elif args.optimizer == "Adam":
		optimizer = optim.Adam(model.parameters(), lr=args.lr)
		optimizer_centloss = torch.optim.SGD(center_loss.parameters(), lr=0.5)
	else:
		raise ValueError("the author is too lazy to add the optimizer {}".format(args.opt))
	
	# set lr scheduler
	if args.lr_scheduler == 'cosine_decay':
		lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, len(source_train_loader), eta_min=0)
	elif args.lr_scheduler == "step_decay":
		# warning! the milestones should be modified by your own demand
		lr_scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[5, 10, 15], gamma=0.1)
	elif args.lr_scheduler == "custom":
		raise ValueError("custom learning rate scheduler is under the todo list")
	else:
		raise ValueError("warning, wrong learning rate scheduler {}".format(args.lr_schedule))
	
	# training process
	for epoch in trange(args.start_epoch, args.num_epoch):
		writer.add_scalar("train_epoch", epoch)
		logging.info("train_epoch: {}".format(epoch))
		gamma = 2 / (1 + math.exp(-10 * (epoch) / args.num_epoch)) - 1  # set gamma for each epoch
		if args.method == "train":
			losses, mmdlosses, softmaxloss, train_acc = train(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=gamma, k=args.k)
		elif args.method == "train_mixup":
			losses, mmdlosses, softmaxloss, train_acc = train_mixup(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=gamma)
		elif args.method == "train_mixmatch":
			losses, softmaxloss, l2loss,  mmdlosses, train_acc = train_mixmatch(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=gamma, k=args.k)
			writer.add_scalar("l2_loss", l2loss)
			logging.info("l2_loss: {}".format(l2loss))
		elif args.method == "consistency_regu":
			losses, softmaxloss, l2loss,  mmdlosses, train_acc = train_consistency_regu(source_train_loader, target_train_loader, model, criterion, optimizer, epoch, gamma=gamma, k=args.k)
		elif args.method == "train_ada":
			losses, mmdlosses, softmaxloss, train_acc = train_ada(source_train_loader, target_train_loader, model, criterion, center_loss, optimizer, optimizer_centloss, epoch, gamma=gamma, k=2)
		else:
			raise ValueError("other tricks is under the todo list")
	
		# writer tensorboard value
		writer.add_scalar("train_loss", losses)
		writer.add_scalar("train_MMDloss", mmdlosses)
		writer.add_scalar("train_CEloss", softmaxloss)
		writer.add_scalar("train_acc", train_acc)
		
		# write result to log file
		logging.info("train_loss: {}".format(losses))
		logging.info("train_MMDloss: {}".format(mmdlosses))
		logging.info("train_CEloss: {}".format(softmaxloss))
		logging.info("train_acc: {}".format(train_acc))

		is_best = True
		save_checkpoint({
			'epoch': epoch + 1,
			'arch': args.arch,
			'state_dict': model.state_dict(),
		}, is_best)
		if (epoch + 1) % args.eval_freq == 0 or epoch == args.num_epoch - 1:
			acc, val_loss = validate(target_test_loader, model, criterion)
			print("validation Epoch: {}, loss: {}, acc: {}".format(epoch+1, val_loss, acc))
			writer.add_scalar("val_epoch", epoch)
			writer.add_scalar("val_acc", acc)
			writer.add_scalar("val_loss", val_loss)
			logging.info("val_epoch: {}".format(epoch))
			logging.info("val_acc: {}".format(acc))
			logging.info("val_loss: {}".format(val_loss))
		
		if args.lr_scheduler:
			lr_scheduler.step()  # lr scheduler for each epoch
			print("Epoch: {} lr: {}".format(epoch+1, lr_scheduler.get_lr()))
		
		
if __name__ == "__main__":
	main()
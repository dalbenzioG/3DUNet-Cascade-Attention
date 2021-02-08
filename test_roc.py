import torch
import torch.nn as nn
import torchvision
import numpy as np
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, roc_curve, auc
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
import torch.optim as optim
from torch.optim import lr_scheduler
import argparse
import json
import os
import cv2
from PIL import Image
from network.models import model_selection
from network.mesonet import Meso4
from dataset.transform import xception_default_data_transforms
from dataset.mydataset import MyDataset
from dataset.transform import xception_default_data_transforms, xception_default_data_transforms_256

def main():
	args = parse.parse_args()
	test_list = args.test_list
	batch_size = args.batch_size
	model_path = args.model_path
	torch.backends.cudnn.benchmark=True
	test_dataset = MyDataset(txt_path=test_list, transform=xception_default_data_transforms['test'])
	test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=True, drop_last=True, num_workers=8)
	test_dataset_size = len(test_dataset)
	scaler = MinMaxScaler()
	true_negative = 0
	true_positive = 0
	false_negative = 0
	false_positive = 0
	y_test = list()
	y_pred = list()
	acc = 0
	#model = torchvision.models.densenet121(num_classes=2)
	model = model_selection(modelname='xception', num_out_classes=2, dropout=0.5)
	#model = Meso4()
	model.load_state_dict(torch.load(model_path))
	if isinstance(model, torch.nn.DataParallel):
		model = model.module
	model = model.cuda()
	model.eval()
	dict = {}
	with torch.no_grad():
		for (image, labels) in test_loader:
			image = image.cuda()
			labels = labels.cuda()
			outputs = model(image)
			true_labels = np.array(labels.cpu())
			output_ = scaler.fit_transform(outputs.cpu())
			output_ = output_[:, 1]
			# some doubts
			_, preds = torch.max(outputs.data, 1)

			for i in range(len(preds)):
				if preds[i] == 0 and labels.data[i] == 0:
					true_negative += 1
				elif preds[i] == 1 and labels.data[i] == 1:
					true_positive += 1
				elif preds[i] == 1 and labels.data[i] == 0:
					false_positive += 1
				elif preds[i] == 0 and labels.data[i] == 1:
					false_negative += 1

			# y_test = y_test + true_labels
			# y_pred = y_pred + output_
			y_test.extend(true_labels)
			y_pred.extend(output_)

		if 'Deepfakes' in test_list:
			dataset = 'xception_nt_df_c23'
		elif 'Face2Face' in test_list:
			dataset = 'xception_nt_f2f_c23'
		elif 'FaceSwap' in test_list:
			dataset = 'xception_nt_fs_c23'
		elif 'NeuralTextures' in test_list:
			dataset = 'xception_nt_nt_c23'

		# print('Iteration Acc {:.4f}'.format(torch.sum(preds == labels.data).to(torch.float32)/batch_size))
		acc = (true_positive + true_negative) / (true_negative+false_negative+false_positive+true_positive)
		with open('plots/' + dataset + '_labels.txt', 'w') as f:
			json.dump([int(i) for i in y_test], f)
		with open('plots/' + dataset + '_prediction.txt', 'w') as f:
			json.dump(y_pred, f)
		# roc_curve
		false_positive_rate, true_positive_rate, thresholds = roc_curve(y_test, y_pred)
		auc_value = auc(false_positive_rate, true_positive_rate)

		print('true negative: {}, false negative: {}, false positive: {}, true positive: {}'.format(true_negative, false_negative, false_positive, true_positive))
		print('Test Acc: {:.4f}'.format(acc))
		print('AUC score: {:.4f}'.format(auc_value))

		plt.title('ROC curve')
		plt.plot(false_positive_rate, true_positive_rate, 'blue', label='AUC = %0.3f' % auc_value)
		plt.legend(loc='lower right')
		plt.plot([0, 1], [0, 1], 'm--')
		plt.xlim([0, 1])
		plt.ylim([0, 1.1])
		plt.ylabel('True Positive Rate')
		plt.xlabel('False Positive Rate')
		plt.savefig('plots/'+dataset+'.png')

		# fh = open(test_list, 'r')
		#
		# for line in fh:
		# 	line = line.rstrip()
		# 	words = line.split()
		# 	fn = words[0]
		# 	img = Image.open(fn).convert('RGB')
		# 	img = img.transform(img)
		# 	label = words[1]
		#
		# 	img = img.cuda()
		# 	label = label.cuda()
		# 	output = model(img)
		#
		# 	if 'manipulated_sequences' in fn:
		# 		folder = fn[-16:-9]
		# 	else:
		# 		folder = fn[-12:-9]
		#
		# 	if folder in dict.keys():
		# 		if label == output:
		# 			dict[folder] = dict[folder] + '1'
		# 		else:
		# 			dict[folder] = dict[folder] + '0'
		# 	else:
		# 		if label == output:
		# 			dict[folder] = '1'
		# 		else:
		# 			dict[folder] = '0'
		#
		# for key in dict:
		# 	if len(key) == 7 and dict[key].count('0') > dict[key].count('1'):
		# 		true_negative += 1
		# 	if len(key) == 7 and dict[key].count('1') > dict[key].count('0'):
		# 		false_positive += 1
		# 	if len(key) == 3 and dict[key].count('0') > dict[key].count('1'):
		# 		false_negative += 1
		# 	if len(key) == 3 and dict[key].count('1') > dict[key].count('0'):
		# 		true_positive += 1

	# print('true_negative: {}'.format(true_negative))
	# print('false_positive: {}'.format(false_positive))
	# print('false_negative: {}'.format(false_negative))
	# print('true_positive: {}'.format(true_positive))

if __name__ == '__main__':
	parse = argparse.ArgumentParser(
		formatter_class=argparse.ArgumentDefaultsHelpFormatter)
	parse.add_argument('--batch_size', '-bz', type=int, default=32)
	parse.add_argument('--test_list', '-tl', type=str, default='./data_list/Deepfakes_c0_test.txt')
	parse.add_argument('--model_path', '-mp', type=str, default='./pretrained_model/df_c0_best.pkl')
	main()
	print('Hello world!!!')

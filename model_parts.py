import torch
import torch.nn as nn

from args import *
from resnet import *


'''
Siam parts
'''
class Backbone(nn.Module):
	def __init__(self):
		super(Backbone, self).__init__()
		self.model = resnet50(pretrained=True)
		self.adjust = nn.Conv2d(1024, 256, kernel_size=1) 

	def forward(self, x):
		search_cat = self.model(x)
		out = self.adjust(search_cat[4])
		return search_cat, out


class ScoreBranch(nn.Module):
	def __init__(self):
		super(ScoreBranch, self).__init__()
		self.branch = nn.Sequential(
			nn.Conv2d(256, 256, kernel_size=1),
			nn.BatchNorm2d(256),
			nn.ReLU(),
			nn.Conv2d(256, NUM_CLASSES, kernel_size=1),
			nn.Sigmoid()
			)

	def forward(self, x):
		score = self.branch(x)
		pos_list = torch.tensor([], dtype=int).to(device)
		for i in range(score.size(0)):
			max_value = score[i][0].max()
			pos = (score[i] == max_value).nonzero()[0][1:].unsqueeze(0)
			pos_list = torch.cat([pos_list, pos], dim=0)
		return score, pos_list		


class MaskBranch(nn.Module):
	def __init__(self):
		super(MaskBranch, self).__init__()
		self.deconv = nn.ConvTranspose2d(256, 32, 16, 16)

	def forward(self, masks_feat):
		# print(masks_feat.shape)
		out = self.deconv(masks_feat)
		# print(out.shape)
		return out


if __name__ == '__main__':
	model = Backbone()
	tensor = torch.rand([BATCH_SIZE, 3, 128, 128])
	search_cat, out = model(tensor)
	print(out.shape)
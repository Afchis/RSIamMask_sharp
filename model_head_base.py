import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *
from model_parts import *


'''
Model head
'''
class ModelDisigner(nn.Module):
	def __init__(self):
		super(ModelDisigner, self).__init__()
		self.backbone = Backbone()
		self.score_branch = ScoreBranch()
		self.mask_branch = MaskBranch()
		self.up = nn.Upsample(scale_factor=2, mode='nearest')
		self.final = nn.Sequential(
			nn.Conv2d(1, NUM_CLASSES, kernel_size=1),
			nn.Sigmoid()
			)

	def Correlation_func(self, s_f, t_f): # s_f-->search_feat, t_f-->target_feat
		t_f = t_f.reshape(-1, 1, t_f.size(2), t_f.size(3))
		out = s_f.reshape(1, -1, s_f.size(2), s_f.size(3)) # 1, b*ch, 32, 32
		out = F.conv2d(out, t_f, groups=t_f.size(0))
		out = out.reshape(-1, s_f.size(1), out.size(2), out.size(3))
		return out

	def Chiose_RoW(self, corr_feat, pos_list):
		corr_feat = corr_feat.reshape(BATCH_SIZE, 17, 17, 256)
		j_tensors = torch.tensor([]).to(device)
		for j in range(corr_feat.size(0)):
			j_tensor = corr_feat[j][pos_list[j, 0]][pos_list[j, 1]].unsqueeze(0)
			j_tensors = torch.cat([j_tensors, j_tensor], dim=0)
		j_tensors = j_tensors.unsqueeze(2).unsqueeze(3)
		return j_tensors


	def Choise_feat(self, feat, pos_list, x):
		feat = feat.reshape(TIMESTEPS, BATCH_SIZE, feat.size(1), feat.size(2), feat.size(3))
		feat = feat.permute(0, 1, 3, 4, 2)

		i_tensors = torch.tensor([]).to(device)
		for i in range(feat.size(0)):
			j_tensors = torch.tensor([]).to(device)
			for j in range(feat.size(1)):
				j_tensor = feat[i][j][x*pos_list[i][j][0]:x*pos_list[i][j][0]+x*16, x*pos_list[i][j][1]:x*pos_list[i][j][1]+x*16, :].unsqueeze(0)
				j_tensors = torch.cat([j_tensors, j_tensor], dim=0)
			i_tensor = j_tensors.unsqueeze(0)
			i_tensors = torch.cat([i_tensors, i_tensor], dim=0)

		feat = i_tensors.permute(0, 1, 4, 2, 3)
		feat = feat.reshape(TIMESTEPS*BATCH_SIZE, feat.size(2), feat.size(3), feat.size(4))
		return feat


	def forward(self, target, searchs):
		_,  target_feat = self.backbone(target)
		search_cats, searchs_feat = self.backbone(searchs)
		corr_feat = self.Correlation_func(searchs_feat, target_feat)
		##### Score Branch #####
		score, pos_list = self.score_branch(corr_feat)
		# print(pos_list)
		##### Mask Branch #####
		masks_feat = self.Chiose_RoW(corr_feat, pos_list)
		mask = self.mask_branch(masks_feat).reshape(BATCH_SIZE, 1, 64, 64)
		mask = self.up(mask)
		mask = self.final(mask)
		return score, mask


if __name__ == '__main__':
	model = ModelDisigner()
	model = model.to(device)
	target = torch.rand([BATCH_SIZE, 3, 128, 128]).to(device)
	searchs = torch.rand([BATCH_SIZE, 3, 256, 256]).to(device)
	score, mask = model(target, searchs)
	print('score.shape: ', score.shape)
	print('mask.shape: ', mask.shape)

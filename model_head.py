import torch
import torch.nn as nn
import torch.nn.functional as F

from args import *
from model_parts import *


'''
Model head
'''
class Refine(nn.Module):
	def __init__(self):
		super(Refine, self).__init__()
		self.v0 = nn.Sequential(
			nn.Conv2d(64, 16, 3, padding=1), nn.ReLU(),
			nn.Conv2d(16, 4, 3, padding=1),nn.ReLU()
			)
		self.v1 = nn.Sequential(
			nn.Conv2d(256, 64, 3, padding=1), nn.ReLU(),
			nn.Conv2d(64, 16, 3, padding=1), nn.ReLU()
			)
		self.v2 = nn.Sequential(
			nn.Conv2d(512, 128, 3, padding=1), nn.ReLU(),
			nn.Conv2d(128, 32, 3, padding=1), nn.ReLU()
			)
		self.h2 = nn.Sequential(
			nn.Conv2d(32, 32, 3, padding=1), nn.ReLU(),
			nn.Conv2d(32, 32, 3, padding=1), nn.ReLU()
			)
		self.h1 = nn.Sequential(
			nn.Conv2d(16, 16, 3, padding=1), nn.ReLU(),
			nn.Conv2d(16, 16, 3, padding=1), nn.ReLU()
			)
		self.h0 = nn.Sequential(
			nn.Conv2d(4, 4, 3, padding=1), nn.ReLU(),
			nn.Conv2d(4, 4, 3, padding=1), nn.ReLU()
			)
		self.post0 = nn.Conv2d(32, 16, 3, padding=1)
		self.post1 = nn.Conv2d(16, 4, 3, padding=1)
		self.post2 = nn.Conv2d(4, 1, 3, padding=1)

	def Choise_feat(self, feat, pos_list, x):
		feat = feat.permute(0, 2, 3, 1)
		j_tensors = torch.tensor([]).to(device)
		for j in range(feat.size(0)):
			j_tensor = feat[j][x*pos_list[j][0]:x*pos_list[j][0]+x*16, x*pos_list[j][1]:x*pos_list[j][1]+x*16, :].unsqueeze(0)
			j_tensors = torch.cat([j_tensors, j_tensor], dim=0)
		feat = j_tensors.permute(0, 3, 1, 2)
		return feat

	def forward(self, mask, search_cats, pos_list):
		feat3 = self.Choise_feat(search_cats[3], pos_list, 1)
		mask = self.post0(F.upsample(self.v2(feat3) + self.h2(mask), size=(32, 32)))
		feat2 = self.Choise_feat(search_cats[2], pos_list, 2)
		mask = self.post1(F.upsample(self.v1(feat2) + self.h1(mask), size=(64, 64)))
		feat1 = self.Choise_feat(search_cats[1], pos_list, 4)
		mask = self.post2(F.upsample(self.v0(feat1) + self.h0(mask), size=(128, 128)))	
		return mask


class ModelDisigner(nn.Module):
	def __init__(self):
		super(ModelDisigner, self).__init__()
		self.backbone = Backbone()
		self.score_branch = ScoreBranch()
		self.mask_branch = MaskBranch()
		self.refine = Refine()
		self.final = nn.Sequential(
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


	def forward(self, target, searchs):
		_,  target_feat = self.backbone(target)
		search_cats, searchs_feat = self.backbone(searchs)
		corr_feat = self.Correlation_func(searchs_feat, target_feat)
		##### Score Branch #####
		score, pos_list = self.score_branch(corr_feat)
		##### Mask Branch #####
		masks_feat = self.Chiose_RoW(corr_feat, pos_list)
		mask = self.mask_branch(masks_feat)
		mask = self.refine(mask, search_cats, pos_list)
		mask = self.final(mask)
		return score, mask


if __name__ == '__main__':
	model = ModelDisigner()
	model = model.to(device)
	target = torch.rand([BATCH_SIZE, 3, 128, 128]).to(device)
	searchs = torch.rand([BATCH_SIZE, 3, 256, 256]).to(device)
	score, mask = model(target, searchs)
	# print('score.shape: ', score.shape)
	# print('mask.shape: ', mask.shape)

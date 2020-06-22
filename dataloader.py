import json
import os
import numpy as np

from PIL import Image, ImageDraw
import scipy.ndimage.morphology as morph

import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchvision
import torchvision.transforms as transforms

from args import *


class Data_train(Dataset):
    def __init__(self):
        super().__init__()
        self.data_path = "/storage/ProtopopovI/_project_/RSiamMask/pathignore/data/train/"
        self.trans = transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            transforms.ToTensor()
            ])
        self.ccrop = transforms.Compose([
            transforms.Resize((256, 256), interpolation=0),
            transforms.CenterCrop((128, 128)),
            transforms.ToTensor()
            ])

    def list_dir(self, object):
        return sorted(os.listdir(self.data_path + 'bike/' + object))

    def len_dit(self):
        return len(self.list_dir('/imgs'))

    def transform_score_label(self, depth2):
        depth2 = depth2.reshape(1, 1, depth2.size(0), depth2.size(1))
        max_value = depth2.max()
        depth2 = (depth2 == max_value).float()
        score_label = F.max_pool2d(depth2, kernel_size=(16, 16), padding=8, stride=16)
        score_zero = (score_label == 0).float()
        score_label = torch.stack([score_zero, score_label], dim=1).squeeze()
        return score_label

    def get_labels(self, object):
        labels = torch.tensor([])
        depths = torch.tensor([])
        score_labels = torch.tensor([])
        label1 = (object==0).float()
        depth1 = torch.tensor(morph.distance_transform_edt(np.asarray(label1[0])))
        label2 = (label1==0).float()
        depth2 = torch.tensor(morph.distance_transform_edt(np.asarray(label2[0])))
        depth = (depth1 + depth2).float().unsqueeze(0)
        label = torch.stack([label1, label2], dim=1)
        labels = torch.cat([labels, label], dim=0)
        depths = torch.cat([depths, depth], dim=0)
        score_label = self.transform_score_label(depth2).unsqueeze(0)
        score_labels = torch.cat([score_labels, score_label], dim=0)
        labels = labels.squeeze()
        score_labels = score_labels.squeeze()
        return labels, depths, score_labels

    def Choise_feat(self, label, score_label, x=8):
        score_label = score_label[0][1]
        max_value = score_label.max()
        pos = (score_label == max_value).nonzero()#.unsqueeze(0)

        label = label.permute(0, 2, 3, 1)
        i_tensors = torch.tensor([])
        for i in range(label.size(0)):
            i_tensor = label[i][x*pos[i][0]:x*pos[i][0]+x*16, x*pos[i][1]:x*pos[i][1]+x*16, :].unsqueeze(0)
            i_tensors = torch.cat([i_tensors, i_tensor], dim=0)

        label = i_tensors.permute(0, 3, 1, 2)
        return label
    
    def __len__(self):
        return 32
        # return len(self.list_dir('imgs'))*len(self.list_dir('imgs'))
    
    def  __getitem__(self, idx):
        search_name = self.list_dir('imgs')[(idx%self.len_dit())]
        # target_name = self.list_dir('imgs')[(idx//self.len_dit())]
        mask_name = self.list_dir('masks')[(idx%self.len_dit())]
        # search_name = self.list_dir('imgs')[idx*4]
        target_name = self.list_dir('imgs')[10]
        # mask_name = self.list_dir('masks')[idx*4]
        search = Image.open(self.data_path + 'bike/imgs/' + search_name).convert('RGB')
        target = Image.open(self.data_path + 'bike/imgs/' + target_name).convert('RGB')
        mask = Image.open(self.data_path + 'bike/masks/' + mask_name).convert('L')
        search = self.trans(search)
        target = self.ccrop(target)
        mask = self.trans(mask)
        label, depth, score_label = self.get_labels(mask)
        search, label, depth, score_label = search.unsqueeze(0), label.unsqueeze(0), depth.unsqueeze(0), score_label.unsqueeze(0)

        label = self.Choise_feat(label, score_label)
        depth = self.Choise_feat(depth, score_label)

        search, label, depth, score_label = search.squeeze(), label.squeeze(), depth.squeeze(), score_label.squeeze()

        return target, search, label[1:], depth, score_label[1:]

if BATCH_SIZE == 1:
    Shuffle = False
else:
    Shuffle = True

train_dataset = Data_train()
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=BATCH_SIZE,
                          num_workers=1,
                          shuffle=Shuffle)


data_loaders = {
    'train' : train_loader,
    # 'valid' : valid_loader
}

if __name__ == '__main__':
    print('Write number of image in dataset: ')
    inp = int(input())
    search, target, label, depth, score_label = train_dataset[inp]
    # print('search', search, 'target', target, 'label', label, 'depth', depth, 'score_label', score_label)
    print('target.shape', target.shape)
    print('search.shape', search.shape)
    print('label.shape', label.shape)
    print('depth.shape', depth.shape)
    print('score_label.shape', score_label.shape)
    # print(label[0, 0])
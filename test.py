import torch

from args import *
from model_head import *
from dataloader import *
from loss_metric import *

if BATCH_SIZE == 1:

	to_pil = transforms.ToPILImage()

	model = ModelDisigner()
	model = model.to(device)
	print('Tensorboard graph name: ')
	GRAPH_NAME = input()
	model.load_state_dict(torch.load('pathignore/weights/%s.pth' % GRAPH_NAME))
	# model.load_state_dict(torch.load('pathignore/weights/base_model.pth'))

	def save_img(object, object2,  j):
		# for batch in range(BATCH_SIZE):
		imgs = object[0][0]
		img = (imgs > 0.5).float()
		# print(imgs.shape)
		img = to_pil(img)
		img.save("test_output/frame%d_mask.png" % i)

		masks = object2[0][0]
		mask = (masks > 0.5).float()
		mask = to_pil(mask)
		mask.save("test_output/frame%d_pred.png" % i)
		print('save!!!', i)


	for i, data in enumerate(train_loader):
		target, searchs, labels, depths, score_labels = data
		target, searchs, labels, depths, score_labels \
			 = target.to(device), searchs.to(device), labels.to(device), depths.to(device), score_labels.to(device)
		pred_score, pred_mask = model(target, searchs)
		save_img(labels.cpu(), pred_mask.cpu(), i)

else:
	print('Change batch_size')


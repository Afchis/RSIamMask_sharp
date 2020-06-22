import torch
from torch.utils.tensorboard import SummaryWriter

from args import *
from model_head import *
from dataloader import *
from loss_metric import *

print('BATCH_SIZE:' , BATCH_SIZE)
print('Tensorboard graph name: ')
GRAPH_NAME = 1#input()
print('Learning_rate: ')
LEARNING_RATE = 0.01#float(input())
print('Num epochs: ')
NUM_EPOCH = 10#int(input())


writer = SummaryWriter()

model = ModelDisigner()
model = model.to(device)
# model.load_state_dict(torch.load('pathignore/weights/run_00.pth'))

try:
	model.load_state_dict(torch.load('pathignore/weights/base.pth'), strict=False)# 
except FileNotFoundError:
	print('!!!Create new weights!!!')
	pass

optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)



# for name, param in model.named_parameters():
#     print('name: ', name)
#     print(type(param))
#     print('param.shape: ', param.shape)
#     print('param.requires_grad: ', param.requires_grad)
#     print('=====')

from model_head_base import ModelDisigner as model_base
base_model = model_base()

for name, param in base_model.named_children():
    print('name: ', name)
    print(type(param))
    print('=====')



print('done!')
quit()



iter = 0
for epoch in range(NUM_EPOCH):
	for phase in ['train']:#, 'valid'#
		if phase == 'train':
			accuracy = []
			loss_mean = []
			print('*'*10, 'epoch: ', epoch, '*'*10)
			for i, data in enumerate(data_loaders[phase]):
				target, searchs, labels, depths, score_labels = data
				target, searchs, labels, depths, score_labels \
				 = target.to(device), searchs.to(device), labels.to(device), depths.to(device), score_labels.to(device)
				pred_scores, pred_masks = model(target, searchs)
				metric = iou_metric(pred_masks, labels, depths).mean().item()
				loss = all_losses(pred_masks, labels, depths, pred_scores, score_labels)
				loss.backward()
				optimizer.step()
				scheduler.step()
				optimizer.zero_grad()
				accuracy.append(metric)
				loss_mean.append(loss.mean().item())
				if iter % 1 == 0 and iter !=0:
					print('iter: ', iter, 'loss: ', torch.tensor(loss_mean).mean().item(), 'accuracy: ', torch.tensor(accuracy).mean().item())
					writer.add_scalars('%s_loss' % GRAPH_NAME, {'train' : torch.tensor(loss_mean).mean()}, iter)
					writer.add_scalars('%s_accuracy' % GRAPH_NAME, {'train' : torch.tensor(accuracy).mean()}, iter)
				if iter % 1 == 0:
					accuracy = []
					loss_mean = []
				iter += 1
			if epoch % 5 == 0:
				torch.save(model.state_dict(), 'pathignore/weights/%s.pth' % GRAPH_NAME)
				print('WEIGHTS IS SAVED: pathignore/weights/%s.pth' % GRAPH_NAME)
			print('LEARNING_RATE:', LEARNING_RATE)
			print('Valid:')
		elif phase == 'valid':
			accuracy = []
			loss_mean = []
			for i, data in enumerate(data_loaders[phase]):
				target, searchs, labels, depths, score_labels = data
				target, searchs, labels, depths, score_labels \
				 = target.to(device), searchs.to(device), labels.to(device), depths.to(device), score_labels.to(device)
				# try:
				pred_scores, pred_masks = model(target, searchs)
				metric = iou_metric(pred_masks, labels, depths).mean().item()
				loss = all_losses(pred_masks, labels, depths, pred_scores, score_labels)
				accuracy.append(metric)
				loss_mean.append(loss.mean().item())
				if iter % 10 == 0:
					print('iter: ', iter, 'loss: ', torch.tensor(loss_mean).mean().item(), 'accuracy: ', torch.tensor(accuracy).mean().item())
					writer.add_scalars('%s_loss' % GRAPH_NAME, {'valid' : torch.tensor(loss_mean).mean()}, iter)
					writer.add_scalars('%s_accuracy' % GRAPH_NAME, {'valid' : torch.tensor(accuracy).mean()}, iter)
				if iter % 10 == 0:
					accuracy = []
					loss_mean = []
				iter += 1



		# except RuntimeError:
		# 	pass
print('Tensorboard graph name: ', GRAPH_NAME)
writer.close()

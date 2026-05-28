import torch
import random
import numpy as np
import torch.nn.functional as F
from utils.dataloader import TrainDataset
from utils.LRScheduler import CosineDecay
from config import Config
from tqdm import tqdm
from utils.loss import L_color, L_spa, L_exp, L_TV
from utils.AdaX import AdaXW


def structure_loss(logits, mask):
	"""
    loss function (ref: F3Net-AAAI-2020)

    pred: logits without activation
    mask: binary mask {0, 1}
    """
	weit = 1 + 5 * torch.abs(F.avg_pool2d(mask, kernel_size=31, stride=1, padding=15) - mask)
	wbce = F.binary_cross_entropy_with_logits(logits, mask, reduction='mean')
	wbce = (weit * wbce).sum(dim=(2, 3)) / weit.sum(dim=(2, 3))

	pred = torch.sigmoid(logits)
	inter = ((pred * mask) * weit).sum(dim=(2, 3))
	union = ((pred + mask) * weit).sum(dim=(2, 3))
	wiou = 1 - (inter + 1) / (union - inter + 1)
	return (wbce + wiou).mean()


def train():
	global model, train_datald, optimizer, cfg, scheduler

	l_color = L_color()
	l_spa = L_spa()
	l_exp = L_exp(16)
	l_TV = L_TV()
	for epoch in range(cfg.epochs):
		model.train()

		loss_iter = []
		for org_img, image, dop, gt in tqdm(train_datald):
			optimizer.zero_grad()

			img = org_img.to(cfg.device)
			dop = dop.to(cfg.device)
			gt = gt.to(cfg.device)

			out1, out2, out3, out4, e_x, x_r, x_cod = model(img, dop)

			loss_TV = 1600 * l_TV(x_r)
			loss_spa = torch.mean(l_spa(e_x, img))
			loss_col = 5 * torch.mean(l_color(e_x))
			loss_exp = 10 * torch.mean(l_exp(e_x, 0.6))
			lle_loss = loss_TV + loss_spa + loss_exp + loss_col
			oloss = structure_loss(x_cod, gt) * 0.25
			loss1 = structure_loss(out1, gt)
			loss2 = structure_loss(out2, gt) * 0.5
			loss3 = structure_loss(out3, gt) * 0.25
			loss4 = structure_loss(out4, gt) * 0.125
			loss = loss1 + loss2 + loss3 + loss4 + lle_loss + oloss
			loss.backward()

			optimizer.step()
			loss_iter.append(loss.item())

		print(f'Epoch: {epoch + 1}, LR: {np.round(scheduler.get_lr(), 8)}, Loss: {np.round(np.mean(loss_iter), 8)}')
		scheduler.step()

		if (epoch+1) % 5 == 0 or epoch == cfg.epochs-1:
			torch.save(model.state_dict(),  f'save_pth/epoch_{epoch+1}.pth')


if __name__ == '__main__':
	seed = 123456
	random.seed(seed)
	np.random.seed(seed)
	torch.manual_seed(seed)
	torch.cuda.manual_seed(seed)
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False
	torch.backends.cudnn.enabled = False

	cfg = Config()

	from Model.C2Net import C2Net
	model = C2Net(channels=48, lle_channels=16).to(cfg.device)

	train_dataset = TrainDataset(image_root=[cfg.dp.train_PCOD_imgs],
	                             gt_root=[cfg.dp.train_PCOD_masks],
	                             dop_root=[cfg.dp.train_PCOD_dops],
	                             trainsize=cfg.trainsize)
	train_datald = torch.utils.data.DataLoader(dataset=train_dataset,
	                                           batch_size=cfg.batch_size,
	                                           shuffle=True,
	                                           num_workers=cfg.num_workers,
	                                           pin_memory=True)


	optimizer = AdaXW(model.parameters(), lr=cfg.learning_rate, weight_decay=cfg.weight_decay)
	scheduler = CosineDecay(optimizer, max_lr=cfg.learning_rate, min_lr=cfg.min_lr, max_epoch=cfg.epochs)

	train()

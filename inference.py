import os
import torch
import numpy as np
from tqdm import tqdm
import torch.nn.functional as F
import cv2
from Model.C2Net import C2Net
from config import Config
from utils.dataloader import TestDataset


def inference(datasets):
	global model, cfg
	model.eval()
	for dataset in datasets:
		assert dataset in ['PCOD']
		save_path = os.path.join('prediction_maps', dataset)
		os.makedirs(save_path, exist_ok=True)

		test_dataset = TestDataset(image_root=[cfg.dp.test_PCOD_imgs],
		                           gt_root=[cfg.dp.test_PCOD_masks],
		                           dop_root=[cfg.dp.test_PCOD_dops],
		                           testsize=cfg.trainsize,
		                           edge_root=None)

		for org_img, image, dop, gt, gt_origin, name in tqdm(test_dataset):
			img = org_img.unsqueeze(0).to(cfg.device)
			dop = dop.unsqueeze(0).to(cfg.device)
			gt = gt_origin.to(cfg.device)
			out1, out2, out3, out4, e_x, x_r, x_cod = model(img, dop)
			out1 = F.interpolate(out1, size=gt_origin.shape[1:], mode='bilinear', align_corners=True)
			out1 = torch.sigmoid(out1) * 255
			out1 = out1.squeeze(0).squeeze(0).detach().cpu().numpy().astype(np.uint8)
			# save preds
			cv2.imwrite(os.path.join(save_path, name), out1)


if __name__ == '__main__':
	pth_path = 'save_pth/epoch_5.pth'

	cfg = Config()
	model = C2Net(channels=48, lle_channels=16).to(cfg.device)
	model.load_state_dict(torch.load(pth_path))

	datasets = ['PCOD']
	inference(datasets)

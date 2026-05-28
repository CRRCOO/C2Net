import os
import torch


class Config:
    def __init__(self):
        dataset_dir = './data/COD'
        self.dp = DataPath(dataset_dir)
        self.num_workers = 8

        self.CUDA = True
        self.device = torch.device('cuda' if self.CUDA else 'cpu')

        self.epochs = 100
        self.trainsize = 352
        self.batch_size = 4
        self.weight_decay = 4e-7
        self.learning_rate = 8e-5
        self.min_lr = 1e-7


class DataPath:
    def __init__(self, dataset_dir):
        self.dataset_dir = dataset_dir

        ''' Train Dataset '''
        # PCOD_1200
        self.train_PCOD_imgs = os.path.join(self.dataset_dir, 'PCOD_1200', 'train', 'train-rgb')
        self.train_PCOD_masks = os.path.join(self.dataset_dir, 'PCOD_1200', 'train', 'train-gt')
        self.train_PCOD_dops = os.path.join(self.dataset_dir, 'PCOD_1200', 'train', 'train-dop')
        ''' Test Dataset '''
        # PCOD_1200
        self.test_PCOD_imgs = os.path.join(self.dataset_dir, 'PCOD_1200', 'test', 'test-rgb')
        self.test_PCOD_masks = os.path.join(self.dataset_dir, 'PCOD_1200', 'test', 'test-gt')
        self.test_PCOD_dops = os.path.join(self.dataset_dir, 'PCOD_1200', 'test', 'test-dop')

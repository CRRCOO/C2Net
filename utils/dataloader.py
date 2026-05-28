import os
import torch
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
from utils.data_augmentation import cv_random_flip, randomCrop, randomRotation, randomPeper, colorEnhance
import cv2
from utils.tools import normalize


class TrainDataset(Dataset):
    def __init__(self, image_root, gt_root, dop_root, trainsize, edge_root=None, rVFlip=True, rCrop=True, rRotate=True,
                 colorEnhance=False, rPeper=False):
        self.edge_root = edge_root
        self.trainsize = trainsize
        self.rVFlip = rVFlip
        self.rCrop = rCrop
        self.rRotate = rRotate
        self.colorEnhance = colorEnhance
        self.rPeper = rPeper

        self.images, self.gts, self.dops, self.edges, self.enhance_imgs = [], [], [], [], []
        for i in range(len(image_root)):
            self.images.extend([os.path.join(image_root[i], f) for f in os.listdir(image_root[i]) if
                           f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tiff')])
            self.gts.extend([os.path.join(gt_root[i], f) for f in os.listdir(gt_root[i]) if f.endswith('.png') or f.endswith('.jpg')])
            self.dops.extend([os.path.join(dop_root[i], f) for f in os.listdir(dop_root[i]) if f.endswith('.jpg') or f.endswith('.tiff')])
            if edge_root is not None:
                self.edges.extend([os.path.join(edge_root[i], f) for f in os.listdir(edge_root[i]) if f.endswith('.jpg')
                              or f.endswith('.png')])

        # sorted files
        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.dops = sorted(self.dops)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        self.img_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])
        self.gt_transform = transforms.Compose([
            transforms.Resize((self.trainsize, self.trainsize)),
            transforms.ToTensor()])

        self.size = len(self.images)
        print('>>> trainig/validing with {} samples'.format(self.size))

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        dop = self.binary_loader(self.dops[index])
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        # Data Augmentation
        # random horizental flipping
        if self.edge_root is not None:
            if self.rVFlip:
                image, gt, dop, edge = cv_random_flip([image, gt, dop, edge])
            if self.rCrop:
                image, gt, dop, edge = randomCrop([image, gt, dop, edge])
            if self.rRotate:
                image, gt, dop, edge = randomRotation([image, gt, dop, edge])
        else:
            if self.rVFlip:
                image, gt, dop = cv_random_flip([image, gt, dop])
            if self.rCrop:
                image, gt, dop = randomCrop([image, gt, dop])
            if self.rRotate:
                image, gt, dop = randomRotation([image, gt, dop])
        # bright, contrast, color, sharp jitters
        if self.colorEnhance:
            image = colorEnhance(image)
        # random peper noise
        if self.rPeper:
            gt = randomPeper(gt)

        org_img = self.gt_transform(image)

        image = self.img_transform(image)
        gt = self.gt_transform(gt)

        # processing DOP
        dop_origin = transforms.PILToTensor()(dop)  # 0-255 no-resize
        dop_resize = transforms.Resize((self.trainsize, self.trainsize))(dop_origin).to(
            torch.float32)  # 0-small_value resize
        dop = normalize(dop_resize)  # 0-1 resize

        if self.edge_root is not None:
            edge = self.gt_transform(edge)
        if self.edge_root is not None:
            return image, dop, gt, edge
        else:
            return org_img, image, dop, gt

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGRA2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(img, mode='L')

    def resize(self, img, gt):
        assert img.size == gt.size
        w, h = img.size
        if h < self.trainsize or w < self.trainsize:
            h = max(h, self.trainsize)
            w = max(w, self.trainsize)
            return img.resize((w, h), Image.BILINEAR), gt.resize((w, h), Image.NEAREST)
        else:
            return img, gt

    def __len__(self):
        return self.size


class TestDataset(Dataset):
    def __init__(self, image_root, gt_root, dop_root, testsize, edge_root=None):
        self.testsize = testsize
        self.edge_root = edge_root

        self.images, self.gts, self.dops, self.edges, self.enhance_imgs = [], [], [], [], []
        for i in range(len(image_root)):
            self.images.extend([os.path.join(image_root[i], f) for f in os.listdir(image_root[i]) if
                                f.endswith('.jpg') or f.endswith('.png') or f.endswith('.tiff')])
            self.gts.extend(
                [os.path.join(gt_root[i], f) for f in os.listdir(gt_root[i]) if f.endswith('.tif') or f.endswith('.png') or f.endswith('.jpg')])
            self.dops.extend([os.path.join(dop_root[i], f) for f in os.listdir(dop_root[i]) if f.endswith('.tif') or f.endswith('.jpg') or f.endswith('.tiff')])
            if edge_root is not None:
                self.edges.extend([os.path.join(edge_root[i], f) for f in os.listdir(edge_root[i]) if f.endswith('.jpg')
                                   or f.endswith('.png')])

        self.images = sorted(self.images)
        self.gts = sorted(self.gts)
        self.dops = sorted(self.dops)
        if edge_root is not None:
            self.edges = sorted(self.edges)

        self.transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406],
                                 [0.229, 0.224, 0.225])])

        self.gt_transform = transforms.Compose([
            transforms.Resize((self.testsize, self.testsize)),
            transforms.ToTensor()])

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.rgb_loader(self.images[index])
        gt = self.binary_loader(self.gts[index])
        dop = self.binary_loader(self.dops[index])
        if self.edge_root is not None:
            edge = self.binary_loader(self.edges[index])

        org_img = self.gt_transform(image)

        image = self.transform(image)
        gt_origin = transforms.PILToTensor()(gt)
        gt = self.gt_transform(gt)

        # processing DOP
        dop = transforms.PILToTensor()(dop)  # 0-255 no-resize
        dop = transforms.Resize((self.testsize, self.testsize))(dop).to(torch.float32)  # 0-small_value resize
        dop = normalize(dop)  # 0-1 resize

        if self.edge_root is not None:
            edge_origin = transforms.PILToTensor()(edge)
            edge = self.gt_transform(edge)
        name = self.images[index].split('/')[-1]
        if '\\' in name:
            name = name.split('\\')[-1]
        if name.endswith('.jpg'):
            name = name.split('.jpg')[0] + '.png'
        if self.edge_root is not None:
            return image, dop, gt, gt_origin, edge, edge_origin, name
        else:
            return org_img, image, dop, gt, gt_origin, name

    def rgb_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, code=cv2.COLOR_BGRA2RGB)
        img = Image.fromarray(img, mode='RGB')
        return img

    def binary_loader(self, path):
        img = cv2.imread(path, flags=cv2.IMREAD_GRAYSCALE)
        return Image.fromarray(img, mode='L')

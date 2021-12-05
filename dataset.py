import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset
import torchvision.transforms as transforms
import random
from utils import crop, flip

class P2PDataset(Dataset):
    def __init__(self, root_dir, config):
        self.root_dir = root_dir
        self.config = config
        self.list_files = os.listdir(self.root_dir)

    def __len__(self):
        return len(self.list_files)

    def __getitem__(self, index):
        img_file = self.list_files[index]
        img_path = os.path.join(self.root_dir, img_file)
        image = Image.open(img_path).convert("RGB")

        w, h = image.size
        w2 = int(w / 2)

        if (self.config.TRAIN_FLIP):
            input_image = image.crop((w2, 0, w, h))
            target_image = image.crop((0, 0, w2, h))
        else:
            input_image = image.crop((0, 0, w2, h))
            target_image = image.crop((w2, 0, w, h))

        # apply the same transform to both A and B
        transform_params = self.get_params(input_image.size)
        A_transform = self.get_transform(transform_params)
        B_transform = self.get_transform(transform_params)

        input_image = A_transform(input_image)
        target_image = B_transform(target_image)

        return input_image, target_image

    def get_params(self, size):
        w, h = size
        new_h = h
        new_w = w
        new_h = new_w = 286

        x = random.randint(0, np.maximum(0, new_w - 256))
        y = random.randint(0, np.maximum(0, new_h - 256))

        flip = random.random() > 0.5

        return {'crop_pos': (x, y), 'flip': flip}


    def get_transform(self, opt, params=None, method=Image.BICUBIC, convert=True):
        transform_list = []
        osize = [256, 256]
        transform_list.append(transforms.Resize(osize, method))
        if params is None:
            transform_list.append(transforms.RandomCrop(256))
        else:
            transform_list.append(transforms.Lambda(lambda img: crop(img, params['crop_pos'], 256)))

        if params is None:
            transform_list.append(transforms.RandomHorizontalFlip())
        elif params['flip']:
            transform_list.append(transforms.Lambda(lambda img: flip(img, params['flip'])))

        if convert:
            transform_list += [transforms.ToTensor()]
            transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
        return transforms.Compose(transform_list)
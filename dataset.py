import numpy as np
import os
from PIL import Image
from torch.utils.data import Dataset

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
        image = np.array(Image.open(img_path))
        input_image = image[:, :self.config.IMAGE_SPLIT_POS, :]
        target_image = image[:, self.config.IMAGE_SPLIT_POS: self.config.IMAGE_LENGTH, :]

        augmentations = self.config.both_transform(image=input_image, image0=target_image)
        input_image = augmentations["image"]
        target_image = augmentations["image0"]

        input_image = self.config.transform_only_input(image=input_image)["image"]
        target_image = self.config.transform_only_mask(image=target_image)["image"]

        return input_image, target_image
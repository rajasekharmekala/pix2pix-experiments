from configurations import all_configs
from dataset import P2PDataset
from torch.utils.data import DataLoader
from generator import Generator
from utils import load_checkpoint, evaluate_model_examples
import torch.optim as optim


import os

import argparse

parser = argparse.ArgumentParser(description='Evaluate Pix2Pix Model.')
parser.add_argument('--dataset', metavar='d', type=str ,required=True,
                    help='provide datasetname')
args = parser.parse_args()
from box import Box



merge_config = all_configs["common"]
merge_config.update(all_configs[args.dataset])
config = Box(merge_config)

try:
    folder = os.environ['SAVE_FOLDER']
    if(folder is not None):
        config.SAVE_FOLDER = folder
    folder = os.environ['EVALUATE_FOLDER']
    if(folder is not None):
        config.EVALUATE_FOLDER = folder
except:
    pass


def evaluate():
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    load_checkpoint(config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, config)

    val_dataset = P2PDataset(root_dir=config.VAL_DIR, config=config)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)
    evaluate_model_examples(gen, val_loader, config, folder=config.EVALUATE_FOLDER)

if __name__ == "__main__":
    evaluate()


import torch
from utils import save_checkpoint, load_checkpoint, save_some_examples
import torch.nn as nn
import torch.optim as optim
from configurations import all_configs
from dataset import P2PDataset
from generator import Generator
from discriminator import Discriminator
from torch.utils.data import DataLoader
from tqdm import tqdm
import os

from box import Box

import argparse

parser = argparse.ArgumentParser(description='Train Pix2Pix Model.')
parser.add_argument('--dataset', metavar='d', type=str ,required=True,
                    help='provide datasetname')
parser.add_argument('--n_layers',default=3, metavar='l', type=int ,
                    help='provide layers count to be used in patchGAN')

parser.add_argument('--train_flip', type=bool, metavar='f',
                    help='set True to flip direction of training')

args = parser.parse_args()

merge_config = all_configs["common"]
merge_config.update(all_configs[args.dataset])
config = Box(merge_config)

try:
    folder = os.environ['SAVE_FOLDER']
    if(folder is not None):
        config.SAVE_FOLDER = folder
    folder = os.environ['SAMPLES']
    if(folder is not None):
        config.SAMPLES = folder
except:
    pass


def train_fn(
    disc, gen, loader, opt_disc, opt_gen, l1_loss, bce, g_scaler, d_scaler,
):
    loop = tqdm(loader, leave=True)

    for idx, (x, y) in enumerate(loop):
        x = x.to(config.DEVICE)
        y = y.to(config.DEVICE)

        # Train Discriminator
        with torch.cuda.amp.autocast():
            y_fake = gen(x)
            D_real = disc(x, y)
            D_real_loss = bce(D_real, torch.ones_like(D_real))
            D_fake = disc(x, y_fake.detach())
            D_fake_loss = bce(D_fake, torch.zeros_like(D_fake))
            D_loss = (D_real_loss + D_fake_loss) / 2

        disc.zero_grad()
        D_loss.backward()
        opt_disc.step()

        # Train generator
        with torch.cuda.amp.autocast():
            D_fake = disc(x, y_fake)
            G_fake_loss = bce(D_fake, torch.ones_like(D_fake))
            L1 = l1_loss(y_fake, y) * config.L1_LAMBDA
            G_loss = G_fake_loss + L1

        opt_gen.zero_grad()
        G_loss.backward()
        opt_gen.step()

        if idx % 10 == 0:
            loop.set_postfix(
                D_real=torch.sigmoid(D_real).mean().item(),
                D_fake=torch.sigmoid(D_fake).mean().item(),
            )


def main():
    disc = Discriminator(in_channels=3, num_layers=args.n_layers).to(config.DEVICE)
    gen = Generator(in_channels=3, features=64).to(config.DEVICE)
    opt_disc = optim.Adam(disc.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999),)
    opt_gen = optim.Adam(gen.parameters(), lr=config.LEARNING_RATE, betas=(0.5, 0.999))
    BCE = nn.BCEWithLogitsLoss()
    L1_LOSS = nn.L1Loss()

    if config.LOAD_MODEL:
        try:
            load_checkpoint(
                config.CHECKPOINT_GEN, gen, opt_gen, config.LEARNING_RATE, config
            )
            load_checkpoint(
                config.CHECKPOINT_DISC, disc, opt_disc, config.LEARNING_RATE, config
            )
            print("Successfully loaded previous checkpoint")
        except:
            print("Failed to load previous checkpoint. Learning from scratch")
            pass

    train_dataset = P2PDataset(root_dir=config.TRAIN_DIR, config=config)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        # num_workers=config.NUM_WORKERS,
    )
    g_scaler = torch.cuda.amp.GradScaler()
    d_scaler = torch.cuda.amp.GradScaler()
    val_dataset = P2PDataset(root_dir=config.VAL_DIR, config=config)
    val_loader = DataLoader(val_dataset, batch_size=1, shuffle=False)

    for epoch in range(1, config.NUM_EPOCHS+1):
        train_fn(
            disc, gen, train_loader, opt_disc, opt_gen, L1_LOSS, BCE, g_scaler, d_scaler,
        )

        if config.SAVE_MODEL and (epoch % 20 ==0 or epoch == config.NUM_EPOCHS ):
            save_checkpoint(gen, opt_gen, epoch, config, filename=config.CHECKPOINT_GEN)
            save_checkpoint(disc, opt_disc, epoch, config, filename=config.CHECKPOINT_DISC)

        save_some_examples(gen, val_loader, epoch, config, folder=config.SAMPLES)


if __name__ == "__main__":
    main()
import torch
from torchvision.utils import save_image
import os
from torch.autograd import Variable
from tqdm import tqdm

def save_some_examples(gen, val_loader, epoch, config, folder):
    x, y = next(iter(val_loader))
    x, y = x.to(config.DEVICE), y.to(config.DEVICE)
    gen.eval()
    if not(os.path.isdir(folder)): os.mkdir(folder)
    with torch.no_grad():
        y_fake = gen(x)
        y_fake = y_fake * 0.5 + 0.5  # remove normalization#
        save_image(y_fake, folder + f"/{epoch}_y_gen.png")
        save_image(x * 0.5 + 0.5, folder + f"/{epoch}_input.png")
        if epoch == 1:
            save_image(y * 0.5 + 0.5, folder + f"/label_{epoch}.png")
    gen.train()

def evaluate_model_examples(gen, val_loader, config, folder):
    gen.eval()
    loop = tqdm(val_loader, leave=True)

    for idx, batch in enumerate(loop):
        x, y = Variable(batch[0]).to(config.DEVICE), Variable(batch[1]).to(config.DEVICE)
        # x, y = x.to(config.DEVICE), y.to(config.DEVICE)
        if not(os.path.isdir(folder)): os.mkdir(folder)
        with torch.no_grad():
            y_fake = gen(x)
            y_fake = y_fake * 0.5 + 0.5  # remove normalization#
            save_image(y_fake, folder + f"/{idx}_y_gen.png")
            save_image(x * 0.5 + 0.5, folder + f"/{idx}_input.png")
            save_image(y * 0.5 + 0.5, folder + f"/{idx}_label.png")

def save_checkpoint(model, optimizer, epoch, config, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    checkpoint = {
        "state_dict": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, config.SAVE_FOLDER+"/" + "epoch_"+str(epoch)+filename)
    torch.save(checkpoint, config.SAVE_FOLDER+"/" + filename)


def load_checkpoint(checkpoint_file, model, optimizer, lr, config):
    print("=> Loading checkpoint")
    checkpoint = torch.load(config.SAVE_FOLDER+"/" +checkpoint_file, map_location=config.DEVICE)
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])

    for param_group in optimizer.param_groups:
        param_group["lr"] = lr

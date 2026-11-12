"""
Step 3: segmentation. Which pixels belong to the pet?

What is NEW compared to 02_detection.py
---------------------------------------
* Dense prediction: one output per pixel (pet / not pet) instead of one label
  or one box per image. The targets are the trimaps of the dataset
  (pixel value 1 = pet).
* The U-Net architecture (models.py): an encoder that halves the resolution
  five times (224 -> 7) to see context, a decoder that upsamples back to 224,
  and skip connections that pass the fine details of each encoder level to the
  decoder level of the same resolution.
* The BCE + Dice loss and the Dice coefficient metric, which measure the
  overlap of whole masks and are not swamped by the many background pixels.
"""

import argparse

import torch

from models import get_segmentation_model
from utils import add_common_args, bce_dice_loss, dice_coeff, get_device, \
    get_loaders, save_predictions, set_seed, train


def loss_fn(logits, y):
    return bce_dice_loss(y, logits)


@torch.no_grad()
def evaluate(model, loader, args):
    # stream over the test set: keeping every predicted mask in memory at
    # once would need several GB
    model.eval()
    total, n = 0.0, 0
    for j, (x, y) in enumerate(loader):
        if args.max_eval_batches is not None and j >= args.max_eval_batches:
            break
        prob = torch.sigmoid(model(x.to(args.device))).cpu()
        total += dice_coeff(y, prob, hard=True).item() * len(x)
        n += len(x)
    return f'Dice: {total / n:.4f}'


def main(args):
    set_seed(args.seed)
    args.device = get_device()
    train_loader, test_loader = get_loaders(args, 'segmentation')
    model = get_segmentation_model(args)
    print(f'U-Net parameters: '
          f'{sum(p.numel() for p in model.parameters()) / 1e6:.1f}M')
    train(model, loss_fn, evaluate, train_loader, test_loader, args,
          'models/segmentation_model.pth')
    save_predictions(model, test_loader, args, 'segmentation',
                     'outputs/segmentation.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pet segmentation with a U-Net')
    add_common_args(parser, batch_size=5, n_iters=10_000, lr=3e-5)
    main(parser.parse_args())

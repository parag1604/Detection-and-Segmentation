"""
Step 1: classification. Is this a cat or a dog?

What this file teaches
----------------------
* Transfer learning: an ImageNet-pretrained ResNet-18 with its last layer
  replaced by one output (the logit of "dog"), fine-tuned end to end.
* Binary classification with a single logit and binary cross-entropy.
* Data handling for the Oxford-IIIT Pet dataset (utils.py); the label comes
  from the file name (cat breeds are capitalised).
The weights saved here are reused by step 2.
"""

import argparse

import torch
import torch.nn.functional as F

from models import get_classification_model
from utils import add_common_args, get_device, get_loaders, predict_all, \
    save_predictions, set_seed, train


def loss_fn(logits, y):
    return F.binary_cross_entropy_with_logits(logits.squeeze(1), y)


def evaluate(model, loader, args):
    logits, y = predict_all(model, loader, args)
    acc = ((logits.squeeze(1) > 0).float() == y).float().mean().item()
    return f'accuracy: {acc:.4f}'


def main(args):
    set_seed(args.seed)
    args.device = get_device()
    train_loader, test_loader = get_loaders(args, 'classification')
    model = get_classification_model(args)
    train(model, loss_fn, evaluate, train_loader, test_loader, args,
          'models/classification_model.pth')
    save_predictions(model, test_loader, args, 'classification',
                     'outputs/classification.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Cat vs dog classification')
    add_common_args(parser, batch_size=32, n_iters=500, lr=1e-4)
    parser.add_argument('--no_pretrained', dest='pretrained',
                        action='store_false',
                        help='start from random weights instead of ImageNet')
    main(parser.parse_args())

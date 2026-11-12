"""
Step 2: localisation. Where is the pet's head?

What is NEW compared to 01_classification.py
--------------------------------------------
* The output is no longer a class but a box: 4 numbers (x1, y1, x2, y2), the
  corners of the head's bounding box as fractions of the image size. This is
  regression, trained with the mean squared error.
* Transfer from step 1: the network starts from the cat/dog classifier, whose
  features already know what a pet looks like; only the head is new.
* A new metric: intersection over union (IoU) between the predicted and the
  true box (1 = perfect overlap, 0 = disjoint).
Every image has exactly one box, so this is "classification + localisation";
detecting a variable number of objects (e.g. Faster R-CNN, YOLO) additionally
needs many candidate boxes and a way to match them to the objects.
"""

import argparse

import torch.nn.functional as F

from models import get_detection_model
from utils import add_common_args, box_iou, get_device, get_loaders, \
    predict_all, save_predictions, set_seed, train


def loss_fn(boxes, y):
    return F.mse_loss(boxes, y)


def evaluate(model, loader, args):
    boxes, y = predict_all(model, loader, args)
    iou = box_iou(boxes, y)
    return (f'mean IoU: {iou.mean().item():.4f}, '
            f'IoU > 0.5: {(iou > 0.5).float().mean().item():.4f}')


def main(args):
    set_seed(args.seed)
    args.device = get_device()
    train_loader, test_loader = get_loaders(args, 'detection')
    model = get_detection_model(args)
    train(model, loss_fn, evaluate, train_loader, test_loader, args,
          'models/detection_model.pth')
    save_predictions(model, test_loader, args, 'detection',
                     'outputs/detection.png')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Pet head localisation')
    add_common_args(parser, batch_size=32, n_iters=5000, lr=1e-4)
    parser.add_argument('--classification_ckpt',
                        default='models/classification_model.pth')
    parser.add_argument('--no_pretrained', dest='pretrained',
                        action='store_false',
                        help='no ImageNet weights if step 1 was not run')
    main(parser.parse_args())

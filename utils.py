"""
Helpers shared by the three scripts: the Oxford-IIIT Pet data, metrics,
losses, a training loop and a plot of the predictions.

Expected layout (see README.md, "Data"):
    data/images/<Breed_name>_<n>.jpg
    data/annotations/xmls/<Breed_name>_<n>.xml     (head bounding boxes)
    data/annotations/trimaps/<Breed_name>_<n>.png  (1 pet, 2 background, 3 border)
In this dataset the file names of CAT breeds start with a capital letter
(Abyssinian_1.jpg) and DOG breeds with a lower-case letter (beagle_1.jpg).
"""

import os
import random
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

IMG_SIZE = 224
IMAGENET_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_device() -> torch.device:
    # prefer an NVIDIA GPU, then Apple silicon, then the CPU
    if torch.cuda.is_available():
        return torch.device('cuda')
    if torch.backends.mps.is_available():
        return torch.device('mps')
    return torch.device('cpu')


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ---------------------------------------------------------------- data ----

def get_dataset_split(data_path, mode, seed, train_frac=0.8):
    """Random 80/20 split of the images that have the annotation `mode` needs."""
    img_dir = os.path.join(data_path, 'images')
    if not os.path.isdir(img_dir):
        raise FileNotFoundError(
            f'{img_dir} not found: download the Oxford-IIIT Pet dataset first '
            '(see README.md, "Data").')
    # the images folder also contains a few .mat files; keep only the jpgs
    names = sorted(n[:-4] for n in os.listdir(img_dir) if n.endswith('.jpg'))
    if mode == 'detection':
        # head boxes exist only for the "trainval" images
        xml_dir = os.path.join(data_path, 'annotations', 'xmls')
        names = [n for n in names
                 if os.path.exists(os.path.join(xml_dir, n + '.xml'))]
    names = np.array(names)
    idxs = np.random.RandomState(seed).permutation(len(names))
    n_train = int(len(names) * train_frac)
    return names[idxs[:n_train]], names[idxs[n_train:]]


def read_box(xml_path):
    """The first <bndbox> of a Pascal-VOC style xml: (xmin, ymin, xmax, ymax)."""
    box = ET.parse(xml_path).getroot().find('object').find('bndbox')
    return [float(box.find(k).text) for k in ('xmin', 'ymin', 'xmax', 'ymax')]


class PetDataset(torch.utils.data.Dataset):
    def __init__(self, data_path, names, mode):
        self.data_path, self.names, self.mode = data_path, names, mode

    def __len__(self):
        return len(self.names)

    def __getitem__(self, i):
        name = self.names[i]
        # some images are grayscale or RGBA: force 3 channels
        img = Image.open(os.path.join(
            self.data_path, 'images', name + '.jpg')).convert('RGB')
        w, h = img.size
        img = np.asarray(img.resize((IMG_SIZE, IMG_SIZE)), dtype=np.float32)
        # normalise with the ImageNet statistics the pretrained ResNet expects
        img = (img / 255.0 - IMAGENET_MEAN) / IMAGENET_STD
        x = torch.from_numpy(img).permute(2, 0, 1)  # (3, H, W)

        if self.mode == 'classification':
            y = torch.tensor(0.0 if name[0].isupper() else 1.0)  # cat 0, dog 1
        elif self.mode == 'detection':
            x1, y1, x2, y2 = read_box(os.path.join(
                self.data_path, 'annotations', 'xmls', name + '.xml'))
            # box corners as fractions of the image width / height, in [0, 1]
            y = torch.tensor([x1 / w, y1 / h, x2 / w, y2 / h],
                             dtype=torch.float32)
        elif self.mode == 'segmentation':
            trimap = Image.open(os.path.join(
                self.data_path, 'annotations', 'trimaps', name + '.png'))
            # NEAREST: interpolating label values would invent new labels
            trimap = np.array(trimap.resize((IMG_SIZE, IMG_SIZE),
                                            Image.NEAREST))
            y = torch.from_numpy((trimap == 1).astype(np.float32))[None]
        else:
            raise ValueError(f'Invalid mode: {self.mode}')
        return x, y


def get_loaders(args, mode):
    train_names, test_names = get_dataset_split(args.data_path, mode, args.seed)
    print(f'{len(train_names)} training and {len(test_names)} test images')
    train_loader = torch.utils.data.DataLoader(
        PetDataset(args.data_path, train_names, mode), args.batch_size,
        shuffle=True, num_workers=args.workers, drop_last=True)
    test_loader = torch.utils.data.DataLoader(
        PetDataset(args.data_path, test_names, mode), args.batch_size,
        shuffle=False, num_workers=args.workers)
    return train_loader, test_loader


def infinite(loader):
    while True:
        for batch in loader:
            yield batch


# ----------------------------------------------------- metrics, losses ----

def box_iou(pred, true):
    """IoU of matching boxes (N, 4) given as (x1, y1, x2, y2)."""
    ix1 = torch.max(pred[:, 0], true[:, 0])
    iy1 = torch.max(pred[:, 1], true[:, 1])
    ix2 = torch.min(pred[:, 2], true[:, 2])
    iy2 = torch.min(pred[:, 3], true[:, 3])
    inter = (ix2 - ix1).clamp(min=0) * (iy2 - iy1).clamp(min=0)
    area = lambda b: (b[:, 2] - b[:, 0]).clamp(min=0) * \
        (b[:, 3] - b[:, 1]).clamp(min=0)
    union = area(pred) + area(true) - inter
    return inter / union.clamp(min=1e-8)


def dice_coeff(y_true, y_prob, smooth=1.0, hard=True):
    """
    Dice = 2 |A n B| / (|A| + |B|), per image, averaged over the batch.
    hard=True thresholds the prediction at 0.5 (the metric); hard=False uses
    the probabilities themselves, which keeps it differentiable (the loss).
    The smooth term avoids 0/0 on empty masks.
    """
    y_pred = (y_prob > 0.5).float() if hard else y_prob
    dims = (1, 2, 3)
    inter = (y_true * y_pred).sum(dims)
    total = y_true.sum(dims) + y_pred.sum(dims)
    return ((2.0 * inter + smooth) / (total + smooth)).mean()


def bce_dice_loss(y_true, logits):
    """
    Binary cross-entropy + soft dice loss.
    BCE is a per-pixel loss and is dominated by the (large) background;
    dice measures the overlap of the whole mask and does not care about class
    imbalance. Together they train faster and more stably than either alone.
    """
    bce = F.binary_cross_entropy_with_logits(logits, y_true)
    return bce + 1.0 - dice_coeff(y_true, torch.sigmoid(logits), hard=False)


# ------------------------------------------------------ training loop ----

def train(model, loss_fn, evaluate_fn, train_loader, test_loader, args, ckpt):
    """Iteration-based training; evaluates and saves every args.eval_every."""
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)
    batches = infinite(train_loader)
    os.makedirs(os.path.dirname(ckpt) or '.', exist_ok=True)
    model.train()
    for i in range(args.n_iters + 1):
        x, y = next(batches)
        x, y = x.to(args.device), y.to(args.device)
        loss = loss_fn(model(x), y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if i % args.print_every == 0:
            print(f'Iteration: {i}, Loss: {loss.item():.4f}')
        if i % args.eval_every == 0 or i == args.n_iters:
            score = evaluate_fn(model, test_loader, args)
            print(f'Iteration: {i}, Test {score}')
            torch.save(model.state_dict(), ckpt)
            model.train()


@torch.no_grad()
def predict_all(model, loader, args):
    """Model outputs and targets for (at most args.max_eval_batches of) loader."""
    model.eval()
    outs, ys = [], []
    for j, (x, y) in enumerate(loader):
        if args.max_eval_batches is not None and j >= args.max_eval_batches:
            break
        outs.append(model(x.to(args.device)).cpu())
        ys.append(y)
    return torch.cat(outs), torch.cat(ys)


def denormalize(x):
    img = x.permute(1, 2, 0).cpu().numpy() * IMAGENET_STD + IMAGENET_MEAN
    return np.clip(img, 0, 1)


@torch.no_grad()
def save_predictions(model, loader, args, mode, path, n=4):
    """Plot a few test images with the model's predictions."""
    model.eval()
    x, y = next(iter(loader))
    x, y = x[:n], y[:n]
    out = model(x.to(args.device)).cpu()
    rows = 3 if mode == 'segmentation' else 1  # image, true mask, predicted
    fig, axes = plt.subplots(rows, len(x), figsize=(3 * len(x), 3 * rows),
                             squeeze=False)
    for j in range(len(x)):
        ax = axes[0, j]
        ax.imshow(denormalize(x[j]))
        ax.axis('off')
        if mode == 'classification':
            p = torch.sigmoid(out[j]).item()
            ax.set_title(f'{"Dog" if p > 0.5 else "Cat"} (p_dog = {p:.2f})')
        elif mode == 'detection':
            for box, color in [(y[j], 'lime'), (out[j], 'red')]:
                x1, y1, x2, y2 = (box * IMG_SIZE).tolist()
                ax.add_patch(patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1, linewidth=2,
                    edgecolor=color, facecolor='none'))
            ax.set_title('green: truth, red: prediction')
        else:
            axes[1, j].imshow(y[j, 0], cmap='gray')
            axes[1, j].set_title('true mask')
            axes[2, j].imshow(torch.sigmoid(out[j, 0]), cmap='gray')
            axes[2, j].set_title('predicted mask')
            axes[1, j].axis('off')
            axes[2, j].axis('off')
    plt.tight_layout()
    os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
    plt.savefig(path, dpi=100)
    plt.close(fig)
    print(f'Saved example predictions to {path}')


def add_common_args(parser, batch_size, n_iters, lr):
    parser.add_argument('--data_path', default='data')
    parser.add_argument('--seed', type=int, default=2022)
    parser.add_argument('--batch_size', type=int, default=batch_size)
    parser.add_argument('--n_iters', type=int, default=n_iters)
    parser.add_argument('--lr', type=float, default=lr)
    parser.add_argument('--workers', type=int, default=4)
    parser.add_argument('--print_every', type=int, default=10)
    parser.add_argument('--eval_every', type=int, default=250)
    parser.add_argument('--max_eval_batches', type=int, default=None,
                        help='evaluate on at most this many test batches')
    return parser

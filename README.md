# Detection and Segmentation

Code for three computer-vision tasks of increasing detail on the Oxford-IIIT Pet dataset: classification (cat or dog?), localisation (where is the pet's head?) and segmentation (which pixels are the pet?).

---

## Theory

Please note that the theory corresponding to the codes is hosted on this [page](https://parag1604.github.io/notes/Machine-Learning/Convolutional-Neural-Networks) (Convolutional Neural Networks lecture, sections "Convolutional neural networks", "Object localization and detection" and "Segmentation")

## Python Environment Setup

Use python >= 3.10 and install the requirements:

```bash
pip install -r requirements.txt
```

- PyTorch - models and training (install the hardware accelerated build if possible; the code picks CUDA, then Apple MPS, then CPU)
- Torchvision - the ResNet-18 model and its ImageNet weights (downloaded automatically by torchvision on first use, about 45 MB; pass `--no_pretrained` to start from random weights instead)
- Numpy, Pillow - image loading
- Matplotlib - plots of the predictions

## Data

The Oxford-IIIT Pet dataset (https://www.robots.ox.ac.uk/~vgg/data/pets/): 7,349 photos of 37 cat and dog breeds, with a head bounding box for the "trainval" images and a pixel-level trimap (pet / background / border) for every image. Download the images (`images.tar.gz`, 790 MB) and the ground truth (`annotations.tar.gz`, 19 MB) and extract both into `./data`:

```bash
mkdir -p data && cd data
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/images.tar.gz
curl -O https://www.robots.ox.ac.uk/~vgg/data/pets/data/annotations.tar.gz
tar xzf images.tar.gz && tar xzf annotations.tar.gz
cd ..
```

The code expects:

```
data/
    images/          Abyssinian_1.jpg, beagle_1.jpg, ...
    annotations/
        xmls/        head bounding boxes (Pascal VOC format, trainval images only)
        trimaps/     1 = pet, 2 = background, 3 = border
        list.txt, trainval.txt, test.txt, README
```

The labels for step 1 come from the file names: cat breeds are capitalised (`Abyssinian_1.jpg`), dog breeds are not (`beagle_1.jpg`). Each script makes a random 80/20 train/test split (fixed by `--seed`) of the images that have the annotation it needs.

## Running

Every script accepts `--batch_size`, `--n_iters`, `--lr`, `--eval_every`, `--max_eval_batches` and `--workers`. The test metric is printed every `--eval_every` iterations, the weights are saved to `models/`, and at the end a figure with a few test predictions is written to `outputs/`. A quick check that a script runs:

```bash
python 01_classification.py --n_iters 20 --eval_every 10 --max_eval_batches 2
```

### 1. Classification

```bash
python 01_classification.py
```

An ImageNet-pretrained ResNet-18 with its last layer replaced by a single logit ("dog"), fine-tuned with binary cross-entropy for 500 iterations (batch 32, Adam, lr 1e-4). Prints the test accuracy; writes `models/classification_model.pth` and `outputs/classification.png`.

### 2. Localisation

```bash
python 02_detection.py
```

New: the output is a box, the four corners of the pet's head as fractions of the image size, regressed with the mean squared error, and the metric is the intersection over union (IoU) of the predicted and true boxes. New too: transfer from step 1, the network starts from the cat/dog classifier and only the last layer is replaced (run step 1 first). Default: 5000 iterations. Every image has exactly one box, so this is classification plus localisation; detectors for many objects (Faster R-CNN, YOLO) build on the same box regression. Writes `outputs/detection.png` (green: truth, red: prediction).

### 3. Segmentation

```bash
python 03_segmentation.py
```

New: a prediction for every pixel, by a U-Net trained from scratch (an encoder down to 7x7, a decoder back up to 224x224, and skip connections between levels of the same resolution), with the BCE + Dice loss and the Dice coefficient as the metric. Default: 10,000 iterations with batch 5 and lr 3e-5. Writes `outputs/segmentation.png` (image, true mask, predicted mask).

# Model Notes – CycleGAN for Monet Style Transfer

## Overview

This project uses the CycleGAN architecture to perform image-to-image translation between two domains:

- Real landscape photographs
- Paintings in the style of Claude Monet

Unlike supervised image translation methods, CycleGAN does not require paired datasets. Instead, it learns mappings between domains using cycle consistency.

---

## CycleGAN Architecture

The architecture consists of four neural networks:

- Generator G : Photo → Monet
- Generator F : Monet → Photo
- Discriminator Dx : distinguishes real photos from generated ones
- Discriminator Dy : distinguishes real Monet paintings from generated ones

Generators try to fool the discriminators, while discriminators try to distinguish real images from generated ones.

This adversarial training enables realistic style transfer.

---

## Loss Functions

CycleGAN training relies on three main losses:

### 1. Adversarial Loss

Encourages generators to produce realistic images that fool the discriminator.

### 2. Cycle Consistency Loss

Ensures that translating an image to the other domain and back reconstructs the original image.

Example:

Photo → Monet → Photo ≈ Original Photo

### 3. Identity Loss

Encourages generators to preserve color composition when possible.

---

## Dataset

The model is trained on the Monet2Photo dataset.

Two image domains are used:

- Monet paintings
- Landscape photographs

Images are typically resized to 256×256 for training.

---

## Training Details

Typical configuration:

- Optimizer: Adam
- Learning rate: 0.0002
- Batch size: 1
- Epochs: 100–200

Training can be accelerated using GPUs or TPUs.

---

## Observed Results

The trained model produces images that mimic:

- Impressionist brush strokes
- Monet-style color palettes
- Soft lighting typical of Monet landscapes

However, limitations remain:

- Some loss of fine details
- Slight blur in generated images

---

## Possible Improvements

Future improvements could include:

- Training for more epochs
- Using higher resolution images
- Applying attention-based GAN architectures
- Adding perceptual loss

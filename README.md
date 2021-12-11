# pix2pix-experiments
Implementation of pix2pix from scratch and experiments on various settings on the research paper. Deep Generative Models (CS 274E) course project


#### Train
```
<!-- Training (Note that there are no scripts for downloading datasets mentioned in the paper. The dataset paths and model options can be configured in configuration.py) -->

python train.py --dataset maps --n_layers 3


```

#### Test/Val

```
<!-- Running on validation/test set -->
python evaluate.py 
```

Inspired from Aladdin Persson gan tutorials


### Abstract

In this project, we re-implement Isola et al.’s work on "Image-to-Image Translation
with Conditional Adversarial Networks" which aims to learn mappings from class
of input images to class of output images. The previous works on image-to-image
translation typically focus on deterministic mappings, and use L2 reconstruction
loss functions that can induce blurriness. These approaches are problematic since
the realism of the results depend heavily on proper loss function design and there
is a lack of variability in image generation. In this approach, conditional generative
adversarial networks are used to learn the image mappings using the L1 reconstruc-
tion loss. Training of generative adversarial networks are generally unstable and
may lead to mode collapse which we hope to overcome through the addition of tai-
lored loss functions on the GAN’s input random variable. We will demonstrate that
this approach is general and can be utilized effectively for many different datasets,
such as at synthesizing photos from label maps, reconstructing facades from man-
ually annotated images among other tasks. The ideal consequence of the project
would be to gain a deeper understanding of training robust GAN networks, and pro-
vide methods that will improve both visual quality and realism in image translation.


#### Datasets

We test the above-mentioned architecture on both facades (400 images from CMP Facades) and
maps (1096 training images scraped from Google Maps) datasets for both upstream and downstream
data generation tasks. For the Image colorization task, we sampled 5,000 images of different object
categories for training from the COCO image dataset.


#### Authors

Rajasekhar Reddy Mekala
rmekala@uci.edu

Agniraj Baikani
abaikani@uci.edu


Andrew Jiang
andrewj3@uci.edu



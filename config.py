import random

import numpy as np
import torch
from torch.backends import cudnn

# Random seed to maintain reproducible results
random.seed(0)
torch.manual_seed(0)
np.random.seed(0)
# Use GPU for training by default
device = torch.device("cuda", 0)
# Turning on when the image size does not change during training can speed up training
cudnn.benchmark = True
# Image magnification factor
upscale_factor = 2
# Current configuration parameter method
mode = "test"
# Experiment name, easy to save weights and log files
exp_name = "SRCNN_x2"

if mode == "train":
    # Dataset
    train_image_dir = "/home/nghiapd/Code/DAT_model/T91"
    test_lr_image_dir = "/home/nghiapd/Code/DAT_model/Set5/image_SRF_2"
    test_hr_image_dir = "/home/nghiapd/Code/DAT_model/Set5/image_SRF_2"

    image_size = 32
    batch_size = 16
    num_workers = 4

    # Incremental training and migration training
    resume = ""

    # Total number of epochs
    epochs = 50800

    # SGD optimizer parameter
    model_lr = 1e-4
    model_momentum = 0.9
    model_weight_decay = 1e-4
    model_nesterov = False

    # How many iterations to print the training result
    print_frequency = 200

if mode == "test":
    # Test data address
    lr_dir = f"/home/nghiapd/Code/DAT_model/Set5/image_SRF_2"
    sr_dir = f"/home/nghiapd/Code/DAT_model/results/test/{exp_name}"
    hr_dir = f"/home/nghiapd/Code/DAT_model/Set5/image_SRF_2"

    model_path = "/home/nghiapd/Code/DAT_model/results/SRCNN_x2/best.pth.tar"
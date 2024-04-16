import json
import os
import sys
import argparse

import numpy
import torch
from torch import nn
from torchvision import transforms as pth_transforms
from timm.models import create_model

from PIL import Image, ImageFile

import utils
import modeling_vqkd 

if __name__ == "__main__":
    model_path = "vqkd_encoder_base_decoder_1x768x12_clip"
    pretrain_weights = "pretrain/vqkd_encoder_base_decoder.pth"
    model = create_model(
        model_path,
        pretrained=True,
        pretrained_weight=pretrain_weights,
        as_tokenzer=True,
    ).eval()
    print(model.quantize.embedding.weight)
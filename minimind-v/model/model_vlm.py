import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from .model import *
from .VLMConfig import VLMConfig

warnings.filterwarnings("ignore")


class VisionProj(nn.Module):
    def __init__(self, ve_dim=768, lm_dim=512):
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.vision_proj = nn.Sequential(nn.Linear(self.ve_dim, self.lm_dim))

    def forward(self, image_encoders):
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj

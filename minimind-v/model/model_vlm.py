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


class MiniMindVLM(MiniMindLM):
    config_class = VLMConfig

    def __init__(self, params, VLMConfig):
        super().__init__(params)
        if not params:
            params = VLMConfig()
        self.params = params
        self.vision_encoder, self.processor = self.__class__.get_vision_encoder(
            params
        )
        self.vision_proj = VisionProj(lm_dim=params.dim)

    @staticmethod
    def get_vision_encoder(self, params, model_path):
        """
        Get the vision encoder and processor from HuggingFace's CLIP model.
        """
        model = CLIPModel.from_pretrained(model_path)
        processos = CLIPModel.from_pretrained(model_path)
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processos

    def image2tensor(image, processor):
        if image.mode in ["Rage", "LA"]:
            image = image.convert("RGB")
        inputs = processor(images=image)

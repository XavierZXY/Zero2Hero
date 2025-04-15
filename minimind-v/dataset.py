import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .model_vlm import MiniMindVLM

os.environ["TOKENIZERS_PARALLELISM"] = "false"

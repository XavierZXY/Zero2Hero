from .VLMConfig import VLMConfig
from .model import *
from typing import Optional, Tuple, List
from torch import nn
import warnings
from transformers import CLIPProcessor, CLIPModel
import torch

warnings.filterwarnings('ignore')
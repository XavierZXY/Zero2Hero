import os
import platform
import argparse
import time
import math
import warnings
import json

import pandas as pd
import torch
import torch.nn.functional as F
import torch.distributed as dist
from contextlib import nullcontext

from torch import optim, nn
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer, AutoModel
from model.model_vlm import MiniMindVLM
from model.VLMConfig import VLMConfig
from model.dataset import VLMDataset

warnings.filterwarnings('ignore')
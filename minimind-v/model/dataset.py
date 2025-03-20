import json
import os

import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset

from .model_vlm import MiniMindVLM


class VLMDataset(Dataset):
    def __init__(
        self,
        jsonl_path,
        images_path,
        tokenizer,
        preprocess=None,
        max_length=128,
        image_special_token="@" * 196,
    ):
        """_summary_

        Args:
            jsonl_path (_type_): _description_
            images_path (_type_): _description_
            tokenizer (_type_): _description_
            preprocess (_type_, optional): _description_. Defaults to None.
            max_length (int, optional): _description_. Defaults to 128.
            image_special_token (_type_, optional): _description_. Defaults to '@'*196.
        """
        super().__init__()
        self.sample = self.load_data(jsonl_path)
        self.images_path = images_path
        self.max_length = max_length
        self.preprocess = preprocess
        self.image_token = image_special_token
        self.tokenizer = tokenizer
        self.bos_id = tokenizer(
            "<s>assistant\n", add_special_tokens=False
        ).input_ids
        self.eos_id = tokenizer("</s>\n", add_special_tokens=False).input_ids

    def __len__(self):
        return len(self.sample)

    def load_data(self, jsonl_path):
        samples = []
        with open(path, "r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, 1):
                data = json.loads(line.strip())
                samples.append(data)
        return samples

    def _create_chat_prompt(self, conversations):
        messages = []
        for i, turn in enumerate(conversations):
            role = "user" if i % 2 == 0 else "assistant"
            messages.append(
                {
                    "role": role,
                    "content": turn["content"].replace(
                        "<image>", self.image_token
                    ),
                }
            )

        return self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )

    def _generate_loss_mask(self, input_ids):
        loss_mask = [0] * len(input_ids)
        i = 0
        while i < len(input_ids):
            if input_ids[i : i + len(self.bos_id)] == self.bos_id:
                start = i + len(self.bos_id)
                end = start
                while end < len(input_ids):
                    if input_ids[end : end + len(self.eos_id)] == self.eos_id:
                        break
                    end += 1
                for j in range(
                    start + 1, min(end + len(self.eos_id) + 1, self.max_length)
                ):
                    loss_mask[j] = 1
                i = (
                    end + len(self.eos_id)
                    if end < len(input_ids)
                    else len(input_ids)
                )
            else:
                i += 1
        return loss_mask

    def __getitem__(self, index: int):
        sample = self.sample[index]
        image_paths = sample["image"]
        prompt = self._create_chat_prompt(sample["conversation"])
        input_ids = self.tokenizer(prompt).input_ids[: self.max_length]
        input_ids += [self.tokenizer.pad_token_id] * (
            self.max_length - len(input_ids)
        )
        loss_mask = self._generate_loss_mask(input_ids)

        X = torch.tensor(input_ids[:-1], dtype=torch.long)
        Y = torch.tensor(input_ids[1:], dtype=torch.long)
        loss_mask = torch.tensor(loss_mask[1:], dtype=torch.long)

        image_tensors = []
        for image_name in image_paths.split(","):
            image_name = image_name.strip()
            image = Image.open(f"{self.images_path}/{image_name}")
            image_tensor = MiniMindVLM.image2tensor(image, self.preprocess)
            image_tensors.append(image_tensor)
        image_tensors = torch.stack(image_tensors)

        return X, Y, loss_mask, image_tensors

"""
Vision-Language Model implementation for MiniMind.
This module integrates vision capabilities with the base language model,
allowing the model to process both text and images.
"""

import warnings
from typing import List, Optional, Tuple

import torch
from torch import nn
from transformers import CLIPModel, CLIPProcessor

from .model import *
from .VLMConfig import VLMConfig

warnings.filterwarnings("ignore")


class VisionProj(nn.Module):
    """
    Vision projection module that maps vision encoder dimensions to language model dimensions.

    This module projects the output from the vision encoder (typically CLIP) to match
    the embedding dimensions of the language model, enabling multi-modal capabilities.

    Args:
        ve_dim (int, optional): Dimension of vision encoder output. Defaults to 768.
        lm_dim (int, optional): Dimension of language model. Defaults to 512.
    """

    def __init__(self, ve_dim=768, lm_dim=512):
        super().__init__()
        self.ve_dim = ve_dim
        self.lm_dim = lm_dim
        self.vision_proj = nn.Sequential(nn.Linear(self.ve_dim, self.lm_dim))

    def forward(self, image_encoders):
        """
        Project vision encoder outputs to language model dimension space.

        Args:
            image_encoders (torch.Tensor): Output from vision encoder

        Returns:
            torch.Tensor: Projected image representations matching LM dimensions
        """
        vision_proj = self.vision_proj(image_encoders)
        return vision_proj


class MiniMindVLM(MiniMindLM):
    """
    Vision-Language Model extending the base MiniMind language model.

    This class adds vision capabilities to the base language model by incorporating
    a vision encoder (CLIP) and mechanisms to process and integrate visual information.

    Args:
        params: Model parameters
        VLMConfig: Configuration for the Vision-Language model
    """

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

        Args:
            self: Instance of the class
            params: Model parameters
            model_path (str): Path to the CLIP model

        Returns:
            tuple: (vision_model, processor) - The CLIP model and its processor
        """
        model = CLIPModel.from_pretrained(model_path)
        processos = CLIPModel.from_pretrained(model_path)
        # Freeze the vision encoder parameters to prevent fine-tuning
        for param in model.parameters():
            param.requires_grad = False
        return model.eval(), processos

    @staticmethod
    def image2tensor(image, processor):
        """
        Convert an image to tensor format using the CLIP processor.

        Args:
            image: PIL image to convert
            processor: CLIP processor for image preprocessing

        Returns:
            torch.Tensor: Processed image tensor
        """
        # Convert RGBA or LA images to RGB for compatibility
        if image.mode in ["RGBA", "LA"]:
            image = image.convert("RGB")
        inputs = processor(
            image,
            return_tensors="pt",
        )["pixel_values"]
        return inputs

    @staticmethod
    def get_image_embeddings(image_tensors, vision_model):
        """
        Extract embeddings from images using the vision model.

        Args:
            image_tensors (torch.Tensor): Processed image tensors
            vision_model: CLIP vision model

        Returns:
            torch.Tensor: Image embeddings from the vision model
        """
        with torch.no_grad():
            outputs = vision_model.vision_model(pixel_values=image_tensors)
        # Extract embeddings, skipping the CLS token [0]
        img_embedding = outputs.last_hidden_state[:, 1:, :].squeeze()
        return img_embedding

    def count_vision_proj(
        self,
        tokens,
        h,
        vision_tensors=None,
        seqlen=512,
    ):
        """
        Replace image token sequences with projected image embeddings.

        This method identifies special image tokens in the input sequence and
        replaces them with the corresponding processed image embeddings.

        Args:
            tokens (torch.Tensor): Input token IDs
            h (torch.Tensor): Token embeddings
            vision_tensors (torch.Tensor, optional): Image embeddings
            seqlen (int, optional): Maximum sequence length. Defaults to 512.

        Returns:
            torch.Tensor: Modified embedding sequence with image embeddings inserted
        """

        def find_indices(tokens, image_ids):
            """
            Find the indices of image tokens in the input sequence.

            Args:
                tokens (torch.Tensor): Input token IDs
                image_ids (list): IDs representing image tokens

            Returns:
                dict or None: Mapping of batch indices to list of (start, end) indices
                              of image token sequences, or None if no matches found
            """
            image_ids_tensor = torch.tensor(image_ids).to(tokens.device)
            len_image_ids = len(image_ids)
            if len_image_ids > tokens.size(1):
                return None
            # Create sliding windows to find image token sequences
            tokens_view = tokens.unfold(1, len_image_ids, 1)
            matches = (tokens_view == image_ids_tensor).all(dim=2)
            return {
                batch_idx: [
                    (idx.item(), idx.item() + len_image_ids - 1)
                    for idx in matches[batch_idx].nonzero(as_tuple=True)[0]
                ]
                for batch_idx in range(tokens.size(0))
                if matches[batch_idx].any()
            } or None

        # Find where image tokens are located in the input
        image_indices = find_indices(tokens, self.params.image_ids)

        # Replace image tokens with vision embeddings if available
        if vision_tensors is not None and image_indices:
            vision_proj = self.vision_proj(vision_tensors)
            if len(vision_proj.shape) == 3:
                vision_proj = vision_proj.unsqueeze(0)
            new_h = []
            for i in range(h.size(0)):
                if i in image_indices:
                    h_i = h[i]
                    img_idx = 0
                    for start_idx, end_idx in image_indices[i]:
                        if img_idx < vision_proj.size(1):
                            # Replace the token embeddings with vision embeddings
                            h_i = torch.cat(
                                (
                                    h_i[:start_idx],
                                    vision_proj[i][img_idx],
                                    h_i[end_idx + 1 :],
                                ),
                                dim=0,
                            )[:seqlen]
                            img_idx += 1
                    new_h.append(h_i)
                else:
                    new_h.append(h[i])
            return torch.stack(new_h, dim=0)
        return h

    def forward(
        self,
        input_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[
            List[Tuple[torch.Tensor, torch.Tensor]]
        ] = None,
        use_cache: bool = False,
        **args,
    ):
        """
        Forward pass of the Vision-Language Model.

        Processes both text and image inputs, integrating vision features
        with language representation.

        Args:
            input_ids (torch.Tensor, optional): Input token IDs
            past_key_values (List[Tuple[torch.Tensor, torch.Tensor]], optional):
                Cached key values for faster inference
            use_cache (bool, optional): Whether to use cached key values. Defaults to False.
            **args: Additional arguments including image data

        Returns:
            dict: Model outputs including logits, aux_loss, and past_key_values
        """
        start_pos = args.get("start_pos", 0)
        pixel_tensors = args.get("pixel_tensors", None)
        h = self.tok_embeddings(input_ids)

        # Process image inputs if available and we're at the start of the sequence
        if pixel_tensors is not None and start_pos == 0:
            if len(pixel_tensors.shape) == 6:
                pixel_tensors = pixel_tensors.squeeze(2)
            bs, num, c, im_h, im_w = pixel_tensors.shape
            # Choose stacking dimension based on batch size
            stack_dim = 1 if bs > 1 else 0
            # Process each image in the batch
            vision_tensors = torch.stack(
                [
                    MiniMindVLM.get_image_embeddings(
                        pixel_tensors[:, i, :, :, :], self.vision_encoder
                    )
                    for i in range(num)
                ],
                dim=stack_dim,
            )
            # Replace image tokens with vision embeddings
            h = self.count_vision_proj(
                tokens=input_ids,
                h=h,
                vision_tensors=vision_tensors,
                seqlen=input_ids.shape[1],
            )

        # Get positional embeddings for the current sequence segment
        pos_cis = self.pos_cis[start_pos : start_pos + input_ids.shape[1]]
        past_kvs = []
        # Process through transformer layers
        for la, layer in enumerate(self.layers):
            h, past_kv = layer(
                h,
                pos_cis,
                past_key_value=past_key_values[la] if past_key_values else None,
                use_cache=use_cache,
            )
            past_kvs.append(past_kv)

        # Generate logits from final hidden states
        logits = self.output(self.norm(h))
        # Calculate auxiliary loss for MoE layers if present
        aux_loss = sum(
            l.feed_forward.aux_loss
            for l in self.layers
            if isinstance(l.feed_forward, MOEFeedForward)
        )

        # Prepare output dictionary
        self.OUT.__setitem__("logits", logits)
        self.OUT.__setitem__("aux_loss", aux_loss)
        self.OUT.__setitem__("past_key_values", past_kvs)
        return self.OUT

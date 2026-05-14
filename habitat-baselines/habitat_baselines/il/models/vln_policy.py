#!/usr/bin/env python3

# Copyright (c) Meta Platforms, Inc. and affiliates.
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import hashlib
import re
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


ACTION_NAMES: Tuple[str, ...] = (
    "stop",
    "move_forward",
    "turn_left",
    "turn_right",
)
START_PREV_ACTION = 0


class HashingTokenizer:
    r"""Tiny tokenizer used for tests and offline smoke runs.

    The production config uses a HuggingFace tokenizer. This fallback keeps the
    policy class usable when a pretrained text model is intentionally disabled.
    """

    def __init__(self, vocab_size: int = 30522, max_length: int = 80) -> None:
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.pad_token_id = 0
        self.unk_token_id = 1
        self._pattern = re.compile(r"[A-Za-z0-9']+|[^\sA-Za-z0-9]")

    def _token_to_id(self, token: str) -> int:
        digest = hashlib.sha1(token.encode("utf8")).hexdigest()
        return int(digest, 16) % (self.vocab_size - 2) + 2

    def __call__(
        self,
        texts: Sequence[str],
        padding: bool = True,
        truncation: bool = True,
        max_length: Optional[int] = None,
        return_tensors: str = "pt",
    ) -> Dict[str, Tensor]:
        if return_tensors != "pt":
            raise ValueError("HashingTokenizer only supports return_tensors='pt'")

        max_len = max_length or self.max_length
        token_ids: List[List[int]] = []
        for text in texts:
            ids = [
                self._token_to_id(tok.lower())
                for tok in self._pattern.findall(text)
            ]
            if truncation:
                ids = ids[:max_len]
            token_ids.append(ids or [self.unk_token_id])

        if padding:
            padded_len = max(len(ids) for ids in token_ids)
            padded_len = min(padded_len, max_len)
        else:
            padded_len = max_len

        input_ids = torch.full(
            (len(token_ids), padded_len), self.pad_token_id, dtype=torch.long
        )
        attention_mask = torch.zeros_like(input_ids)
        for row, ids in enumerate(token_ids):
            ids = ids[:padded_len]
            input_ids[row, : len(ids)] = torch.tensor(ids, dtype=torch.long)
            attention_mask[row, : len(ids)] = 1

        return {"input_ids": input_ids, "attention_mask": attention_mask}


class PretrainedVisualEncoder(nn.Module):
    r"""ImageNet-pretrained torchvision encoder projected to policy features."""

    def __init__(
        self,
        backbone: str = "resnet18",
        output_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        image_size: int = 224,
    ) -> None:
        super().__init__()

        import torchvision.models as tv_models

        try:
            weights = (
                tv_models.get_model_weights(backbone).DEFAULT
                if pretrained
                else None
            )
            model = tv_models.get_model(backbone, weights=weights)
        except AttributeError:
            model_fn = getattr(tv_models, backbone)
            model = model_fn(pretrained=pretrained)

        if not hasattr(model, "fc"):
            raise ValueError(
                f"Unsupported visual backbone '{backbone}'. "
                "Use a torchvision ResNet-style model with an fc layer."
            )

        feature_dim = model.fc.in_features
        model.fc = nn.Identity()
        self.backbone = model
        self.proj = nn.Sequential(
            nn.Linear(feature_dim, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )
        self.image_size = image_size
        self.freeze_backbone = freeze

        self.register_buffer(
            "image_mean",
            torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1),
        )
        self.register_buffer(
            "image_std",
            torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1),
        )

        if freeze:
            for param in self.backbone.parameters():
                param.requires_grad = False

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_backbone:
            self.backbone.eval()
        return self

    def _preprocess(self, rgb: Tensor) -> Tensor:
        if rgb.dim() != 4:
            raise ValueError(
                "Visual encoder expects rgb with shape (B,C,H,W) or "
                f"(B,H,W,C), got {tuple(rgb.shape)}"
            )

        if rgb.size(1) not in (1, 3) and rgb.size(-1) in (1, 3):
            rgb = rgb.permute(0, 3, 1, 2).contiguous()
        if rgb.size(1) == 1:
            rgb = rgb.repeat(1, 3, 1, 1)
        if rgb.size(1) > 3:
            rgb = rgb[:, :3]

        rgb = rgb.float()
        if rgb.numel() > 0 and rgb.max() > 2.0:
            rgb = rgb / 255.0

        if rgb.shape[-2:] != (self.image_size, self.image_size):
            rgb = F.interpolate(
                rgb,
                size=(self.image_size, self.image_size),
                mode="bilinear",
                align_corners=False,
            )
        return (rgb - self.image_mean) / self.image_std

    def forward(self, rgb: Tensor) -> Tensor:
        had_time_dim = rgb.dim() == 5
        if had_time_dim:
            batch, steps = rgb.shape[:2]
            rgb = rgb.reshape(batch * steps, *rgb.shape[2:])

        feats = self.backbone(self._preprocess(rgb))
        feats = self.proj(feats)

        if had_time_dim:
            feats = feats.view(batch, steps, -1)
        return feats


class TransformerTextEncoder(nn.Module):
    r"""Pretrained HuggingFace text encoder with pooled sentence features."""

    def __init__(
        self,
        model_name: str = "distilbert-base-uncased",
        output_dim: int = 256,
        pretrained: bool = True,
        freeze: bool = True,
        local_files_only: bool = False,
    ) -> None:
        super().__init__()

        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, local_files_only=local_files_only
        )
        if pretrained:
            self.encoder = AutoModel.from_pretrained(
                model_name, local_files_only=local_files_only
            )
        else:
            text_config = AutoConfig.from_pretrained(
                model_name, local_files_only=local_files_only
            )
            self.encoder = AutoModel.from_config(text_config)

        hidden_size = self.encoder.config.hidden_size
        self.proj = nn.Sequential(
            nn.Linear(hidden_size, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

        if freeze:
            for param in self.encoder.parameters():
                param.requires_grad = False
        self.freeze_encoder = freeze

    def train(self, mode: bool = True):
        super().train(mode)
        if self.freeze_encoder:
            self.encoder.eval()
        return self

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
            pooled = outputs.pooler_output
        else:
            token_feats = outputs.last_hidden_state
            mask = attention_mask.unsqueeze(-1).type_as(token_feats)
            pooled = (token_feats * mask).sum(dim=1) / mask.sum(
                dim=1
            ).clamp_min(1.0)
        return self.proj(pooled)


class TrainableTextEncoder(nn.Module):
    r"""GRU text encoder used when no pretrained transformer is requested."""

    def __init__(
        self,
        vocab_size: int = 30522,
        output_dim: int = 256,
        embedding_dim: int = 128,
        max_length: int = 80,
    ) -> None:
        super().__init__()
        self.tokenizer = HashingTokenizer(vocab_size=vocab_size, max_length=max_length)
        self.embedding = nn.Embedding(
            vocab_size, embedding_dim, padding_idx=self.tokenizer.pad_token_id
        )
        self.rnn = nn.GRU(embedding_dim, output_dim, batch_first=True)
        self.out = nn.Sequential(nn.LayerNorm(output_dim), nn.ReLU(inplace=True))

    def forward(self, input_ids: Tensor, attention_mask: Tensor) -> Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(input_ids.long())
        packed = pack_padded_sequence(
            embedded, lengths, batch_first=True, enforce_sorted=False
        )
        _, hidden = self.rnn(packed)
        return self.out(hidden[-1])


class GatedMultimodalFusion(nn.Module):
    r"""Gated fusion between visual observations and the instruction vector."""

    def __init__(
        self,
        visual_dim: int,
        text_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.visual_proj = nn.Linear(visual_dim, hidden_dim)
        self.text_proj = nn.Linear(text_dim, hidden_dim)
        self.gate = nn.Sequential(nn.Linear(2 * hidden_dim, hidden_dim), nn.Sigmoid())
        self.out = nn.Sequential(nn.LayerNorm(hidden_dim), nn.Dropout(dropout))

    def forward(self, visual_feats: Tensor, text_feats: Tensor) -> Tensor:
        if visual_feats.dim() == 3:
            text_feats = text_feats.unsqueeze(1).expand(
                -1, visual_feats.size(1), -1
            )

        visual = torch.tanh(self.visual_proj(visual_feats))
        text = torch.tanh(self.text_proj(text_feats))
        gate = self.gate(torch.cat([visual, text], dim=-1))
        return self.out(gate * visual + (1.0 - gate) * text)


class VLNPolicyNet(nn.Module):
    r"""Vision-language navigation policy for imitation learning.

    Inputs are RGB observations, tokenized natural-language instructions, and
    previous actions. The model predicts Habitat's discrete VLN actions at each
    timestep: stop, move_forward, turn_left, turn_right.
    """

    def __init__(
        self,
        num_actions: int = len(ACTION_NAMES),
        visual_backbone: str = "resnet18",
        visual_pretrained: bool = True,
        freeze_visual_encoder: bool = True,
        text_model_name: str = "distilbert-base-uncased",
        text_pretrained: bool = True,
        freeze_text_encoder: bool = True,
        transformers_local_files_only: bool = False,
        hidden_size: int = 256,
        recurrent_hidden_size: int = 512,
        prev_action_embedding_dim: int = 32,
        dropout: float = 0.1,
        image_size: int = 224,
        simple_text_vocab_size: int = 30522,
        max_instruction_length: int = 80,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.max_instruction_length = max_instruction_length

        self.visual_encoder = PretrainedVisualEncoder(
            backbone=visual_backbone,
            output_dim=hidden_size,
            pretrained=visual_pretrained,
            freeze=freeze_visual_encoder,
            image_size=image_size,
        )

        if text_model_name:
            self.text_encoder = TransformerTextEncoder(
                model_name=text_model_name,
                output_dim=hidden_size,
                pretrained=text_pretrained,
                freeze=freeze_text_encoder,
                local_files_only=transformers_local_files_only,
            )
        else:
            self.text_encoder = TrainableTextEncoder(
                vocab_size=simple_text_vocab_size,
                output_dim=hidden_size,
                max_length=max_instruction_length,
            )

        self.tokenizer = self.text_encoder.tokenizer
        self.prev_action_embedding = nn.Embedding(
            num_actions + 1,
            prev_action_embedding_dim,
            padding_idx=START_PREV_ACTION,
        )
        self.fusion = GatedMultimodalFusion(
            hidden_size, hidden_size, hidden_size, dropout=dropout
        )
        self.rnn = nn.GRU(
            hidden_size + prev_action_embedding_dim,
            recurrent_hidden_size,
            batch_first=True,
        )
        self.policy_head = nn.Sequential(
            nn.LayerNorm(recurrent_hidden_size),
            nn.Dropout(dropout),
            nn.Linear(recurrent_hidden_size, num_actions),
        )

    def tokenize(
        self, instructions: Sequence[str], device: Optional[torch.device] = None
    ) -> Dict[str, Tensor]:
        encoded = self.tokenizer(
            instructions,
            padding=True,
            truncation=True,
            max_length=self.max_instruction_length,
            return_tensors="pt",
        )
        if device is not None:
            encoded = {key: val.to(device) for key, val in encoded.items()}
        return encoded

    def forward(
        self,
        rgb: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        prev_actions: Tensor,
        lengths: Optional[Tensor] = None,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(1)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1)

        batch, steps = rgb.shape[:2]
        visual_feats = self.visual_encoder(rgb)
        text_feats = self.text_encoder(input_ids, attention_mask)
        fused = self.fusion(visual_feats, text_feats)
        prev_emb = self.prev_action_embedding(prev_actions.long())
        rnn_input = torch.cat([fused, prev_emb], dim=-1)

        if lengths is not None:
            lengths = lengths.clamp_min(1).cpu()
            packed = pack_padded_sequence(
                rnn_input, lengths, batch_first=True, enforce_sorted=False
            )
            packed_out, hidden_state = self.rnn(packed, hidden_state)
            rnn_out, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=steps
            )
        else:
            rnn_out, hidden_state = self.rnn(rnn_input, hidden_state)

        logits = self.policy_head(rnn_out.reshape(batch * steps, -1))
        logits = logits.view(batch, steps, self.num_actions)
        return logits, hidden_state

    def step(
        self,
        rgb: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        prev_action: Tensor,
        hidden_state: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        logits, hidden_state = self.forward(
            rgb,
            input_ids,
            attention_mask,
            prev_action,
            lengths=None,
            hidden_state=hidden_state,
        )
        return logits[:, -1], hidden_state

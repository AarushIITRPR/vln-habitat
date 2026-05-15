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


class DepthEncoder(nn.Module):
    r"""Compact depth encoder for geometry cues in CPU/Intel-friendly VLN runs."""

    def __init__(self, output_dim: int = 256) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, kernel_size=5, stride=2, padding=2),
            nn.GroupNorm(8, 32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.GroupNorm(8, 128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
            nn.Linear(128, output_dim),
            nn.LayerNorm(output_dim),
            nn.ReLU(inplace=True),
        )

    def forward(self, depth: Tensor) -> Tensor:
        had_time_dim = depth.dim() == 5
        if had_time_dim:
            batch, steps = depth.shape[:2]
            depth = depth.reshape(batch * steps, *depth.shape[2:])
        if depth.dim() != 4:
            raise ValueError(
                "Depth encoder expects depth with shape (B,C,H,W) or "
                f"(B,T,C,H,W), got {tuple(depth.shape)}"
            )
        if depth.size(1) != 1 and depth.size(-1) == 1:
            depth = depth.permute(0, 3, 1, 2).contiguous()
        if depth.size(1) > 1:
            depth = depth[:, :1]
        depth = depth.float().clamp(0.0, 1.0)
        feats = self.encoder(depth)
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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_tokens: bool = False,
    ) -> Tensor:
        outputs = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask
        )
        if return_tokens:
            return self.proj(outputs.last_hidden_state)

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

    def forward(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
        return_tokens: bool = False,
    ) -> Tensor:
        lengths = attention_mask.long().sum(dim=1).clamp_min(1).cpu()
        embedded = self.embedding(input_ids.long())
        if return_tokens:
            outputs, _ = self.rnn(embedded)
            return self.out(outputs)

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


class CMAMultimodalFusion(nn.Module):
    r"""Cross-modal attention fusion for VLN.

    Each visual timestep queries the instruction token sequence. This keeps the
    recurrent action policy from Task 2, but replaces pooled sentence fusion
    with instruction-token attention similar to compact CMA-style VLN agents.
    """

    def __init__(
        self,
        hidden_dim: int,
        num_heads: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        if hidden_dim % num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} must be divisible by num_heads={num_heads}"
            )
        self.visual_proj = nn.Linear(hidden_dim, hidden_dim)
        self.text_proj = nn.Linear(hidden_dim, hidden_dim)
        self.attn = nn.MultiheadAttention(
            hidden_dim, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.gate = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.Sigmoid(),
        )
        self.out = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Dropout(dropout),
        )

    def forward(
        self,
        visual_feats: Tensor,
        text_token_feats: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        visual = torch.tanh(self.visual_proj(visual_feats))
        text = torch.tanh(self.text_proj(text_token_feats))
        key_padding_mask = attention_mask == 0
        attended_text, _ = self.attn(
            visual,
            text,
            text,
            key_padding_mask=key_padding_mask,
            need_weights=False,
        )
        gate = self.gate(torch.cat([visual, attended_text], dim=-1))
        return self.out(gate * visual + (1.0 - gate) * attended_text)


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
        fusion_type: str = "gated",
        cma_num_heads: int = 4,
        use_depth: bool = False,
        separate_stop_head: bool = False,
    ) -> None:
        super().__init__()
        self.num_actions = num_actions
        self.max_instruction_length = max_instruction_length
        self.fusion_type = fusion_type.lower()
        self.use_depth = use_depth
        self.separate_stop_head = separate_stop_head

        self.visual_encoder = PretrainedVisualEncoder(
            backbone=visual_backbone,
            output_dim=hidden_size,
            pretrained=visual_pretrained,
            freeze=freeze_visual_encoder,
            image_size=image_size,
        )
        if self.use_depth:
            self.depth_encoder = DepthEncoder(output_dim=hidden_size)
            self.rgbd_fusion = nn.Sequential(
                nn.Linear(hidden_size * 2, hidden_size),
                nn.LayerNorm(hidden_size),
                nn.ReLU(inplace=True),
                nn.Dropout(dropout),
            )
        else:
            self.depth_encoder = None
            self.rgbd_fusion = None

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
        if self.fusion_type == "cma":
            self.fusion = CMAMultimodalFusion(
                hidden_size, num_heads=cma_num_heads, dropout=dropout
            )
        elif self.fusion_type == "gated":
            self.fusion = GatedMultimodalFusion(
                hidden_size, hidden_size, hidden_size, dropout=dropout
            )
        else:
            raise ValueError(
                f"Unsupported fusion_type '{fusion_type}'. Use 'gated' or 'cma'."
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
        if self.separate_stop_head:
            self.stop_head = nn.Sequential(
                nn.LayerNorm(recurrent_hidden_size),
                nn.Dropout(dropout),
                nn.Linear(recurrent_hidden_size, 1),
            )
        else:
            self.stop_head = None

    def _encode_observation(
        self,
        rgb: Tensor,
        depth: Optional[Tensor] = None,
    ) -> Tensor:
        visual_feats = self.visual_encoder(rgb)
        if self.use_depth:
            if depth is None:
                raise ValueError("use_depth=True requires a depth tensor")
            if self.depth_encoder is None or self.rgbd_fusion is None:
                raise RuntimeError("Depth encoder is not initialized")
            depth_feats = self.depth_encoder(depth)
            visual_feats = self.rgbd_fusion(
                torch.cat([visual_feats, depth_feats], dim=-1)
            )
        return visual_feats

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
        depth: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(1)
        if depth is not None and depth.dim() == 4:
            depth = depth.unsqueeze(1)
        if prev_actions.dim() == 1:
            prev_actions = prev_actions.unsqueeze(1)

        batch, steps = rgb.shape[:2]
        visual_feats = self._encode_observation(rgb, depth)
        if self.fusion_type == "cma":
            text_feats = self.text_encoder(
                input_ids, attention_mask, return_tokens=True
            )
            fused = self.fusion(visual_feats, text_feats, attention_mask)
        else:
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
        if self.stop_head is not None:
            stop_logits = self.stop_head(rnn_out.reshape(batch * steps, -1))
            stop_logits = stop_logits.view(batch, steps)
            return (logits, stop_logits), hidden_state
        return logits, hidden_state

    def encode_instruction(
        self,
        input_ids: Tensor,
        attention_mask: Tensor,
    ) -> Tensor:
        if self.fusion_type == "cma":
            return self.text_encoder(input_ids, attention_mask, return_tokens=True)
        return self.text_encoder(input_ids, attention_mask)

    def step(
        self,
        rgb: Tensor,
        input_ids: Tensor,
        attention_mask: Tensor,
        prev_action: Tensor,
        hidden_state: Optional[Tensor] = None,
        instruction_features: Optional[Tensor] = None,
        depth: Optional[Tensor] = None,
    ) -> Tuple[Tensor, Tensor]:
        if instruction_features is None:
            output, hidden_state = self.forward(
                rgb,
                input_ids,
                attention_mask,
                prev_action,
                lengths=None,
                hidden_state=hidden_state,
                depth=depth,
            )
            if isinstance(output, tuple):
                action_logits, stop_logits = output
                return (action_logits[:, -1], stop_logits[:, -1]), hidden_state
            return output[:, -1], hidden_state

        if rgb.dim() == 4:
            rgb = rgb.unsqueeze(1)
        if depth is not None and depth.dim() == 4:
            depth = depth.unsqueeze(1)
        if prev_action.dim() == 1:
            prev_action = prev_action.unsqueeze(1)

        visual_feats = self._encode_observation(rgb, depth)
        if self.fusion_type == "cma":
            fused = self.fusion(
                visual_feats, instruction_features, attention_mask
            )
        else:
            fused = self.fusion(visual_feats, instruction_features)

        prev_emb = self.prev_action_embedding(prev_action.long())
        rnn_input = torch.cat([fused, prev_emb], dim=-1)
        rnn_out, hidden_state = self.rnn(rnn_input, hidden_state)
        logits = self.policy_head(rnn_out[:, -1])
        if self.stop_head is not None:
            stop_logits = self.stop_head(rnn_out[:, -1]).squeeze(-1)
            return (logits, stop_logits), hidden_state
        return logits, hidden_state

    def select_action(
        self,
        model_output,
        stop_threshold: float = 0.5,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        if isinstance(model_output, tuple):
            action_logits, stop_logits = model_output
            stop_prob = torch.sigmoid(stop_logits)
            action = action_logits[..., 1:].argmax(dim=-1) + 1
            action = torch.where(
                stop_prob >= stop_threshold,
                torch.zeros_like(action),
                action,
            )
            return action, stop_prob
        return model_output.argmax(dim=-1), None

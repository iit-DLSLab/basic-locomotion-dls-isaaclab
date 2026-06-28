from __future__ import annotations

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, Dataset, random_split


def _make_group_norm(num_channels: int) -> nn.GroupNorm:
    for num_groups in (8, 4, 2, 1):
        if num_channels % num_groups == 0:
            return nn.GroupNorm(num_groups=num_groups, num_channels=num_channels)
    raise ValueError(f"Could not build GroupNorm for {num_channels} channels.")


def _sinusoidal_position_embedding(length: int, dim: int, device: torch.device, dtype: torch.dtype) -> Tensor:
    positions = torch.arange(length, device=device, dtype=dtype).unsqueeze(1)
    frequency = torch.exp(torch.arange(0, dim, 2, device=device, dtype=dtype) * (-math.log(10000.0) / max(dim, 1)))
    angles = positions * frequency.unsqueeze(0)

    embedding = torch.zeros(length, dim, device=device, dtype=dtype)
    embedding[:, 0::2] = torch.sin(angles)
    if dim > 1:
        embedding[:, 1::2] = torch.cos(angles[:, : embedding[:, 1::2].shape[1]])
    return embedding


def _prepare_lidar_sequence(lidar_data: Tensor) -> Tensor:
    if lidar_data.dim() == 3:
        return lidar_data.unsqueeze(1)
    if lidar_data.dim() == 4:
        return lidar_data
    raise ValueError(
        "lidar_data must have shape (B, N, F) or (B, T_lidar, N, F), "
        f"but got {tuple(lidar_data.shape)}"
    )


def _prepare_proprio_history(robot_info: Tensor) -> Tensor:
    if robot_info.dim() == 2:
        return robot_info.unsqueeze(1)
    if robot_info.dim() == 3:
        return robot_info
    raise ValueError(
        "robot_info must have shape (B, F) or (B, T_prop, F), "
        f"but got {tuple(robot_info.shape)}"
    )


def _select_point_indices(num_points: int, max_points: int | None, device: torch.device) -> Tensor | None:
    if max_points is None or num_points <= max_points:
        return None
    return torch.linspace(0, num_points - 1, max_points, device=device).round().long()


def _split_points_and_mask(lidar_sequence: Tensor) -> tuple[Tensor, Tensor]:
    points = lidar_sequence[..., :3]
    if lidar_sequence.shape[-1] >= 4:
        valid_mask = lidar_sequence[..., 3] > 0.5
    else:
        valid_mask = torch.isfinite(points).all(dim=-1)

    points = torch.nan_to_num(points, nan=0.0, posinf=0.0, neginf=0.0)
    return points, valid_mask


class ConvNormAct(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, stride: int = 1, padding: int = 0):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
            _make_group_norm(out_channels),
            nn.GELU(),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.block = nn.Sequential(
            ConvNormAct(in_channels, out_channels, kernel_size=3, padding=1),
            ConvNormAct(out_channels, out_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class ConditionalUNetRefiner(nn.Module):
    def __init__(self, input_channels: int, base_channels: int):
        super().__init__()
        self.enc1 = DoubleConv(input_channels, base_channels)
        self.enc2 = DoubleConv(base_channels, base_channels * 2)
        self.bottleneck = DoubleConv(base_channels * 2, base_channels * 4)
        self.dec1 = DoubleConv(base_channels * 4 + base_channels * 2, base_channels * 2)
        self.dec2 = DoubleConv(base_channels * 2 + base_channels, base_channels)
        self.output_projection = nn.Conv2d(base_channels, 1, kernel_size=1)

    def forward(self, x: Tensor) -> Tensor:
        skip_1 = self.enc1(x)
        skip_2 = self.enc2(F.max_pool2d(skip_1, kernel_size=2))
        bottleneck = self.bottleneck(F.max_pool2d(skip_2, kernel_size=2))

        x = F.interpolate(bottleneck, size=skip_2.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec1(torch.cat((x, skip_2), dim=1))
        x = F.interpolate(x, size=skip_1.shape[-2:], mode="bilinear", align_corners=False)
        x = self.dec2(torch.cat((x, skip_1), dim=1))
        return self.output_projection(x)


class ProprioceptiveHistoryEncoder(nn.Module):
    def __init__(self, proprio_dim: int, embed_dim: int, hidden_dim: int):
        super().__init__()
        self.input_norm = nn.LayerNorm(proprio_dim)
        self.encoder = nn.Sequential(
            nn.Linear(proprio_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, embed_dim),
        )
        self.output_norm = nn.LayerNorm(embed_dim)

    def forward(self, proprio_history: Tensor) -> Tensor:
        batch_size, history_steps, _ = proprio_history.shape
        tokens = self.encoder(self.input_norm(proprio_history))
        position_embedding = _sinusoidal_position_embedding(history_steps, tokens.shape[-1], tokens.device, tokens.dtype)
        tokens = self.output_norm(tokens + position_embedding.view(1, history_steps, -1))
        return tokens.reshape(batch_size, history_steps, -1)


class FeedForwardBlock(nn.Module):
    def __init__(self, embed_dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.block(x)


class CrossAttentionBlock(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float):
        super().__init__()
        self.query_norm = nn.LayerNorm(embed_dim)
        self.context_norm = nn.LayerNorm(embed_dim)
        self.attention = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.feedforward_norm = nn.LayerNorm(embed_dim)
        self.feedforward = FeedForwardBlock(embed_dim=embed_dim, dropout=dropout)

    def forward(self, query_tokens: Tensor, context_tokens: Tensor, context_padding_mask: Tensor | None = None) -> Tensor:
        normalized_query = self.query_norm(query_tokens)
        normalized_context = self.context_norm(context_tokens)
        attention_output, _ = self.attention(
            query=normalized_query,
            key=normalized_context,
            value=normalized_context,
            key_padding_mask=context_padding_mask,
            need_weights=False,
        )
        fused = query_tokens + attention_output
        fused = fused + self.feedforward(self.feedforward_norm(fused))
        return fused


@dataclass
class TerrainReconstructionOutput:
    rough_heightmap: Tensor
    refined_heightmap: Tensor
    hidden_state: Tensor | None


class PointFrameEncoder(nn.Module):
    def __init__(self, point_feature_dim: int, embed_dim: int, max_points: int | None = 2048):
        super().__init__()
        self.point_feature_dim = point_feature_dim
        self.max_points = max_points
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_feature_dim),
            nn.Linear(point_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
            nn.GELU(),
        )
        self.frame_projection = nn.Sequential(
            nn.LayerNorm(embed_dim * 2),
            nn.Linear(embed_dim * 2, embed_dim),
            nn.GELU(),
        )

    def forward(self, lidar_sequence: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, lidar_steps, num_points, feature_dim = lidar_sequence.shape
        point_indices = _select_point_indices(num_points, self.max_points, lidar_sequence.device)
        if point_indices is not None:
            lidar_sequence = lidar_sequence.index_select(dim=2, index=point_indices)

        points, valid_mask = _split_points_and_mask(lidar_sequence)
        point_features = lidar_sequence[..., :feature_dim].clone()
        point_features[..., :3] = points

        flat_features = point_features.reshape(batch_size * lidar_steps, point_features.shape[-2], feature_dim)
        flat_mask = valid_mask.reshape(batch_size * lidar_steps, valid_mask.shape[-1])
        encoded = self.point_encoder(flat_features)

        mask_f = flat_mask.unsqueeze(-1).to(encoded.dtype)
        valid_count = mask_f.sum(dim=1).clamp_min(1.0)
        mean_pooled = (encoded * mask_f).sum(dim=1) / valid_count
        max_pooled = encoded.masked_fill(~flat_mask.unsqueeze(-1), -1.0e6).max(dim=1).values
        max_pooled = torch.where(flat_mask.any(dim=1, keepdim=True), max_pooled, torch.zeros_like(max_pooled))

        frame_features = self.frame_projection(torch.cat((mean_pooled, max_pooled), dim=-1))
        frame_features = frame_features.view(batch_size, lidar_steps, -1)
        global_context = frame_features.mean(dim=1)
        return frame_features, global_context


class LidarTokenEncoder(nn.Module):
    def __init__(self, point_feature_dim: int, embed_dim: int, max_points: int | None = 2048):
        super().__init__()
        self.point_feature_dim = point_feature_dim
        self.max_points = max_points
        self.point_encoder = nn.Sequential(
            nn.LayerNorm(point_feature_dim),
            nn.Linear(point_feature_dim, 64),
            nn.GELU(),
            nn.Linear(64, 128),
            nn.GELU(),
            nn.Linear(128, embed_dim),
        )
        self.token_norm = nn.LayerNorm(embed_dim)

    def forward(self, lidar_sequence: Tensor) -> tuple[Tensor, Tensor, Tensor | None]:
        batch_size, lidar_steps, num_points, feature_dim = lidar_sequence.shape
        point_indices = _select_point_indices(num_points, self.max_points, lidar_sequence.device)
        if point_indices is not None:
            lidar_sequence = lidar_sequence.index_select(dim=2, index=point_indices)
            num_points = lidar_sequence.shape[2]

        points, valid_mask = _split_points_and_mask(lidar_sequence)
        point_features = lidar_sequence[..., :feature_dim].clone()
        point_features[..., :3] = points

        tokens = self.point_encoder(point_features)
        time_pos = _sinusoidal_position_embedding(lidar_steps, tokens.shape[-1], tokens.device, tokens.dtype)
        point_pos = _sinusoidal_position_embedding(num_points, tokens.shape[-1], tokens.device, tokens.dtype)
        tokens = tokens + time_pos.view(1, lidar_steps, 1, -1) + point_pos.view(1, 1, num_points, -1)
        tokens = self.token_norm(tokens)

        flat_tokens = tokens.reshape(batch_size, lidar_steps * num_points, -1)
        flat_mask = valid_mask.reshape(batch_size, lidar_steps * num_points)
        padding_mask = ~flat_mask
        padding_mask = padding_mask if padding_mask.any() else None

        mask_f = flat_mask.unsqueeze(-1).to(flat_tokens.dtype)
        valid_count = mask_f.sum(dim=1).clamp_min(1.0)
        global_context = (flat_tokens * mask_f).sum(dim=1) / valid_count
        return flat_tokens, global_context, padding_mask


class PointNetGRUTerrainReconstructor(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        point_feature_dim: int = 4,
        heightmap_size: tuple[int, int] = (7, 9),
        embed_dim: int = 128,
        proprio_hidden_dim: int = 256,
        lidar_gru_hidden_dim: int = 128,
        proprio_gru_hidden_dim: int = 128,
        recurrent_layers: int = 1,
        fusion_hidden_dim: int = 256,
        refinement_context_channels: int = 32,
        refinement_base_channels: int = 32,
        max_lidar_points: int | None = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heightmap_size = heightmap_size
        self.lidar_gru_hidden_dim = lidar_gru_hidden_dim
        self.proprio_gru_hidden_dim = proprio_gru_hidden_dim
        self.refinement_context_channels = refinement_context_channels
        self.lidar_encoder = PointFrameEncoder(point_feature_dim, embed_dim=embed_dim, max_points=max_lidar_points)
        self.proprio_encoder = ProprioceptiveHistoryEncoder(proprio_dim, embed_dim=embed_dim, hidden_dim=proprio_hidden_dim)
        self.lidar_memory = nn.GRU(
            embed_dim,
            lidar_gru_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.proprio_memory = nn.GRU(
            embed_dim,
            proprio_gru_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )

        fused_dim = lidar_gru_hidden_dim + proprio_gru_hidden_dim
        self.rough_decoder = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, heightmap_size[0] * heightmap_size[1]),
        )
        self.context_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, refinement_context_channels * heightmap_size[0] * heightmap_size[1]),
            nn.GELU(),
        )
        self.refiner = ConditionalUNetRefiner(1 + refinement_context_channels, base_channels=refinement_base_channels)

    def forward(self, lidar_data: Tensor, robot_info: Tensor, hidden_state: Tensor | None = None) -> TerrainReconstructionOutput:
        lidar_sequence = _prepare_lidar_sequence(lidar_data)
        proprio_history = _prepare_proprio_history(robot_info)

        if hidden_state is None:
            lidar_hidden_state = None
            proprio_hidden_state = None
        else:
            lidar_hidden_state = hidden_state[..., : self.lidar_gru_hidden_dim].contiguous()
            proprio_hidden_state = hidden_state[..., self.lidar_gru_hidden_dim :].contiguous()

        lidar_features, lidar_context = self.lidar_encoder(lidar_sequence)
        proprio_features = self.proprio_encoder(proprio_history)
        lidar_memory_tokens, lidar_hidden_state = self.lidar_memory(lidar_features, lidar_hidden_state)
        proprio_memory_tokens, proprio_hidden_state = self.proprio_memory(proprio_features, proprio_hidden_state)

        fused_state = torch.cat((lidar_memory_tokens[:, -1], proprio_memory_tokens[:, -1]), dim=-1)
        rough_heightmap = self.rough_decoder(fused_state).view(-1, 1, *self.heightmap_size)
        lidar_context = self.context_decoder(lidar_context).view(-1, self.refinement_context_channels, *self.heightmap_size)
        refined_heightmap = rough_heightmap + self.refiner(torch.cat((rough_heightmap, lidar_context), dim=1))
        stacked_hidden = torch.cat((lidar_hidden_state, proprio_hidden_state), dim=-1)
        return TerrainReconstructionOutput(rough_heightmap, refined_heightmap, stacked_hidden)


class LidarTransformerTerrainReconstructor(nn.Module):
    def __init__(
        self,
        proprio_dim: int,
        point_feature_dim: int = 4,
        heightmap_size: tuple[int, int] = (7, 9),
        embed_dim: int = 128,
        proprio_hidden_dim: int = 256,
        num_attention_heads: int = 4,
        num_cross_attention_layers: int = 2,
        recurrent_hidden_dim: int = 192,
        recurrent_layers: int = 1,
        refinement_context_channels: int = 32,
        refinement_base_channels: int = 32,
        max_lidar_points: int | None = 2048,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heightmap_size = heightmap_size
        self.refinement_context_channels = refinement_context_channels
        self.lidar_encoder = LidarTokenEncoder(point_feature_dim, embed_dim=embed_dim, max_points=max_lidar_points)
        self.proprio_encoder = ProprioceptiveHistoryEncoder(proprio_dim, embed_dim=embed_dim, hidden_dim=proprio_hidden_dim)
        self.cross_attention_blocks = nn.ModuleList(
            [CrossAttentionBlock(embed_dim, num_attention_heads, dropout) for _ in range(num_cross_attention_layers)]
        )
        self.memory = nn.GRU(
            embed_dim,
            recurrent_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.rough_decoder = nn.Sequential(
            nn.LayerNorm(recurrent_hidden_dim),
            nn.Linear(recurrent_hidden_dim, recurrent_hidden_dim * 2),
            nn.GELU(),
            nn.Linear(recurrent_hidden_dim * 2, heightmap_size[0] * heightmap_size[1]),
        )
        self.context_decoder = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, refinement_context_channels * heightmap_size[0] * heightmap_size[1]),
            nn.GELU(),
        )
        self.refiner = ConditionalUNetRefiner(1 + refinement_context_channels, base_channels=refinement_base_channels)

    def forward(self, lidar_data: Tensor, robot_info: Tensor, hidden_state: Tensor | None = None) -> TerrainReconstructionOutput:
        lidar_sequence = _prepare_lidar_sequence(lidar_data)
        proprio_history = _prepare_proprio_history(robot_info)

        lidar_tokens, lidar_context, lidar_padding_mask = self.lidar_encoder(lidar_sequence)
        fused_tokens = self.proprio_encoder(proprio_history)
        for block in self.cross_attention_blocks:
            fused_tokens = block(fused_tokens, lidar_tokens, context_padding_mask=lidar_padding_mask)

        memory_tokens, hidden_state = self.memory(fused_tokens, hidden_state)
        rough_heightmap = self.rough_decoder(memory_tokens[:, -1]).view(-1, 1, *self.heightmap_size)
        lidar_context = self.context_decoder(lidar_context).view(-1, self.refinement_context_channels, *self.heightmap_size)
        refined_heightmap = rough_heightmap + self.refiner(torch.cat((rough_heightmap, lidar_context), dim=1))
        return TerrainReconstructionOutput(rough_heightmap, refined_heightmap, hidden_state)


def compute_reconstruction_losses(prediction: TerrainReconstructionOutput, target_heightmap: Tensor) -> dict[str, Tensor]:
    rough_loss = F.mse_loss(prediction.rough_heightmap, target_heightmap)
    refined_loss = F.l1_loss(prediction.refined_heightmap, target_heightmap)
    return {"loss": rough_loss + refined_loss, "rough_loss": rough_loss, "refined_loss": refined_loss}


class FakeLidarTerrainReconstructionDataset(Dataset):
    def __init__(
        self,
        num_samples: int = 128,
        lidar_history: int = 5,
        proprio_history: int = 50,
        num_points: int = 512,
        point_feature_dim: int = 4,
        proprio_dim: int = 48,
        heightmap_size: tuple[int, int] = (7, 9),
        seed: int = 0,
    ):
        super().__init__()
        generator = torch.Generator().manual_seed(seed)
        xyz = torch.rand(num_samples, lidar_history, num_points, 3, generator=generator) * 2.0 - 1.0
        if point_feature_dim >= 4:
            valid = (torch.rand(num_samples, lidar_history, num_points, 1, generator=generator) > 0.1).float()
            extras = [valid]
            if point_feature_dim > 4:
                extras.append(torch.randn(num_samples, lidar_history, num_points, point_feature_dim - 4, generator=generator))
            self.lidar_data = torch.cat((xyz * valid, *extras), dim=-1)
        else:
            self.lidar_data = xyz[..., :point_feature_dim]
        self.robot_info = torch.randn(num_samples, proprio_history, proprio_dim, generator=generator)
        self.heightmaps = self._build_targets(heightmap_size)

    def _build_targets(self, heightmap_size: tuple[int, int]) -> Tensor:
        batch_size = self.lidar_data.shape[0]
        height, width = heightmap_size
        x_grid = torch.linspace(-1.0, 1.0, width).view(1, 1, 1, width)
        y_grid = torch.linspace(-1.0, 1.0, height).view(1, 1, height, 1)
        valid = self.lidar_data[..., 3:4] if self.lidar_data.shape[-1] >= 4 else torch.ones_like(self.lidar_data[..., :1])
        z_mean = (self.lidar_data[..., 2:3] * valid).sum(dim=(1, 2)) / valid.sum(dim=(1, 2)).clamp_min(1.0)
        last_robot_state = self.robot_info[:, -1]
        target = z_mean.view(batch_size, 1, 1, 1) + 0.2 * last_robot_state[:, 0].view(batch_size, 1, 1, 1) * x_grid
        target = target + 0.2 * last_robot_state[:, 1].view(batch_size, 1, 1, 1) * y_grid
        return target.clamp(-2.0, 2.0)

    def __len__(self) -> int:
        return self.lidar_data.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.lidar_data[index], self.robot_info[index], self.heightmaps[index]


class SavedLidarTerrainReconstructionDataset(Dataset):
    def __init__(self, dataset_path: str):
        super().__init__()
        dataset = torch.load(dataset_path, map_location="cpu")
        required_keys = {"lidar_data", "robot_info", "heightmaps"}
        missing_keys = required_keys.difference(dataset)
        if missing_keys:
            missing = ", ".join(sorted(missing_keys))
            raise KeyError(f"Dataset at {dataset_path} is missing required keys: {missing}")

        self.lidar_data = dataset["lidar_data"].float()
        self.robot_info = dataset["robot_info"].float()
        self.heightmaps = dataset["heightmaps"].float()
        self.metadata = dataset.get("metadata", {})

        dataset_size = self.lidar_data.shape[0]
        if self.robot_info.shape[0] != dataset_size or self.heightmaps.shape[0] != dataset_size:
            raise ValueError("lidar_data, robot_info, and heightmaps must all contain the same number of samples.")

    def __len__(self) -> int:
        return self.lidar_data.shape[0]

    def __getitem__(self, index: int) -> tuple[Tensor, Tensor, Tensor]:
        return self.lidar_data[index], self.robot_info[index], self.heightmaps[index]


def _build_model(
    model_type: str,
    proprio_dim: int,
    point_feature_dim: int,
    heightmap_size: tuple[int, int],
    max_lidar_points: int | None,
) -> nn.Module:
    if model_type == "pointnet_gru":
        return PointNetGRUTerrainReconstructor(
            proprio_dim=proprio_dim,
            point_feature_dim=point_feature_dim,
            heightmap_size=heightmap_size,
            max_lidar_points=max_lidar_points,
        )
    if model_type == "transformer":
        return LidarTransformerTerrainReconstructor(
            proprio_dim=proprio_dim,
            point_feature_dim=point_feature_dim,
            heightmap_size=heightmap_size,
            max_lidar_points=max_lidar_points,
        )
    raise ValueError(f"Unsupported model_type: {model_type}")


def _run_epoch(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> dict[str, float]:
    is_training = optimizer is not None
    model.train(mode=is_training)
    total_loss = total_rough_loss = total_refined_loss = 0.0
    total_samples = 0

    for lidar_data, robot_info, target_heightmap in data_loader:
        lidar_data = lidar_data.to(device)
        robot_info = robot_info.to(device)
        target_heightmap = target_heightmap.to(device)

        if is_training:
            optimizer.zero_grad(set_to_none=True)

        with torch.set_grad_enabled(is_training):
            prediction = model(lidar_data=lidar_data, robot_info=robot_info)
            losses = compute_reconstruction_losses(prediction, target_heightmap)

        if is_training:
            losses["loss"].backward()
            optimizer.step()

        batch_size = lidar_data.shape[0]
        total_samples += batch_size
        total_loss += losses["loss"].item() * batch_size
        total_rough_loss += losses["rough_loss"].item() * batch_size
        total_refined_loss += losses["refined_loss"].item() * batch_size

    return {
        "loss": total_loss / max(total_samples, 1),
        "rough_loss": total_rough_loss / max(total_samples, 1),
        "refined_loss": total_refined_loss / max(total_samples, 1),
    }


def train_terrain_reconstructor(
    model: nn.Module,
    train_loader: DataLoader,
    validation_loader: DataLoader | None = None,
    num_epochs: int = 5,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    device: str | torch.device = "cpu",
) -> list[dict[str, float]]:
    device = torch.device(device)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    history: list[dict[str, float]] = []
    for epoch in range(num_epochs):
        train_metrics = _run_epoch(model, train_loader, device, optimizer)
        epoch_metrics = {
            "epoch": float(epoch + 1),
            "train_loss": train_metrics["loss"],
            "train_rough_loss": train_metrics["rough_loss"],
            "train_refined_loss": train_metrics["refined_loss"],
        }
        if validation_loader is not None:
            with torch.no_grad():
                validation_metrics = _run_epoch(model, validation_loader, device, optimizer=None)
            epoch_metrics.update(
                {
                    "val_loss": validation_metrics["loss"],
                    "val_rough_loss": validation_metrics["rough_loss"],
                    "val_refined_loss": validation_metrics["refined_loss"],
                }
            )

        history.append(epoch_metrics)
        summary = (
            f"Epoch {epoch + 1}/{num_epochs} | "
            f"train: total={epoch_metrics['train_loss']:.4f}, "
            f"rough={epoch_metrics['train_rough_loss']:.4f}, "
            f"refined={epoch_metrics['train_refined_loss']:.4f}"
        )
        if validation_loader is not None:
            summary += (
                f" | val: total={epoch_metrics['val_loss']:.4f}, "
                f"rough={epoch_metrics['val_rough_loss']:.4f}, "
                f"refined={epoch_metrics['val_refined_loss']:.4f}"
            )
        print(summary)

    return history


def _default_artifact_dir(dataset_path: str, model_type: str) -> Path:
    return Path(dataset_path).expanduser().resolve().parent / f"{model_type}_training"


def _save_heightmap_comparison_png(
    model: nn.Module,
    data_loader: DataLoader,
    device: torch.device,
    output_path: str | Path,
    max_samples: int = 4,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    lidar_data, robot_info, target_heightmap = next(iter(data_loader))
    lidar_data = lidar_data.to(device)
    robot_info = robot_info.to(device)
    target_heightmap = target_heightmap.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(lidar_data=lidar_data, robot_info=robot_info)

    max_samples = max(1, min(max_samples, target_heightmap.shape[0]))
    target = target_heightmap[:max_samples, 0].detach().cpu()
    predicted = prediction.refined_heightmap[:max_samples, 0].detach().cpu()
    error = predicted - target

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(max_samples, 3, figsize=(9, 3 * max_samples), squeeze=False)
    for sample_index in range(max_samples):
        target_map = target[sample_index].numpy()
        predicted_map = predicted[sample_index].numpy()
        error_map = error[sample_index].numpy()
        value_min = min(float(target_map.min()), float(predicted_map.min()))
        value_max = max(float(target_map.max()), float(predicted_map.max()))
        error_abs_max = max(abs(float(error_map.min())), abs(float(error_map.max())), 1e-6)

        images = (
            axes[sample_index, 0].imshow(target_map, cmap="viridis", vmin=value_min, vmax=value_max),
            axes[sample_index, 1].imshow(predicted_map, cmap="viridis", vmin=value_min, vmax=value_max),
            axes[sample_index, 2].imshow(error_map, cmap="coolwarm", vmin=-error_abs_max, vmax=error_abs_max),
        )
        axes[sample_index, 0].set_title("Target")
        axes[sample_index, 1].set_title("Predicted")
        axes[sample_index, 2].set_title("Error")
        for axis, image in zip(axes[sample_index], images):
            axis.set_xticks([])
            axis.set_yticks([])
            fig.colorbar(image, ax=axis, fraction=0.046, pad=0.04)

    fig.tight_layout()
    fig.savefig(output_path, dpi=160)
    plt.close(fig)
    print(f"[INFO] Saved heightmap comparison PNG to: {output_path}")


def train_from_saved_dataset(
    dataset_path: str,
    model_type: str = "pointnet_gru",
    device: str | torch.device | None = None,
    batch_size: int = 32,
    num_epochs: int = 50,
    validation_fraction: float = 0.2,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    max_lidar_points: int | None = 2048,
    model_path: str | None = None,
    comparison_png_path: str | None = None,
    comparison_samples: int = 4,
) -> None:
    torch.manual_seed(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = SavedLidarTerrainReconstructionDataset(dataset_path)
    val_size = int(len(dataset) * validation_fraction)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(f"Dataset at {dataset_path} has {len(dataset)} samples, too small for validation.")

    if val_size > 0:
        train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = dataset
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    point_feature_dim = dataset.lidar_data.shape[-1]
    heightmap_size = tuple(dataset.heightmaps.shape[-2:])
    proprio_dim = dataset.robot_info.shape[-1]
    model = _build_model(model_type, proprio_dim, point_feature_dim, heightmap_size, max_lidar_points)
    history = train_terrain_reconstructor(
        model,
        train_loader,
        validation_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    artifact_dir = _default_artifact_dir(dataset_path, model_type)
    model_path = Path(model_path or artifact_dir / f"{model_type}_lidar_terrain_reconstructor.pt")
    comparison_png_path = Path(comparison_png_path or artifact_dir / "heightmap_comparison.png")
    model_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "model_state_dict": {name: parameter.detach().cpu() for name, parameter in model.state_dict().items()},
            "model_type": model_type,
            "model_config": {
                "proprio_dim": proprio_dim,
                "point_feature_dim": point_feature_dim,
                "heightmap_size": heightmap_size,
                "max_lidar_points": max_lidar_points,
            },
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "dataset_metadata": dataset.metadata,
            "history": history,
        },
        model_path,
    )
    print(f"[INFO] Saved LiDAR model checkpoint to: {model_path}")

    comparison_loader = val_loader if val_loader is not None else train_loader
    _save_heightmap_comparison_png(model, comparison_loader, device, comparison_png_path, max_samples=comparison_samples)
    final_metrics = history[-1]
    print(
        "LiDAR training finished | "
        f"model={model_type}, samples={len(dataset)}, train_samples={train_size}, val_samples={val_size}, "
        f"lidar={tuple(dataset.lidar_data.shape[1:])}, robot_info={tuple(dataset.robot_info.shape[1:])}, "
        f"heightmap={tuple(dataset.heightmaps.shape[1:])}, "
        f"final_val_loss={final_metrics.get('val_loss', final_metrics['train_loss']):.4f}"
    )


def run_fake_data_smoke_test(
    model_type: str = "pointnet_gru",
    device: str | torch.device | None = None,
    num_samples: int = 96,
    batch_size: int = 8,
    num_epochs: int = 3,
    max_lidar_points: int | None = 512,
) -> None:
    torch.manual_seed(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FakeLidarTerrainReconstructionDataset(num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size], generator=torch.Generator().manual_seed(0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = _build_model(
        model_type,
        proprio_dim=dataset.robot_info.shape[-1],
        point_feature_dim=dataset.lidar_data.shape[-1],
        heightmap_size=tuple(dataset.heightmaps.shape[-2:]),
        max_lidar_points=max_lidar_points,
    )
    history = train_terrain_reconstructor(model, train_loader, val_loader, num_epochs=num_epochs, device=device)
    sample_lidar, sample_robot_info, sample_target = next(iter(val_loader))
    model.eval()
    with torch.no_grad():
        prediction = model(sample_lidar.to(device), sample_robot_info.to(device))
    assert prediction.rough_heightmap.shape == sample_target.to(device).shape
    assert prediction.refined_heightmap.shape == sample_target.to(device).shape
    final_metrics = history[-1]
    print(
        "LiDAR smoke test passed | "
        f"model={model_type}, lidar={tuple(sample_lidar.shape)}, robot_info={tuple(sample_robot_info.shape)}, "
        f"heightmap={tuple(prediction.refined_heightmap.shape)}, "
        f"final_val_loss={final_metrics.get('val_loss', final_metrics['train_loss']):.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train LiDAR-to-heightmap terrain reconstructors.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a lidar_terrain_reconstruction_dataset.pt file. If omitted, runs a fake-data smoke test.",
    )
    parser.add_argument("--model", type=str, default="pointnet_gru", choices=("pointnet_gru", "transformer"))
    parser.add_argument("--device", type=str, default=None, help="Training device. Defaults to cuda if available.")
    parser.add_argument("--num_samples", type=int, default=96, help="Number of fake samples for smoke tests.")
    parser.add_argument("--batch_size", type=int, default=None, help="Batch size.")
    parser.add_argument("--epochs", type=int, default=None, help="Number of training epochs.")
    parser.add_argument("--validation_fraction", type=float, default=0.2, help="Fraction of samples used for validation.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--max_lidar_points",
        type=int,
        default=2048,
        help="Maximum points per LiDAR frame consumed by the model. Use <=0 to keep all points.",
    )
    parser.add_argument("--model_path", type=str, default=None, help="Where to save the trained model.")
    parser.add_argument("--comparison_png_path", type=str, default=None, help="Where to save the target-vs-predicted PNG.")
    parser.add_argument("--comparison_samples", type=int, default=4, help="Number of samples to include in the PNG.")
    return parser.parse_args()


__all__ = [
    "FakeLidarTerrainReconstructionDataset",
    "LidarTransformerTerrainReconstructor",
    "PointNetGRUTerrainReconstructor",
    "SavedLidarTerrainReconstructionDataset",
    "TerrainReconstructionOutput",
    "compute_reconstruction_losses",
    "run_fake_data_smoke_test",
    "train_from_saved_dataset",
    "train_terrain_reconstructor",
]


if __name__ == "__main__":
    args = _parse_args()
    max_lidar_points = args.max_lidar_points if args.max_lidar_points and args.max_lidar_points > 0 else None
    if args.dataset_path:
        train_from_saved_dataset(
            dataset_path=args.dataset_path,
            model_type=args.model,
            device=args.device,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            num_epochs=args.epochs if args.epochs is not None else 50,
            validation_fraction=args.validation_fraction,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            max_lidar_points=max_lidar_points,
            model_path=args.model_path,
            comparison_png_path=args.comparison_png_path,
            comparison_samples=args.comparison_samples,
        )
    else:
        run_fake_data_smoke_test(
            model_type=args.model,
            device=args.device,
            num_samples=args.num_samples,
            batch_size=args.batch_size if args.batch_size is not None else 8,
            num_epochs=args.epochs if args.epochs is not None else 3,
            max_lidar_points=max_lidar_points,
        )

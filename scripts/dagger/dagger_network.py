from __future__ import annotations

import torch
import torch.nn as nn
from torch import Tensor


class DaggerReplayBuffer:
    """Fixed-size CPU aggregation buffer for DAgger supervision."""

    def __init__(self, capacity: int, depth_dtype: torch.dtype = torch.float16):
        if capacity <= 0:
            raise ValueError("DaggerReplayBuffer capacity must be positive.")

        self.capacity = capacity
        self.depth_dtype = depth_dtype
        self.size = 0
        self.next_idx = 0
        self._depth_sequences: Tensor | None = None
        self._common_obs: Tensor | None = None
        self._expert_actions: Tensor | None = None

    def __len__(self) -> int:
        return self.size

    def _lazy_init(self, depth_sequences: Tensor, common_obs: Tensor, expert_actions: Tensor) -> None:
        self._depth_sequences = torch.empty(
            (self.capacity, *depth_sequences.shape[1:]),
            dtype=self.depth_dtype,
            device="cpu",
        )
        self._common_obs = torch.empty((self.capacity, common_obs.shape[-1]), dtype=torch.float32, device="cpu")
        self._expert_actions = torch.empty(
            (self.capacity, expert_actions.shape[-1]),
            dtype=torch.float32,
            device="cpu",
        )

    def add_batch(
        self,
        depth_sequences: Tensor,
        common_obs: Tensor,
        expert_actions: Tensor,
        max_samples: int | None = None,
    ) -> int:
        """Add a batch of labeled states without saving anything to disk."""
        if depth_sequences.shape[0] == 0:
            return 0

        batch_size = depth_sequences.shape[0]
        sample_count = batch_size
        if max_samples is not None and max_samples > 0:
            sample_count = min(sample_count, max_samples)
        sample_count = min(sample_count, self.capacity)

        if sample_count < batch_size:
            selected_indices = torch.randperm(batch_size, device=depth_sequences.device)[:sample_count]
            depth_sequences = depth_sequences[selected_indices]
            common_obs = common_obs[selected_indices]
            expert_actions = expert_actions[selected_indices]

        if self._depth_sequences is None or self._common_obs is None or self._expert_actions is None:
            self._lazy_init(depth_sequences=depth_sequences, common_obs=common_obs, expert_actions=expert_actions)

        write_indices = (torch.arange(sample_count) + self.next_idx) % self.capacity
        self._depth_sequences[write_indices] = depth_sequences.detach().to("cpu", dtype=self.depth_dtype)
        self._common_obs[write_indices] = common_obs.detach().to("cpu", dtype=torch.float32)
        self._expert_actions[write_indices] = expert_actions.detach().to("cpu", dtype=torch.float32)

        self.next_idx = (self.next_idx + sample_count) % self.capacity
        self.size = min(self.capacity, self.size + sample_count)
        return sample_count

    def sample(self, batch_size: int, device: torch.device | str) -> tuple[Tensor, Tensor, Tensor]:
        if self.size == 0:
            raise RuntimeError("Cannot sample from an empty DAgger replay buffer.")
        if self._depth_sequences is None or self._common_obs is None or self._expert_actions is None:
            raise RuntimeError("DAgger replay buffer storage was not initialized.")

        indices = torch.randint(0, self.size, (batch_size,))
        depth_sequences = self._depth_sequences[indices].to(device=device, dtype=torch.float32, non_blocking=True)
        common_obs = self._common_obs[indices].to(device=device, non_blocking=True)
        expert_actions = self._expert_actions[indices].to(device=device, non_blocking=True)
        return depth_sequences, common_obs, expert_actions


class DepthCnnEncoder(nn.Module):
    """Encodes each depth frame into a compact token for recurrent memory."""

    def __init__(self, depth_channels: int = 1, feature_dim: int = 64):
        super().__init__()
        self.backbone = nn.Sequential(
            nn.Conv2d(depth_channels, 16, kernel_size=5, stride=2, padding=2),
            nn.ELU(inplace=True),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(32, 48, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.Conv2d(48, 64, kernel_size=3, stride=2, padding=1),
            nn.ELU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            nn.Flatten(),
        )
        self.projection = nn.Sequential(
            nn.Linear(64, feature_dim),
            nn.ELU(inplace=True),
        )

    def forward(self, depth_frames: Tensor) -> Tensor:
        return self.projection(self.backbone(depth_frames))


class DaggerNet(nn.Module):
    """Depth-map DAgger policy.

    The policy first embeds depth frames with a CNN, passes the depth token sequence
    through a GRU, concatenates the latest depth memory with ``obs["common"]``, and
    predicts the expert action.
    """

    def __init__(
        self,
        vec_size: int,
        output_size: int,
        depth_channels: int = 1,
        cnn_dim: int = 64,
        gru_hidden: int = 128,
        gru_layers: int = 1,
        head_hidden: int = 128,
        dropout: float = 0.0,
    ):
        super().__init__()
        self.depth_encoder = DepthCnnEncoder(depth_channels=depth_channels, feature_dim=cnn_dim)
        self.depth_gru = nn.GRU(
            input_size=cnn_dim,
            hidden_size=gru_hidden,
            num_layers=gru_layers,
            batch_first=True,
            dropout=dropout if gru_layers > 1 else 0.0,
        )
        self.action_head = nn.Sequential(
            nn.Linear(gru_hidden + vec_size, head_hidden),
            nn.ELU(inplace=True),
            nn.Linear(head_hidden, head_hidden),
            nn.ELU(inplace=True),
            nn.Linear(head_hidden, output_size),
        )

    def _prepare_depth_sequence(self, depth_data: Tensor) -> Tensor:
        if depth_data.dim() == 4:
            return depth_data.unsqueeze(1)
        if depth_data.dim() == 5:
            return depth_data
        raise ValueError(
            "depth_data must have shape (B, C, H, W) or (B, T, C, H, W), "
            f"but got {tuple(depth_data.shape)}"
        )

    def _prepare_common_obs(self, common_obs: Tensor) -> Tensor:
        if common_obs.dim() == 2:
            return common_obs
        if common_obs.dim() == 3:
            return common_obs[:, -1]
        raise ValueError(
            "common_obs must have shape (B, F) or (B, T, F), "
            f"but got {tuple(common_obs.shape)}"
        )

    def forward(self, depth_data: Tensor, common_obs: Tensor, hidden: Tensor | None = None) -> tuple[Tensor, Tensor]:
        depth_sequence = self._prepare_depth_sequence(depth_data)
        common_features = self._prepare_common_obs(common_obs)

        batch_size, sequence_length, channels, height, width = depth_sequence.shape
        depth_tokens = self.depth_encoder(depth_sequence.reshape(batch_size * sequence_length, channels, height, width))
        depth_tokens = depth_tokens.view(batch_size, sequence_length, -1)

        depth_memory, new_hidden = self.depth_gru(depth_tokens, hidden)
        fused_features = torch.cat((depth_memory[:, -1], common_features), dim=-1)
        actions = self.action_head(fused_features)
        return actions, new_hidden

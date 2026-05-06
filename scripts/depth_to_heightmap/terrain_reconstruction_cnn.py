from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.utils.data import DataLoader, random_split

from terrain_reconstruction_transformer import (
    ConditionalUNetRefiner,
    ConvNormAct,
    FakeTerrainReconstructionDataset,
    ProprioceptiveHistoryEncoder,
    SavedTerrainReconstructionDataset,
    TerrainReconstructionOutput,
    compute_reconstruction_losses,
    train_terrain_reconstructor,
)


class DepthFeatureEncoder(nn.Module):
    def __init__(self, depth_channels: int, feature_dim: int, context_channels: int):
        super().__init__()
        self.backbone = nn.Sequential(
            ConvNormAct(depth_channels, 32, kernel_size=5, stride=2, padding=2),
            ConvNormAct(32, 64, kernel_size=3, stride=2, padding=1),
            ConvNormAct(64, 96, kernel_size=3, stride=2, padding=1),
            ConvNormAct(96, 128, kernel_size=3, padding=1),
        )
        self.sequence_pool = nn.AdaptiveAvgPool2d((4, 4))
        self.feature_projection = nn.Sequential(
            nn.Flatten(start_dim=1),
            nn.Linear(128 * 4 * 4, feature_dim),
            nn.GELU(),
        )
        self.context_projection = nn.Sequential(
            nn.Conv2d(128, context_channels, kernel_size=1),
            nn.GELU(),
        )

    def forward(self, depth_sequence: Tensor) -> tuple[Tensor, Tensor]:
        batch_size, depth_steps, channels, _, _ = depth_sequence.shape
        encoded = self.backbone(depth_sequence.reshape(batch_size * depth_steps, channels, *depth_sequence.shape[-2:]))

        sequence_features = self.feature_projection(self.sequence_pool(encoded))
        sequence_features = sequence_features.view(batch_size, depth_steps, -1)

        depth_context = self.context_projection(encoded)
        depth_context = depth_context.view(batch_size, depth_steps, depth_context.shape[1], *depth_context.shape[-2:]).mean(dim=1)
        return sequence_features, depth_context


class CNNGRUTerrainReconstructor(nn.Module):
    """Terrain reconstructor baseline with CNN encoders and GRU fusion, without attention."""

    def __init__(
        self,
        proprio_dim: int,
        depth_channels: int = 1,
        heightmap_size: tuple[int, int] = (20, 20),
        depth_feature_dim: int = 128,
        proprio_feature_dim: int = 128,
        proprio_hidden_dim: int = 256,
        depth_gru_hidden_dim: int = 128,
        proprio_gru_hidden_dim: int = 128,
        recurrent_layers: int = 1,
        fusion_hidden_dim: int = 256,
        refinement_context_channels: int = 32,
        refinement_base_channels: int = 32,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.heightmap_size = heightmap_size
        self.depth_gru_hidden_dim = depth_gru_hidden_dim
        self.proprio_gru_hidden_dim = proprio_gru_hidden_dim

        self.depth_encoder = DepthFeatureEncoder(
            depth_channels=depth_channels,
            feature_dim=depth_feature_dim,
            context_channels=refinement_context_channels,
        )
        self.proprio_encoder = ProprioceptiveHistoryEncoder(
            proprio_dim=proprio_dim,
            embed_dim=proprio_feature_dim,
            hidden_dim=proprio_hidden_dim,
        )

        self.depth_memory = nn.GRU(
            input_size=depth_feature_dim,
            hidden_size=depth_gru_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )
        self.proprio_memory = nn.GRU(
            input_size=proprio_feature_dim,
            hidden_size=proprio_gru_hidden_dim,
            num_layers=recurrent_layers,
            batch_first=True,
            dropout=dropout if recurrent_layers > 1 else 0.0,
        )

        fused_dim = depth_gru_hidden_dim + proprio_gru_hidden_dim
        self.rough_decoder = nn.Sequential(
            nn.LayerNorm(fused_dim),
            nn.Linear(fused_dim, fusion_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(fusion_hidden_dim, heightmap_size[0] * heightmap_size[1]),
        )
        self.refinement_context_projection = nn.Sequential(
            nn.Conv2d(refinement_context_channels, refinement_base_channels, kernel_size=3, padding=1),
            nn.GELU(),
        )
        self.refiner = ConditionalUNetRefiner(
            input_channels=1 + refinement_base_channels,
            base_channels=refinement_base_channels,
        )

    def _prepare_depth_sequence(self, depth_data: Tensor) -> Tensor:
        if depth_data.dim() == 4:
            return depth_data.unsqueeze(1)
        if depth_data.dim() == 5:
            return depth_data
        raise ValueError(
            "depth_data must have shape (B, C, H, W) or (B, T_depth, C, H, W), "
            f"but got {tuple(depth_data.shape)}"
        )

    def _prepare_proprio_history(self, robot_info: Tensor) -> Tensor:
        if robot_info.dim() == 2:
            return robot_info.unsqueeze(1)
        if robot_info.dim() == 3:
            return robot_info
        raise ValueError(
            "robot_info must have shape (B, F) or (B, T_prop, F), "
            f"but got {tuple(robot_info.shape)}"
        )

    def forward(
        self,
        depth_data: Tensor,
        robot_info: Tensor,
        hidden_state: Tensor | None = None,
    ) -> TerrainReconstructionOutput:
        depth_sequence = self._prepare_depth_sequence(depth_data)
        proprio_history = self._prepare_proprio_history(robot_info)

        if hidden_state is None:
            depth_hidden_state = None
            proprio_hidden_state = None
        else:
            depth_hidden_state = hidden_state[..., : self.depth_gru_hidden_dim].contiguous()
            proprio_hidden_state = hidden_state[..., self.depth_gru_hidden_dim :].contiguous()

        depth_features, depth_context = self.depth_encoder(depth_sequence)
        proprio_features = self.proprio_encoder(proprio_history)

        depth_memory_tokens, depth_hidden_state = self.depth_memory(depth_features, depth_hidden_state)
        proprio_memory_tokens, proprio_hidden_state = self.proprio_memory(proprio_features, proprio_hidden_state)

        fused_state = torch.cat((depth_memory_tokens[:, -1], proprio_memory_tokens[:, -1]), dim=-1)
        rough_heightmap = self.rough_decoder(fused_state).view(-1, 1, *self.heightmap_size)

        depth_context = self.refinement_context_projection(depth_context)
        depth_context = F.interpolate(depth_context, size=self.heightmap_size, mode="bilinear", align_corners=False)

        refinement_input = torch.cat((rough_heightmap, depth_context), dim=1)
        refinement_residual = self.refiner(refinement_input)
        refined_heightmap = rough_heightmap + refinement_residual

        stacked_hidden = torch.cat((depth_hidden_state, proprio_hidden_state), dim=-1)
        return TerrainReconstructionOutput(
            rough_heightmap=rough_heightmap,
            refined_heightmap=refined_heightmap,
            hidden_state=stacked_hidden,
        )


def run_fake_data_smoke_test(
    device: str | torch.device | None = None,
    num_samples: int = 96,
    batch_size: int = 8,
    num_epochs: int = 3,
) -> None:
    torch.manual_seed(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    dataset = FakeTerrainReconstructionDataset(num_samples=num_samples)
    train_size = int(0.8 * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size],
        generator=torch.Generator().manual_seed(0),
    )

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = CNNGRUTerrainReconstructor(proprio_dim=dataset.robot_info.shape[-1])
    history = train_terrain_reconstructor(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        num_epochs=num_epochs,
        device=device,
    )

    sample_depth, sample_robot_info, sample_target = next(iter(val_loader))
    sample_depth = sample_depth.to(device)
    sample_robot_info = sample_robot_info.to(device)
    sample_target = sample_target.to(device)

    model.eval()
    with torch.no_grad():
        sample_prediction = model(depth_data=sample_depth, robot_info=sample_robot_info)
        losses = compute_reconstruction_losses(prediction=sample_prediction, target_heightmap=sample_target)

    assert sample_prediction.rough_heightmap.shape == sample_target.shape
    assert sample_prediction.refined_heightmap.shape == sample_target.shape

    final_metrics = history[-1]
    print(
        "CNN-GRU smoke test passed | "
        f"depth={tuple(sample_depth.shape)}, "
        f"robot_info={tuple(sample_robot_info.shape)}, "
        f"heightmap={tuple(sample_prediction.refined_heightmap.shape)}, "
        f"eval_loss={losses['loss'].item():.4f}, "
        f"final_val_loss={final_metrics.get('val_loss', final_metrics['train_loss']):.4f}"
    )


def _default_artifact_dir(dataset_path: str) -> Path:
    return Path(dataset_path).expanduser().resolve().parent / "cnn_gru_training"


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

    depth_data, robot_info, target_heightmap = next(iter(data_loader))
    depth_data = depth_data.to(device)
    robot_info = robot_info.to(device)
    target_heightmap = target_heightmap.to(device)

    model.eval()
    with torch.no_grad():
        prediction = model(depth_data=depth_data, robot_info=robot_info)

    max_samples = max(1, min(max_samples, target_heightmap.shape[0]))
    target = target_heightmap[:max_samples, 0].detach().cpu()
    predicted = prediction.refined_heightmap[:max_samples, 0].detach().cpu()
    error = predicted - target
    num_samples = target.shape[0]

    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    fig, axes = plt.subplots(num_samples, 3, figsize=(9, 3 * num_samples), squeeze=False)
    for sample_index in range(num_samples):
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
    device: str | torch.device | None = None,
    batch_size: int = 32,
    num_epochs: int = 50,
    validation_fraction: float = 0.2,
    learning_rate: float = 3e-4,
    weight_decay: float = 1e-4,
    model_path: str | None = None,
    comparison_png_path: str | None = None,
    comparison_samples: int = 4,
) -> None:
    torch.manual_seed(0)
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)

    dataset = SavedTerrainReconstructionDataset(dataset_path)
    val_size = int(len(dataset) * validation_fraction)
    train_size = len(dataset) - val_size
    if train_size <= 0:
        raise ValueError(
            f"Dataset at {dataset_path} has {len(dataset)} samples, which is too small "
            f"for validation_fraction={validation_fraction}."
        )

    if val_size > 0:
        train_dataset, val_dataset = random_split(
            dataset,
            [train_size, val_size],
            generator=torch.Generator().manual_seed(0),
        )
        val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    else:
        train_dataset = dataset
        val_loader = None

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    depth_channels = dataset.depth_data.shape[-3]
    heightmap_size = tuple(dataset.heightmaps.shape[-2:])
    proprio_dim = dataset.robot_info.shape[-1]
    model = CNNGRUTerrainReconstructor(
        proprio_dim=proprio_dim,
        depth_channels=depth_channels,
        heightmap_size=heightmap_size,
    )
    history = train_terrain_reconstructor(
        model=model,
        train_loader=train_loader,
        validation_loader=val_loader,
        num_epochs=num_epochs,
        learning_rate=learning_rate,
        weight_decay=weight_decay,
        device=device,
    )

    artifact_dir = _default_artifact_dir(dataset_path)
    model_path = model_path or str(artifact_dir / "cnn_gru_terrain_reconstructor.pt")
    comparison_png_path = comparison_png_path or str(artifact_dir / "heightmap_comparison.png")

    model_path = Path(model_path)
    model_path.parent.mkdir(parents=True, exist_ok=True)
    model_state_dict = {name: parameter.detach().cpu() for name, parameter in model.state_dict().items()}
    torch.save(
        {
            "model_state_dict": model_state_dict,
            "model_config": {
                "proprio_dim": proprio_dim,
                "depth_channels": depth_channels,
                "heightmap_size": heightmap_size,
            },
            "dataset_path": str(Path(dataset_path).expanduser().resolve()),
            "dataset_metadata": dataset.metadata,
            "history": history,
        },
        model_path,
    )
    print(f"[INFO] Saved CNN-GRU model checkpoint to: {model_path}")

    comparison_loader = val_loader if val_loader is not None else train_loader
    _save_heightmap_comparison_png(
        model=model,
        data_loader=comparison_loader,
        device=device,
        output_path=comparison_png_path,
        max_samples=comparison_samples,
    )

    final_metrics = history[-1]
    print(
        "CNN-GRU training finished | "
        f"samples={len(dataset)}, "
        f"train_samples={train_size}, "
        f"val_samples={val_size}, "
        f"depth={tuple(dataset.depth_data.shape[1:])}, "
        f"robot_info={tuple(dataset.robot_info.shape[1:])}, "
        f"heightmap={tuple(dataset.heightmaps.shape[1:])}, "
        f"final_val_loss={final_metrics.get('val_loss', final_metrics['train_loss']):.4f}"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the CNN-GRU terrain reconstructor.")
    parser.add_argument(
        "--dataset_path",
        type=str,
        default=None,
        help="Path to a collected terrain_reconstruction_dataset.pt file. If omitted, runs a fake-data smoke test.",
    )
    parser.add_argument("--device", type=str, default=None, help="Training device. Defaults to cuda if available.")
    parser.add_argument("--num_samples", type=int, default=96, help="Number of fake samples.")
    parser.add_argument(
        "--batch_size",
        type=int,
        default=None,
        help="Training batch size. Defaults to 32 for saved datasets and 8 for fake-data smoke tests.",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs. Defaults to 50 for saved datasets and 3 for fake-data smoke tests.",
    )
    parser.add_argument("--validation_fraction", type=float, default=0.2, help="Fraction of samples used for validation.")
    parser.add_argument("--learning_rate", type=float, default=3e-4, help="AdamW learning rate.")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="AdamW weight decay.")
    parser.add_argument(
        "--model_path",
        type=str,
        default=None,
        help="Where to save the trained model. Defaults to <dataset_dir>/cnn_gru_training/cnn_gru_terrain_reconstructor.pt.",
    )
    parser.add_argument(
        "--comparison_png_path",
        type=str,
        default=None,
        help="Where to save the target-vs-predicted PNG. Defaults to <dataset_dir>/cnn_gru_training/heightmap_comparison.png.",
    )
    parser.add_argument("--comparison_samples", type=int, default=4, help="Number of samples to include in the PNG.")
    return parser.parse_args()


__all__ = [
    "CNNGRUTerrainReconstructor",
    "run_fake_data_smoke_test",
    "train_from_saved_dataset",
]


if __name__ == "__main__":
    args = _parse_args()
    if args.dataset_path:
        train_from_saved_dataset(
            dataset_path=args.dataset_path,
            device=args.device,
            batch_size=args.batch_size if args.batch_size is not None else 32,
            num_epochs=args.epochs if args.epochs is not None else 50,
            validation_fraction=args.validation_fraction,
            learning_rate=args.learning_rate,
            weight_decay=args.weight_decay,
            model_path=args.model_path,
            comparison_png_path=args.comparison_png_path,
            comparison_samples=args.comparison_samples,
        )
    else:
        run_fake_data_smoke_test(
            device=args.device,
            num_samples=args.num_samples,
            batch_size=args.batch_size if args.batch_size is not None else 8,
            num_epochs=args.epochs if args.epochs is not None else 3,
        )

import torch
from torch.utils.data import Dataset
import math
import random
import os
import tempfile


class CustomDataset(Dataset):
    def __init__(self, max_size=None):
        self.data = []
        self.labels = []
        self.max_size = max_size

    def add_sample(self, input_data, label):
        # There is a problem! we append (num_envs, features) as one element inside the list, not N!

        # Save only random 128 element from the input_data and label
        random_idx = random.sample(range(input_data.size(0)), min(128, input_data.size(0)))
        input_data_cpu = input_data[random_idx].clone().detach().cpu()
        label_cpu = label[random_idx].clone().detach().cpu()

        #input_data_cpu = input_data.clone().detach().cpu()
        #label_cpu = label.clone().detach().cpu()
        self.data.append(input_data_cpu)
        self.labels.append(label_cpu)

        # Check if the buffer exceeds the maximum size
        if self.max_size is not None and len(self.data) > self.max_size:
            # Remove a random sample to maintain the buffer size
            idx_to_remove = random.randint(0, len(self.data) - 1)
            del self.data[idx_to_remove]
            del self.labels[idx_to_remove]


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx], self.labels[idx]


class SupervisedNetworkBase(torch.nn.Module):
    network_type = "base"

    def __init__(self, in_features, out_features, dataset_max_size=80000):
        super().__init__()
        self.input_features = in_features
        self.output_features = out_features
        self.dataset = CustomDataset(max_size=dataset_max_size)

    def train_network(self, batch_size=512, epochs=1000, learning_rate=1e-3, device='cpu', validation_split=0.2):
        """Train the network with validation loss tracking.
        
        Args:
            batch_size: Batch size for training
            epochs: Number of training epochs
            learning_rate: Learning rate for optimizer
            device: Device to train on ('cpu' or 'cuda')
            validation_split: Fraction of data to use for validation (0.0 to 1.0)
        """
        # Split dataset into training and validation
        dataset_size = len(self.dataset)
        if dataset_size == 0:
            print("Warning: Dataset is empty. Cannot train.")
            return
        
        val_size = int(dataset_size * validation_split)
        train_size = dataset_size - val_size
        
        train_dataset, val_dataset = torch.utils.data.random_split(
            self.dataset, 
            [train_size, val_size]
        )
        
        # Define optimizer and loss function
        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        loss_fn = torch.nn.MSELoss()

        # Create DataLoaders for training and validation
        train_loader = torch.utils.data.DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True
        )
        
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False
        )

        # Training loop
        best_val_loss = float('inf')
        
        for epoch in range(epochs):
            with torch.inference_mode(False):
                with torch.enable_grad():
                # Training phase
                    self.train()
                    train_loss = 0.0
                    train_batches = 0
                    
                    for inputs, targets in train_loader:
                        # Forward pass
                        inputs = inputs.reshape(-1, inputs.size(-1)).clone().to(device)
                        targets = targets.reshape(-1, targets.size(-1)).clone().to(device)
                        predictions = self(inputs)

                        loss = loss_fn(predictions, targets)

                        # Backward pass and optimization
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()
                        
                        train_loss += loss.item()
                        train_batches += 1
                    
                    avg_train_loss = train_loss / train_batches if train_batches > 0 else 0.0
            
            # Validation phase
            self.eval()
            val_loss = 0.0
            val_batches = 0
            
            with torch.no_grad():
                for inputs, targets in val_loader:
                    inputs = inputs.reshape(-1, inputs.size(-1)).clone().to(device)
                    targets = targets.reshape(-1, targets.size(-1)).clone().to(device)
                    predictions = self(inputs)
                    
                    loss = loss_fn(predictions, targets)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches if val_batches > 0 else 0.0
            
            # Track best validation loss
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
            
            # Print progress
            if (epoch + 1) % 10 == 0 or epoch == 0:
                print(f"Epoch {epoch + 1}/{epochs} - Train Loss: {avg_train_loss:.6f}, Val Loss: {avg_val_loss:.6f}")
        
        self.eval()
        print(f"Training complete. Best validation loss: {best_val_loss:.6f}")
        print("Model set to evaluation mode.")

    def _checkpoint_kwargs(self):
        return {}

    def save_network(self, filepath, device='cpu'):
        """Save the network state dict to a file."""
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath) if os.path.dirname(filepath) else '.', exist_ok=True)
        
        # Move model to CPU for saving (optional, saves GPU memory)
        original_device = next(self.parameters()).device
        #self.cpu()
        
        # Save the state dict
        torch.save({
            'model_state_dict': self.state_dict(),
            'input_features': self.input_features,
            'output_features': self.output_features,
            'network_type': self.network_type,
            'model_kwargs': self._checkpoint_kwargs(),
        }, filepath)
        
        print(f"Network saved to {filepath}")
        
        # Move model back to original device
        #self.to(original_device)


class MlpNN(SupervisedNetworkBase):
    network_type = "mlp"

    def __init__(self, in_features, out_features):
        super().__init__(in_features, out_features)
        self.fc1 = torch.nn.Linear(in_features, 128)
        self.fc2 = torch.nn.Linear(128, 128)
        self.fc3 = torch.nn.Linear(128, 128)
        self.fc4 = torch.nn.Linear(128, out_features)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = torch.relu(self.fc3(x))
        x = torch.tanh(self.fc4(x))*2.0
        return x


class Chomp1d(torch.nn.Module):
    def __init__(self, chomp_size):
        super().__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        if self.chomp_size == 0:
            return x
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(torch.nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super().__init__()
        padding = (kernel_size - 1) * dilation
        self.conv1 = torch.nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp1 = Chomp1d(padding)
        self.conv2 = torch.nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size,
            padding=padding,
            dilation=dilation,
        )
        self.chomp2 = Chomp1d(padding)
        self.dropout = torch.nn.Dropout(dropout)
        self.downsample = torch.nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None

    def forward(self, x):
        residual = x if self.downsample is None else self.downsample(x)
        x = self.conv1(x)
        x = self.chomp1(x)
        x = torch.relu(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.chomp2(x)
        x = torch.relu(x)
        x = self.dropout(x)
        return torch.relu(x + residual)


class TemporalConvNN(SupervisedNetworkBase):
    network_type = "tcn"

    def __init__(
        self,
        in_features,
        out_features,
        sequence_length,
        hidden_channels=(128, 128, 128),
        kernel_size=3,
        dropout=0.0,
    ):
        super().__init__(in_features, out_features)
        if sequence_length <= 0:
            raise ValueError("sequence_length must be positive.")
        if in_features % sequence_length != 0:
            raise ValueError(
                f"in_features ({in_features}) must be divisible by sequence_length ({sequence_length})."
            )

        self.sequence_length = sequence_length
        self.features_per_step = in_features // sequence_length
        self.hidden_channels = tuple(hidden_channels)
        self.kernel_size = kernel_size
        self.dropout = dropout

        layers = []
        in_channels = self.features_per_step
        for idx, out_channels in enumerate(self.hidden_channels):
            layers.append(
                TemporalBlock(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    kernel_size=kernel_size,
                    dilation=2**idx,
                    dropout=dropout,
                )
            )
            in_channels = out_channels

        self.temporal_net = torch.nn.Sequential(*layers)
        self.fc1 = torch.nn.Linear(in_channels, 128)
        self.fc2 = torch.nn.Linear(128, out_features)

    def forward(self, x):
        if x.dim() == 1:
            x = x.unsqueeze(0)

        x = x.reshape(x.size(0), self.sequence_length, self.features_per_step)
        x = x.transpose(1, 2)
        x = self.temporal_net(x)
        x = x[:, :, -1]
        x = torch.relu(self.fc1(x))
        x = torch.tanh(self.fc2(x))*2.0
        return x

    def _checkpoint_kwargs(self):
        return {
            'sequence_length': self.sequence_length,
            'hidden_channels': self.hidden_channels,
            'kernel_size': self.kernel_size,
            'dropout': self.dropout,
        }


class FrozenRandomMlpEncoder(torch.nn.Module):
    def __init__(self, in_features, latent_features, hidden_features=128, seed=0):
        super().__init__()
        self.input_features = in_features
        self.latent_features = latent_features
        self.hidden_features = hidden_features
        self.seed = seed
        self.encoder = torch.nn.Sequential(
            torch.nn.Linear(in_features, hidden_features),
            torch.nn.ReLU(),
            torch.nn.Linear(hidden_features, latent_features),
            torch.nn.Tanh(),
        )
        self._reset_parameters(seed)
        for parameter in self.parameters():
            parameter.requires_grad_(False)
        self.eval()

    def _reset_parameters(self, seed):
        generator = torch.Generator(device="cpu")
        generator.manual_seed(seed)
        for module in self.encoder:
            if isinstance(module, torch.nn.Linear):
                bound = 1.0 / math.sqrt(module.in_features)
                module.weight.data.uniform_(-bound, bound, generator=generator)
                module.bias.data.uniform_(-bound, bound, generator=generator)

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        return self.encode(x)


def create_supervised_network(in_features, out_features, network_type="mlp", **kwargs):
    """Create a supervised network.

    Args:
        in_features: Flattened input dimension.
        out_features: Output dimension.
        network_type: "mlp" for MlpNN or "tcn" for TemporalConvNN.
        **kwargs: Extra architecture args. TCN requires sequence_length.
    """
    kwargs = dict(kwargs)
    normalized_type = (network_type or "mlp").lower()
    if normalized_type in ("mlp", "simple", "simple_nn"):
        return MlpNN(in_features, out_features)
    if normalized_type in ("tcn", "temporal_cnn", "temporal_conv", "temporal_convolution"):
        return TemporalConvNN(in_features, out_features, **kwargs)
    raise ValueError(f"Unknown supervised network type: {network_type}")


def load_network(filepath, device='cpu'):
    """Load the network state dict from a file."""
    if not os.path.isfile(filepath):
        raise FileNotFoundError(f"No such file: '{filepath}'")
    
    checkpoint = torch.load(filepath, map_location=device)
    
    input_features = checkpoint.get('input_features')
    output_features = checkpoint.get('output_features')
    if input_features is None or output_features is None:
        raise ValueError("Checkpoint does not contain model architecture information.")
    # Reinitialize the model with the correct architecture
    network_type = checkpoint.get('network_type', 'mlp')
    model_kwargs = checkpoint.get('model_kwargs', {})
    model = create_supervised_network(input_features, output_features, network_type=network_type, **model_kwargs)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    
    print(f"Network loaded from {filepath}")
    return model



if __name__ == "__main__":
    torch.manual_seed(0)

    sequence_length = 5
    features_per_step = 48
    input_features = sequence_length * features_per_step
    output_features = 3
    privileged_features = 33
    latent_features = 8
    batch_size = 16

    test_inputs = torch.randn(batch_size, input_features)
    test_targets = torch.randn(batch_size, output_features).clamp(-2.0, 2.0)
    privileged_targets = torch.randn(batch_size, privileged_features).clamp(-2.0, 2.0)

    with tempfile.TemporaryDirectory() as tmp_dir:
        for network_type in ("mlp", "tcn"):
            model = create_supervised_network(
                input_features,
                output_features,
                network_type=network_type,
                sequence_length=sequence_length,
            )

            predictions = model(test_inputs)
            assert predictions.shape == (batch_size, output_features), (
                f"{network_type} output shape mismatch: {predictions.shape}"
            )

            for _ in range(5):
                model.dataset.add_sample(test_inputs, test_targets)

            model.train_network(batch_size=2, epochs=2, learning_rate=1e-3, validation_split=0.2)

            checkpoint_path = os.path.join(tmp_dir, f"{network_type}.pth")
            model.save_network(checkpoint_path)
            loaded_model = load_network(checkpoint_path)
            loaded_predictions = loaded_model(test_inputs)
            assert loaded_predictions.shape == (batch_size, output_features), (
                f"Loaded {network_type} output shape mismatch: {loaded_predictions.shape}"
            )

            print(f"{network_type} smoke test passed.")

        random_encoder = FrozenRandomMlpEncoder(privileged_features, latent_features, hidden_features=32, seed=123)
        latent_targets = random_encoder.encode(privileged_targets)
        assert latent_targets.shape == (batch_size, latent_features), (
            f"Latent target shape mismatch: {latent_targets.shape}"
        )
        assert all(not parameter.requires_grad for parameter in random_encoder.parameters())

        latent_predictor = create_supervised_network(
            input_features,
            latent_features,
            network_type="tcn",
            sequence_length=sequence_length,
        )
        for _ in range(5):
            latent_predictor.dataset.add_sample(test_inputs, latent_targets)
        latent_predictor.train_network(batch_size=2, epochs=2, learning_rate=1e-3, validation_split=0.2)
        latent_predictions = latent_predictor(test_inputs)
        assert latent_predictions.shape == (batch_size, latent_features), (
            f"Latent predictor shape mismatch: {latent_predictions.shape}"
        )

        print("random-latent RMA smoke test passed.")

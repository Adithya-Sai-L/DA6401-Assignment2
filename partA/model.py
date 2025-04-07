import torch
from torch import nn
import pytorch_lightning as pl
from typing import List, Tuple, Union


class ConvNN(pl.LightningModule):
    def __init__(
        self,
        input_shape: Tuple[int, int, int] = (3, 800, 800),
        num_filters: List[int] = [32, 64, 128, 256, 512],
        kernel_sizes: List[int] = [3, 3, 3, 3, 3],
        conv_activations: Union[List[str], str] = "ReLU",
        dense_neurons: int = 512,
        dense_activation: str = "ReLU",
        learning_rate: float = 1e-3,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Parameter validation
        if len(num_filters) != 5:
            raise ValueError("num_filters must contain exactly 5 elements")
        if len(kernel_sizes) != 5:
            raise ValueError("kernel_sizes must contain exactly 5 elements")
            
        if isinstance(conv_activations, str):
            conv_activations = [conv_activations] * 5
        if len(conv_activations) != 5:
            raise ValueError("conv_activations must contain exactly 5 elements or be a single string")

        # Build convolutional blocks
        self.conv_blocks = nn.Sequential()
        in_channels = input_shape[0]
        
        for i in range(5):
            # Calculate padding to maintain spatial dimensions before pooling
            padding = (kernel_sizes[i] - 1) // 2
            
            self.conv_blocks.append(
                nn.Conv2d(
                    in_channels=in_channels,
                    out_channels=num_filters[i],
                    kernel_size=kernel_sizes[i],
                    padding=padding
                )
            )
            self.conv_blocks.append(getattr(nn, conv_activations[i])())
            self.conv_blocks.append(nn.MaxPool2d(kernel_size=2, stride=2))
            in_channels = num_filters[i]

        # Calculate flattened size dynamically
        with torch.no_grad():
            dummy_input = torch.randn(1, *input_shape)
            dummy_output = self.conv_blocks(dummy_input)
            flattened_size = dummy_output.view(1, -1).size(1)

        # Build dense layers
        self.dense = nn.Sequential(
            nn.Linear(flattened_size, dense_neurons),
            getattr(nn, dense_activation)(),
            nn.Linear(dense_neurons, 10)
        )

        # Loss function and metrics
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = learning_rate

    def forward(self, x):
        x = self.conv_blocks(x)
        x = x.view(x.size(0), -1)  # Flatten
        x = self.dense(x)
        return x

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)
    
    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        self.log("train_loss", loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)
        loss = self.loss_fn(y_hat, y)
        acc = (y_hat.argmax(dim=1) == y).float().mean()
        self.log("test_loss", loss)
        self.log("test_acc", acc)


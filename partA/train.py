import wandb
import pytorch_lightning as pl
from dataloader import get_training_dataloaders
from model import ConvNN
import os

def train():
    wandb.init()
    config = wandb.config
    wandb.run.name = f"num_filters_{config.num_filters}_conv_act_{config.conv_activation}_dense_act_{config.dense_activation}_dense_neurons_{config.dense_neurons}_batch_norm_{config.batch_normalization}_drop_out_{config.drop_out}_lr_{config.learning_rate}_optim_{config.optimizer}_batch_sz_{config.batch_size}_data_aug_{config.data_augmentation}"
    wandb.run.save
    # Model with sweep parameters
    model = ConvNN(
        num_filters=wandb.config.num_filters,
        conv_activations=wandb.config.conv_activation,
        dense_activation=wandb.config.dense_activation,
        dense_neurons=wandb.config.dense_neurons,
        optimizer=wandb.config.optimizer,
        learning_rate=wandb.config.learning_rate,
        batch_normalization=wandb.config.batch_normalization,
        drop_out=wandb.config.drop_out,
    )
    
    # Get DataLoaders
    train_loader, val_loader = get_training_dataloaders(
        data_dir="/home/adithyal/DL_PA2/data/nature_12K/inaturalist_12K",
        batch_size=wandb.config.batch_size,
        augmentation=wandb.config.data_augmentation
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="auto",
        logger=pl.loggers.WandbLogger(),
        max_epochs=20,
        enable_checkpointing=True
    )
    
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)

if __name__ == "__main__":
    from sweep_config import sweep_config
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    sweep_id = wandb.sweep(sweep_config, project="DA6401_PA2_partA")
    wandb.agent(sweep_id=sweep_id,
                function=train,
                count=50)

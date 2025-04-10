import wandb
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from dataloader import get_training_dataloaders
from model import ConvNN
import os
import argparse

def create_parser():
    parser = argparse.ArgumentParser(description='Accept Command line arguments for DA6401 Programming Assignment 2 - Part A')
    
    # Added arguments to accept hyperparameter configuration
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')    
    parser.add_argument('-we','--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    
    parser.add_argument('-f', '--num_filters', type=int, nargs='+', default=[32]*5, help='List of number of filters for each conv layer')
    parser.add_argument('-k', '--kernel_sizes', type=int, nargs='+', default=[3]*5, help='List of kernel sizes for each conv layer')
    parser.add_argument('-ac', '--conv_activation', type=str, default='ReLU', help='Activation function for conv layers')
    parser.add_argument('-ad', '--dense_activation', type=str, default='ReLU', help='Activation function for dense layer')
    parser.add_argument('-d', '--dense_neurons', type=int, default=512, help='Number of Neurons in dense layer')    
    parser.add_argument('-o', '--optimizer', type=str, default='Adam', help='Optimizer to use for training')
    parser.add_argument('-lr', '--learning_rate', type=float, default=1e-3, help='Learning rate for optimizer')
    parser.add_argument('-bn', '--batch_normalization', action='store_true', default=False, help='Use batch normalization')
    parser.add_argument('-dp', '--drop_out', type=float, default=0.0, help='Dropout rate for conv layers and dense layer')

    parser.add_argument('-da', '--data_augmentation', action='store_true', default=False, help='Use data augmentation')
    parser.add_argument('-bs', '--batch_size', type=int, default=32, help='Batch size for training')

    return parser

def train():
    parser = create_parser()
    args = parser.parse_args()

    wandb.init(project=args.wandb_project, entity=args.wandb_entity, config=vars(args))
    config = wandb.config
    wandb.run.name = f"num_filters_{config.num_filters}_kernel_sizes_{config.kernel_sizes}_conv_act_{config.conv_activation}_dense_act_{config.dense_activation}_dense_neurons_{config.dense_neurons}_batch_norm_{config.batch_normalization}_drop_out_{config.drop_out}_lr_{config.learning_rate}_optim_{config.optimizer}_batch_sz_{config.batch_size}_data_aug_{config.data_augmentation}"
    wandb.run.save
    # Model with sweep parameters
    model = ConvNN(
        num_filters=config.num_filters,
        kernel_sizes=config.kernel_sizes,
        conv_activations=config.conv_activation,
        dense_activation=config.dense_activation,
        dense_neurons=config.dense_neurons,
        optimizer=config.optimizer,
        learning_rate=config.learning_rate,
        batch_normalization=config.batch_normalization,
        drop_out=config.drop_out,
    )
    
    # Get DataLoaders
    train_loader, val_loader = get_training_dataloaders(
        data_dir="/home/adithyal/DL_PA2/data/nature_12K/inaturalist_12K",
        batch_size=config.batch_size,
        augmentation=config.data_augmentation
    )

    # Save only the best model checkpoint for each hyperparameter configuration
    checkpoint_callback = ModelCheckpoint(
    monitor="val_acc",       # Metric to monitor
    mode="max",               # "min" for loss, "max" for accuracy
    save_top_k=1,             # Save only the best model
    dirpath="./checkpoints/"+wandb.run.name,  # Directory to save checkpoints
    filename="best_model-{epoch:02d}-{val_acc:.2f}"  # Filename format
    )

    trainer = pl.Trainer(
        accelerator="gpu",
        devices="auto",
        strategy="auto",
        logger=pl.loggers.WandbLogger(),
        max_epochs=20,
        callbacks=[checkpoint_callback],
        enable_checkpointing=True
    )
    
    trainer.fit(model=model,
                train_dataloaders=train_loader,
                val_dataloaders=val_loader)
    
    wandb.finish()

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    train()
from train import train
import wandb
import argparse
from sweep_config import sweep_config

parser = argparse.ArgumentParser(description='Accept Command line arguments of wandb_project, wandb_entity')
parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')
parser.add_argument('-we','--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')

args = parser.parse_args()

sweep_id = wandb.sweep(sweep_config, entity=args.wandb_entity, project=args.wandb_project)
print(sweep_id)

wandb.agent(sweep_id, function=train, count=75)
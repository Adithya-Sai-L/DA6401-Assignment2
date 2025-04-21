import torch
import wandb
import matplotlib.pyplot as plt
from model import ConvNN
from dataloader import get_testing_dataloader
import argparse
import os
import torch.nn.functional as F
import numpy as np

def create_parser():
    parser = argparse.ArgumentParser(description='Accept Command line arguments for DA6401 Programming Assignment 2 - Part A')
    
    # Added arguments to accept hyperparameter configuration
    parser.add_argument('-wp', '--wandb_project', type=str, default='myprojectname',
                        help='Project name used to track experiments in Weights & Biases dashboard')    
    parser.add_argument('-we','--wandb_entity', type=str, default='myname',
                        help='Wandb Entity used to track experiments in the Weights & Biases dashboard.')
    parser.add_argument('-p', '--path', type=str, default='checkpoints/num_filters_[128, 128, 128, 128, 128]_kernel_sizes_[3, 3, 3, 3, 3]_conv_act_Mish_dense_act_ReLU_dense_neurons_128_batch_norm_True_drop_out_0.047004290996061904_lr_0.00023849526793374195_optim_AdamW_batch_sz_32_data_aug_True/best_model-epoch=18-val_acc=0.48.ckpt')
   
    return parser

parser = create_parser()
args = parser.parse_args()

# Load best model
best_model = ConvNN.load_from_checkpoint(args.path)
best_model.eval()

# Load test data
test_loader = get_testing_dataloader(
    data_dir="/home/adithyal/DL_PA2/data/nature_12K/inaturalist_12K",
    batch_size=32
)

# Initialize WandB
wandb.init(project=args.wandb_project, entity=args.wandb_entity, job_type="test")

# Compute test accuracy
correct = 0
total = 0

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        outputs = best_model(images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

test_accuracy = 100 * correct / total
wandb.log({"test_accuracy": test_accuracy})

# Generate prediction grid
num_classes = 10
samples_per_class = 3
class_data = {i: [] for i in range(num_classes)}
    

# Collect samples with no gradients
with torch.no_grad():
    for images, labels in test_loader:
        images, labels = images.cuda(), labels.cuda()
        
        # Get model predictions
        logits = best_model(images)
        probs = F.softmax(logits, dim=1)
        preds = torch.argmax(probs, dim=1)
        
        # Store samples for each class
        for img, label, pred, prob in zip(images, labels, preds, probs):
            cls = label.item()
            confidence = prob[pred].item() * 100  # Convert to percentage
            
            if len(class_data[cls]) < samples_per_class:
                class_data[cls].append((
                    img.cpu(), 
                    label.item(), 
                    pred.item(), 
                    confidence
                ))
        
        # Break if we have collected enough samples for each class
        if all(len(class_data[i]) >= samples_per_class for i in range(num_classes)):
            break

# Create the visualization grid
fig, axes = plt.subplots(
    num_classes, 
    samples_per_class, 
    figsize=(samples_per_class * 3.5, num_classes * 3.5)
)

# Handle the case when samples_per_class is 1
if samples_per_class == 1:
    axes = axes.reshape(-1, 1)

# Track metrics for each class
class_metrics = []
class_names = ['Amphibia', 'Animalia', 'Arachnida', 'Aves', 'Fungi', 'Insecta', 'Mammalia', 'Mollusca', 'Plantae', 'Reptilia']
# Fill the grid with images and predictions
for row_idx, cls in enumerate(range(num_classes)):
    correct_count = 0
    total_conf = 0
    
    for col_idx in range(samples_per_class):
        if col_idx < len(class_data[cls]):  # Safety check
            img, label, pred, confidence = class_data[cls][col_idx]
            
            # Convert tensor to numpy for display
            img_np = img.numpy().transpose(1, 2, 0)
            img_np = np.clip(img_np, 0, 1)
            
            # Get class names
            pred_label = class_names[pred]
            actual_label = class_names[label]
            
            # Track metrics
            is_correct = pred == label
            if is_correct:
                correct_count += 1
            total_conf += confidence
            
            # Set border color based on correctness
            color = "green" if is_correct else "red"
            
            # Plot the image
            ax = axes[row_idx, col_idx]
            ax.imshow(img_np)
            
            # Style the borders
            for side in ["top", "bottom", "left", "right"]:
                ax.spines[side].set_color(color)
                ax.spines[side].set_linewidth(4)
            
            # Remove ticks
            ax.set_xticks([])
            ax.set_yticks([])
            
            # Add prediction text
            if is_correct:
                ax.text(
                    0.5, 1.17, 
                    f"Predicted {pred_label} right!", 
                    transform=ax.transAxes,
                    color=color, 
                    fontsize=9, 
                    ha="center", 
                    va="bottom", 
                    fontweight="bold"
                )
            else:
                ax.text(
                    0.5, 1.17, 
                    f"Predicted {pred_label} instead of {actual_label}", 
                    transform=ax.transAxes,
                    color=color, 
                    fontsize=9, 
                    ha="center", 
                    va="bottom", 
                    fontweight="bold"
                )
            
            # Add confidence text
            ax.text(
                0.5, 1.05, 
                f"with {confidence:.2f}% confidence", 
                transform=ax.transAxes,
                color="black", 
                fontsize=9, 
                ha="center", 
                va="bottom"
            )
    
    # Calculate class metrics
    if samples_per_class > 0:  # Avoid division by zero
        avg_acc = correct_count / samples_per_class
        avg_conf = total_conf / samples_per_class
        class_metrics.append({
            'class': class_names[cls],
            'accuracy': avg_acc,
            'avg_confidence': avg_conf
        })

# Add title and adjust layout
fig.suptitle("Model Predictions Grid", fontsize=16, y=1.02)
plt.tight_layout(pad=2.0)

# Add a text box with overall metrics
overall_acc = sum(m['accuracy'] for m in class_metrics) / len(class_metrics)
overall_conf = sum(m['avg_confidence'] for m in class_metrics) / len(class_metrics)

plt.figtext(
    0.5, -0.05,
    f"Overall Accuracy: {overall_acc:.2f} | Average Confidence: {overall_conf:.2f}%",
    ha="center",
    fontsize=12,
    bbox={"boxstyle": "round,pad=0.5", "facecolor": "lightgray", "alpha": 0.8}
)

columns = list(class_metrics[0].keys())  # Get column names from the first dict
table = wandb.Table(columns=columns)

for row in class_metrics:
    table.add_data(*[row[col] for col in columns])

wandb.log({
            "Prediction Grid": fig,
            "Class Metrics": table
        })

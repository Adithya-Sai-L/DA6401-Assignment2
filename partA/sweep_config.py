sweep_config = {
    'method': 'bayes',
    'metric': {'name': 'val_acc', 'goal': 'maximize'},
    'parameters': {
        'num_filters': {'values': [[32]*5, [64]*5, [16, 32, 64, 128, 256], [256, 128, 64, 32, 16]]},
        'conv_activation': {'values': ['ReLU', 'GELU', 'SiLU', 'Mish']},
        'dense_activation': {'values': ['ReLU', 'Sigmoid', 'Tanh']},
        'dense_neurons': {'values': [512, 256, 128]},
        'batch_normalization': {'values': [True, False]},
        'drop_out': {'min': 0.0, 'max': 0.5},
        'learning_rate': {'min': 1e-5, 'max': 1e-3},
        'optimizer': {'values': ['Adam', 'AdamW', 'NAdam']},

        'batch_size': {'values': [16, 32, 64]},
        'data_augmentation': {'values': [True, False]},
    }
}
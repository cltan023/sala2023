import wandb

project = 'running_time' # project name
config = {
    'program': 'main.py', # the main program that usually recieves the arguments
    'name': 'la_resnet50', # sweep name 
    'method': 'grid',
    'metric': {'goal': 'maximize', 'name': 'best_acc'},
    'parameters': 
    {
        'seed': {'value': 1}, # fixed hyper-parameter 
        'scheduler': {'value': 'cosine'},
        'data_name': {'value': 'CIFAR-100'},
        'num_classes': {'value': 100},
        'arch': {'value': 'resnet50'},
        'optimizer': {'value': 'SGD'},
        # 'use_cutmix': {'value': True},
        'use_la': {'value': True},
        # 'use_sam': {'value': True},
        # 'sam_rho': {'values': [0.1, 0.2]},
        # 'use_sala': {'value': True},
        'alpha': {'value': 0.8},
        'k': {'value': 5},
        # 'sala_rho': {'values': [0.1, 0.2]},
        'learning_rate': {'values': [0.01, 0.05, 0.1]}, # hyper-parameter needs to be tuned
        'batch_size_train': {'value': 128},
        'weight_decay': {'values': [5.0e-4, 1.0e-3, 2.0e-3]}
     }
}

sweep_id = wandb.sweep(sweep=config, project=project)
print(sweep_id)

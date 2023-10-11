# sala2023
SALA: A Novel Optimizer for Accerlating Convergence and Improving Generalization

Tested Environment: Ubuntu 22.04, GPU 4090, Pytorch 1.12, CUDA 11.6 

Requirement: wandb, prefetch_generator, colorama, tqdm  
Also can create a new conda environment with the configuration file environment.yml

Usage:
```
from opt import SALA

# define a base optimizer as in standard training  
base_optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
optimizer = SALA(base_optimizer, sala_k=k, sala_alpha=alpha, sala_rho=rho)

for i in range(num_epochs):
    for batch in train_dataloader:
        ...
        loss.backward()
        optimizer.step()

    # calculate the validation error
    optimizer.backup_and_load_cache() # use cached weights
    validate(...)
    optimizer.clear_and_load_backup() # restore weights to continue training
```
A simple test can be executed by runing:
```
sh run.sh
```

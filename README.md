# sala2023
SALA: A Novel Optimizer for Accerlating Convergence and Improving Generalization

Tested Environment: Ubuntu 22.04, GeForce RTX 4090, Pytorch 1.12, CUDA 11.6 

Requirement: wandb, prefetch_generator, colorama, tqdm  
Also can create a new conda environment with the configuration file environment.yml

Usage:
```
from opt import SALA

# define a base optimizer as in standard training  
base_optimizer = torch.optim.SGD(net.parameters(), lr=0.05)
# by default, we set k=2, alpha=0.5, rho=0.2
optimizer = SALA(base_optimizer, sala_k=k, sala_alpha=alpha, sala_rho=rho)

for i in range(num_epochs):
    for data, target in train_dataloader:
        ...
        def closure():
            loss = train_loss_func(net(data), target)
            loss.backward()
            return loss
                       
        output = net(data)
        loss = train_loss_func(output, target)
        loss.backward()
        optimizer.step(closure)
        ...

    # calculate the validation error
    optimizer.backup_and_load_cache() # use cached weights
    validate(...)
    optimizer.clear_and_load_backup() # restore weights to continue training
```
A simple test can be executed by runing:
```
sh run.sh
```
To tune hyper-parameters via wandb, please run for example
```
python sweep.py
```
that returns something like
```
https://wandb.ai/$your_wandb_username/$project_name/sweeps/$wandb_id
```
Then, you can run a sweep by the following command
```
export CUDA_VISIBLE_DEVICES=0
wandb agent $your_wandb_username/$project_name/$wandb_id
```
To use another GPU on the same server or any other server, just open another terminal, and run for example
```
export CUDA_VISIBLE_DEVICES=1
wandb agent $your_wandb_username/$project_name/$wandb_id
```
Please do not hesitate in contacting me should you have any questions about the implementations (email: cltan023 at outlook.com).

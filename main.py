import numpy as np
import random
import torch
import os
from os.path import join
import argparse
from tqdm import tqdm
from colorama import Fore, Style
import wandb
import json
import math
from prefetch_generator import BackgroundGenerator
import shutil
import time

from models import ResNet18, ResNet50, ResNet101
from models import WideResNet, PyramidNet
from opt import Lookahead
from opt import SAM
from opt import SALA
from data import SoftCrossEntropyLoss
from data import get_cifar_dataloader

def load_checkpoint(model, optimizer, lr_sched, logs, best_acc, filename='checkpoint.pth.tar'):
    # Note: Input model & optimizer should be pre-defined.  This routine only updates their states.
    start_epoch = 0
    if os.path.isfile(filename):
        print("=> loading checkpoint '{}'".format(filename))
        checkpoint = torch.load(filename)
        start_epoch = checkpoint['epoch']
        model.load_state_dict(checkpoint['model'])
        optimizer.load_state_dict(checkpoint['optimizer'])
        lr_sched = checkpoint['lr_sched']
        logs = checkpoint['logs']
        best_acc = checkpoint['best_acc']
        print("=> loaded checkpoint '{}' (epoch {})"
                  .format(filename, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(filename))

    return model, optimizer, start_epoch, lr_sched, logs, best_acc
            
def init_random_state(seed):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False
    
def main():
    parser = argparse.ArgumentParser(description='Accerlating the convergence of the optimization process')
    
    # data configuration
    parser.add_argument('--data_dir', default='./public_dataset', type=str)
    parser.add_argument('--data_name', default='CIFAR-100', choices=['CIFAR-10', 'CIFAR-100', 'ImageNet-1k'])
    parser.add_argument('--num_classes', default=100, type=int)
    parser.add_argument('--batch_size_train', default=128, type=int)
    parser.add_argument('--batch_size_eval', default=256, type=int)
    parser.add_argument('--use_cutout', default=False, type=bool)
    parser.add_argument('--length', default=8, type=int) # 16 for cifar-10, 8 for cifar-100
    parser.add_argument('--use_auto_augment', default=False, type=bool)
    parser.add_argument('--use_rand_augment', default=False, type=bool)
    parser.add_argument('--use_random_erasing', default=False, type=bool)
    parser.add_argument('--use_cutmix', default=False, type=bool)
    parser.add_argument('--cutmix_alpha', default=1.0, type=float)
    parser.add_argument('--use_mixup', default=False, type=bool)
    parser.add_argument('--mixup_alpha', default=1.0, type=float)
    parser.add_argument('--num_workers', default=8, type=int)
    parser.add_argument('--pin_memory', default=True, type=bool)
    
    # optimizer configuration
    parser.add_argument('--optimizer', default='SGD', choices=['SGD', 'Adam', 'AdamW'])
    parser.add_argument('--use_la', default=False, type=bool)
    parser.add_argument('--use_sala', default=False, type=bool)
    parser.add_argument('--use_sam', default=False, type=bool)
    parser.add_argument('--learning_rate', default=0.1, type=float)
    parser.add_argument('--momentum', default=0.9, type=float)
    parser.add_argument('--weight_decay', default=5.0e-4, type=float)
    parser.add_argument('--scheduler', default='cosine', choices=['cosine', 'multistep', 'const'])
    parser.add_argument('--milestones', default=[100, 150], type=int, nargs='+')
    parser.add_argument('--decay_gamma', default=0.1, type=float)
    
    # hyper-parameter for LookAhead and SALA
    parser.add_argument('--k', default=5, type=int)
    parser.add_argument('--alpha', default=0.5, type=float)
    parser.add_argument('--sala_rho', default=0.05, type=float)
    
    # hyper-parameter for SAM
    parser.add_argument('--sam_rho', default=0.05, type=float)
    
    # architecture configuration
    parser.add_argument('--arch', default='resnet18')
    # some other choices
    
    # train configuration
    parser.add_argument('--seed', default=1, type=int)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--log_interval', default=1, type=int)
    parser.add_argument('--save_dir', default='test', type=str)
    parser.add_argument('--project', default=None)
    parser.add_argument('--restart', default=False, type=bool)
    parser.add_argument('--wandb', default=True, type=bool)
    parser.add_argument('--gpu_id', default=0, type=int)
    parser.add_argument('--checkpoint', default='', type=str)
    
    args = parser.parse_args()
    
    device = torch.device(f'cuda:{args.gpu_id}' if torch.cuda.is_available() else 'cpu')
    
    if args.project is None:
        args.project = f'{args.arch}_{args.data_name}_'
        if args.use_cutout:
            args.project += 'cutout'
        elif args.use_auto_augment:
            args.project += 'auto_augment'
        elif args.use_rand_augment:
            args.project += 'rand_augment'
        elif args.use_cutmix:
            args.project += 'cutmix'
        elif args.use_mixup:
            args.project += 'mixup'
        else:
            args.project += 'basic'
    args.project += f'_{args.optimizer}'
    
    if args.use_sam:
        args.instance = f'sam_rho_{args.sam_rho}_'
    elif args.use_la:
        args.instance = f'la_k={args.k}_alpha={args.alpha}_'
    elif args.use_sala:
        args.instance = f'sala_k={args.k}_alpha={args.alpha}_rho={args.sala_rho}_'
    else:
        args.instance = 'vanilla_'
    args.instance += f'lr={args.learning_rate}_bs={args.batch_size_train}_wd={args.weight_decay}_seed={args.seed}'

    timestamp = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    save_dir = join(args.save_dir, args.project, args.instance, timestamp)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    
    # log important files 
    # if not os.path.exists(join(save_dir, 'scripts')):
    #     os.makedirs(join(save_dir, 'scripts'))
    # scripts_to_save = ['main.py', 'run.sh', 'sweep.py', 'opt/sala.py']
    # for script in scripts_to_save:
    #     dst_file = os.path.join(save_dir, 'scripts', os.path.basename(script))
    #     shutil.copyfile(script, dst_file)
    with open(join(save_dir, 'config.json'), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    
    if args.wandb:
        # wandb_run = wandb.init(config=args, project=args.project, name=args.instance, settings=wandb.Settings(code_dir="."))
        wandb_run = wandb.init(config=args, project=args.project, name=args.instance)
        
    init_random_state(args.seed)
    
    train_loader, test_loader = get_cifar_dataloader(args)
    
    if args.arch == 'resnet18':
        net = ResNet18(num_classes=args.num_classes).to(device)
    elif args.arch == 'resnet50':
        net = ResNet50(num_classes=args.num_classes).to(device)
    elif args.arch == 'resnet101':
        net = ResNet101(num_classes=args.num_classes).to(device)
    elif args.arch == 'wideresnet':
        net = WideResNet(depth=28, num_classes=args.num_classes, widen_factor=10).to(device)
    elif args.arch == 'pyramidnet':
        net = PyramidNet(args.data_name, 110, 270, args.num_classes, False).to(device)
        
    optim_param = {'lr': args.learning_rate, 'weight_decay': args.weight_decay}
    if args.optimizer == 'SGD':
        base_optimizer = torch.optim.SGD
        optim_param.update({'momentum': args.momentum, 'nesterov': True})
    elif args.optimizer == 'Adam':
        base_optimizer = torch.optim.Adam
    elif args.optimizer == 'AdamW':
        base_optimizer = torch.optim.AdamW
    else:
        raise NotImplementedError(f'{args.optimizer} is currently not implemented!')
    
    if args.use_sam:
        optim_param.update({'rho': args.sam_rho})
        optimizer = SAM(net.parameters(), base_optimizer, **optim_param)
    elif args.use_la:
        optimizer = Lookahead(base_optimizer(net.parameters(), **optim_param), k=args.k, alpha=args.alpha)
    elif args.use_sala:
        optimizer = SALA(base_optimizer(net.parameters(), **optim_param), sala_k=args.k, sala_alpha=args.alpha, sala_rho=args.sala_rho)
    else:
        optimizer = base_optimizer(net.parameters(), **optim_param)
    
    if args.scheduler == 'cosine':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
    elif args.scheduler == 'multistep':
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=args.milestones, gamma=args.decay_gamma)
    elif args.scheduler == 'const':
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, args.epochs)
        print('use constant learning rate, be careful!')
    else:
        raise NotImplementedError(f'{args.scheduler} is currently not implemented!')
    
    if args.use_mixup or args.use_cutmix:
        train_loss_func = SoftCrossEntropyLoss(reduction='mean')
    else:
        train_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    test_loss_func = torch.nn.CrossEntropyLoss(reduction='mean')
    
    if args.use_sam:
        batch_updater = sam_batch_updater
    elif args.use_sala:
        batch_updater = sam_batch_updater
    else:
        batch_updater = basic_batch_updater
    
    start_epoch = 0    
    best_acc = 0.0
    logs = []
    if args.checkpoint != '':
        net, optimizer, start_epoch, scheduler, logs, best_acc = load_checkpoint(net, optimizer, scheduler, logs, best_acc, filename=args.checkpoint)
    
    with tqdm(total=args.epochs, colour='MAGENTA', ascii=True) as pbar:
        for epoch in range(start_epoch+1, args.epochs+1):
            train_loss, train_acc = trainer(net, optimizer, batch_updater, train_loader, train_loss_func, device)
            if args.scheduler != 'const':
                scheduler.step()
            
            if epoch % args.log_interval == 0:
                if args.use_sala:
                    optimizer.backup_and_load_cache()
                elif args.use_la:
                    optimizer.sync_lookahead()
                    
                test_loss, test_acc = validate(net, test_loader, test_loss_func, device)
                    
                logs.append([train_acc, test_acc, train_loss, test_loss])
                
                if best_acc <= test_acc:
                    best_acc = test_acc
                    # uncomment to save model weights in order for estimation of Hessian
                    # checkpoint = { 
                    #     'epoch': epoch,
                    #     'model': net.state_dict(),
                    #     'optimizer': optimizer.state_dict(),
                    #     'lr_sched': scheduler,
                    #     'logs': logs,
                    #     'best_acc': best_acc}
                    # if not os.path.exists(join(save_dir, 'checkpoints')):
                    #     os.makedirs(join(save_dir, 'checkpoints'))
                    # torch.save(checkpoint, join(save_dir, 'checkpoints', f'checkpoint-best.pth.tar'))
                if args.use_sala:
                    optimizer.clear_and_load_backup()
                
                if args.wandb:
                    if args.use_sala:
                        wandb_run.log({'train_acc': train_acc*100, 'test_acc': test_acc*100, 'train_loss': train_loss, 'test_loss': test_loss, 'best_acc': best_acc*100, 'lr': scheduler.get_last_lr()[0], 'param_dist': optimizer.param_dist})
                    else:
                        wandb_run.log({'train_acc': train_acc*100, 'test_acc': test_acc*100, 'train_loss': train_loss, 'test_loss': test_loss, 'best_acc': best_acc*100, 'lr': scheduler.get_last_lr()[0]})
                    
                message = f'epoch: {epoch} '
                message += f'lr: {scheduler.get_last_lr()[0]:.6f} '
                message += f'train_loss: {Fore.RED}{train_loss:.4f}{Style.RESET_ALL} '
                message += f'train_acc: {Fore.RED}{train_acc*100:.2f}%{Style.RESET_ALL} '
                message += f'test_loss: {Fore.GREEN}{test_loss:.4f}{Style.RESET_ALL} '
                message += f'test_acc: {Fore.GREEN}{test_acc*100:.2f}%{Style.RESET_ALL} '
                message += f'best_acc: {Fore.MAGENTA}{best_acc*100:.2f}%{Style.RESET_ALL} '
                
                pbar.set_description(message)
                pbar.update()
                
        logs = torch.tensor(logs)
        torch.save(logs, join(save_dir, 'logs.pt'))
                 
def trainer(net, optimizer, batch_updater, train_loader, train_loss_func, device):
    net.train()
    tot_loss = 0.0
    tot_correct = 0.0
    for i, batch in BackgroundGenerator(enumerate(train_loader)):
        loss, correct = batch_updater(net, optimizer, batch, train_loss_func, device)
        tot_loss += loss / len(train_loader.dataset)
        tot_correct += correct / len(train_loader.dataset)
    return tot_loss, tot_correct

def validate(net, test_loader, test_loss_func, device):
    net.eval()
    tot_loss = 0.0
    tot_correct = 0.0
    with torch.no_grad():
        for data, target in BackgroundGenerator(test_loader):
            data, target = data.to(device), target.to(device)
            output = net(data)
            tot_loss += test_loss_func(output, target) * len(output)
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            tot_correct += pred.eq(target.view_as(pred)).sum().item()
    return tot_loss / len(test_loader.dataset), tot_correct / len(test_loader.dataset)

def basic_batch_updater(net, optimizer, batch, train_loss_func, device):
    optimizer.zero_grad()
    data, target = batch
    if not isinstance(target, (tuple, list)):
        target = target.to(device)        
    output = net(data.to(device))
    loss = train_loss_func(output, target)
    loss.backward()
    optimizer.step()    
    _, preds = torch.max(output.data, 1)
    if isinstance(target, (tuple, list)):
        targets1, targets2, lam = target
        targets1, targets2 = targets1.to(device), targets2.to(device)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        correct = lam * correct1 + (1 - lam) * correct2
    else:
        correct = preds.eq(target).sum().item()
    
    return loss.item() * len(data), correct

def sam_batch_updater(net, optimizer, batch, train_loss_func, device):
    optimizer.zero_grad()
    data, target = batch
    if not isinstance(target, (tuple, list)):
        target = target.to(device)
        
    def closure():
        loss = train_loss_func(net(data.to(device)), target)
        loss.backward()
        return loss.item()
                   
    output = net(data.to(device))
    loss = train_loss_func(output, target)
    loss.backward()
    optimizer.step(closure)
    
    _, preds = torch.max(output.data, 1)
    if isinstance(target, (tuple, list)):
        targets1, targets2, lam = target
        targets1, targets2 = targets1.to(device), targets2.to(device)
        correct1 = preds.eq(targets1).sum().item()
        correct2 = preds.eq(targets2).sum().item()
        correct = lam * correct1 + (1 - lam) * correct2
    else:
        correct = preds.eq(target).sum().item()
    
    return loss.item() * len(data), correct
            
if __name__ == '__main__':
    main()

import torch
import torchvision
import torch.nn.functional as F
from torchvision import datasets, transforms
import os

from .cutout import Cutout
from .cutmix import CutMixCollator
from .mixup import MixUpCollator

def get_imagenet_dataloader(args):
    train_dir = os.path.join(args.data_dir, 'train')
    test_dir = os.path.join(args.data_dir, 'val')
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    basic_transform = [transforms.RandomResizedCrop(224),transforms.RandomHorizontalFlip()]

    if args.use_auto_augment:
        basic_transform.append(transforms.AutoAugment(transforms.AutoAugmentPolicy.IMAGENET))
    if args.use_rand_augment:
        basic_transform.append(transforms.RandAugment(num_ops=2, magnitude=9))
    if args.use_random_erasing:
        basic_transform.append(transforms.RandomErasing(p=0.25))
        
    basic_transform += [transforms.ToTensor(), normalize]
    if args.use_cutout:
        basic_transform.append(Cutout(n_holes=1, length=args.length))
        
    train_transform = transforms.Compose(basic_transform)
    test_transform = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        normalize,
    ])
    
    if args.use_cutmix:
        collator = CutMixCollator(args.cutmix_alpha)
    elif args.use_mixup:
        collator = MixUpCollator(args.mixup_alpha)
    else:
        collator = torch.utils.data.dataloader.default_collate
    
    train_dataset = datasets.ImageFolder(train_dir, train_transform)
    test_dataset = datasets.ImageFolder(test_dir, test_transform)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=args.batch_size_train,
        shuffle=True,
        collate_fn=collator,
        num_workers=args.num_workers,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=args.batch_size_eval,
        num_workers=args.num_workers,
        shuffle=False,
        pin_memory=args.pin_memory,
        drop_last=False,
    )
    return train_loader, test_loader
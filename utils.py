import torch
import torchvision
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder
import sys, os
from pathlib import Path
file_dir = Path(__file__).parent
sys.path.append(str(file_dir))
from tiny_imagenet import TinyImageNet


cifar10_mean = (0.4914, 0.4822, 0.4465)
cifar10_std = (0.2471, 0.2435, 0.2616)

def data_prepare(args):

    transform_train =transforms.Compose([
            transforms.Pad(4, padding_mode='reflect'),
            transforms.RandomCrop(32),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

    transform_test =transforms.Compose([
        transforms.ToTensor()
    ])


    if args.dataset == "CIFAR10":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_train)
            ,batch_size=args.batch_size, shuffle=True, num_workers=8
            )
        train_loader_eval = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=args.data_root, train=True, download=True, transform=transform_test)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
            )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10(root=args.data_root, train=False, download=True, transform=transform_test)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
            )
    
    elif args.dataset == "CIFAR100":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_train)
            ,batch_size=args.batch_size, shuffle=True, num_workers=8
        )
        train_loader_eval = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=args.data_root, train=True, download=True, transform=transform_test)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100(root=args.data_root, train=False, download=True, transform=transform_test)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
        )
    
    if args.dataset == "MNIST":
        transform = transforms.Compose([
            transforms.ToTensor()
        ])
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
            , batch_size=args.batch_size, shuffle=True, num_workers=8
        )
        train_loader_eval = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=args.data_root, train=True, download=True, transform=transform)
            ,batch_size=args.batch_size, shuffle=False, num_workers=8
        )
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root=args.data_root, train=False, download=True, transform=transform)
            , batch_size=args.batch_size, shuffle=False, num_workers=8
        )
    
    if args.dataset == "svhn":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=args.data_root, split='train', download=True, transform=transform_test)
            , batch_size=args.batch_size, shuffle=True, num_workers=8
        )

        train_loader_eval = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=args.data_root, split='train', download=True, transform=transform_test)
            , batch_size=args.batch_size, shuffle=False, num_workers=8
        )

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root=args.data_root, split='test', download=True, transform=transform_test)
            , batch_size=args.batch_size, shuffle=False, num_workers=8
        )
    
    if args.dataset == "TinyImageNet":
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

        transform_test = transforms.Compose([
            # transforms.Resize(64),  
            transforms.ToTensor()
        ])

        train_loader = torch.utils.data.DataLoader(
            TinyImageNet(args.data_root, split='train',  transform=transform_train, download=True),
            batch_size=args.batch_size, shuffle=True, num_workers=8
        )

        train_loader_eval = torch.utils.data.DataLoader(
            TinyImageNet(args.data_root, split='train',  transform=transform_test, download=True),
            batch_size=args.batch_size, shuffle=False, num_workers=8
        )
        
        test_loader = torch.utils.data.DataLoader(
            TinyImageNet(args.data_root, split='val',  transform=transform_test, download=True),
            batch_size=args.batch_size, shuffle=False, num_workers=8
        )

    if args.dataset == "CINIC10":
        cinic_root = args.data_root

        train_loader = torch.utils.data.DataLoader(
            ImageFolder(root=os.path.join(cinic_root, 'train'), transform=transform_train),
            batch_size=args.batch_size, shuffle=True, num_workers=8
        )

        train_loader_eval = torch.utils.data.DataLoader(
            ImageFolder(root=os.path.join(cinic_root, 'train'), transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=8
        )

        test_loader = torch.utils.data.DataLoader(
            ImageFolder(root=os.path.join(cinic_root, 'test'), transform=transform_test),
            batch_size=args.batch_size, shuffle=False, num_workers=8
        )
    
    if args.dataset in ["Imagenette", "ImageNet"] :
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor()
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])

        train_dir = os.path.join(args.data_root, 'train')
        val_dir = os.path.join(args.data_root, 'val')

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_train)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=8)
        train_loader_eval = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)

      
    
    if args.phase == 'eval':
        return train_loader_eval, test_loader
    else:
        return train_loader, test_loader








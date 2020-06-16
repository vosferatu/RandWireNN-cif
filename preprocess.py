import torch

from torchvision import datasets, transforms


# CIFAR-10,
# mean, [0.4914, 0.4822, 0.4465]
# std, [0.2470, 0.2435, 0.2616]
# CIFAR-100,
# mean, [0.5071, 0.4865, 0.4409]
# std, [0.2673, 0.2564, 0.2762]
def load_data(args):
    train_loader = None
    test_loader = None
    if args.dataset_mode == "CIFAR10":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR10('data/cifar10', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
    elif args.dataset_mode == "CIFAR100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        transform_test = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data/cifar100', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
        test_loader = torch.utils.data.DataLoader(
            datasets.CIFAR100('data/cifar100', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2
        )
    elif args.dataset_mode == "MNIST":
        transform_train = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.MNIST('data/mnist', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
    elif args.dataset_mode == "FASHION_MNIST":
        transform_train = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])

        transform_test = transforms.Compose([
            transforms.Grayscale(3),
            transforms.ToTensor(),
            transforms.Normalize((0.1307,), (0.3081,)),
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fashion_mnist', train=True, download=True, transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('data/fashion_mnist', train=False, transform=transform_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2
        )
    elif args.dataset_mode == "IMAGENET":
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                         std=[0.229, 0.224, 0.225])
        transform_train = transforms.Compose([
            transforms.RandomResizedCrop(224),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            normalize,
        ])
        train_loader = torch.utils.data.DataLoader(
            datasets.ImageNet('/disk/two/imagenet/', split='train', transform=transform_train),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

        test_loader = torch.utils.data.DataLoader(
            datasets.FashionMNIST('/disk/two/imagenet/', split='val', transform=transform_test),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=4
        )

    return train_loader, test_loader

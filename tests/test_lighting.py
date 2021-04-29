def test_litvae():
    import os
    import torch
    from tempfile import gettempdir
    from torchvision import transforms, datasets
    from torch.utils.data import random_split, DataLoader

    from torchvae.lighting import LitVAE as VAE

    tmp = gettempdir()
    batch_size = 128

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0,), (1,)),
        transforms.Lambda(lambda x: torch.flatten(x))
    ])

    root = '~/.data/'
    mnist_train_val = datasets.MNIST(root, download=True, train=True, transform=transform)

    # Load train and validation sets
    loader_args = dict(batch_size=128, pin_memory=True)
    mnist_train, mnist_val = random_split(mnist_train_val, [50000, 10000])
    train_loader = DataLoader(mnist_train, shuffle=True, **loader_args)
    val_loader = DataLoader(mnist_val, **loader_args)

    model = VAE(in_features=784, hidden_features=[400, 5])

    from torchvae.lighting import train_vae
    train_vae(model, min_epochs=5, max_epochs=100,
              train_loader=train_loader, val_loader=val_loader,
              logger=True, logger_path=tmp)
    print('Results of the model can be found at:', tmp)


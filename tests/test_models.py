def test_forward():
    import torch
    from torchvae.models import VAE
    x = torch.rand(1, 784)
    model = VAE(in_features=784, hidden_features=[256, 3])
    model(x)
    x_recon = model.reconstruct(x)

    assert x_recon.shape == x.shape

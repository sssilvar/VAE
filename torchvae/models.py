import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class Encoder(nn.Module):
    def __init__(self, in_features, hidden_features=2):
        super(Encoder, self).__init__()

        self.mu_enc = nn.Linear(in_features=in_features, out_features=hidden_features)
        self.logvar2_enc = nn.Linear(in_features=in_features, out_features=hidden_features)

    def forward(self, x):
        mu = self.mu_enc(x)
        sigma = self.logvar2_enc(x).exp().sqrt()
        return Normal(mu, sigma)


class Decoder(nn.Module):
    def __init__(self, hidden_features, out_features):
        super(Decoder, self).__init__()

        self.mu_dec = nn.Linear(hidden_features, out_features)
        self.logvar2_dec = nn.Parameter(torch.Tensor(out_features))

        nn.init.uniform_(self.logvar2_dec, 1, math.e)

    def forward(self, z):
        return Normal(self.mu_dec(z), self.logvar2_dec.exp().sqrt())


class VAE(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(VAE, self).__init__()

        if isinstance(hidden_features, list):
            hidden_features.insert(0, in_features)

            # [784, 128, 10, 2] >> [784, 128] >> [128, 10]
            n_features = zip(hidden_features[:-2], hidden_features[1:-1])
            self.hidden_layers_enc = (nn.Sequential(nn.Linear(i, o), nn.ReLU()) for i, o in n_features)

            self.encode = nn.Sequential(*self.hidden_layers_enc, Encoder(hidden_features[-2], hidden_features[-1]))

            hidden_features.reverse()
            n_features = zip(hidden_features[:-2], hidden_features[1:-1])
            self.hidden_layers_dec = (nn.Sequential(nn.Linear(i, o), nn.ReLU()) for i, o in n_features)
            self.decode = nn.Sequential(*self.hidden_layers_dec, Decoder(hidden_features[-2], hidden_features[-1]))

        else:
            self.encode = Encoder(in_features, hidden_features)
            self.decode = Decoder(hidden_features, in_features)

        self.kld = None

    def forward(self, x):
        z = self.encode(x)
        self.kld = kl_divergence(z, Normal(0, 1)).sum()

        return self.decode(z.rsample())

    def reconstruct(self, x):
        z = self.encode(x).loc
        return self.decode(z).loc

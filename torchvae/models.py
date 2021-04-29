import math
import torch
import torch.nn as nn
from torch.distributions import Normal, kl_divergence


class VAE(nn.Module):
    def __init__(self, in_features, hidden_features):
        super(VAE, self).__init__()

        self.hidden_features = hidden_features

        if self.is_multilayer:
            hidden_features.insert(0, in_features)

            # Encoder
            n_features = zip(hidden_features[:-2], hidden_features[1:-1])
            self.hidden_layers_enc = nn.Sequential()
            for i, (hid_in, hid_out) in enumerate(n_features):
                self.hidden_layers_enc.add_module(f'enc_hid_{i}', nn.Linear(hid_in, hid_out))
                self.hidden_layers_enc.add_module(f'act_{i}', nn.ReLU())

            self.enc_mu = nn.Linear(hidden_features[-2], hidden_features[-1])
            self.enc_logvar = nn.Linear(hidden_features[-2], hidden_features[-1])

            # Decoder
            hidden_features.reverse()
            n_features = zip(hidden_features[:-2], hidden_features[1:-1])
            self.hidden_layers_dec = nn.Sequential()
            for i, (hid_in, hid_out) in enumerate(n_features):
                self.hidden_layers_dec.add_module(f'dec_hid_{i}', nn.Linear(hid_in, hid_out))
                self.hidden_layers_dec.add_module(f'act_{i}', nn.ReLU())

            self.dec_mu = nn.Linear(hidden_features[-2], hidden_features[-1])
            self.dec_logvar = nn.Parameter(torch.Tensor(hidden_features[-1]))

        else:
            # Single-layer encoder
            self.enc_mu = nn.Linear(in_features, self.hidden_features)
            self.enc_logvar = nn.Linear(in_features, self.hidden_features)

            # single-layer decoder
            self.dec_mu = nn.Linear(hidden_features, in_features)
            self.dec_logvar = nn.Parameter(torch.Tensor(in_features))

        self.kld = None
        nn.init.uniform_(self.dec_logvar, 1, math.e)

    @property
    def is_multilayer(self):
        if isinstance(self.hidden_features, (tuple, list)):
            if len(self.hidden_features) == 1:
                self.hidden_features = self.hidden_features[0]
                return False
            else:
                return True
        return False

    def encode(self, x):
        if self.is_multilayer:
            x = self.hidden_layers_enc(x)
        return Normal(self.enc_mu(x), self.enc_logvar(x).exp().sqrt())

    def decode(self, x):
        if self.is_multilayer:
            x = self.hidden_layers_dec(x)
        return Normal(self.dec_mu(x), self.dec_logvar.exp().sqrt())

    def forward(self, x):
        z = self.encode(x)
        self.kld = kl_divergence(z, Normal(0, 1)).sum()

        return self.decode(z.rsample())

    def reconstruct(self, x):
        z = self.encode(x).loc
        return self.decode(z).loc

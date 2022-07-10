#!/usr/bin/env python
# coding: utf-8

# ***
# ## Generative Model - VAE
# 
# Experiment implementing a **MultiLayer Perceptron Variational AutoEncoder** as generative model of data needed for the Generative Replay technique.
# 
# The Variational AutoEncoder (VAE) is an architecture composed of an encoder, a decoder and a loss function, that is trained to minimize the reconstruction error between the encoded-decoded data and the initial data.
# 
# *Code is based in part on the work:*
#  - https://github.com/TLESORT/Generative_Continual_Learning 
#  - https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html#vae
# ***

# In[1]:


# --- LIBRARIES AND UTILS ---
from abc import abstractmethod

import torch
import torch.nn as nn
from torchvision.transforms import Compose, Normalize

import avalanche
from avalanche.models import MLP, Flatten, BaseModel


# ### Encoder
# Encoder part of the VAE, it is a fully connected neural network that takes in an input and generates a much smaller, dense representation (encoding) specifically useful for reconstructing its own input by mean of a decoder.
# 
# #### Input parameters:
#  - shape = shape of the network input (1, height, width)
#  - latent_dim = dimension of the last hidden layer

# In[2]:


class VAE_encoder(nn.Module):
    
    def __init__(self, shape, latent_dim=128):
        super(VAE_encoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        self.encode = nn.Sequential(
            Flatten(),      # nn.Module to flatten each tensor of a batch of tensors
            nn.Linear(in_features=flattened_size, out_features=400), # single layer MLP
            nn.BatchNorm1d(400),
            nn.LeakyReLU(), # no vanishing gradient
            MLP([400, latent_dim], last_activation=True), # MLP with BatchNorm and Relu activations
        )

    def forward(self, x, y=None):
        x = self.encode(x)
        return x


# ### Decoder
# Decoder part of the VAE, it has the same network structure of the encoder; it takes the output of the encoder and attempts to recreate an output identical to the input.
# 
# #### Input parameters:
#  - shape = shape of the network output (1, height, width)
#  - nhid = dimension of the input

# In[3]:


class VAE_decoder(nn.Module):

    def __init__(self, shape, nhid=16):
        super(VAE_decoder, self).__init__()
        flattened_size = torch.Size(shape).numel()
        self.shape = shape
        self.decode = nn.Sequential(
            MLP([nhid, 64, 128, 256, flattened_size], last_activation=False),
            nn.Sigmoid(),
        )
        self.invTrans = Compose([Normalize((0.1307,), (0.3081,))])

    def forward(self, z, y=None):
        if y is None:
            return self.invTrans(self.decode(z).view(-1, *self.shape))
        else:
            return self.invTrans(
                self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape)
            )


# ***
# 
# ### Variational AutoEncoder module and loss
# Fully connected Variational Autoencoder
# 
# #### Input parameters:
#  - shape = shape of each input sample
#  - nhid = dimension of the latent space of the Encoder
#  - num_classes = number of classes, it defines the classification head's dimension

# In[4]:


##### base abstract class for generators
class Generator(BaseModel):

    @abstractmethod
    def generate(self, batch_size=None, condition=None):
        """
        Lets the generator sample random samples.
        """


class VAE_model(Generator, nn.Module):

    def __init__(self, shape, nhid=16, n_classes=10, device="cpu"):
        
        super(VAE_model, self).__init__()
        self.dim = nhid
        self.device = device
        self.encoder = VAE_encoder(shape, latent_dim=128)
        self.calc_mean = MLP([128, nhid], last_activation=False)
        self.calc_logvar = MLP([128, nhid], last_activation=False)
        self.classification = MLP([128, n_classes], last_activation=False)
        self.decoder = VAE_decoder(shape, nhid)

    # Get features for encoder part given input x
    def get_features(self, x):

        return self.encoder(x)

    # Random samples generator
    def generate(self, batch_size=None): # :param batch_size = samples to generate

        z = (torch.randn((batch_size, self.dim)).to(self.device) # batch of samples of size "batch_size"
            if batch_size
            else torch.randn((1, self.dim)).to(self.device))     # if batch_size = None, single sample
        
        res = self.decoder(z)
        
        if not batch_size:
            res = res.squeeze(0)
        return res
    
    # Reparametrization trick
    def sampling(self, mean, logvar):
        
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):

        representations = self.encoder(x)
        mean, logvar = self.calc_mean(representations), self.calc_logvar(representations)
        z = self.sampling(mean, logvar)
        return self.decoder(z), mean, logvar


# ### Loss function
# Loss function of VAE using mean squared error for reconstruction loss (used for VAE training)
# 
# #### Input parameters:
#  - X = input batch
#  - forward_output = [reconstructed input after autoencoder, mean of the VAE output distribution, logvar of the VAE output distribution]

# In[5]:


MSE_loss = nn.MSELoss(reduction="sum")# Criterion that measures mean squared error (L2 norm)

def VAE_loss(X, forward_output):

    X_hat, mean, logvar = forward_output
    reconstruction_loss = MSE_loss(X_hat, X)
    KL_divergence = 0.5 * torch.sum(-1 - logvar + torch.exp(logvar) + mean ** 2)
    return reconstruction_loss + KL_divergence


# ***
# List of strings defining what symbols are exported in the module

# In[6]:


__all__ = ["VAE_model", "VAE_loss"]

#!/usr/bin/env python
# coding: utf-8

# ***
# ## Generative Model - VAE
# 
# Experiment implementing a **fully-connected Variational AutoEncoder** as generative model of data needed for the Generative Replay technique.
# 
# The Variational AutoEncoder (VAE) is an architecture composed of an encoder, a decoder and a loss function, that is trained to minimize the reconstruction error between the encoded-decoded data and the initial data.
# 
# *Code is based in part on the work:*
#  - https://blog.keras.io/building-autoencoders-in-keras.html
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


# ***
# ### Encoder
# Encoder part of the VAE, it is a fully connected neural network that takes in an input and generates a much smaller, dense representation (encoding) specifically useful for reconstructing its own input by mean of a decoder.
# 
# #### Input parameters:
#  - shape = shape of the network input (channels, height, width) of the handled images
#  - latent_dim = dimension of the latent space Z

# In[2]:


class VAE_encoder(nn.Module):
    
    def __init__(self, shape, latent_dim=128):
        super(VAE_encoder, self).__init__()
        flattened_size = torch.Size(shape).numel() # original input shape
        self.encode = nn.Sequential(
            Flatten(),      # nn.Module to flatten each tensor of a batch of tensors
            nn.Linear(in_features=flattened_size, out_features=400), # 1st layer (image -> 400 units)
            nn.BatchNorm1d(400),
            nn.LeakyReLU(), # no vanishing gradient
            MLP([400, latent_dim], last_activation=True), # 2nd layer (400 units -> Z latent space)
        )

    # Encoder maps an input-vector X to a vector of latent variables Z 
    def forward(self, x, y=None):
        x = self.encode(x)
        return x


# ***
# ### Decoder
# Decoder part of the VAE, it has the same network structure of the encoder; it takes the output of the encoder and attempts to recreate an output identical to the input.
# 
# #### Input parameters:
#  - shape = shape of the network output: (channels, height, width) of the handled images
#  - nhid = parameters in the latent space (mean and logvar)

# In[3]:


class VAE_decoder(nn.Module):

    def __init__(self, shape, nhid=16):
        super(VAE_decoder, self).__init__()
        flattened_size = torch.Size(shape).numel() # original input shape
        self.shape = shape
        self.decode = nn.Sequential(
            MLP([nhid, 400, 400, flattened_size], last_activation=False),
            #MLP([nhid, 64, 128, 256, flattened_size], last_activation=False),
            nn.Sigmoid(),
        )
        self.invTrans = Compose([Normalize((0.1307,), (0.3081,))])

    # Decoder maps the latent variables Z to a reconstructed original input data  
    def forward(self, z, y=None):
        if y is None:
            return self.invTrans(self.decode(z).view(-1, *self.shape))
        else:
            return self.invTrans(self.decode(torch.cat((z, y), dim=1)).view(-1, *self.shape))


# ***
# 
# ### Variational AutoEncoder module and loss
# Fully connected Variational Autoencoder
# 
# #### Input parameters:
#  - shape = shape of each input sample (channels, height, width) of the handled images
#  - nhid = parameters in the latent space

# In[4]:


##### base abstract class for generators
class Generator(BaseModel):

    @abstractmethod
    def generate(self, batch_size=None, condition=None):
        """
        Lets the generator sample random samples.
        """


class VAE_model(Generator, nn.Module):

    def __init__(self, shape, nhid=16, device="cpu"):
        
        super(VAE_model, self).__init__()
        self.dim = nhid
        self.device = device
        # ENCODER
        self.encoder = VAE_encoder(shape, latent_dim=128)
        # parameters in the latent space
        self.calc_mean = MLP([128, nhid], last_activation=False)
        self.calc_logvar = MLP([128, nhid], last_activation=False)
        # DECODER
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
    
    # Reparametrization trick (Z = mean + eps*sigma)
    def sampling(self, mean, logvar):
        
        eps = torch.randn(mean.shape).to(self.device)
        sigma = 0.5 * torch.exp(logvar)
        return mean + eps * sigma

    def forward(self, x):
        # map input samples to latent distribution parameters
        representations = self.encoder(x)
        mean, logvar = self.calc_mean(representations), self.calc_logvar(representations)
        # randomly samples similar points from the latent space
        z = self.sampling(mean, logvar)
        # outputs the reconstructed samples from the latent space points
        return self.decoder(z), mean, logvar


# ### Loss function
# Loss function of VAE (*theory:* https://gregorygundersen.com/blog/2018/04/29/reparameterization/)
# 
# The parameters of the model are trained via two loss functions:
#  - **Reconstruction loss:** forcing the decoded samples to match the initial inputs
#  - **Variational loss:** KL divergence between the learned latent distribution and the prior distribution, acting as a regularization term.
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


## Description

### INTRO
**Deep generative replay framework** that generates fake inputs from learned past input distribution to retain knowledge without revisiting actual past data. The network in fact is jointly optimized using an ensemble of generated past data and real current data.

Performance depend on the quality of the generator and its ability to recover the input distribution.

### 1. Generative Model
Hereby it is considered a variational autoencoder deep generative model based on a fully connected neural network that is able to mimic complex samples like images. The term generative model refers to any model that generates observable samples.

**The Variational AutoEncoder (VAE)** is an architecture composed of an encoder, a decoder and a loss function, that is trained to minimize the reconstruction error between the encoded-decoded data and the initial data:
<div align="center">

<img src="autoencoder.png" alt="drawing" style="width:700px;"/>
</div> 

Reference script: *GenerativeModel.py*

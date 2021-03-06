## Description

### INTRO
**Deep generative replay framework** that generates fake inputs from learned past input distribution to retain knowledge without revisiting actual past data. The network in fact is jointly optimized using an ensemble of generated past data and real current data.

The generative replay strategy implemented relies on the framework proposed by [Shin et al.](https://arxiv.org/abs/1705.08690?context=cs), which consists of a main model for solving the tasks and a separate generative model trained to generate the data to be replayed:
<div align="center">

<img src="framework.png" alt="drawing" style="width:500px;"/>
</div>

### 1. Generative Model
Hereby it is considered a variational autoencoder deep generative model based on a fully connected neural network that is able to mimic complex samples like images. The term generative model refers to any model that generates observable samples.

**The Variational AutoEncoder (VAE)** is an architecture composed of an encoder, a decoder and a loss function, that is trained to minimize the reconstruction error between the encoded-decoded data and the initial data:
<div align="center">

<img src="VAE_schematic.png" alt="drawing" style="width:350px;"/>
</div> 

The VAE is symmetric, the architectures are similar to the base neural network: both the encoder and the decoder are fully connected NNs with two hidden layers containing units with ReLU non-linearity. To train the generator, the same hyperparameters (optimizer, batch sizes, learning rate) are used as for training the main model.

How does the VAE work?
<div align="center">

<img src="VAE_Architecture.png" alt="drawing" style="width:450px;"/>
</div> 

- **Autoencoder** maps the input samples to latent distribution parameters z<sub>mean</sub> and z<sub>log-sigma</sub> ; 
- **Encoder** randomly samples similar points z from the latent space;
- **Decoder** takes these latent space points and outputs the corresponding reconstructed samples.

Reference script: *GenerativeModel.py*. Any kind of generative model can be used, the script utilizes a VAE generator.  
Performance depend on the quality of the generator and its ability to recover the input distribution.

### 2. Generator
The **Generator** for the scholar model is wrapped into a trainable strategy, to be passed to the generator_strategy parameter when using Generative Replay plugin of Avalanche. The data to be replayed is produced by first sampling the inputs from the generative model, whereafter those generated inputs are presented to the main model.

The key idea behind the estimator utilized in the variational autoencoder generator ([intro to VAEs](https://arxiv.org/abs/1906.02691)) is to sample latent space Z using the reparametrization trick and then condition on Z to generate a realistic sample of X.  
Concretely, for each pass through the computational graph:
<div align="center">

<img src="VAE_ComputationalGraph.png" alt="drawing" style="width:400px;"/>
</div>

Every variable in this computational graph is computed through a sequence of differentiable operations, so backpropagation is used to compute the required gradients:
<div align="center">

<img src="VAE_Diagram.png" alt="drawing" style="width:450px;"/>
</div>

Reference script: *Generator.py*. Script serves as a test to visualize the samples generated by the VAE generative model.

### 3. Training Instances
Instances of the **training strategies** for the VAE generative model and Generative Replay, based on the SupervisedTemplate of Avalanche, which is a basic training loop and callback system that allows to execute code at each experience of the training loop.

Reference script: *TrainingInstances.py*.

### 4. Generative Replay
Experiment reproducing Generative Replay strategy: Deep Generative Replay for a Scholar consisting of a Solver and Generator.  
The strategy relies on the *GenerativeReplayPlugin* of Avalanche. 

The scholar model is built with a solver that has suitable architecture for solving a task sequence and a generator trained using a variational autoencoder. However, this framework can employ any deep generative model as a generator.  
The strategy sequentially trains the neural network on all tasks of the task protocol, whereby the network has access only to the data of the current task: the current minibatch of the strategy is updated before training an experience by *sampling the generator* and *concatenating* the replay data to the current batch.

<div align="center">

<img src="training.png" alt="drawing" style="width:500px;"/>
</div> 

The parameters to be used for the training strategy are:
 - replay_size: batch size of replay added to each data batch;
 - increasing_replay_size: if set to True, each experience this will double the amount of replay data added to each data batch.

Reference script: *GenerativeReplay.py*.

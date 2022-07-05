<div align="center">
    
# Generative Replay for Continual Learning
</div>
Repository that is intended to keep track of the work for the project of the course Embedded Systems and that contains all the reference material and the scripts developed to this end.

### PROJECT in two lines
The presented work aims at providing interesting improvements to replay and memory-based approaches that store past data to alleviate catastrophic forgetting in the field of **Continual Learning**, by performing extensive experimental evaluation on the novel framework of **Generative Replay**. 

I propose an alternative approach to sequentially train NNs without referring to past data. In this generative replay framework in fact, the model retains previously acquired knowledge by training a deep generative model (VAE) to mimic past data to be paired with corresponding response from the past task solver to represent old tasks.

The model can be applied to **any** task as long as the trained generator reliably reproduces the input space.

References                                             | link         
-------------------------------------------------------|---------------------------------
Continual Learning with Deep Generative Replay              | https://arxiv.org/abs/1705.08690
Brain-inspired replay for Continual Learning                | https://www.nature.com/articles/s41467-020-17866-2
An investigation of Replay-based approaches for Continual Learning | https://arxiv.org/abs/2108.06758
Generative Negative Replay for Continual Learning           | https://openreview.net/forum?id=MWQCPYSJRN

**In the context of AI-powered edge devices**, this work also reports the trade-off between memory footprint, latency, and accuracy for learning a new class with Generative Replay when targeting an image classification task on the MNIST dataset; Memory and computational requirements of the Continual Learning algorithm are first quantified and later evaluated in a simulated environment but from a platform-perspective:
- Memory evaluation
- Latency
- Accuracy
- Energy estimation

References                                             | link         
-------------------------------------------------------|---------------------------------
Continual Learning on a RISC-V Node                    | https://arxiv.org/abs/2007.13631
On-Device Training under 256KB memory                  | https://arxiv.org/abs/2206.15472

***

### Avalanche - *end-to-end library for Continual Learning*
Open-source end-to-end library for continual learning based on Pytorch, devised to ease the implementation, assessment and replication of continual learning algorithms across different settings.

<div align="center">
  
**[Avalanche Website](https://avalanche.continualai.org)** | **[Avalanche Repository](https://github.com/ContinualAI/avalanche)**

<img src="avalanche_library.png" alt="drawing" style="width:400px;"/>
</div>

All the code is based on the `Avalanche` framework and is publicy available.
***

## Action plan

### 1. EXPERIMENT
Provide a set of experiments reproducing **Generative Replay strategy** in Continual Learning.
Scripts are (so far) organized as follows:
 - *Neural Network Architecture*
   - MultiLayer Perceptron (MLP): *MLP_NNmodel.py* defines the neural network utilized for training and evaluating the CL technique; 
 - *Generative Model* 
   - Variational AutoEncoder (VAE): *GenerativeModel.py* defines the VAE generative model of data;
 - *Generator Strategy*
   - *GeneratorStrategy.py* wraps the VAE generator in a trainable strategy, later passed to the generator_strategy parameter;
 - *Learning Strategy*
   - Generative Replay: *GenerativeReplayStrategy.py* implements the Continual Learning strategy of Latent Generative Replay.
   
### 2. EVALUATE
Bechmark and evaluate the performance together with the computational and memory requirements of the CL strategy.

### 3. DISCUSSION
Discuss the results and conclude.

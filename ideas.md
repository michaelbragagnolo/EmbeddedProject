## Next steps
### Strategy training
**Criteria for assessing quality** of the Generative Replay strategy:
 - In each iteration, use only a fixed amount of samples to replay. Total number of samples not depending on the number of previous tasks;
 - Vary the amount of samples per mini-batch to test whether the amount of replay needed could be reduced further;
 - How important is the quality of the generative model? Replay does not need to perfect in order to be useful;
 - Vary the number of hidden units in the VAE model used for producing replay to test how good the quality of the replay needs to be;
 - VAE model loss vs epoch;
 - Accuracy vs # of classes so far;
 - ...
  
 **... ideas of improvements:**
 - Improve the quality of the generator using recent progress in generative modeling (very challenging problem, not very efficient and very costly to train).

### Strategy evaluation
**How to test in a MCU-perspective?**
 - Quantify memory and computational requirements of CL algorithm;
 - Identify the requirements for enabling onboard ML for microcontroller class devices (https://arxiv.org/abs/2205.14550);
 - Describe the edge platform to run fictitiously the algorithm (https://arxiv.org/abs/2007.13631);
 - Qualitatively report the trade-off between memory footprint, latency and accuracy for learning a new class with Generative Replay via Variational Autoencoder, when targeting an image classification task.

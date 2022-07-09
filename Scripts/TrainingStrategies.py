#!/usr/bin/env python
# coding: utf-8

# ***
# ## Training strategies
# 
# Experiment that defines the **training strategies for the VAE model and Generative Replay**.
# 
# *Code is based on the SupervisedTemplate of Avalanche:*
#  - https://avalanche-api.continualai.org/en/v0.2.0/generated/avalanche.training.templates.SupervisedTemplate.html?highlight=supervised#avalanche.training.templates.SupervisedTemplate

# In[1]:


# --- LIBRARIES AND UTILS ---
from typing import Optional, Sequence, List, Union

import torch
import torch.nn as nn
from torch.nn import Module, CrossEntropyLoss
from torch.optim import Optimizer, SGD

import avalanche
from avalanche.training.templates.base import BaseTemplate
from avalanche.training.templates.supervised import SupervisedTemplate
from avalanche.training.plugins import (SupervisedPlugin, GenerativeReplayPlugin, EvaluationPlugin, TrainGeneratorAfterExpPlugin)
from avalanche.training.plugins.evaluation import default_evaluator

from avalanche.logging import InteractiveLogger


# ***
# ### VAE training class
# 
# Experiment that defines the training strategies for the **VAE model**.

# In[2]:


# # Import the VAE generative model and its loss function
from GenerativeModel import VAE_model, VAE_loss

class VAE_TrainingStrategy(SupervisedTemplate):

    def __init__(
        self,
        # --- Strategy instantiation --- # 
        # 1. Model
        # 2. Optimizer
        # 3. Criterion
        model: Module, optimizer: Optimizer, criterion=VAE_loss,
        # additional arguments
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = EvaluationPlugin(
            loggers=[InteractiveLogger()],
            suppress_warnings=True,),
        eval_every=-1,
        **base_kwargs):

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs)
        
        
    # The criterion function is overwritten to adapt the input to the VAE loss function,
    def criterion(self):
        return self._criterion(self.mb_x, self.mb_output)


# ***
# ### Generative Replay training class
# 
# Experiment that defines the training strategy of **Generative Replay**.

# In[3]:


class GenerativeReplay(SupervisedTemplate):

    def __init__(
        self,
        # --- Strategy instantiation --- # 
        # 1. Model
        # 2. Optimizer
        # 3. Criterion
        model: Module, optimizer: Optimizer, criterion=CrossEntropyLoss(),
        # additional arguments
        train_mb_size: int = 1,
        train_epochs: int = 1,
        eval_mb_size: int = None,
        device=None,
        plugins: Optional[List[SupervisedPlugin]] = None,
        evaluator: EvaluationPlugin = default_evaluator,
        eval_every=-1,
        
        # param: generator_strategy takes as input the VAE generator
        generator_strategy: BaseTemplate = None,
        
        # replay data size to be concatenated to the current training batch
        replay_size: int = None,
        increasing_replay_size: bool = False, # double the amount of replay data added to each data batch
        **base_kwargs):
        
        # any kind of generative model (implemented in GenerativeModel.py)
        if generator_strategy is not None:
            self.generator_strategy = generator_strategy

        rp = GenerativeReplayPlugin(
            generator_strategy=self.generator_strategy,
            replay_size=replay_size,
            increasing_replay_size=increasing_replay_size)
        
        tgp = TrainGeneratorAfterExpPlugin()

        if plugins is None:
            plugins = [tgp, rp]
        else:
            plugins.append(tgp)
            plugins.append(rp)

        super().__init__(
            model, optimizer, criterion,
            train_mb_size=train_mb_size,
            train_epochs=train_epochs,
            eval_mb_size=eval_mb_size,
            device=device,
            plugins=plugins,
            evaluator=evaluator,
            eval_every=eval_every,
            **base_kwargs)


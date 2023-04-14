import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.nn.functional import softplus, softmax
from torch.distributions import constraints
from torch import optim
from torch.optim import Adam
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.distributions.util import broadcast_shape
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO
from pyro.contrib.examples.scanvi_data import get_data
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from data_utils import *
import os

from inspect import isclass


class MLPForOne(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128],last_acitivation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.model = nn.Sequential()
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.output_size))
            if last_acitivation is not None:
                self.model.add_module('relu_out',last_acitivation())
        else:
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    self.model.add_module('input_layer', nn.Linear(self.input_size, self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
                else:
                    self.model.add_module('hidden_layer{}'.format(i),
                                        nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], self.output_size))
            if last_acitivation is not None:
                self.model.add_module('relu_out',last_acitivation())

        print('create encoder ')
        print(self.model)

    def forward(self, x):
        
        if isinstance(x,tuple):
            x=torch.cat(x,-1)

        out_put_ = self.model(x)
        return out_put_
    
    def get_predict(self,X):
        self.eval()
        if isinstance(X,tuple):
            X=torch.cat(X,-1)
        output_=self.model(X).cpu().detach().numpy()
        self.train()
        return output_

class MLPForTwo(nn.Module):
    #output_size: mu or sigma size
    def __init__(self, input_size, output_size, hidden_layers=[128],last_acitivation=(None,None)):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.moutput_size=2*output_size
        self.hidden_layers = hidden_layers
        self.activation_1=last_acitivation[0]
        self.activation_2=last_acitivation[1]

        self.model = nn.Sequential()
        if last_acitivation[0] is not None:
            self.activation_1=last_acitivation[0]()
        if last_acitivation[1] is not None:
            self.activation_2=last_acitivation[1]()
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.moutput_size))
              
        else:
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    self.model.add_module('input_layer', nn.Linear(self.input_size, self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
                else:
                    self.model.add_module('hidden_layer{}'.format(i),
                                        nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], self.moutput_size))
           
                
        print('create encoder ')
        print(self.model)

    def forward(self,x):
        if isinstance(x,tuple):
            x=torch.cat(x,-1)

        out_put_ = self.model(x)
        loc,scale=split_in_half(out_put_)
        scale=self.activation_2(scale)

        if self.activation_1 is not None:
            loc=self.activation_1(loc)
        if self.activation_2 is not None:
            scale=self.activation_2(scale)
        return loc,scale
    
    def get_predict(self,X):
        self.eval()
        if isinstance(X,tuple):
            X=torch.cat(X,-1)
        out_put_ = self.model(X)
        loc,scale=split_in_half(out_put_)
        if self.activation_1 is not None:
            loc=self.activation_1(loc)
        if self.activation_2 is not None:
            scale=self.activation_2(scale)
        loc_=loc.detach().cpu().numpy()
        scale_=scale.detach().cpu().numpy()
        self.train()
        return loc_,scale_



class MLPForOneNB(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128],last_acitivation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.model = nn.Sequential()
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.output_size))
            if last_acitivation is not None:
                self.model.add_module('relu_out',last_acitivation())
        else:
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    self.model.add_module('input_layer', nn.Linear(self.input_size, self.hidden_layers[i]))
                   # self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
                else:
                    self.model.add_module('hidden_layer{}'.format(i),
                                        nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
                    #self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], self.output_size))
            if last_acitivation is not None:
                self.model.add_module('relu_out',last_acitivation())

        print('create encoder ')
        print(self.model)

    def forward(self, x):
        
        if isinstance(x,tuple):
            x=torch.cat(x,-1)

        out_put_ = self.model(x)
        return out_put_
    
    def get_predict(self,X):
        self.eval()
        if isinstance(X,tuple):
            X=torch.cat(X,-1)
        output_=self.model(X).cpu().detach().numpy()
        self.train()
        return output_

class MLPForOneC(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128],last_acitivation=None):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.model = nn.Sequential()
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.output_size))
            if last_acitivation is not None:
                self.model.add_module('relu_out',last_acitivation())
        else:
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    self.model.add_module('input_layer', nn.Linear(self.input_size, self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module("Dropout{}".format(i),nn.Dropout(0.4))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
                else:
                    self.model.add_module('hidden_layer{}'.format(i),
                                        nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.LeakyReLU(inplace=True))
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], self.output_size))
            if last_acitivation is not None:
                self.model.add_module('relu_out',last_acitivation())

        print('create encoder ')
        print(self.model)

    def forward(self, x):
        
        if isinstance(x,tuple):
            x=torch.cat(x,-1)

        out_put_ = self.model(x)
        return out_put_
    
    def get_predict(self,X):
        self.eval()
        if isinstance(X,tuple):
            X=torch.cat(X,-1)
        
        output_=self.model(X).cpu().detach().numpy()
        self.train()
        return output_
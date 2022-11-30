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


class Exp(nn.Module):
    """
    a custom module for exponentiation of tensors
    """

    def __init__(self):
        super().__init__()

    def forward(self, val):
        return torch.exp(val)

class ExpM(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, val):
        z1 = torch.exp(val)
        z2 = torch.max(z1, 0.01 * torch.ones_like(z1))
        y = torch.min(z2, 5000. * torch.ones_like(z2))
        return y

class SigmoidM(nn.Module):
    def __init__(self, scale = 1000):
        super().__init__()
        self.scale = scale

    def forward(self, val):
        z = nn.Sigmoid()(val)
        return self.scale * z

class ConcatModule(nn.Module):
    """
    a custom module for concatenation of tensors
    """

    def __init__(self, allow_broadcast=False):
        self.allow_broadcast = allow_broadcast
        super().__init__()

    def forward(self, *input_args):
        # we have a single object
        if len(input_args) == 1:
            # regardless of type,
            # we don't care about single objects
            # we just index into the object
            input_args = input_args[0]

        # don't concat things that are just single objects
        if torch.is_tensor(input_args):
            return input_args
        else:
            if self.allow_broadcast:
                shape = broadcast_shape(*[s.shape[:-1] for s in input_args]) + (-1,)
                input_args = [s.expand(shape) for s in input_args]
            return torch.cat(input_args, dim=-1)


class ListOutModule(nn.ModuleList):
    """
    a custom module for outputting a list of tensors from a list of nn modules
    """

    def __init__(self, modules):
        super().__init__(modules)

    def forward(self, *args, **kwargs):
        # loop over modules in self, apply same args
        return [mm.forward(*args, **kwargs) for mm in self]


def call_nn_op(op):
    """
    a helper function that adds appropriate parameters when calling
    an nn module representing an operation like Softmax

    :param op: the nn.Module operation to instantiate
    :return: instantiation of the op module with appropriate parameters
    """
    if op in [nn.Softmax, nn.LogSoftmax]:
        return op(dim=1)
    else:
        return op()


class MLP(nn.Module):
    def __init__(
        self,
        mlp_sizes,
        activation=nn.ReLU,
        output_activation=None,
        post_layer_fct=lambda layer_ix, total_layers, layer: None,
        post_act_fct=lambda layer_ix, total_layers, layer: None,
        allow_broadcast=False,
        use_cuda=False,
    ):
        # init the module object
        super().__init__()

        assert len(mlp_sizes) >= 2, "Must have input and output layer sizes defined"

        # get our inputs, outputs, and hidden
        input_size, hidden_sizes, output_size = (
            mlp_sizes[0],
            mlp_sizes[1:-1],
            mlp_sizes[-1],
        )

        # assume int or list
        assert isinstance(
            input_size, (int, list, tuple)
        ), "input_size must be int, list, tuple"

        # everything in MLP will be concatted if it's multiple arguments
        last_layer_size = input_size if type(input_size) == int else sum(input_size)

        # everything sent in will be concatted together by default
        all_modules = [ConcatModule(allow_broadcast)]

        # loop over l
        for layer_ix, layer_size in enumerate(hidden_sizes):
            assert type(layer_size) == int, "Hidden layer sizes must be ints"

            # get our nn layer module (in this case nn.Linear by default)
            cur_linear_layer = nn.Linear(last_layer_size, layer_size)

            # for numerical stability -- initialize the layer properly
            cur_linear_layer.weight.data.normal_(0, 0.001)
            cur_linear_layer.bias.data.normal_(0, 0.001)

            # use GPUs to share data during training (if available)
            if use_cuda:
                cur_linear_layer = nn.DataParallel(cur_linear_layer)

            # add our linear layer
            all_modules.append(cur_linear_layer)

            # handle post_linear
            post_linear = post_layer_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )

            # if we send something back, add it to sequential
            # here we could return a batch norm for example
            if post_linear is not None:
                all_modules.append(post_linear)

            # handle activation (assumed no params -- deal with that later)
            all_modules.append(activation())

            # now handle after activation
            post_activation = post_act_fct(
                layer_ix + 1, len(hidden_sizes), all_modules[-1]
            )

            # handle post_activation if not null
            # could add batch norm for example
            if post_activation is not None:
                all_modules.append(post_activation)

            # save the layer size we just created
            last_layer_size = layer_size

        # now we have all of our hidden layers
        # we handle outputs
        assert isinstance(
            output_size, (int, list, tuple)
        ), "output_size must be int, list, tuple"

        if type(output_size) == int:
            all_modules.append(nn.Linear(last_layer_size, output_size))
            if output_activation is not None:
                all_modules.append(
                    call_nn_op(output_activation)
                    if isclass(output_activation)
                    else output_activation
                )
        else:

            # we're going to have a bunch of separate layers we can spit out (a tuple of outputs)
            out_layers = []

            # multiple outputs? handle separately
            for out_ix, out_size in enumerate(output_size):

                # for a single output object, we create a linear layer and some weights
                split_layer = []

                # we have an activation function
                split_layer.append(nn.Linear(last_layer_size, out_size))

                # then we get our output activation (either we repeat all or we index into a same sized array)
                act_out_fct = (
                    output_activation
                    if not isinstance(output_activation, (list, tuple))
                    else output_activation[out_ix]
                )

                if act_out_fct:
                    # we check if it's a class. if so, instantiate the object
                    # otherwise, use the object directly (e.g. pre-instaniated)
                    split_layer.append(
                        call_nn_op(act_out_fct) if isclass(act_out_fct) else act_out_fct
                    )

                # our outputs is just a sequential of the two
                out_layers.append(nn.Sequential(*split_layer))

            all_modules.append(ListOutModule(out_layers))

        # now we have all of our modules, we're ready to build our sequential!
        # process mlps in order, pretty standard here
        self.sequential_mlp = nn.Sequential(*all_modules)

    # pass through our sequential for the output!
    def forward(self, *args, **kwargs):
        return self.sequential_mlp.forward(*args, **kwargs)


def split_in_half(t):
    return t.reshape(t.shape[:-1] + (2, -1)).unbind(-2)

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
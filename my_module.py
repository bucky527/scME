import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scvi import REGISTRY_KEYS
from scvi._compat import Literal 
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
import torch
import torch.nn as nn
from torch import optim, relu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
import seaborn as sns
from collections import Counter
from data_utils import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score


DATA_PATH='D:/zhoub/neuron/SingleCellMerge/data'


class Encoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128],last_relu=False):
        super().__init__()
        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers

        self.last_relu=last_relu
        self.model = nn.Sequential()
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.output_size))
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
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], output_size))
            if self.last_relu:
                self.model.add_module('relu_out',nn.LeakyReLU(inplace=True))
        print('create encoder  ')
        print(self.model)

    def forward(self, x):
        out_put_ = self.model(x)
        return out_put_
    
    def get_predict(self,X):
        self.eval()
        return self.model(X).cpu().detach().numpy()

class Decoder(nn.Module):
    def __init__(self, input_size, output_size, hidden_layers=[128],last_sigmoid=False):
        super().__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.hidden_layers = hidden_layers
        self.last_sigmoid=last_sigmoid

        self.model = nn.Sequential()
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.output_size))
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
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], output_size))
            if self.last_sigmoid:
                self.model.add_module('sigmoid',nn.Sigmoid())

        print('create decoder ')
        print(self.model)

    def forward(self, x):
        out_put_ = self.model(x)
        return out_put_
    def get_predict(self,X):
        self.eval()
        return self.model(X).cpu().detach().numpy()



class Classifier(nn.Module):

    def __init__(self,input_size,class_num=6,hidden_layers=[128]):
        super().__init__()
        self.class_num=class_num
        self.hidden_layers=hidden_layers
        self.input_size=input_size
        self.model=nn.Sequential()
        self.sm=nn.Softmax(dim=1)
            
        if len(self.hidden_layers) == 0:
            self.model.add_module('parameters_layer', nn.Linear(self.input_size, self.output_size))
        else:
            for i in range(len(self.hidden_layers)):
                if i == 0:
                    self.model.add_module('input_layer', nn.Linear(self.input_size, self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
                    #self.model.add_module('dropout_1',nn.Dropout(0.4))
                else:
                    self.model.add_module('hidden_layer{}'.format(i),
                                        nn.Linear(self.hidden_layers[i - 1], self.hidden_layers[i]))
                    self.model.add_module("BN{}".format(i),nn.BatchNorm1d(self.hidden_layers[i]))
                    self.model.add_module('relu{}'.format(i), nn.ReLU(inplace=True))
            self.model.add_module('output_layer', nn.Linear(hidden_layers[-1], self.class_num))

        print('Create Classifer ')
        print(self.model)

    def forward(self,x):
        out_put_=self.model(x)
        return out_put_

    def get_predict(self,x):
        
        self.eval()
        x=self.model(x).detach()
        return self.sm(x).detach().cpu().numpy()
    


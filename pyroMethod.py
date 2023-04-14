import os
os.environ["CUDA_VISIBLE_DEVICES"]="0"
import numpy as np
import pandas as pd
from sklearn.model_selection import KFold, train_test_split
import torch
import torch.nn as nn
from torch.distributions import constraints
from torch.optim import Adam
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO,JitTrace_ELBO, JitTraceEnum_ELBO,Trace_ELBO
from pyro.contrib.examples.scanvi_data import get_data
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import os
from data_utils import *
# from pyro_utils import *
from mlp_net import *
import argparse
import umap
from sklearn.metrics.cluster import adjusted_rand_score,normalized_mutual_info_score

class ScMESVI_Batch(nn.Module):
    def __init__(self,rna_dim,protein_dim,latent_dim=24,scale_factor=1.0,use_cuda=True,aux_loss_multiplier=10):
        
        #alpha :ratio of classifiaction loss
        #latent_dim: dim of zm
        #class_num: max classification  num
        super().__init__()
        self.rna_dim=rna_dim
        self.protein_dim=protein_dim
        self.latent_dim=latent_dim
        self.epsilon = 0.006
        self.use_cuda=use_cuda
        self.scale_factor=scale_factor
        self.aux_loss_multiplier=aux_loss_multiplier
    def setup_network(self,rna_class_num,protein_class_num,batch_num,l_loc,l_scale,rna_latent_dim=32,protein_latent_dim=20):
        self.rna_class_num=rna_class_num
        self.protein_class_num=protein_class_num
        self.batch_num=batch_num

        #hyperparameters l
        #均值和方差 直接计算然后输入 每个批次计算
        self.l_loc=torch.ones((self.batch_num,self.rna_dim))
        self.l_scale=torch.ones((self.batch_num,self.rna_dim))


    def model(self,rna,protein,batch_onehot=None):
        pyro.module("svimerge",self)
        batch_size=r.size(0)
        options=dict(dtype=r.dtype,device=r.device)
        theta=pyro.param("inverse_dispresion",10*torch.ones((self.batch_num,self.rna_dim)),constraint=constraints.positive)
        phi=pyro.param("protein_inverse_dispresion",torch.ones((self.batch_num,self.protein_dim)),constraint=constraints.positive)
        

        with pyro.plate('data',len(rna)):

            zm_loc=torch.zeros(batch_size,self.latent_dim,**options)
            zm_scale=torch.ones(batch_size,self.latent_dim,**options)

            zm=pyro.sample("zm",dist.Normal(zm_loc,zm_scale).to_event(1))

            rna_p=self.rnap_decoder(zm)
    
    def guide(self,r,e,y1=None,y2=None):
        pyro.module("svimerge",self)
        with pyro.plate('data',len(r)):

            zr_loc,zr_scale=self.zr_encoder(r)
            zr=pyro.sample('zr',dist.Normal(zr_loc,zr_scale).to_event(1))
            ze_loc,ze_scale=self.ze_encoder(e)
            ze=pyro.sample('ze',dist.Normal(ze_loc,ze_scale).to_event(1))
            zm_loc,zm_scale=self.zm_encoder((zr,ze))

            zm=pyro.sample('zm',dist.Normal(zm_loc,zm_scale).to_event(1))


            if y1 is None:
                y1_logits=self.classifer1(zm)
                y1=pyro.sample("y1",dist.OneHotCategorical(logits=y1_logits))
            
            
            if y2 is None:
                y2_logits=self.classifer2(zm)
                y2=pyro.sample("y2",dist.OneHotCategorical(logits=y2_logits))


    def model_classify(self, r,e,y1= None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("svimerge",self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data',len(r)):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if y1 is not None:
                zr,_=self.zr_encoder(r)
                
                ze,_=self.ze_encoder(e)
                zm_loc,zm_scale=self.zm_encoder((zr,ze))
                zm=pyro.sample('zm1_aux',dist.Normal(zm_loc,zm_scale).to_event(1))
                y_logits=self.classifer1(zm)

                with pyro.poutine.scale(scale = 100 * self.aux_loss_multiplier):
                    y1_aux = pyro.sample('y1_aux', dist.OneHotCategorical(logits=y_logits), obs = y1)
                   

    def guide_classify(self, r,e ,y = None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


    def model_classify2(self, r,e,y2= None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("svimerge",self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data',len(r)):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if y2 is not None:
                zr,_=self.zr_encoder(r)
                ze,_=self.ze_encoder(e)
                zm,_=self.zm_encoder((zr,ze))
                zm_loc,zm_scale=self.zm_encoder((zr,ze))
                zm=pyro.sample('zm2_aux',dist.Normal(zm_loc,zm_scale).to_event(1))
                y_logits=self.classifer2(zm)

                with pyro.poutine.scale(scale = 100 * self.aux_loss_multiplier):
                    y2_aux = pyro.sample('y2_aux', dist.OneHotCategorical(logits=y_logits), obs = y2)
                   
                # alpha = self.encoder_zy_y(zs)
                # with pyro.poutine.scale(scale = 50 * self.aux_loss_multiplier):
                #     ys_aux = pyro.sample('y_aux', dist.OneHotCategorical(alpha), obs = ys)

    def guide_classify2(self, r,e ,y1= None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass

    def get_y_predict(self,r,e):

        self.eval()

        zr_loc,zr_scale=self.zr_encoder(r)
        ze_loc,ze_scale=self.ze_encoder(e)

        zm_loc,zm_scale=self.zm_encoder((zr_loc,ze_loc))
        y_logits=self.classifer1(zm_loc)

        self.train()
        return y_logits.detach().cpu().numpy()
    
    def get_y2_predict(self,r,e):
        self.eval()
        zr_loc,zr_scale=self.zr_encoder(r)
        ze_loc,ze_scale=self.ze_encoder(e)

        zm_loc,zm_scale=self.zm_encoder((zr_loc,ze_loc))
        y_logits=self.classifer2(zm_loc)
        self.train()
        return y_logits.detach().cpu().numpy()
    
    def get_zm_predict(self,r,e):

        self.eval() 
        zr_loc,_=self.zr_encoder(r)
        ze_loc,_=self.ze_encoder(e)
        zm_loc,_=self.zm_encoder((zr_loc,ze_loc))
        self.train()
        return zm_loc.detach().cpu().numpy()

class ScMESVI(nn.Module):
    def __init__(self,rna_class_num,protein_class_num,latent_dim=32,scale_factor=1.0,device="cpu",aux_loss_multiplier=10):
        
        #alpha :ratio of classifiaction loss
        #latent_dim: dim of zm
        #class_num: max classification  num

        super().__init__()
        self.latent_dim=latent_dim
        self.epsilon = 0.006
        self.class_num1=rna_class_num
        self.class_num2=protein_class_num
        self.scale_factor=scale_factor
        self.aux_loss_multiplier=aux_loss_multiplier
        self.device=device
    
    def setup_network(self,rna_size,protein_size,rna_latent_dim=32,protein_latent_dim=20):
         #Decoder
        self.cutoff = nn.Threshold(1.0e-9, 1.0e-9)
        self.rna_size=rna_size
        self.protin_size=protein_size
        self.rna_latent_dim=rna_latent_dim
        self.protein_latent_dim=protein_latent_dim

        self.zm_rdecoder=MLPForTwo(input_size=self.latent_dim+self.class_num1,output_size=rna_latent_dim,hidden_layers=[128,256],last_acitivation=[nn.ReLU,nn.Softplus])
        self.zr_size=self.zm_rdecoder.output_size
        self.zm_edecoder=MLPForTwo(input_size=self.latent_dim+self.class_num2,output_size=protein_latent_dim,hidden_layers=[64,32],last_acitivation=[nn.ReLU,nn.Softplus])

        self.ze_size=self.zm_edecoder.output_size

        self.zr_decoder=MLPForOne(input_size=self.zm_rdecoder.output_size,output_size=rna_size,hidden_layers=[256,512],last_acitivation=nn.ReLU)#nn.LogSmoid

        self.ze_decoder=MLPForTwo(input_size=self.zm_edecoder.output_size,output_size=protein_size,hidden_layers=[64,64],last_acitivation=[nn.ReLU,nn.Softplus])

        #Encoder
        
        self.zr_encoder=MLPForTwo(input_size=rna_size,output_size= self.zr_size,hidden_layers=[1000,400,128],last_acitivation=[nn.ReLU,nn.Softplus])

        self.ze_encoder=MLPForTwo(input_size=protein_size,output_size= self.ze_size,hidden_layers=[128,64],last_acitivation=[nn.ReLU,nn.Softplus])

        self.zm_encoder=MLPForTwo(input_size=self.ze_size+self.zr_size,output_size=self.latent_dim,hidden_layers=[128,64],last_acitivation=[nn.ReLU,nn.Softplus])

        self.classifer1=MLPForOne(input_size=self.latent_dim,output_size=self.class_num1,hidden_layers=[128,64],last_acitivation=None)
        self.classifer2=MLPForOne(input_size=self.latent_dim,output_size=self.class_num2,hidden_layers=[128,64],last_acitivation=None)
    

    def model(self,r,e,y1=None,y2=None):
        pyro.module("svimerge",self)

        batch_size=r.size(0)
        options=dict(dtype=r.dtype,device=r.device)

        with pyro.plate('data',len(r)):
            # Define a unit Normal prior distribution for z1

            # y1_prior=torch.zeros(batch_size,self.class_num1,**options)

            # y1=pyro.sample("y1",dist.OneHotCategorical(logits=y1_prior),obs=y1)
            # y2_prior=torch.zeros(batch_size,self.class_num2,**options)

            # y2=pyro.sample("y2",dist.OneHotCategorical(logits=y2_prior),obs=y2)

            #zm prior
            zm_loc=torch.zeros(batch_size,self.latent_dim,**options)
            zm_scale=torch.ones(batch_size,self.latent_dim,**options)

            zm=pyro.sample("zm",dist.Normal(zm_loc,zm_scale).to_event(1))

            zr_loc,zr_scale=self.zm_rdecoder((zm,y1))

            ze_loc,ze_scale=self.zm_edecoder((zm,y2))

            zr=pyro.sample('zr',dist.Normal(zr_loc,zr_scale).to_event(1))
            
            ze=pyro.sample('ze',dist.Normal(ze_loc,ze_scale).to_event(1))

            concentrate = self.zr_decoder(zr)
            concentrate = self.cutoff(concentrate)

            max_count = torch.ceil(r.sum(1).max()).int().item()
            pyro.sample('r', dist.DirichletMultinomial(total_count = max_count, concentration = concentrate,is_sparse=True), obs = r)
            
            #the epyhs obs (assume it is a N() distribution)
            e_loc,e_scale=self.ze_decoder(ze)
            pyro.sample('e',dist.Normal(e_loc,e_scale).to_event(1),obs=e)
    def guide(self,r,e,y1=None,y2=None):
        pyro.module("svimerge",self)
        with pyro.plate('data',len(r)):

            zr_loc,zr_scale=self.zr_encoder(r)
            zr=pyro.sample('zr',dist.Normal(zr_loc,zr_scale).to_event(1))
            ze_loc,ze_scale=self.ze_encoder(e)
            ze=pyro.sample('ze',dist.Normal(ze_loc,ze_scale).to_event(1))
            zm_loc,zm_scale=self.zm_encoder((zr,ze))

            zm=pyro.sample('zm',dist.Normal(zm_loc,zm_scale).to_event(1))


            if y1 is None:
                y1_logits=self.classifer1(zm)
                y1=pyro.sample("y1",dist.OneHotCategorical(logits=y1_logits))
            
            
            if y2 is None:
                y2_logits=self.classifer2(zm)
                y2=pyro.sample("y2",dist.OneHotCategorical(logits=y2_logits))
               
            # else:
            #     classification_loss=y_dist.log_prob(y)

            #     # and the guide log_prob appears in the ELBO as -log q
            #     pyro.factor("classification_loss", -self.alpha * classification_loss, has_rsample=False)

        

    def model_classify(self, r,e,y1= None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("svimerge",self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data',len(r)):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if y1 is not None:
                zr,_=self.zr_encoder(r)
                
                ze,_=self.ze_encoder(e)
                zm_loc,zm_scale=self.zm_encoder((zr,ze))
                zm=pyro.sample('zm1_aux',dist.Normal(zm_loc,zm_scale).to_event(1))
                y_logits=self.classifer1(zm)

                with pyro.poutine.scale(scale = 100 * self.aux_loss_multiplier):
                    y1_aux = pyro.sample('y1_aux', dist.OneHotCategorical(logits=y_logits), obs = y1)
                   

    def guide_classify(self, r,e ,y = None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


    def model_classify2(self, r,e,y2= None):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("svimerge",self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data',len(r)):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if y2 is not None:
                zr,_=self.zr_encoder(r)
                ze,_=self.ze_encoder(e)
                zm,_=self.zm_encoder((zr,ze))
                zm_loc,zm_scale=self.zm_encoder((zr,ze))
                zm=pyro.sample('zm2_aux',dist.Normal(zm_loc,zm_scale).to_event(1))
                y_logits=self.classifer2(zm)

                with pyro.poutine.scale(scale = 100 * self.aux_loss_multiplier):
                    y2_aux = pyro.sample('y2_aux', dist.OneHotCategorical(logits=y_logits), obs = y2)
                   
                # alpha = self.encoder_zy_y(zs)
                # with pyro.poutine.scale(scale = 50 * self.aux_loss_multiplier):
                #     ys_aux = pyro.sample('y_aux', dist.OneHotCategorical(alpha), obs = ys)

    def guide_classify2(self, r,e ,y1= None):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass

    def get_y_predict(self,r,e):

        self.eval()

        zr_loc,zr_scale=self.zr_encoder(r)
        ze_loc,ze_scale=self.ze_encoder(e)

        zm_loc,zm_scale=self.zm_encoder((zr_loc,ze_loc))
        y_logits=self.classifer1(zm_loc)

        self.train()
        return y_logits.detach().cpu().numpy()
    
    def get_y2_predict(self,r,e):
        self.eval()
        zr_loc,zr_scale=self.zr_encoder(r)
        ze_loc,ze_scale=self.ze_encoder(e)

        zm_loc,zm_scale=self.zm_encoder((zr_loc,ze_loc))
        y_logits=self.classifer2(zm_loc)
        self.train()
        return y_logits.detach().cpu().numpy()
    
    def get_zm_predict(self,r,e):

        self.eval() 
        zr_loc,_=self.zr_encoder(r)
        ze_loc,_=self.ze_encoder(e)
        zm_loc,_=self.zm_encoder((zr_loc,ze_loc))
        self.train()
        return zm_loc.detach().cpu().numpy()


class ScMESVI_2(nn.Module):
    def __init__(self,rna_dim,protein_dim,latent_dim=32,scale_factor=1.0,use_cuda=True,aux_loss_multiplier=500,config_enum=None):
        
        #alpha :ratio of classifiaction loss
        #latent_dim: dim of zm
        #class_num: max classification  num
        super().__init__()
        self.rna_dim=rna_dim
        self.protein_dim=protein_dim
        self.latent_dim=latent_dim
        self.epsilon = 0.006
        self.use_cuda=use_cuda
        self.scale_factor=scale_factor
        self.aux_loss_multiplier=aux_loss_multiplier
        self.allow_braodcast=config_enum=='parallel'


    def setup_network(self,rna_class_num,protein_class_num,l_loc,l_scale,c_loc,c_scale,
                    rna_latent_dim=24,protein_latent_dim=24
                    ):
        self.rna_class_num=rna_class_num
        self.protein_class_num=protein_class_num
        self.rna_latent_dim=rna_latent_dim
        self.protein_latent_dim=protein_latent_dim

        #hyperparameters l
        #均值和方差 直接计算然后输入 每个批次计算
        self.l_loc=l_loc
        self.l_scale=l_scale
        self.c_loc=c_loc
        self.c_scale=c_scale

        # inference encoder
        # #zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
        # self.zr_encoder=MLP(
        #     [self.rna_dim+self.rna_class_num]+[1000,256,64]+[[self.rna_latent_dim,self.rna_latent_dim,1,1]],    
        #     activation=nn.ReLU,
        #     output_activation=[None,nn.Softplus,None,nn.Softplus],
        #     post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
        #     use_cuda=self.use_cuda,
        #     allow_broadcast=self.allow_braodcast
        # )
        # # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        # self.zp_encoder=MLP(
        #     [self.protein_dim+self.protein_class_num]+[256,128,32]+[[self.protein_latent_dim,self.protein_latent_dim,1,1,self.protein_dim]],    
        #     activation=nn.ReLU,
        #     output_activation=[None,nn.Softplus,None,nn.Softplus,nn.Sigmoid],
        #     post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
        #     use_cuda=self.use_cuda,
        #     allow_broadcast=self.allow_braodcast
        # )
        # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna))
        self.zr_encoder=MLP(
            [self.rna_dim]+[1000,256,64]+[[self.rna_latent_dim,self.rna_latent_dim,1,1]],    
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus,None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        self.zp_encoder=MLP(
            [self.protein_dim]+[256,128,32]+[[self.protein_latent_dim,self.protein_latent_dim,1,1,self.protein_dim]],    
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus,None,nn.Softplus,nn.Sigmoid],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        # zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
        self.zm_encoder=MLP(
            [self.rna_latent_dim+self.protein_latent_dim]+[64,32]+[[self.latent_dim,self.latent_dim]],    
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )

        #generate decoder
        # zr_loc,zr_scale=self.zr_decoder(zm)
        self.zr_decoder=MLP(
            [self.latent_dim]+[32,64]+[[self.rna_latent_dim,self.rna_latent_dim]],
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )

        #zp_loc,zp_scale=self.zp_decoder(zm)
        self.zp_decoder=MLP(
            [self.latent_dim]+[32,64]+[[self.protein_latent_dim,self.protein_latent_dim]],
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        # gate_logits, mu = self.rna_decoder(zr)
        self.rna_decoder=MLP(
            [self.rna_latent_dim]+[64,256,1000]+[[self.rna_dim,self.rna_dim]],
            activation=nn.ReLU,
            output_activation=[None,nn.Softmax],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        #  gamma_p,pi_p=self.protein_decoder(zp) gamma_p [1,+inf]
        self.protein_decoder=MLP(
            [self.protein_latent_dim]+[32,128,256]+[[self.protein_dim,self.protein_dim]],
            activation=nn.ReLU,
            output_activation=[nn.ReLU,nn.Sigmoid],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )

        #classfier
        self.classifier1=MLP(
            [self.latent_dim]+[128,64]+[self.rna_class_num],
            activation=nn.ReLU,
            output_activation=None,
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        self.classifier2=MLP(
            [self.latent_dim]+[128,64]+[self.protein_class_num],
            activation=nn.ReLU,
            output_activation=None,
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )


    def model(self,rna,protein,yr=None,yp=None):
        pyro.module("svimerge",self)
        batch_size=rna.size(0)
        options=dict(dtype=rna.dtype,device=rna.device)
        theta=pyro.param("inverse_dispresion",1000*rna.new_ones(self.rna_dim),constraint=constraints.positive)
        phi=pyro.param("protein_inverse_dispresion",100*rna.new_ones(self.protein_dim),constraint=constraints.positive)
        with pyro.plate('data',len(rna)):
            # Define a unit Normal prior distribution for z1

            # y1_prior=torch.zeros(batch_size,self.class_num1,**options)

            # y1=pyro.sample("y1",dist.OneHotCategorical(logits=y1_prior),obs=y1)
            # y2_prior=torch.zeros(batch_size,self.class_num2,**options)
            # y2=pyro.sample("y2",dist.OneHotCategorical(logits=y2_prior),obs=y2)
            #zm prior
            zm_loc=torch.zeros(batch_size,self.latent_dim,**options)
            zm_scale=torch.ones(batch_size,self.latent_dim,**options)
            zm=pyro.sample("zm",dist.Normal(zm_loc,zm_scale).to_event(1))

            zr_loc,zr_scale=self.zr_decoder(zm)
            zp_loc,zp_scale=self.zp_decoder(zm)

            zr=pyro.sample("zr",dist.Normal(zr_loc,zr_scale).to_event(1))
            zp=pyro.sample("zp",dist.Normal(zp_loc,zp_scale).to_event(1))
            l_scale=self.l_scale*rna.new_ones(1)
            l_loc=self.l_loc*rna.new_ones(1)
            l=pyro.sample("l",dist.LogNormal(self.l_loc,l_scale).to_event(1))

            gate_logits, mu = self.rna_decoder(zr)
            #gate, mu = self.rna_decoder(zr)

            nb_logits = (l * mu+ self.epsilon).log() - (theta + self.epsilon).log()
            rna_dist = dist.ZeroInflatedNegativeBinomial(total_count=theta,gate_logits=gate_logits, 
                                                       logits=nb_logits)
            # Observe the datapoint x using the observation distribution x_dist
            pyro.sample("rna_count", rna_dist.to_event(1), obs=rna)


            #protein totalvi
            c_scale=self.c_scale*protein.new_ones(1)
            c_loc=self.c_loc*protein.new_ones(1)
            beta=pyro.sample("beta",dist.LogNormal(self.c_loc,c_scale).to_event(1))
            
            #mixture of negative binomial distribution
            gamma_p,pi_p=self.protein_decoder(zp)
            # v_p=pyro.sample("v_p",dist.Bernoulli(pi_p).to_event(1))
            nb_logits=((beta*gamma_p)+ self.epsilon).log()-(phi+self.epsilon).log()
            protein_dist=dist.NegativeBinomial(total_count=beta,logits=nb_logits)
            pyro.sample("protein_count",protein_dist.to_event(1),obs=protein)
    
    def guide(self,rna,protein,yr=None,yp=None):
        pyro.module("svimerge",self)
        with pyro.plate('data',len(rna)):
            # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
            # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))s
            zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
            zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
            pyro.sample("l",dist.LogNormal(l_loc,l_scale).to_event(1))
            zr=pyro.sample("zr",dist.Normal(zr_loc,zr_scale).to_event(1))

            # pyro.sample("v_p",dist.Bernoulli(pi).to_event(1))
            pyro.sample("beta",dist.LogNormal(c_loc,c_scale).to_event(1))
            zp=pyro.sample("zp",dist.Normal(zp_loc,zp_scale).to_event(1))
            zm_loc,zm_scale=self.zm_encoder((zr,zp))
            zm=pyro.sample("zm",dist.Normal(zm_loc,zm_scale).to_event(1))

    def model_classify(self,rna,protein,yr,yp):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("svimerge",self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data',len(rna)):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if yr is not None:
                # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
                # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
                zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
                zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
                zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
                # zm=pyro.sample('zm1_aux',dist.Normal(zm_loc,zm_scale).to_event(1))
                yr_logits=self.classifier1(zm_loc)
                yp_logits=self.classifier2(zm_loc)
                with pyro.poutine.scale(scale = self.aux_loss_multiplier):
                    y1_aux = pyro.sample('y1_aux', dist.OneHotCategorical(logits=yr_logits), obs = yr)
                    y2_aux = pyro.sample('y2_aux', dist.OneHotCategorical(logits=yp_logits), obs = yp)
                   

    def guide_classify(self,rna,protein,yr,yp):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass

    def inference_zm(self,rna,protein):
        """
        this function is used to get the latent space representation of the data
         return zm_loc,zr_loc,zp_loc
        """
        self.eval() 
        # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
        # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
        zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
        zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
        self.train()
        return zm_loc.detach().cpu().numpy(),zr_loc.detach().cpu().numpy(),zp_loc.detach().cpu().numpy()

    def inference(self,rna,protein):
        """
        this function is used to get the latent space representation of the data
         return zm_loc
        """
        self.eval() 
        # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
        # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
        zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
        zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
        self.train()
        return zm_loc.detach().cpu().numpy()
    


class ScMESVI_MIXNB(nn.Module):
    def __init__(self,rna_dim,protein_dim,latent_dim=32,scale_factor=1.0,use_cuda=True,aux_loss_multiplier=500,config_enum=None):
        
        #alpha :ratio of classifiaction loss
        #latent_dim: dim of zm
        #class_num: max classification  num
        super().__init__()
        self.rna_dim=rna_dim
        self.protein_dim=protein_dim
        self.latent_dim=latent_dim
        self.epsilon = 0.006
        self.use_cuda=use_cuda
        self.scale_factor=scale_factor
        self.aux_loss_multiplier=aux_loss_multiplier
        self.allow_braodcast=config_enum=='parallel'


    def setup_network(self,rna_class_num,protein_class_num,l_loc,l_scale,c_loc,c_scale,
                    rna_latent_dim=24,protein_latent_dim=24
                    ):
        self.rna_class_num=rna_class_num
        self.protein_class_num=protein_class_num
        self.rna_latent_dim=rna_latent_dim
        self.protein_latent_dim=protein_latent_dim

        #hyperparameters l
        #均值和方差 直接计算然后输入 每个批次计算
        self.l_loc=l_loc
        self.l_scale=l_scale
        self.c_loc=c_loc
        self.c_scale=c_scale

        # inference encoder
        # #zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
        # self.zr_encoder=MLP(
        #     [self.rna_dim+self.rna_class_num]+[1000,256,64]+[[self.rna_latent_dim,self.rna_latent_dim,1,1]],    
        #     activation=nn.ReLU,
        #     output_activation=[None,nn.Softplus,None,nn.Softplus],
        #     post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
        #     use_cuda=self.use_cuda,
        #     allow_broadcast=self.allow_braodcast
        # )
        # # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        # self.zp_encoder=MLP(
        #     [self.protein_dim+self.protein_class_num]+[256,128,32]+[[self.protein_latent_dim,self.protein_latent_dim,1,1,self.protein_dim]],    
        #     activation=nn.ReLU,
        #     output_activation=[None,nn.Softplus,None,nn.Softplus,nn.Sigmoid],
        #     post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
        #     use_cuda=self.use_cuda,
        #     allow_broadcast=self.allow_braodcast
        # )
        # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna))
        self.zr_encoder=MLP(
            [self.rna_dim]+[1000,256,64]+[[self.rna_latent_dim,self.rna_latent_dim,1,1]],    
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus,None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        self.zp_encoder=MLP(
            [self.protein_dim]+[256,128,32]+[[self.protein_latent_dim,self.protein_latent_dim,1,1,self.protein_dim]],    
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus,None,nn.Softplus,nn.Sigmoid],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        # zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
        self.zm_encoder=MLP(
            [self.rna_latent_dim+self.protein_latent_dim]+[64,32]+[[self.latent_dim,self.latent_dim]],    
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )

        #generate decoder
        # zr_loc,zr_scale=self.zr_decoder(zm)
        self.zr_decoder=MLP(
            [self.latent_dim]+[32,64]+[[self.rna_latent_dim,self.rna_latent_dim]],
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )

        #zp_loc,zp_scale=self.zp_decoder(zm)
        self.zp_decoder=MLP(
            [self.latent_dim]+[32,64]+[[self.protein_latent_dim,self.protein_latent_dim]],
            activation=nn.ReLU,
            output_activation=[None,nn.Softplus],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        # gate_logits, mu = self.rna_decoder(zr)
        self.rna_decoder=MLP(
            [self.rna_latent_dim]+[64,256,1000]+[[self.rna_dim,self.rna_dim]],
            activation=nn.ReLU,
            output_activation=[None,nn.Softmax],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        #  gamma_p,pi_p=self.protein_decoder(zp) gamma_p [1,+inf]
        self.protein_decoder=MLP(
            [self.protein_latent_dim]+[32,128,256]+[[self.protein_dim,self.protein_dim]],
            activation=nn.ReLU,
            output_activation=[nn.ReLU,nn.Sigmoid],
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )

        #classfier
        self.classifier1=MLP(
            [self.latent_dim]+[128,64]+[self.rna_class_num],
            activation=nn.ReLU,
            output_activation=None,
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )
        self.classifier2=MLP(
            [self.latent_dim]+[128,64]+[self.protein_class_num],
            activation=nn.ReLU,
            output_activation=None,
            post_layer_fct=lambda layer_ix, total_layers,layer: nn.BatchNorm1d(layer.out_features) if layer_ix <=total_layers else None,
            use_cuda=self.use_cuda,
            allow_broadcast=self.allow_braodcast
        )


    def model(self,rna,protein,yr=None,yp=None):
        pyro.module("svimerge",self)
        batch_size=rna.size(0)
        options=dict(dtype=rna.dtype,device=rna.device)
        theta=pyro.param("inverse_dispresion",100*rna.new_ones(self.rna_dim),constraint=constraints.positive)
        phi=pyro.param("protein_inverse_dispresion",rna.new_ones(self.protein_dim),constraint=constraints.positive)
        with pyro.plate('data',len(rna)):
            # Define a unit Normal prior distribution for z1

            # y1_prior=torch.zeros(batch_size,self.class_num1,**options)

            # y1=pyro.sample("y1",dist.OneHotCategorical(logits=y1_prior),obs=y1)
            # y2_prior=torch.zeros(batch_size,self.class_num2,**options)
            # y2=pyro.sample("y2",dist.OneHotCategorical(logits=y2_prior),obs=y2)
            #zm prior
            zm_loc=torch.zeros(batch_size,self.latent_dim,**options)
            zm_scale=torch.ones(batch_size,self.latent_dim,**options)
            zm=pyro.sample("zm",dist.Normal(zm_loc,zm_scale).to_event(1))

            zr_loc,zr_scale=self.zr_decoder(zm)
            zp_loc,zp_scale=self.zp_decoder(zm)

            zr=pyro.sample("zr",dist.Normal(zr_loc,zr_scale).to_event(1))
            zp=pyro.sample("zp",dist.Normal(zp_loc,zp_scale).to_event(1))
            l_scale=self.l_scale*rna.new_ones(1)
            l_loc=self.l_loc*rna.new_ones(1)
            l=pyro.sample("l",dist.LogNormal(self.l_loc,l_scale).to_event(1))

            gate_logits, mu = self.rna_decoder(zr)
            #gate, mu = self.rna_decoder(zr)

            nb_logits = (l * mu+ self.epsilon).log() - (theta + self.epsilon).log()
            rna_dist = dist.ZeroInflatedNegativeBinomial(total_count=theta,gate_logits=gate_logits, 
                                                       logits=nb_logits)
            # Observe the datapoint x using the observation distribution x_dist
            pyro.sample("rna_count", rna_dist.to_event(1), obs=rna)


            #protein totalvi
            c_scale=self.c_scale*protein.new_ones(1)
            c_loc=self.c_loc*protein.new_ones(1)
            beta=pyro.sample("beta",dist.LogNormal(self.c_loc,c_scale).to_event(1))
            
            #mixture of negative binomial distribution
            gamma_p,pi_p=self.protein_decoder(zp)
            gamma_p=gamma_p+1
            v_p=pyro.sample("v_p",dist.Bernoulli(pi_p).to_event(1))
            nb_logits=(v_p*beta+(1-v_p)*(beta*gamma_p)+ self.epsilon).log()-(phi+self.epsilon).log()
            protein_dist=dist.NegativeBinomial(total_count=beta,logits=nb_logits)
            pyro.sample("protein_count",protein_dist.to_event(1),obs=protein)
    
    def guide(self,rna,protein,yr=None,yp=None):
        pyro.module("svimerge",self)
        with pyro.plate('data',len(rna)):
            # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
            # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))s
            zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
            zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
            pyro.sample("l",dist.LogNormal(l_loc,l_scale).to_event(1))
            zr=pyro.sample("zr",dist.Normal(zr_loc,zr_scale).to_event(1))
            pyro.sample("v_p",dist.Bernoulli(pi).to_event(1))
            pyro.sample("beta",dist.LogNormal(c_loc,c_scale).to_event(1))
            zp=pyro.sample("zp",dist.Normal(zp_loc,zp_scale).to_event(1))
            zm_loc,zm_scale=self.zm_encoder((zr,zp))
            zm=pyro.sample("zm",dist.Normal(zm_loc,zm_scale).to_event(1))

    def model_classify(self,rna,protein,yr,yp):
        """
        this model is used to add auxiliary (supervised) loss as described in the
        Kingma et al., "Semi-Supervised Learning with Deep Generative Models".
        """
        # register all pytorch (sub)modules with pyro
        pyro.module("svimerge",self)

        # inform pyro that the variables in the batch of xs, ys are conditionally independent
        with pyro.plate('data',len(rna)):
            # this here is the extra term to yield an auxiliary loss that we do gradient descent on
            if yr is not None:
                # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
                # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
                zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
                zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
                zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
                # zm=pyro.sample('zm1_aux',dist.Normal(zm_loc,zm_scale).to_event(1))
                yr_logits=self.classifier1(zm_loc)
                yp_logits=self.classifier2(zm_loc)
                with pyro.poutine.scale(scale = self.aux_loss_multiplier):
                    y1_aux = pyro.sample('y1_aux', dist.OneHotCategorical(logits=yr_logits), obs = yr)
                    y2_aux = pyro.sample('y2_aux', dist.OneHotCategorical(logits=yp_logits), obs = yp)
                   

    def guide_classify(self,rna,protein,yr,yp):
        """
        dummy guide function to accompany model_classify in inference
        """
        pass


    def inference_zm(self,rna,protein):
        """
        this function is used to get the latent space representation of the data
         return zm_loc,zr_loc,zp_loc
        """
        self.eval() 
        # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
        # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
        zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
        zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
        self.train()
        return zm_loc.detach().cpu().numpy(),zr_loc.detach().cpu().numpy(),zp_loc.detach().cpu().numpy()

    def inference(self,rna,protein):
        """
        this function is used to get the latent space representation of the data
         return zm_loc,zr_loc,zp_loc
        """
        self.eval() 
        # zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder((rna,yr))
        # zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder((protein,yp))
        zr_loc,zr_scale,l_loc,l_scale=self.zr_encoder(rna)
        zp_loc,zp_scale,c_loc,c_scale,pi=self.zp_encoder(protein)
        zm_loc,zm_scale=self.zm_encoder((zr_loc,zp_loc))
        self.train()
        return zm_loc.detach().cpu().numpy()


def train_scme(model,max_epochs,dataloader,lr,milestones=[80]):
    pyro.clear_param_store()
    pyro.enable_validation(True)

    scheduler= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                         'optim_args': {'lr':lr},
                          'milestones': milestones})
    elbo=Trace_ELBO()
    svi=SVI(model.model,model.guide,scheduler,elbo)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    for epoch in range(max_epochs):
        losses=[]
        for i,(rna,protein,yr,yp) in enumerate(dataloader):
            rna=rna.to(device)
            protein=protein.to(device)
            yr=yr.to(device)
            yp=yp.to(device)
            loss=svi.step(rna,protein,yr,yp)
            losses.append(loss)
        print('epoch :{:0>4d}  loss:{:.5f}'.format(epoch,np.mean(losses)))
        scheduler.step()
    return model

def train_scme_aux(model,max_epochs,dataloader,lr,lr_cla,milestones=[80],save_model=False,save_dir=None):
    pyro.clear_param_store()
    pyro.enable_validation(True)

    scheduler= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                         'optim_args': {'lr':lr},
                          'milestones': milestones})
    scheduler2= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                            'optim_args': {'lr':lr_cla},
                            'milestones': milestones})
    elbo=Trace_ELBO()
    elbo2=Trace_ELBO()
    svi=SVI(model.model,model.guide,scheduler,elbo)
    svi2=SVI(model.model_classify,model.guide_classify,scheduler2,elbo2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.device=device
    model.to(device)
    for epoch in range(max_epochs):
        losses=[]
        losses_cls=[]
        losses_ae=[]
        if save_model and epoch%10==0:
            torch.save(model.state_dict(),os.path.join(save_dir,'model.pt'))
        for i,(rna,protein,yr,yp) in enumerate(dataloader):
            rna=rna.to(device)
            protein=protein.to(device)
            yr=yr.to(device)
            yp=yp.to(device)
            loss1=svi.step(rna,protein,yr,yp)
            loss2=svi2.step(rna,protein,yr,yp)
            losses_ae.append(loss1)
            losses_cls.append(loss2)
            losses.append(loss1+loss2)
        print('epoch :{:0>4d}  loss:{:.5f} loss_ae:{:.4f} loss_cls:{:.4f}'.format(epoch,np.mean(losses),np.mean(losses_ae),np.mean(losses_cls)))
        scheduler.step()
        scheduler2.step()
    return model

def train_scme_aux_best(model,max_epochs,dataloader,rnadata,proteindata,lr,lr_cla,milestones=[80],cell_type=None,save_model=False,save_dir=None):
    pyro.clear_param_store()
    pyro.enable_validation(True)

    scheduler= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                         'optim_args': {'lr':lr},
                          'milestones': milestones})
    scheduler2= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                            'optim_args': {'lr':lr_cla},
                            'milestones': milestones})
    if isinstance(rnadata.X,np.ndarray):
        rna_torch=torch.from_numpy(rnadata.X).float()
    else:
        rna_torch=torch.from_numpy(rnadata.X.todense()).float()
    if isinstance(proteindata.X,np.ndarray):
        protein_torch=torch.from_numpy(proteindata.X).float()
    else:
        protein_torch=torch.from_numpy(proteindata.X.todense()).float()

    rna_leiden=torch.from_numpy(dataloader.dataset.rnaleiden_onehot).float()
    protein_leiden=torch.from_numpy(dataloader.dataset.protein_leiden_onehot).float()

    elbo=Trace_ELBO()
    elbo2=Trace_ELBO()
    svi=SVI(model.model,model.guide,scheduler,elbo)
    svi2=SVI(model.model_classify,model.guide_classify,scheduler2,elbo2)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    best_ari=0
    best_nmi=0
    best_epoch=0
    for epoch in range(max_epochs):
        losses=[]
        losses_cls=[]
        losses_ae=[]

        if save_model and epoch%5==0:
            zm,zr,zp=model.inference_zm(rna_torch,protein_torch,rna_leiden,protein_leiden)
            rnadata.obsm["scme"]=zm
            sc.pp.neighbors(rnadata,n_neighbors=30,use_rep="scme")
            sc.tl.leiden(rnadata,resolution=1)
            ari=adjusted_rand_score(cell_type,rnadata.obs["leiden"])
            nmi=normalized_mutual_info_score(cell_type,rnadata.obs["leiden"])
            if ari >=best_ari or nmi>=best_nmi:
                if ari >=best_ari:
                    best_ari=ari
                if nmi>=best_nmi:
                    best_nmi=nmi
                best_epoch=epoch
                print("best ari:{:.4f} best nmi:{:.4f}".format(best_ari,best_nmi))
                torch.save(model.state_dict(),os.path.join(save_dir,'model_best.pt'))
            
        for i,(rna,protein,yr,yp) in enumerate(dataloader):
            rna=rna.to(device)
            protein=protein.to(device)
            yr=yr.to(device)
            yp=yp.to(device)
            loss1=svi.step(rna,protein,yr,yp)
            loss2=svi2.step(rna,protein,yr,yp)
            losses_ae.append(loss1)
            losses_cls.append(loss2)
            losses.append(loss1+loss2)
        print('epoch :{:0>4d}  loss:{:.5f} loss_ae:{:.4f} loss_cls:{:.4f}'.format(epoch,np.mean(losses),np.mean(losses_ae),np.mean(losses_cls)))
        scheduler.step()
        scheduler2.step()
        model.best_epoch=best_epoch
        model.best_ari=best_ari
        model.best_nmi=best_nmi
    return model


def svi_run(true_label,max_epochs=100,lr=1e-3,lr_step=[100],batch_size=100,aux_loss_multiplier=25,filenamelist=['rnafilename','efeaturesfilename','metafilename','targetcolname_inmeta'],
            latent_dim=32,if_smote=False,mydata=None,
            result_path='D:/zhoub/neuron/GANTransform/data/patchseq/000020/result',data_path='D:/zhoub/neuron/GANTransform/data/patchseq/000020'
            ):


    now=time.strftime("%Y-%m-%d-%H_%M_%S",time.localtime())
    result_path=os.path.join(result_path,f'{now}')
    if not os.path.exists(result_path):
        os.makedirs(result_path)

    embedding_path=os.path.join(result_path,f'embedding{now}')
    if not os.path.exists(embedding_path):
        os.makedirs(embedding_path)
    
    if torch.cuda.is_available():
        device = torch.device('cuda:0')
    else:
        device = torch.device('cpu')
    if mydata==None:
        mydata=MyData(data_path=data_path,rnafilename=filenamelist[0],efeaturesfilename=filenamelist[1],
                    metafilename=filenamelist[2],targetname1=filenamelist[3],targetname2=filenamelist[4],reduct='umap')
    print("lr:",lr)

    label1=mydata.meta[true_label[0]]
    label2=mydata.meta[true_label[1]]
    label1=np.array(label1)
    label2=np.array(label2)

    #分类目标
    label=np.concatenate([mydata.label1_digit,mydata.label2_digit],axis=1)
    re_train,re_test,y_train,y_test=train_test_split(mydata.rna_df,label,test_size=0.2,shuffle=True)
    train_cells,test_cells=np.array(re_train.index).squeeze(),np.array(re_test.index).squeeze()
    pd.DataFrame(train_cells).to_csv(os.path.join(result_path,'traincells.csv'))
    pd.DataFrame(test_cells).to_csv(os.path.join(result_path,'testcells.csv'))

    y1_train,y1_test=y_train[:,0],y_test[:,0]
    y2_train,y2_test=y_train[:,1],y_test[:,1]


    r_train,r_test=np.array(mydata.rna_df.loc[train_cells,:]),np.array(mydata.rna_df.loc[test_cells,:])
    e_train,e_test=np.array(mydata.efeatures_df.loc[train_cells,:]),np.array(mydata.efeatures_df.loc[test_cells,:])

    if if_smote:
        print("data smote....")
        re_train=np.concatenate((r_train,e_train),axis=1)
        re_train,y_train=SMOTE(k_neighbors=6).fit_resample(re_train,y_train)
        y_train=y_train.reshape((-1,1))
        r_train,e_train=re_train[:,:mydata.rna.shape[1]],re_train[:,mydata.rna.shape[1]:]


    pyro.clear_param_store()
    pyro.enable_validation(True)
    
    best_acc1=0
    best_acc2=0
     ##init SVIMerge
    svimerge=ScMESVI(latent_dim=latent_dim,class_num1=mydata.labelclassnum1,class_num2=mydata.labelclassnum2,alpha=50,scale_factor=1.0,aux_loss_multiplier=aux_loss_multiplier,device=device)

    svimerge.setup_network(rna_size=mydata.rnanums,ephys_size=mydata.ephysfeaturesnums)
    print("Class1 Num:",svimerge.class_num1)
    print("Class2 Num:",svimerge.class_num2)
    svimerge.to(device)

    scheduler = MultiStepLR({'optimizer': Adam,
                         'optim_args': {'lr': lr},
                         'gamma': 0.1, 'milestones': lr_step})


    # guide = config_enumerate(svimerge.guide, "sequential", expand=True)
    # elbo = TraceEnum_ELBO(strict_enumeration_warning=False)
    # svi=SVI(svimerge.model,guide,scheduler,elbo)

    elbo=Trace_ELBO()
    svi=SVI(svimerge.model,svimerge.guide,scheduler,elbo)

    elbo_2= Trace_ELBO()
    loss_aux = SVI(svimerge.model_classify, svimerge.guide_classify, scheduler, loss = elbo_2)
    elbo_3= Trace_ELBO()
    loss_aux2 = SVI(svimerge.model_classify2, svimerge.guide_classify2, scheduler, loss = elbo_3)


    trian_set=CVData(rnadata=mydata.rna,ephysdata=mydata.efeatures,label1=mydata.label1_digit,label2=mydata.label2_digit,
                        class_num1=svimerge.class_num1,class_num2=svimerge.class_num2,one_hot=True)
    mydataloader=DataLoader(dataset=trian_set,batch_size=batch_size,shuffle=True,drop_last=True)

    test_set=CVData(rnadata=r_test,ephysdata=e_test,label1=y1_test,label2=y2_test,
                    class_num1=svimerge.class_num1,class_num2=svimerge.class_num2,one_hot=True)
    testloader=DataLoader(dataset=test_set,batch_size=batch_size,shuffle=True,drop_last=True)

    r_test,e_test=torch.from_numpy(r_test).float(),torch.from_numpy(e_test).float()
    r_test,e_test=r_test.to(device),e_test.to(device)

    rdata,edata=mydata.rna,mydata.efeatures
    rdata,edata=torch.from_numpy(rdata).float(),torch.from_numpy(edata).float()
    rdata,edata=rdata.to(device),edata.to(device)


    netlog_dict={'title':f"GANTranslation{now}_logdict",'maxepoch':max_epochs,'batch_size':batch_size,
                'lr':lr,'lr_step':lr_step,'aux_loss_multiplier':aux_loss_multiplier,
                'latentdims':latent_dim,'if_smote':if_smote,'elbo1':str(elbo),
                'filename':mydata.rnafilename+'\n'+mydata.efeaturesfilename,'filenamelist':filenamelist}

    with open(os.path.join(result_path,f'Netlog{now}.txt'),'a') as f:
        f.write(str(netlog_dict))
        f.close()
    
    cells=mydata.cells

    for epoch in range(max_epochs):
        losses = []
        # Take a gradient step for each mini-batch in the dataset
        for r,e, y1,y2 in mydataloader:
            if y1 is not None:
                y1 = y1.type_as(r)
                y1=y1.to(device)
            if y2 is not None:
                y2 = y2.type_as(r)
                y2=y2.to(device)
            r=r.to(device)
            e=e.to(device)
            loss = svi.step(r,e,y1,y2)
            loss=loss+loss_aux.step(r,e,y1)
            loss=loss+loss_aux2.step(r,e,y2)
            losses.append(loss)
        
    
        scheduler.step()

        y_pre=svimerge.get_y_predict(rdata,edata)
        y_pre=np.argmax(y_pre,axis=1)
        acc=accuracy_score(y_pre,mydata.label1_digit)

        y2_pre=svimerge.get_y2_predict(rdata,edata)
        y2_pre=np.argmax(y2_pre,axis=1)
        acc2=accuracy_score(y2_pre,mydata.label2_digit)

        if acc>=best_acc1 and acc2>=best_acc2:
            zm_pre=svimerge.get_zm_predict(rdata,edata)
            df_z=pd.DataFrame(zm_pre,index=cells)
            df_z.to_csv(os.path.join(embedding_path,f'zm{now}_best.csv'))
            torch.save(svimerge,os.path.join(result_path,f"svimerge_best{now}.pt"))


        if epoch%10==0:
            zm_pre=svimerge.get_zm_predict(rdata,edata)
            df_z=pd.DataFrame(zm_pre,index=cells)
            df_z.to_csv(os.path.join(embedding_path,f'zm{epoch}.csv'))
            print("ploting...")
            two_pics_plot(zm=zm_pre,label1=label1,label2=label2,save_path=result_path,pic_title=f'{now}{epoch}',s=1,reduct='umap')
            # two_pics_plot(zm=zm_pre,label1=label1,label2=label2,save_path=result_path,pic_title=f'tsne{now}{epoch}',s=1,reduct='tsne')
                
        print("[Epoch %02d]  Loss: %.5f  ACC1:%.5f  ACC1:%.5f it:%s" % (epoch, np.mean(losses),acc,acc2,now))



    param_list=[]
    for name, value in pyro.get_param_store().items():
        param_list.append(str(name)+'\n')
    with open(os.path.join(result_path,f'Param{now}.txt'),'a') as f:
        f.write(str(param_list))
        f.close()

    zm_best=pd.read_csv(os.path.join(embedding_path,f'zm{now}_best.csv'),index_col=0)
    two_pics_plot(zm=zm_best,label1=label1,label2=label2,save_path=result_path,pic_title=f'{now}_best',s=1,reduct='umap')

    zm_pre=svimerge.get_zm_predict(rdata,edata)
    cells=mydata.cells
    df_z=pd.DataFrame(zm_pre,index=cells)
    df_z.to_csv(os.path.join(embedding_path,f'zm{now}.csv'))
    torch.save(svimerge,os.path.join(result_path,f"svimerge_last{now}.pt"))
    two_pics_plot(zm=zm_pre,label1=label1,label2=label2,save_path=result_path,pic_title=f'{now}{epoch}',s=1,reduct='umap')
    print("Finished training!")

    return now





if __name__=='__main__':

    # parser=argparse.ArgumentParser()
    # parser.add_argument('--rna',type=str,help='rna csv data path')
    # parser.add_argument('--protein',type=str,help='protein csv data path')
    # parser.add_argument('--output-dir',type=str,help='output directory')
    # parser.add_argument('--max-epochs',type=int,default=1000,help='max epochs')    
    # args=parser.parse_args()
    # rnadata=pd.read_csv(args.rna,index_col=0)
    # proteindata=pd.read_csv(args.protein,index_col=0)
    # rna,protein=rna_protein_preprocess(rnadata,proteindata)
    # traindataset=AnnDataset(rna,protein)
    # scmesvi=train_scme(dataset=traindataset,max_epochs=args.max_epochs)

    # rnatorch,proteintorch=torch.from_numpy(np.array(rnadata)),torch.from_numpy(np.array(proteindata))
    # rnatorch,proteintorch=rnatorch.to(scmesvi.device),proteintorch.to(scmesvi.device)
    # scmesvi.eval()
    # zm_pre=scmesvi.get_zm_predict(rnatorch,proteintorch)
    # zm_pre=pd.DataFrame(zm_pre,index=rnadata.index)
    # zm_pre.to_csv(os.path.join(args.output_dir,f'scME_result.csv'))


    # bm2=sc.read_h5ad("/home/zhoub/work/scME/data/sanbox/sandboxsite1.h5ad")
    # rna_df=pd.DataFrame(bm2.X.todense(),index=bm2.obs_names,columns=bm2.var_names)
    # protein_df=pd.DataFrame(bm2.obsm["protein_expression"].todense(),index=bm2.obs_names)
    # rna,protein=rna_protein_preprocess(rna_df,protein_df,res=0.5)
    sandbox=sc.read_h5ad("/home/zhoub/work/scME/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad")
    bmsite1=sandbox[sandbox.obs.Site=="site1",:]
    bmsite1_rna=bmsite1[:,bmsite1.var.feature_types=="GEX"]
    bmsite1_protein=bmsite1[:,bmsite1.var.feature_types=="ADT"]
    bmsite1_rna.X=bmsite1_rna.layers["counts"]
    bmsite1_protein.X=bmsite1_protein.layers["counts"]
    rna,protein=rna_protein_preprocess(bmsite1_rna,bmsite1_protein,res=0.5)
    rna.X=rna.layers["counts"]
    protein.X=protein.layers["counts"]
    train_dataset=AnnDataset(rna,protein,to_onehot=True)
    scme=ScMESVI_2(rna_dim=rna.shape[1],protein_dim=protein.shape[1],
        latent_dim=32,aux_loss_multiplier=500)
    l_sum=np.sum(train_dataset.rna,axis=1)
    c_sum=np.sum(train_dataset.protein,axis=1)
    l_loc,l_scale=np.mean(l_sum),np.std(l_sum)
    c_loc,c_scale=np.mean(c_sum),np.std(c_sum)
    scme.setup_network(rna_class_num=train_dataset.rna_class_num,protein_class_num=train_dataset.protein_class_num,
            l_loc=l_loc,l_scale=l_scale,c_loc=c_loc,c_scale=c_scale,
            rna_latent_dim=32,protein_latent_dim=32)
    
    train_dataloader=DataLoader(train_dataset,batch_size=128,shuffle=True)
    pt_path="/home/zhoub/work/scME/checkpoints/1"
    if os.path.exists(pt_path):
        print(pt_path)
    else:
        os.makedirs(pt_path)
    scme=train_scme_aux_best(scme,rnadata=rna,proteindata=protein,max_epochs=150,dataloader=train_dataloader,lr=1e-4,lr_cla=1e-5,
                            milestones=[100],save_model=True,save_dir=pt_path)

    rna_torch=torch.from_numpy(rna.X.todense()).float()
    protein_torch=torch.from_numpy(protein.X.todense()).float()
    rna_leiden=torch.from_numpy(train_dataset.rnaleiden_onehot).float()
    protein_leiden=torch.from_numpy(train_dataset.protein_leiden_onehot).float()
    zm,zr,zp=scme.inference_zm(rna_torch,protein_torch,rna_leiden,protein_leiden)

    zm_umap=umap.UMAP(n_components=2).fit_transform(zm)
    plt.figure(figsize=(10,10))
    umap_rep=pd.DataFrame(zm_umap,index=bmsite1.obs_names)
    umap_rep['type']=bmsite1.obs["cell_type"]
    umap_rep.columns=['x','y','type']
    sns.scatterplot(x='x',y='y',hue='type',data=umap_rep,palette='Set1',s=4)
    #set legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    now_time=time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    print(now_time)
    #save figure
    plt.savefig(f"/home/zhoub/work/scME/add_results/figures/{now_time}.png",dpi=300)
    #get best model zm
    #load model
    best_epoch=scme.best_epoch
    scme.load_state_dict(state_dict=torch.load(os.path.join(pt_path,f"model_best.pt")))
    zm,zr,zp=scme.inference_zm(rna_torch,protein_torch,rna_leiden,protein_leiden)
    zm_umap=umap.UMAP(n_components=2).fit_transform(zm)
    plt.figure(figsize=(10,10))
    umap_rep=pd.DataFrame(zm_umap,index=bmsite1.obs_names)
    umap_rep['type']=bmsite1.obs["cell_type"]
    umap_rep.columns=['x','y','type']
    sns.scatterplot(x='x',y='y',hue='type',data=umap_rep,palette='Set1',s=4)
    #set legend outside
    plt.legend(bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0.)
    plt.savefig(f"/home/zhoub/work/scME/add_results/figures/best_{now_time}.png",dpi=300)
    print(f"/home/zhoub/work/scME/add_results/figures/best_{now_time}.png")

    
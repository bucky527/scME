# gpu selection
# os.environ['CUDA_VISIBLE_DEVICES'] = "1"

import torch
import torch.nn as nn
from torch.optim import Adam
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO,JitTrace_ELBO, JitTraceEnum_ELBO,Trace_ELBO
from scvi._compat import Literal 
from scvi.data import AnnDataManager, fields
from scvi.data._constants import _MODEL_NAME_KEY, _SETUP_ARGS_KEY
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import os
import numpy as np
import pandas as pd
from data_utils import *
from pyro_utils import *
from pyroMethod import *
from mlp_net import *
import argparse

def train_scme(dataset,inputargs,max_epochs=100,batch_size=100,lr=1e-3,lr_step=[100],aux_loss_multiplier=20,latent_dim=32):
    
    if torch.cuda.is_available():
        device = torch.device("cuda")
    else:
        device = torch.device("cpu")
    
    traindataloader=DataLoader(dataset,batch_size=batch_size,shuffle=True)

    pyro.clear_param_store()
    pyro.enable_validation(True)
    svimerge=ScMESVI(rna_class_num=np.unique(dataset.rna_leiden).shape[0],protein_class_num=np.unique(dataset.protein_leiden).shape[0],
                        latent_dim=latent_dim,scale_factor=1.0,aux_loss_multiplier=aux_loss_multiplier,device=device)

    svimerge.setup_network(rna_size=dataset.rnadata.shape[1],protein_size=dataset.proteindata.shape[1],rna_latent_dim=inputargs.rna_latentdim,protein_latent_dim=inputargs.protein_latentdim)
    svimerge.to(device)
    scheduler = MultiStepLR({'optimizer': Adam,
                         'optim_args': {'lr': lr},
                         'gamma': 0.1, 'milestones': lr_step})
    elbo=Trace_ELBO()

    svi=SVI(svimerge.model,svimerge.guide,scheduler,elbo)

    elbo_2= Trace_ELBO()
    loss_aux = SVI(svimerge.model_classify, svimerge.guide_classify, scheduler, loss = elbo_2)
    elbo_3= Trace_ELBO()
    loss_aux2 = SVI(svimerge.model_classify2, svimerge.guide_classify2, scheduler, loss = elbo_3)
    for epoch in range(max_epochs):
        losses = []
        # Take a gradient step for each mini-batch in the dataset
        for r,e, y1,y2 in traindataloader:
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
        print("[Epoch %02d]  Loss: %.5f  " % (epoch, np.mean(losses)))
    return svimerge

def build_scme(rnadata,proteindata,train_dataset,protein_dist="MNB",rna_latent_dim=24,protein_latent_dim=20,latent_dim=32,if_preprocess=False):

    if if_preprocess:
        pass
    else:
        rna,protein=rna_protein_preprocess(rnadata,proteindata)
        traindataset=AnnDataset(rna,protein,to_onehot=True)
        rna.X=rna.layers["counts"]
        protein.X=protein.layers["counts"]
    if protein_dist=="MNB":
        scme_model=ScMESVI_MIXNB(rna_dim=rnadata.shape[1],protein_dim=proteindata.shape[1],
                latent_dim=latent_dim,aux_loss_multiplier=1000)
        l_sum=np.sum(train_dataset.rna,axis=1)
        c_sum=np.sum(train_dataset.protein,axis=1)
        l_loc,l_scale=np.mean(l_sum),np.std(l_sum)
        c_loc,c_scale=np.mean(c_sum),np.std(c_sum)
        scme_model.setup_network(rna_class_num=train_dataset.rna_class_num,protein_class_num=train_dataset.protein_class_num,
                        l_loc=l_loc,l_scale=l_scale,c_loc=c_loc,c_scale=c_scale,
                        rna_latent_dim=rna_latentdim,protein_latent_dim=protein_latent_dim)
    else:
        scme_model=ScMESVI_2(rna_dim=rnadata.shape[1],protein_dim=proteindata.shape[1],
                latent_dim=latent_dim,aux_loss_multiplier=1000)
        l_sum=np.sum(train_dataset.rna,axis=1)
        c_sum=np.sum(train_dataset.protein,axis=1)
        l_loc,l_scale=np.mean(l_sum),np.std(l_sum)
        c_loc,c_scale=np.mean(c_sum),np.std(c_sum)
        scme_model.setup_network(rna_class_num=train_dataset.rna_class_num,protein_class_num=train_dataset.protein_class_num,
                        l_loc=l_loc,l_scale=l_scale,c_loc=c_loc,c_scale=c_scale,
                        rna_latent_dim=rna_latent_dim,protein_latent_dim=protein_latent_dim)
    scme_model.train_dataset=train_dataset
    return scme_model

def train_model(model,max_epochs,batchsize=256,lr=1e-4,lr_cla=1e-4,milestones=[80],save_model=False,save_dir=None):
    pyro.clear_param_store()
    pyro.enable_validation(True)

    scheduler= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                         'optim_args': {'lr':lr},
                          'milestones': milestones})
    scheduler2= MultiStepLR({'optimizer': Adam,'gamma': 0.2,
                            'optim_args': {'lr':lr_cla},
                            'milestones': milestones})
    train_dataset=model.train_dataset
    train_dataloader=DataLoader(train_dataset,batch_size=batchsize,shuffle=True)                        
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
        for i,(rna,protein,yr,yp) in enumerate(train_dataloader):
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
if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--rna',type=str,help='rna count data .csv data path',required=True)
    parser.add_argument('--protein',type=str,help='protein count data .csv data path',required=True)
    parser.add_argument('--output-dir',type=str,help='output directory to save cells embeddings',required=True)
    parser.add_argument('--max-epochs',type=int,default=120,help='train max epochs')    
    parser.add_argument('--batch-size',type=int,default=100,help='train dataset batch size')
    parser.add_argument('--lr',type=float,default=1e-4,help='learning rate')
    parser.add_argument('--lr_classify',type=float,default=1e-4,help='learning rate for classify loss')
    parser.add_argument('--latentdim',type=int,default=32,help='dimension for embedding')
    parser.add_argument('--aux-loss-multiplier',type=int,default=1000,help='auxiliary loss multiplier')
    parser.add_argument('--rna-latentdim',type=int,default=24,help='rna latent dimension')
    parser.add_argument('--protein-latentdim',type=int,default=20,help='protein latent dimension')
    parser.add_argument('--lr-step',type=int,nargs='+',default=[100],help='learning rate decay step')
    parser.add_argument('--cuda',type=bool,default=True,help='use cuda')
    parser.add_argument('--use-mnb',type=bool,default=True,help='use mixture negative binomial distribution or not for proteindata')
    args=parser.parse_args()

    #read data
    rnadata=pd.read_csv(args.rna,index_col=0)
    proteindata=pd.read_csv(args.protein,index_col=0)

    rna,protein=rna_protein_preprocess(rnadata,proteindata)
    rna.X=rna.layers["counts"]
    protein.X=protein.layers["counts"]
    traindataset=AnnDataset(rna,protein,to_onehot=True)

    if args.use_mnb:
        scme_model=ScMESVI_MIXNB(rna_dim=rna.shape[1],protein_dim=protein.shape[1],
                latent_dim=args.latentdim,aux_loss_multiplier=args.aux_loss_multiplier)
        l_sum=np.sum(train_dataset.rna,axis=1)
        c_sum=np.sum(train_dataset.protein,axis=1)
        l_loc,l_scale=np.mean(l_sum),np.std(l_sum)
        c_loc,c_scale=np.mean(c_sum),np.std(c_sum)
        scme_model.setup_network(rna_class_num=train_dataset.rna_class_num,protein_class_num=train_dataset.protein_class_num,
                        l_loc=l_loc,l_scale=l_scale,c_loc=c_loc,c_scale=c_scale,
                        rna_latent_dim=args.rna_latentdim,protein_latent_dim=args.protein_latentdim)
    else:
        scme_model=ScMESVI_2(rna_dim=rna.shape[1],protein_dim=protein.shape[1],
                latent_dim=args.latentdim,aux_loss_multiplier=args.aux_loss_multiplier)
        l_sum=np.sum(train_dataset.rna,axis=1)
        c_sum=np.sum(train_dataset.protein,axis=1)
        l_loc,l_scale=np.mean(l_sum),np.std(l_sum)
        c_loc,c_scale=np.mean(c_sum),np.std(c_sum)
        scme_model.setup_network(rna_class_num=train_dataset.rna_class_num,protein_class_num=train_dataset.protein_class_num,
                        l_loc=l_loc,l_scale=l_scale,c_loc=c_loc,c_scale=c_scale,
                        rna_latent_dim=args.rna_latentdim,protein_latent_dim=args.protein_latentdim)

    #train
    train_dataloader=DataLoader(train_dataset,batch_size=args.batch_size,shuffle=True)
    scme_model=train_scme_aux(scme_model,args.max_epochs,train_dataloader,lr=args.lr,lr_cla=args.lr_classify,milestones=args.lr_step,save_model=False,save_dir=None)
    # scmesvi=train_scme(dataset=traindataset,inputargs=args,max_epochs=args.max_epochs,lr=args.lr,batch_size=args.batch_size,)

    rnatorch,proteintorch=torch.from_numpy(np.array(rna.X)),torch.from_numpy(np.array(protein.X))
    rnatorch,proteintorch=rnatorch.to(scme_model.device),proteintorch.to(scme_model.device)
    scme_model.eval()
    zm=scme_model.inference(rna, protein)
    zm=pd.DataFrame(zm,index=rnadata.index)
    zm.to_csv(os.path.join(args.output_dir,f'scME_result.csv'))
    scme_model.train()
    # scmesvi.eval()
    # zm_pre=scmesvi.get_zm_predict(rnatorch,proteintorch)
    # zm_pre=pd.DataFrame(zm_pre,index=rnadata.index)
    # zm_pre.to_csv(os.path.join(args.output_dir,f'scME_result.csv'))




import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.optim import Adam
import pyro
import pyro.distributions as dist
import pyro.poutine as poutine
from pyro.optim import MultiStepLR
from pyro.infer import SVI, config_enumerate, TraceEnum_ELBO,JitTrace_ELBO, JitTraceEnum_ELBO,Trace_ELBO
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
import os
from data_utils import *
from pyro_utils import *
from pyroMethod import *
import argparse


import os

# gpu selection
os.environ['CUDA_VISIBLE_DEVICES'] = "1"

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



if __name__=='__main__':

    parser=argparse.ArgumentParser()
    parser.add_argument('--rna',type=str,help='rna csv data path',required=True)
    parser.add_argument('--protein',type=str,help='protein csv data path',required=True)
    parser.add_argument('--output-dir',type=str,help='output directory',required=True)
    parser.add_argument('--max-epochs',type=int,default=100,help='max epochs')    
    parser.add_argument('--batch-size',type=int,default=100,help='batch size')
    parser.add_argument('--lr',type=float,default=1e-3,help='learning rate')
    parser.add_argument('--latentdim',type=int,default=32,help='latent dimension')
    parser.add_argument('--aux-loss-multiplier',type=int,default=20,help='auxiliary loss multiplier')
    parser.add_argument('--rna-latentdim',type=int,default=32,help='rna latent dimension')
    parser.add_argument('--protein-latentdim',type=int,default=20,help='protein latent dimension')

    args=parser.parse_args()
    rnadata=pd.read_csv(args.rna,index_col=0)
    proteindata=pd.read_csv(args.protein,index_col=0)
    rna,protein=rna_protein_preprocess(rnadata,proteindata)
    traindataset=AnnDataset(rna,protein,to_onehot=True)
    scmesvi=train_scme(dataset=traindataset,inputargs=args,max_epochs=args.max_epochs,lr=args.lr,batch_size=args.batch_size,)

    rnatorch,proteintorch=torch.from_numpy(np.array(rna.X)),torch.from_numpy(np.array(protein.X))
    rnatorch,proteintorch=rnatorch.to(scmesvi.device),proteintorch.to(scmesvi.device)
    scmesvi.eval()
    zm_pre=scmesvi.get_zm_predict(rnatorch,proteintorch)
    zm_pre=pd.DataFrame(zm_pre,index=rnadata.index)
    zm_pre.to_csv(os.path.join(args.output_dir,f'scME_result.csv'))



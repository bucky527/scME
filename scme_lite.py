from collections import Counter
from data_utils import *
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split 
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
import torch
import torch.nn as nn
from torch import optim, relu
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torch.nn import functional as F
import pytorch_lightning as pl
import matplotlib.pyplot as plt
import argparse
import numpy as np
import pandas as pd
import seaborn as sns
from my_module import *
from data_utils import *
import scanpy as sc
import anndata as ad


class ScMELite(pl.LightningModule):
    def __init__(self,rna_classnum,protein_classnum,rna_dim=2000,protein_dim=25,latent=32,rna_latent=32,protein_latent=32,classify_loss_ratio=0.8) -> None:
        super().__init__()
        self.rna_dim=rna_dim
        self.protein_dim=protein_dim
        self.latent=latent
        self.rna_latent=rna_latent
        self.protein_latent=protein_latent
        self.rna_encoder=Encoder(self.rna_dim,rna_latent,hidden_layers=[1000,400,128],last_relu=True)
        self.protein_encoder=Encoder(self.protein_dim,protein_latent,hidden_layers=[128,64],last_relu=True)
        self.concate_encoder=Encoder(rna_latent+protein_latent,latent,hidden_layers=[64],last_relu=True)
        self.rna_decoder=Decoder(latent,self.rna_dim,hidden_layers=[128,400,1000])
        self.protein_decoder=Decoder(latent,self.protein_dim,hidden_layers=[64,128])
        self.rna_loss=nn.MSELoss()
        self.protein_loss=nn.MSELoss()
        self.classify_loss_ratio=classify_loss_ratio
        self.rna_classifier=Classifier(self.latent,rna_classnum,hidden_layers=[64,32])
        self.protein_classifier=Classifier(self.latent,protein_classnum,hidden_layers=[64,32])
        self.save_hyperparameters()


    def forward(self,x):
        rna,protein=x
        rna=rna.float()
        protein=protein.float()
        rna_latent=self.rna_encoder(rna)
        protein_latent=self.protein_encoder(protein)
        latent=torch.cat([rna_latent,protein_latent],dim=1)
        latent=self.concate_encoder(latent)
        return latent

       
    def training_step(self, batch, batch_idx):
        rna,protein,rnaclass,proteinclass=batch
        rna=rna.float()
        protein=protein.float()
        rnaclass=rnaclass.long()
        proteinclass=proteinclass.long()
        rna_latent=self.rna_encoder(rna)  
        protein_latent=self.protein_encoder(protein)
        latent=torch.cat([rna_latent,protein_latent],dim=1)
        latent=self.concate_encoder(latent)
        rna_recon=self.rna_decoder(latent)
        protein_recon=self.protein_decoder(latent)

        rna_loss=self.rna_loss(rna_recon,rna)
        protein_loss=self.protein_loss(protein_recon,protein)
        recon_loss=rna_loss+protein_loss
        self.log('reconloss', recon_loss)
        rna_class=self.rna_classifier(latent)
        protein_class=self.protein_classifier(latent)
        rna_class_loss=F.cross_entropy(rna_class,rnaclass)
        protein_class_loss=F.cross_entropy(protein_class,proteinclass)
        class_loss=rna_class_loss+protein_class_loss
        self.log('classloss', class_loss)
        loss=recon_loss+self.classify_loss_ratio*class_loss
        self.log('train_loss', loss)
        return loss
    
    def validation_step(self, batch, batch_idx):
        rna,protein,rnaclass,proteinclass=batch
        rna=rna.float()
        protein=protein.float()
        rnaclass=rnaclass.long()
        proteinclass=proteinclass.long()
        rna_latent=self.rna_encoder(rna)
        protein_latent=self.protein_encoder(protein)
        latent=torch.cat([rna_latent,protein_latent],dim=1)
        latent=self.concate_encoder(latent)
        rna_recon=self.rna_decoder(latent)
        protein_recon=self.protein_decoder(latent)
        rna_loss=self.rna_loss(rna_recon,rna)
        protein_loss=self.protein_loss(protein_recon,protein)
        rna_class=self.rna_classifier(latent)
        protein_class=self.protein_classifier(latent)
        rna_class_loss=F.cross_entropy(rna_class,rnaclass)
        protein_class_loss=F.cross_entropy(protein_class,proteinclass)
        loss=rna_loss+protein_loss+self.classify_loss_ratio*(rna_class_loss+protein_class_loss)
        self.log('val_loss', loss)
        return loss
    
    def test_step(self, batch, batch_idx):
        rna,protein,rnaclass,proteinclass=batch
        rna=rna.float()
        protein=protein.float()
        rnaclass=rnaclass.long()
        proteinclass=proteinclass.long()
        rna_latent=self.rna_encoder(rna)
        protein_latent=self.protein_encoder(protein)
        latent=torch.cat([rna_latent,protein_latent],dim=1)
        latent=self.concate_encoder(latent)
        rna_recon=self.rna_decoder(latent)
        protein_recon=self.protein_decoder(latent)
        rna_loss=self.rna_loss(rna_recon,rna)
        protein_loss=self.protein_loss(protein_recon,protein)
        rna_class=self.rna_classifier(latent)
        protein_class=self.protein_classifier(latent)
        rna_class_loss=F.cross_entropy(rna_class,rnaclass)
        protein_class_loss=F.cross_entropy(protein_class,proteinclass)
        loss=rna_loss+protein_loss+self.classify_loss_ratio*(rna_class_loss+protein_class_loss)
        self.log('test_loss', loss)
        return loss

    
    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=1e-3)
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=100, gamma=0.1)
        return { 'optimizer': optimizer, 'lr_scheduler': scheduler}


def train_scmelite(rnadata,proteindata,batch_size=256):
    
   
    rna,protein=rna_protein_preprocess(rnadata,proteindata)
    
    mydataset=AnnDataset(rna,protein)
    #init classifier for scMELite model
    scmelite=ScMELite(rna_classnum=np.unique(mydataset.rna_leiden).shape[0],protein_classnum=np.unique(mydataset.protein_leiden).shape[0],
                        rna_dim=rna.X.shape[1],protein_dim=protein.X.shape[1])
   
    dataloader=DataLoader(mydataset,batch_size=batch_size,shuffle=True,num_workers=24)
    trainer = pl.Trainer(gpus=1, max_epochs=100)
    trainer.fit(scmelite,dataloader)

if __name__ == "__main__":

    
    adata=sc.read_h5ad('/home/zhoub/work/scME/data/GSE194122_openproblems_neurips2021_cite_BMMC_processed.h5ad')
    rna=adata[:,adata.var.feature_types=="GEX"]
    protein=adata[:,adata.var.feature_types=="ADT"]
    rna=pd.DataFrame(rna.X.todense(),columns=list(rna.var.index),index=rna.obs.index)
    protein=pd.DataFrame(protein.X.toarray(),columns=protein.var.index,index=protein.obs.index)
    train_scmelite(rna,protein)

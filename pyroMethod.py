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
from pyro_utils import *
import argparse

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
        self.alpha=alpha
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

    parser=argparse.ArgumentParser()
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



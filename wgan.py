import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
from torch.autograd import grad as torch_grad

from torch.utils.tensorboard import SummaryWriter

from functools import partial, reduce

import numpy as np
import pickle
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from umap import UMAP
import matplotlib.pyplot as plt
import plotly.express as px
import pandas as pd

import os
import gc
 
#import seaborn as sns

import math
import torch
from torch import Tensor

from torch.optim import Optimizer


 
class CustomBankDataset(Dataset):
 
    def __init__(self) -> None:
        super().__init__()
        dataset = pd.read_csv('data/bank.csv')
        dataset.rename(columns={'y': 'Yresult'}, inplace=True)
        self.raw_data = dataset
        string_cols = dataset.select_dtypes(include='object')
        self.cat_cols = string_cols.columns
        self.cat_dims = tuple(len(dataset[x].unique()) for x in self.cat_cols)
        self.orig_cols = dataset.columns
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # 
       
        print(f"dataset: {dataset.head()}")
        
        #for col in string_cols:
         #   #if (len(dataset[col]) > dataset.ndim):
          #  dataset = pd.concat([dataset, pd.get_dummies(dataset[col], prefix="oht")], axis=1)
        #dataset = pd.get_dummies(dataset, columns=string_cols.columns, prefix="oht")
        self.scaler = StandardScaler()
        self.scaler.fit(dataset[dataset.select_dtypes(exclude='object').columns])

        dataset[dataset.select_dtypes(exclude='object').columns] = self.scaler.transform(dataset[dataset.select_dtypes(exclude='object').columns]).astype('float32')
        dataset = pd.get_dummies(dataset, columns=string_cols.columns, prefix=[x for x in string_cols.columns])
        

        self.dataset = dataset.fillna(0)#.drop(columns=string_cols.columns)
        self.converted_cols = dataset.columns

    def __len__(self):
        return len(self.dataset)
 
 
    def __getitem__(self, idx) :
        return torch.tensor(self.dataset.iloc[idx]).to(self.device)

    def __get_raw_data__(self):
        return self.raw_data



def SimpleWassersteinLoss(x, y):
    return torch.mean(x * y) 

class WGAN_GEN(nn.Module):
    def __init__(self, input_dim, latent_dim) -> None:
        super().__init__()
 
        #print(f"generator input dim: {input_dim} of type {type(input_dim)}")
        self.input_dim = input_dim
        self.latent_dim = latent_dim
        self.hidden_size = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # "cuda" if torch.cuda.is_available() else
       
 
        #self.flatten = torch.flatten()
 
        self.model = nn.Sequential(
            nn.Linear(self.latent_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear((self.hidden_size), (self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size),
            nn.ReLU(),
            nn.Linear((self.hidden_size), (self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size),
            
            nn.Linear((self.hidden_size), (self.input_dim)),
        ).to(self.device)
        print(self.model)
        #NoneType is not iterable, really

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.model(x)
        return logits
 
       
 
 

class WGAN_DISC(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()
 
        self.input_dim = input_dim
        self.hidden_size = 256
        self.device = "cuda" if torch.cuda.is_available() else "cpu" # 
        
 
        #self.flatten = torch.flatten()
 
        self.model = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear(self.hidden_size, (self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear((self.hidden_size), (self.hidden_size)),
            nn.BatchNorm1d(self.hidden_size),
            nn.LeakyReLU(),
            nn.Linear((self.hidden_size), (1)),
            #nn.Sigmoid(),
            #here ks doesn't use sigmoid
        ).to(self.device)
        print(self.model)
    

    def forward(self, x):
        #x = self.flatten(x)
        logits = self.model(x)
        return logits


    
class WGAN_FEATURE_CRITIC(nn.Module):
    def __init__(self, input_dim) -> None:
        super().__init__()

        self.device = "cuda" if torch.cuda.is_available() else "cpu" #

        self.input_dim = input_dim
        self.hidden_size = 256

        self.model1 = nn.Sequential(
            nn.Linear(self.input_dim, self.hidden_size),
            nn.BatchNorm1d(self.hidden_size),   
            nn.LeakyReLU(),

        ).to(self.device)
        
        self.model2 = nn.Sequential(nn.Linear(1204, 10)) #1024, 10 in original. read paper for dimensions
     

        def forward(self, input,matching = True):
            output = self.main(input)
            feature = output.view(-1,1024)
            output = self.main2(feature)
     
            return feature, output
    


class WGAN_NET(nn.Module):
    def __init__(self, input_dim, latent_dim, orig_data) -> None:
        super().__init__()
        self.device = "cuda" if torch.cuda.is_available() else "cpu" 
        self.writer = SummaryWriter()
        self.input_dim = input_dim
        self.generator = WGAN_GEN(input_dim=input_dim, latent_dim=latent_dim)
        self.critic = WGAN_DISC(input_dim=self.input_dim)
        self.gp_weight = 10
        self.cat_dims = bank_data.cat_dims 
        self.strap_num = None
        self.orig_data = orig_data
        

    def calc_corr(logits):
        return torch.corrcoef(logits)

    #not implemented like ks
    def sample_gumbel(self, logits):
        #print(f"logit size before gump sampling: {logits.size()}")
        
        start_dim = logits.size()[1] - sum(self.cat_dims)
        #print(f"cat dims: {self.cat_dims} start dim: {start_dim}")
        for dim in self.cat_dims:  # Draw gumbel soft-max for each categorical variable
            #print(f"dim: {dim: }")
            torch.cuda.empty_cache()
            #print(f"dim for which x is allocated to tensor: {dim} ")
            temp_logits = logits[:, start_dim:start_dim + dim]
            #print(f"temp logits size pre softm: {temp_logits.size()}")

            #temp_logits = F.gumbel_softmax(logits=temp_logits, tau=0.2)
            eps=1e-20

            #TODO: devude temp by number of epochs!
            temperature=0.2
            shape = temp_logits.size()
            #print(f"shape: {shape}")
            U = torch.rand(shape).to(self.device)
            x = -torch.log(-torch.log(U + eps) + eps).to(self.device)
            #print(f"x shape: {x.shape}")
            #The size of tensor a (53) must match the size of tensor b (12) at non-singleton dimension 1
            y = temp_logits + x
            #print(f"y shape: {y.shape}")
            #sm = F.softmax(y.size())
            temp_logits = F.softmax(input=(y / temperature), dim=-1)
            #print(f"temp logits size post softm: {temp_logits.size()}")
            #print(f"logits: {logits} \n temp logits: {temp_logits}, \n start dim: {start_dim}")
            
            logits = torch.cat((logits[:, :start_dim], temp_logits, logits[:, start_dim + dim:]), dim=1)
            start_dim += dim
            #print(f"logit size after apppending: {logits.size()}")
        return logits

    def grad_penalty(self, real, fake):
        batch_size = real.size()[0]
        feature_size = real.size()[1]

        self.generator.train(False)
        # Calculate interpolation
        alpha = torch.rand(size=[batch_size, 1]) #last change: made second dim 1 and disabled expand_as
        #print(f"alpah shape: {alpha.shape}")
        
        #alpha = alpha.expand_as(real)
        
        if self.device == "cuda":
            alpha = alpha.cuda()
        
        #interpolated = alpha * real.data + (1 - alpha) * fake.data
        interpolated = alpha * real + (1 - alpha) * fake
        
        interpolated = Variable(interpolated, requires_grad=True)
        if self.device == "cuda":
            interpolated = interpolated.cuda()

        # Calculate probability of interpolated examples
        #TODO: functools partial könnte auswirkung haben
        prob_interpolated = self.critic(interpolated)

        # Calculate gradients of probabilities with respect to examples
        gradients = torch_grad(outputs=prob_interpolated, inputs=interpolated,
                               grad_outputs=torch.ones(prob_interpolated.size()).cuda() if self.device == "cuda" else torch.ones(
                               prob_interpolated.size()),
                               create_graph=True, retain_graph=True)[0]

        # Gradients have shape (batch_size, num_channels, img_width, img_height),
        # so flatten to easily take norm per example in batch
        #TODO: Experiment to see if flattening is even necessary with our input
        gradients = gradients.view(batch_size, -1)

        # Derivatives of the gradient close to 0 can cause problems because of
        # the square root, so manually calculate norm and add epsilon
        gradients_norm = torch.sqrt(torch.sum(gradients**2, dim=1) + 1e-12) ##TODO: Difference here: difference between original **2 and torch.square bzw. tf.square used in the last run. 

        gp = self.gp_weight * ((gradients_norm - 1) ** 2).mean() 

        self.generator.train()
        return gp

    def bootstrap_C(self, strap_num):
        self.generator.train(False)
        losses = torch.empty(1).to(self.device)

        
        
        #Task, speed up with lambda, map, reduce

        #compute_fake = lambda fake_pred: SimpleWassersteinLoss(fake_pred, -torch.ones_like(fake_pred))
        #compute_real = lambda real_pred: SimpleWassersteinLoss(real_pred, torch.ones_like(real_pred))
        #compute_loss = lambda fake_loss, real_loss: fake_loss + real_loss
        #store_loss  = lambda loss: losses.append(loss)

        #map(compute_fake, range(strap_num)) + map(compute_real, )
        #total_loss = reduce(lambda x: fake_loss + real_loss)
        
        
        for i in range(strap_num):
            
            batch_size = 64

            real = self.orig_data.__getitem__(np.random.randint(0,  self.orig_data.__len__()-1)).view(self.orig_data.__getitem__(0).size()[0], 1).to(self.device)

            for i in range(batch_size -1): #works
            

                real = torch.cat((real, self.orig_data.__getitem__(np.random.randint(0,  self.orig_data.__len__()-1)) #works
                .view(self.orig_data.__getitem__(0).size()[0], 1)), dim=1).to(self.device)
               

            #print(f"real sample size: {real.size()}")
            fake = self.generator(torch.rand(size=[batch_size, 128]).to(self.device))
            #print(f"fake sample size: {fake.size()}")

            fake_pred = self.critic(fake).to(self.device)
        
            real_pred = self.critic(real.T).to(self.device) #works
          
           
            fake_loss = SimpleWassersteinLoss(fake_pred, -torch.ones_like(fake_pred))
         
            real_loss = SimpleWassersteinLoss(real_pred, torch.ones_like(real_pred))
            
            loss = fake_loss + real_loss
            
            losses = torch.cat((losses, loss.view(1))).to(self.device)
            #losses.append(loss)
        loss = torch.mean(losses)
        

        self.generator.train(True)
        return loss

    def correlation_loss(real, fake):
        real_corr = torch.corrcoef(real)
        fake_corr = torch.corrcoef(fake)
        corr_loss = F.mse_loss(fake_corr, real_corr)
        return corr_loss
        
 
    def train_g(self, databatch, corr_loss=None):
        
        size = databatch.size()


        #print(f"training G with batch size: {size}")
       
        #print(f"databatch size: {len(databatch)}")
        #print(f"databatch: {databatch}")
       
        self.generator.train()

        
        noise = torch.randn(size=[size[0], 128]).to(self.device)
        #print(f"noise size: {noise.size()}")
 
        # Compute prediction error
        fake = self.generator(noise).to(self.device)
        if (self.cat_dims != None):
            #print(f"In C samplng gumbel")
            fake = self.sample_gumbel(fake).to(self.device)
        
        fake_pred = self.critic(fake)

        #print(f"generated sample from noise: {fake}")

        loss = SimpleWassersteinLoss(fake_pred, torch.ones_like(fake_pred))


        if corr_loss == True:
            loss += corr_loss(databatch, fake)


        #print(f"train_g loss size: {loss.size()}")
        
        #coeff_loss = nn.MSELoss(self.calc_corr(databatch), self.calc_corr(fake))
        #loss += coeff_loss
        
        #print(f"generator loss: {loss}")
        
 
        # Backpropagation
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-07)
        op2 = torch.optim.RAdam(self.generator.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        return loss
 
   
    def train_c(self, databatch, dataset, smoothing=None):
    
        size = databatch.size()
        self.critic.train()

        #print(f"training C with batch size: {size}")
 
        noise = torch.randn(size=[size[0], 128]).to(self.device)
       
        real = databatch.float().to(self.device)
        #print(f"real data batch used to train c: {real} {real.size()}")
 
        # Compute prediction error
        fake = self.generator(noise).to(self.device)


        if (self.cat_dims != None):
            #print(f"In C samplng gumbel")
            fake = self.sample_gumbel(fake).to(self.device)
        
        if (self.strap_num != None):
            loss = self.bootstrap_C(self.strap_num)
        else:
        
            fake_pred = self.critic(fake).to(self.device)

            #print(f"generated prediction from noise: {fake} {fake.size()}")

            real_pred = self.critic(real).to(self.device)

            #print(f"generated prediction from data: {real}")


            if (smoothing != None):
                #Smoothes the lable by the specified smoothing factor to decrease overconfidence of the critic. 

                #smooth_fake = torch.full(fake_pred.size(), (-1 + smoothing)).to(self.device)
                smooth_real = torch.full(real_pred.size(), (1 - smoothing)).to(self.device)
                #fake_loss = SimpleWassersteinLoss(fake_pred, smooth_fake)
                real_loss = SimpleWassersteinLoss(real_pred, smooth_real)

            else:
                real_loss = SimpleWassersteinLoss(real_pred, torch.ones_like(real_pred))
            
            
            fake_loss = SimpleWassersteinLoss(fake_pred, -torch.ones_like(fake_pred))
            #print(f"fake_loss {fake_loss} real_loss {real_loss}")
            loss = fake_loss + real_loss

        gp = self.grad_penalty(real, fake)

        loss += gp


        optimizer = torch.optim.Adam(self.critic.parameters(), lr=1e-4, betas=(0.5, 0.9), eps=1e-07)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        return loss
        #print(f"critic loss: {loss}")

    
    def train_feature_critic():
        pass



        
 
           
 
    def train_net(self, dataloader, epochs, cat_cols=None, strap_num=None, smoothing=0.1):
        self.cat_cols = cat_cols

        #print(f"dataloader next iter: {next(iter(dataloader))}")

        self.writer.add_graph(self.generator, torch.rand(size=[64, 128]).to(self.device))
        #TODO: Fix: currently not working: 
        self.writer.add_graph(self.critic, next(iter(dataloader)))

        for ep in range(epochs):
            print(f"training epoch: {ep}")

            
 
            for x, batch in enumerate(dataloader):
            
                #print(f"db x: {x}")
                #print(f"batch {x} size: {batch.size()}")

                #print(f"data batch: {x, x.size()}")
               
                #print(f"batch size: {batch.size()}")
                #print(f"trainig c")

                c_loss = self.train_c(batch, dataloader, smoothing)


                self.writer.add_scalar('Critic Loss',c_loss, ep)
                
                #print(f"x % 5: {x % 5}")


                if (x % 5 == 0):
                    #print(f"trainig G")
                    g_loss = self.train_g(batch)
                    self.writer.add_scalar('Generator Loss',g_loss, ep)


        self.writer.flush()
        self.writer.close()

        
               
 
            #if batch % 100 == 0:
             #       loss = loss.item()
              #      print(f"loss: {loss:>7f}")
       
    def test():
        pass

    def undummify(self, df: pd.DataFrame, prefix_sep="_"):
        print(df)
    
        cols2collapse = {
            item.split(prefix_sep)[0]: (prefix_sep in item) for item in df.columns
        }
        series_list = []
        for col, needs_to_collapse in cols2collapse.items():
            #print(f"col: {col}, needs to collapse: {needs_to_collapse}")
            if needs_to_collapse:
                #list index out of range because of .apply()[1]. for the column y there are some other values, like pdays, day instad of the wanted pairs [y, yes/no] included
                test = df.filter(like=col).idxmax(axis=1).apply(lambda x: x.split(prefix_sep, maxsplit=1))
                #print(f"test: {test}")
                undummified = (
                    df.filter(like=col)
                    .idxmax(axis=1)
                    .apply(lambda x: x.split(prefix_sep, maxsplit=1)[1])
                    .rename(col)
                )
                series_list.append(undummified)
            else:
                series_list.append(df[col])
        undummified_df = pd.concat(series_list, axis=1)
        return undummified_df
 
    def sample(self, size, cols, scaler):
        self.generator.train(False)
        if (self.device == "cuda"):
            raw_data = self.sample_gumbel(self.generator(torch.randn(size=[size, 128]).to(self.device))).detach().cpu().numpy()
        else:
            raw_data = self.sample_gumbel(self.generator(torch.randn(size=[size, 128]).to(self.device))).detach().numpy()
        #print(f"raw_data: {raw_data}")
        synth_data = self.undummify(df=pd.DataFrame(raw_data, columns=cols))#.astype('float'))

        #print(f"sample pre inverse transform {synth_data.head()}")
        synth_data[synth_data.select_dtypes(exclude='object').columns] = scaler.inverse_transform(synth_data[synth_data.select_dtypes(exclude='object').columns])
        print(f"sample pst inverse transform {synth_data.head()}")

        self.generator.train()
        return synth_data

    def visualize_cluster(self, real_data, fake_data):
        
        real_data = real_data[:fake_data.shape[1]].select_dtypes(exclude='object')
        fake_data = fake_data.select_dtypes(exclude='object')

        scaled_real = StandardScaler().fit_transform(real_data)
        scaled_fake = StandardScaler().fit_transform(fake_data)


        pca = PCA(n_components='mle')
        pca.fit(real_data)
        pca_real = (pd.DataFrame(pca.transform(scaled_real)).assign(Data='Real'))
        pca.fit(fake_data)
        pca_synthetic = (pd.DataFrame(pca.transform(scaled_fake)).assign(Data='Synthetic'))
     
        pca_plot = px.scatter(pd.concat((pca_real, pca_synthetic), axis=0), x=0, y=1, color='Data')

        tsne = TSNE()
        tsne_data = np.concatenate((scaled_real, scaled_fake), axis=0)
        tsne_result = tsne.fit_transform(tsne_data)
        tsne_plot = px.scatter(tsne_result)


        umap = UMAP()
        umap_real = (pd.DataFrame(umap.fit_transform(scaled_real)).assign(Data='Real'))
        umap_fake = (pd.DataFrame(umap.fit_transform(scaled_fake)).assign(Data='Fake'))
        umap_plot = px.scatter(pd.concat((umap_real, umap_fake), axis=0), color='Data')


        return pca_plot, tsne_plot, umap_plot
 
    def logging():
        pass
    
    def save(self, path):
        with open(path, 'wb') as f:
            pickle.dump(self, f)




if __name__ == '__main__':
    #sinkhorn = SinkhornDistance(eps=1e-3, max_iter=200)
    epochs = 10
    print(f"Starting Workflow....")

    os.chdir("C:/Users/Administrator/Documents/Imperial/Code (POCs & pre-Github)")
    print(f"working directory: {os.getcwd()}")

    print(f"Inter-Op Threads: {torch.get_num_interop_threads()} \n Intra-Op Threads: {torch.get_num_threads()}")
    

    bank_data = CustomBankDataset()
    conv_cols = bank_data.converted_cols
    cat_cols = bank_data.cat_cols
    print(f"dataset size: {bank_data.__len__()}, {len(bank_data.__getitem__(1))}, \n conv_cols: {conv_cols}")
    #print(f"bank dataset item: {bank_data.__getitem__(0)}")
 
    bank_batch = DataLoader(bank_data, batch_size=64, drop_last=True, shuffle=True)
    #print(f"data loader: {next(iter(bank_data))} \n data length: {bank_batch.__len__()} \n batch size: {next(iter(bank_data)).size()} \n batch length: {len(bank_batch)} ")
    
    cols = len(bank_data.__getitem__(1))
    print(f"Cols: {cols}")
    mywgan = WGAN_NET(cols, 128, bank_data)


    #mywgan.load_state_dict(torch.load('my_wgan_state_dict'))

    mywgan.train_net(dataloader=bank_batch, epochs=epochs, cat_cols=cat_cols)#, sinkhorn=sinkhorn)

    torch.save(mywgan, 'my_wgan_state_dict')
 
    samples = mywgan.sample(1000, conv_cols, bank_data.scaler)
    print(f"sampled: data: {samples.head()}")
    samples.rename({'Yresult': 'y'}, inplace=True)

    pca_plot, tsne_plot, umap_plot = mywgan.visualize_cluster(bank_data.__get_raw_data__(), samples)
    umap_plot.show()

    #samples.to_csv('mywgan_ks_copy_300ep_samples.csv')


    #TODO: tsne, pca & umap visualization
    

    #TODO: Disentangel the GAN for the solo GAN version. 

    #TODO: Can we use transfer learning / inductive transfer to turn the discriminator into the generator and vice versa? since the "features" to be learned are the same for both networks, could we benefit from using the 
    # discriminators learned knoledge about real data to instantly/quickly get better new generated data? Could this lead to a resolution for scenarios when the discriminator has "outtrained" the generator? How effective 
    # is this (and it's mirror variant) compared to the usual approach of doing x steps of disc training per genearator training?
    # Can wel also use transfer learnign between an decoder architecture and the generator architecture in a GAN as well as the encoder and discriminator? This yields the problem of the two networks having completely different tasks,
    # leading to the infeasability to use classical transer learning. However, could we use an attention mechanism to identify where the discriminator focuses on for successful discrimination and then direct the generator the 
    # focus on improviing the identified aera? 

    #TODO: implement feature matching meachanism as a training paradigm choice

    #TODO: implement minibatch discrimination (equivalent to PacGAN?)
  
    #TODO: check if data is absolutely equal before training starts


    #TODO: Experiment Dimensions:
    #- strapping critic (repeated prediction and averaging of gradients of critic to obtain less noisy mapping of critic's state/gradient) yes vs no
    #- label smooting (change labels to e.g. 0.9 to penalize overconfidence of critic when predicting) yes vs no
    #- correlation loss vs no correlation loss
    #- Adam vs RAdam optimizer
    #- what other purposes do the discriminator and/or generator have? What could i use them for?
    #- how much better do classifications get with mmore synthetic data added?

    #####################
    """REMINDER: Unless explicitly specified otherwise, this verions on the Home-PC will always be the main version of the code because of the GPU.""" 
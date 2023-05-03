# Copyright 2022-2023 Marcello Laurenti
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from .m_h_attn import MHAttn
from ..ncps_mod.torch import CfC
import tqdm
import numpy as np




#####################################################
############    DEFINE MODEL CLASSES   ##############
#####################################################



class CfCNet(nn.Module):
    def __init__(self,config):
        super(CfCNet,self).__init__()
        
        wirings = config['wiring']
        input_size = config['input_size']
        output_size = config['output_size']
        config_attn = config['config_attn']
        self.device = config['device']
        self.dtype = config['dtype']


        if config['activation'] == 'sigmoid':
            activation = nn.Sigmoid()
        elif config['activation'] == 'relu':
            activation = nn.LeakyReLU()
        elif config['activation'] == 'hardtanh':
            activation = nn.Hardtanh()



        self.hidden = wirings[0].units
        motor_size = wirings[0].output_dim
        self.cfc = nn.ModuleList()
        self.attn = nn.ModuleList()
        self.n_layers_cfc = config['n_layers_cfc']
        for i,wiring in enumerate(wirings):
            cfc = CfC(input_size,wiring,mixed_memory = True) if i ==0 else CfC(motor_size,wiring,mixed_memory = True)
            self.init_weight(cfc)
            self.cfc.append(cfc) 
            self.attn.append(MHAttn(config_attn))

        layers_fcn = config['layers_fcn']
        width_fcn = config['width_fcn']
        decay_fcn = config['decay_fcn']
        fcn = nn.ModuleList()
        fcn.append(nn.Linear(motor_size,width_fcn))
        for i in range(0,layers_fcn-1):
            if i != layers_fcn - 2:
                layer_in = width_fcn//(decay_fcn**i)
                layer_out = width_fcn//(decay_fcn**(i+1))
                fcn.append(nn.Linear(layer_in,layer_out))
                if config['norm']:
                    fcn.append(nn.LayerNorm(layer_out,eps = 1e-6))
                fcn.append(activation)
            else:
                layer_in = width_fcn//(decay_fcn**i)
                fcn.append(nn.Linear(layer_in,output_size))
        self.fcn = nn.Sequential(*fcn)
        self.init_weight_nn(self.fcn)
    
    def forward(self,x,t,state):
        batch_size = x.shape[0]
        h,c = state
        output = x
        new_h,new_c = self.init_hidden(batch_size)
        for i,(cfc,attn) in enumerate(zip(self.cfc,self.attn)):
            output_nonattn,(new_h[:,i],new_c[:,i]) = cfc(output,(h[:,i],c[:,i]),t)
            output = attn(output_nonattn,output_nonattn,output_nonattn)
        out = self.fcn(output[:,-1])
        return out,(new_h,new_c)

    
    def init_hidden(self,batch_size):
        weight = next(self.parameters()).data
        h = weight.new(batch_size,self.n_layers_cfc, self.hidden).zero_()
        c = weight.new(batch_size,self.n_layers_cfc, self.hidden).zero_()
        return (h.data,c.data)
    
    def init_weight(self,block):
        for name, param in block.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                nn.init.orthogonal_(param)
    def init_weight_nn(self,block):
        for name, param in block.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0.0)
            elif 'weight' in name:
                nn.init.normal_(param)


    def fit(self,loader_train,loader_test,params):
        self.train()
        epochs = params['epochs']
        lr = params['lr']
        gamma = params['gamma']
        batch_size = params['batch_size_train']
        mse = nn.MSELoss()
        optimizer = torch.optim.Adam(self.parameters(),lr = lr)
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = gamma)
        epoch_iterator = tqdm.tqdm(range(1,epochs+1),desc = 'Epochs',position = 0,leave = None)
        history = []
        for epoch in epoch_iterator:
            minibatch_iterator = tqdm.tqdm(loader_train,disable = True)
            list_error = []
            (h,c) = self.init_hidden(batch_size)
            for x_batch,y_batch,t_batch in minibatch_iterator:
                optimizer.zero_grad()
                h,c = h.data,c.data
                out,_ = self(x_batch.to(device = self.device,dtype = self.dtype),t_batch.to(device = self.device,dtype = self.dtype),(h,c))
                loss = mse(out,y_batch.to(device = self.device,dtype = self.dtype))
                loss.backward()
                list_error.append(loss.item())
                optimizer.step()
            lr_scheduler.step()
            error = np.mean(np.array(list_error))
            error_test = self.predict_t(loader_test,params)
            errors = [error,error_test]
            history.append(errors)
            epoch_iterator.set_postfix(Loss = error)

        return np.array(history)



    def predict(self,loader,config):
        self.eval()
        batch_size = config['batch_size_test']
        with torch.no_grad():
            out_predicted = []
            y_real = []
            x_used = []
            minibatch_iterator = tqdm.tqdm(loader,disable = True)
            first = False
            for x,y,t in minibatch_iterator:
                state = self.init_hidden(batch_size)
                _,state = self(x.to(device = self.device,dtype = self.dtype),t.to(device = self.device,dtype = self.dtype),state)  
                out,_ = self(x.to(device = self.device,dtype = self.dtype),t.to(device = self.device,dtype = self.dtype),state)  
                out_predicted.append(out)
                y_real.append(y.to(device = self.device,dtype = self.dtype))
                x_used.append(x[:,-1].to(device = self.device,dtype = self.dtype))
            return torch.cat(out_predicted,dim = 0),torch.cat(y_real,dim = 0),torch.cat(x_used,dim = 0)


    def predict2(self,loader,batch_size):
        self.eval()
        with torch.no_grad():
            out_predicted = []
            x_used = []
            minibatch_iterator = tqdm.tqdm(loader,disable = True)
            first = False
            for x,t in minibatch_iterator:
                state = self.init_hidden(batch_size)
                _,state = self(x.to(device = self.device,dtype = self.dtype),t.to(device = self.device,dtype = self.dtype),state)  
                out,_ = self(x.to(device = self.device,dtype = self.dtype),t.to(device = self.device,dtype = self.dtype),state)  
                out_predicted.append(out)
                x_used.append(x[:,-1].to(device = self.device,dtype = self.dtype))
            return torch.cat(out_predicted,dim = 0),torch.cat(x_used,dim = 0)


    def predict_t(self,loader,config):
        self.eval()
        batch_size = config['batch_size_test']
        with torch.no_grad():
            
            out_predicted = []
            y_used = []
            minibatch_iterator = tqdm.tqdm(loader,disable = True)
            first = False
            for x,y,t in minibatch_iterator:
                state = self.init_hidden(batch_size)
                _,state = self(x.to(device = self.device,dtype = self.dtype),t.to(device = self.device,dtype = self.dtype),state)  
                out,state = self(x.to(device = self.device,dtype = self.dtype),t.to(device = self.device,dtype = self.dtype),state)  
                out_predicted.append(out)
                y_used.append(y.to(device = self.device,dtype = self.dtype))
            mse = nn.MSELoss()
            out_pred = torch.cat(out_predicted,dim = 0)
            y_real = torch.cat(y_used,dim = 0)
            error  = mse(out_pred,y_real).item()
            return error
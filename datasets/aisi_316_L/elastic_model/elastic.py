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


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import os
import joblib

os.environ['KMP_DUPLICATE_LIB_OK']='True'

class CustomScale():
    def __init__(self,mode = 'centre'):
        self.init = False
        self.mode = mode
        
    def fit_transform(self,X):
        if not self.init:
            self.init = True
            self.min = X.min()
            self.max = X.max()
            self.centre = (self.max+self.min)/2
            self.coeff = (self.max-self.min)/2
        if self.mode == 'centre' or self.mode == 'center':
            return (X-self.centre)/(self.coeff+1e-8)
        else:
            return (X-self.min)/(self.max-self.min+1e-8)
    def inverse_transform(self,X):
        if self.mode == 'centre' or self.mode == 'center':
            return X*(self.coeff+1e-8)+self.centre
        else:
            return X*(self.max-self.min+1e-8)+self.min

T = [24,90,150,200,260,320,370,430,480,540,590,650,700,760,820]
E = [195.12,193.74,189.61,185.47,181.33,176.51,171.68,166.85,162.03,157.20,153.06,148.24,143.41,137.90,131.69] #GPa


if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
dtype = torch.float32

T = np.array(T)[:,np.newaxis]
E = np.array(E)[:,np.newaxis]

T_scaler = CustomScale('positive')
E_scaler = CustomScale('positive')

T = T_scaler.fit_transform(T)
E = E_scaler.fit_transform(E)

joblib.dump(T_scaler,'./T_scaler_elastic.gz')
joblib.dump(E_scaler,'./E_scaler_elastic.gz')


T_torch = torch.from_numpy(T).to(device = device,dtype = dtype)
E_torch = torch.from_numpy(E).to(device = device,dtype = dtype)



T_torch_train = T_torch[1:-1]
E_torch_train = E_torch[1:-1]

T_torch_test = torch.cat((T_torch[0].unsqueeze(1),T_torch[-1].unsqueeze(1))).to(device = device,dtype = dtype)
E_torch_test = torch.cat((E_torch[0].unsqueeze(1),E_torch[-1].unsqueeze(1))).to(device = device,dtype = dtype)

print(T_torch.shape)
print(T_torch_train.shape)
print(T_torch_test.shape)

class DenseNET(nn.Module):
	def __init__(self):
		super().__init__()
		layers_fcn = 3
		width_fcn = 16
		decay_fcn = 2
		output_size = 1
		input_size = 1
		norm = False
		activation = nn.Sigmoid()

		fcn = nn.ModuleList()
		fcn.append(nn.Linear(input_size,width_fcn))
		for i in range(0,layers_fcn-1):
			if i != layers_fcn - 2:
				layer_in = width_fcn//(decay_fcn**i)
				layer_out = width_fcn//(decay_fcn**(i+1))
				fcn.append(nn.Linear(layer_in,layer_out))
				if norm:
					fcn.append(nn.LayerNorm(layer_out,eps = 1e-6))
				#fcn.append(nn.Dropout(0.1))
				fcn.append(activation)
			else:
				layer_in = width_fcn//(decay_fcn**i)
				fcn.append(nn.Linear(layer_in,output_size))
		self.fcn = nn.Sequential(*fcn)
		self.init_weight_nn(self.fcn)

	def forward(self,x):
		return self.fcn(x)



	def init_weight_nn(self,block):
		for name, param in block.named_parameters():
			if 'bias' in name:
				nn.init.constant_(param, 0.0)
			elif 'weight' in name:
				nn.init.normal_(param)

model = DenseNET().to(device)

model_exist = os.path.exists('./elastic_model.pth')
override = False

if not model_exist or override:

	epochs = 1000
	lr = 5e-2
	gamma = 0.9998
	optimizer = torch.optim.Adam(model.parameters(),lr = lr)
	lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer,gamma = gamma)
	mse = nn.MSELoss()
	epoch_iterator = tqdm.tqdm(range(1,epochs+1),desc = 'Epochs',position = 0,leave = None)
	for epoch in epoch_iterator:
		model.train()
		optimizer.zero_grad()
		out = model(T_torch_train)
		loss = mse(out,E_torch_train)
		loss.backward()
		optimizer.step()
		lr_scheduler.step()
		with torch.no_grad():
			model.eval()
			out_test = model(T_torch_test)
			loss_test = mse(out_test,E_torch_test)

		message = {'Train/Test' : '{:.2e}'.format(loss.item()) + '/' + '{:.2e}'.format(loss_test.item())}
		epoch_iterator.set_postfix(message)

	with torch.no_grad():
		model.eval()
		out = model(T_torch_test)


	E_pred = out.cpu().detach().numpy()
	E_pred = E_scaler.inverse_transform(E_pred)
	E_real = E_torch_test.cpu().detach().numpy()
	E_real = E_scaler.inverse_transform(E_real)

	loss = np.sqrt(np.mean(np.square(E_pred-E_real)))


	print('RMSE Loss on Test: {}'.format(loss))

	torch.save(model.state_dict(),'./elastic_model.pth') # Saves zipped

else:
	model.load_state_dict(torch.load('./elastic_model.pth'))


T_synt = np.linspace(20,850,1000)[:,np.newaxis]
T_synt_n = T_scaler.fit_transform(T_synt)
T_synt_torch = torch.from_numpy(T_synt_n).to(device = device,dtype = dtype)


with torch.no_grad():
	model.eval()
	E_synt_torch = model(T_synt_torch)

E_synt = E_synt_torch.cpu().detach().numpy()
E_synt = E_scaler.inverse_transform(E_synt)

T = [24,90,150,200,260,320,370,430,480,540,590,650,700,760,820]

E = [195.12,193.74,189.61,185.47,181.33,176.51,171.68,166.85,162.03,157.20,153.06,148.24,143.41,137.90,131.69] #GPa

plt.plot(T_synt,E_synt,color = 'C0', label = 'Syntetic')
plt.plot(T,E,color = 'C1', label = 'Real')
plt.xlabel('Temperature °C')
plt.ylabel('Elastic Modulus GPa')
plt.legend()
plt.show()
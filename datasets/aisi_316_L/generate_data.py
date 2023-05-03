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
import pandas as pd
import json
import numpy as np
import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import TensorDataset, DataLoader
import joblib

from scipy.signal import savgol_filter as sf


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




def mkdir(path):
	if not os.path.exists(path):
		os.mkdir(path)



os.environ['KMP_DUPLICATE_LIB_OK']='True'
seed = 1
rnd = np.random.RandomState(seed)

if torch.cuda.is_available():
	device = 'cuda'
else:
	device = 'cpu'
dtype = torch.float32

model = DenseNET().to(device)
model.load_state_dict(torch.load('./elastic_model/elastic_model.pth'))

T_scaler = joblib.load('./elastic_model/T_scaler_elastic.gz')
E_scaler = joblib.load('./elastic_model/E_scaler_elastic.gz')

valid_temp = [20.0,450.0]
valid_sr = [-4.0,6.0]


##TRAIN GENERATION OF POINTS

train_temp = [30.0,400.0]
train_sr = [-3.0,5.0]
test_temp = train_temp
test_sr = train_sr

t_min,t_max = train_temp
sr_min,sr_max = train_sr

grid_sr = rnd.rand(10,10)*(sr_max-sr_min)+sr_min
grid_temp = rnd.rand(10,10)*(t_max-t_min)+t_min

#FLATTEN BOTH, COMBINE AND REMATRIX AND IT'S DONE

grid_sr = grid_sr.reshape(-1,)
grid_temp = grid_temp.reshape(-1,)

train_grid = [[sr,temp] for sr,temp in zip(grid_sr,grid_temp)]




#VALIDATION GENERATION OF POINTS

t_min_valid,t_max_valid = valid_temp
sr_min_valid,sr_max_valid = valid_sr

grid_sr_valid = rnd.rand(100,100)*(sr_max_valid-sr_min_valid)+sr_min_valid
grid_temp_valid = rnd.rand(100,100)*(t_max_valid-t_min_valid)+t_min_valid

grid_sr_valid = grid_sr_valid.reshape(-1,)
grid_temp_valid = grid_temp_valid.reshape(-1,)



sub_sr_idx = np.arange(grid_sr_valid.shape[0])
sub_temp_idx = np.arange(grid_temp_valid.shape[0])

sub_sr_idx = np.sort(rnd.choice(sub_sr_idx,100,False))
sub_temp_idx = np.sort(rnd.choice(sub_temp_idx,100,False))

grid_sr_valid = grid_sr_valid[sub_sr_idx]
grid_temp_valid = grid_temp_valid[sub_temp_idx]



sr_idx = np.ones(grid_sr_valid.shape[0],dtype = 'bool')
temp_idx = np.ones(grid_temp_valid.shape[0],dtype = 'bool')

for i in range(grid_sr_valid.shape[0]):
	if (grid_sr_valid[i] <= sr_max and grid_sr_valid[i] >= sr_min) and (grid_temp_valid[i] <= t_max and grid_temp_valid[i] >= t_min):
		sr_idx[i] = False
		temp_idx[i] = False
		


grid_sr_valid = grid_sr_valid[sr_idx]
grid_temp_valid = grid_temp_valid[temp_idx]

valid_grid = [[sr,temp] for sr,temp in zip(grid_sr_valid,grid_temp_valid)]



num_points = 10000
num_dmg = 1000
strain1 = np.linspace(0,0.004,num_points//2+1,dtype = 'float')
strain2 = np.linspace(0.004,0.10,num_points//2,dtype = 'float')
strain = np.concatenate((strain1[:-1],strain2))[:,np.newaxis]

path_data = './data/'
mkdir(path_data)

def model_data(strain,sr,temp,type = 'M1'):

	#M1
	if type == 'M1':
		A = 305.0
		B = 1161.0
		C = 0.01
		n = 0.61
		m = 0.517
		e0 = 1.0
		
	elif type == 'M2' : 
		A = 305.0
		B = 441.0
		C = 0.057
		n = 0.1
		m = 1.041
		e0 = 1.0

	elif type == 'M3' : 
		A = 301.0
		B = 1472.0
		C = 0.09
		n = 0.807
		m = 0.623
		e0 = 0.001		

	elif type == 'M4' : 
		A = 280.0
		B = 1750.0
		C = 0.1
		n = 0.8
		m = 0.85
		e0 = 200.0

	elif type == 'M5' : 
		A = 514.0
		B = 514.0
		C = 0.042
		n = 0.508
		m = 0.533
		e0 = 0.001	

	t_room = 20 #°C
	t_melt = 1399 # °C
	stress_plastic = (A + B*np.power(strain,n)) * (1 + C*np.log(sr/e0))* (1-np.power((temp-t_room)/(t_melt-t_room),m))
	temp_np = np.array([temp])[:,np.newaxis]
	temp_np_n = T_scaler.fit_transform(temp_np)
	temp_torch = torch.from_numpy(temp_np_n).to(device = device,dtype = dtype)
	with torch.no_grad():
		model.eval()
		e_torch = model(temp_torch)
	e_np_n = e_torch.cpu().detach().numpy()
	e_np = E_scaler.inverse_transform(e_np_n) #GPa
	e_np *= 1000 #MPa
	stress_elastic = strain*e_np

	elastic_idx = stress_elastic < stress_plastic
	plastic_idx = stress_elastic > stress_plastic

	stress_elastic_true = stress_elastic[elastic_idx]
	stress_plastic_true = stress_plastic[plastic_idx]

	stress = np.concatenate((stress_elastic_true,stress_plastic_true))
	#stress = sf(stress,50,2)


	
	return stress[:,np.newaxis]

data_type = 'M1'

fig,ax = plt.subplots(1,2, figsize = (7*2,5))
ax.flatten()
test_temp = [30.0,400.0]
test_sr = [-3.0,5.0]
columns = ['Time(s)','Strain(%)','Stress(MPa)','Temp(C)','Log_SR(s-1)']


gen_path = path_data + data_type + '/'
mkdir(gen_path)





counter_train = 0
counter_valid = 0
counter_test = 0
test_data = []
train_data = []
valid_data = []
for log_sr_,temp_cur in train_grid:
	sr_cur = np.power(10,log_sr_)
	time = strain/sr_cur 
	train_data.append([temp_cur,log_sr_])
	temp_array = np.ones(num_points)[:,np.newaxis]*temp_cur
	sr_array = np.ones(num_points)[:,np.newaxis]*log_sr_
	stress = model_data(strain,sr_cur,temp_cur,data_type)
	data = np.hstack((time,strain,stress,temp_array,sr_array))
	dmg_idx = rnd.choice(np.arange(1,num_points,dtype = 'int'),size = num_dmg-1,replace = False)
	dmg_idx = np.sort(np.concatenate(([0],dmg_idx)))
	data = data[dmg_idx]
	df = pd.DataFrame(data,columns = columns)
	name_data = 'train_' + str(counter_train) + '.csv'
	counter_train += 1
	df.to_csv(gen_path + name_data,sep=',', encoding='utf-8',index=False)
	ax[0].plot(data[:,1],data[:,2])
	ax[0].set(xlabel = 'Strain', ylabel = 'Stress', title = data_type)


for log_sr_,temp_cur in valid_grid:
	sr_cur = np.power(10,log_sr_)
	time = strain/sr_cur 
	valid_data.append([temp_cur,log_sr_])
	temp_array = np.ones(num_points)[:,np.newaxis]*temp_cur
	sr_array = np.ones(num_points)[:,np.newaxis]*log_sr_
	stress = model_data(strain,sr_cur,temp_cur,data_type)
	data = np.hstack((time,strain,stress,temp_array,sr_array))
	dmg_idx = rnd.choice(np.arange(1,num_points,dtype = 'int'),size = num_dmg-1,replace = False)
	dmg_idx = np.sort(np.concatenate(([0],dmg_idx)))
	data = data[dmg_idx]
	df = pd.DataFrame(data,columns = columns)
	name_data = 'valid_' + str(counter_valid) + '.csv'
	counter_valid += 1
	df.to_csv(gen_path + name_data,sep=',', encoding='utf-8',index=False)
	ax[0].plot(data[:,1],data[:,2])
	ax[0].set(xlabel = 'Strain', ylabel = 'Stress', title = data_type)


strain_test = np.linspace(0,0.10,num_dmg,dtype = 'float')[:,np.newaxis]
for t_test in test_temp:
	for sr_test in test_sr:
		test_data.append([t_test,sr_test])
		sr_cur = np.power(10,sr_test)
		time = strain_test/sr_cur
		temp_array = np.ones(num_dmg)[:,np.newaxis]*t_test
		sr_array = np.ones(num_dmg)[:,np.newaxis]*sr_test
		stress = model_data(strain_test,sr_cur,t_test,data_type)
		data = np.hstack((time,strain_test,stress,temp_array,sr_array))
		df = pd.DataFrame(data,columns = columns)
		name_data = 'test_' + str(counter_test) + '.csv'
		counter_test += 1
		df.to_csv(gen_path + name_data,sep=',', encoding='utf-8',index=False)



train_data = np.array(train_data)
test_data = np.array(test_data)
valid_data  =np.array(valid_data)
ax[1].scatter(train_data[:,0],train_data[:,1],color = 'C0',label = 'Train')
ax[1].scatter(test_data[:,0],test_data[:,1],color = 'C1',label = 'Test')
ax[1].scatter(valid_data[:,0],valid_data[:,1],color = 'C3', label = 'Valid')
ax[1].legend()





plt.show()


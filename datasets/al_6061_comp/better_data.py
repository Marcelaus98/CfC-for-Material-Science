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

import numpy as np
import pandas as pd
import torch
import sys
import os
import openpyxl 
import pyexcel as p
import matplotlib.pyplot as plt
import joblib
from copy import deepcopy
from scipy.signal import savgol_filter
import json


TITLE = 'Al 6061 Comp'

from os import listdir
from os.path import isfile, join
from  os.path import dirname as parent


directory = os.path.abspath(__file__)
directory = parent(parent(parent(directory)))
sys.path.append(directory)
import core
from core.utils import CustomScale, mkdir,adjust_time,SlidingWindowLoader


seed = 1
rnd = np.random.RandomState(seed)




def check_failure(stress):
	stress_len = stress.shape[0]
	tol = 3
	counter = 0
	for i in range(stress_len-1):
		counter += 1
		if (stress[i+1] - stress[i]) < 0:
			if np.abs(stress[i+1] - stress[i]) > tol:
				break
	return counter

def array_composition(identifier,lenght):
	comp_data = pd.read_csv(path_c + 'composition.csv')
	comp_data = comp_data.loc[comp_data['Lot'] == identifier]
	comp_data.drop(['Lot'],axis=1,inplace = True)
	comp_array = comp_data.to_numpy()*np.ones((lenght,8))
	return comp_array


#FARLA MEGLIO!!!!!!!!!!!
def find_break(stress):
	len_stress = stress.shape[0]
	tol = 0.4
	counter = 0
	for i in range(len_stress-1):
		if (stress[i] - stress[i+1]) > tol:
			break
		else:
			counter += 1
	return counter

def find_break2(strain,limit):
	len_strain = strain.shape[0]
	tol = 0.4
	counter = 0
	for i in range(len_strain):
		if strain[i] > limit:
			break
		else:
			counter += 1
	return counter


def find_break3(stress):
	len_stress = stress.shape[0]
	uts = stress.max()
	tol = 0.4
	counter = 0
	for i in range(len_stress):
		if stress[i] == uts:
			counter += 1
			break
		else:
			counter += 1
	return counter





def augment_data(df, k = 6):
    raw_out = df.drop(['Time','Strain', 'Temp','SR','Si','Fe','Cu','Mn','Mg','Cr','Zn','Ti'], axis=1).to_numpy()
    raw_in = raw.drop(['Stress'], axis=1).to_numpy()

    len_raw = raw_in.shape[0]
    cols_in = raw_in.shape[1]
    cols_out = raw_out.shape[1]
    new_raw_out = np.zeros((len_raw*k-(k-1),cols_out))
    new_raw_in = np.zeros((len_raw*k-(k-1),cols_in))

    for i in range(len_raw):
        if i != len_raw-1:
            coeff_ang = (raw_out[i+1] - raw_out[i])/(raw_in[i+1,1] - raw_in[i,1])
            intercept = (raw_in[i+1,1]*raw_out[i]-raw_in[i,1]*raw_out[i+1])/(raw_in[i+1,1] - raw_in[i,1])
            new_raw_in_temp = np.zeros((k,cols_in))

            for col_in in range(cols_in):
            	new_raw_in_temp[:,col_in] = np.linspace(raw_in[i,col_in],raw_in[i+1,col_in],k+1)[:-1]


            new_raw_out_temp = new_raw_in_temp[:,1][:,np.newaxis]*coeff_ang+intercept
            new_raw_in[i*k:i*k+k] = new_raw_in_temp
            new_raw_out[i*k:i*k+k] = new_raw_out_temp 
        else:
            new_raw_out[-1] = raw_out[-1]
            new_raw_in[-1] = raw_in[-1]
    columns = ['Time','Strain', 'Temp','SR','Si','Fe','Cu','Mn','Mg','Cr','Zn','Ti','Stress']
    data = np.hstack((new_raw_in,new_raw_out))
    df = pd.DataFrame(data,columns = columns)
    return df



path_data = './csv/'
path_numpy = './numpy/'
mkdir(path_numpy)
path_scaler = './scaler/'
mkdir(path_scaler)



columns_name = ['Time','Strain','Stress','Temp','SR','Si','Fe','Cu','Mn','Mg','Cr','Zn','Ti']

columns_name_save = ['Time','Strain(%)','Stress(MPa)','Temp(C)','SR(%/s)','Si(%)','Fe(%)','Cu(%)','Mn(%)','Mg(%)','Cr(%)','Zn(%)','Ti(%)']


#GENERATE_NEW_SCALERS

in_flat = []
out_flat = []

onlyfiles = [f for f in listdir(path_data) if isfile(join(path_data, f))]

for file in onlyfiles:
    raw = pd.read_csv(path_data + file)
    columns_name = ['Time','Strain','Stress','Temp','SR','Si','Fe','Cu','Mn','Mg','Cr','Zn','Ti']
    raw.columns = columns_name
    raw['SR'] = np.log10(raw['SR'].to_numpy())
    out_flat_temp = raw['Stress'].to_numpy()[:,np.newaxis]
    in_flat_temp = raw.drop(['Time','Stress'],axis = 1).to_numpy()
    in_flat.append(in_flat_temp)
    out_flat.append(out_flat_temp)


    
in_foo = np.concatenate(in_flat,axis = 0)
out_foo = np.concatenate(out_flat,axis = 0)





out_scaler = CustomScale('positive')
in_scaler = CustomScale('positive')

_ = in_scaler.fit_transform(in_foo)
_ = out_scaler.fit_transform(out_foo)





joblib.dump(in_scaler,path_scaler + 'in_scaler.gz')
joblib.dump(out_scaler,path_scaler + 'out_scaler.gz')




#GENRRATE SEQUENCE AND DUMP IN NUMPY FILE


test_files = ['T_020_A_1.csv','T_100_D_2.csv','T_150_G_1.csv','T_300_I_1.csv']
lookback = 100
lenght_data = 1000


raw_in_list_train = []
raw_out_list_train = []
dt_list_train = []

raw_in_list_test = []
raw_out_list_test = []
dt_list_test = []

counter_test = 0
counter_train = 0


for file in onlyfiles:
    raw = pd.read_csv(path_data + file)
    
    columns_name = ['Time','Strain','Stress','Temp','SR','Si','Fe','Cu','Mn','Mg','Cr','Zn','Ti']
    raw.columns = columns_name
    raw['SR'] = np.log10(raw['SR'].to_numpy())

    raw = augment_data(raw)
    tot_idx = np.arange(raw.shape[0])
    dmg_idx = np.sort(rnd.choice(tot_idx,lenght_data,False))

    time = raw['Time'].to_numpy()[:,np.newaxis]
    time = time[dmg_idx]
    time = np.vstack((np.arange(0,lookback)[:,np.newaxis],time-time.min()+lookback))
    d_time = adjust_time(time)
    dt_container = SlidingWindowLoader(d_time,window = lookback,padding = False,cpu = True)
    dt = dt_container.out_numpy()

    raw_in_temp = raw.drop(['Time','Stress'], axis=1).to_numpy()
    raw_in_temp = in_scaler.fit_transform(raw_in_temp)
    raw_out_temp = raw.drop(['Time','Strain', 'Temp','SR','Si','Fe','Cu','Mn','Mg','Cr','Zn','Ti'], axis=1).to_numpy()
    raw_out_temp = out_scaler.fit_transform(raw_out_temp)
    raw_in_temp = raw_in_temp[dmg_idx]
    raw_out_temp = raw_out_temp[dmg_idx]
    #print(raw_in_temp.shape)
    raw_in_container = SlidingWindowLoader(raw_in_temp,window = lookback,padding = True,cpu = True)
    raw_in = raw_in_container.out_numpy()
    raw_out = raw_out_temp

    if file in test_files:
        raw_in_list_test.append(raw_in)
        raw_out_list_test.append(raw_out)
        dt_list_test.append(dt)
        counter_test += 1
    else:
        raw_in_list_train.append(raw_in)
        raw_out_list_train.append(raw_out)
        dt_list_train.append(dt)
        counter_train += 1



x_train = np.concatenate(raw_in_list_train,axis = 0)
y_train = np.concatenate(raw_out_list_train,axis = 0)
dt_train = np.concatenate(dt_list_train,axis = 0)


x_test = np.concatenate(raw_in_list_test,axis = 0)
y_test = np.concatenate(raw_out_list_test,axis = 0)
dt_test = np.concatenate(dt_list_test,axis = 0)


np.save(path_numpy + "train_x.npy", x_train)
np.save(path_numpy + "train_y.npy", y_train)
np.save(path_numpy + "train_dt.npy", dt_train)

np.save(path_numpy + "test_x.npy", x_test)
np.save(path_numpy + "test_y.npy", y_test)
np.save(path_numpy + "test_dt.npy", dt_test)




dataset_specs = {
    'train_curves' : counter_train,
    'test_curves' : counter_test,
    'input_names' : ['Strain (%)','Temp (°C)','LogSR (s-1)','Si (%)','Fe (%)','Cu (%)','Mn (%)','Mg (%)','Cr (%)','Zn (%)','Ti (%)'],
    'output_names' : ['Stress (MPa)'],
    'title' : TITLE,
}

json_object = json.dumps(dataset_specs, indent=4)
f = open(path_numpy + "dataset_specs.json", "w")
f.write(json_object)
f.close()



##AUGMENT DATA WHEN < 1000!!!
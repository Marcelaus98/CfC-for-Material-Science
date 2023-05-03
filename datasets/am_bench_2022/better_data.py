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
import joblib
from copy import deepcopy
from scipy.signal import savgol_filter
import json
import argparse


TITLE = 'Inconel'

from os import listdir
from os.path import isfile, join
from  os.path import dirname as parent


directory = os.path.abspath(__file__)
directory = parent(parent(parent(directory)))
sys.path.append(directory)
import core
from core.utils import CustomScale, mkdir,adjust_time,SlidingWindowLoader



def cut_uts(df):
	stress = df['Stress(MPa)'].to_numpy()
	uts = stress.max()
	counter = 0
	for i in range(stress.shape[0]):
		if stress[i] < uts:
			counter += 1
		else:
			counter += 1
			break
	return df[:counter]




w = 2.54 #mm
t = 1.5 #mm
l = 5.08 #mm 

A = w*t



parser = argparse.ArgumentParser()

parser.add_argument('--no-damage', dest = 'damaged_flag', action = 'store_true', default = False, help = 'Avoid Dataset damage, not implemented yet')
args = parser.parse_args()
damaged_flag = args.damaged_flag

path_data = './csv/'
path_scaler = './scaler/'
mkdir(path_scaler)
path_numpy = './numpy/'
mkdir(path_numpy)

columns = ['Time','Strain','Stress','SR','Temp','Angle']
onlyfiles = [f for f in listdir(path_data) if isfile(join(path_data, f))]


in_flat = []
out_flat = []
for file in onlyfiles:
    raw = pd.read_csv(path_data + file)
    raw.columns = columns
    in_flat_temp = raw.drop(['Time','Stress','SR','Temp'],axis = 1).to_numpy()
    out_flat_temp = raw.drop(['Time','Strain','SR','Temp','Angle'],axis = 1).to_numpy()
    in_flat.append(in_flat_temp)
    out_flat.append(out_flat_temp)


in_foo = np.concatenate(in_flat,axis = 0)
out_foo = np.concatenate(out_flat,axis = 0)


in_scaler = CustomScale('positive')
out_scaler = CustomScale('positive')


_ = in_scaler.fit_transform(in_foo)
_ = out_scaler.fit_transform(out_foo)


joblib.dump(in_scaler,path_scaler + 'in_scaler.gz')
joblib.dump(out_scaler,path_scaler + 'out_scaler.gz')





raw_in_list_train = []
raw_out_list_train = []
dt_list_train = []

raw_in_list_test = []
raw_out_list_test = []
dt_list_test = []


test_files = ['0-XY.05-3.csv','30-XY.18-1.csv','45-XY.08-4.csv','60-XY.04-1.csv','90-XY.17-1.csv']


#DAMAGED

lookback = 100
lenght = 1000
seed = 1
rnd = np.random.RandomState(seed)

counter_test = 0
counter_train = 0

for file in onlyfiles:
    raw = pd.read_csv(path_data + file)
    raw.columns = columns
    tot_idx = np.arange(raw.shape[0])
    dmg_idx = np.sort(rnd.choice(tot_idx,size = lenght,replace = False))
    

    
    raw_in_temp = raw.drop(['Time','Stress','Temp','SR'], axis=1).to_numpy()
    raw_out_temp = raw.drop(['Time','Strain', 'Temp','SR','Angle'], axis=1).to_numpy()
    raw_in_temp = in_scaler.fit_transform(raw_in_temp)
    raw_out_temp = out_scaler.fit_transform(raw_out_temp)
    time = raw['Time'].to_numpy()[:,np.newaxis]

    raw_in_temp,raw_out_temp,time = raw_in_temp[dmg_idx],raw_out_temp[dmg_idx],time[dmg_idx]




    time = np.vstack((np.arange(0,lookback)[:,np.newaxis],time-time.min()+lookback))
    d_time = adjust_time(time)
    dt_container = SlidingWindowLoader(d_time,window = lookback,padding = False,cpu = True)
    dt = dt_container.out_numpy()


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



#TEST SCENARIO

angles_test = [0,15,30,45,60,75,90]
sr = 0.00025370340306410486 #s-1
sr_in = 4.36146471601588e-05 # s-1
ratio_len = 0.1625
len_mean = 1000 #8436
strain_mean = 0.17
len_in = int(len_mean*ratio_len)
len_fin = len_mean - len_in

artificial_in = np.linspace(0,0.0035,len_in + 1)
artificial_fin = np.linspace(artificial_in[-1],strain_mean,len_fin)
artificial_strain = np.concatenate((artificial_in[:-1],artificial_fin))[:,np.newaxis]

artificial_time_in = artificial_in/sr_in
artificial_time_fin = artificial_fin/sr

artificial_time = np.concatenate((artificial_time_in[:-1],artificial_time_fin))[:,np.newaxis]

raw_in_list_test = []
dt_list_test = []
len_valid = []


for angle in angles_test:
    angle_array = np.ones(len_mean)[:,np.newaxis]*angle
    syn_test = np.hstack((artificial_strain,angle_array))
    syn_test = in_scaler.fit_transform(syn_test)

    syn_test_container = SlidingWindowLoader(syn_test,window = lookback,padding = True,cpu = True)
    syn_test = syn_test_container.out_numpy()

    time = np.vstack((np.arange(0,lookback)[:,np.newaxis],artificial_time-artificial_time.min()+lookback))
    d_time = adjust_time(time)
    dt_container = SlidingWindowLoader(d_time,window = lookback,padding = False,cpu = True)
    dt = dt_container.out_numpy()

    raw_in_list_test.append(syn_test)
    dt_list_test.append(dt)
    len_valid.append(syn_test.shape[0])


x_test = np.concatenate(raw_in_list_test,axis = 0)
dt_test = np.concatenate(dt_list_test,axis = 0)
len_valid = np.array(len_valid)




np.save(path_numpy + "valid_x.npy", x_test)
np.save(path_numpy + "valid_dt.npy", dt_test)



dataset_specs = {
    'train_curves' : counter_train,
    'test_curves' : counter_test,
    'valid_curves' : len(angles_test),
    'input_names' : ['Strain (%)','Angle (deg)'],
    'output_names' : ['Stress (MPa)'],
    'title' : TITLE,
}

json_object = json.dumps(dataset_specs, indent=4)
f = open(path_numpy + "dataset_specs.json", "w")
f.write(json_object)
f.close()
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


TITLE = 'AISI 316L'

from os import listdir
from os.path import isfile, join
from  os.path import dirname as parent


directory = os.path.abspath(__file__)
directory = parent(parent(parent(directory)))
sys.path.append(directory)
import core
from core.utils import CustomScale, mkdir,adjust_time,SlidingWindowLoader





data_type = 'M1'
path_data = './data/' + data_type + '/'
path_scaler = './scaler/'
path_numpy = './numpy/'
mkdir(path_scaler)
mkdir(path_numpy)
onlyfiles = [f for f in listdir(path_data) if isfile(join(path_data, f))]
columns = ['Time','Strain','Stress','Temp','SR']

strains_max = []
strains_min = []
stress_max = []
stress_min = []
temps_max = []
temps_min = []
sr_max = []
sr_min = []
dtime_max = []
dtime_min = []
for file in onlyfiles:
    raw = pd.read_csv(path_data + file)
    raw.columns = columns
    strain = raw['Strain'].to_numpy()
    stress = raw['Stress'].to_numpy()
    temp = raw['Temp'].to_numpy()
    sr = raw['SR'].to_numpy()
    strains_max.append(strain.max())
    strains_min.append(strain.min())
    stress_max.append(stress.max())
    stress_min.append(stress.min())
    temps_max.append(temp.max())
    temps_min.append(temp.min())
    sr_max.append(sr.max())
    sr_min.append(sr.min())


strain_max = np.array(strains_max).max()
strain_min = np.array(strains_min).min()
stress_max = np.array(stress_max).max()
stress_min = np.array(stress_min).min()
temp_max = np.array(temps_max).max()
temp_min = np.array(temps_min).min()
sr_max = np.array(sr_max).max()
sr_min = np.array(sr_min).min()

strain_foo = np.array([strain_min,strain_max])[:,np.newaxis]
stress_foo = np.array([stress_min,stress_max])[:,np.newaxis]
temp_foo = np.array([temp_min,temp_max])[:,np.newaxis]
sr_foo = np.array([sr_min,sr_max])[:,np.newaxis]

in_foo = np.hstack((stress_foo,temp_foo,sr_foo))
out_foo = stress_foo




in_scaler = CustomScale('positive')
out_scaler = CustomScale('positive')

_ = in_scaler.fit_transform(in_foo)
_ = out_scaler.fit_transform(out_foo)


joblib.dump(in_scaler, path_scaler + 'in_scaler.gz')
joblib.dump(out_scaler, path_scaler + 'out_scaler.gz')


lookback = 100 
raw_in_list_train = []
raw_out_list_train = []
dt_list_train = []

raw_in_list_test = []
raw_out_list_test = []
dt_list_test = []


raw_in_list_valid = []
raw_out_list_valid = []
dt_list_valid = []

counter_train = 0
counter_test = 0
counter_valid = 0


for file in onlyfiles:
    raw = pd.read_csv(path_data + file)
    raw.columns = columns
    time = raw['Time'].to_numpy()[:,np.newaxis]
    time = np.vstack((np.arange(0,lookback)[:,np.newaxis],time-time.min()+lookback))
    d_time = adjust_time(time)

    raw_in_temp = raw.drop(['Time','Stress'], axis=1).to_numpy()
    raw_in_temp = in_scaler.fit_transform(raw_in_temp)
    raw_out_temp = raw.drop(['Time','Strain', 'Temp','SR'], axis=1).to_numpy()
    raw_out_temp = out_scaler.fit_transform(raw_out_temp)
    
    

    dt_container = SlidingWindowLoader(d_time,window = lookback,padding = False,cpu = True)
    dt = dt_container.out_numpy()

    raw_in_container = SlidingWindowLoader(raw_in_temp,window = lookback,padding = True,cpu = True)
    raw_in = raw_in_container.out_numpy()
    raw_out = raw_out_temp

    if 'test' in file:
        raw_in_list_test.append(raw_in)
        raw_out_list_test.append(raw_out)
        dt_list_test.append(dt)
        counter_test +=1

    elif 'valid' in file:
        raw_in_list_valid.append(raw_in)
        raw_out_list_valid.append(raw_out)
        dt_list_valid.append(dt)
        counter_valid +=1


    else:
        raw_in_list_train.append(raw_in)
        raw_out_list_train.append(raw_out)
        dt_list_train.append(dt)
        counter_train +=1
        

x_train = np.concatenate(raw_in_list_train,axis = 0)
y_train = np.concatenate(raw_out_list_train,axis = 0)
dt_train = np.concatenate(dt_list_train,axis = 0)


x_test = np.concatenate(raw_in_list_test,axis = 0)
y_test = np.concatenate(raw_out_list_test,axis = 0)
dt_test = np.concatenate(dt_list_test,axis = 0)


x_valid = np.concatenate(raw_in_list_valid,axis = 0)
y_valid = np.concatenate(raw_out_list_valid,axis = 0)
dt_valid = np.concatenate(dt_list_valid,axis = 0)


print(x_train.shape)
print(x_test.shape)
print(x_valid.shape)


np.save(path_numpy + "train_x.npy", x_train)
np.save(path_numpy + "train_y.npy", y_train)
np.save(path_numpy + "train_dt.npy", dt_train)


np.save(path_numpy + "test_x.npy", x_test)
np.save(path_numpy + "test_y.npy", y_test)
np.save(path_numpy + "test_dt.npy", dt_test)

np.save(path_numpy + "valid_x.npy", x_valid)
np.save(path_numpy + "valid_y.npy", y_valid)
np.save(path_numpy + "valid_dt.npy", dt_valid)

dataset_specs = {
    'train_curves' : counter_train,
    'test_curves' : counter_test,
    'valid_curves' : counter_valid,
    'input_names' : ['Strain (%)','Temp (°C)','LogSR (s-1)'],
    'output_names' : ['Stress (MPa)'],
    'title' : TITLE,
}

json_object = json.dumps(dataset_specs, indent=4)
f = open(path_numpy + "dataset_specs.json", "w")
f.write(json_object)
f.close()
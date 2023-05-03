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

TITLE = 'Al 6061 Aged'

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
lenght_data = 1000





lookback = 100

path_csv = './csv/'
path_numpy = './numpy/'
mkdir(path_numpy)
path_scaler = './scaler/'
mkdir(path_scaler)
in_list = []
out_list = []
columns_name = ['Time','Strain','Stress','Temp','Ang']
onlyfiles = [f for f in listdir(path_csv) if isfile(join(path_csv, f))]
for file in onlyfiles:
    raw = pd.read_csv(path_csv+file,usecols = [0,3,5,6,7],skiprows = [0])
    #print(raw.shape)
    raw.columns = columns_name
    in_list_temp = raw.drop(['Time','Stress'],axis = 1).to_numpy()
    out_list_temp = raw.drop(['Time','Strain','Temp','Ang'],axis = 1).to_numpy()
    in_list.append(in_list_temp)
    out_list.append(out_list_temp)

in_foo = np.concatenate(in_list,axis = 0)
out_foo = np.concatenate(out_list,axis = 0)

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


test_files = ['400F_HT_05.csv','775F_HT_03.csv','AR_45_03.csv','AR_RD_06.csv','AR_TD_06.csv']
problematic_files = ['400F_HT_05.csv','525F_HT_01.csv']
less_problematic_files = ['775F_HT_02.csv']

def problematic_strategy(raw):
    strains = raw['Strain']
    stresses = raw['Stress']
    temps = raw['Temp']
    angs = raw['Ang']
    times = raw['Time']
    first = True
    strain_temp = []
    stress_temp = []
    temp_temp = []
    ang_temp = []
    time_temp = []
    for strain,stress,temp,ang,time in zip(strains,stresses,temps,angs,times):
        if first:
            first = False
            strain_temp.append(strain)
            stress_temp.append(stress)
            temp_temp.append(temp)
            ang_temp.append(ang)
            time_temp.append(time)
            prev_strain = strain
        else:
            if strain > prev_strain:
                prev_strain = strain
                strain_temp.append(strain)
                stress_temp.append(stress)
                temp_temp.append(temp)
                ang_temp.append(ang)
                time_temp.append(time)

    strain_temp = np.array(strain_temp)[:,np.newaxis]
    stress_temp = np.array(stress_temp)[:,np.newaxis]
    temp_temp = np.array(temp_temp)[:,np.newaxis]
    ang_temp = np.array(ang_temp)[:,np.newaxis]
    time_temp = np.array(time_temp)[:,np.newaxis]

    data = np.hstack((time_temp,strain_temp,stress_temp,temp_temp,ang_temp))
    df = pd.DataFrame(data)


    return df


def augment_data(df, k = 2):
    raw_out = df.drop(['Time','Strain', 'Temp','Ang'], axis=1).to_numpy()
    raw_in = df.drop(['Stress'], axis=1).to_numpy()

    len_raw = raw_in.shape[0]
    cols_in = raw_in.shape[1]
    cols_out = raw_out.shape[1]
    new_raw_out = np.zeros((len_raw*k-(k-1),cols_out))
    new_raw_in = np.zeros((len_raw*k-(k-1),cols_in))

    for i in range(len_raw):
        if i != len_raw-1:
            if raw_in[i+1,1] == raw_in[i,1]:
                continue
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


    columns = ['Time','Strain','Temp','Ang','Stress']
    data = np.hstack((new_raw_in,new_raw_out))
    df = pd.DataFrame(data,columns = columns)
    return df



counter_test = 0
counter_train = 0
for file in onlyfiles:
    raw = pd.read_csv(path_csv+file,usecols = [0,3,5,6,7],skiprows = [0])
    raw.columns = columns_name

    if file in problematic_files:
        raw = problematic_strategy(raw)
        raw.columns = columns_name

    if file in less_problematic_files:
        raw_split = (raw.shape[0]//5)*2
        raw_1,raw_2 = raw[:raw_split],raw[raw_split:]
        raw_1 = problematic_strategy(raw_1)
        raw_1.columns = columns_name
        raw = pd.concat([raw_1,raw_2])
        raw.columns = columns_name

    if '775' in file:
        span = 100
    else:
        span = 50
    deg = 3

    raw['Temp'] = (raw['Temp']-32)*5/9#°C
    raw['Stress'] = raw['Stress']*0.00689476 #MPa
    raw = augment_data(raw)
    savgol3_stress = savgol_filter(raw['Stress'], span, deg)
    raw['Stress'] = savgol3_stress

    raw_in_temp = raw.drop(['Time','Stress'], axis=1).to_numpy()
    raw_in_temp = in_scaler.fit_transform(raw_in_temp)
    raw_out_temp = raw.drop(['Time','Strain', 'Temp','Ang'], axis=1).to_numpy()
    raw_out_temp = out_scaler.fit_transform(raw_out_temp)
    time_temp = raw['Time'].to_numpy()[:,np.newaxis]

    tot_idx = np.arange(raw_in_temp.shape[0])
    dmg_idx = np.sort(rnd.choice(tot_idx,lenght_data,False))
    raw_in_temp = raw_in_temp[dmg_idx]
    raw_out_temp = raw_out_temp[dmg_idx]
    time_temp = time_temp[dmg_idx]



    time_temp = np.vstack((np.arange(0,lookback)[:,np.newaxis],time_temp-time_temp.min()+lookback))
    d_time = adjust_time(time_temp)
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


dataset_specs = {
    'train_curves' : counter_train,
    'test_curves' : counter_test,
    'input_names' : ['Strain (%)','Temp (°C)','Ang (deg)'],
    'output_names' : ['Stress (MPa)'],
    'title' : TITLE,
}

json_object = json.dumps(dataset_specs, indent=4)
f = open(path_numpy + "dataset_specs.json", "w")
f.write(json_object)
f.close()
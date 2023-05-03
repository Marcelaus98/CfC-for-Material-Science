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
from torch.utils.data import TensorDataset, DataLoader
import tqdm
import sys
import numpy as np
import random

import core
from core.model import CfCNet,CfCNet_VAR
from core.ncps_mod.wirings import AutoNCP
from core.utils import mkdir
from core.utils import load_data
from core.utils import save_output
import joblib
import argparse
import json
import datetime



#####################################################
#############    DEFINE MAIN PROGRAM   ##############
#####################################################


def main(config_main):

    epochs = config_main['epochs']
    train_loader = config_main['train_loader']
    test_loader = config_main['test_loader']
    test_flag = config_main['test_flag']
    var_flag = config_main['var_flag']
    path_save  = config_main['path_save']
    batch_size_train = config_main['batch_size_train']
    batch_size_test = config_main['batch_size_test']
    in_scaler = config_main['in_scaler']
    out_scaler = config_main['out_scaler']
    device = config_main['device']
    dtype = config_main['dtype']
    input_names = config_main['input_names']
    output_names = config_main['output_names']
    title = config_main['title']
    config_model = config_main['config_model']

    config_wiring = config_model['config_wiring']
    n_layers_cfc = config_wiring['n_layers_cfc']
    wiring = []
    for _ in range(n_layers_cfc):
        wiring.append(AutoNCP(config_wiring['total_neurons'],config_wiring['motor_neurons'],sparsity_level = config_wiring['sparsity']))
    config_attn = config_model['config_attn']

    config_model = {
        'wiring' : wiring,
        'motor_neurons' :  config_wiring['motor_neurons'],
        'n_layers_cfc' : n_layers_cfc,
        'input_size' : input_size,
        'output_size' : output_size,
        'config_attn': config_attn, 
        'activation' : config_model['activation'],
        'norm' :  config_model['norm'],
        'layers_fcn' : config_model['layers_fcn'],
        'width_fcn' : config_model['width_fcn'],
        'decay_fcn' : config_model['decay_fcn'],
        'device' : device,
        'dtype' : dtype
    }


    config_model_json = {
        'config_wiring' : config_wiring,
        'config_attn': config_attn,
        'n_layers_cfc' : n_layers_cfc,
        'motor_neurons' :  config_model['motor_neurons'],
        'layers_fcn' : config_model['layers_fcn'],
        'width_fcn' : config_model['width_fcn'],
        'decay_fcn' : config_model['decay_fcn'],
        'norm' :  config_model['norm'],
        'activation' : config_model['activation'],
        'variational' : var_flag,
        'input_size' : config_model['input_size'],
        'output_size' : config_model['output_size'],
        'lr' : config_model['lr'],
        'gamma' : comfig_model['gamma']
    }

    model = CfCNet_VAR(config_model).to(device = device,dtype = dtype) if var_flag else CfCNet(config_model).to(device = device,dtype = dtype)


    epochs_train = 2 if test_flag else epochs
    train_config = {
        'epochs' : epochs_train,
        'lr' : config_model['lr'],
        'gamma': comfig_model['gamma']
        'batch_size_train' : batch_size_train,
        'batch_size_test' : batch_size_test,
    }



    history = model.fit(train_loader,test_loader,train_config)
    epochs = np.arange(1,history.shape[0]+1)

    if var_flag:

        mu_test_pred,var_test_pred,y_test_real,x_test_used = model.predict(test_loader,train_config)
        mu_test_pred_numpy = mu_test_pred.cpu().detach().numpy()
        var_test_pred_numpy = var_test_pred.cpu().detach().numpy()

    else:
        y_test_pred,y_test_real,x_test_used = model.predict(test_loader,train_config)
        y_test_pred_numpy = y_test_pred.cpu().detach().numpy()

    y_test_real_numpy = y_test_real.cpu().detach().numpy()
    x_test_used_numpy = x_test_used.cpu().detach().numpy()

    if var_flag:
        mu_test_pred_numpy = out_scaler.inverse_transform(mu_test_pred_numpy)
        var_test_pred_numpy = out_scaler.inverse_transform(var_test_pred_numpy)
    else:
        y_test_pred_numpy = out_scaler.inverse_transform(y_test_pred_numpy)

    y_test_real_numpy = out_scaler.inverse_transform(y_test_real_numpy)
    x_test_used_numpy = in_scaler.inverse_transform(x_test_used_numpy)



    test_pred_numpy = (mu_test_pred_numpy,var_test_pred_numpy) if var_flag else y_test_pred_numpy
    names = (input_names,output_names,title)
    save_test = (x_test_used_numpy,y_test_real_numpy,test_pred_numpy,test_curves,var_flag)
    save_run = (history,epochs,model)
    save_variables = (save_test,save_run,config_model_json,names)

    save_output(path_save,save_variables)









#GLOBAL VARIABLES


parser = argparse.ArgumentParser()

parser.add_argument('--test', dest = 'test_flag', action = 'store_true', default = False, help = 'Set Epochs to 2 for testing before deploying. By Default is disabled')
parser.add_argument('-epochs', dest = 'epochs', action = 'store', default = 500, help = 'Set Train Epochs. Default is 500')
parser.add_argument('--var', dest = 'var_flag', action = 'store_true', default = False, help = 'Enable Variational Approach. By Default is disabled')
parser.add_argument('-dataset', dest = 'dataset', action = 'store', default = 'aisi', help = 'Select the proper dataset, available datasets are: AISI, AL_COMP, AL_AGED, \
    INCONEL, STEEL, CUSTOM. Default is AISI. More information on the README doc')
parser.add_argument('-batch_train', dest = 'batch_train', action = 'store', default = 1000, help = 'Set Train Batch Size. Default is 1000')
parser.add_argument('-batch_test', dest = 'batch_test', action = 'store', default = 1000, help = 'Set Test Batch Size. Default is 1000')
parser.add_argument('--cuda', dest = 'cuda_flag', action = 'store_true', default = False, help = 'Use CUDA instead of CPU. Default is CPU')
parser.add_argument('--float16', dest = 'float16_flag', action = 'store_true', default  = False, help = 'Use float16 instead of float32. Default is float32')
parser.add_argument('-seed',dest = 'seed',action = 'store', default = 1, help = 'Assign custom seed value for random process. Default is 1')

args = parser.parse_args()
test_flag = args.test_flag
var_flag = args.var_flag
cuda_flag = args.cuda_flag
float16_flag  = args.float16_flag
epochs = int(args.epochs)
seed = int(args.seed)
rnd = np.random.RandomState(seed)
torch.manual_seed(seed)
random.seed(seed)

if args.dataset == 'Al_Comp' or args.dataset == 'al_comp' or args.dataset == 'AL_COMP':
    dataset_type = 'al_6061_comp'
elif args.dataset == 'Al_aged' or args.dataset == 'al_aged' or args.dataset == 'AL_AGED':
    dataset_type = 'al_6061_aging'
elif args.dataset == 'inconel' or args.dataset == 'INCONEL' or args.dataset == 'Inconel':
    dataset_type = 'am_bench_2022'
elif args.dataset == 'aisi' or args.dataset == 'Aisi' or args.dataset == 'AISI':
    dataset_type = 'aisi_316_L'
elif args.dataset == 'steel' or args.dataset == 'Steel' or args.dataset == 'STEEL':
    dataset_type = 'steel_spring'
elif args.dataset == 'custom' or args.dataset == 'CUSTOM' or args.dataset == 'Custom':
    dataset_type = 'custom'
    if not os.path.exists('./datasets/custom/numpy/') or not os.path.exists('./datasets/custom/scaler/'):
        sys.exit('Please Insert data in Custom folder compliant with rules present in the README doc')
else:
    sys.exit('Please Insert a valid dataset. For more informations read the README doc')


path_data = './datasets/' + dataset_type + '/'
path_scaler = path_data + 'scaler/'
path_numpy = path_data + 'numpy/'

with open(path_numpy + 'dataset_specs.json', 'r') as openfile:
 
    # Reading from json file
    json_object = json.load(openfile)



train_curves = json_object['train_curves']
test_curves = json_object['test_curves']
input_names = json_object['input_names']
output_names = json_object['output_names']
title = json_object['title']

batch_size_train = int(args.batch_train)
batch_size_test = int(args.batch_test)


#IMPROVE THIS SECTION TO ENABLE DATASET CHOICE
x_train,y_train,dt_train,x_test,y_test,dt_test = load_data(path_numpy,dataset_type)

in_scaler = joblib.load(path_scaler + 'in_scaler.gz')
out_scaler = joblib.load(path_scaler + 'out_scaler.gz')

input_size = x_train.shape[2]
output_size = y_train.shape[1]


if cuda_flag:
    if torch.cuda.is_available():
        device = 'cuda'
        torch.backends.cudnn.benchmark = True
    else:
        print('CUDA not available on this machine, overriding to cpu')
        device = 'cpu'
else:
    device = 'cpu'

if float16_flag:
    dtype = torch.float16
else:
    dtype = torch.float32


x_train_torch = torch.from_numpy(x_train)#.to(device = device,dtype = dtype)
y_train_torch = torch.from_numpy(y_train)#.to(device = device,dtype = dtype)
dt_train_torch = torch.from_numpy(dt_train)#.to(device = device,dtype = dtype)

x_test_torch = torch.from_numpy(x_test)#.to(device = device,dtype = dtype)
y_test_torch = torch.from_numpy(y_test)#.to(device = device,dtype = dtype)
dt_test_torch = torch.from_numpy(dt_test)#.to(device = device,dtype = dtype)

train_dataset = TensorDataset(x_train_torch,y_train_torch,dt_train_torch)
train_loader = DataLoader(train_dataset,shuffle = True,drop_last = True,batch_size = batch_size_train,pin_memory = True)

test_dataset = TensorDataset(x_test_torch,y_test_torch,dt_test_torch)
test_loader = DataLoader(test_dataset,shuffle = False,drop_last = True,batch_size = batch_size_test,pin_memory = True)

path_results_general = './results/'
mkdir(path_results_general)
path_results_model = path_results_general + args.dataset.lower() + '/'
mkdir(path_results_model)
if var_flag:
    path_results_model_type = path_results_model + 'var/'
else:
    path_results_model_type = path_results_model + 'normal/'

mkdir(path_results_model_type)

now = datetime.datetime.now()
timestamp = now.strftime("%Y_%m_%d_%H_%M")
path_save = path_results_model_type + timestamp + '/'
mkdir(path_save)



with open('./config_model.json', 'r') as openfile:
 
    # Reading from json file
    config_model = json.load(openfile)

config_main = {
    'train_loader' : train_loader,
    'test_loader' : test_loader,
    'input_names' : input_names,
    'output_names': output_names,
    'title' : title,
    'var_flag' : var_flag,
    'test_flag' : test_flag,
    'batch_size_train' : batch_size_train,
    'batch_size_test' : batch_size_test,
    'epochs' : epochs,
    'path_save' : path_save,
    'in_scaler' : in_scaler,
    'out_scaler' : out_scaler,
    'device' : device,
    'dtype' : dtype,
    'config_model' : config_model,
}


if __name__ == '__main__':
    main(config_main)
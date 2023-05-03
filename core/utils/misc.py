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
import torch
import os
from copy import deepcopy
import json
import pandas as pd
import os
from os import listdir
from os.path import isfile, join
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

os.environ['KMP_DUPLICATE_LIB_OK']='True'





def mkdir(path):
    isExist = os.path.exists(path)
    if not isExist:
        os.makedirs(path)



def del_tree(path):
    for f in os.listdir(path):
        joined_path = os.path.join(path, f)
        if os.path.isfile(joined_path):
            os.remove(joined_path)
        else:
            del_tree(joined_path)
            os.rmdir(joined_path)


def adjust_time(t):
    delta_t = np.zeros((t.shape),dtype = float)
    t_pad = np.vstack((np.zeros((1,1),dtype = float),t))
    for i in range(delta_t.shape[0]):
        delta_t[i] = deepcopy(t_pad[i+1] - t_pad[i])
    return delta_t



def load_data(path,dataset_type):




    x_train = np.load(path + 'train_x.npy',allow_pickle = True)
    y_train = np.load(path + 'train_y.npy',allow_pickle = True)
    dt_train = np.load(path + 'train_dt.npy',allow_pickle = True)


    x_test = np.load(path + 'test_x.npy',allow_pickle = True)
    y_test = np.load(path + 'test_y.npy',allow_pickle = True)
    dt_test = np.load(path + 'test_dt.npy',allow_pickle = True)
        

    return x_train,y_train,dt_train,x_test,y_test,dt_test


def save_output(path,save_variables):
    save_test,save_run,config_model_json,names = save_variables
    input_names,output_names,title = names
    history,epochs,model = save_run
    x_test,y_test,test_pred,test_curves,var_flag = save_test

    path_plot = path + 'plots/'
    mkdir(path_plot)
    path_data = path + 'data/'
    mkdir(path_data)

    x_test_split = np.array_split(x_test,test_curves)
    y_test_split = np.array_split(y_test,test_curves)

    if var_flag:
        y_test_pred,var_test_pred = test_pred
        y_test_pred_split = np.array_split(y_test_pred,test_curves)
        var_test_pred_split = np.array_split(var_test_pred,test_curves)
    else:
        y_test_pred_split = np.array_split(test_pred,test_curves)
        var_test_pred_split = y_test_pred_split

    name_figs = []
    save_strings = []
    for counter,(x_test_cur,y_test_cur,mu_test_pred_cur,var_test_pred_cur) in enumerate(zip(x_test_split,y_test_split,y_test_pred_split,var_test_pred_split)):

        fig,ax = plt.subplots(1,1)
        ax.plot(x_test_cur[:,0],y_test_cur,'C1')
        ax.plot(x_test_cur[:,0],mu_test_pred_cur,'C0')
        if var_flag:
            lower,upper = mu_test_pred_cur-np.sqrt(var_test_pred_cur),mu_test_pred_cur+np.sqrt(var_test_pred_cur)
            ax.fill_between(x_test_cur[:,0],lower[:,-1],upper[:,-1],color = 'C0',alpha = 0.3)

        #MAKE GENERAL THE TITLE AND THE SAVING, RIGHT NOW IT IS NOT GOOD


        name_fig = title + " "
        for i in range(1,len(input_names)):
            name_fig += str(int(x_test_cur[0,i])) + input_names[i].split(" ")[1].replace("(","").replace(")","") + " "
        
        save_string = str(counter)
        name_figs.append(name_fig)
        save_strings.append(save_string)
        ax.set(xlabel = input_names[0],ylabel = output_names[0], title = name_fig)
        if var_flag:
            ax.legend(['Real','Mean Predicted Test','Variance Predicted Test'])
        else:
            ax.legend(['Real','Predicted Test'])
        fig.savefig(path_plot+ "prediction_test_"+ save_string + ".jpg")
        plt.close(fig)


    fig,ax = plt.subplots(1,1)
    ax.plot(epochs,history[:,0],'C0')
    ax.plot(epochs,history[:,1],'C1')
    if var_flag:
        ax.set(xlabel = 'Epochs',ylabel = 'NLL', title = 'Error')
    else:
        ax.set(xlabel = 'Epochs',ylabel = 'logMSE', title = 'Error')
        ax.set_yscale('log')
    ax.legend(['Train','Test'])
    fig.savefig(path_plot+ "error.jpg")
    plt.close(fig)





    ##  CSV FILES

    txt_error = np.hstack((epochs[:,np.newaxis],history))
    data = pd.DataFrame(txt_error,columns = ["Epochs", "Error_Train","Error_Test"])
    data.to_csv(path_data + "error_data.csv", sep=',', encoding='utf-8',index=False)

    for counter,(x_test_cur,y_test_cur,mu_test_pred_cur,var_test_pred_cur) in enumerate(zip(x_test_split,y_test_split,y_test_pred_split,var_test_pred_split)):


        #MAKE GENERAL THE TITLE AND THE SAVING, RIGHT NOW IT IS NOT GOOD


        save_string = str(counter)

        if var_flag:
            lower,upper = mu_test_pred_cur-np.sqrt(var_test_pred_cur),mu_test_pred_cur+np.sqrt(var_test_pred_cur)
            data_fig_test = np.hstack((x_test_cur,y_test_cur,mu_test_pred_cur,lower,upper))
            columns_in = [s.split(" ")[0] + "_Test" for s in input_names]
            columns_out = [s.split(" ")[0] + "_Test" for s in output_names]
            columns_out_pred = [s.split(" ")[0] + "_Pred_Test" for s in output_names]
            columns_lower_pred = [s.split(" ")[0] + "_Lower_Pred_Test" for s in output_names]
            columns_upper_pred = [s.split(" ")[0] + "_Upper_Pred_Test" for s in output_names]
            columns = columns_in + columns_out + columns_out_pred + columns_lower_pred + columns_upper_pred
        else:
            data_fig_test = np.hstack((x_test_cur,y_test_cur,mu_test_pred_cur))
            columns_in = [s.split(" ")[0] + "_Test" for s in input_names]
            columns_out = [s.split(" ")[0] + "_Test" for s in output_names]
            columns_out_pred = [s.split(" ")[0] + "_Pred_Test" for s in output_names]
            columns = columns_in + columns_out + columns_out_pred


        data_fig_test = pd.DataFrame(data_fig_test,columns = columns)

        data_fig_test.to_csv(path_data + "test_pred_data_"+ save_string + ".csv", sep=',', encoding='utf-8',index=False)

    ##  MODEL STATE DICTIONARY

    torch.save(model.state_dict(), path + 'model_noZip.pth', _use_new_zipfile_serialization=False) # Saves pickle
    torch.save(model.state_dict(), path + 'model.pth') # Saves zipped

    #SAVE&UPLOAD JSON CONFIG FILE

    json_object = json.dumps(config_model_json, indent=4)
    f = open(path + "config_model.json", "w")
    f.write(json_object)
    f.close()


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
import os
import sys
import torch
from copy import deepcopy




#####################################################
##########    DEFINE SeqCreator CLASSES   ###########
#####################################################





class SlidingWindow():
    def __init__(self,data_inputs,window,padding = False,cpu = False):
        if torch.cuda.is_available() and not cpu:
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        if not isinstance(data_inputs,torch.Tensor):
            self.data_inputs = torch.from_numpy(data_inputs).to(dtype = torch.float32)
        else:
            self.data_inputs = data_inputs
            self.data_inputs = self.data_inputs.to(dtype = torch.float32)
            
        if window > self.data_inputs.size(0):
            print("Window too large, setting equal to input first dim")
            self.window = self.data_inputs.size(0)
        else:
            self.window = window
            
        
        if padding:
            dim_inputs_vect = self.data_inputs.shape
            first_inputs = True
            dim_inputs_count = 0
            dim_inputs_pad = []
            for dim_inputs in dim_inputs_vect:
                if first_inputs:
                    first_inputs = False
                    dim_inputs_pad.append(window)
                    continue
                dim_inputs_pad.append(dim_inputs)
            padding_inputs = torch.zeros(torch.Size(dim_inputs_pad), dtype = torch.float32)
            self.data_inputs = torch.cat((padding_inputs, self.data_inputs), dim = 0)
            
        self.final_shape = self.data_inputs.size(0)
        self.n_windows = self.final_shape-self.window
        self.position = 0

    def out(self):
        inputs = self.data_inputs[self.position:self.window+self.position]
        return inputs.to(device = self.device)
    
    def step(self):
        if self.position >= self.final_shape - self.window:
            print("Edge of data reached")
        else:
            self.position += 1
    
    def reset(self):
        self.position = 0 



class SlidingWindowLoader():
    def __init__(self,data_inputs,window = None,padding = False,cpu = False):            
        if window is None:
            sys.exit("Please enter window value")
        
        inputs_windowed = SlidingWindow(data_inputs,window,padding,cpu)
        self.device = inputs_windowed.device
    
        inputs = None
        for window in range(inputs_windowed.n_windows):
            inputs_windowed.step()
            if inputs == None:
                inputs = inputs_windowed.out()
                inputs = torch.unsqueeze(inputs,dim = 0)
            else:
                inputs_temp = inputs_windowed.out()
                inputs_temp = torch.unsqueeze(inputs_temp,dim = 0)
                inputs = torch.cat((inputs,inputs_temp),dim = 0)
                
                
        self.inputs = inputs
        
    def out_numpy(self):
        if self.device != 'cpu':
            inputs = self.inputs.cpu()
        else:
            inputs = self.inputs  
        inputs = inputs.numpy()
        return inputs
    
    def out_torch(self):
        return self.inputs
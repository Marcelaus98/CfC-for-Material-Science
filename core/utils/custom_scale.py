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


#####################################################
#########    DEFINE CUSTOM NORMALIZE DATA   #########
#####################################################


class CustomScale():
    def __init__(self,mode = 'centre'):
        self.init = False
        self.mode = mode
        
    def fit_transform(self,X):
        if not self.init:
            self.init = True
            self.min = X.min(axis = 0)
            self.max = X.max(axis = 0) 
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


     
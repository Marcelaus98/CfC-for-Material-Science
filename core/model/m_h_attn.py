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
import torch.nn as nn
import torch.nn.functional as F
import sys



#####################################################
############    DEFINE MODEL CLASSES   ##############
#####################################################




class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature, attn_dropout=0.1):
        super().__init__()
        self.temperature = temperature
        self.dropout = nn.Dropout(attn_dropout)

    def forward(self, q, k, v, mask=None):

        attn = torch.matmul(q / self.temperature, k.transpose(2, 3))

        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)

        attn = self.dropout(F.softmax(attn, dim=-1))
        output = torch.matmul(attn, v)

        return output




class MHAttn(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self, config):
        super().__init__()
        key_size = config['key_size']
        query_size = config['query_size']
        value_size = config['value_size']
        self.heads = config['heads']
        dropout = config['dropout']
    
        if value_size%self.heads != 0:
            sys.exit("Please enter a number of heads multiple of the values dimentions")
        self.dv = int(value_size/self.heads)
        symmetric = config['symmetric']
        if symmetric:
            self.dk = self.dv
        else:
            self.dk = config['dk']
        self.w_qs = nn.Linear(query_size, self.heads * self.dk, bias=False)
        self.w_ks = nn.Linear(key_size, self.heads * self.dk, bias=False)
        self.w_vs = nn.Linear(value_size, self.heads * self.dv, bias=False)
        self.fc = nn.Linear(self.heads * self.dv, value_size, bias=False)

        self.attention = ScaledDotProductAttention(temperature=self.dk ** 0.5,attn_dropout = dropout)

        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(value_size, eps=1e-6)


    def forward(self, q, k, v, mask=None):

        d_k, d_v, n_head = self.dk, self.dv, self.heads
        sz_b, len_q, len_k, len_v = q.size(0), q.size(1), k.size(1), v.size(1)

        residual = q

        # Pass through the pre-attention projection: b x lq x (n*dv)
        # Separate different heads: b x lq x n x dv
        q = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v = self.w_vs(v).view(sz_b, len_v, n_head, d_v)

        # Transpose for attention dot product: b x n x lq x dv
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)

        if mask is not None:
            mask = mask.unsqueeze(1)   # For head axis broadcasting.

        q= self.attention(q, k, v, mask=mask)

        # Transpose to move the head dimension back: b x lq x n x dv
        # Combine the last two dimensions to concatenate all the heads together: b x lq x (n*dv)
        q = q.transpose(1, 2).contiguous().view(sz_b, len_q, -1)
        q = self.dropout(self.fc(q))
        q += residual

        q = self.layer_norm(q)

        return q
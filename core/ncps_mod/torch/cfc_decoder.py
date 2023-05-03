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
from torch import nn
from typing import Optional, Union
from ... import ncps_mod
from . import CfCCell, WiredCfCCell
from .lstm import LSTMCell


class CfCDecoder(nn.Module):
    def __init__(
        self,
        input_size: Union[int, ncps_mod.wirings.Wiring],
        output_size : int,
        units,
    ):
        """Applies a `Closed-form Continuous-time <https://arxiv.org/abs/2106.13898>`_ RNN to an input sequence.

        Examples::

             >>> from ncps.torch import CfC
             >>>
             >>> rnn = CfC(20,50)
             >>> x = torch.randn(2, 3, 20) # (batch, time, features)
             >>> h0 = torch.zeros(2,50) # (batch, units)
             >>> output, hn = rnn(x,h0)

        :param input_size: Number of input features
        :param units: Number of hidden units
        :param proj_size: If not None, the output of the RNN will be projected to a tensor with dimension proj_size (i.e., an implict linear output layer)
        :param return_sequences: Whether to return the full sequence or just the last output
        :param batch_first: Whether the batch or time dimension is the first (0-th) dimension
        :param mixed_memory: Whether to augment the RNN by a `memory-cell <https://arxiv.org/abs/2006.04418>`_ to help learn long-term dependencies in the data
        :param mode: Either "default", "pure" (direct solution approximation), or "no_gate" (without second gate).
        :param activation: Activation function used in the backbone layers
        :param backbone_units: Number of hidden units in the backbone layer (default 128)
        :param backbone_layers: Number of backbone layers (default 1)
        :param backbone_dropout: Dropout rate in the backbone layers (default 0)
        """

        super(CfCDecoder, self).__init__()
        self.input_size = input_size
        self.wiring_or_units = units
        self.wiring = units
        self.state_size = self.wiring.units
        self.motor_size = self.wiring.output_dim
        self.output_size = output_size
        self.rnn_cell = WiredCfCCell(
            input_size,
            self.wiring_or_units,
            'default',
        )

        self.lstm = LSTMCell(input_size, self.state_size)


        #ATTENTION MECHANISM
        self.fcn_dec = nn.Linear(self.state_size*2,self.state_size,bias = False)
        self.fcn_enc = nn.Linear(self.motor_size,self.state_size)
        self.act = nn.Tanh()
        self.fcn_score = nn.Sequential(
            nn.Linear(self.state_size,1,bias = False),
            nn.Softmax(dim = 1))
        self.fcn_y_attn = nn.Linear(self.motor_size+self.input_size,self.input_size)
        self.fcn_out = nn.Sequential(
            nn.Linear(2*self.motor_size,self.motor_size),
            nn.LeakyReLU(),
            nn.Linear(self.motor_size,self.output_size))



        
    def forward(self, input,h_en, hx=None, timespans=None):
        """

        :param input: Input tensor of shape (L,C) in batchless mode, or (B,L,C) if batch_first was set to True and (L,B,C) if batch_first is False
        :param hx: Initial hidden state of the RNN of shape (B,H) if mixed_memory is False and a tuple ((B,H),(B,H)) if mixed_memory is True. If None, the hidden states are initialized with all zeros.
        :param timespans:
        :return: A pair (output, hx), where output and hx the final hidden state of the RNN
        """
        device = input.device
        batch_dim = 0 
        seq_dim = 1 
        batch_size, seq_len = input.size(batch_dim), input.size(seq_dim)

        if hx is None:
            h_state = torch.zeros((batch_size, self.state_size), device=device)
            c_state = torch.zeros((batch_size, self.state_size), device=device)
        else:
            if isinstance(hx, torch.Tensor):
                raise RuntimeError(
                    "Running a CfC with mixed_memory=True, requires a tuple (h0,c0) to be passed as state (got torch.Tensor instead)"
                )
            h_state, c_state = hx 
            if h_state.dim() != 2:
                msg = (
                    "For batched 2-D input, hx and cx should "
                    f"also be 2-D but got ({h_state.dim()}-D) tensor"
                )
                raise RuntimeError(msg)

        output_sequence = []
        proj_enc = self.fcn_enc(h_en)
        for t in range(seq_len):
            inputs = input[:, t]
            ts = 1.0 if timespans is None else timespans[:, t]

            #ATTENTION FORWARD
            proj_dec = self.fcn_dec(torch.cat((h_state,c_state),dim = 1)).unsqueeze(1)
            score = self.fcn_score(self.act(proj_dec+proj_enc))
            h_en_attn = score*h_en
            c = h_en_attn.sum(1)
            inputs_hat = self.fcn_y_attn(torch.cat((inputs,c),dim = 1))


            #STANDARD CFC-LSTM
            h_state, c_state = self.lstm(inputs_hat, (h_state, c_state))
            h_out, h_state = self.rnn_cell.forward(inputs_hat, h_state, ts)


        out = self.fcn_out(torch.cat((h_out,c),dim = 1))  
        hx = (h_state, c_state)
        return out, hx
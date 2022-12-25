# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:25 2022

@author: dirty
"""

from typing import Dict

import torch
import torch.nn as nn
import numpy as np

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class NFL_LSTM_classifier(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_layers: int,
        bidirectional: bool,
        intput_size:int,
        intput_data_length :int,
        output_size:int,
        batch_size:int
    ) -> None:
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bidirectional = bidirectional
        self.intput_size = intput_size
        self.intput_data_length = intput_data_length
        self.output_size = output_size
        self.batch_size = batch_size
        
        super(NFL_LSTM_classifier, self).__init__()
        
        # TODO: model architecture

        self.lstm = torch.nn.LSTM(
            input_size = intput_size,
            hidden_size = hidden_size,
            batch_first = True,
            num_layers = num_layers,
            bidirectional = bidirectional
        )

        self.seq = nn.Sequential(
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.1),
            torch.nn.Linear(2 * self.hidden_size * self.intput_data_length, 32),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.1),
            torch.nn.Linear(32, 16),
            torch.nn.ReLU(),
            #torch.nn.Dropout(0.1),
            torch.nn.Linear(16, self.output_size),
            #torch.nn.ReLU(),
            torch.nn.Softmax(dim=1),
        )


        
    @property
    def encoder_output_size(self) -> int:
        # TODO: calculate the output dimension of rnn
        raise NotImplementedError

    def forward(self, batch) -> Dict[str, torch.Tensor]:
        # TODO: implement model forward

        batchSize = batch.shape[0]

        h0 = torch.randn(self.num_layers * (2 if self.bidirectional else 1), batchSize, self.hidden_size).requires_grad_().to(device)
        c0 = torch.randn(self.num_layers * (2 if self.bidirectional else 1), batchSize, self.hidden_size).requires_grad_().to(device)


        L, (hn, Cn) = self.lstm(batch, (h0, c0))
            
        flatten = torch.flatten(L, start_dim=1)
        ans = self.seq(flatten)
        
        return ans.reshape(batchSize,self.output_size)


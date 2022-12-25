# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:16:40 2022

@author: ted
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import optim
from traceingAnimate import animate_player_movement
from PreProcess import preProcess
from model import NFL_LSTM_classifier
from tqdm import trange

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
    
    path = './data/'

# =============================================================================
#     df_games = pd.read_csv(path + 'games.csv')
#     df_plays = pd.read_csv(path + 'plays.csv')
#     df_players = pd.read_csv(path +'players.csv')
#     df_pffScouting = pd.read_csv(path +'pffScoutingData.csv')
#     
#     df_tracking  = pd.DataFrame()
#     
#     for i in range(1,9,1):
#         buffer = pd.read_csv(path + 'week' + str(i) + '.csv')
#         df_tracking = pd.concat([df_tracking,buffer])
#         #df_tracking.append(buffer, ignore_index=True)
# 
#         
#     # select random play
#     df_sample_plays = df_plays.iloc[100]
#     gameID = df_sample_plays['gameId']
#     playID = df_sample_plays['playId']
#     playDesc = df_sample_plays['playDescription']
#     
#     df_sample_tracking = df_tracking.loc[(df_tracking['gameId'] == gameID) & (df_tracking['playId'] == playID)]
#     
#     game = df_games.loc[df_games['gameId'] == gameID]
#     team_home = game['homeTeamAbbr']
#     team_visit = game['visitorTeamAbbr']
# =============================================================================
    
    #df_preProcess = preProcess()
    #df_preProcess.head()
    
    
    # model parameter 
    hidden_size = 512 # 預設512，可調整
    num_layers = 2 # 預設2，可調整
    bidirectional = True
    intput_size = 23 * 9 # (22個選手+球) * 9個feature
    intput_data_length = 64  # Frame size
    output_size = 4 # 4種傳球結果 C、I、S、IN
    batch_size = 32 # 預設32，可調整
    
    model = NFL_LSTM_classifier(hidden_size,num_layers,bidirectional,intput_size,intput_data_length,output_size,batch_size).to(device)
    
    optimizer = optim.Adam(model.parameters(), 1e-3)

    epoch_size = 100
    epoch_pbar = trange(epoch_size, desc="Epoch")
    
    for epoch in epoch_pbar:
        # TODO: Training loop - iterate over train dataloader and update model weights
        train_loop()
        # TODO: Evaluation loop - calculate accuracy and save model weights
        vaild_loop()
        pass

    
    ckpt_path = "./checkpoint/checkpoint.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    
    # video create
    gameID = 2021091600
    playID = 65
    weekNum = 2
    video = animate_player_movement(weekNum,playID,gameID)
    video.save(path + 'video.mp4')


def train_loop():
    raise NotImplementedError

def vaild_loop():
    
    raise NotImplementedError


if __name__ == "__main__":
    main()
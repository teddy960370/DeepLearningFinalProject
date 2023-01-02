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
from PreProcess2 import preProcess
from model import NFL_LSTM_classifier
from tqdm import trange
from dataset import NFL_Track_Dataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

import torch

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def main():
# =============================================================================
#     
#     path = './data/'
#     
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
#     print("Data load complete \n")
#     
#     # 找出各play的frame數
#     df_tracking_temp = df_tracking[['gameId','playId','frameId']].groupby(['gameId','playId'], as_index=False).max()
#     
#     # 加入key
#     df_tracking_temp['key'] = df_tracking_temp['gameId'].astype(str) + df_tracking_temp['playId'].astype(str)
#     df_plays['key'] = df_plays['gameId'].astype(str) + df_plays['playId'].astype(str)
#     # 移除大於64的結果
#     df_tracking_temp = df_tracking_temp.loc[df_tracking_temp['frameId'] <= 64]
#     df_plays = df_plays.loc[df_plays['key'].isin(df_tracking_temp['key'])]
#     # passResult整理
#     df_plays = df_plays.loc[df_plays['passResult'].isin(['C','I','S','IN'])]
#     
#     
#     df_gp = df_plays[['gameId','playId','possessionTeam','defensiveTeam','passResult']]
# =============================================================================
    import pickle
    path = './data/'
    df_data = pd.DataFrame()
    with open(path + 'preProcessdata.pickle', 'rb') as f:
    # The protocol version used is detected automatically, so we do not
    # have to specify it.
        #data = pickle.load(f)
        df_data = pd.read_pickle(f)
    
    
    train,vaild = train_test_split(df_data, test_size=0.2)

    batch_size = 32
    train_dataset = NFL_Track_Dataset(train,df_games,df_pffScouting,df_players,df_tracking,'train')
    train_loader = DataLoader(train_dataset, batch_size = batch_size, shuffle = False,collate_fn=train_dataset.collate_fn)

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
        train_loop(train_loader,model, optimizer,intput_data_length,output_size)
        # TODO: Evaluation loop - calculate accuracy and save model weights
        #vaild_loop()
        pass

    
    ckpt_path = "./checkpoint/checkpoint.pt"
    torch.save(model.state_dict(), str(ckpt_path))
    
    # video create
# =============================================================================
#     gameID = 2021091600
#     playID = 65
#     weekNum = 2
#     video = animate_player_movement(weekNum,playID,gameID)
#     video.save(path + 'video.mp4')
# =============================================================================


def train_loop(dataloader,model,optimizer,intput_data_length,output_size):
    
    loss_function = torch.nn.CrossEntropyLoss()
    num_batches = len(dataloader)
    model.train()
    total_loss = 0
    average_loss = 0
    # model初始化
    model.zero_grad()
    
    for data,label in dataloader:
        
        x = torch.from_numpy(data).float().to(device)
        y = torch.from_numpy(label).float().to(device)
        
        output = model(x)
        # 計算loss
        loss = loss_function(output,y)

        # 梯度清零
        optimizer.zero_grad()
        # 反向傳播
        loss.backward()
        # 更新參數
        optimizer.step()
        
        total_loss += loss
    
    
    average_loss = total_loss / num_batches
    print(f"\n training Loss: {average_loss} ")
    return average_loss



def vaild_loop():
    
    raise NotImplementedError


if __name__ == "__main__":
    main()
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 29 11:37:34 2022

@author: dirty
"""

from typing import List, Dict
from torch.utils.data import Dataset
import numpy as np
from PIL import Image
import torchvision.transforms as transforms
from sklearn.preprocessing import OneHotEncoder ,StandardScaler,LabelEncoder
import pandas as pd

class NFL_Track_Dataset(Dataset):
    def __init__(
        self,
        df_gp : pd.DataFrame(),
        df_games : pd.DataFrame(),
        df_pffScouting : pd.DataFrame(),
        df_players : pd.DataFrame(),
        df_tracking : pd.DataFrame(),
        mode: str,
    ):
        self.df_gp = df_gp
        self.df_games = df_games
        self.df_pffScouting = df_pffScouting
        self.df_players = df_players
        self.df_tracking = df_tracking
        self.mode = mode
        
        #self.collate_fn()
    def __len__(self) -> int:
        return len(self.df_gp)
    
    def __getitem__(self, index) -> Dict:
        instance = self.df_gp.iloc[index]
        
        gameId = instance['gameId']
        playId = instance['playId']
        label = instance['passResult']
        
        label_one_hot = {'C': [1.0, 0.0, 0.0, 0.0],'I': [0.0, 1.0, 0.0, 0.0],
                    'S': [0.0, 0.0, 1.0, 0.0],'IN': [0.0, 0.0, 0.0, 1.0]}
        
        label_encoder = label_one_hot[label]
        
        # 取得該play所有選手NFL ID
        nflId = self.df_pffScouting.loc[(self.df_pffScouting['gameId'] == gameId) & (self.df_pffScouting['playId'] == playId)]
        nflId = nflId['nflId'].tolist()
        
        # 取得該play所有選手資訊
        players = self.df_players[self.df_players['nflId'].isin(nflId)]
        
        # 取得該play每個frame資訊
        tracking = self.df_tracking.loc[(self.df_tracking['gameId'] == gameId) & (self.df_tracking['playId'] == playId)]
        tracking = tracking.join(players.set_index('nflId'), on='nflId').sort_values(['frameId', 'team','officialPosition'])
        
        #instance['feature'] = [feature for feature in tracking]
        
        max_frame = tracking['frameId'].max()
        
        feature = []
        last_frame = []
        for frame_index in range(1,65,1):

            if(frame_index <= max_frame) :
                last_frame = []
                tracking_frame = tracking.loc[tracking['frameId'] == frame_index]
                for index,player in tracking_frame.iterrows():
                    temp = player[['nflId','x','y','s','a','dis','o','dir','weight']]
                    last_frame.append(temp.to_numpy())
             
            feature.append(np.array(last_frame).flatten())
                
        
        return np.array(feature),np.array(label_encoder)
    

    @property
    def num_classes(self) -> int:
        return len(self.label_mapping)


    def collate_fn(self,samples):
        
        #batch = [np.array(list(data.values())) for data in samples]
        
        data = []
        label = []
        for sample in samples :
            temp = [data for data in sample]
            data.append(temp[0].astype(np.float))
            label.append(temp[1].astype(np.float))
        

        return np.array(data),np.array(label)



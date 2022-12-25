# -*- coding: utf-8 -*-
"""
Created on Sun Dec 25 14:28:13 2022

@author: ted
"""

# This Python 3 environment comes with many helpful analytics libraries installed
# It is defined by the kaggle/python Docker image: https://github.com/kaggle/docker-python
# For example, here's several helpful packages to load

import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
from pandarallel import pandarallel as pl
pl.initialize(progress_bar=True, verbose=0)
# Input data files are available in the read-only "../input/" directory
# For example, running this (by clicking run or pressing Shift+Enter) will list all files under the input directory

import os

def Add_playerData_To_TrackingData(row ,playerDf):
    #apply to tracking data
    playerid = row['nflId']
    player = playerDf.loc[(playerDf['nflId'] == playerid)]
    #print(f'playerid :{playerid}')
    #print(f'player :{player}')
    try:
        nflId_displayName      = player.displayName.values[0]
        nflId_officialPosition = player.officialPosition.values[0]
        nflId_height           = player.height.values[0]
        nflId_weight           = player.weight.values[0]
    except:
        nflId_displayName=''
        nflId_officialPosition=''
        nflId_height=''
        nflId_weight=''
        
    row['nflId_displayName']      = nflId_displayName
    row['nflId_officialPosition'] = nflId_officialPosition
    row['nflId_height']           = nflId_height
    row['nflId_weight']           = nflId_weight
    
    return row

# Collect a record of the number of players in each position in every play.
# This will be used later to determine the maxiumum possible number of players
# for each position, which in turn determines how many features are needed
# to be collected (or zero-padded) at every position.
def get_position_counts(row,trackingDf):
    expr = (trackingDf['gameId'] == row['gameId']) & (trackingDf['playId'] == row['playId'])
    df = trackingDf.loc[expr]
    # 每frame參與play的位置數量應該相同
    # 在play中，所以只統計第一frame的位置數。
    vc = df.loc[df['frameId'] == 1, 'nflId_officialPosition'].value_counts()
    return pd.DataFrame(np.array([vc.values]), columns=vc.keys())

def format_features(row,tracking_data_source_df=None, all_positions=None ,feature_cols=None):
   
    #print(f'row:{row}')
    filter = (tracking_data_source_df['gameId'] == row['gameId']) \
             & (tracking_data_source_df['playId'] == row['playId'])
    
    ''' 取出相對應的play資料'''
    play_df = tracking_data_source_df.loc[filter]
    '''取得frames的集合並排序'''
    frames = sorted(play_df['frameId'].unique())
    #print(f'frames:{frames}')
    
    frame_data = [] #容器用以裝入處理好的frame data
    
    '''  Iterate through every frame in the play'''
    for fid in frames:
         #print(f'fid:{fid}')
         features = np.array([])
         '''Iterate每個位置，不一定在play每個位置都會有''' 
         for pos in all_positions:
                if pos!='':
                    pos_data = play_df.loc[(play_df['frameId'] == fid) &
                                   (play_df['nflId_officialPosition'] == pos),
                                   feature_cols].values
                    
                    pos_data =np.nan_to_num(np.hstack(pos_data))
                    #print(f'pos:{pos}')
                    #print(f'Typr:{type(pos_data)}')
                    #print(f'pos_data:{pos_data}')
                    features = np.hstack((features, pos_data))    
         #print(f'features:{features}')
         print(f'features len:{len(features)}')
         frame_data.append(features) 
         #break 
          
    return frame_data


def get_pass_results(row , plays_df):
    
    return plays_df.loc[(plays_df['gameId'] == row['gameId']) &
                        (plays_df['playId'] == row['playId']),
                        'passResult'].values[0]

def preProcess() :
    
    # 設定使用環境
    #environment = 'kaggle'
    #environment = 'colab'
    environment = 'client'
    
    df_games = pd.DataFrame()
    df_pffScouting = pd.DataFrame()
    df_plays = pd.DataFrame()
    df_players = pd.DataFrame()
    df_tracking = pd.DataFrame()
    
    if (environment == 'kaggle'):
    
        for dirname, _, filenames in os.walk('/kaggle/input'):
            for filename in filenames:
                print(os.path.join(dirname, filename))
                if 'week5' in filename:
                    weeks = os.path.join(dirname, filename)
                if 'players' in filename:
                    players = os.path.join(dirname, filename)
                if 'pffScoutingData' in filename:
                    pffScoutingData = os.path.join(dirname, filename)
                if 'plays' in filename:
                    plays = os.path.join(dirname, filename)
        # You can write up to 20GB to the current directory (/kaggle/working/) that gets preserved as output when you create a version using "Save & Run All" 
        # You can also write temporary files to /kaggle/temp/, but they won't be saved outside of the current session
        
        
        df_pffScouting = pd.read_csv(pffScoutingData)
        df_plays = pd.read_csv(plays)
        df_players = pd.read_csv(players)
        df_tracking =  pd.read_csv(weeks)
        
    elif(environment == 'colab') :
        raise NotImplementedError
    
    elif(environment == 'client') :
        path = './data/'
        
        df_games = pd.read_csv(path + 'games.csv')
        df_plays = pd.read_csv(path + 'plays.csv')
        df_players = pd.read_csv(path +'players.csv')
        df_pffScouting = pd.read_csv(path +'pffScoutingData.csv')
             
        
        for i in range(1,2,1):
            buffer = pd.read_csv(path + 'week' + str(i) + '.csv')
            df_tracking = pd.concat([df_tracking,buffer])
            df_tracking.append(buffer, ignore_index=True)
    else:
        raise NotImplementedError
        
    
    if (df_pffScouting.empty | df_plays.empty | df_players.empty | df_tracking.empty):
        raise Exception("檔案未讀取")
    
    df_gp = df_tracking[['gameId', 'playId']].drop_duplicates()
    df_gp.reset_index(drop=True, inplace=True)
    df_gp = df_gp[0:1]
    
    #把球員data塞進track data
    player = df_tracking.parallel_apply(Add_playerData_To_TrackingData , axis=1,playerDf = df_players )
    
    df_tracking = player
    
    #統計每個position數量
    pos_counts = df_gp.parallel_apply(get_position_counts, axis=1 , trackingDf = df_tracking)
    pos_counts_max = pd.concat([df for df in pos_counts]).max().astype(int)
    del pos_counts  # Mark temp object for deletion to free memory
    pos_counts_max.head()
    
    filter = (df_tracking['gameId'] == 2021100700) &(df_tracking['playId'] == 95)\
         &(df_tracking['frameId'] == 1) & (df_tracking['nflId_officialPosition'] == 'T')

    df_tracking[filter]
    
    feature_label_df = df_gp.copy()
    feature_label_df['labels'] = feature_label_df.parallel_apply(get_pass_results, axis=1 , plays_df = df_plays)
    feature_cols = ['nflId','x', 'y', 's', 'a', 'dis', 'o', 'dir','nflId_weight']
    all_positions = sorted(pos_counts_max.keys())
    feature_label_df['features']= df_gp.parallel_apply(format_features, axis=1, args=[ df_tracking,all_positions,feature_cols])
    
    return feature_label_df
# -*- coding: utf-8 -*-
"""
Created on Mon Jan  2 09:40:36 2023

@author: ted
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import time

def getFeatures(row,df_tracking):
    
    gameId = row['gameId']
    playId = row['playId']
    
    # 取得該play每個frame資訊
    tracking = df_tracking.loc[(df_tracking['gameId'] == gameId) & (df_tracking['playId'] == playId)]

    max_frame = tracking['frameId'].max()
    
    feature = []
    last_frame = []
    for frame_index in range(1,65,1):

        if(frame_index <= max_frame) :
            last_frame = []
            tracking_frame = tracking.loc[tracking['frameId'] == frame_index]
            for index,player in tracking_frame.iterrows():
                
                if player['playDirection'] == 'left':
                    player['x'] = 120.0 - player['x']
                    player['y'] = 160/3 - player['y']
                    
                
                height = str(player['height']).split('-')
                if(len(height) == 2):
                    player['height'] = float(height[0]) * 12 + float(height[1])
                else : 
                    player['height']  = 0
                temp = player[['x','y','s','a','dis','o','dir','weight','height']]
                last_frame.append(temp.to_numpy())
         
        feature.append(np.array(last_frame).flatten())
        
    return np.array(feature)
    
def trackNormailizeX(row):
    
    if row['playDirection'] == 'left':
        return 120.0 - row['x'],160/3 - row['y']
    else:
        return row['x'],row['y']

def preProcess() :
    path = './data/'
    
    df_games = pd.read_csv(path + 'games.csv')
    df_plays = pd.read_csv(path + 'plays.csv')
    df_players = pd.read_csv(path +'players.csv')
    df_pffScouting = pd.read_csv(path +'pffScoutingData.csv')
    
    df_tracking  = pd.DataFrame()
    
    for i in range(1,9,1):
        buffer = pd.read_csv(path + 'week' + str(i) + '.csv')
        df_tracking = pd.concat([df_tracking,buffer])
        #df_tracking.append(buffer, ignore_index=True)

    

    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"Data load complete , time : {current_time}\n")
    
    # tracking data join player data
    df_tracking = df_tracking.join(df_players.set_index('nflId'), on='nflId').sort_values(['gameId','playId','frameId', 'team','officialPosition'])
    
    # tracking 資料補0
    df_tracking['officialPosition'] = df_tracking['officialPosition'].fillna('ball')
    df_tracking['o'] = df_tracking['o'].fillna(0)
    df_tracking['dir'] = df_tracking['dir'].fillna(0)
    df_tracking['weight'] = df_tracking['weight'].fillna(0)
    df_tracking['height'] = df_tracking['height'].fillna(0)
    #df_tracking[['height_foot', 'height_inches']] = df_tracking['height'].str.split('-', 1, expand=True)
    #df_tracking['height'] = df_tracking['height_foot'].astype(float) * 12 + df_tracking['height_inches'].astype(float)
    #df_tracking['height'] = df_tracking['height'].astype(float)
    
    # 加入key
    df_tracking['key'] = df_tracking['gameId'].astype(str) + df_tracking['playId'].astype(str)
    df_plays['key'] = df_plays['gameId'].astype(str) + df_plays['playId'].astype(str)
    
    #df_tracking['x'] = MinMaxScaler().fit_transform(np.array(df_tracking['x']).reshape(-1,1))
    #df_tracking['y'] = MinMaxScaler().fit_transform(np.array(df_tracking['y']).reshape(-1,1))
    

    # 找出各play的frame數
    df_tracking_temp = df_tracking[['key','frameId']].groupby(['key'], as_index=False).max()
    

    # 移除大於64的結果
    df_tracking_temp = df_tracking_temp.loc[df_tracking_temp['frameId'] <= 64]
    df_plays = df_plays.loc[df_plays['key'].isin(df_tracking_temp['key'])]
    
    # passResult整理，移除passResult為R的結果
    df_plays = df_plays.loc[df_plays['passResult'].isin(['C','I','S','IN'])]
    
    # 再對tracking做一次清理
    df_tracking = df_tracking.loc[df_tracking['key'].isin(df_tracking_temp['key'])]
    
    # tracking 資料normalize
    
    #df_tracking['x_norm'],df_tracking['y_norm'] = df_tracking.progress_apply(trackNormailizeX, axis=1)
    
    #df_players['height'] = 
    
    df_gp = df_plays[['gameId','playId','possessionTeam','defensiveTeam','passResult']]
    
    
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"get Feature start, time : {current_time}\n")
    
    df_temp100 = df_gp.iloc[:2000]
    tqdm.pandas()
    df_temp100['Features'] = df_temp100.progress_apply(getFeatures, axis=1 , df_tracking = df_tracking)
    
    current_time = time.strftime("%H:%M:%S", time.localtime())
    print(f"get Feature end, time : {current_time}\n")
    
    #df_temp100.to_csv(path + 'preprocessData.csv',index=False)
    
    import pickle
    with open(path + 'preProcessdata.pickle', 'wb') as f:
    # Pickle the 'data' dictionary using the highest protocol available.
        pickle.dump(df_temp100, f, pickle.HIGHEST_PROTOCOL)
    # The following example reads the resulting pickled data.

    
def main() :
    preProcess()
    
    
if __name__ == "__main__":
    main()
    
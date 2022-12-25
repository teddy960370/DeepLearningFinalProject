# -*- coding: utf-8 -*-
"""
Created on Sun Dec 11 14:16:40 2022

@author: ted
"""

from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from traceingAnimate import animate_player_movement
from PreProcess import preProcess

def main():
    
    path = '../data/'

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
    
    df_preProcess = preProcess()
    df_preProcess.head()
    #video = animate_player_movement(2,65,2021091600)
    
    #video.save(path + 'video.mp4')

    
if __name__ == "__main__":
    main()
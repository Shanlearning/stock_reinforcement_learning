# -*- coding: utf-8 -*-
"""
Created on Sun Apr 17 16:26:23 2022

@author: zhong
"""

import os
os.chdir("C:\\Users\\zhong\\Dropbox\\github\\stock_reinforcement _learning\\code_torch")

import pandas as pd
import numpy as np
from sklearn.preprocessing import PowerTransformer
from torch.nn.functional import mse_loss


_dat_ = pd.read_csv("data\\dat_518_companies.csv",index_col=0)
_dat_ = _dat_[['return_t', 'return_t_plus_1', 'date', 'ticker','cci', 'macdh', 'rsi_14', 'kdjk', 'wr_14', 'atr_percent','cmf']].copy()
_dat_ = _dat_[~(_dat_.isna()).T.any()].copy()
_dat_['date'] = pd.DatetimeIndex(_dat_['date'])

available_years = [item for item in range(2002,2020)]
available_days = list(set(_dat_['date']))
available_days.sort()

_input_val =  ['return_t','cci','macdh','rsi_14','kdjk','wr_14', 'atr_percent', 'cmf']
_output_val = ['return_t_plus_1']

class StockEnv(object):
    """
    ### Description
    
    """
    def __init__(self):
        self.action_space = 5
        self.observation_space = [-4.5,4.5]
        self.state = None
        self.year = 2010
        self.transformer = None
        self.available_years = available_years
        self.available_days_train = available_days
        self.available_days_test = available_days
        self.dat_train = None
        self.dat_test = None
        self.original_dat = _dat_
        
    def update_year(self,year):
        self.year = year
        _dat_ = self.original_dat
        
        train_df = _dat_[pd.DatetimeIndex(_dat_['date']).year < year ].copy()
        test_df = _dat_[ pd.DatetimeIndex(_dat_['date']).year == year ].copy()  
        
        _train_X = np.asarray(train_df[_input_val])
        
        X_transformer = PowerTransformer(method='yeo-johnson', standardize=True).fit(_train_X)
        _train_X = X_transformer.transform(_train_X)
        #_train_X  = np.nan_to_num(_train_X)
        _train_X  = np.asarray(pd.DataFrame(_train_X).clip(-4.5,4.5))
        
        _test_X =  np.asarray(test_df[_input_val])
        _test_X =  X_transformer.transform(np.asarray(test_df[_input_val]))
        #_test_X  = np.nan_to_num(_test_X)
        _test_X  = np.asarray(pd.DataFrame(_test_X).clip(-4.5,4.5))
        train_df[_input_val] = _train_X
        test_df[_input_val] = _test_X
        
        available_days_train = list(set(train_df['date']))
        available_days_train.sort()
        available_days_test = list(set(test_df['date']))
        available_days_test.sort()
        
        available_days_train_pd = pd.DataFrame()
        available_days_train_pd['dates'] = available_days_train
        
        available_days_test_pd = pd.DataFrame()
        available_days_test_pd['dates'] = available_days_test
        
        self.transformer = X_transformer
        self.dat_train = train_df 
        self.dat_test = test_df
        
        self.available_days_train = available_days_train
        self.available_days_test = available_days_test
                
        self.available_days_train_pd = available_days_train_pd
        self.available_days_test_pd = available_days_test_pd
        
    def step(self, action ,training=True):
        if training:
            available_days_pd = self.available_days_train_pd
            _df = self.dat_train
            # _df = train_df
        else:
            available_days_pd = self.available_days_test_pd
            _df = self.dat_test
        
        #available_stocks = state
        available_stocks = self.state 
        reward = mse_loss(action , available_stocks['return_t_plus_1'])
        
        _ticker = list(set(available_stocks['ticker']))
        _day = list(set(available_stocks['date']))[0]
        _next_day = available_days_pd[available_days_pd['dates'] > _day].iloc[0]['dates']
        available_stocks = _df[_df['date'] == _next_day].copy()
        self.state = available_stocks[available_stocks['ticker'].isin(_ticker) ]
        
        if _day == available_days_pd.iloc[-1]['dates']:
            done = True
        return self.state, reward, done, {}

    def reset(self,training=True):
        if training:
            available_days = self.available_days_train
            _df = self.dat_train
            # _df = train_df
        else:
            available_days = self.available_days_test
            _df = self.dat_test
            
        _day = np.random.choice(available_days)
        available_stocks = _df[_df['date'] == _day].copy()
        _len_stocks = len(available_stocks)
        self.state = available_stocks.iloc[np.random.choice(_len_stocks, 5)]
        #return self.state
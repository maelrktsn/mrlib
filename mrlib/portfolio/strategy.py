import pandas as pd
import numpy as np
import datetime as dt

# --- Value Strategy ---
def value_strategy(date: pd.Timestamp, free_cf: pd.DataFrame, pe_ratio: pd.DataFrame, pb_ratio: pd.DataFrame, n_inst: int):

    free_cf  = free_cf.loc[free_cf.index == date]
    pe_ratio = pe_ratio.loc[pe_ratio.index == date]
    pb_ratio = pb_ratio.loc[pb_ratio.index == date]

    # Check empty - last available date
    count_date = 0
    while len(free_cf) == 0:
        count_date += 1
        new_date    = date - dt.timedelta(count_date)
        free_cf     = free_cf.loc[free_cf.index == new_date]

    count_date = 0
    while len(pe_ratio) == 0:
        count_date += 1
        new_date = date - dt.timedelta(count_date)
        pe_ratio = pe_ratio.loc[pe_ratio.index == new_date]

    count_date = 0
    while len(pb_ratio) == 0:
        count_date += 1
        new_date = date - dt.timedelta(count_date)
        pb_ratio = pb_ratio.loc[pb_ratio.index == new_date]

    free_cf  = free_cf.T
    pe_ratio = pe_ratio.T
    pb_ratio = pb_ratio.T

    # Score from 1 to 10 on each indicators
    df_value                  = pd.merge(free_cf, pe_ratio, how= 'inner', left_on = free_cf.index, right_on = pe_ratio.index)
    df_value                  = pd.merge(df_value, pb_ratio, how= 'inner', left_on = 'key_0', right_on = pb_ratio.index)
    df_value.columns          = ['Ticker', 'Free_CF', 'PE_Ratio', 'PB_Ratio']
    df_value['Score_CF']      = pd.qcut(df_value['Free_CF'], 10, labels = list(range(1,11)))
    df_value['Score_PE']      = pd.qcut(df_value['PE_Ratio'], 10, labels = list(range(10,0,-1))) 
    df_value['Score_PB']      = pd.qcut(df_value['PB_Ratio'], 10, labels = list(range(10,0,-1)))
    df_value['average_score'] = np.mean(df_value[['Score_CF','Score_PE','Score_PB']], axis = 1) # Equal weight average, but we can define a weighted average according the most important indicator for the PM.


    # Screening - N best stocks 
    rank    = df_value[['Ticker','average_score']].sort_values(by='average_score', ascending=False)
    tickers = rank.iloc[:n_inst,0].tolist()

    return tickers
import pandas as pd
import numpy as np
def tnr(conf_matrix):
    """compute true negative rate / recall from a confusion matrix in
    which the true label is in the ith row, predicted label is in the
    jth column"""
    tnr = conf_matrix[0,0]/(conf_matrix[0,0]+conf_matrix[0,1])
    return np.round(tnr, 2)

def tpr(conf_matrix):
    """compute true positive rate / sensitivity / recall from a confusion matrix 
    in which the true label is in the ith row, 
    predicted label is in the jth column"""
    tpr = conf_matrix[1,1]/(conf_matrix[1,1]+conf_matrix[1,0])
    return np.round(tpr, 2)

def trade_from_close_close(stock,sd,begin_price = None):
    '''Args: stock: dataframe of stock data | sd: dictionary of staggered weeks,
    i.e. {2017-02:2017-01, ... 2021-52:2021-51} for gettin prior week adjusted
    close | Begin Price: optional parameter for prior period adj close
    Return a dataframe with weekly return factors based on first day of week
    open and last day of week close.'''
    
    stock_by_week = pd.DataFrame(
        columns=['Year','Year_Week',"Begin_Price","Adj Close", 'Return'])
    
    year_weeks = stock["Year_Week"].unique()
    
    for inx, yw in enumerate(year_weeks):
        this_week = stock[stock["Year_Week"]==yw]
        try:
            begin_price = stock[stock["Year_Week"]==sd[yw]].\
            tail(1)['Adj Close'].values[0]
        except (IndexError,KeyError): 
            if begin_price is not None:
                pass
            else:
                begin_price = this_week.head(1)['Open'].values[0]
        close_price = this_week.tail(1)['Adj Close'].values[0]
        r = close_price/begin_price
        y = this_week.tail(1)['Year'].values[0]
        stock_by_week.loc[inx,:] = [y,yw,begin_price,close_price,r]
    return stock_by_week

def trade_labels(df, year, predictions):
    """Args: df with weekly Return, year of interest, and predictions in binary 
    with 1 being a trade week and 0 being a hold week.  Returns the final portfolio 
    value of $100 invested in this strategy"""
    
    return np.round(100*np.prod(df.query(f'Year == {year}')[
    predictions.astype(np.bool_)].Return),2)

def buy_and_hold(df, year):
    '''Return the buy & hold ending portfolio from 100 invested at the beginning of the year.'''
    return np.round(100*df.query(f'Year == {year}')["Begin_Price"].values[-1]/\
    df.query(f'Year == {year}')["Begin_Price"].values[0],2)



    
import pandas as pd
import numpy as np
from sklearn.metrics import accuracy_score
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

def trade_from_close_close(stock,begin_price = None):
    """Args: stock: dataframe of stock data | sd: dictionary of staggered weeks, i.e. {2017-02:2017-01, ... 2021-52:2021-51} for getting prior week adjusted close | Begin Price: optional parameter for prior period adj close
    Return a dataframe with weekly return factors based on first day of week
    open and last day of week close."""
    
    staggered = stagger_weeks(stock)
    open_prices = stock.loc[:,['Year_Week','Open']]
    stock_by_week = pd.DataFrame(
        columns = ['Year','Year_Week',"Begin_Price","Adj Close", 'Return'])
    year_weeks = stock["Year_Week"].unique()
    for inx, yw in enumerate(year_weeks):
        this_week = stock[stock["Year_Week"]==yw]
        try:
            begin_price = stock[stock["Year_Week"]==staggered[yw]].\
            tail(1)['Adj Close'].values[0]
        except (IndexError,KeyError): 
            if begin_price is not None:
                pass
            else:
                begin_price = open_prices[open_prices["Year_Week"]==yw]['Open'].head(1).values[0]
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

def stagger_weeks(stock_daily_info):
    """return dictionary correspondence for 1 week lags"""
    staggered_weeks_dict = dict(zip(stock_daily_info.Year_Week.unique()[1:], 
                           stock_daily_info.Year_Week.unique()[:-1]))
    return staggered_weeks_dict

def head_tail(df, n=2):
    """return top and bottom n rows of a DataFrame"""
    return pd.concat([df.head(n),df.tail(n)])


def X_matrix(stock, y = 2020, w = 5, measure = 'Adj Close'):
    """Return a matrix of weekly lagged values for a given feature."""
    (begin,stop) = (stock[stock['Year']==y].index[0],
                    stock[stock['Year']==y].index[-1]+1)
    X_lags = [stock[f'{measure}'][i-w:i].values for i in range(begin,stop)]
    return X_lags
# HW 8: Trading With Linear Models
def poly_expansion(X,d):
    """Return a polynomially expanded matrix."""
    X_pe = [[x_i**d_i for d_i in range(1,d+1) for x_i in x] for x in X]
    return X_pe

def add_intercept(X_pe):
    """Return matrix with intercept prepended to each row."""
    X_pe = np.asarray(X_pe, dtype = 'float64')
    X_i_pe = np.vstack([np.ones(len(X_pe)),X_pe.T]).T
    return X_i_pe

def labeler(differences):
        """Return labels for weekly price movements: 1 for up, 0 for down."""
        labels = np.where(differences > 0, 1, 0)
        shifted = np.delete(np.insert(labels,0,1),-1)
        labels = np.where(differences == 0, shifted, labels)
        return labels

def X_transform(stock, y = 2020, w = 5, d = 2, measure = 'Adj Close'):
    """Return a matrix for which coefficients will be fit."""
    return np.asarray(add_intercept(
        poly_expansion(X_matrix(stock,y=y,w=w,measure=measure),d=d)), dtype='float64')

def polynomial_accuracy(stock,y = 2020, w = 5, d = 2, test = 1, measure = 'Adj Close', weekly=True, **kwargs):
    """Return the accuracy, np.lstsq results, and a tuple of predictions, 
    true labels, and false labels for a particular model formulation."""
    (y1,y2) = (y,y+test)
    if weekly:
        stock = trade_from_close_close(stock)
    X_tr = X_transform(stock,y=y1,w=w,d=d,measure=measure)
    X_te = X_transform(stock,y=y2,w=w,d=d,measure=measure)
    y_tr = np.asarray(stock[stock['Year'] == y1][measure].values,
                      dtype='float64')
    y_te = np.asarray(stock[stock['Year'] == y2][measure].values,
                     dtype='float64')
    lstsq_results = np.linalg.lstsq(X_tr,y_tr,rcond=1.e-10)
    coeffs = lstsq_results[0]
    predictions = X_te @ coeffs
    modeled_difference = predictions - stock.loc[stock[
        stock.Year == y2].index -1,f'{measure}'].values
    actual_difference  = stock[stock.Year == y2][f'{measure}'].values - \
    stock.loc[stock[
        stock.Year == y2].index -1,f'{measure}'].values
    return (accuracy_score(labeler(actual_difference),
                          labeler(modeled_difference)),lstsq_results,
            {'predictions' : predictions,
             'true_labels':labeler(actual_difference),
             'predicted_labels':labeler(modeled_difference)})
    
def hyperparameter_grid(stock,year_train,w_array,d_array,test,measure):
    hp = {}
    for d in d_array:
        for w in w_array:
            hp[(d,w)] = polynomial_accuracy(stock,y=year_train,w=w,d=d,
                            test=test, measure=measure)[0]
    
    accuracies_df = pd.DataFrame([tuple(list(k)+[hp[k]]) for k in hp.keys()],
                                 columns = ['d','w','accuracy'])
    return accuracies_df

def hp_g(a,b):
    return a+b
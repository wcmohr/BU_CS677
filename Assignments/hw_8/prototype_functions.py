def X_matrix(stock, y = 2020, w = 5, measure = 'Adj Close'):
    """Return a matrix of weekly lagged values for a given feature."""
    (begin,stop) = (stock[stock['Year']==y].index[0],
                    stock[stock['Year']==y].index[-1]+1)
    X_lags = [stock[f'{measure}'][i-w:i].values for i in range(begin,stop)]
    return X_lags

def add_intercept()

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
    
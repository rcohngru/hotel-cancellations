import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import statsmodels.api as sm

from imblearn.over_sampling import SMOTE, RandomOverSampler
from imblearn.under_sampling import RandomUnderSampler
from sklearn.model_selection import KFold
from sklearn.metrics import precision_score, recall_score, f1_score

def dummify(X):
    dummy_cols = ['total_nights', 'total_of_special_requests', 'market_segment', 'party_size']
    X = pd.get_dummies(X, columns=dummy_cols, drop_first=True)
    return X.to_numpy()

def clean_data(data):
    feats = ['hotel','market_segment', 'total_of_special_requests', 
         'total_nights', 'room_difference', 'party_size', 'booking_changes']
    X = data[feats].copy()
    
    
    dummifiable_df = pd.DataFrame({
            'hotel' : [0, 1, 0, 1, 0, 0, 1, 1],
            'market_segment' : ['Offline TA/TO', 'Online TA', 
                                'Groups', 'Direct', 'Corporate',
                                'Groups', 'Direct', 'Corporate'],
            'total_of_special_requests' : [0, 1, 2, 0, 1, 2, 0, 1] ,
            'total_nights' : [1, 2, 3, 4, 5, 6, 7, 8],
            'room_difference' : [0, 1, 0, 1, 0, 1, 0, 1],
            'party_size' : [1, 2, 3, 3, 2, 1, 1, 2],
            'booking_changes' : [0, 0, 1, 1, 0, 0, 1, 1]
            })
    
    X = X.append(dummifiable_df, ignore_index=True)
    
    
    X.replace(['Resort Hotel', 'City Hotel'], [0, 1], inplace=True)
    X.loc[np.argwhere((X['total_nights'] >= 8).values).flatten(), 'total_nights'] = 8
    X.loc[np.argwhere((X['total_of_special_requests'] >= 2).values).flatten(), 'total_of_special_requests'] = 2
    X.loc[np.argwhere(X['market_segment'].isin(['Aviation', 'Complementary']).values).flatten(), 'market_segment'] = 'Corporate'
    X.loc[np.argwhere((X['party_size'] >= 3).values).flatten(), 'party_size'] = 3
    X.loc[np.argwhere((X['booking_changes'] >= 1).values).flatten(), 'booking_changes'] = 1
    X = dummify(X)
        
    y = data[['is_canceled']].to_numpy()
    
    return X[0 : data.shape[0], :], y

def get_day_counts(data):
    start_day = (data['full_date'] - np.min(data['full_date'])).dt.days
    days = len(data['full_date'].unique())
    resort_bookings_by_day = np.zeros(days)
    city_bookings_by_day = np.zeros(days)
    resort_cancellations_by_day = np.zeros(days)
    city_cancellations_by_day = np.zeros(days)

    for day, nights, hotel_type, canceled in zip(start_day, data['total_nights'], data['hotel'], data['is_canceled']):
        if canceled == 1:
            if hotel_type == 1:
                city_cancellations_by_day[day:(day + nights + 1)] += 1
            else:
                resort_cancellations_by_day[day:(day + nights + 1)] += 1
        else:  
            if hotel_type == 1:
                city_bookings_by_day[day:(day + nights + 1)] += 1
            else:
                resort_bookings_by_day[day:(day + nights + 1)] += 1
                
    return resort_bookings_by_day, resort_cancellations_by_day, city_bookings_by_day, city_cancellations_by_day


def balance_train_data(X, y, method=None):
    '''
    Balances the data passed in according to the specified method.
    Returns balanced numpy arrays.
    '''
    if method == None:
        return X, y
    
    elif method == 'undersampling':
        rus = RandomUnderSampler()
        X_train, y_train = rus.fit_resample(X, y)
        return X_train, y_train
    
    elif method == 'oversampling':    
        ros = RandomOverSampler()
        X_train, y_train = ros.fit_resample(X, y)
        return X_train, y_train
    
    elif method == 'smote':
        smote = SMOTE()
        X_train, y_train = smote.fit_resample(X, y)
        return X_train, y_train
    
    elif method == 'both':
        smote = SMOTE(sampling_strategy=0.75)
        under = RandomUnderSampler(sampling_strategy=1)
        X_train, y_train = smote.fit_resample(X, y)
        X_train, y_train = under.fit_resample(X_train, y_train)
        return X_train, y_train

    else:
        print('Incorrect balance method')
        return

def plot_cross_val(models, X, y, ax, sampling_method, names, n_splits=5):
    kf = KFold(n_splits=n_splits, shuffle=True)

    
    precisions = [] 
    recalls = []
    f1 = []
    for i in range(len(models)):
        precisions.append([])
        recalls.append([])
        f1.append([])
    
    for train, test in kf.split(X):
        X_test, y_test = X[test], y[test]
        X_train, y_train = X[train], y[train]
        
        X_train, y_train = balance_train_data(X_train, y_train, method=sampling_method)
         
        for i, model in enumerate(models):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            
            precisions[i].append(precision_score(y_test, y_pred))
            recalls[i].append(recall_score(y_test, y_pred))
            f1[i].append(f1_score(y_test, y_pred))
    
    x = range(0, n_splits)
    colormap = {0 : 'r',
                1 : 'b',
                2 : 'g', 
                3 : 'c', 
                4 : 'm'}
    
    
    for i in range(len(models)):
        ax.plot(x, f1[i], c=colormap[i], 
                linewidth=1, linestyle='-',
                label='%s F1 Score' % names[i])
        #ax.plot(x, precisions[i], c=colormap[i], 
        #        linewidth=1, linestyle='-',
        #        label='%s Precision' % names[i])
        #ax.plot(x, recalls[i], c=colormap[i], 
        #        linewidth=1, linestyle='--',
        #        label='%s Recall' % names[i])
        
        
def moving_avg(values, window=3):
    bef = np.floor(window/2)
    aft = np.ceil(window/2)
    avg = []
    for i in range(len(values)):
        before = 0 if i < bef else i - bef
        after = len(values) if i + aft > len(values) else i + aft
        avg.append(np.mean(values[int(before) : int(after)]))
    return avg

def timedelta(series):
    deltas = []
    for s in series:
        deltas.append(datetime.timedelta(days=s))
        
    return pd.Series(deltas, index=series.index)


def plot_trend_data(ax, name, series):
    ax.plot(series.index.date, series, linewidth=2, label='Raw Data')

def make_design_matrix(arr):
    """Construct a design matrix from a numpy array, converting to a 2-d array
    and including an intercept term."""
    return sm.add_constant(arr.reshape(-1, 1), prepend=False)

def fit_linear_trend(series):
    """Fit a linear trend to a time series.  Return the fit trend as a numpy array."""
    X = make_design_matrix(np.arange(len(series)) + 1)
    linear_trend_ols = sm.OLS(series.values, X).fit()
    linear_trend = linear_trend_ols.predict(X)
    return linear_trend

def plot_linear_trend(ax, name, series):
    linear_trend = fit_linear_trend(series)
    plot_trend_data(ax, name, series)
    ax.plot(series.index.date, linear_trend, linewidth=2, label='Trend')
    
def create_monthly_dummies(series):
    month = series.index.week
    # Only take 11 of the 12 dummies to avoid strict colinearity.
    return pd.get_dummies(month, drop_first=True)

def fit_seasonal_trend(series):
    dummies = create_monthly_dummies(series)
    X = sm.add_constant(dummies.values, prepend=False)
    seasonal_model = sm.OLS(series.values, X).fit()
    return seasonal_model.predict(X)

def plot_seasonal_trend(ax, name, series):
    seasons_average_trend = fit_seasonal_trend(series)
    plot_trend_data(ax, name, series, )
    ax.plot(series.index.date, seasons_average_trend, '-', label='Seasonal Trend', linewidth=2)
    
def plot_decomposition(ax, name, series):
    plot_trend_data(ax[0], name, series)
    ax[0].set_title('Raw Series - %s' % (name), fontsize=20)
    
    linear_trend = fit_linear_trend(series)
    plot_trend_data(ax[1], name, pd.Series(linear_trend, series.index))
    ax[1].set_title('Trend Component - %s' % (name), fontsize=20)
    
    if name == 'Resort':
        ax[1].set_ylim(150, 350)
    else:
        ax[1].set_ylim(200, 600)
    
    seasonal_trend = fit_seasonal_trend(series - linear_trend)
    plot_trend_data(ax[2], name, pd.Series(seasonal_trend, series.index))
    ax[2].set_title('Seasonal Component - %s' % (name), fontsize=20)
    
    plot_trend_data(ax[3], name, series - seasonal_trend - linear_trend)
    ax[3].set_title('Residual Component - %s' % (name), fontsize=20)
    
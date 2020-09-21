import numpy as np
import pickle
import statsmodels.api as sm
import pandas as pd
import sys

sys.path.insert(0, '../python')

from sklearn.ensemble import RandomForestClassifier

class Model:
    def __init__(self):
        self.high_demand_thresh = 0.81
        self.low_demand_thresh = 0.21
        self.model = RandomForestClassifier(criterion='entropy', max_depth=10,
                                      max_features='sqrt', n_estimators=75)
        self.resort_rooms = 250
        self.city_rooms = 350

    def make_design_matrix(self, arr):
            """Construct a design matrix from a numpy array, converting to a 2-d array
            and including an intercept term."""
            return sm.add_constant(arr.reshape(-1, 1), prepend=False)

    def fit_linear_trend(self, series):
        """Fit a linear trend to a time series.  Return the fit trend as a numpy array."""
        X = self.make_design_matrix(np.arange(len(series)) + 1)
        linear_trend_ols = sm.OLS(series.values, X).fit()
        linear_trend = linear_trend_ols.predict(X)
        return linear_trend

    def fit_classifier(self, X, y):
        self.X = X
        self.y = y.flatten()
        self.model.fit(self.X, self.y)

    def fit_demand_forecaster(self, resort_dates, city_dates):
        df = pd.DataFrame()
        df['ds'] = resort_dates.index
        df['y'] = resort_dates.values
        df = df[15:]

        df['month'] = df['ds'].dt.month
        df['day_of_month'] = df['ds'].dt.day
        df['year'] = df['ds'].dt.year

        self.resort_dates = df.copy()
        l = self.fit_linear_trend(resort_dates)
        self.resort_slope = 1 + (l[1] - l[0] / 1)

        df = pd.DataFrame()
        df['ds'] = city_dates.index
        df['y'] = city_dates.values
        df = df[15:]

        df['month'] = df['ds'].dt.month
        df['day_of_month'] = df['ds'].dt.day
        df['year'] = df['ds'].dt.year

        self.city_dates = df.copy()
        l = self.fit_linear_trend(city_dates)
        self.city_slope = 1 + (l[1] - l[0] / 1)

    def predict_demand(self, hotel_type, date):
        if type(date) == str:
            [yr, m, d] = date.split('-')
        else:
            m = date.month
            d = date.day

        # hotel_type 1: city
        if hotel_type == 1:
            return (self.city_dates[(self.city_dates['month'] == int(m)) &
                           (self.city_dates['day_of_month'] == int(d))]['y'].mean() * self.city_slope)
        else:
            return (self.resort_dates[(self.resort_dates['month'] == int(m)) &
                           (self.resort_dates['day_of_month'] == int(d))]['y'].mean() * self.resort_slope)


    def predict_proba(self, X, dates):
        y_probas = self.model.predict_proba(X)[:, 1]

        demands = []
        for hotel_type, date in zip(X[:, 0], dates):
            forecasted_demand = self.predict_demand(hotel_type, date)
            demands.append(int(forecasted_demand))

        return (y_probas, np.array(demands))


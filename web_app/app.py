import pickle
import psycopg2 as pg2
import pandas as pd
import flask
import sys
import os
import datetime

sys.path.insert(0, '../src/python/')
from model import Model
from helpers import *

APP = flask.Flask(__name__)

@APP.route('/')
def index():
    return flask.render_template('index.html')


@APP.route('/dash')
def dash():
    query = '''
    SELECT
    id, timestamp, hotel, party_size, market_segment,
    room_difference, booking_changes, total_of_special_requests, total_nights,
    arrival_date_year, arrival_date_month, arrival_date_day_of_month
    FROM reservations
    ORDER BY timestamp DESC
    '''
    cur.execute(query)
    obj = cur.fetchall()
    df = pd.DataFrame(obj, columns=['id', 'timestamp', 'hotel', 'party_size', 'market_segment',
                                'room_difference', 'booking_changes', 'total_of_special_requests', 'total_nights',
                                'arrival_year', 'arrival_month', 'arrival_day'])
    X = clean_data(df, y_included=False)

    dates = []
    for y, m, d in zip(df['arrival_year'], df['arrival_month'], df['arrival_day']):
        date = str(y) + '-' + str(m) + '-' + str(d)
        dates.append(datetime.datetime.strptime(date, '%Y-%B-%d'))
    dates = pd.DatetimeIndex(dates)
    probs, bookings = model.predict_proba(X, dates)

    return flask.render_template('index.html')

if __name__ == '__main__':
    APP.debug=True

    print(os.getcwd())
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    conn = pg2.connect(dbname='hotels', user='postgres',
                      host='localhost', port='5432',
                      password='password')
    cur = conn.cursor()
    conn.autocommit = True

    APP.run()

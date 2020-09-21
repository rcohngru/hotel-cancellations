import psycopg2 as pg2
import pandas as pd
from flask import Flask, render_template
import sys
import os
import datetime

APP = Flask(__name__)

@APP.route('/')
def index():
    return render_template('index.html')

@APP.route('/detail')
def detail():
    return ''

@APP.route('/dash')
def dash():
    query = '''
    SELECT
    id, timestamp,
    arrival_date_year, arrival_date_month, arrival_date_day_of_month,
    gross_profit, cancel_prob, hotel, predicted_demand
    FROM predictions
    ORDER BY timestamp DESC
    '''
    cur.execute(query)
    obj = cur.fetchall()
    df = pd.DataFrame(obj, columns=['id', 'timestamp',
                              'arrival_date_year',
                              'arrival_date_month',
                              'arrival_date_day_of_month',
                              'gross_profit',
                              'cancel_prob',
                              'hotel',
                              'predicted_demand'])
    arrival_date = []
    for y, m, d in zip(df['arrival_date_year'],
                      df['arrival_date_month'],
                      df['arrival_date_day_of_month']):

        date = str(y) + '-' + str(m) + '-' + str(d)
        arrival_date.append(datetime.datetime.strptime(date, '%Y-%B-%d').strftime('%Y-%m-%d'))

    capacity = [350 if h == 'City Hotel' else 250 for h in df['hotel']]
    booking_date = [datetime.datetime.fromtimestamp(timestamp) for timestamp in df['timestamp']]

    return render_template('index.html',
            items=zip(booking_date,
                      arrival_date,
                      df['hotel'],
                      df['gross_profit'],
                      df['cancel_prob'],
                      df['predicted_demand'],
                      capacity))

if __name__ == '__main__':
    APP.debug=True

    conn = pg2.connect(dbname='hotels', user='postgres',
                      host='localhost', port='5432',
                      password='password')
    cur = conn.cursor()
    conn.autocommit = True

    APP.run()

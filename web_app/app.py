import psycopg2 as pg2
import pandas as pd
import numpy as np
from flask import Flask, render_template, request
import sys
import os
import datetime

APP = Flask(__name__)

@APP.route('/detail/<id>', methods=['GET'])
def detail(id):
    query = '''
      SELECT *
      FROM reservations
      WHERE id=%s;
    ''' % (id)

    cur.execute(query)
    obj = cur.fetchall()
    df = pd.DataFrame(obj, columns=['id', 'hotel', 'timestamp', 'is_canceled', 'lead_time',
            'arrival_date_year', 'arrival_date_month', 'arrival_date_week_number',
            'arrival_date_day_of_month', 'stays_in_weekend_nights', 'stays_in_week_nights',
            'adults', 'children', 'babies', 'meal', 'country', 'market_segment', 'distribution_channel',
            'is_repeated_guest', 'previous_cancellations', 'previous_bookings_not_canceled', 'reserved_room_type',
            'assigned_room_type', 'booking_changes', 'deposit_type', 'agent', 'company', 'days_in_waiting_list',
            'customer_type', 'adr', 'required_car_parking_spaces', 'total_of_special_requests',
            'reservation_status', 'total_nights', 'party_size','is_family', 'room_difference', 'had_agent', 'had_company'])

    booking_date = [datetime.datetime.fromtimestamp(timestamp) for timestamp in df['timestamp']]
    arrival_date = []
    for y, m, d in zip(df['arrival_date_year'],
                      df['arrival_date_month'],
                      df['arrival_date_day_of_month']):

        date = str(y) + '-' + str(m) + '-' + str(d)
        arrival_date.append(datetime.datetime.strptime(date, '%Y-%B-%d').strftime('%Y-%m-%d'))
    return render_template('detail.html',
                            data=df,
                            book_date=booking_date,
                            arr_date=arrival_date)

@APP.route('/')
@APP.route('/dash')
def dash():
    query = '''
    SELECT
    id, timestamp,
    arrival_date_year, arrival_date_month, arrival_date_day_of_month,
    gross_profit, cancel_prob, hotel, predicted_demand, total_nights, adr
    FROM predictions
    ORDER BY timestamp DESC;
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
                              'predicted_demand',
                              'total_nights',
                              'adr'])

    arrival_date = []
    for y, m, d in zip(df['arrival_date_year'],
                      df['arrival_date_month'],
                      df['arrival_date_day_of_month']):

        date = str(y) + '-' + str(m) + '-' + str(d)
        arrival_date.append(datetime.datetime.strptime(date, '%Y-%B-%d').strftime('%Y-%m-%d'))

    capacity = [350 if h == 'City Hotel' else 250 for h in df['hotel']]
    booking_date = [datetime.datetime.fromtimestamp(timestamp) for timestamp in df['timestamp']]
    items=zip(df['id'],
              booking_date,
              arrival_date,
              df['hotel'],
              df['gross_profit'],
              df['cancel_prob'],
              df['predicted_demand'],
              capacity)


    cts = df['hotel'].value_counts()
    cols = ["#20B2AA", "#32CD32"]
    set_ = zip(cts.values, cts.index, cols)
    max_ = np.max(cts)

    low_risk = np.sum((df['cancel_prob'] < 0.5) * 1)
    med_risk = np.sum(((df['cancel_prob'] >= 0.5) & (df['cancel_prob'] < 0.75)) * 1)
    high_risk = np.sum((df['cancel_prob'] >= 0.75) * 1)
    labels=['low risk', 'medium risk', 'high risk']
    values=[low_risk, med_risk, high_risk]
    max_risk=low_risk

    profits = []
    dates = []
    for day in np.arange(1, 32):
        adr = round(df[(df['arrival_date_month'] == 'August') &
                  (day >= df['arrival_date_day_of_month']) &
                  (day < (df['arrival_date_day_of_month'] + df['total_nights']))]['adr'].mean(), 2)
        rooms_occ = int(df[(df['arrival_date_month'] == 'August') &
                  (day >= df['arrival_date_day_of_month']) &
                  (day < (df['arrival_date_day_of_month'] + df['total_nights']))].groupby('hotel')['predicted_demand'].mean().sum())

        profits.append(adr * rooms_occ)

        dates.append('8/' + str(day))

    max_profit = np.max(profits)


    return render_template('index.html',
                      items=items,
                      max=max_,
                      set=set_,
                      bar_labels=labels,
                      bar_values=values,
                      bar_max_risk=max_risk,
                      profits=profits,
                      dates=dates,
                      max_profit=max_profit)

if __name__ == '__main__':
    APP.debug=True

    conn = pg2.connect(dbname='hotels', user='postgres',
                      host='localhost', port='5432',
                      password='password')
    cur = conn.cursor()
    conn.autocommit = True

    APP.run()

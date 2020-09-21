import psycopg2 as pg2
import sys
import datetime
import time
import pickle
import pandas as pd

sys.path.insert(0, '../src/')
from model import Model
from helpers import *


def main():
    while(True):
        query = '''
        SELECT
        r.id, r.timestamp, r.arrival_date_year,
        r.arrival_date_month, r.arrival_date_day_of_month,
        r.hotel, r.party_size, r.market_segment, r.room_difference,
        r.booking_changes, r.total_of_special_requests,
        r.adr, r.total_nights, (r.adr * r.total_nights) as gross_profit
        FROM reservations r
        WHERE r.id NOT IN (SELECT id from predictions)
        '''
        cur.execute(query)
        obj = cur.fetchall()

        if len(obj) == 0:
            continue
        df = pd.DataFrame(obj, columns=['id', 'timestamp',
                                'arrival_date_year', 'arrival_date_month', 'arrival_date_day_of_month',
                                'hotel', 'party_size', 'market_segment', 'room_difference',
                                'booking_changes', 'total_of_special_requests',
                                'adr', 'total_nights', 'gross_profit'])

        X, _ = clean_data(df, y_included=False)
        dates = []
        for y, m, d in zip(df['arrival_date_year'],
                            df['arrival_date_month'],
                            df['arrival_date_day_of_month']):
            date = str(y) + '-' + str(m) + '-' + str(d)
            dates.append(datetime.datetime.strptime(date, '%Y-%B-%d'))
        dates = pd.DatetimeIndex(dates)
        probs, bookings = model.predict_proba(X, dates)

        df['cancel_prob'] = probs
        df['predicted_demand'] = bookings

        query = 'INSERT INTO predictions\nVALUES '
        for row, item in df.iterrows():
            query += ('(%d, \'%s\', %d, %d, \'%s\', %d, %5.2f, %d, %6.2f, %4.3f, %d)' %
                    (item['id'], item['hotel'], item['timestamp'],
                    item['arrival_date_year'], item['arrival_date_month'],
                    item['arrival_date_day_of_month'], item['adr'],
                    item['total_nights'], item['gross_profit'],
                    item['cancel_prob'], item['predicted_demand']))
            if row != df.index[-1]:
                query += ',\n'
        query += ';'
        cur.execute(query)

        time.sleep(60)




if __name__ == '__main__':
    with open('model.pkl', 'rb') as f:
        model = pickle.load(f)

    print('Connecting...')
    conn = pg2.connect(dbname='hotels', user='postgres',
                      host='localhost', port='5432',
                      password='password')
    cur = conn.cursor()
    conn.autocommit = True
    print('Successfully connected!')
    main()
#!/usr/bin/python3
# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from fbprophet import Prophet
import arrow
from sqlalchemy import create_engine


def date_to_str(x):
    x = str(x)
    return x

def load_date(reg_df, date_name):
    reg_df_date=reg_df.loc[:,date_name:date_name]
    array_reg_df_date = np.array(reg_df_date).tolist()
    date_list=[]
    for i in range(len(array_reg_df_date)):
        date_list.append(array_reg_df_date[i][0])
    return date_list

def typhoon_list(con):
    sql_str = '''SELECT start_date, end_date-start_date days FROM fruveg.typhoon where level in ("中度", "強烈") 
    and end_date-start_date in (1,2,3,4)'''
    typhoon_df = pd.read_sql(sql_str, con)
    typhoon_df["start_date"] = typhoon_df["start_date"].apply(date_to_str)
    typhoon_list = load_date(typhoon_df, "start_date")
    return typhoon_list
    #typhoon

def L_new_year_list(con):
    sql_str = 'SELECT WDate FROM fruveg.date_map where LMonth = 1 and LDay = "初一";'
    L_new_year_df = pd.read_sql(sql_str, con)
    L_new_year_df["WDate"] = L_new_year_df["WDate"].apply(date_to_str)
    L_new_year_list = load_date(L_new_year_df, "WDate")
    return L_new_year_list
    # 除夕、初一～初五

def Qingming_list(con):
    sql_str = 'SELECT WDate FROM fruveg.date_map where Holiday_Name = "清明";'
    Qingming_df = pd.read_sql(sql_str, con)
    Qingming_df["WDate"] = Qingming_df["WDate"].apply(date_to_str)
    Qingming_list = load_date(Qingming_df, "WDate")
    return Qingming_list
    # 清明

def Dragon_boat_festival_list(con):
    sql_str = 'SELECT WDate FROM fruveg.date_map where Holiday_Name = "端午";'
    Dragon_boat_festival_df = pd.read_sql(sql_str, con)
    Dragon_boat_festival_df["WDate"] = Dragon_boat_festival_df["WDate"].apply(date_to_str)
    Dragon_boat_festival_list = load_date(Dragon_boat_festival_df, "WDate")
    return Dragon_boat_festival_list
    # 端午

def Zhongyuan_list(con):
    sql_str = 'SELECT WDate FROM fruveg.date_map where Holiday_Name = "中元";'
    Zhongyuan_df = pd.read_sql(sql_str, con)
    Zhongyuan_df["WDate"] = Zhongyuan_df["WDate"].apply(date_to_str)
    Zhongyuan_list = load_date(Zhongyuan_df, "WDate")
    return Zhongyuan_list
    # 中元

def Mid_autumn_festival_list(con):
    sql_str = 'SELECT WDate FROM fruveg.date_map where Holiday_Name = "中秋";'
    Mid_autumn_festival_df = pd.read_sql(sql_str, con)
    Mid_autumn_festival_df["WDate"] = Mid_autumn_festival_df["WDate"].apply(date_to_str)
    Mid_autumn_festival_list = load_date(Mid_autumn_festival_df, "WDate")
    return Mid_autumn_festival_list
    # 中秋



def main():
    # market_parameter = pd.read_csv('./market_parameter.csv')
    market_parameter = pd.read_csv('/home/lazyso/anaconda3/envs/Prophetenv/banana_project_prophet_amount/market_parameter.csv')

    con = create_engine("mysql+pymysql://dbuser:20200428@localhost:3306/fruveg")

    L_new_year = L_new_year_list(con)
    Qingming = Qingming_list(con)
    Dragon_boat_festival = Dragon_boat_festival_list(con)
    Zhongyuan =Zhongyuan_list(con)
    Mid_autumn_festival = Mid_autumn_festival_list(con)
    typhoon = typhoon_list(con)


    for i in range(market_parameter.shape[0]):
        sql = """SELECT trade_date, amount 
        FROM fruveg.Prediction_Source 
        where market_no={};""".format(market_parameter.loc[i]['market'])

        train_data_pr = pd.read_sql_query(sql, con)

        train_data_pr.rename(columns={'trade_date':'ds','amount':'y'}, inplace=True)
        train_data_pr['ds']=pd.to_datetime(train_data_pr['ds'], errors='coerce')

        L = pd.DataFrame({
            'holiday': 'L_new_year',
            'ds': pd.to_datetime(L_new_year),
            'lower_window': -3,
            'upper_window': 4,
        })
        Q = pd.DataFrame({
            'holiday': 'Qingming',
            'ds': pd.to_datetime(Qingming),
            'lower_window': -2,
            'upper_window': 0,
        })
        D = pd.DataFrame({
            'holiday': 'Dragon_boat_festival',
            'ds': pd.to_datetime(Dragon_boat_festival),
            'lower_window': -2,
            'upper_window': 0,
        })
        Z = pd.DataFrame({
            'holiday': 'Zhongyuan',
            'ds': pd.to_datetime(Zhongyuan),
            'lower_window': -2,
            'upper_window': 0,
        })
        M = pd.DataFrame({
            'holiday': 'Mid_autumn_festival',
            'ds': pd.to_datetime(Mid_autumn_festival),
            'lower_window': -2,
            'upper_window': 0,
        })
        t = pd.DataFrame({
            'holiday': 'typhoon_list',
            'ds': pd.to_datetime(typhoon),
            'lower_window': -1,
            'upper_window': 7,
        })

        holidays = pd.concat((L, Q, D, Z, M,t))

        model = Prophet(growth='linear',
                        changepoints=None,
                        n_changepoints=market_parameter.loc[i]['n_changepoints'],
                        changepoint_range=market_parameter.loc[i]['changepoint_range'],
                        yearly_seasonality=False,
                        weekly_seasonality=market_parameter.loc[i]['weekly_seasonality'],
                        daily_seasonality=False,
                        holidays=holidays,
                        seasonality_mode='additive',
                        seasonality_prior_scale=market_parameter.loc[i]['seasonality_prior_scale'],
                        holidays_prior_scale=market_parameter.loc[i]['holidays_prior_scale'],
                        changepoint_prior_scale=market_parameter.loc[i]['changepoint_prior_scale'],
                        mcmc_samples=0,
                        interval_width=0.85,
                        uncertainty_samples=1000,
                        stan_backend=None)

        model.fit(train_data_pr)
        future = model.make_future_dataframe(periods=30, freq='D')
        forecast = model.predict(future)

        train_data = forecast[['ds', 'yhat']].tail(30)
        train_data.reset_index(drop=True, inplace=True)
        train_data.rename(columns={'yhat': 'amount'}, inplace=True)
        train_data.rename(columns={'ds': 'date'}, inplace=True)
        train_data['amount'] = train_data.amount.astype(int) #取整數
        train_data['predict_date'] = arrow.now().format("YYYY-MM-DD") #取現在時間
        train_data['market_no'] = market_parameter.loc[i]['market']

        train_data.to_sql("amount_predictions", con,  index=False ,if_exists='append')

        # print(train_data)
        # print(train_data.T)
        print("market_no:{} is update on {}".format(market_parameter.loc[i]['market'],arrow.now().format("YYYY-MM-DD")))

if __name__ == "__main__":
    main()

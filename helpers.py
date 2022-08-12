#https://betterprogramming.pub/how-to-write-proper-docstrings-for-a-python-function-7c40b8d2e153
#https://peps.python.org/pep-0257/

from sqlalchemy import create_engine
import psycopg2
import numpy as np
import pandas as pd
import datetime as dt #or should I be using the np version?
import dateutil

from statsmodels.tsa.holtwinters import ExponentialSmoothing as hwes
from sklearn.metrics import mean_squared_error

import matplotlib.pyplot as plt


def get_demand(key_list, engine):
    """
    Get monthly demand data from the Redshift table r2ibp.f_actual_demand
    
    Args:
        key_list: a list of ISBN/country keys to retrieve
        engine: the SQLAlchemy engine for the Redhsift connection i.e. 
    
    Returns:
        A Pandas dataframe of monthly demand data  
    
    """
    
    key_tuple = tuple(key_list)
    
    query = f"""
    select 
        isbn + ship_to_country_key as key,
        isbn,
        ship_to_country_key as country,
        last_day(date) as month,
        sum(quantity_demanded) as qty
    from r2ibp.f_demand_actual
    where key in {key_tuple}
    and month <= current_date
    group by month, isbn, ship_to_country_key
    order by month asc
    """

    conn = engine.connect()
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    return df


def convert_to_ts(df, period = 0, start = '1970-01-01'):
    """
    Convert a dataframe of monthly demand for a single ISBN/country to a timeseries
    
    Args:
        df: the input dataframe
        period: the number of months that will be forecast with this timeseries. Used to ensure that the timeseries is long enough for the forecasting method being used
        start: the first month of the timeseries. If not set then will calculate based on the period and first month in df
    
    Returns:
        A timeseries for the ISBN/country
    """
   
    current_date = dt.date.today()
    last_full_month = dt.date(current_date.year, current_date.month, 1) + dt.timedelta(days=-1)
    #HWES needs 2 full years of data (+allow for test data period)             
    latest_ts_start = last_full_month - dateutil.relativedelta.relativedelta(months=24+period)

    #Start ts with first month of actual demand - so don't base prediction on zeros before the start
    #BUT need at least 2 years for HWES ANB may want to override
    if start == '1970-01-01':   
        start = df['month'].min()
        if start > latest_ts_start:
            start = latest_ts_start
    else:
        start = pd.to_datetime(start)
    
    idx = pd.date_range(start, last_full_month, freq='M') 
    
    ts = df.copy()[['month', 'qty']]

    ts.set_index(pd.to_datetime(ts.month), inplace=True)
    ts.drop("month", axis=1, inplace=True)

    #This is used to fill in the missing days
    ts = ts.reindex(idx, fill_value=0).astype('float64')  #For some reason it was converting to Object!

    #Replace -ve values with zeros - Needed for non-SAP sites
    ts[ts < 0] = 0
    
    return ts


def predict_using_hwes(df_demand, period, config = ['add', True, 'add', False]):
    """
    Create forecasts for multiple ISBN/countries using Holt-Winters with exponential smoothing. 
    
    Args:
        df_demand: Dataframe of monthly demand by ISBN/country
        period: the number of months to forecast
        config: a list of HWES hyperparameters
        
    Returns:
        df_errors: A dataframe of any errored ISBN/countries
        df_hwes_forecasts: A dataframe by ISBN/country and month of the actual, predicted and naive-1 value for each month in the forecast period.        
    """
    
    t,d,s,b = config       
    
    start = dt.datetime.now()

    df_errors = pd.DataFrame (columns = ['key', 'error_type', 'text'])
    df_hwes_forecasts = pd.DataFrame (columns = ['key', 'month', 'actuals', 'pred', 'naive-1'])

    key_list = df_demand['key'].unique()
        
    for key in key_list:

        ts_actuals = convert_to_ts(df_demand[df_demand['key'] == key], period)
        #Add a tiny value to avoid divide by zero errors
        ts_actuals['qty'] += 1e-10   
        
        ts_train = ts_actuals[:-period]
        ts_test = ts_actuals[-period:]
        ts_naive1 = ts_actuals[-(12+period):-12].shift(periods = 12, freq = 'M')

        #Need to catch errors if the number of months is too short (need 24 months for HW to work)
        #The other option is to pad the start of the TS so that there is enough data
        try:
            mod = hwes(ts_train,  
                
                trend= t,
                damped_trend = d,
                seasonal= s,
                use_boxcox = b,
                       
                seasonal_periods = 12,
                initialization_method="estimated",
                freq = 'M'
                )
        except ValueError as e:
            df_temp = pd.DataFrame([[key, 'ValueError', e]], columns =['key', 'error_type', 'text'])
            df_errors = df_errors.append(df_temp, ignore_index=True)
        except:
            df_temp = pd.DataFrame([[key, 'other error', '']], columns =['key', 'error_type', 'text'])

            df_errors = pd.concat([df_errors, df_temp], ignore_index = True)
            df_errors = df_errors.append(df_temp, ignore_index=True)       
        else:    
            res = mod.fit()
            pred = res.predict(len(ts_train),len(ts_train)+(period-1))
            
            #If a negatve forecast, when using additive set to zero?
            #220221 - play with this as should make HW worse
            #pred[pred<0] = 0

            #Append the results to df_forecasts
            df_temp = ts_test.copy().reset_index()
            df_temp.insert(loc=0, column='key', value=key)
            df_temp.rename(columns = {'index':'month', 'qty':'actuals'}, inplace = True)
            df_temp['pred'] = pred.values
            df_temp['naive-1'] = ts_naive1['qty'].values

            df_hwes_forecasts = pd.concat([df_hwes_forecasts, df_temp], ignore_index=True)


    print('To produce these Holt-Winters forecasts from the dataframe took', dt.datetime.now() - start) 
    print('Tried to forecast', len(key_list))
    print('Errored', df_errors.shape[0])
    
    return df_errors, df_hwes_forecasts


def calc_prediction_metrics(df_forecasts, percent = False):
    """
    Calculate accuracy metrics for the ISBN/countries in df_forecasts
    
    Args:
        df_forecasts: Dataframe of actuals, predictions and naive-1 forecasts by ISBN/country and month
        percent: Normalise accuracy metrics as percentage?
        
    Returns: A dataframe of metrics. One row per ISBN/country.
        
    """   
    
    df_metrics = pd.DataFrame (columns = ['key', 'sum_naive1', 'sum_pred', 'sum_act', 'diff_naive1_act', 'diff_pred_act',
                                      'abs_pred_closer', 'rmse_naive1', 'rmse_pred', 'pred_rmse_lower', 'rmse_pc_diff'
                                     ])
    
    key_list = df_forecasts['key'].unique()
    
    for key in key_list:
        
        #Extract the series for calculation
        df_key = df_forecasts[df_forecasts['key'] == key].fillna(0)
        y_test = df_key['actuals'] #Rename this when I tidy up!
        yhat = df_key['pred']
        y_naive1 = df_key['naive-1']
    
        #Sum each series
        sum_pred = yhat.sum()
        sum_naive1 = y_naive1.sum()
        sum_act = y_test.sum()
              
        if percent:
            diff_naive1_act = round(((sum_naive1 - sum_act)/sum_act)*100, 1)
            diff_pred_act = round(((sum_pred - sum_act)/sum_act)*100, 1)
            abs_pred_closer = (abs(diff_pred_act) < abs(diff_naive1_act))
            #Calculate the RMSE
            rmse_naive1 = round((mean_squared_error(y_test, y_naive1, squared = False)/ sum_act)*100, 1)
            rmse_pred = round((mean_squared_error(y_test, yhat, squared = False)/ sum_act)*100, 1)
          #And compare
            pred_rmse_lower = (rmse_pred < rmse_naive1)
            rmse_pc_diff = round(((rmse_pred - rmse_naive1)/rmse_naive1)*100, 1)
        else:
            diff_pred_act = round(sum_pred - sum_act, 1)
            diff_naive1_act = round(sum_naive1 - sum_act, 1)
            abs_pred_closer = (abs(diff_pred_act) < abs(diff_naive1_act))
            #Calculate the RMSE
            rmse_pred = round(mean_squared_error(y_test, yhat, squared = False), 1)
            rmse_naive1 = round(mean_squared_error(y_test, y_naive1, squared = False), 1)
            #And compare
            pred_rmse_lower = (rmse_pred < rmse_naive1)
            rmse_pc_diff = round((rmse_pred-rmse_naive1)*100/rmse_naive1, 1)
             
        sum_pred = round(sum_pred, 1) # to display more cleanly

        df_temp = pd.DataFrame([[key, sum_naive1, sum_pred, sum_act, diff_naive1_act, diff_pred_act, abs_pred_closer,
                                rmse_naive1, rmse_pred, pred_rmse_lower, rmse_pc_diff]],
                               columns = df_metrics.columns)
        
        df_metrics = pd.concat([df_metrics, df_temp], ignore_index=True)    
        
    return df_metrics


def plot_pred_naive1(df_metrics):
    """
    Visually compare the aggregate accuracy metrics for the prediction against naive-1
    
    Args: df_metrics
    
    Returns: Plots of number of cases where statistical forecast is better than naive-1. Both for RMSE and total quantity in forecast period.
    
    """
    fig, axes = plt.subplots(1, 2, figsize = (16,6))
    
    #Plot the RMSE comparison
    ax = axes[0]
    series = df_metrics['pred_rmse_lower'].sort_values(ascending = False).value_counts(sort=False)
    x = series.keys().map({True: 'True', False: 'False'})
    ax.bar(x, series)
    ax.set_title('RMSE for Statistical better than Naive-1?"')

    #Plot the totals comparison
    ax = axes[1]
    series = df_metrics['abs_pred_closer'].sort_values(ascending = False).value_counts(sort=False)
    x = series.keys().map({True: 'True', False: 'False'})
    ax.bar(x, series)
    ax.set_title('Total forecast quantity for Statistical better than Naive-1?"')
   
    plt.show();
    
    print('Total number of forecasts:', len(df_metrics))
    number = df_metrics['pred_rmse_lower'].sum()
    percent = round(df_metrics['pred_rmse_lower'].sum()*100/ len(df_metrics) , 1)
    print(f'RMSE Stastical lower than Naive-1: {number} ({percent}%)') 


def plot_sample_preds(plot_list, df_demand, df_forecasts, period):
    """
    Plot actuals, prediction and naive-1 for ISBN/country combinations
    
    Args:
        plot_list: list of ISBN/counties to plot
        df_demand: dataframe of order history
        df_forecasts: dataframe of forecasts
        period: number of months to forecast
        
    Returns: Grid of ISBN/country plots
    
    """
        
    rows = int(np.ceil(len(plot_list)/2))  #round up
    fig, axes = plt.subplots(rows, 2, figsize = (16,rows*4))
    #The following is to iterate the axes
    axes_flat = axes.flat

    for i, key in enumerate(plot_list):

        #Get the actuals and naive-1
        ts_actuals = convert_to_ts(df_demand[df_demand['key'] == key], period)
        ts_naive1 = ts_actuals[-(12+period):-12].shift(periods = 12, freq = 'M')

        #Get the forecast
        ts_pred = df_forecasts.copy()[df_forecasts['key'] == key][['month', 'pred']]
        ts_pred.set_index(pd.to_datetime(ts_pred.month), inplace=True)
        ts_pred.drop("month", axis=1, inplace=True)

        ax = axes_flat[i]
        ax.plot(ts_actuals[-24:], '-o', label="actuals")
        ax.plot(ts_pred, '-o', label="pred")
        ax.plot(ts_naive1, '-o', label="naive-1")
        ax.grid()
        ax.legend(fontsize=12)
        ax.set_title(key);

    plt.tight_layout()
    plt.show();
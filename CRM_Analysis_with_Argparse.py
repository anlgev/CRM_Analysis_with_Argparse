####################################################
# Project CRM Analysis
####################################################

# You can

import datetime as dt
import pandas as pd
import pymysql
import mysql.connector
from sqlalchemy import create_engine
from sklearn.preprocessing import MinMaxScaler
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
import argparse
import time

pd.set_option('display.max_columns', None)


def check_data(dataframe):
    """
    You can see the dataset information
    Parameters
    ----------
    dataframe

    Returns
    You can see first 5 observation, dataset shape, columns type, NaN information and some statistic information
    -------

    """
    print('##################### HEAD ####################')
    print(dataframe.head())
    print('##################### SHAPE ###################')
    print(dataframe.shape)
    print('##################### INFO ####################')
    print(dataframe.info())
    print('##################### NA ######################')
    print(dataframe.isnull().sum())
    print('##################### DESCRIBE ################')
    print(dataframe.describe([0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]).T)
    print('###############################################')


def outlier_thresholds(dataframe, variable):
    """
    Check outlier value
    Parameters
    ----------
    dataframe
    variable: str, column name

    Returns
    low limit and up limit outlier value
    -------

    """
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    """
    Change the maximum value with outlier limits
    Parameters
    ----------
    dataframe
    variable: str, column name

    Returns
    -------

    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    # dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


def crm_prep_data(dataframe):
    """
    Data preparation
    Parameters
    ----------
    dataframe

    Returns
    Dataframe
    -------

    """
    dataframe.dropna(axis=0, inplace=True)
    dataframe = dataframe[~dataframe['Invoice'].str.startswith('C', na=False)]
    dataframe = dataframe[(dataframe['Quantity'] > 0) & (dataframe['Price'] > 0)]
    replace_with_thresholds(dataframe, 'Quantity')
    replace_with_thresholds(dataframe, 'Price')
    dataframe['TotalPrice'] = dataframe["Quantity"] * dataframe["Price"]
    dataframe.rename(columns={'Customer ID': 'CustomerID'}, inplace=True)
    return dataframe




def create_rfm_t(dataframe):
    """
    Create RFM segments
    Parameters
    ----------
    dataframe

    Returns:
    Dataframe
    -------

    """
    # Warning! I use the nunique value of invoice for frequency.

    today_date = dt.datetime(2011, 12, 11)
    dataframe = dataframe.groupby('CustomerID').agg({'InvoiceDate': lambda date: (today_date - date.max()).days,
                                                     'Invoice': lambda x: x.nunique(),
                                                     'TotalPrice': lambda price: price.sum()})

    dataframe.columns = ['recency', 'frequency', 'monetary']

    # Control
    dataframe = dataframe[(dataframe['monetary'] > 0)]

    # Calculate the RFM scores
    dataframe['recency_score'] = pd.qcut(dataframe['recency'], 5, labels=[5, 4, 3, 2, 1])
    dataframe["frequency_score"] = pd.qcut(dataframe["frequency"].rank(method="first"), 5, labels=[1, 2, 3, 4, 5])

    # Create segmentation
    seg_map = {
        r'[1-2][1-2]': 'hibernating',
        r'[1-2][3-4]': 'at_risk',
        r'[1-2]5': 'cant_loose',
        r'3[1-2]': 'about_to_sleep',
        r'33': 'need_attention',
        r'[3-4][4-5]': 'loyal_customers',
        r'41': 'promising',
        r'51': 'new_customers',
        r'[4-5][2-3]': 'potential_loyalists',
        r'5[4-5]': 'champions'
    }

    dataframe['rfm_segment'] = dataframe['recency_score'].astype(str) + dataframe['frequency_score'].astype(str)
    dataframe['rfm_segment'] = dataframe['rfm_segment'].replace(seg_map, regex=True)
    dataframe = dataframe[['recency', 'frequency', 'monetary', 'rfm_segment']]
    return dataframe


def create_cltv_c_t(dataframe, pm=0.05):
    """
    Data preparation for CLTV Calculation
    Parameters
    ----------
    dataframe
    pm: int, profit margin

    Returns
    Dataframe
    -------

    """
    # Average_Order_Value = Total_Revenue / Total_Number_of_Orders
    dataframe['average_order_value'] = dataframe['monetary'] / dataframe['frequency']

    # Purchase_Frequency =  Total_Number_of_Orders / Total_Number_of_Customers
    dataframe['purchase_frequency'] = dataframe['frequency'] / dataframe.shape[0]

    # Churn_Rate = 1 - Repeat_Rate
    repeat_rate = dataframe[dataframe['frequency'] > 1].shape[0] / dataframe.shape[0]
    churn_rate = 1 - repeat_rate

    # Profit_margin
    dataframe['profit_margin'] = dataframe['monetary'] * pm

    # Customer_Value = Average_Order_Value  * Purchase_Frequency
    dataframe['cv'] = dataframe['average_order_value'] * dataframe['purchase_frequency']

    # CLTV = (Customer_Value / Churn_Rate) x Profit_margin.
    dataframe['cltv'] = (dataframe['cv'] / churn_rate) * dataframe['profit_margin']

    # MinMaxScaler
    scaler = MinMaxScaler(feature_range=(1, 100))
    scaler.fit(dataframe[['cltv']])
    dataframe['cltv_c_score'] = scaler.transform(dataframe[['cltv']])

    # cltv_c segmentasyon
    dataframe['cltv_c_segment'] = pd.qcut(dataframe['cltv_c_score'], 3, labels=['C', 'B', 'A'])

    dataframe = dataframe[["recency", "frequency", "monetary", "rfm_segment",
                           "cltv_c_score", "cltv_c_segment"]]

    return dataframe



def create_cltv_p_prep(dataframe):
    """
    Data preparation for CLTV Prediction (gamagama and bgnbd_
    Parameters
    ----------
    dataframe

    Returns
    Dataframe
    -------

    """

    today_date = dt.datetime(2011, 12, 11)
    rfm = dataframe.groupby('CustomerID').agg({'InvoiceDate': [lambda date: (today_date - date.max()).days,
                                                               lambda date: (today_date - date.min()).days],
                                               'Invoice': lambda x: x.nunique(),
                                               'TotalPrice': lambda price: price.sum()})

    rfm.columns = rfm.columns.droplevel(0)
    rfm.columns = ['recency', 'T', 'frequency', 'monetary']

    # Calculate monetary avg
    temp_df = dataframe.groupby(['CustomerID', 'Invoice']).agg({'TotalPrice': ['mean']}).reset_index()
    temp_df.columns = temp_df.columns.droplevel(0)
    temp_df.columns = ["CustomerID", "Invoice", "total_price_mean"]
    temp_df2 = temp_df.groupby(["CustomerID"], as_index=False).agg({"total_price_mean": ["mean"]})
    temp_df2.columns = temp_df2.columns.droplevel(0)
    temp_df2.columns = ["CustomerID", "monetary_avg"]

    if rfm.index.isin(temp_df2["CustomerID"]).all():
        rfm = rfm.merge(temp_df2, how='left', on='CustomerID')
    else:
        print('merge, index problem')

    rfm.set_index("CustomerID", inplace=True)
    rfm.index = rfm.index.astype(int)

    # convert daily values to weekly for recency and tenure
    rfm["recency_weekly"] = rfm["recency"] / 7
    rfm["T_weekly"] = rfm["T"] / 7

    return rfm


# Gamagama ve Bnbd prediction uyarlama fonksiyonlari

def create_cltv_pred(dataframe, w=4, m=1):
    """
    Gamagama and BGNBD model and prediction
    Parameters
    ----------
    dataframe
    w: int, week information for BGNBD model
    m: int, month information for gamama model

    Returns
    Dataframe
    -------

    """
    # BGNBD

    dataframe = dataframe[dataframe["monetary_avg"] > 0]
    dataframe["frequency"] = dataframe["frequency"].astype(int)

    bgf = BetaGeoFitter(penalizer_coef=0.001)
    bgf.fit(dataframe['frequency'],
            dataframe['recency_weekly'],
            dataframe['T_weekly'])

    dataframe[f'exp_sales_{w}_week'] = bgf.predict(w,
                                                   dataframe['frequency'],
                                                   dataframe['recency_weekly'],
                                                   dataframe['T_weekly'])

    # Gamagama - expected_average_profit
    ggf = GammaGammaFitter(penalizer_coef=0.001)
    ggf.fit(dataframe['frequency'], dataframe['monetary_avg'])
    dataframe["expected_average_profit"] = ggf.conditional_expected_average_profit(dataframe['frequency'],
                                                                                   dataframe['monetary_avg'])

    # CLTV Prediction
    cltv = ggf.customer_lifetime_value(bgf,
                                       dataframe['frequency'],
                                       dataframe['recency_weekly'],
                                       dataframe['T_weekly'],
                                       dataframe['monetary_avg'],
                                       time=m,
                                       freq="W",
                                       discount_rate=0.01)

    dataframe[f'cltv_p_{m}_month'] = cltv

    scaler = MinMaxScaler(feature_range=(1, 100))
    dataframe['cltv_p_score'] = scaler.fit_transform(dataframe[[f'cltv_p_{m}_month']])

    # cltv_p Segment
    dataframe['cltv_p_segment'] = pd.qcut(dataframe['cltv_p_score'], 3, labels=['C', 'B', 'A'])

    new_col = dataframe.columns[~dataframe.columns.isin(['recency', 'frequency', 'monetary'])]
    dataframe = dataframe[new_col]

    return dataframe

# Argparse information

parser = argparse.ArgumentParser(description='CRM Analysis')

parser.add_argument('-w', '--week', metavar='', type=int, default=4,
                    help="How many weeks later would you like to see the expected sales")
parser.add_argument('-m', '--month', metavar='', type=int, default=1,
                    help="How many month later would you like to see the cltv")
parser.add_argument('-op', '--openreport', metavar='', type=str, default=None,
                    help="If you want to see process pleas write 'openreport'")

args = parser.parse_args()


def automated_crm_r(w=4, m=1):
    """
    To get the report print
    Parameters
    ----------
    w: int, week information for BGNBD model
    m: int, month information for gamagama model

    Returns
    Dataframe
    -------

    """
    start_time = time.perf_counter()
    print('################ Reading Data ########################')
    df_ = pd.read_excel('C:/Users/Asus/PycharmProjects/bootcamp/dataset/online_retail_II.xlsx',
                        sheet_name="Year 2010-2011")
    df = df_.copy()
    end_time = time.perf_counter()
    print(df.head())
    print(f'Data reading time: {end_time - start_time: .4f} seconds')
    print('################# Data Reading Finish #################')
    print('-------------------------------------------------------------')

    start_time = time.perf_counter()
    print('################# Row Data Preparation ###########')
    df_prep = crm_prep_data(df)
    print(check_data(df_prep))
    end_time = time.perf_counter()
    print(f'Data prep time: {end_time - start_time: .4f} seconds')
    print('################# Data Preparation Finish ################')
    print('-------------------------------------------------------------')

    print('################# Create RFM Data ################')
    start_time = time.perf_counter()
    rfm_df = create_rfm_t(df_prep)
    print(rfm_df.head())
    end_time = time.perf_counter()
    print(f'Create RFM data time: {end_time - start_time: .4f} seconds')
    print('################# RFM Data Ready ################')
    print('-------------------------------------------------------------')

    start_time = time.perf_counter()
    print('################# Calculated CLTV Data ################')
    rfm_cltv_df = create_cltv_c_t(rfm_df)
    print(rfm_cltv_df.head())
    end_time = time.perf_counter()
    print(f'Calculated CLTV data time: {end_time - start_time: .4f} seconds')
    print('################# Claculated Data Ready ################')
    print('-------------------------------------------------------------')

    start_time = time.perf_counter()
    print('################# Preparation Data for CLTV Pred ################')
    rfm_cltv_prep = create_cltv_p_prep(df_prep)
    print(rfm_cltv_prep.head())
    end_time = time.perf_counter()
    print(f'Preparation data time: {end_time - start_time: .4f} seconds')
    print('################# Data Ready for Prediction CLTV ################')
    print('-------------------------------------------------------------')

    start_time = time.perf_counter()
    print('################# Prediction CLTV Data ################')
    rfm_cltv_pred = create_cltv_pred(rfm_cltv_prep, w, m)
    crm_final = rfm_cltv_df.merge(rfm_cltv_pred, on='CustomerID', how='left')
    crm_final.index = crm_final.index.astype(int)
    end_time = time.perf_counter()
    print(f'Prediction CLTV time: {end_time - start_time: .4f} seconds')
    print('################# Predictin CLTV Data Ready ################')
    print('-------------------------------------------------------------')
    return crm_final.head()


def automated_crm(w=4, m=1):
    """
    To get print without report
    Parameters
    ----------
    w: int, week information for BGNBD model
    m: int, month information for gamagama model

    Returns
    Dataframe
    -------

    """
    print('############## Process Started ########################')
    df_ = pd.read_excel('C:/Users/Asus/PycharmProjects/bootcamp/dataset/online_retail_II.xlsx',
                        sheet_name="Year 2010-2011")
    df = df_.copy()
    df_prep = crm_prep_data(df)
    rfm_df = create_rfm_t(df_prep)
    rfm_cltv_df = create_cltv_c_t(rfm_df)
    rfm_cltv_prep = create_cltv_p_prep(df_prep)
    rfm_cltv_pred = create_cltv_pred(rfm_cltv_prep, w, m)
    crm_final = rfm_cltv_df.merge(rfm_cltv_pred, on='CustomerID', how='left')
    crm_final.index = crm_final.index.astype(int)
    return crm_final.head()



if __name__ == '__main__':
    if args.openreport == 'openreport':
        print(automated_crm_r(args.week, args.month))
    else:
        print(automated_crm(args.week, args.month))


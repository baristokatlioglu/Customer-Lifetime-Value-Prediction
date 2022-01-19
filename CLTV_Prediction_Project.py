import datetime as dt
import pandas as pd
import matplotlib.pyplot as plt
from lifetimes import BetaGeoFitter
from lifetimes import GammaGammaFitter
from lifetimes.plotting import plot_period_transactions
from sklearn.preprocessing import MinMaxScaler

def outlier_thresholds(dataframe, variable):
    quartile1 = dataframe[variable].quantile(0.01)
    quartile3 = dataframe[variable].quantile(0.99)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit


def replace_with_thresholds(dataframe, variable):
    low_limit, up_limit = outlier_thresholds(dataframe, variable)
    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit
    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit

# Reading data from excel
df_ = pd.read_excel("online_retail_II.xlsx",
                    sheet_name="Year 2010-2011")
df = df_.copy()

# Data preprocessing
df.dropna(inplace=True)
df = df[~df["Invoice"].str.contains("C", na=False)]
df = df[(df["Quantity"] > 0)]
df = df[(df["Price"] > 0)]

replace_with_thresholds(df, "Quantity")
replace_with_thresholds(df, "Price")

df.describe().T

df["TotalPrice"] = df["Quantity"]*df["Price"]
today_date = dt.datetime(2011,12,11)

#########################
# Preparation of Lifetime Data Structure
#########################

# recency: recency: last count purchase - first purchase date. Weekly. (according to analysis day on cltv_df, user specific here)
# The date the customer made the last purchase - the date the customer made the first purchase
# T: The age of the customer. Weekly. (how long before the analysis date the first purchase was made)
# frequency: total number of repeat purchases (frequency>1)
# monetary_value: average earnings per purchase

# CLTV for UK customers
df_uk = df[df["Country"] == "United Kingdom"]
df_uk.head()

cltv_df = df_uk.groupby("Customer ID").agg({"InvoiceDate": [lambda InvoiceDate : (InvoiceDate.max() - InvoiceDate.min()).days,
                                                         lambda date : (today_date - date.min()).days],
                                         "Invoice" : lambda Invoice : Invoice.nunique(),
                                         "TotalPrice" : lambda TotalPrice : TotalPrice.sum()
                                         })
cltv_df.head()
cltv_df.columns = cltv_df.columns.droplevel(0) # It removes the level difference in the columns after groupby.
cltv_df.columns = ["recency", "T", "frequency", "monetary"]
cltv_df.head()

cltv_df["monetary"] = cltv_df["monetary"] / cltv_df["frequency"]
# Average monetary value for cltv p expected by the gamma gamma model.

cltv_df = cltv_df[(cltv_df['frequency'] > 1)]
# customers with multiple purchases

cltv_df.describe().T


bgf = BetaGeoFitter(penalizer_coef=0.001)
bgf.fit(cltv_df["frequency"], cltv_df["recency"], cltv_df["T"])

cltv_df["expected_purc_1_week"] = bgf.predict(1,
                                              cltv_df['frequency'],
                                              cltv_df['recency'],
                                              cltv_df['T'])

cltv_df["expected_purc_1_month"] = bgf.predict(4,
                                               cltv_df['frequency'],
                                               cltv_df['recency'],
                                               cltv_df['T'])
cltv_df.head()
####
# Establishment of the Gamma Gamma model
####

ggf = GammaGammaFitter(penalizer_coef=0.01)
ggf.fit(cltv_df['frequency'], cltv_df['monetary'])

cltv_df["expected_average_profit"] = ggf.conditional_expected_average_profit(cltv_df['frequency'],
                                                                             cltv_df['monetary'])
# If monetary values less than 0 are observed, values greater than 0 can be taken.

#######
# For BG NBD model, Recency and T values should be Weekly.
#######

cltv_df["recency"] = cltv_df["recency"] / 7
cltv_df["T"] = cltv_df["T"] / 7
cltv_df.head()

#####
# Calculation of CLTV
#####

cltv = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=6,  # 6 aylık
                                   freq="W",  # T'nin frekans bilgisi.
                                   discount_rate=0.01)

cltv.head(10)
cltv = cltv.reset_index()
cltv_final = cltv_df.merge(cltv, on="Customer ID", how="left")
cltv_final.sort_values(by="clv", ascending=False).head(10)

# Since we cannot compare, the clv value is standardized between 0-1.
scaler = MinMaxScaler(feature_range=(0, 1))
scaler.fit(cltv_final[["clv"]])
cltv_final["scaled_clv"] = scaler.transform(cltv_final[["clv"]])

cltv_final.sort_values(by="scaled_clv", ascending=False).head()

####
# 1-month and 12-month CLTV calculation
####

cltv_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=1,  # 1-month
                                   freq="W",  # Frequency information of T.
                                   discount_rate=0.01)

cltv_month = cltv_month.reset_index()
cltv_month.head()
cltv_12_month = ggf.customer_lifetime_value(bgf,
                                   cltv_df['frequency'],
                                   cltv_df['recency'],
                                   cltv_df['T'],
                                   cltv_df['monetary'],
                                   time=12,  # 12-month
                                   freq="W",  # Frequency information of T.
                                   discount_rate=0.01)
cltv_12_month = cltv_12_month.reset_index()

cltv_month.sort_values(by="clv", ascending=False).head(10)
cltv_12_month.sort_values(by="clv", ascending=False).head(10)

# While customers with 16000 IDs are in the 7th place in the 1-month forecast, they are seen in the 9th place in the 12-month forecast.
# This means that we can say that the buying pattern is less than that of other customers.

####
# Divide UK customers into 4 segments based on 6-month CLTV.
####

cltv_final["segment"] = pd.qcut(cltv_final["scaled_clv"], 4, labels=["D", "C", "B", "A"])
cltv_final.head()

cltv_final.groupby("segment").agg(
    {"count", "mean", "sum"})

# In order for customers in the C segment to shop more, discounts and coupons can be defined for these users.
# Promotions and gifts can be given to customers in the B segment to create brand loyalty.
from utils_bert import *
from utils_time_series import *
from pipeline import Pipeline

import yfinance as yf


start = "2019-01-01"
end   = "2020-03-20"
threshold = "2019-11-01"
window = 10
epochs = 10#250

# Download series
stock_names = ["MSFT"]
stocks = {}
for stock in stock_names:
    df = yf.download(stock, start= start, end= end)
    stocks[stock] = df

series = stocks["MSFT"]
columns = ["Close", "Volume"] #["Close","Volume"]
target = "Close"
series = pd.DataFrame(series[columns])

dates = pd.DataFrame(index = pd.date_range(start,end))
series = pd.merge(dates,series, right_index=True,left_index=True, how = "left").bfill().ffill()


standardize = True
dummy = True
returns = True

if dummy:
    series = dummy_series(200,2,True, start)
    threshold = "2019-05-20"
    target = 0

p = Pipeline(series,target_column= target)

# Create and transform features
p.add_moving_averages(7,10, 20)

p.dropna()
if returns:
    p.prices_to_returns()
if standardize:
    p.standarize()
p.create_rolling_windows_and_target(window)

# create train and test datasets
p.split_by_date(threshold)
n_features = p.X.shape[-1]

#define and train model
p.model = lstm_model(window,n_features)
p.fit_model(epochs=epochs)
p.make_predictions()
if standardize:
    p.destandarize_predictions()
if returns:
    p.returns_to_prices_predictions()
p.plots()






#




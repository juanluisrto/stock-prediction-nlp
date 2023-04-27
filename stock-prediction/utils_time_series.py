import pandas as pd, numpy as np
from matplotlib import pyplot as plt
from rolling_window import rolling_window

LOAD_TENSORFLOW = False
if LOAD_TENSORFLOW:
    from tensorflow.keras.layers import LSTM, Dense, Input, Flatten
    from tensorflow.keras.layers import TimeDistributed, Dropout, BatchNormalization, MaxPooling1D
    from tensorflow.keras.models import Sequential

    
import pmdarima as pm
import statsmodels as st

from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.metrics import mean_squared_error

from skforecast.ForecasterAutoreg import ForecasterAutoreg
from skforecast.ForecasterAutoregDirect import ForecasterAutoregDirect
from skforecast.model_selection import backtesting_forecaster
from skforecast.model_selection import grid_search_forecaster

# computes and returns intersection among series
def series_intersection(a,b):
    a.index = pd.DatetimeIndex(a.index)
    b.index = pd.DatetimeIndex(b.index)
    intersection = pd.DatetimeIndex([value for value in a.index if value in b.index])
    return a.loc[intersection], b.loc[intersection]


def rolling_window_bert_2nd_dim(a, window):
    shape = (a.shape[0] - window + 1, window, a.shape[1])
    strides = (a.strides[0], a.strides[1]*a.shape[1], a.strides[1])
    #print(shape, strides)
    return np.lib.stride_tricks.as_strided(a, shape=shape, strides=strides)


def rolling_window_2d(x, steps):
    roll = rolling_window(x, window = (steps,x.shape[-1]))
    return np.squeeze(roll)[:-1]


#Computes de moving average of a series for the past n days
def moving_avg(s,days):
    if len(s) < days:
        raise Exception(f"Can't compute moving average of lenght {days} on a series of lenght {len(s)}")
    df = pd.DataFrame(s)
    for i in range(1,days):
        df[str(i)] = s.shift(i)
    return pd.Series(df.values.mean(axis = 1), index=s.index, name = "ma" + str(days))


def ma5(s):
    return moving_avg(s,5)

def ma20(s):
    return moving_avg(s,20)

def ma50(s):
    return moving_avg(s,50)

def destandarize(data, mean, std):
    return data*std + mean

def standarize(data):
    return (data - data.mean())/data.std(), data.mean(), data.std()

def r2p(d, start = 100):
    return returns_to_prices(d,start)

def p2r(d):
    return pd.DataFrame(d).pct_change(1).fillna(0).values.squeeze()


def returns_to_prices(array, start = 100):
    """Transform dataframe columns from returns to prices with a start value"""
    array = array.T
    if array.ndim == 1:
        array = array.reshape(1,-1)
    zeros = np.zeros(array.shape[0])
    concat = np.column_stack((zeros, array))
    cum_returns = np.apply_along_axis(np.cumprod,1,concat + 1)
    prices = np.apply_along_axis(lambda x: np.multiply(start, x), 0, cum_returns)
    return prices.T

def dummy_series(length, dim, as_df = True, start = "2019-01-01"):
    df = pd.DataFrame(np.array([dim*[i] for i in range(1, length + 1)]),pd.date_range(start, periods = length))
    return df if as_df else df.values

def generate_labels(array, window, index):
    return array[index][window:]

def diff2p(d, start):
    prices = []
    cum_diff = start
    for diff in d:
        prices.append(diff + cum_diff)
        cum_diff += diff
    return prices

def lstm_model_orig(n_steps, n_features = 5):
    model = Sequential()
    model.add(LSTM(50, input_shape = (n_steps, n_features), return_sequences = True))
    model.add(TimeDistributed(Dense(20, activation='elu')))
    model.add(Flatten())
    model.add(Dense(1, activation='elu'))
    model.compile(loss='mean_squared_error', optimizer='adam')
    print(model.summary())
    return model

def lstm_model(n_steps, n_features):
    model = Sequential()
    # Adding the first LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True, input_shape=(n_steps, n_features)))
    model.add(Dropout(0.2))
    # Adding a second LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a third LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))
    # Adding a fourth LSTM layer and some Dropout regularisation
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))
    # Adding the output layer
    model.add(Dense(units=1,activation='elu'))
    # Compiling the RNN
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model


def lstm_train_and_predict(X_train, X_test, y_train, y_test, window, standarize_output=False, mode="returns", start=100,
                           epochs=10):
    y_test_orig = y_test.values
    y_train_orig = y_train.values

    if standarize_output:
        y_test, y_test_mean, y_test_std = standarize(y_test)
        y_train, y_train_mean, y_train_std = standarize(y_train)

    X_train_roll = rolling_window_bert_2nd_dim(X_train.values, window)
    X_test_roll = rolling_window_bert_2nd_dim(X_test.values, window)
    y_train_roll = y_train[window - 1:].values
    y_test_roll = y_test[window - 1:].values

    lstm = lstm_model(window)
    history = lstm.fit(X_train_roll, y_train_roll, validation_data=(X_test_roll, y_test_roll), epochs=epochs)
    pred_tr = lstm.predict(X_train_roll)
    pred_te = lstm.predict(X_test_roll)

    if standarize_output:
        pred_tr = destandarize(pred_tr, y_train_mean, y_train_std)
        pred_te = destandarize(pred_te, y_test_mean, y_test_std)
        y_train_roll = destandarize(y_train_roll, y_train_mean, y_train_std)
        y_test_roll = destandarize(y_test_roll, y_test_mean, y_test_std)

    if mode == "returns":
        pred_prices_tr = r2p(pred_tr, start)
        pred_prices_te = r2p(pred_te, start)
        real_prices_tr = r2p(y_train_orig, start)
        real_prices_te = r2p(y_test_orig, start)

    if mode == "diff":
        pred_prices_tr = diff2p(pred_tr, start)
        pred_prices_te = diff2p(pred_te, start)
        real_prices_tr = diff2p(y_train_orig, start)
        real_prices_te = diff2p(y_test_orig, start)

    print("TEST DATA")
    plt.plot(real_prices_te)
    plt.plot(pred_prices_te)
    plt.legend(["y_real", "y_pred"])
    plt.show()

    print("TRAIN DATA")
    plt.plot(real_prices_tr)
    plt.plot(pred_prices_tr)
    plt.legend(["y_real", "y_pred"])
    plt.show()

    plt.plot(history.history["loss"])
    plt.plot(history.history["val_loss"])
    plt.legend(["loss", "val_loss"])
    plt.show()
    return lstm, pred_tr, pred_te


###############################################
####            TFM NEW ATTEMPT            ####
###############################################


def train_test_split(df, threshold = None, train_pctg = None):
    df = df.copy()
    assert (threshold is None and train_pctg is not None) or (train_pctg is None and threshold is not None), "Provide either threshold or train_pctg"
    if threshold:
        return (df[:threshold], df[threshold:])
    elif train_pctg:
        i = int(len(df)*train_pctg)
        return (df.iloc[:i], df.iloc[i:])

def fit_model(df, target_col = "Close", exog_cols = None, **kwargs):
    
    forecaster = ForecasterAutoreg(
                     regressor = GradientBoostingRegressor(random_state=123),
                     lags  = range(1,14)
                 )
    
    forecaster.fit(
        y    = df[target_col],
        exog = df[exog_cols] if exog_cols else None
    )
    return forecaster

def fit_grid_model(df, target_col = "Close", exog_cols = None, **kwargs):
    
    forecaster = ForecasterAutoreg(
                 regressor = GradientBoostingRegressor(random_state=123),
                 lags      = range(1,14)
             )
    # Lags used as predictors
    lags_grid = [3, 7, 14]

    # Regressor's hyperparameters
    
    param_grid = {
            'max_depth': [3 ,5, 7],
            'n_estimators': [50, 100, 150]
            #'learning_rate': [0.1, 0.01, 0.05]
        }
    
    grid_search_result =   grid_search_forecaster(
                        forecaster         = forecaster,
                        y                  = df[target_col],
                        param_grid         = param_grid,
                        lags_grid          = lags_grid,
                        steps              = 1,
                        refit              = True,
                        metric             = 'mean_squared_error',
                        initial_train_size = int(len(df)*0.70),
                        fixed_train_size   = False,
                        return_best        = True,
                        verbose            = False,
                        exog               = df[exog_cols] if exog_cols else None
               )
    return forecaster


def predict_ahead(forecaster, df, exog_cols = None):
    
    predictions = forecaster.predict(
                    steps = len(df),
                    exog = df[exog_cols] if exog_cols else None
                    )
    df["pred"] = pd.Series(predictions.values, index = df.index)
    return df.copy()


from pmdarima.arima import ndiffs


def fit_arima_model(df, target_col = "Close", exog_cols = None):
    
    kpss_diffs = ndiffs(df.Close, alpha=0.05, test='kpss', max_d=6)
    adf_diffs = ndiffs(df.Close, alpha=0.05, test='adf', max_d=6)
    n_diffs = max(adf_diffs, kpss_diffs)
    
    forecaster = pm.auto_arima(
        y = df[target_col],
        X = df[exog_cols] if exog_cols else None,
        d = n_diffs,
        seasonal=False, 
        stepwise=True,
        suppress_warnings=True,
        max_p=6,
        trace=2
        )
    return forecaster

def predict_arima_ahead(forecaster, df, exog_cols = None):
    
    predictions = forecaster.predict(
                    n_periods = len(df),
                    X = df[exog_cols] if exog_cols else None
                    )
    df["pred"] = pd.Series(predictions.values, index = df.index)
    return df.copy()

def predict_arima_ahead_update(forecaster, df,  target_col = "Close", exog_cols = None):

    predictions = []
    for date, row in df.iterrows():
        exog_data = pd.DataFrame(row[exog_cols]).T if exog_cols else None
        next_pred, _ = forecaster.predict(  
                                n_periods=1,
                                return_conf_int=True,
                                X = exog_data 
                                )
        if isinstance(next_pred,pd.Series):
            next_pred = next_pred.values
        predictions.append(next_pred[0])
        forecaster.update(row[target_col], X = exog_data)
    df["pred"] = pd.Series(predictions, index = df.index)
    return df.copy()

def compute_test_error(pred_df, y_true_col = "Close", y_pred_col = "pred", exog_cols = None):
    error_mse = mean_squared_error(
                    y_true = pred_df[y_true_col],
                    y_pred = pred_df[y_pred_col]
                )**0.5
    print(f"Test error (rmse): {error_mse}")
    if exog_cols:
        print(f"Exog cols : {exog_cols}")
    pred_df[[y_true_col, y_pred_col]].plot()
    return error_mse
    





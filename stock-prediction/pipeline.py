from utils_time_series import *
from sklearn.preprocessing import StandardScaler

class Pipeline:

    def __init__(self, df, is_standardized = False, is_price = True, target_column = "Close"):
        self.df_orig = df.copy()
        self.df = df
        self.is_standardized = is_standardized
        self.is_price = is_price
        self.target_column = target_column
        self.scaler = StandardScaler()
        self.X = None
        self.y = None
        self.model = None
        self.max_ma = 0

    def add_moving_averages(self, *args):
        self.max_ma = max(args)
        for k in args:
            ma_k = moving_avg(self.df[self.target_column], k)
            ma_k = ma_k.fillna(method= "bfill")
            self.df[ma_k.name] = ma_k
        #self.df_orig = self.df.copy()

    def standarize(self):
        self.df[:] = self.scaler.fit_transform(self.df)
        self.is_standardized = True

    def destandarize(self, series):
        ix = self.df.columns.get_loc(self.target_column)
        #mean = self.scaler.mean_[ix]
        #std = self.scaler.scale_[ix]
        #return series * std + mean
        array = np.zeros((len(series), self.scaler.n_features_in_))
        array[:, ix] = series.values.flatten()
        original =  self.scaler.inverse_transform(array)[:,ix]
        return pd.DataFrame(original, index= series.index)


    def destandarize_predictions(self):
        self.train_predictions = self.destandarize(self.train_predictions)
        self.test_predictions = self.destandarize(self.test_predictions)

    def dropna(self):
        self.df = self.df.dropna()

    def returns_to_prices_df(self, start = 100):
        self.df[:] = returns_to_prices(self.df, start)
        self.is_price = True

    def prices_to_returns(self):
        self.df[:] = p2r(self.df)
        self.is_price = False

    def create_rolling_windows_and_target(self,window):
        self.window = window
        self.y = self.df[self.target_column][window + self.max_ma:]
        rolling_windows = rolling_window_2d(self.df.values[self.max_ma:], window)
        self.X = rolling_windows  #pd.DataFrame(rolling_windows, index = self.y.index)

    def split_by_date(self, date):
        date = pd.Timestamp(date)
        self.threshold = date
        assert(self.df.index.get_loc(self.threshold) > self.window + self.max_ma), "Threshold <= window + maximum moving average"
        self.y_train_prices = self.df_orig[self.target_column][:date].iloc[self.window + self.max_ma:]
        self.y_test_prices = self.df_orig[self.target_column][date + pd.DateOffset(1):]
        self.y_train_returns = self.y[:date]
        self.y_test_returns  = self.y[date + pd.DateOffset(1):]
        self.y_train = self.y_train_returns.values
        self.y_test = self.y_test_returns.values
        self.X_train, self.X_test = self.X[:len(self.y_train)], self.X[len(self.y_train):]

    def fit_model(self, epochs):
        self.history = self.model.fit(self.X_train, self.y_train,
                                      validation_data=(self.X_test, self.y_test),
                                      epochs=epochs,
                                      batch_size = 32)
    def make_predictions(self):
        self.train_predictions = pd.DataFrame(self.model.predict(self.X_train), index = self.y_train_returns.index)
        self.test_predictions = pd.DataFrame(self.model.predict(self.X_test), index = self.y_test_returns.index)


    def returns_to_prices_predictions(self):
        date_train = self.y.index[0] # + pd.DateOffset(days = - 1)
        date_test = pd.Timestamp(self.threshold)  + pd.DateOffset(days = 1)
        pos = self.df_orig.columns.get_loc(self.target_column)
        start_price_train = self.df_orig.loc[date_train][pos]
        start_price_test = self.df_orig.loc[date_test][pos]
        self.train_predictions = returns_to_prices(self.train_predictions, start_price_train)
        self.test_predictions = returns_to_prices(self.test_predictions, start_price_test)
        # We put predictions inside dataframes
        self.train_predictions = pd.DataFrame(self.train_predictions , index = pd.date_range(date_train, periods = len(self.train_predictions) ))
        self.test_predictions = pd.DataFrame(self.test_predictions, index = pd.date_range(date_test, periods = len(self.test_predictions)))

    def add_sentiment(self,path):
        sentiment = pd.read_pickle(path)
        sentiment.columns = [f"{i}_stars" for i in range(1, 6)]
        self.df = pd.merge(self.df, sentiment, right_index=True, left_index=True, how="left").bfill().ffill()


    def plots(self):
        fig, ax = plt.subplots(figsize = (10,6))
        #ax.title = "REAL VS PREDICTED PRICES"
        plt.plot(self.y_test_prices)
        plt.plot(self.test_predictions)
        plt.plot(self.y_train_prices)
        plt.plot(self.train_predictions)
        plt.legend(["y_test_prices","y_test_pred", "y_train_prices","y_train_pred"])
        plt.show()

        fig, ax = plt.subplots(figsize=(7, 5))
        plt.plot(self.history.history["loss"])
        plt.plot(self.history.history["val_loss"])
        plt.legend(["loss", "val_loss"])
        plt.show()


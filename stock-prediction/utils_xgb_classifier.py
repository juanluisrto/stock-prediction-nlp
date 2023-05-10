import xgboost as xgb
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

from datetime import datetime
from dataclasses import dataclass

from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.metrics import roc_auc_score, confusion_matrix
from sklearn.model_selection import StratifiedKFold, TimeSeriesSplit
from sklearn.utils.class_weight import compute_sample_weight

from yellowbrick.classifier import ROCAUC


label_encoding = {0: "sell", 1: "hold", 2: "buy"}


def timer(start_time=None):
    if not start_time:
        start_time = datetime.now()
        return start_time
    elif start_time:
        thour, temp_sec = divmod((datetime.now() - start_time).total_seconds(), 3600)
        tmin, tsec = divmod(temp_sec, 60)
        print('\n Time taken: %i hours %i minutes and %s seconds.' % (thour, tmin, round(tsec, 2)))


def _get_base_xgb_clf():
    return xgb.XGBClassifier(objective="multi:softprob",
                             eval_metric = ["auc", "mlogloss", "merror"],
                             early_stopping_rounds = 20,
                            )

def _get_sample_weights(y):
    #class_weights = {0 : 0.4, 1 : 0.2, 2: 0.4}
    return compute_sample_weight(class_weight= "balanced", y=y)
    



def simple_xgb_clf(X_train, y_train, X_test, y_test):
    
    clf = _get_base_xgb_clf()
    sample_weights = _get_sample_weights(y_train)
    
    eval_set = [(X_train, y_train),(X_test, y_test)]
    
    clf.fit(X_train, y = y_train,
            sample_weight = sample_weights,
            eval_set = eval_set,
            verbose = False)
    return clf


def random_search_xgb_clf(X_train, y_train, X_test, y_test):
    "Random search over xgboost"
    # A parameter grid for XGBoost
    params = {
            'min_child_weight': [1, 5, 10],
            'gamma': [0.5, 1, 1.5, 2, 5],
            'subsample': [0.6, 0.8, 1.0],
            'colsample_bytree': [0.6, 0.8, 1.0],
            'max_depth': [3, 4, 5],
            'learning_rate' : [0.2, 0.1, 0.01],
            'n_estimators' : [200, 400, 600],
            }

    model = _get_base_xgb_clf()
    sample_weights = _get_sample_weights(y_train)
    eval_set = [(X_train, y_train),(X_test, y_test)]


    folds = 4
    param_comb = 120

    skf = StratifiedKFold(n_splits=folds, shuffle = True, random_state = 42)
    #ts_split = TimeSeriesSplit(n_splits=folds)

    random_search = RandomizedSearchCV(model, 
                                       param_distributions=params, 
                                       n_iter=param_comb,
                                       scoring = "f1_weighted",
                                       n_jobs=4,
                                       cv= skf.split(X_train,y_train),
                                       verbose=0,
                                       random_state=42 )

    # Here we go
    start_time = timer(None) # timing starts from this point for "start_time" variable
    random_search.fit(X_train, y_train, eval_set = eval_set, sample_weight = sample_weights, verbose = False)
    timer(start_time)
    return random_search



def plot_model_metrics(model):
    results = model.evals_result()
    n_metrics = len(model.eval_metric)

    fig, axes = plt.subplots(nrows = 1, ncols = n_metrics, figsize = (20,8))


    for metric, ax in zip(model.eval_metric, axes):
        ax.plot(results['validation_0'][metric], label='Train')
        ax.plot(results['validation_1'][metric], label='Test')
        ax.legend()
        ax.set_ybound(0,1.2)
        ax.set_title(metric, {"fontsize" : 15})
    return fig

def plot_model_tree(model):
    fig, ax = plt.subplots(figsize = (12,12), dpi = 300)
    xgb.plot_tree(model, ax = ax)
    plt.show()
    
    
def plot_ROC_curve(model, xtrain, ytrain, xtest, ytest, **kwargs):

    # Creating visualization with the readable labels
    visualizer = ROCAUC(model, encoder=label_encoding, **kwargs)
                                        
    # Fitting to the training data first then scoring with the test data                                    
    visualizer.fit(xtrain, ytrain)
    visualizer.score(xtest, ytest)
    visualizer.show()
    
    return visualizer



def confusion_matrix_df(df, target_col = "target", pred_col = "pred"):
    # Get unique class labels
    labels = sorted(df[target_col].unique())

    # Compute confusion matrix
    matrix = confusion_matrix(df[target_col], df[pred_col], labels=labels)

    # Create DataFrame from confusion matrix
    df_matrix = pd.DataFrame(matrix, index=labels, columns=labels)

    # Print confusion matrix DataFrame
    df_matrix = df_matrix.add_suffix('_pred')
    df_matrix = df_matrix.rename('{}_target'.format)
    return df_matrix
    

@dataclass
class TrainedClassifier:
    """Class for keeping track of an item in inventory."""
    name : str
    X_train: pd.DataFrame
    y_train: pd.Series
    X_test: pd.DataFrame
    y_test: pd.Series
    clf: None
    replace_labels : bool = False
    
    
    
    def __post_init__(self):
        y_pred = self.clf.predict(self.X_test)
        self.name_pred = self.name.replace("_clf", "_pred")
        self.y_pred = pd.Series(name = self.name_pred, data = y_pred, index = self.X_test.index)
        if self.replace_labels:
            self.y_pred.replace(label_encoding, inplace = True)
        self.feature_importance_df = (pd.DataFrame(
                                        {"features" : self.clf.feature_names_in_, 
                                         "importance" : self.clf.feature_importances_})
                                          .sort_values("importance", ascending=False)
                                     )
        
    
    def plot_ROC_curve(self):
        return plot_ROC_curve(self.clf,
                              self.X_train, self.y_train,
                              self.X_test, self.y_test,
                              per_class = False)
    

    def plot_model_metrics(self):
        return plot_model_metrics(self.clf)
    
    def plot_model_tree(self):
        return plot_model_tree(self.clf)
    
    def confusion_matrix_df(self):
        df = pd.DataFrame(self.y_pred)
        df["target"] = self.y_test
        if self.replace_labels:
            df = df.replace(label_encoding)
        return confusion_matrix_df(df, pred_col = self.name_pred)
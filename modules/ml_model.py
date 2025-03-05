from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn import svm
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import numpy as np
import pandas as pd

def train_temp_prediction_model(X,y,span):
    X_smoothed = causal_filter(X,span=span)
    df_X_final=pd.concat([X_smoothed,X],axis=1)
    X_train, X_test, y_train, y_test = train_test_split(df_X_final, y, test_size=0.2, random_state=0)
    # print(X_train)
    #est = RandomForestRegressor(n_estimators=200, max_depth=12, random_state=0)
    est = GradientBoostingRegressor(n_estimators=200, learning_rate=0.11, max_depth=4, random_state=0, loss='squared_error')
    #est=svm.SVR()
    #est=Ridge(solver = 'lsqr', alpha=1.5, tol=0.0001, random_state = 0)
    #est=ElasticNet(alpha = 0.1)
    #est = linear_model.Lasso(alpha=0.1)
    est.fit(X_train, y_train)
    mse = mean_squared_error(y_test, est.predict(X_test))
    mse_train = mean_squared_error(y_train, est.predict(X_train))
    print(f"The mean squared error (MSE) on test set: {mse:.4f}")
    print(f"The overfitting ratio is : {mse/mse_train:.4f}")
    est.fit(df_X_final,y)
    return est

def causal_filter(x,span,amplitude_gain=1):
    df_temp = x.ewm(span=span).mean()
    m = x.mean()
    df_temp = (df_temp-m)*amplitude_gain+m
    old_names=x.columns.tolist()
    new_names=[str(e) + '_smoothed' for e in old_names]
    return df_temp.rename(columns=dict(zip(old_names, new_names)))

# class Binarize:
#     def __init__(self,a_min,a_max,est=None):
#         self.a_min,self.a_max=a_min,a_max
#         self.est=est
#     def fit(self,*args):
#         self.est.fit(args)
#     def transform(self,x):
#         return np.around((np.clip(x, a_min = self.a_min, a_max = self.a_max)-self.a_min)/(self.a_max-self.a_min))*(self.a_max-self.a_min)+self.a_min
#     def fit_transform(self,x):
#         return self.transform(x)
#     def predict(self,x):
#         return self.transform(est.predict(x))

def predict_temp(model,X,span):
    X_smoothed = causal_filter(X,span=span)
    df_X_final=pd.concat([X_smoothed,X],axis=1)
    y_pred=model.predict(df_X_final)
    df_y_pred=pd.DataFrame(y_pred,index=X.index)
    #df_y_pred=causal_filter(df_y_pred,span=1,amplitude_gain=post_gain)
    return df_y_pred

def predict_heat_time(model,X,date,target_temp,span):
    X.reset_index(inplace=True)
    X.loc[X['date'] >= date, 'temp_target'] = target_temp
    X.set_index(['date'], inplace=True)
    X['temp_target_smoothed'] = causal_filter(X,span=span)['temp_target_smoothed']
    y_pred=model.predict(X)
    df_y_pred=pd.DataFrame(y_pred,index=X.index)
    #df_y_pred=causal_filter(df_y_pred,span=1,amplitude_gain=post_gain)
    df_y_pred=df_y_pred[df_y_pred['date']>=date]
    filtered_df=df_y_pred[df_y_pred['0_smoothed']>=target_temp]
    return filtered_df.index.min

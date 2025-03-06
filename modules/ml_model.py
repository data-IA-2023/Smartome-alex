from sklearn.ensemble import GradientBoostingRegressor, GradientBoostingClassifier, RandomForestRegressor, RandomForestClassifier
from sklearn.linear_model import Ridge
from sklearn import linear_model
from sklearn import svm
from sklearn.svm import SVR
from sklearn.linear_model import ElasticNet
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.signal import medfilt

import numpy as np
import pandas as pd


def train_temp_prediction_model(X,y,span,model_type='rand_forest'):
    X_smoothed = causal_filter(X,span=span)
    df_X_final=pd.concat([X_smoothed,X],axis=1)
    #df_X_final=X_smoothed
    X_train, X_test, y_train, y_test = train_test_split(df_X_final, y, test_size=0.2, random_state=0)
    # print(X_train)
    est1 = RandomForestRegressor(n_estimators=100, max_depth=24, random_state=0)
    est2 = GradientBoostingRegressor(n_estimators=50, learning_rate=0.11, max_depth=9, random_state=0, loss='squared_error')
    est3 = MLPRegressor(random_state=0, max_iter=10000, tol=0.0001, hidden_layer_sizes=(150, 75))
    est4 = KernelRidge(alpha=1.0)
    est5 = SVR(C=1.0, epsilon=0.2)
    est6 = LinearRegression()
    model_dict={'rand_forest':est1,'grad_boost':est2,'mlp':est3,'kernel':est4,'svr':est5,'lin':est6}
    est=model_dict[model_type]
    #est=svm.SVR()
    #est=Ridge(solver = 'lsqr', alpha=1.5, tol=0.0001, random_state = 0)
    #est=ElasticNet(alpha = 0.1)
    #est = linear_model.Lasso(alpha=0.1)
    est.fit(X_train, y_train)
    mse = mean_squared_error(y_test, est.predict(X_test))
    mse_train = mean_squared_error(y_train, est.predict(X_train))
    print(f"The mean squared error (MSE) on test set: {mse:.4f} (MSE < 1 is good)")
    print(f"The overfitting ratio is : {mse/mse_train:.4f} (trying not make it too high)")
    est.fit(df_X_final,y)
    return est

def causal_filter(x,span,amplitude_gain=1):
    df_temp = x.ewm(span=span).mean()
    m = x.mean()
    df_temp = (df_temp-m)*amplitude_gain+m
    old_names=x.columns.tolist()
    new_names=[str(e) + '_smoothed' for e in old_names]
    return df_temp.rename(columns=dict(zip(old_names, new_names)))


def predict_temp(model,X,span):

    X_smoothed = causal_filter(X,span=span)
    df_X_final=pd.concat([X_smoothed,X],axis=1)

    y_pred=model.predict(df_X_final)
    df_y_pred=pd.DataFrame(y_pred,index=X.index)

    return df_y_pred

def predict_heat_time(model,X,date_s,start_target_temp,target_temp,span,tol):

    X.reset_index(inplace=True)
    X.loc[X['date'] >= date_s, 'temp_target'] = target_temp
    X.loc[pd.isna(X['temp_target']),'temp_target'] = start_target_temp
    X.set_index(['date'], inplace=True)

    X_smoothed = causal_filter(X,span=span)
    df_X_final=pd.concat([X_smoothed,X],axis=1)

    y_pred=model.predict(df_X_final)

    df_y_pred=pd.DataFrame(y_pred,index=df_X_final.index)

    df_y_pred.reset_index(inplace=True)
    df_y_pred['date'] = pd.to_datetime(df_y_pred['date'])


    filtered_df=df_y_pred[(np.abs(df_y_pred[0]-target_temp)<=tol) & (df_y_pred['date']>=date_s)]
    filtered_df.set_index(['date'], inplace=True)
    print(filtered_df.index.min()-pd.to_datetime(date_s))
    df_y_pred.set_index(['date'], inplace=True)
    return df_y_pred

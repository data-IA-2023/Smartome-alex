import sys
sys.path.append('modules')
from csv_preprocessing import *
from database import *
from openmeteo_api import *
from ml_model import *
import seaborn as sns
import matplotlib.pyplot as plt




if __name__=='__main__':
    db_path='./data/database.sqlite'
    df_X,df_y=create_temp_training_set(db_path,'roland_04',"2024-01-01","2024-04-05")
    #print(df_y)



    #sns.lineplot(df_X['temp_target'])
    span=100

    est=train_temp_prediction_model(df_X,df_y,span=span)

    #y_pred=predict_temp(est,df_X,span)
    #df_y_pred=pd.Series(y_pred,index=df_y.index)

    # df_y_pred,df_temp2=causal_filter(df_y_pred,span=50,amplitude_gain=2.25,mult=4)
    #df_y_pred,df_temp_target=predict_heat_time(est,df_X_final,"2024-02-08",22,span,1.1)
    #sns.lineplot(y_pred)
    # df_y_pred=causal_filter(df_y_pred,span=30,amplitude_gain=1.25)
    # clamp=Binarize(7,22)
    # y_pred=clamp.transform(y_pred)
    #df_y_pred=pd.Series(y_pred,index=df_y.index)
    #sns.lineplot(df_X['temperature_2m'])

    #sns.lineplot(df_temp_target)
    # sns.lineplot(df_temp2)
    sns.lineplot(df_y, color='red')
    # sns.lineplot(df_y_pred)


    X_fc=create_prediction_set(db_path,'roland_04',"2024-01-01","2025-03-07")
    X_fc=
    X_fc.loc[X_fc['date']>="2025-03-01",'temp_target']=7
    y_fc=predict_temp(est,X_fc,span)
    sns.lineplot(X_fc['temp_target'])
    sns.lineplot(X_fc['temperature_2m'])
    sns.lineplot(y_fc)
    plt.show()

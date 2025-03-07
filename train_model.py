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
    span=80
    k=10
    med_coef=9

    est=train_temp_prediction_model(df_X,df_y,span=span,model_type='rand_forest',k=k,med_coef=med_coef)
    #est2=train_temp_prediction_model(df_X,df_y,span=span,model_type='rand_forest')
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
    sns.lineplot(df_X['temp_target'])
    #sns.lineplot(df_X['temperature_2m'])
    sns.lineplot(df_y, color='red')



    X_fc=create_prediction_set(db_path,'roland_04',"2024-02-01","2024-04-09")
    #print(X_fc)
    # X_fc=
    # X_fc.loc[X_fc['date']>="2025-03-01",'temp_target']=7
    df_y_pred=predict_temp(est,df_X,span,k)
    sns.lineplot(df_y_pred)
    y_fc,h_time=predict_heating_time(est,X_fc,"2024-04-06",start_target_temp=7,target_temp=22,span=span,tol=2,k=k)
    print(f'Heating time at ± 5% : {h_time}')
    sns.lineplot(X_fc['temp_target'])
    sns.lineplot(X_fc['temperature_2m'])
    sns.lineplot(y_fc)
    #print(predict_heat_time(est,X_fc,"2024-03-07",15,span))
    plt.show()

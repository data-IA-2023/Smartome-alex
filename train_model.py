import sys
sys.path.append('modules')
from csv_preprocessing import *
from database import *
from openmeteo_api import *
from ml_model import *
import seaborn as sns
import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
from mlflow.models import infer_signature
import json
import pickle
import os


db_path='./data/database.sqlite'
span=5
k=5
med_coef=9 #19 for linear and 9 for tree based
id_buil='roland_04'
tol=2
start_date="2024-01-01"
end_date="2024-04-05"
model_type='rand_forest'
start_date2="2024-02-05"
end_date2="2024-04-09"
heating_date="2024-04-06 8:00:00"

def train_model(db_path,span,k,med_coef,id_buil,tol,start_date,end_date,model_type):

    df_X,df_y=create_temp_training_set(db_path,id_buil,start_date,end_date)
    #print(df_y)
    #sns.lineplot(df_X['temp_target'])

    est=train_temp_prediction_model(df_X,df_y,span=span,model_type=model_type,k=k,med_coef=med_coef)
    #est2=train_temp_prediction_model(df_X,df_y,span=span,model_type='mlp',k=k,med_coef=19)
    #est=CombineModels([est,est2])
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
    #sns.lineplot(df_X['temp_target'])
    #sns.lineplot(df_X['temperature_2m'])
    #sns.lineplot(df_y, color='red')




    #sns.lineplot(y_fc2)
    #print(predict_heat_time(est,X_fc,"2024-03-07",15,span))
    model_path = "./temp/latest_model.pkl"
    newpath = './temp'
    if not os.path.exists(newpath):
        os.makedirs(newpath)

    pickle.dump(est, open(model_path, "wb"))
    loaded_model = pickle.load(open(model_path, "rb"))

    signature = infer_signature(df_X, df_y)

    mlflow.set_tracking_uri("sqlite:///mlruns.db")
    mlflow.sklearn.log_model(
        loaded_model,
        "sk_learn",
        serialization_format="cloudpickle",
        signature=signature,
        registered_model_name=id_buil,
        metadata={'span':span,'k':k,'med_coef':med_coef}
    )
    # with open(f"./mlruns/models/{id_buil}.json", "w") as file:
    #     json.dump({'span':span,'k':k,'med_coef':med_coef}, file)

def predict_heating_time_with_model(est,db_path,span,k,med_coef,id_buil,tol,start_date,end_date,heating_date,start_target_temp=7,target_temp=22):
    X_fc=create_prediction_set(db_path,id_buil,start_date,end_date)
    #print(X_fc)
    # X_fc=
    # X_fc.loc[X_fc['date']>="2025-03-01",'temp_target']=7
    #df_y_pred=predict_temp(est,df_X,span,k)
    # sns.lineplot(df_y_pred)
    y_fc,h_time=predict_heating_time(est,X_fc,heating_date,start_target_temp=7,target_temp=22,span=span,tol=tol,k=k)
    #y_fc2,h_time2=predict_heating_time(est2,X_fc,"2024-04-06 8:00:00",start_target_temp=7,target_temp=22,span=span,tol=tol,k=k)
    print(f'Heating time (not accurate) : {h_time}')
    #print(X_fc)
    sns.lineplot(X_fc['temp_target'])
    sns.lineplot(X_fc['temperature_2m'])
    sns.lineplot(y_fc)
    # plt.show()
    return h_time

def get_latest_model_from_mlflow(id_buil):
    con = sqlite3.connect('./mlruns.db')
    df_models=pd.read_sql_query("SELECT * FROM model_versions", con)
    run_id=df_models.iloc[df_models[df_models['name']==id_buil]['version'].idxmax()]['run_id']
    est = pickle.load(open(f'./mlruns/0/{run_id}/artifacts/sk_learn/model.pkl', 'rb'))
    return est


if __name__=='__main__':
    df_X,df_y=create_temp_training_set(db_path,id_buil,start_date,end_date)
    sns.lineplot(df_y, color='red')
    # train_model(db_path,span,k,med_coef,id_buil,tol,start_date,end_date,model_type)
    est=get_latest_model_from_mlflow(id_buil)
    predict_heating_time_with_model(est,db_path,span,k,med_coef,id_buil,tol,start_date2,end_date2,heating_date,start_target_temp=7,target_temp=22)
    plt.show()

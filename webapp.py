from fastapi import FastAPI, Request, Form, Depends, Response, HTTPException, Cookie, Header, status
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse
from pydantic import BaseModel

import sys
sys.path.append('modules')
from csv_preprocessing import *
from database import *
from openmeteo_api import *
from ml_model import *
from train_model import *
import pandas as db
import numpy as np

app = FastAPI()
#app.mount("/static", StaticFiles(directory="static"), name="static")
db_path='./data/database.sqlite'
span=100
k=5
med_coef=9 #19 for linear and 9 for tree based
#id_buil='roland_04'
tol=2


class TemperatureData(BaseModel):
    id_buil : list[str]
    temp_int : list[float]
    date : list[str]

class BuildingData(BaseModel):
    id_buil : list[str]
    DPE : list[float]
    surface : list[float]
    city : list[str]

class GetWeather(BaseModel):
    id_buil : list[str]
    start_date : list[str]
    end_date : list[str]

class TrainModel(BaseModel):
    id_buil : list[str]
    start_date : list[str]
    end_date : list[str]
    model_type : list[str]
    preprocessing_med_coef : list[int]
    preprocessing_k : list[int]
    preprocessing_span : list[int]

class PredictHeatingTimes(BaseModel):
    id_buil : list[str]
    start_date : list[str]
    end_date : list[str]
    heating_date : list[str]
    base_temp : list[float]
    target_temp : list[float]
    tol : list[float]

@app.get("/")
async def root():
    return {"message": "Hello World"}



@app.post("/upload_data/temperature_data")
async def upload_data_temp(data: TemperatureData):
    d=vars(data)
    measurements_df=pd.DataFrame(d,index=['date','id_buil'])
    measurements_to_db
    return {"message": "Seems to work"}

@app.post("/upload_data/new_building")
async def upload_data_building(data: BuildingData):
    d=vars(data)
    building_df=pd.DataFrame(d,index=['id_buil'])
    building_to_db(building_df,file_path)
    return {"message": "Seems to work"}

@app.post("/upload_data/update_weather")
async def upload_data_weather(data: GetWeather):
    d=vars(data)
    df_buil=pd.read_sql_query("SELECT * FROM buildings", con)
    for i in range(len(d['id_buil'])):
        coords=get_coords(df_buil[df_buil['id_buil']==d['id_buil'][i]]['city'][0])
        df_wea=get_weather_data(coords['latitude'],coords['longitude'],d['start_date'][i],d['end_date'][i])
        weather_to_db(df_wea,db_path)
    return {"message": "Seems to work"}

@app.post("/train_model")
async def train_models(data: TrainModel):
    d=vars(data)
    for i in range(len(d['id_buil'])):
        train_model(db_path,d['span'][i],d['k'][i],d['med_coef'][i],d['id_buil'][i],d['tol'][i],d['start_date'][i],d['end_date'][i],d['model_type'][i])
    return {"message": "Seems to work"}

@app.post("/predict_heating_times")
async def predict_heating_times(data: PredictHeatingTimes):
    d=vars(data)
    L=[]
    for i in range(len(d['id_buil'])):
        est=get_latest_model_from_mlflow(id_buil)
        L.append(predict_heating_time_with_model(est,db_path,span=span,k=k,med_coef=med_coef,id_buil=d['id_buil'][i],tol=d['tol'][i],start_date=d['start_date'][i],end_date=d['end_date'][i],heating_date=d['heating_date'][i],start_target_temp=d['base_temp'][i],target_temp=d['target_temp'][i]))
    return {'heating_times':L}

import sys
sys.path.append('modules')
from csv_preprocessing import *
from database import *
from openmeteo_api import *

if __name__=='__main__':
    db_path='./data/database.sqlite'
    create_db(db_path)
    building_df=pd.DataFrame({'id_buil':['roland_04'],'DPE':[None],'surface':[None],'city':['Luz-Saint-Sauveur']}).set_index(['id_buil'])
    building_to_db(building_df,db_path)

    weather_df=create_df('Luz-Saint-Sauveur',"2024-01-01","2024-12-31")
    weather_to_db(weather_df,db_path)

    df_mes=read_val_roland('./data/Val de Roland/Export_Data_Val de Roland_2024-05-09_15h49.csv')
    measurements_to_db(df_mes,db_path)



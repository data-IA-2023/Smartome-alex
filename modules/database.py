import sqlite3
import pandas as pd
import numpy as np

def create_db(file_path):
    con = sqlite3.connect(file_path)

    cur = con.cursor()
    cur.execute("CREATE TABLE IF NOT EXISTS buildings(id_buil, DPE, surface, city)")
    cur.execute("CREATE TABLE IF NOT EXISTS measurements(date,id_buil,temp_int, temp_target)")
    cur.execute("CREATE TABLE IF NOT EXISTS weather(city, date, temperature_2m, relative_humidity_2m, precipitation, cloud_cover, wind_speed_10m)")

    con.close()

def weather_to_db(weather_df,file_path):
    con = sqlite3.connect(file_path)
    weather_df.to_sql(name='weather', con=con, if_exists='append')
    con.close()

def measurements_to_db(measurements_df,file_path):
    measurements_df.to_sql(name='measurements', con=con, if_exists='append')
    con.close()

def building_to_db(building_df,file_path):
    con = sqlite3.connect(file_path)
    building_df.to_sql(name='buildings', con=con, if_exists='append')
    con.close()

def join_and_interpolate_df(df1,df2):
    df1=df1.tz_localize(None)
    df2=df2.tz_localize(None)
    df = pd.merge(df1, df2, left_index=True, right_index=True, how='outer')
    df = df.interpolate(method='linear').resample('h').mean().sort_index()
    return df

def create_temp_training_set(file_path,building_id,start_date,end_date):
    con = sqlite3.connect(file_path)

    # df_buil = pd.read_sql_table('buildings', con=con)
    # df_mea = pd.read_sql_table('measurements', con=con)
    # df_wea = pd.read_sql_table('weather', con=con)
    df_buil=pd.read_sql_query("SELECT * FROM buildings", con)
    df_mea=pd.read_sql_query("SELECT * FROM measurements", con)
    df_wea=pd.read_sql_query("SELECT * FROM weather", con)

    con.close()

    df_mea['date'] = pd.to_datetime(df_mea['date'])
    df_wea['date'] = pd.to_datetime(df_wea['date'])
    city=df_buil[df_buil['id_buil'] == building_id]['city'].loc[0]

    mask_mea = (df_mea['date'] >= start_date) & (df_mea['date'] <= end_date) & (df_mea['id_buil'] == building_id)
    mask_wea = (df_wea['date'] >= start_date) & (df_wea['date'] <= end_date) & (df_wea['city'] == city)
    df_mea=df_mea.loc[mask_mea]
    df_wea=df_wea.loc[mask_wea]
    df_mea.set_index(['date'], inplace=True)
    df_wea.set_index(['date'], inplace=True)
    df_mea = df_mea.drop('id_buil', axis=1)
    df_wea = df_wea.drop('city', axis=1)
    df = join_and_interpolate_df(df_mea,df_wea)
    
    #df_shift = pd.DataFrame(np.random.randint(1,100,size=(df.shape[0]-100, 1)), columns=['shift'], index=df[100:].index)
    #df_shift = pd.concat([pd.DataFrame([i+1 for i in range(100)], columns=['shift'], index=df[:100].index),df_shift],axis=0)
    
    #print(df_shift)

    #old_names=df.columns.tolist()
    #new_names=[e + '_shifted' for e in old_names]
    #training_df=df['temp_int'].shift(nb_hours)[nb_hours:]
    #training_df=df['temp_target'].shift(nb_hours)[nb_hours:]
    #training_df=pd.concat([df[nb_hours:].drop(columns=['temp_int']),training_df],axis=1)
    training_df=df[['temp_target','temperature_2m','cloud_cover']] # only useful features, other features would create some kind of positional encoding wich would lead to overfitting
    
    return training_df,df['temp_int']
    
def create_prediction_set(file_path,building_id,start_date,end_date):
    import openmeteo_api as om

    con = sqlite3.connect(file_path)
    df_buil=pd.read_sql_query("SELECT * FROM buildings", con)
    city=df_buil[df_buil['id_buil'] == building_id]['city'].loc[0]

    coords=om.get_coords(city)
    print(coords)
    df_wea=om.get_weather_forecast_data(coords['latitude'],coords['longitude'],start_date,end_date)



    df_mea=pd.read_sql_query("SELECT * FROM measurements", con)
    df_wea2=pd.read_sql_query("SELECT * FROM weather", con)
    #print(df_wea)
    df_mea['date'] = pd.to_datetime(df_mea['date'])
    df_wea['date'] = pd.to_datetime(df_wea['date'])
    df_wea2['date'] = pd.to_datetime(df_wea['date'])

    df_mea.set_index(['date'], inplace=True)
    df_wea.set_index(['date'], inplace=True)
    df_wea2.set_index(['date'], inplace=True)

    df_wea=df_wea.combine_first(df_wea2)

    con.close()

    df_wea.reset_index(inplace=True)
    df_mea.reset_index(inplace=True)


    mask_mea = (df_mea['date'] >= start_date) & (df_mea['date'] <= end_date) & (df_mea['id_buil'] == building_id)
    mask_wea = (df_wea['date'] >= start_date) & (df_wea['date'] <= end_date) & (df_wea['city'] == city)
    df_mea=df_mea.loc[mask_mea]
    df_wea=df_wea.loc[mask_wea]
    df_mea.set_index(['date'], inplace=True)
    df_wea.set_index(['date'], inplace=True)
    df_mea = df_mea.drop('id_buil', axis=1)
    df_wea = df_wea.drop('city', axis=1)
    df = join_and_interpolate_df(df_mea,df_wea)

    final_df=df[['temp_target','temperature_2m','cloud_cover']]
    return final_df

if __name__=='__main__':
    db_path='./data/database.sqlite'
    # import openmeteo_api as om
    # create_db('./test.sqlite')
    # weather_df=om.create_df('Tours',"2024-01-01","2024-12-31")
    # weather_to_db(weather_df,'./test.sqlite')
    print(create_prediction_set(db_path,'roland_04',"2024-03-07","2025-03-07"))

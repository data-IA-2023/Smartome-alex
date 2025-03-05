import pandas as pd
import numpy as np
# import datetime
# import time
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib.dates as mdates

def frstr2float(x)->float:
    y=str(x)
    a=y.split(",")
    if len(a)==1 : return float(x)
    res=float(a[0])
    res+=float(f'0.{a[1]}')
    return res

def plot_date_serie(fig,ax,s:pd.Series,log:bool=False,title:str='')->None:
    # s.index = s.index.map(date2stamp)
    s=s.apply(str2float)
    sns.lineplot(data=s,ax=ax)
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))

    # Set tick interval (e.g., daily)
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=1))

    plt.xticks(rotation=45)
    if log : plt.yscale('log')
    plt.title(title)

def read_val_roland(file_path='../data/Val de Roland/Export_Data_Val de Roland_2024-05-09_15h49.csv',vis=False):
    df2024val=pd.read_csv(file_path,sep=';')
    df2024val['Date']=pd.to_datetime(df2024val['Date'])
    df2024val_pivot=pd.pivot_table(df2024val,values='Valeur',index='Date',columns=['Objet','Équipement','Commande'], aggfunc="sum", fill_value=np.nan)
    df_temp1=df2024val_pivot['LOGEMENT04 - A302 (piloté)']['LOGEMENT04_TEMP']['Température'].dropna()
    df_temp2=df2024val_pivot['LOGEMENT04 - A302 (piloté)']['Thermostat Logement 4']['Consigne'].dropna().astype(float).reindex(df_temp1.index, method='pad')
    df_temp3=df2024val_pivot['LOGEMENT01 - B001 (piloté)']['LOGEMENT01_4PM']['0 - CHAUFFAGE Etat '].dropna()
    df_temp4=df2024val_pivot['LOGEMENT01 - B001 (piloté)']['LOGEMENT01_TEMP']['Température'].dropna()

    df_temp2.fillna(value=22,inplace=True)

    df_temp1=df_temp1.str.replace(',', '.').astype(float)

    #df_diff = df_temp2-df_temp1


    fig, ax = plt.subplots()
    # print(df_temp1)
    # print(df_temp2)
    # plot_date_serie(fig, ax, df_temp3,False,'LOGEMENT01_4 état chauffage')
    # plot_date_serie(fig, ax, df_temp4,False,'LOGEMENT01_4 température en °C')

    df_bat=pd.DataFrame({'id_buil':['roland_04']*df_temp1.shape[0]},index=df_temp1.index)

    df_concat=pd.concat([df_temp1,df_bat,df_temp2,],axis=1).reset_index()
    df_concat.rename(columns={'Date': 'date','Température': 'temp_int', 'Consigne': 'temp_target'}, inplace=True)
    df_concat.set_index(['date','id_buil'], inplace=True)
    # print(df_concat)
    if vis:
        plot_date_serie(fig, ax, df_temp1,False,'LOGEMENT04_TEMP / Température en °C')
        plot_date_serie(fig, ax, df_temp2,False,'LOGEMENT04_TEMP / Température consigne en °C')
        # plot_date_serie(fig, ax, df_diff,False,'LOGEMENT04_TEMP / Erreur en °C')
        plt.show()
    return df_concat

if __name__=='__main__':
    print(read_val_roland())


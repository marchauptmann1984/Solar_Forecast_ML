import appdaemon.plugins.hass.hassapi as hass
from datetime import datetime, timedelta
import requests
import pandas as pd
import numpy as np
import calendar
import pytz
from scipy import stats
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import SGDRegressor
from joblib import load
from datetime import datetime, timedelta

# add other imports later

class Forecast(hass.Hass):
    def initialize(self): 
        # load machine learning models and scaler
        sgd_reg_best=load('/config/apps/sol_pred_mod_sgd.gz')
        scale=load('/config/apps/weather_scl_sgd.gz')
        # set API parameters, all times in UTC
        parameters = {
            "latitude": 51.3225,
            "longitude": 4.9447,
            "tilt": 60,
            "azimuth": 77,
            "hourly": ['soil_temperature_0cm', 'snow_depth', 'global_tilted_irradiance', 'is_day'],
        }
        parameters_2 = {
           "latitude": 51.3225,
            "longitude": 4.9447,
            "tilt": 60,
            "azimuth": -103,
            "hourly": ['global_tilted_irradiance'],
        }              
        # define start time and call set_sensor
        now = datetime.now()
        #now=datetime(year=2024, month=3, day=15, hour=23, minute=0)
        #print(now)       
        self.run_every(self.set_sensor,now, 600, pars=parameters, pars2=parameters_2, scaler=scale, model=sgd_reg_best) 
    def set_sensor(self, kwargs): 
        # call open meteo API and get weather forecast
        response = requests.get("https://api.open-meteo.com/v1/forecast",params=kwargs["pars"])
        weather_fc_int=pd.DataFrame(response.json()['hourly'])
        # call open meteo API and get weather forecast again for 2nd roof
        response = requests.get("https://api.open-meteo.com/v1/forecast",params=kwargs["pars2"])
        weather_fc_int_roof2=pd.DataFrame(response.json()['hourly'])
        # combine roofs
        weather_fc_int.global_tilted_irradiance=weather_fc_int.global_tilted_irradiance+weather_fc_int_roof2.global_tilted_irradiance
        # format weather
        weather_fc_int['time_']=weather_fc_int['time']
        weather_fc_int['time'] = pd.to_datetime(weather_fc_int['time'])
        weather_fc_int['time_']=weather_fc_int['time']
        weather_fc_int['time'] = pd.to_datetime(weather_fc_int['time'])
        weather_fc_int['time']=(weather_fc_int.time.dt.year-2024)*(365 + calendar.isleap(datetime.now().year))*24+weather_fc_int.time.dt.day_of_year*24+weather_fc_int.time.dt.hour
        weather_fc_int.set_index('time', inplace=True)
        weather_fc_int.columns=['soil_temperature_0_to_7cm', 'snow_depth', 'global_tilted_irradiance','is_day', 'time_']
        weather_fc_int_1=weather_fc_int.loc[weather_fc_int.index[:-1]]
        weather_fc_int_2=weather_fc_int.loc[weather_fc_int.index[:-1]+1]
        weather_fc_int_2.index=weather_fc_int.index[:-1]
        weather_fc_int=weather_fc_int_2
        weather_fc_int.time_=weather_fc_int_1.time_
        weather_fc_int.soil_temperature_0_to_7cm=(weather_fc_int_1.soil_temperature_0_to_7cm+weather_fc_int_2.soil_temperature_0_to_7cm)/2
        weather_fc_int.snow_depth=(weather_fc_int_1.snow_depth+weather_fc_int_2.snow_depth)/2
        # merge and prep
        weather_fc_int.drop('time_', axis=1,inplace=True)
        data_fc_comb=weather_fc_int
        data_fc_comb = data_fc_comb.astype(float)
        scale=kwargs["scaler"]
        data_fc_comb_sc=pd.DataFrame(scale.transform(data_fc_comb.drop(['is_day'], axis=1)),columns=data_fc_comb.drop(['is_day'], axis=1).columns,index=data_fc_comb.index)        
        # prediction
        sgd_reg_best=kwargs["model"]
        predictions=sgd_reg_best.predict(data_fc_comb_sc[data_fc_comb.is_day>0])
        predictions[predictions<0]=0 # remove negative
        predictions[predictions>4.5]=4.5 # remove values larger than what inverter can do
        df_fc=pd.DataFrame(0, columns=['Time','ForeCast'], index=data_fc_comb.index)
        df_fc['ForeCast'] = df_fc['ForeCast'].astype('float')
        df_fc.loc[data_fc_comb.is_day>0, "ForeCast"]=predictions
        df_fc.Time=weather_fc_int_1.time_    
        #print(df_fc.Time.values)
        # integrate
        df_fc['Time'] = df_fc['Time'].astype('datetime64[ns]')
        df_fc_d=df_fc.resample('D', on='Time').sum()
        # forecast for current hour
        ind_now=datetime.now().hour-1
        # forecast for next 24 hours and conversion to battery threshold
        # get the min threshold
        batt_thresh_min=float(self.entities.input_number.batt_thresh_min.state)
        batt_thres=max(min(100-df_fc.ForeCast.values[ind_now:(ind_now+23)].sum()/14.2*100*2/3,95),batt_thresh_min) # assume 1/3 of it is consumed straightaway
        #print(batt_thres)
        # export
        #print(datetime.now().minute)
        #print((df_fc.ForeCast.values[ind_now+1]-df_fc.ForeCast.values[ind_now])*datetime.now().minute/60+df_fc.ForeCast.values[ind_now])
        self.set_state("sensor.solar_forecast_hourly",state=np.round(df_fc.ForeCast.values[ind_now],1),attributes={"friendly_name":"Hourly Solar Forecast", "unit_of_measurement": "kWh", "PeakTimes":list(df_fc.Time.dt.strftime('%Y-%m-%dT%H:%M:%S%z+00:00').values), "PeakHeights": list(np.round(df_fc.ForeCast.values,1))})
        self.set_state("sensor.solar_forecast_daily",state=np.round(df_fc_d.ForeCast.values[0],1),attributes={"friendly_name":"Daily Solar Forecast", "unit_of_measurement": "kWh", "PeakTimes":list(df_fc_d.index.strftime('%Y-%m-%dT%H:%M:%S%z+00:00').values), "PeakHeights": list(np.round(df_fc_d.ForeCast.values,1))})
        self.set_state("sensor.solar_forecast_next12h",state=np.round(df_fc.ForeCast.values[ind_now:(ind_now+23)].sum(),1),attributes={"friendly_name":"Solar Forecast Next 24h", "unit_of_measurement": "kWh"})        
        self.set_state("sensor.solar_forecast_nexthour",state=np.round((df_fc.ForeCast.values[ind_now+1]-df_fc.ForeCast.values[ind_now])*datetime.now().minute/60+df_fc.ForeCast.values[ind_now],1),attributes={"friendly_name":"Solar Forecast Next Hour", "unit_of_measurement": "kWh"})
        self.set_state("input_number.Battery_management_threshold",state=np.round(batt_thres,0),attributes={"friendly_name":"Battery management threshold", "unit_of_measurement": "%"})

        

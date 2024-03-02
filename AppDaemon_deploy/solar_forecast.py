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
from sklearn.ensemble import RandomForestRegressor
from joblib import load
from pysolar.solar import *
from datetime import datetime, timedelta

# add other imports later

class Forecast(hass.Hass):
    def initialize(self): 
        # load machine learning models and scaler
        forest_reg_best=load('/config/apps/sol_pred_mod.gz')
        scale=load('/config/apps/weather_scl.gz')
        # set API parameters, all times in UTC
        parameters = {
            "latitude": 51.3225,    
            "longitude": 4.9447,            
            "hourly": ['direct_normal_irradiance', 'diffuse_radiation', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high' , 'soil_temperature_0cm', 'snow_depth'],
            }                  
        # define start time and call set_sensor
        now = datetime.now()       
        self.run_every(self.set_sensor,now, 600, pars=parameters, scaler=scale, model=forest_reg_best) 
    def set_sensor(self, kwargs): 
        # call open meteo API and get weather forecast
        response = requests.get("https://api.open-meteo.com/v1/forecast",params=kwargs["pars"])
        weather_fc_int=pd.DataFrame(response.json()['hourly'])
        # format weather
        weather_fc_int['time_']=weather_fc_int['time']
        weather_fc_int['time'] = pd.to_datetime(weather_fc_int['time'])
        weather_fc_int['time_']=weather_fc_int['time']
        weather_fc_int['time'] = pd.to_datetime(weather_fc_int['time'])
        weather_fc_int['time']=(weather_fc_int.time.dt.year-2023)*(365 + calendar.isleap(datetime.now().year))*24+weather_fc_int.time.dt.day_of_year*24+weather_fc_int.time.dt.hour
        weather_fc_int.set_index('time', inplace=True)
        weather_fc_int.columns=['direct_normal_irradiance', 'diffuse_radiation', 'cloud_cover_low',
       'cloud_cover_mid', 'cloud_cover_high', 'soil_temperature_0_to_7cm',
       'snow_depth', 'time_']
        weather_fc_int_1=weather_fc_int.loc[weather_fc_int.index[:-1]]
        weather_fc_int_2=weather_fc_int.loc[weather_fc_int.index[:-1]+1]
        weather_fc_int_2.index=weather_fc_int.index[:-1]
        weather_fc_int=weather_fc_int_2
        weather_fc_int.time_=weather_fc_int_1.time_
        weather_fc_int.cloud_cover_high=(weather_fc_int_1.cloud_cover_high+weather_fc_int_2.cloud_cover_high)/2
        weather_fc_int.cloud_cover_low=(weather_fc_int_1.cloud_cover_low+weather_fc_int_2.cloud_cover_low)/2
        weather_fc_int.cloud_cover_mid=(weather_fc_int_1.cloud_cover_mid+weather_fc_int_2.cloud_cover_mid)/2
        weather_fc_int.soil_temperature_0_to_7cm=(weather_fc_int_1.soil_temperature_0_to_7cm+weather_fc_int_2.soil_temperature_0_to_7cm)/2
        weather_fc_int.snow_depth=(weather_fc_int_1.snow_depth+weather_fc_int_2.snow_depth)/2
        # solar prediction
        suntimeutc=pd.to_datetime(weather_fc_int.time_).dt.tz_localize('UTC').dt.tz_convert('UTC').dt.strftime('%Y-%m-%dT%H:%M') # need to convert to utc
        datelist=list(map(lambda x:datetime.strptime(x,'%Y-%m-%dT%H:%M')+timedelta(minutes=30), suntimeutc)) # half hour later for our interval
        parameters=kwargs["pars"]
        def sun_utc(x,par='azimuth'):
            utc_dt=pytz.utc.localize(x)
            if par=='azimuth':
                out=get_azimuth(parameters['latitude'],parameters['longitude'],utc_dt)
            else:
                out=get_altitude(parameters['latitude'],parameters['longitude'],utc_dt)             
            return out
        azimuth_fc=list(map(lambda x: sun_utc(x,par='azimuth'), datelist))
        altitude_fc=list(map(lambda x: sun_utc(x,par='altitude'), datelist))
        sun_fc_int=pd.DataFrame({'time': weather_fc_int.time_,
                         'azimuth': azimuth_fc,
                         'altitude': altitude_fc})
        sun_fc_int['time'] = pd.to_datetime(sun_fc_int['time'])
        sun_fc_int['time']=(sun_fc_int.time.dt.year-2023)*24*(365 + calendar.isleap(datetime.now().year))+sun_fc_int.time.dt.day_of_year*24+sun_fc_int.time.dt.hour
        sun_fc_int.set_index('time', inplace=True)
        # calculate angle of incidence etc.
        a_m=45 # altitude panel
        a_s=sun_fc_int.altitude.values # altitude sun
        A_m_1=77 # Azimuth module 1
        A_m_2=257 # Azimuth module 2
        A_s=sun_fc_int.azimuth.values # Azimuth sun
        sun_fc_int=sun_fc_int.assign(AOI_1=np.arccos(np.cos(a_m/180*np.pi)*np.cos(a_s/180*np.pi)*np.cos((A_m_1-A_s)/180*np.pi)+np.sin(a_m/180*np.pi)*np.sin(a_s/180*np.pi))*180/np.pi)
        sun_fc_int=sun_fc_int.assign(AOI_2=np.arccos(np.cos(a_m/180*np.pi)*np.cos(a_s/180*np.pi)*np.cos((A_m_2-A_s)/180*np.pi)+np.sin(a_m/180*np.pi)*np.sin(a_s/180*np.pi))*180/np.pi)
        # merge and prep
        weather_fc_int.drop('time_', axis=1,inplace=True)
        data_fc_comb=sun_fc_int.join(weather_fc_int)
        data_fc_comb = data_fc_comb.astype(float)
        # calculate the direct global irradiation and diffuse irradiation per plane
        G_direct_1 = data_fc_comb.direct_normal_irradiance*np.cos(data_fc_comb.AOI_1*np.pi/180)
        G_direct_2 = data_fc_comb.direct_normal_irradiance*np.cos(data_fc_comb.AOI_2*np.pi/180)
        G_diff = data_fc_comb.diffuse_radiation*(1+np.cos(45*np.pi/180))/2
        data_fc_comb=data_fc_comb.assign(G_direct=G_direct_1+G_direct_2)
        data_fc_comb=data_fc_comb.assign(G_diff=G_diff)
        data_fc_comb=data_fc_comb.drop(['diffuse_radiation', 'direct_normal_irradiance', 'azimuth', 'AOI_1', 'AOI_2'], axis=1)
        scale=kwargs["scaler"]
        data_fc_comb_sc=pd.DataFrame(scale.transform(data_fc_comb.drop(['altitude'], axis=1)),columns=data_fc_comb.drop(['altitude'], axis=1).columns,index=data_fc_comb.index)        
        # prediction
        forest_reg_best=kwargs["model"]
        predictions=forest_reg_best.predict(data_fc_comb_sc[data_fc_comb.altitude>0])
        df_fc=pd.DataFrame(0, columns=['Time','ForeCast'], index=data_fc_comb.index)
        df_fc.ForeCast[data_fc_comb.altitude>0]=predictions
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
        batt_thres=max(min(100-df_fc.ForeCast.values[ind_now:(ind_now+23)].sum()/14.2*100,95),batt_thresh_min)
        #print(batt_thres)
        # export
        self.set_state("sensor.solar_forecast_hourly",state=np.round(df_fc.ForeCast.values[ind_now],1),attributes={"friendly_name":"Hourly Solar Forecast", "unit_of_measurement": "kWh", "PeakTimes":list(df_fc.Time.dt.strftime('%Y-%m-%dT%H:%M:%S%z+00:00').values), "PeakHeights": list(np.round(df_fc.ForeCast.values,1))})
        self.set_state("sensor.solar_forecast_daily",state=np.round(df_fc_d.ForeCast.values[0],1),attributes={"friendly_name":"Daily Solar Forecast", "unit_of_measurement": "kWh", "PeakTimes":list(df_fc_d.index.strftime('%Y-%m-%dT%H:%M:%S%z+00:00').values), "PeakHeights": list(np.round(df_fc_d.ForeCast.values,1))})
        self.set_state("sensor.solar_forecast_next12h",state=np.round(df_fc.ForeCast.values[ind_now:(ind_now+23)].sum(),1),attributes={"friendly_name":"Solar Forecast Next 24h", "unit_of_measurement": "kWh"})
        self.set_state("input_number.Battery_management_threshold",state=np.round(batt_thres,0),attributes={"friendly_name":"Battery management threshold", "unit_of_measurement": "%"})

        

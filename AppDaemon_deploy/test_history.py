import appdaemon.plugins.hass.hassapi as hass
from datetime import datetime, timedelta, time
import pandas as pd
import pytz
from sklearn.metrics import mean_squared_error
import numpy as np
import requests
import calendar

class HistoryExample(hass.Hass):

    def initialize(self):
        # set API parameters
        dayz=14
        parameters = {
            "past_days": dayz,
            "latitude": 51.3225,
            "longitude": 4.9447,
            "tilt": 60,
            "azimuth": 77,
            "hourly": ['relative_humidity_2m', 'precipitation', 'cloud_cover_low', 'cloud_cover_mid', 'cloud_cover_high' , 'soil_temperature_0cm', 'snow_depth', 'global_tilted_irradiance'],
        }
        parameters_2 = {
                "past_days": dayz,
                "latitude": 51.3225,
                "longitude": 4.9447,
                "tilt": 60,
                "azimuth": -103,
                "hourly": ['global_tilted_irradiance'],
        }    
        # time zone
        tz_name = self.args.get("tz", "Europe/Amsterdam") 
        # change to your HA timezone 
        tz = pytz.timezone(tz_name) 
        now = datetime.now(tz)
        print(now)
        # Build today's 23:59 in local time
        run_time = tz.localize(datetime.combine(now.date()-timedelta(days=1), time(22, 30)))        
        print(run_time)
        self.run_daily(self.calc_mse, run_time, tzinfo=tz, daysback=dayz, tzname=tz_name, pars=parameters, pars2=parameters_2, url_i="https://ehklxvz47g.execute-api.eu-north-1.amazonaws.com/prod/predict", url_t="https://ehklxvz47g.execute-api.eu-north-1.amazonaws.com/prod/retrain", key="my-home-automation-secret")
        #self.run_every(self.calc_mse, now, 120, daysback=dayz, tzname=tz_name, pars=parameters, pars2=parameters_2, url_i="https://ehklxvz47g.execute-api.eu-north-1.amazonaws.com/prod/predict", url_t="https://ehklxvz47g.execute-api.eu-north-1.amazonaws.com/prod/retrain", key="my-home-automation-secret")
        
    def days_in_year(self, year=datetime.now().year):
        return 365 + calendar.isleap(year)
    
    def get_forecast(self, para, para_2):
        response = requests.get("https://api.open-meteo.com/v1/forecast",params=para)
        weather_fc_int=pd.DataFrame(response.json()['hourly'])
        response_roof2 = requests.get("https://api.open-meteo.com/v1/forecast",params=para_2)
        weather_fc_int_roof2=pd.DataFrame(response_roof2.json()['hourly'])
        weather_fc_int.global_tilted_irradiance=weather_fc_int.global_tilted_irradiance+weather_fc_int_roof2.global_tilted_irradiance
        weather_fc_int['is_day']=weather_fc_int.global_tilted_irradiance>0
        weather_fc_int['time_']=weather_fc_int['time']
        weather_fc_int['time'] = pd.to_datetime(weather_fc_int['time'])
        #weather_fc_int['checktime']=weather_fc_int['time']
        weather_fc_int['time']=(weather_fc_int.time.dt.year-2024)*self.days_in_year()*24+weather_fc_int.time.dt.day_of_year*24+weather_fc_int.time.dt.hour
        weather_fc_int.set_index('time', inplace=True)
        weather_fc_int.columns=['relative_humidity_2m', 'precipitation', 'cloud_cover_low',
            'cloud_cover_mid', 'cloud_cover_high', 'soil_temperature_0_to_7cm', 'snow_depth', 'global_tilted_irradiance',
            'is_day', 'time_']        
        # make 2 versions at begin and end of interval
        weather_fc_int_1=weather_fc_int.loc[weather_fc_int.index[:-1]]
        weather_fc_int_2=weather_fc_int.loc[weather_fc_int.index[:-1]+1]
        weather_fc_int_2.index=weather_fc_int.index[:-1]
        # precipitation and irradiance is preceeding hour, others are instant --> average over interval
        weather_fc_int=weather_fc_int_2
        weather_fc_int.time_=weather_fc_int_1.time_
        weather_fc_int.relative_humidity_2m=(weather_fc_int_1.relative_humidity_2m+weather_fc_int_2.relative_humidity_2m)/2
        weather_fc_int.cloud_cover_high=(weather_fc_int_1.cloud_cover_high+weather_fc_int_2.cloud_cover_high)/2
        weather_fc_int.cloud_cover_low=(weather_fc_int_1.cloud_cover_low+weather_fc_int_2.cloud_cover_low)/2
        weather_fc_int.cloud_cover_mid=(weather_fc_int_1.cloud_cover_mid+weather_fc_int_2.cloud_cover_mid)/2
        weather_fc_int.soil_temperature_0_to_7cm=(weather_fc_int_1.soil_temperature_0_to_7cm+weather_fc_int_2.soil_temperature_0_to_7cm)/2
        weather_fc_int.snow_depth=(weather_fc_int_1.snow_depth+weather_fc_int_2.snow_depth)/2
        weather_fc_int.loc[weather_fc_int.snow_depth<0, 'snow_depth']=0
        weather_fc_int.loc[np.isnan(weather_fc_int.snow_depth), 'snow_depth']=0
        Time=pd.to_datetime(weather_fc_int_1.time_)
        weather_fc_int.drop('time_', axis=1,inplace=True)
        data_fc_comb=weather_fc_int
        data_fc_comb = data_fc_comb.astype(float)
        return data_fc_comb, Time
    
    def calc_mse(self, kwargs):
        days_back=kwargs["daysback"]
        # timezone-aware "now" from AppDaemon
        tz_name = kwargs["tzname"] 
        #print(tz_name)
        power = kwargs.get("entity_id", "sensor.daily_power_generated")
        forecast = kwargs.get("entity_id", "sensor.solar_forecast_hourly")
        now = self.get_now().astimezone(pytz.UTC)
        print(now)
        # start of local day (00:00:00)
        start_of_window = (now-timedelta(days=days_back)).replace(hour=0, minute=0, second=0, microsecond=0)

        # fetch history starting one day before midnight to ensure we have a value
        # *before* the first boundary (this fixes the "first entry" problem)
        history_start = start_of_window - timedelta(hours=1)

        history = self.get_history(entity_id=power, start_time=history_start, end_time=now)
        
        if not history or len(history) == 0:
            self.log(f"No history returned for {power}")
            return

        entries = history[0]
        if len(entries) == 0:
            self.log(f"No history entries for {power}")
            return

        # turn into DataFrame for easier timestamp handling
        df = pd.DataFrame(entries)
        df["last_changed"] = pd.to_datetime(df["last_changed"], utc=True) # to utc
        df["state"] = pd.to_numeric(df["state"], errors="coerce")
        df = df.dropna(subset=["state"]).sort_values("last_changed").reset_index(drop=True)

        if df.empty:
            self.log("No numeric states in history.")
            return

        # helper: find last-known state <= timestamp (walk entries backwards)
        records = df.to_dict("records")
        def state_at(ts):
            # ts: a timezone-aware datetime
            for r in reversed(records):
                if r["last_changed"] <= ts:
                    return r["state"]
            return None

        # build hourly buckets from 00:00 up to now (include current partial hour)
        hour_starts = []
        t = start_of_window
        while t <= now:
            hour_starts.append(t)
            t = t + timedelta(hours=1)
        #print(hour_starts)
        hours = []
        energies = []
        days = []
        for h_start in hour_starts:
            h_end = h_start + timedelta(hours=1)
            if h_end > now:
                h_end = now  # current partial hour -- compute up to now

            v_start = state_at(h_start)
            v_end = state_at(h_end)

            # default behavior for missing values:
            # - If both missing -> treat as 0.0 (no data / no change)
            # - If one side missing -> treat as 0.0 (safe fallback)
            # You can change this behavior if you prefer NaN for "unknown".
            if v_start is None and v_end is None:
                energy = 0.0
            elif v_start is None and v_end is not None:
                # we don't know previous state, assume no consumption in the hour before the first recorded value
                energy = 0.0
            elif v_start is not None and v_end is None:
                energy = 0.0
            else:
                # normal case
                energy = v_end - v_start
                if energy < 0:
                    # meter reset / rollover detected
                    # default: assume a reset happened and take v_end as the energy since reset.
                    # alternative: set energy = 0.0 or energy = None depending on your preference.
                    self.log(f"Detected negative delta for hour {h_start.isoformat()} ({v_start} -> {v_end}). "
                             "Assuming meter reset; using v_end as energy for that hour.")
                    energy = float(v_end)

            hours.append(h_start.hour)
            days.append(h_start.day)
            energies.append(energy)    
        # get forecast
        data_fc_comb, Time=self.get_forecast(kwargs["pars"], kwargs["pars2"])        
        # prediction
        X_in = data_fc_comb[data_fc_comb.is_day>0].iloc[:, 0:8].values.tolist()
        url = kwargs["url_i"]
        headers = {
            "Content-Type": "application/json",
            "x-api-key": kwargs["key"]
        }       
        payload = {
            "features": X_in
        }
        response = requests.post(url, json=payload, headers=headers)
        try:    
            #print(response.json())
            resp=response.json()
            predictions=resp['prediction']
        except Exception:
            print(response.text)
        df_fc=pd.DataFrame(0, columns=['timestamp','forecast_kwh'], index=data_fc_comb.index)
        df_fc['forecast_kwh'] = df_fc['forecast_kwh'].astype('float')
        df_fc.loc[data_fc_comb.is_day>0, "forecast_kwh"]=predictions  
        df_fc.timestamp=pd.to_datetime(Time)        
        # put back actual prediction input data
        df_fc = df_fc.join(data_fc_comb)   
        df_fc['day'] = df_fc.timestamp.dt.day.tolist()  
        df_fc['hour'] = df_fc.timestamp.dt.hour.tolist()  
        # merge
        # Build actual energy DataFrame
        df_actual = pd.DataFrame({
            "day": days,
            "hour": hours,
            "actual_kwh": energies
        })        
        # Merge on hour using inner join to keep only overlapping hours
        df_combined = pd.merge(df_actual, df_fc, on=["hour", "day"], how="inner")

        # Optional: sort by hour
        df_combined = df_combined.sort_values(["day","hour"]).reset_index(drop=True)        
        #df_combined.to_csv("/config/my_data.csv", index=False)
        df_combined=df_combined[df_combined.forecast_kwh>0.0]
        print(df_combined.iloc[:, 0:5].tail(24))        
        # ------ calculate mse between prediction and actual
        # Extract series
        y_true = df_combined['actual_kwh']
        y_pred = df_combined['forecast_kwh']

        # Calculate RMSE       
        mse = mean_squared_error(y_true, y_pred)
        rmse = np.sqrt(mse)
        print(f"Root Mean Squared Error (RMSE): {rmse:.4f}")

        if rmse>0.3:
           print("RMSE is too high! Initiate retraining.")
           X_train = df_combined.iloc[:, [5, 6, 7, 8, 9, 10, 11, 12]]
           print(X_train.columns)
           print(X_train)        
           y_train = df_combined.actual_kwh.tolist()
           X_train = X_train.values.tolist() 
           print(len(X_train))
           print(len(y_train))
           # API call
           headers = {
               "Content-Type": "application/json",
               "x-api-key": kwargs["key"]
           }       
           payload = {
               "features": X_train,
               "targets": y_train
           }
           response = requests.post(kwargs["url_t"], json=payload, headers=headers)
           try:
               print(response.json())
               print("success" in response.text)
           except Exception:
               print(response.text)
               if "Service Unavailable" in response.text:
                   print(f"Lambda is cold. Waiting 30s seconds before retry...")
                   time.sleep(30)
                   response = requests.post(kwargs["url_t"], json=payload, headers=headers)
                   try:
                       print(response.json())
                   except Exception:
                       print(response.text)
        else:
           print("RMSE is acceptable. Retraining not required.")
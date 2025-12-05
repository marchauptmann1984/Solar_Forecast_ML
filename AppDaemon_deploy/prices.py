import appdaemon.plugins.hass.hassapi as hass
import io
import re
import requests
import pdfplumber
from datetime import datetime, timedelta, timezone

class PriceDATS24(hass.Hass):
    def initialize(self):
        URL1 = "https://profile.dats24.be/api/v1/ratecard?energyType=electricity&&contractType=variable&language=nl"
        URL2 = "https://profile.dats24.be/api/v1/ratecard?energyType=gas&&contractType=variable&language=nl"
        now = datetime.now()
        self.run_every(self.get_prices, now, 3600, url1=URL1, url2=URL2, dag=3, nacht=4, injectie=3, gas=2)
    def fetch_pdf_bytes(self, url: str) -> bytes:
        resp = requests.get(url, timeout=15)
        resp.raise_for_status()
        return resp.content
    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        # Read all pages into one big text string
        with pdfplumber.open(io.BytesIO(pdf_bytes)) as pdf:
            pages_text = []
            for page in pdf.pages:
                page_text = page.extract_text() or ""
                pages_text.append(page_text)
        return "\n".join(pages_text)    
    def find_energy_prices(self, text: str) -> dict:
        lines = [l.strip() for l in text.splitlines() if l.strip()]
        result = {
            "tarief_lines": [],
            "afname_lines": [],
            "teruglevering_lines": [],
            "gas_lines": []
        }
        for line in lines:
            lower = line.lower()
            #print(lower)
            if "enkelvoudige" in lower and "tweevoudige" in lower:
                result["tarief_lines"].append(line)
            if "afname" in lower and "c€/kwh" in lower:
                result["afname_lines"].append(line)
            if "teruglevering" in lower and "c€/kwh" in lower:
                result["teruglevering_lines"].append(line)
            if "energieprijs" in lower and "c€/kwh" in lower:
                result["gas_lines"].append(line)                      
        return result
    def get_prices(self, kwargs):
       URL1=kwargs["url1"]
       URL2=kwargs["url2"]
       ind_d=kwargs["dag"]
       ind_n=kwargs["nacht"]
       ind_i=kwargs["injectie"]
       ind_g=kwargs["gas"]
       pdf_bytes = self.fetch_pdf_bytes(URL1)       
       text = self.extract_text_from_pdf(pdf_bytes)
       prices = self.find_energy_prices(text)
       print(float(prices["afname_lines"][0].split()[ind_d].replace(",",".")))
       print(float(prices["afname_lines"][0].split()[ind_n].replace(",",".")))
       print(float(prices["teruglevering_lines"][0].split()[ind_i].replace(",",".")))
       self.set_state("sensor.stroom_dag",state=float(prices["afname_lines"][0].split()[ind_d].replace(",","."))/100,attributes={"friendly_name":"Stroomprijs dag", "unit_of_measurement": "EUR/kWh"})
       self.set_state("sensor.stroom_nacht",state=float(prices["afname_lines"][0].split()[ind_n].replace(",","."))/100,attributes={"friendly_name":"Stroomprijs nacht", "unit_of_measurement": "EUR/kWh"})
       self.set_state("sensor.stroom_injectie",state=float(prices["teruglevering_lines"][0].split()[ind_i].replace(",","."))/100,attributes={"friendly_name":"Stroomprijs injectie", "unit_of_measurement": "EUR/kWh"})      
       pdf_bytes = self.fetch_pdf_bytes(URL2)       
       text = self.extract_text_from_pdf(pdf_bytes)
       prices = self.find_energy_prices(text)
       print(float(prices["gas_lines"][0].split()[ind_g].replace(",",".")))
       self.set_state("sensor.gas_prijs",state=float(prices["gas_lines"][0].split()[ind_g].replace(",","."))/100,attributes={"friendly_name":"Gasprijs", "unit_of_measurement": "EUR/kWh", "icon": "mdi:currency-eur"})      
      
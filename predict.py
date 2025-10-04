from catboost import CatBoostClassifier,Pool
import pickle
from sklearn.linear_model import LogisticRegression
import numpy as np
import shap
import pandas as pd
class ModelNasa():
    def __init__(self,path_model : str, path_knn : str):
       self.model = CatBoostClassifier()
       self.model.load_model(path_model)
       self.columns = ['koi_period', 'koi_time0bk', 'koi_impact', 'koi_duration', 'koi_depth',
       'koi_prad', 'koi_teq', 'koi_insol', 'koi_model_snr', 'koi_tce_plnt_num',
       'koi_steff', 'koi_slogg', 'koi_srad', 'ra', 'dec', 'koi_kepmag']

       self.names = [
           'orbital_period',
        'transit_epoch',
    'impact_parameter','transit_duration',
    'transit_depth',
    'planetary_radius',
    'equilibrium_temperature',
    'insolation_flux',
    'transit_snr',
    'tce_planet_number',
    'stellar_temperature',
    'stellar_surface_gravity',
    'stellar_radius',
    'ra',
    'dec',
    'kepler_band']
       
       self.comp = dict(zip(self.names, self.columns))
       

       self.imputer = pickle.load(open(path_knn, 'rb'))

    def process_data(self,x: pd.DataFrame):
       
       
       data = self.imputer.transform(x)
       return data
    
    def prediction(self,x):
        d = self.process_data(x)
        return  self.model.predict_proba(d)[0][0]
    

    def analys_feat(self,x):
        d = self.process_data(x)
        explainer = shap.TreeExplainer(self.model)
        shap_values = explainer(d)
        df = pd.DataFrame(columns=self.names)
        df.loc[0] = shap_values.values[0]
        return df
    



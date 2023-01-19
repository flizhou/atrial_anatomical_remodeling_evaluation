# constants

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from lightgbm import LGBMClassifier
from imblearn import over_sampling

class constants(object):
    def __init__(self):
        super().__init__()
        self.DIMENSION = 3
        # the number of spatial transformation
        self.N = 3
        self.RESOLUTION = 0.2
        self.CT_RAW = r'data/raw/ct'
        self.CT_PROCESSED = r'data/processed/ct'
        self.SEED = 81921
        self.sampling_funcs = {
            'SMOTE': over_sampling.SMOTE,
            'ADASYN': over_sampling.ADASYN,
            'BorderlineSMOTE': over_sampling.BorderlineSMOTE,
        }
        self.models = {
            'LogisticRegression': LogisticRegression, 
            'RandomForestClassifier': RandomForestClassifier,  
            'LGBMClassifier': LGBMClassifier,
            'GaussianProcessClassifier': GaussianProcessClassifier
        }
        self.params = {
            'LogisticRegression': {'solver': 'liblinear'}, 
            'RandomForestClassifier': {},
            'GaussianProcessClassifier': {},
            'LGBMClassifier': {},
        }
        self.base_params = {
            'LogisticRegression': {
                'solver': 'liblinear',
                'random_state':self.SEED, 
                'class_weight': 'balanced',
            }   
        }

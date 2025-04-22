# data_function.py
from sklearn.preprocessing import FunctionTransformer
import numpy as np
import pandas as pd

#----------------------DATA INITIALIZATION-------------------------

def load_data(data):
# Load dataset
    df = pd.read_csv(data)
    return df

#-----------------------FEATURE ENGINEERING------------------------------

def sin_transformer(period):
    return FunctionTransformer(lambda x: np.sin(x / period * 2 * np.pi))


def cos_transformer(period):
    return FunctionTransformer(lambda x: np.cos(x / period * 2 * np.pi))

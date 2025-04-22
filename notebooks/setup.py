#setup.py

# common packages imports

import sys
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import scipy
import datetime
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

# Automatically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

# custom package imports
from geolife.geolife.config import data_dir, submission_path, data_test_path, data_train_path
from geolife.geolife.data_function import *
from geolife.geolife.visualization_function import *
from geolife.geolife.modeling_function import *

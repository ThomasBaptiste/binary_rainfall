from pathlib import Path
import sys
import os
import opendatasets as od


# Automatically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)

#folder for dataset
dl_path = Path(project_root) / 'data'

# Assign the Kaggle data set URL into variable
dataset = 'https://www.kaggle.com/competitions/playground-series-s5e3/data'# Using opendatasets let's download the data sets
od.download(dataset, data_dir=dl_path)
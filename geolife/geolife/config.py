# config.py

from pathlib import Path
import sys
import os



# Automatically add the project root to sys.path
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if project_root not in sys.path:
    sys.path.append(project_root)



data_dir = Path(project_root) / 'data/playground-series-s5e3'
data_train_path = data_dir / 'train.csv' 
data_test_path = data_dir / 'test.csv' 
submission_path = Path(project_root) / 'submission'


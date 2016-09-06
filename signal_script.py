import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error

train_data = pd.read_csv('trainHousing.csv')
test_data = pd.read_csv('testHousing.csv')

columns = train_data.columns.tolist()

strong_vars = ['RM', 'LSTAT']
class_var = 'LSTAT'
all_vars = [c for c in columns if c not in ['MEDV']]
positive_vars = ['ZN', 'CHAS', 'RM', 'DIS', 'B']
negative_vars = ['CRIM', 'INDUS', 'NOX', 'AGE', 'RAD', 'TAX', 'PTRATIO', 'LSTAT']
weak_vars = ['CHAS', 'DIS', 'B']


RMSE_values = {}

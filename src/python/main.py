import pandas as pd

control_group_data = pd.read_csv('src/data/control_group_data.csv')
treatment_group_data = pd.read_csv('src/data/treatment_group_data.csv')

print(control_group_data.head())
print(treatment_group_data.head())
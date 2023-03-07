import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import MinMaxScaler


#Collect all of the .csv files
files = glob.glob('PGR_Dataset_1/*.csv')

def csv_label_filter(csv_file: str, planes, label_system: str):
    #Create dataframe from .csv
    df = pd.read_csv(csv_file, index_col=0, header=[0,1])
    pd.set_option('display.max_columns', None)
    #Take just the first 300 rows because these are the only ones that are labeeld
    df = df.head(300)

    angles = df.loc[:, df.columns.get_level_values(0).isin(planes)]
    angles.columns = angles.columns.droplevel(level=0)
    s_scaler = MinMaxScaler()
    scaled_angles = pd.DataFrame(s_scaler.fit_transform(angles), columns=angles.columns)

    phase = df.filter(regex='Phase')
    phase.columns = phase.columns.droplevel(level=0)
    labels = phase.filter(regex=label_system)

    final = pd.concat([scaled_angles, labels], axis="columns")

    return final

#Set up a list to catch each processed dataframe
df_list = []

plns = ['Sagittal Plane Left', 'Sagittal Plane Right', 'Posterior Frontal Plane', 'Anterior Frontal Plane']
ls = 'RL - RunLab'

for i, f in enumerate(files, start=0):
    df = csv_label_filter(csv_file=f, planes=plns, label_system=ls)
    
    #Add to list for concatenation later
    df_list.append(df)

master_df = pd.concat(df_list, axis=0, ignore_index=True)
#filename_planes = "_".join(plns)
filename = ls + "training_set.csv"
print(filename + " is complete.")
master_df.to_csv("PGR_Dataset_1/training_sets/" + filename, index=False)

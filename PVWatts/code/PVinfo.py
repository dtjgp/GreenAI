# this code is the function to extract the dataframe from the csv file
import pandas as pd
import numpy as np
import os

def extract_data(file_name):
    '''find file path and extract data'''
    # find the path of the current file
    current_path = os.path.dirname(os.path.realpath(__file__))  
    # print(current_path)
    # find the path of the parent directory
    parent_path = os.path.abspath(os.path.join(current_path, os.pardir))
    # print(parent_path)
    # find the path of the data folder
    data_path = os.path.join(parent_path, 'PVdata')
    # print(data_path)
    # find the files in the data folder
    data_files = os.listdir(data_path)
    # print(data_files)
    # read the Hourly_Data.csv file
    file_path = os.path.join(data_path, 'Hourly_Data.csv')
    hourly_data = pd.read_csv(file_path)
    # print(hourly_data.head())
    for i, row in enumerate(hourly_data.itertuples(index=False)):
        if "Date" in row or "Hour" in row:
            data_end_row = i
            break   
    metadata = pd.read_csv(file_path, nrows=data_end_row - 1)
    metadata = metadata.dropna(axis=1, how='all') # remove extra NaN columns
    for i, row in enumerate(metadata.itertuples(index=False)):
        if "Monthly Irradiance Loss (%)" in row:
            info_row = i
            break   
    metadata_info = metadata.iloc[:info_row]
    metadata_info = metadata_info.dropna(axis=1, how='all')
    metadata_info.columns = ['Parameter', 'Info']
    Irradiance_info = metadata.iloc[info_row:]
    Irradiance_info.columns = ['Irradiance Loss', 'Month', 'Value']
    return metadata_info, Irradiance_info

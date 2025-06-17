#!/usr/bin/env python
# coding: utf-8

import sys

import pickle
import requests
from datetime import datetime
from venv import logger
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction import DictVectorizer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import make_pipeline


def read_dataframe(run_date: datetime, filename: str) -> pd.DataFrame:
    df = pd.read_parquet(filename)
    df['duration'] = df.tpep_dropoff_datetime - df.tpep_pickup_datetime
    df['duration'] = df.duration.dt.total_seconds() / 60
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    year = run_date.year
    month = run_date.month
    df['ride_id'] = f'{year:04d}/{month:02d}_' + df.index.astype('str')

    return df


def prepare_dictionaries(df: pd.DataFrame):
    categorical = ['PULocationID', 'DOLocationID']
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    dicts = df[categorical].to_dict(orient='records')
    return dicts

def load_model():
    # Define the URL where the model can be downloaded
    model_url = 'https://github.com/DataTalksClub/mlops-zoomcamp/raw/refs/heads/main/cohorts/2025/04-deployment/homework/model.bin'
    model_path = './model.bin'

    # Download the model if it doesn't exist
    if not Path(model_path).exists():
        print(f"Downloading model from {model_url}")
        response = requests.get(model_url)
        with open(model_path, 'wb') as f_out:
            f_out.write(response.content)
        print("Model downloaded successfully")

    # Load the model
    with open(model_path, 'rb') as f_in:
        dv, model = pickle.load(f_in)
    return dv, model

def save_results(df, y_pred, output_file):
    df_result = pd.DataFrame()
    df_result['ride_id'] = df['ride_id']
    df_result['predicted_duration'] = y_pred
    df_result.to_parquet(output_file, engine='pyarrow', compression=None, index=False)


def apply_model(run_date, input_file, output_file):

    logger.info(f'reading the data from {input_file}...')
    df = read_dataframe(run_date, input_file)
    dicts = prepare_dictionaries(df)

    logger.info(f'loading the model...')
    dv, model = load_model()

    logger.info(f'applying the model...')
    X_val = dv.transform(dicts)
    y_pred = model.predict(X_val)

    mean_predicted_duration = y_pred.mean()
    logger.info(f'mean predicted duration: {mean_predicted_duration}')
    print(f'mean predicted duration: {mean_predicted_duration}')

    logger.info(f'saving the result to {output_file}...')
    save_results(df, y_pred, output_file)
    return output_file


def get_paths(run_date, taxi_type):
    year = run_date.year
    month = run_date.month

    input_file = f'https://d37ci6vzurychx.cloudfront.net/trip-data/{taxi_type}_tripdata_{year:04d}-{month:02d}.parquet'
    output_file = f'{taxi_type}_predicted_durations_{year:04d}-{month:02d}.parquet'

    return input_file, output_file


def ride_duration_prediction(
        taxi_type: str,
        run_date: datetime):

    input_file, output_file = get_paths(run_date, taxi_type)

    apply_model(
        run_date=run_date,
        input_file=input_file,
        output_file=output_file
    )


def run():
    taxi_type = sys.argv[1]  # yellow
    year = int(sys.argv[2])  # 2023
    month = int(sys.argv[3])  # 3

    ride_duration_prediction(
        taxi_type=taxi_type,
        run_date=datetime(year=year, month=month, day=1)
    )


if __name__ == '__main__':
    run()
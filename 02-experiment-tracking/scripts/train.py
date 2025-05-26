import os
import pickle
import click
import mlflow

from sklearn.metrics import mean_squared_error
from sklearn.ensemble import RandomForestRegressor
import numpy as np


mlflow.set_tracking_uri("sqlite:///mlflow.db") # set the tracking URI to a local SQLite database
mlflow.set_experiment("mlops_zoomcamp_homework_2") # set the experiment name


# Define the model path
def load_pickle(filename: str):
    with open(filename, "rb") as f_in:
        return pickle.load(f_in)

# Click command to run the training process
@click.command()
@click.option(
    "--data_path",
    default="./output",
    help="Location where the processed NYC taxi trip data was saved"
)


# Function to run the training process
def run_train(data_path: str):
    with mlflow.start_run():
        mlflow.sklearn.autolog(log_datasets=False)

        X_train, y_train = load_pickle(os.path.join(data_path, "train.pkl"))
        X_val, y_val = load_pickle(os.path.join(data_path, "val.pkl"))

        rf = RandomForestRegressor(max_depth=10, random_state=0)
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_val)

        rmse =np.sqrt(mean_squared_error(y_val, y_pred))
        print(f"Validation RMSE: {rmse}")



if __name__ == '__main__':
    run_train()
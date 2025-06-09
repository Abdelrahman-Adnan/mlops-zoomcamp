"""
MLOps Orchestration Pipeline for NYC Taxi Trip Duration Prediction

This module implements a Prefect-based workflow for training and validating
a linear regression model to predict taxi trip duration using NYC taxi data.
"""

import pandas as pd
import pickle
from datetime import datetime
from pathlib import Path
import requests
import tempfile

from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
from dateutil.relativedelta import relativedelta

from prefect import task, flow, get_run_logger
from prefect import serve
from prefect.server.schemas.schedules import CronSchedule

import mlflow
import mlflow.sklearn


@task(name="download_and_read_data", retries=3)
def download_and_read_data(url: str) -> pd.DataFrame:
    """
    Download parquet file from URL and read it into a DataFrame.
    
    Args:
        url (str): URL to the parquet file
        
    Returns:
        pd.DataFrame: Loaded data
        
    Raises:
        requests.RequestException: If download fails
        Exception: If file reading fails
    """
    logger = get_run_logger()
    
    logger.info(f"Downloading data from: {url}")
    
    try:
        # Download the file
        response = requests.get(url, timeout=300)  # 5 minute timeout
        response.raise_for_status()
        
        # Create temporary file to store the downloaded data
        with tempfile.NamedTemporaryFile(delete=False, suffix='.parquet') as tmp_file:
            tmp_file.write(response.content)
            tmp_path = tmp_file.name
        
        # Read the parquet file
        df = pd.read_parquet(tmp_path)
        logger.info(f"Successfully downloaded and loaded {len(df)} records")
        
        # Clean up temporary file
        Path(tmp_path).unlink()
        
        return df
        
    except requests.RequestException as e:
        logger.error(f"Failed to download data from {url}: {e}")
        raise
    except Exception as e:
        logger.error(f"Failed to read parquet file: {e}")
        raise


@task(name="prepare_features")
def prepare_features(df: pd.DataFrame, categorical: list, train: bool = True) -> pd.DataFrame:
    """
    Preprocess the taxi trip data by calculating duration and filtering outliers.
    
    Args:
        df (pd.DataFrame): Raw taxi trip data
        categorical (list): List of categorical column names
        train (bool): Whether this is training data (affects logging)
        
    Returns:
        pd.DataFrame: Processed dataframe with duration and cleaned categorical features
    """
    logger = get_run_logger()
    
    # Calculate trip duration in minutes
    df = df.copy()  # Avoid modifying the original dataframe
    df['duration'] = df['tpep_dropoff_datetime'] - df['tpep_pickup_datetime']
    df['duration'] = df.duration.dt.total_seconds() / 60
    
    # Filter trips between 1 and 60 minutes (remove outliers)
    initial_count = len(df)
    df = df[(df.duration >= 1) & (df.duration <= 60)].copy()
    filtered_count = len(df)
    
    logger.info(f"Filtered {initial_count - filtered_count} outlier records")
    
    # Calculate and log mean duration
    mean_duration = df.duration.mean()
    dataset_type = "training" if train else "validation"
    logger.info(f"Mean duration for {dataset_type} set: {mean_duration:.2f} minutes")
    
    # Process categorical features (fill NaN with -1 and convert to string)
    df[categorical] = df[categorical].fillna(-1).astype('int').astype('str')
    
    return df


@task(name="train_linear_regression")
def train_model(df: pd.DataFrame, categorical: list) -> tuple:
    """
    Train a linear regression model using the preprocessed data.
    
    Args:
        df (pd.DataFrame): Preprocessed training data
        categorical (list): List of categorical column names
        
    Returns:
        tuple: Trained model (LinearRegression) and fitted vectorizer (DictVectorizer)
    """
    logger = get_run_logger()
    
    with mlflow.start_run(nested=True):
        # Convert categorical features to dictionary format for vectorization
        train_dicts = df[categorical].to_dict(orient='records')
        
        # Vectorize categorical features
        dv = DictVectorizer()
        X_train = dv.fit_transform(train_dicts)
        y_train = df.duration.values
        
        # Log feature information
        logger.info(f"Training data shape: {X_train.shape}")
        logger.info(f"Number of features after vectorization: {len(dv.feature_names_)}")
        
        # Train linear regression model
        lr = LinearRegression()
        lr.fit(X_train, y_train)
        
        # Calculate and log training performance
        y_pred = lr.predict(X_train)
        rmse = mean_squared_error(y_train, y_pred, squared=False)
        logger.info(f"Training RMSE: {rmse:.4f}")
        
        # Log metrics to MLflow
        mlflow.log_metric("train_rmse", rmse)
        mlflow.log_param("n_features", len(dv.feature_names_))
        mlflow.log_param("n_samples", len(y_train))
        
    return lr, dv


@task(name="validate_model")
def run_model(df: pd.DataFrame, categorical: list, dv: DictVectorizer, lr: LinearRegression) -> float:
    """
    Validate the trained model on validation data and return RMSE.
    
    Args:
        df (pd.DataFrame): Validation dataset
        categorical (list): List of categorical column names
        dv (DictVectorizer): Fitted vectorizer from training
        lr (LinearRegression): Trained model
        
    Returns:
        float: Validation RMSE
    """
    logger = get_run_logger()
    
    # Transform validation data using the fitted vectorizer
    val_dicts = df[categorical].to_dict(orient='records')
    X_val = dv.transform(val_dicts)
    y_val = df.duration.values
    
    # Make predictions and calculate RMSE
    y_pred = lr.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    logger.info(f"Validation RMSE: {rmse:.4f}")
    
    # Log validation metrics to MLflow
    mlflow.log_metric("val_rmse", rmse)
    
    return rmse


@task(name="construct_download_urls")
def get_data_urls(date: str = None) -> tuple:
    """
    Construct download URLs for training and validation datasets based on date.
    
    Training data: 2 months before the given date
    Validation data: 1 month before the given date
    
    Args:
        date (str, optional): Date in YYYY-MM-DD format. Defaults to today.
        
    Returns:
        tuple: (train_url, val_url) - URLs for training and validation files
    """
    logger = get_run_logger()
    
    # Parse date or use today's date
    if date:
        try:
            processed_date = datetime.strptime(date, "%Y-%m-%d")
        except ValueError:
            logger.error(f"Invalid date format: {date}. Expected YYYY-MM-DD")
            raise
    else:
        processed_date = datetime.today()
    
    # Calculate training and validation dates
    train_date = processed_date - relativedelta(months=2)
    val_date = processed_date - relativedelta(months=1)
    
    # Construct download URLs
    base_url = "https://d37ci6vzurychx.cloudfront.net/trip-data"
    train_url = f"{base_url}/yellow_tripdata_{train_date.year}-{str(train_date.month).zfill(2)}.parquet"
    val_url = f"{base_url}/yellow_tripdata_{val_date.year}-{str(val_date.month).zfill(2)}.parquet"
    
    logger.info(f"Training data URL: {train_url}")
    logger.info(f"Validation data URL: {val_url}")
    
    return train_url, val_url


@task(name="save_model_artifacts")
def save_model_artifacts(dv: DictVectorizer, lr: LinearRegression, date: str) -> str:
    """
    Save the trained model and vectorizer to disk.
    
    Args:
        dv (DictVectorizer): Fitted vectorizer
        lr (LinearRegression): Trained model
        date (str): Date string for filename
        
    Returns:
        str: Path where artifacts were saved
    """
    logger = get_run_logger()
    
    # Create models directory if it doesn't exist
    models_dir = Path("./models")
    models_dir.mkdir(exist_ok=True)
    
    # Save vectorizer
    dv_path = models_dir / f"dv-{date}.pkl"
    with open(dv_path, 'wb') as f:
        pickle.dump(dv, f)
    
    # Save model
    model_path = models_dir / f"model-{date}.pkl"
    with open(model_path, 'wb') as f:
        pickle.dump(lr, f)
    
    logger.info(f"Model artifacts saved: {dv_path}, {model_path}")
    
    return str(models_dir)


@flow(name="taxi_duration_prediction_pipeline")
def main(date: str = None) -> None:
    """
    Main orchestration flow for the taxi trip duration prediction pipeline.
    
    This flow:
    1. Sets up MLflow tracking
    2. Constructs data download URLs
    3. Downloads and preprocesses training/validation data
    4. Trains a linear regression model
    5. Validates the model
    6. Saves model artifacts
    
    Args:
        date (str, optional): Target date in YYYY-MM-DD format. 
                            If None, uses current date.
    """
    logger = get_run_logger()
    logger.info("Starting taxi duration prediction pipeline")
    
    # Configure MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("nyc-taxi-experiment")
    mlflow.sklearn.autolog()  # Enable automatic logging for sklearn
    
    # Define categorical features
    categorical_features = ['PULocationID', 'DOLocationID']
    
    with mlflow.start_run():
        # Get data download URLs
        train_url, val_url = get_data_urls(date)
        
        # Download and preprocess training data
        logger.info("Downloading and processing training data...")
        df_train = download_and_read_data(train_url)
        logger.info(f"Number of samples or rows for the training data:{len(df_train)}")
        df_train_processed = prepare_features(df_train, categorical_features, train=True)
        
        # Download and preprocess validation data
        logger.info("Downloading and processing validation data...")
        df_val = download_and_read_data(val_url)
        df_val_processed = prepare_features(df_val, categorical_features, train=False)
        
        # Train the model
        logger.info("Training model...")
        lr, dv = train_model(df_train_processed, categorical_features)
        
        # Validate the model
        logger.info("Validating model...")
        val_rmse = run_model(df_val_processed, categorical_features, dv, lr)
        
        # Prepare date string for saving
        if date is None:
            date = datetime.today().strftime("%Y-%m-%d")
        
        # Save model artifacts
        logger.info("Saving model artifacts...")
        save_model_artifacts(dv, lr, date)
        
        logger.info(f"Pipeline completed successfully. Validation RMSE: {val_rmse:.4f}")


# use for testing only uncomment when in phase of pre-production
# if __name__ == "__main__":
#     main("2023-05-15")


# Production deployment configuration
def create_deployment():
    """Create and return the deployment specification for the pipeline."""
    return main.to_deployment(
        name="taxi-duration-model-training",
        # Run at 6 AM on the 9th of every month
        schedule=CronSchedule(cron="0 6 9 * *"),
        tags=["mlops", "taxi", "duration-prediction", "production"]
    )

# use for deployment
deployment = create_deployment()
# 🚀 MLOps Zoomcamp 2025: Week 3 - Workflow Orchestration with Prefect 📊

## 🔄 What is Workflow Orchestration?

Workflow orchestration is the process of coordinating, scheduling, and integrating various tasks in a data pipeline. For ML pipelines, this includes:

- Data ingestion and preprocessing
- Feature engineering
- Model training and evaluation
- Model deployment
- Retraining schedules
- Monitoring

Orchestration tools help manage complex dependencies, handle failures gracefully, and provide visibility into pipeline execution.

## 🧩 Prefect: A Modern Orchestration Tool for ML

Prefect is designed specifically for data-intensive applications with a Python-first approach. Unlike older orchestration tools, Prefect allows you to convert your existing Python code into production-ready data pipelines with minimal changes.

### Why Prefect for MLOps?

1. **Pythonic**: Works with your existing Python code
2. **Dynamic**: DAGs can be created at runtime
3. **Resilient**: Built-in error handling and recovery
4. **Observable**: Comprehensive UI and monitoring
5. **Distributed**: Can scale across multiple machines
6. **Modern**: Actively maintained and feature-rich

## 🏗️ Prefect Architecture Explained

Prefect has a distributed architecture with distinct components:

```
┌─────────────────┐     ┌───────────────┐     ┌─────────────────┐
│                 │     │               │     │                 │
│  Prefect API    │◄────┤  Prefect UI   │     │  Storage        │
│  (Orchestrator) │     │  (Dashboard)  │     │  (S3, GCS, etc) │
│                 │     │               │     │                 │
└───────┬─────────┘     └───────────────┘     └────────┬────────┘
        │                                              │
        │                                              │
┌───────▼──────────────────────────────────────┐       │
│                                              │       │
│  Prefect Agents                              │◄──────┘
│  (Workers that execute flows)                │
│                                              │
└──────────────────────────────────────────────┘
```

### Core Components:

1. **Prefect API (Orion)**: The orchestration engine that:
   - Schedules and triggers flows
   - Tracks flow and task states
   - Stores execution history
   - Manages concurrency and dependencies

2. **Prefect UI**: Web-based dashboard that provides:
   - Visual representation of flows and tasks
   - Execution history and logs
   - Performance metrics
   - Administrative controls

3. **Storage**: Where flow code and artifacts are stored:
   - Local filesystem
   - Cloud object storage (S3, GCS, etc.)
   - Git repositories
   - Docker images

4. **Agents**: Workers that:
   - Poll for scheduled flows
   - Pull flow code from storage
   - Execute flows in the specified environment
   - Report results back to the API

## 🧠 Prefect Core Concepts in Detail

### 1. Tasks 📋

Tasks are the atomic units of work in Prefect. They encapsulate individual operations and can be composed into larger workflows.

#### Task Definition:

```python
from prefect import task

@task(
    name="extract_data",  # Custom name for the task
    retries=3,            # Retry automatically on failure
    retry_delay_seconds=30,  # Wait between retries
    cache_key_fn=lambda context, **params: params["url"],  # Cache by URL parameter
    cache_expiration=timedelta(hours=12)  # Cache for 12 hours
)
def extract_data(url: str) -> pd.DataFrame:
    """Extract data from a given URL"""
    return pd.read_csv(url)
```

#### Task Properties:

- **Name**: Human-readable identifier for the task
- **Retries**: Number of times to retry on failure
- **Timeout**: Maximum execution time
- **Tags**: Metadata for filtering and organization
- **Cache**: Store and reuse task results
- **Result Storage**: Where to store task outputs
- **Result Handlers**: How to serialize/deserialize outputs

#### Task States:

Tasks can be in various states during execution:
- **Pending**: Ready to run
- **Running**: Currently executing
- **Completed**: Successfully finished
- **Failed**: Encountered an error
- **Retrying**: Failed but will retry
- **Cancelled**: Manually stopped

### 2. Flows 🌊

Flows are the main unit of work in Prefect. They coordinate the execution of tasks and manage dependencies.

#### Flow Definition:

```python
from prefect import flow
from prefect.task_runners import ConcurrentTaskRunner

@flow(
    name="Data Processing Pipeline",
    description="Extracts, transforms, and loads data",
    version="1.0.0",
    task_runner=ConcurrentTaskRunner(),  # Run tasks concurrently
    retries=2,  # Retry the entire flow if it fails
)
def process_data(date: str = None):
    """Main flow to process data"""
    # If no date provided, use today's date
    if date is None:
        date = datetime.today().strftime("%Y-%m-%d")
    
    # Call tasks, which automatically creates dependencies
    data = extract_data(f"data/data-{date}.csv")
    processed = transform_data(data)
    load_data(processed)
    
    return processed
```

#### Flow Features:

- **Task Dependencies**: Automatically determined from function calls
- **Error Handling**: Custom exception handling
- **Subflows**: Nested flows for better organization
- **Parameters**: Runtime configuration
- **State Handlers**: Custom logic on state changes
- **Logging**: Structured logging for monitoring
- **Concurrency**: Parallel task execution

#### Flow Execution:

Flows can be executed in various ways:
- Directly calling the function
- Using `flow.serve()` for real-time API
- Through deployments for scheduled runs
- Via the Prefect UI or API

### 3. Task Runners 🏃‍♂️

Task runners determine how tasks within a flow are executed.

#### Types of Task Runners:

```python
from prefect.task_runners import SequentialTaskRunner  # Default, run tasks in sequence
from prefect.task_runners import ConcurrentTaskRunner  # Run tasks concurrently
from prefect.task_runners import DaskTaskRunner  # Distributed execution with Dask
```

#### Use Cases:

- **SequentialTaskRunner**: Simple workflows, easy debugging
- **ConcurrentTaskRunner**: Independent tasks that can run in parallel
- **DaskTaskRunner**: Large-scale data processing, distributed computing

### 4. Deployments 🚢

Deployments make flows executable outside your local Python environment.

#### Creating a Deployment:

```python
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from prefect.infrastructure.process import Process

# Create deployment from a flow
deployment = Deployment.build_from_flow(
    flow=process_data,
    name="daily-data-processing",
    version="1",
    schedule=CronSchedule("0 0 * * *"),  # Daily at midnight
    infrastructure=Process(),  # Run as a local process
    tags=["production", "data-pipeline"],
    parameters={"date": None},  # Default parameters
)

# Apply deployment to the Prefect API
deployment.apply()
```

#### Deployment Properties:

- **Name**: Identifier for the deployment
- **Schedule**: When to execute the flow
- **Infrastructure**: Where to run the flow
- **Storage**: Where to store the flow code
- **Parameters**: Default parameters for the flow
- **Tags**: For organizing and filtering deployments

### 5. Schedules ⏰

Schedules determine when flows should be executed automatically.

#### Types of Schedules:

```python
from prefect.orion.schemas.schedules import (
    CronSchedule,          # Based on cron expressions
    IntervalSchedule,      # Run at regular intervals
    RRuleSchedule          # Based on iCalendar recurrence rules
)

# Examples
cron_schedule = CronSchedule(cron="0 9 * * 1-5")  # Weekdays at 9 AM
interval_schedule = IntervalSchedule(interval=timedelta(hours=4))  # Every 4 hours
```

### 6. Results and Persistence 💾

Prefect can store and track results from task and flow runs.

#### Persisting Results:

```python
@task(
    persist_result=True,  # Store the result
    result_serializer=JSONSerializer(),  # How to serialize the result
    result_storage=S3Bucket.load("my-bucket"),  # Where to store the result
)
def calculate_metrics(data):
    # Process data and return metrics
    return {"accuracy": 0.95, "precision": 0.92}
```

## 🛠️ Building an ML Pipeline with Prefect: Complete Example

The following example demonstrates a complete ML pipeline for NYC taxi trip duration prediction:

```python
import pandas as pd
import pickle
from datetime import datetime, timedelta
from sklearn.feature_extraction import DictVectorizer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

from prefect import task, flow
from prefect.task_runners import SequentialTaskRunner
from prefect.deployments import Deployment
from prefect.orion.schemas.schedules import CronSchedule
from prefect.logging import get_run_logger

# Data ingestion task with retries
@task(retries=3, retry_delay_seconds=5, name="download_taxi_data")
def get_data(date):
    """Download taxi data for given date range"""
    logger = get_run_logger()
    
    train_date = datetime.strptime(date, '%Y-%m-%d')
    val_date = train_date + timedelta(days=28)
    
    train_month = train_date.month
    val_month = val_date.month
    train_year = train_date.year
    val_year = val_date.year
    
    train_file = f"yellow_tripdata_{train_year}-{train_month:02d}.parquet"
    val_file = f"yellow_tripdata_{val_year}-{val_month:02d}.parquet"
    
    logger.info(f"Downloading training data: {train_file}")
    train_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{train_file}"
    train_data = pd.read_parquet(train_url)
    
    logger.info(f"Downloading validation data: {val_file}")
    val_url = f"https://d37ci6vzurychx.cloudfront.net/trip-data/{val_file}"
    val_data = pd.read_parquet(val_url)
    
    return train_data, val_data

# Data preprocessing task
@task(name="prepare_taxi_features")
def prepare_features(df, categorical_features):
    """Prepare the features for training"""
    logger = get_run_logger()
    
    # Calculate trip duration
    df['duration'] = (df.tpep_dropoff_datetime - df.tpep_pickup_datetime).dt.total_seconds() / 60
    
    # Filter outliers
    logger.info(f"Initial shape: {df.shape}")
    df = df[(df.duration >= 1) & (df.duration <= 60)]
    logger.info(f"Filtered shape: {df.shape}")
    
    # Create features
    df['PU_DO'] = df['PULocationID'].astype(str) + '_' + df['DOLocationID'].astype(str)
    
    # Process categoricals
    df[categorical_features] = df[categorical_features].fillna(-1).astype('str')
    
    return df

# Model training task
@task(name="train_regression_model")
def train_model(df, categorical_features):
    """Train the model with the given data"""
    logger = get_run_logger()
    
    dicts = df[categorical_features].to_dict(orient='records')
    dv = DictVectorizer()
    X_train = dv.fit_transform(dicts)
    y_train = df['duration'].values
    
    logger.info(f"Training on {X_train.shape[0]} examples")
    
    lr = LinearRegression()
    lr.fit(X_train, y_train)
    
    return lr, dv

# Model evaluation task
@task(name="evaluate_model_performance")
def evaluate_model(model, dv, df, categorical_features):
    """Evaluate the model performance"""
    logger = get_run_logger()
    
    dicts = df[categorical_features].to_dict(orient='records')
    X_val = dv.transform(dicts)
    y_val = df['duration'].values
    
    y_pred = model.predict(X_val)
    rmse = mean_squared_error(y_val, y_pred, squared=False)
    
    logger.info(f"RMSE: {rmse:.3f}")
    return rmse

# Model artifacts storage task
@task(name="save_model_artifacts")
def save_model(model, dv, date):
    """Save the model and DictVectorizer"""
    logger = get_run_logger()
    
    # Create directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    with open(f'models/model-{date}.pkl', 'wb') as f:
        pickle.dump(model, f)
    
    with open(f'models/dv-{date}.pkl', 'wb') as f:
        pickle.dump(dv, f)
    
    logger.info(f"Model saved: models/model-{date}.pkl")
    return True

# Main flow that orchestrates all tasks
@flow(name="taxi-duration-prediction", task_runner=SequentialTaskRunner())
def main(date=None):
    """Main flow for taxi duration prediction model training"""
    if date is None:
        date = datetime.today().strftime('%Y-%m-%d')
    
    logger = get_run_logger()
    logger.info(f"Starting training pipeline for date: {date}")
    
    categorical_features = ['PULocationID', 'DOLocationID', 'PU_DO']
    
    # 1. Get data
    train_data, val_data = get_data(date)
    
    # 2. Process training data
    df_train = prepare_features(df=train_data, categorical_features=categorical_features)
    
    # 3. Train model
    model, dv = train_model(df=df_train, categorical_features=categorical_features)
    
    # 4. Process validation data
    df_val = prepare_features(df=val_data, categorical_features=categorical_features)
    
    # 5. Evaluate model
    rmse = evaluate_model(
        model=model,
        dv=dv,
        df=df_val,
        categorical_features=categorical_features
    )
    
    # 6. Save model
    save_model(model=model, dv=dv, date=date)
    
    logger.info(f"Pipeline completed successfully!")
    return rmse

# Define deployment for production
def create_deployment():
    return Deployment.build_from_flow(
        flow=main,
        name="nyc-taxi-monthly-training",
        schedule=CronSchedule("0 0 1 * *"),  # First day of each month at midnight
        tags=["production", "mlops", "taxi-duration"]
    )

# For local development/testing
if __name__ == "__main__":
    main("2023-01-01")
```

## 📊 MLOps Workflow Integration

In the MLOps lifecycle, Prefect serves as the workflow orchestration layer that connects:

1. **Data Engineering**: Extracting and preparing data for ML
2. **Experimentation**: Running and tracking model experiments
3. **Continuous Training**: Automating regular model retraining
4. **Deployment**: Pushing models to production
5. **Monitoring**: Tracking model performance over time

### Integration with MLflow

Prefect works well with MLflow for experiment tracking:

```python
@task
def train_with_mlflow(X_train, y_train, params):
    with mlflow.start_run():
        # Log parameters
        mlflow.log_params(params)
        
        # Train model
        model = RandomForestRegressor(**params)
        model.fit(X_train, y_train)
        
        # Log metrics
        train_rmse = mean_squared_error(y_train, model.predict(X_train), squared=False)
        mlflow.log_metric("train_rmse", train_rmse)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        return model
```

## 🚦 Prefect vs. Other Orchestration Tools

### Prefect vs. Airflow

| Feature | Prefect | Airflow |
|---------|---------|---------|
| Programming Model | Python-native, tasks as functions | DAGs as configuration |
| Dynamic Workflows | Yes, DAGs can change at runtime | Limited, mostly static DAGs |
| Error Handling | Rich, automated retries and custom handlers | Basic retry policies |
| UI/UX | Modern, task-oriented | Traditional, DAG-oriented |
| Parametrization | First-class support | More complex implementation |
| Learning Curve | Lower for Python developers | Steeper |
| Community | Growing | Large, established |

### Prefect vs. Kubeflow

| Feature | Prefect | Kubeflow |
|---------|---------|----------|
| Focus | General workflow orchestration | Kubernetes-native ML pipelines |
| Deployment | Multiple options (local, cloud, k8s) | Primarily Kubernetes |
| Integration | Python ecosystem | Container-based components |
| Complexity | Lower barrier to entry | Higher complexity |
| ML Specific | General purpose, adaptable to ML | Built specifically for ML |

## 🛫 Getting Started with Prefect in 5 Minutes

### Installation

```bash
pip install prefect
```

### Start the Prefect Server

```bash
prefect server start
```

### Create a Simple Flow

```python
from prefect import task, flow

@task
def say_hello(name):
    return f"Hello, {name}!"

@flow
def hello_flow(name="world"):
    result = say_hello(name)
    print(result)
    return result

if __name__ == "__main__":
    hello_flow("MLOps Zoomcamp")
```

### Create a Deployment

```bash
prefect deployment build hello_flow.py:hello_flow -n my-first-deployment -q default
prefect deployment apply hello_flow-deployment.yaml
```

### Start an Agent

```bash
prefect agent start -q default
```

## 🧪 Best Practices for ML Workflows with Prefect

1. **Task Granularity**: Create tasks at the right level - not too fine-grained, not too coarse
2. **Error Boundaries**: Place tasks boundaries around operations that might fail
3. **Parameterize Flows**: Make flows configurable with parameters
4. **Logging**: Use the Prefect logger to capture important information
5. **Resource Management**: Clean up resources in task teardown
6. **Caching Strategy**: Cache expensive computations but be mindful of data changes
7. **Testing**: Test flows and tasks independently
8. **Version Control**: Track flow code in version control
9. **Documentation**: Document flow purpose, inputs, and outputs
10. **Monitoring**: Set up notifications for critical flow failures

## 🔮 Advanced Prefect Features for ML

### Dask Integration for Distributed Training

```python
from prefect.task_runners import DaskTaskRunner

@flow(task_runner=DaskTaskRunner())
def distributed_training_flow():
    # Tasks will be executed in a distributed Dask cluster
    results = []
    for i in range(10):
        results.append(train_model_fold(fold_id=i))
    return results
```

### Storage Options for Large Models

```python
from prefect.filesystems import S3

# Register S3 block
s3_block = S3(bucket_path="my-model-registry")
s3_block.save("model-storage")

# Use in deployment
deployment = Deployment.build_from_flow(
    flow=train_flow,
    name="distributed-training",
    storage=S3.load("model-storage"),
)
```

### Notifications for Critical Failures

```python
from prefect.notifications import SlackWebhook

slack_webhook = SlackWebhook(url="https://hooks.slack.com/services/XXX/YYY/ZZZ")
slack_webhook.save("ml-alerts")

@flow(on_failure=[SlackWebhook.load("ml-alerts")])
def critical_training_flow():
    # This flow will send a Slack message if it fails
    ...
```


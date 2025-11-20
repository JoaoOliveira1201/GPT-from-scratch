import logging
import os
import sys
from typing import Any, Dict, Optional

import mlflow
from dotenv import load_dotenv

try:
    # This is required to my specific setup
    # I wont share this code since it shows some problems on my current setup :)
    # I'll eventually remove it
    import src.logger.cloud_auth
except ImportError:
    pass

logger = logging.getLogger(__name__)


def setup_logging(name: str, save_dir: str = "logs"):
    load_dotenv()

    global logger
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI"))
    mlflow.set_experiment(os.environ.get("MLFLOW_EXPERIMENT_NAME"))

    if mlflow.active_run():
        try:
            mlflow.end_run()
        except Exception as e:
            print(f"Error ending active MLflow run: {e}")

    mlflow.start_run()

    if logger.hasHandlers():
        logger.handlers.clear()

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    # File logging
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fh = logging.FileHandler(os.path.join(save_dir, "logs.log"), mode="w")
    fh.setFormatter(formatter)
    logger.addHandler(fh)

    logger.info(f"Logging initialized for {name}")


def log_params(params: Dict[str, Any]):
    mlflow.log_params(params)
    logger.info(f"Params logged: {params}")


def log_metrics(metrics: Dict[str, float], step: Optional[int] = None):
    mlflow.log_metrics(metrics, step=step)
    logger.info(f"Metrics logged: {metrics} (Step: {step})")


def log_model(model, artifact_path: str = "model"):
    try:
        mlflow.pytorch.log_model(model, artifact_path)
        logger.info(f"Model saved to artifact path: {artifact_path}")
    except Exception as e:
        logger.error(f"Failed to log model: {e}")


def log_artifact(local_path: str, artifact_path: Optional[str] = None):
    try:
        mlflow.log_artifact(local_path, artifact_path)
        logger.info(f"Artifact logged: {local_path} to path: {artifact_path}")
    except Exception as e:
        logger.error(f"Failed to log artifact: {e}")


def end_run():
    mlflow.end_run()
    logger.info("MLflow run ended.")

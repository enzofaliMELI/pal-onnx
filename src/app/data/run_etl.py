import logging

from app.data.training_dataset import MyETL

logger = logging.getLogger(__name__)


if __name__ == "__main__":
    etl = MyETL()
    etl.run_task()

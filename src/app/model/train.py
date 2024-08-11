import logging

from app.conf.settings import MODEL_HYPER_PARAMETERS
from app.data.training_dataset import unserialize_dataset
from app.model.dummy import DummyModel

logger = logging.getLogger(__name__)


def do_train(artifact_data):
    model = DummyModel(**MODEL_HYPER_PARAMETERS)
    dataset = unserialize_dataset(artifact_data)
    model.train(dataset)
    model.validate_training()
    logger.info("Trained DummyModel with params: {}".format(str(MODEL_HYPER_PARAMETERS)))
    return model.serialize()


def main():
    raw_training_data = ""
    artifact = None  # Replace with your artifact manager
    model_data = do_train(raw_training_data)
    # artifact.save_from_bytes(
    #     data=model_data
    # )


if __name__ == "__main__":
    main()

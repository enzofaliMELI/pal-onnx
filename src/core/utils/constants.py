from enum import Enum

GCS_BUCKET_PROD = "gs://pdme000189"
GCS_BUCKET_DEV = "gs://ddme000189"
GCS_ROOT_BLOB = "search-score"
GCS_ALLOWED_BLOBS = ["tmp", "processed_features"]
GCS_FEATURES_BLOB = "processed_features"


class ExecutionEnv(Enum):
    DEV = "DEV"
    PROD = "PROD"


class AWSEnv(Enum):
    SECRET = "AWS_SK"
    KEY = "AWS_AK"


class AWSBuckets(Enum):
    PROD = "my-prod-bucket"
    DEV = "my-dev-bucket"


class OnnxVersion(Enum):
    IR_VERSION = 8
    OPSET_VERSION = 17

from src import config
from src.data import preprocess
from src.utils import read_data
from src.models.train import train_model, save_model

# Pre-process & load data
preprocess.process_data(config.RAW_DATA_PATH, config.PROCESSED_DATA_PATH)
df = read_data(config.PROCESSED_DATA_PATH)

# Train & serialize model
lr_model = train_model(df, config.NUMERICAL_FEATURES, config.CATEGORICAL_FEATURES, config.TARGET)
save_model(lr_model, config.LR_MODEL_PATH)
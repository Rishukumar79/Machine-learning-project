import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation

if __name__ == '__main__':
    logging.info('The execution has started')
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()
        logging.info('Data Ingestion completed!')

        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, preprocessor_path = data_transformation.initiate_data_transformation(
            train_path, test_path
        )
        logging.info('Data Transformation completed!')

        print("X_train shape:", X_train.shape)
        print("X_test shape:", X_test.shape)
        print("Preprocessor saved at:", preprocessor_path)

    except Exception as e:
        logging.info('Custom exception occurred')
        raise CustomException(e, sys)

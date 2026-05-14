import sys
from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.components.data_ingestion import DataIngestion
from src.mlproject.components.data_transformation import DataTransformation
from src.mlproject.components.model_trainer import ModelTrainer

if __name__ == '__main__':
    logging.info('The execution has started')
    try:
        data_ingestion = DataIngestion()
        train_path, test_path = data_ingestion.initiate_data_ingestion()

        data_transformation = DataTransformation()
        X_train, X_test, y_train, y_test, _ = data_transformation.initiate_data_transformation(
            train_path, test_path
        )

        model_trainer = ModelTrainer()
        score = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
        print(f'Final R2 Score: {score:.4f}')

    except Exception as e:
        logging.info('Custom exception occurred')
        raise CustomException(e, sys)
import sys
import os
import pickle
from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object

@dataclass
class DataTransformationConfig:
    preprocessor_obj_file_path: str = os.path.join('artifacts', 'preprocessor.pkl')

class DataTransformation:
    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    def get_data_transformer_object(self):
        try:
            categorical_columns = [
                'school', 'sex', 'address', 'famsize', 'Pstatus',
                'Mjob', 'Fjob', 'reason', 'guardian', 'schoolsup',
                'famsup', 'paid', 'activities', 'nursery', 'higher',
                'internet', 'romantic'
            ]
            numerical_columns = [
                'age', 'Medu', 'Fedu', 'traveltime', 'studytime',
                'failures', 'famrel', 'freetime', 'goout', 'Dalc',
                'Walc', 'health', 'absences', 'G1', 'G2'
            ]

            num_pipeline = Pipeline(steps=[
                ("scaler", StandardScaler())
            ])

            cat_pipeline = Pipeline(steps=[
                ("one_hot_encoder", OneHotEncoder()),
                ("scaler", StandardScaler(with_mean=False))
            ])

            preprocessor = ColumnTransformer([
                ("num_pipeline", num_pipeline, numerical_columns),
                ("cat_pipeline", cat_pipeline, categorical_columns)
            ])

            return preprocessor

        except Exception as e:
            raise CustomException(e, sys)

    def initiate_data_transformation(self, train_path, test_path):
        try:
            train_df = pd.read_csv(train_path, encoding='latin-1')
            test_df = pd.read_csv(test_path, encoding='latin-1')

            train_df.columns.values[0] = 'school'
            test_df.columns.values[0] = 'school'

            train_df = train_df[train_df['G3'] != 0]
            test_df = test_df[test_df['G3'] != 0]

            target_column = 'G3'

            X_train = train_df.drop(columns=[target_column], axis=1)
            y_train = train_df[target_column]

            X_test = test_df.drop(columns=[target_column], axis=1)
            y_test = test_df[target_column]

            preprocessor_obj = self.get_data_transformer_object()

            X_train_transformed = preprocessor_obj.fit_transform(X_train)
            X_test_transformed = preprocessor_obj.transform(X_test)

            save_object(
                file_path=self.data_transformation_config.preprocessor_obj_file_path,
                obj=preprocessor_obj
            )

            logging.info("Data transformation completed!")

            return (
                X_train_transformed,
                X_test_transformed,
                y_train,
                y_test,
                self.data_transformation_config.preprocessor_obj_file_path
            )

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    obj = DataTransformation()
    X_train, X_test, y_train, y_test, _ = obj.initiate_data_transformation(
        'artifacts/train.csv',
        'artifacts/test.csv'
    )
    print("X_train shape:", X_train.shape)
    print("X_test shape:", X_test.shape)
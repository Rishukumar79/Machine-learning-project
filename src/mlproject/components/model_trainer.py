import os
import sys
from dataclasses import dataclass

from xgboost import XGBRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from catboost import CatBoostRegressor
from sklearn.metrics import r2_score

from src.mlproject.logger import logging
from src.mlproject.exception import CustomException
from src.mlproject.utils import save_object

@dataclass
class ModelTrainerConfig:
    trained_model_path: str = os.path.join('artifacts', 'model.pkl')

class ModelTrainer:
    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_trainer(self, X_train, X_test, y_train, y_test):
        try:
            models = {
                "Linear Regression": LinearRegression(),
                "Lasso": Lasso(),
                "Ridge": Ridge(),
                "K-Neighbors Regressor": KNeighborsRegressor(),
                "Decision Tree": DecisionTreeRegressor(),
                "Random Forest": RandomForestRegressor(),
                "XGBRegressor": XGBRegressor(),
                "CatBoost": CatBoostRegressor(verbose=False),
                "AdaBoost": AdaBoostRegressor()
            }

            best_model_name = None
            best_score = 0
            best_model = None

            for name, model in models.items():
                model.fit(X_train, y_train)
                y_pred = model.predict(X_test)
                score = r2_score(y_test, y_pred)

                print(f"{name}: {score:.4f}")

                if score > best_score:
                    best_score = score
                    best_model_name = name
                    best_model = model

            if best_score < 0.6:
                raise CustomException("No best model found!", sys)

            print(f"\nBest Model: {best_model_name} — R2: {best_score:.4f}")

            save_object(
                file_path=self.model_trainer_config.trained_model_path,
                obj=best_model
            )

            logging.info(f"Best model saved: {best_model_name}")
            return best_score

        except Exception as e:
            raise CustomException(e, sys)


if __name__ == "__main__":
    from src.mlproject.components.data_ingestion import DataIngestion
    from src.mlproject.components.data_transformation import DataTransformation

    data_ingestion = DataIngestion()
    train_path, test_path = data_ingestion.initiate_data_ingestion()

    data_transformation = DataTransformation()
    X_train, X_test, y_train, y_test, _ = data_transformation.initiate_data_transformation(
        train_path, test_path
    )

    model_trainer = ModelTrainer()
    score = model_trainer.initiate_model_trainer(X_train, X_test, y_train, y_test)
    print(f"Final R2 Score: {score:.4f}")
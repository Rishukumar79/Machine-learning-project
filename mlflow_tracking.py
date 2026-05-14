import dagshub
dagshub.init(repo_owner='Rishukumar79', repo_name='Machine-learning-project', mlflow=True)

import mlflow
import mlflow.sklearn
import pandas as pd
import numpy as np
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from catboost import CatBoostRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split

# Data load
df = pd.read_csv('artifacts/raw.csv', encoding='latin-1')
df.columns.values[0] = 'school'
df.drop_duplicates(inplace=True)
df = df[df['G3'] != 0]

X = df.drop(columns=['G3'], axis=1)
y = df['G3']

num_features = X.select_dtypes(exclude="object").columns
cat_features = X.select_dtypes(include="object").columns

preprocessor = ColumnTransformer([
    ("OneHotEncoder", OneHotEncoder(), cat_features),
    ("StandardScaler", StandardScaler(), num_features),
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

models = {
    "Linear Regression": LinearRegression(),
    "Lasso": Lasso(),
    "Ridge": Ridge(),
    "K-Neighbors": KNeighborsRegressor(),
    "Decision Tree": DecisionTreeRegressor(),
    "Random Forest": RandomForestRegressor(),
    "XGBRegressor": XGBRegressor(),
    "CatBoost": CatBoostRegressor(verbose=False),
    "AdaBoost": AdaBoostRegressor()
}

for name, model in models.items():
    with mlflow.start_run(run_name=name):
        pipe = Pipeline(steps=[
            ("preprocessor", preprocessor),
            ("model", model)
        ])
        pipe.fit(X_train, y_train)
        y_pred = pipe.predict(X_test)

        r2 = r2_score(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        mae = mean_absolute_error(y_test, y_pred)

        mlflow.log_param('model', name)
        mlflow.log_metric('r2_score', r2)
        mlflow.log_metric('rmse', rmse)
        mlflow.log_metric('mae', mae)
        mlflow.sklearn.log_model(pipe, name)

        print(f"{name}: R2={r2:.4f}")

print("\nSab experiments DagsHub pe log ho gaye!")
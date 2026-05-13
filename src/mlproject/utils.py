import os
import sys
import pandas as pd
import pymysql
from dotenv import load_dotenv
from src.mlproject.exception import CustomException
from src.mlproject.logger import logging

load_dotenv()

host     = os.getenv('host')
user     = os.getenv('user')
password = os.getenv('password')
db       = os.getenv('db')

def read_sql_data():
    logging.info("Reading the MySQL dataset started")
    try:
        mydb = pymysql.connect(
            host=host,
            user=user,
            password=password,
            db=db
        )
        logging.info(f"Connection Established: {mydb}")
        df = pd.read_sql_query("SELECT * FROM `maths-dataset`", mydb)
        return df

    except Exception as ex:
        raise CustomException(ex, sys)
import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f'Object saved at: {file_path}')
    except Exception as e:
        raise CustomException(e, sys)

import pickle

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)
        os.makedirs(dir_path, exist_ok=True)
        with open(file_path, 'wb') as file_obj:
            pickle.dump(obj, file_obj)
        logging.info(f'Object saved at: {file_path}')
    except Exception as e:
        raise CustomException(e, sys)

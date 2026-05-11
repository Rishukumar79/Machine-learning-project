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
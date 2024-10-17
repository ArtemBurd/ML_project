from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime
#import pandas as pd

default_args = {
    'start_date': datetime(2024, 10, 15),
    'retries': 1
}

with DAG('add_train_data', default_args=default_args, schedule_interval=None) as dag:
    def insert_train_data():
        import psycopg2
        import pandas as pd
        import logging

        data_path = '/opt/airflow/dags/data/data.csv'  # Update with the path inside the container
        df = pd.read_csv(data_path)
        logging.info(df.shape)
            
        # Connect to the PostgreSQL database
        conn = psycopg2.connect(
            host='postgres-bakery',
            database='bakery_data',
            user='bakery_admin',
            password='bakery_admin',
            port=5432
        )
        cursor = conn.cursor()
        logging.info('Connected to bakery_data database successfully')

        # Insert data from the DataFrame into the table
        for _, row in df.iterrows():
            cursor.execute(
                """
                INSERT INTO bakery_sales (date, income, croissant, tartelette, boisson_33cl)
                VALUES (%s, %s, %s, %s, %s);
                """,
                (row['date'], row['income'], row['croissant'], row['tartelette'], row['boisson_33cl'])
            )
            
        # Commit the transaction and close the connection
        conn.commit()
        logging.info('Data inserted successfully')
        cursor.close()
        conn.close()
            
    PythonOperator(
        task_id='insert_train_data',
        python_callable=insert_train_data
    )
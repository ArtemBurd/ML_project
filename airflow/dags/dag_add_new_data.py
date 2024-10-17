from airflow import DAG
from airflow.operators.python import PythonOperator
from datetime import datetime

default_args = {
    'start_date': datetime(2024, 10, 15),
    'retries': 1
}

with DAG('add_new_data', default_args=default_args, schedule_interval=None) as dag:
    def insert_new_data():
        import psycopg2
        import pandas as pd
        import logging

        data_path = '/opt/airflow/dags/data/data_new.csv'  # Update with the path inside the container
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

    def retrain_model():
        import requests
        import logging
        response = requests.post('http://model-service:5000/train', json={})
        if response.status_code != 200:
            raise Exception('Failed to retrain the model')
        else:
            logging.info('retrain model triggered successfully')
            
            
    insert_data = PythonOperator(
        task_id='insert_new_data',
        python_callable=insert_new_data
    )

    retrain_model = PythonOperator(
        task_id='retrain_model',
        python_callable=retrain_model
    )

    insert_data >> retrain_model
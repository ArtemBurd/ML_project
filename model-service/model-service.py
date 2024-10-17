from flask import Flask, request, jsonify
import torch
import torch.nn as nn
import joblib
from sklearn.preprocessing import MinMaxScaler
import numpy as np
import pandas as pd
from io import StringIO
import psycopg2

from model_class import NeuralNetwork

app = Flask(__name__)


model_income = torch.load('saved_models/model_income.pth')
model_pastry = torch.load('saved_models/model_pastry.pth')

# Load the MinMaxScaler
scaler_income = joblib.load('saved_models/scaler_income.gz')
scaler_pastry = joblib.load('saved_models/scaler_pastry.gz')

def process_dataframe(df_json):
    data = pd.read_json(StringIO(df_json), orient='split')
    data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')
    data['day_of_week'] = data['date'].dt.dayofweek  # Monday=0, Sunday=6
    data['month'] = data['date'].dt.month  # January=1, December=12
    data.drop(columns=["date"], inplace=True)
    return data

def preprocess(df):
    X_train_income, X_test_income = df.iloc[:-100][["income", "day_of_week", "month"]], df.iloc[-100:][["income", "day_of_week", "month"]]
    X_train_pastry, X_test_pastry = df.iloc[:-100][["croissant",  "tartelette",  "boisson_33cl", "day_of_week", "month"]], df.iloc[-100:][["croissant",  "tartelette",  "boisson_33cl", "day_of_week", "month"]]
        
    X_train_income_scaled = scaler_income.fit_transform(X_train_income)
    X_test_income_scaled = scaler_income.transform(X_test_income)
    X_train_pastry_scaled = scaler_pastry.fit_transform(X_train_pastry)
    X_test_pastry_scaled = scaler_pastry.transform(X_test_pastry)
        
    joblib.dump(scaler_income, 'saved_models/model_income.pth')
    joblib.dump(scaler_pastry, 'saved_models/model_pastry.pth')
        
    return X_train_income_scaled, X_test_income_scaled, X_train_pastry_scaled, X_test_pastry_scaled
    
def create_sequences(data, sequence_length, lookahead):
    sequences = []
    labels = []
    for i in range(len(data) - sequence_length - lookahead):
        sequences.append(data[i:i+sequence_length])
        labels.append(data[i+sequence_length:i+sequence_length+lookahead, :-2])
    return np.array(sequences), np.array(labels)

def train_loop(X_train, y_train, model, loss_fn, optimizer):
    model.train()    
    output = model(X_train)
    loss = loss_fn(output, y_train.reshape(-1, output.shape[1]))
    
    # Backpropagation
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

def test_loop(X_test, y_test, model, loss_fn):
    model.eval()
    with torch.no_grad():
        predictions = model(X_test)
        print(predictions.shape)
        test_loss = loss_fn(predictions, y_test.reshape(-1, predictions.shape[1]))
        print(f'Test Loss: {test_loss.item():.4f}')
    return test_loss.item()

def train_income_model(X_train_income_seq, y_train_income_seq, X_test_income_seq, y_test_income_seq):
    optimizer_income = torch.optim.Adam(model_income.parameters(), lr=0.005)
    scheduler_income = torch.optim.lr_scheduler.ExponentialLR(optimizer_income, gamma=0.997)
    
    epochs = 1000
    best_mse = 1.
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_loop(X_train_income_seq, y_train_income_seq, model_income, nn.MSELoss(), optimizer_income)
        
        print("Results:")
        print("Validation ", end = '')
        current_mse = test_loop(X_test_income_seq, y_test_income_seq, model_income, nn.L1Loss())
        if current_mse < best_mse:
            best_mse = current_mse
            torch.save(model_income, 'saved_models/model_income.pth')
        scheduler_income.step()
    print("Done!")
    
def train_pastry_model(X_train_pastry_seq, y_train_pastry_seq, X_test_pastry_seq, y_test_pastry_seq):
    optimizer_pastry = torch.optim.Adam(model_pastry.parameters(), lr=0.005)
    scheduler_pastry = torch.optim.lr_scheduler.ExponentialLR(optimizer_pastry, gamma=0.997)
    
    epochs = 1000
    best_mse = 1.
    for t in range(epochs):
        print(f"\nEpoch {t+1}\n-------------------------------")
        train_loop(X_train_pastry_seq, y_train_pastry_seq, model_pastry, nn.MSELoss(), optimizer_pastry)
        
        print("Results:")
        print("Validation ", end = '')
        current_mse = test_loop(X_test_pastry_seq, y_test_pastry_seq, model_pastry, nn.L1Loss())
        if current_mse < best_mse:
            best_mse = current_mse
            torch.save(model_pastry, 'saved_models/model_pastry.pth')
        scheduler_pastry.step()
    print("Done!")

@app.route('/health', methods=['GET'])
def health_check():
    return jsonify({"status": "healthy"}), 200

@app.route('/predict-income', methods=['POST'])
def predict_income():
    model_income.eval()
    df_json = request.get_json().get('dataframe')
    df = process_dataframe(df_json)
    data_prepr = scaler_income.transform(df)
    
    X_pred = torch.Tensor(data_prepr).unsqueeze(0)
    
    with torch.no_grad():
        y_pred = model_income(X_pred)
        
    dummy_y_pred = np.hstack((y_pred.numpy().reshape(14, 1), np.zeros((14, 2))))
    y_pred_origrange = scaler_income.inverse_transform(dummy_y_pred)[:, 0]
    return jsonify({'prediction': y_pred_origrange.tolist()})

@app.route('/predict-pastry', methods=['POST'])
def predict_pastry():
    model_pastry.eval()
    df_json = request.get_json().get('dataframe')
    df = process_dataframe(df_json)
    data_prepr = scaler_pastry.transform(df)
    
    X_pred = torch.Tensor(data_prepr).unsqueeze(0)
    
    with torch.no_grad():
        y_pred = model_pastry(X_pred)
        
    dummy_y_pred = np.hstack((y_pred.numpy().reshape(14, 3), np.zeros((14, 2))))
    y_pred_origrange = scaler_pastry.inverse_transform(dummy_y_pred)[:, :-2]
    return jsonify({'prediction': y_pred_origrange.tolist()})

@app.route('/train', methods=['POST'])
def train():    
    conn = psycopg2.connect(
        host="postgres-bakery",
        port=5432,
        database="bakery_data",
        user="bakery_admin",
        password="bakery_admin"
    )
    query = "SELECT * FROM bakery_sales;"
    df = pd.read_sql_query(query, conn)
    df['date'] = pd.to_datetime(df['date'], format='%Y-%m-%d')
    df['day_of_week'] = df['date'].dt.dayofweek  # Monday=0, Sunday=6
    df['month'] = df['date'].dt.month  # January=1, December=12
    df.drop(columns=["date", "id"], inplace=True)
    
    X_train_income_scaled, X_test_income_scaled, X_train_pastry_scaled, X_test_pastry_scaled = preprocess(df)
    
    sequence_length = 28  # Робимо прогноз на основі минулих 28-ти днів
    lookahead = 14    # Прогнозуємо на 14 днів вперед
    
    X_train_income_seq, y_train_income_seq = create_sequences(X_train_income_scaled, sequence_length, lookahead)
    X_test_income_seq, y_test_income_seq = create_sequences(X_test_income_scaled, sequence_length, lookahead)
    
    X_train_income_seq = torch.Tensor(X_train_income_seq)
    y_train_income_seq = torch.Tensor(y_train_income_seq)
    X_test_income_seq = torch.Tensor(X_test_income_seq)
    y_test_income_seq = torch.Tensor(y_test_income_seq)
    
    X_train_pastry_seq, y_train_pastry_seq = create_sequences(X_train_pastry_scaled, sequence_length, lookahead)
    X_test_pastry_seq, y_test_pastry_seq = create_sequences(X_test_pastry_scaled, sequence_length, lookahead)
    
    X_train_pastry_seq = torch.Tensor(X_train_pastry_seq)
    y_train_pastry_seq = torch.Tensor(y_train_pastry_seq)
    X_test_pastry_seq = torch.Tensor(X_test_pastry_seq)
    y_test_pastry_seq = torch.Tensor(y_test_pastry_seq)
    
    train_income_model(X_train_income_seq, y_train_income_seq, X_test_income_seq, y_test_income_seq)
    train_pastry_model(X_train_pastry_seq, y_train_pastry_seq, X_test_pastry_seq, y_test_pastry_seq)
    
    global model_income
    model_income = torch.load('saved_models/model_income.pth')
    global model_pastry
    model_pastry = torch.load('saved_models/model_pastry.pth')
    return "Models are trained"

if __name__ == '__main__':
    app.run(host='0.0.0.0')
import torch.nn as nn

class NeuralNetwork(nn.Module):
    def __init__(self, input_size, hidden_size, lookahead, predict_dim, num_layers=1):
        super(NeuralNetwork, self).__init__()
        self.lookahead = lookahead        # на скільки днів робиться прогноз
        self.predict_dim = predict_dim    # розмірність часового ряду для прогнозу
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, predict_dim*lookahead)
    
    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out
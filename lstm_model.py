import torch
import torch.nn as nn
import torch.nn.functional as F


class VideoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(VideoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Linear(hidden_size, output_size)
        self.hidden_cell = (torch.zeros(1, 1, self.hidden_layer_size),
                            torch.zeros(1, 1, self.hidden_layer_size))

    def forward(self, input_seq):
        # lstm takes input shape (sequence length, batch, input size); batch is 1
        lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        # output shape is (seq_len, batch, num_directions * hidden_size); batch is 1
        predictions = self.linear(lstm_out.view(len(input_seq), -1))
        predictions = F.relu(predictions)
        return predictions


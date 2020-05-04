import torch
import torch.nn as nn
import torch.nn.functional as F


# output_sizes is a tuple containing the sizes of (class one-hot output, bounding box output, confidence output)
class VideoLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_sizes):
        super(VideoLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)

        # fc layer for classification, bounding box, and confidence
        self.fc_class = nn.Linear(hidden_size, output_sizes[0])
        self.fc_bbox = nn.Linear(hidden_size, output_sizes[1])

    def forward(self, input_seq):
        # reset the state of LSTM
        # the state is kept till the end of the sequence
        #h_t = torch.zeros(self.num_layers, input_seq.size(1), self.hidden_size, dtype=torch.float32)
        #c_t = torch.zeros(self.num_layers, input_seq.size(1), self.hidden_size, dtype=torch.float32)

        #output, hidden = self.lstm(input_seq, (h_t, c_t))
        output, hidden = self.lstm(input_seq)
        # reshape to be batches and flatten everything else
        #output = output.view((input_seq.size(1), -1))

        class_pred = F.relu(self.fc_class(output))
        bbox_pred = F.relu(self.fc_bbox(output))

        # batch size MUST be 1! IDK how to deal with bigger batches.
        class_pred = class_pred.squeeze(1)
        bbox_pred = bbox_pred.squeeze(1)

        return class_pred, bbox_pred


'''
        # lstm takes input shape (sequence length, batch, input size); batch is 1
        #lstm_out, self.hidden_cell = self.lstm(input_seq.view(len(input_seq), 1, -1), self.hidden_cell)
        lstm_out, self.hidden_cell = self.lstm(input_seq, self.hidden_cell)

        # output shape is (seq_len, batch, num_directions * hidden_size); batch is 1
        class_pred = self.fc_class(lstm_out.view(len(input_seq), -1))
        bbox_pred = self.fc_bbox(lstm_out.view(len(input_seq), -1))

        class_pred = F.relu(class_pred)
        bbox_pred = F.relu(bbox_pred)

        return class_pred, bbox_pred

        #predictions = self.linear(lstm_out.view(len(input_seq), -1))
        #predictions = F.relu(predictions)
        
        #return predictions
'''


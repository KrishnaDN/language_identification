
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
import numpy as np

class AttentionModel(torch.nn.Module):
    def __init__(self, batch_size, input_size, output_size, hidden_size,num_layers=2):
        super(AttentionModel, self).__init__()
        self.batch_size = batch_size
        self.output_size = output_size
        self.hidden_size = hidden_size
        		
        self.lstm = nn.LSTM(input_size, hidden_size,num_layers,batch_first=True,bidirectional=True)
        self.label = nn.Linear(hidden_size*4, output_size)
        
    def forward(self, input_features):
        
        output, (final_hidden_state, final_cell_state) = self.lstm(input_features) # final_hidden_state.size() = (1, batch_size, hidden_size) 
        mean = torch.mean(output,1)
        std = torch.std(output,1)
        stat = torch.cat((mean,std),1)
		#attn_output = self.attention_net(output, final_hidden_state
        logits = self.label(stat)
        soft_out = F.softmax(logits)
        return soft_out

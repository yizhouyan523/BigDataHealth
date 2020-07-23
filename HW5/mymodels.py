import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable


##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####


class MyMLP(nn.Module):
	def __init__(self):
		super(MyMLP, self).__init__()
		self.input_layer = nn.Linear(178, 16) #unimproved model
		self.output_layer = nn.Linear(16, 5) #unimproved model
		#self.input_layer = nn.Linear(178, 50) #improved model
		#self.output_layer = nn.Linear(50, 5) #improved model
		
	def forward(self, x):
		x = F.sigmoid(self.input_layer(x)) #unimproved model 
		#x = F.relu(self.input_layer(x)) #improved model
		x = self.output_layer(x) 
		
		return x


class MyCNN(nn.Module):
	def __init__(self):
		super(MyCNN, self).__init__()
		self.layer1 = nn.Conv1d(in_channels = 1, out_channels= 6, kernel_size = 5,stride=1) #unimproved model
		self.layer2 = nn.Conv1d(6, 16, 5) #unimproved model
		self.pool = nn.MaxPool1d(kernel_size = 2) #unimproved model
		self.input_layer = nn.Linear(in_features=16 * 41, out_features=128) #unimproved model
		self.output_layer = nn.Linear(128, 5) #unimproved model

		#self.improvement1 = nn.Dropout(p = 0.2) #improved model

	def forward(self, x):
		x = self.pool(F.relu(self.layer1(x))) #unimproved model
		x = self.pool(F.relu(self.layer2(x))) #unimproved model
		x = x.view(-1, 16 * 41) #unimproved model
		x = F.relu(self.input_layer(x)) #unimproved model
		x = self.output_layer(x) #unimproved model
		#x = self.pool(F.relu(self.improvement1(self.layer1(x)))) #improved model
		#x = self.pool(F.relu(self.improvement1(self.layer2(x)))) #improved model
		#x = x.view(-1, 16 * 41) #improved model
		#x = F.relu(self.improvement1(self.input_layer(x))) #improved model
		#x = self.output_layer(x) #improved model
		return x


class MyRNN(nn.Module):
	def __init__(self):
		super(MyRNN, self).__init__()
		self.rnn_model = nn.GRU(input_size= 1, hidden_size= 16, num_layers = 1, batch_first = True) #unimproved model
		#self.input_layer = nn.Linear(in_features = 16, out_features = 5) #unimproved model
		#self.rnn_model = nn.GRU(input_size= 1, hidden_size = 16, num_layers = 1, batch_first = True,dropout=0.2) #improved model
		self.input_layer = nn.Linear(in_features = 16, out_features = 5) #improved model
	def forward(self, x):
		x, _ = self.rnn_model(x) #unimproved model
		#x = self.input_layer(x[:, -1, :]) #unimproved model
		#x, _ = self.rnn_model(x) #improved model
		x = F.relu(x[:, -1, :]) #improved model
		#x = F.sigmoid(x)
		x = self.input_layer(x) #improved model
		return x

class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.input_layer1 = nn.Linear(in_features=dim_input, out_features=32) #unimproved model
		self.rnn_model = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True) #unimproved model
		#self.rnn_model = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
		self.input_layer2 = nn.Linear(in_features=16, out_features=2) #unimproved model
		#self.input_layer2 = nn.Linear(in_features=128,out_features=2)


	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = torch.tanh(self.input_layer1(seqs)) #unimproved model
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True) #unimproved model
		seqs, h = self.rnn_model(seqs) #unimproved model
		seqs, _ = pad_packed_sequence(seqs, batch_first=True) #unimproved model
		seqs = self.input_layer2(seqs[:, -1, :]) #unimproved model
		return seqs
'''
#improved model
class MyVariableRNN(nn.Module):
	def __init__(self, dim_input):
		super(MyVariableRNN, self).__init__()
		self.input_layer1 = nn.Linear(in_features=dim_input, out_features=32) #unimproved model
		self.input_layer1.bias.data.zero_()
		#self.rnn_model = nn.GRU(input_size=32, hidden_size=16, num_layers=1, batch_first=True) #unimproved model
		self.rnn_model = nn.GRU(input_size=32, hidden_size=128, num_layers=1, batch_first=True)
		self.input_layer2 = nn.Linear(in_features=128, out_features=128) #unimproved model
		self.input_layer2.bias.data.zero_()
		#self.input_layer2 = nn.Linear(in_features=128,out_features=128)
		self.rnn_model2 = nn.GRU(input_size=128, hidden_size=128, num_layers=1, batch_first=True)
		self.input_layer3 = nn.Linear(in_features=128, out_features=2) 
		#self.rnno = nn.Sequential(nn.Dropout(p=0.5),nn.Linear(in_features=128, out_features=2))
		#self.rnno[1].bias.data.zero_()

	def forward(self, input_tuple):
		seqs, lengths = input_tuple
		seqs = torch.tanh(self.input_layer1(seqs)) #unimproved model
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True) #unimproved model
		seqs, h = self.rnn_model(seqs) #unimproved model
		seqs, _ = pad_packed_sequence(seqs, batch_first=True) #unimproved model
		seqs = self.input_layer2(seqs) 
		seqs = torch.tanh(seqs)
		seqs = pack_padded_sequence(seqs, lengths, batch_first=True)
		seqs, h = self.rnn_model2(seqs)
		seqs, _ = pad_packed_sequence(seqs, batch_first=True)
		seqs = self.input_layer3(seqs[:, -1, :])
		return seqs
'''
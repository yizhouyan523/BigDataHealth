import numpy as np
import pandas as pd
from scipy import sparse
import torch
from torch.utils.data import TensorDataset, Dataset
from functools import reduce

##### DO NOT MODIFY OR REMOVE THIS VALUE #####
checksum = '169a9820bbc999009327026c9d76bcf1'
##### DO NOT MODIFY OR REMOVE THIS VALUE #####

def load_seizure_dataset(path, model_type):
	"""
	:param path: a path to the seizure data CSV file
	:return dataset: a TensorDataset consists of a data Tensor and a target Tensor
	"""
	# TODO: Read a csv file from path.
	# TODO: Please refer to the header of the file to locate X and y.
	# TODO: y in the raw data is ranging from 1 to 5. Change it to be from 0 to 4.
	# TODO: Remove the header of CSV file of course.
	# TODO: Do Not change the order of rows.
	# TODO: You can use Pandas if you want to.
	df = pd.read_csv(path)
	X = df.drop('y',axis=1).values
	y = (df['y']-1).values

	if model_type == 'MLP':
		data = torch.tensor(X.astype('float32'))
		target = torch.tensor(y)
		dataset = TensorDataset(data, target)
	elif model_type == 'CNN':
		data = torch.from_numpy(X.astype('float32')).unsqueeze(1)
		target = torch.tensor(y)
		dataset = TensorDataset(data, target)
	elif model_type == 'RNN':
		data = torch.from_numpy(X.astype('float32')).unsqueeze(2)
		target = torch.tensor(y)
		dataset = TensorDataset(data, target)
	else:
		raise AssertionError("Wrong Model Type!")

	return dataset


def calculate_num_features(seqs):
	"""
	:param seqs:
	:return: the calculated number of features
	"""
	# TODO: Calculate the number of features (diagnoses codes in the train set)
	f = reduce(lambda x,y: x+y, seqs)
	fr = reduce(lambda x,y: x+y, f)
	no = len(np.unique(fr))
	return no


class VisitSequenceWithLabelDataset(Dataset):
	def __init__(self, seqs, labels, num_features):
		"""
		Args:
			seqs (list): list of patients (list) of visits (list) of codes (int) that contains visit sequences
			labels (list): list of labels (int)
			num_features (int): number of total features available
		"""

		if len(seqs) != len(labels):
			raise ValueError("Seqs and Labels have different lengths")

		self.labels = labels

		# TODO: Complete this constructor to make self.seqs as a List of which each element represent visits of a patient
		# TODO: by Numpy matrix where i-th row represents i-th visit and j-th column represent the feature ID j.
		# TODO: You can use Sparse matrix type for memory efficiency if you want.
		ans = []
		for s in seqs:
			matrices = np.zeros((len(s),num_features))
			i=0
			for a in s:
				for b in a:
					matrices[i,b] = 1
				i+=1
			ans.append(matrices)
			self.seqs = ans 

	def __len__(self):
		return len(self.labels)

	def __getitem__(self, index):
		# returns will be wrapped as List of Tensor(s) by DataLoader
		return self.seqs[index], self.labels[index]


def visit_collate_fn(batch):
	"""
	DataLoaderIter call - self.collate_fn([self.dataset[i] for i in indices])
	Thus, 'batch' is a list [(seq1, label1), (seq2, label2), ... , (seqN, labelN)]
	where N is minibatch size, seq is a (Sparse)FloatTensor, and label is a LongTensor

	:returns
		seqs (FloatTensor) - 3D of batch_size X max_length X num_features
		lengths (LongTensor) - 1D of batch_size
		labels (LongTensor) - 1D of batch_size
	"""

	# TODO: Return the following two things
	# TODO: 1. a tuple of (Tensor contains the sequence data , Tensor contains the length of each sequence),
	# TODO: 2. Tensor contains the label of each sequence
	lines_list = []
	i = 0
	for x,y in batch:
		lines_list.append((x.shape[0],i))
		i+=1
	#lines_list = sorted(lines.items(),key=lambda kv:(kv[1],kv[0]),reverse=True)
	lines_list.sort(key=lambda x:x[0], reverse=True)
	list_seqs=[]
	list_lengths=[]
	list_labels=[]

	for i in range(len(lines_list)):
		idx = lines_list[i][1]
		patient = batch[idx]
		list_labels.append(patient[1])
		list_lengths.append(patient[0].shape[0])
		seq = np.zeros((lines_list[0][0],batch[0][0].shape[1]))
		seq[0:patient[0].shape[0],0:patient[0].shape[1]]=patient[0]
		list_seqs.append(seq)

	seqs_tensor = torch.FloatTensor(list_seqs)
	lengths_tensor = torch.LongTensor(list_lengths)
	labels_tensor = torch.LongTensor(list_labels)

	return (seqs_tensor, lengths_tensor), labels_tensor

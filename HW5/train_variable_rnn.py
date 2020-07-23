import os
import pickle

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torch.optim as optim

from utils import train, evaluate, make_kaggle_submission
from plots import plot_learning_curves, plot_confusion_matrix
from mydatasets import calculate_num_features, VisitSequenceWithLabelDataset, visit_collate_fn
from mymodels import MyVariableRNN
import numpy as np

# Set a correct path to the data files that you preprocessed
PATH_TRAIN_SEQS = "../data/mortality/processed/mortality.seqs.train"
PATH_TRAIN_LABELS = "../data/mortality/processed/mortality.labels.train"
PATH_VALID_SEQS = "../data/mortality/processed/mortality.seqs.validation"
PATH_VALID_LABELS = "../data/mortality/processed/mortality.labels.validation"
PATH_TEST_SEQS = "../data/mortality/processed/mortality.seqs.test"
PATH_TEST_LABELS = "../data/mortality/processed/mortality.labels.test"
PATH_TEST_IDS = "../data/mortality/processed/mortality.ids.test"
PATH_OUTPUT = "../output/mortality/"
os.makedirs(PATH_OUTPUT, exist_ok=True)

NUM_EPOCHS = 10
BATCH_SIZE = 32
USE_CUDA = False  # Set 'True' if you want to use GPU
NUM_WORKERS = 0

device = torch.device("cuda" if torch.cuda.is_available() and USE_CUDA else "cpu")
torch.manual_seed(1)
if device == "cuda":
	torch.backends.cudnn.deterministic = True
	torch.backends.cudnn.benchmark = False

# Data loading
print('===> Loading entire datasets')
train_seqs = pickle.load(open(PATH_TRAIN_SEQS, 'rb'))
train_labels = pickle.load(open(PATH_TRAIN_LABELS, 'rb'))
valid_seqs = pickle.load(open(PATH_VALID_SEQS, 'rb'))
valid_labels = pickle.load(open(PATH_VALID_LABELS, 'rb'))
test_seqs = pickle.load(open(PATH_TEST_SEQS, 'rb'))
test_labels = pickle.load(open(PATH_TEST_LABELS, 'rb'))

num_features = calculate_num_features(train_seqs)

train_dataset = VisitSequenceWithLabelDataset(train_seqs, train_labels, num_features)
valid_dataset = VisitSequenceWithLabelDataset(valid_seqs, valid_labels, num_features)
test_dataset = VisitSequenceWithLabelDataset(test_seqs, test_labels, num_features)

train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
valid_loader = DataLoader(dataset=valid_dataset, batch_size=BATCH_SIZE, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)
# batch_size for the test set should be 1 to avoid sorting each mini-batch which breaks the connection with patient IDs
test_loader = DataLoader(dataset=test_dataset, batch_size=1, shuffle=False, collate_fn=visit_collate_fn, num_workers=NUM_WORKERS)

model = MyVariableRNN(num_features)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

model.to(device)
criterion.to(device)

best_val_acc = 0.0
train_losses, train_accuracies = [], []
valid_losses, valid_accuracies = [], []
for epoch in range(NUM_EPOCHS):
	train_loss, train_accuracy = train(model, device, train_loader, criterion, optimizer, epoch)
	valid_loss, valid_accuracy, valid_results = evaluate(model, device, valid_loader, criterion)

	train_losses.append(train_loss)
	valid_losses.append(valid_loss)

	train_accuracies.append(train_accuracy)
	valid_accuracies.append(valid_accuracy)

	is_best = valid_accuracy > best_val_acc  # let's keep the model that has the best accuracy, but you can also use another metric.
	if is_best:
		best_val_acc = valid_accuracy
		torch.save(model, os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))


best_model = torch.load(os.path.join(PATH_OUTPUT, "MyVariableRNN.pth"))
# TODO: For your report, try to make plots similar to those in the previous task.
# TODO: You may use the validation set in case you cannot use the test set.
class_names = ['Alive', 'Dead']
test_loss, test_accuracy, test_results = evaluate(best_model, device, valid_loader, criterion)
plot_learning_curves(train_losses, valid_losses, train_accuracies, valid_accuracies)
plot_confusion_matrix(test_results, class_names)

# TODO: Complete predict_mortality
def predict_mortality(model, device, data_loader):
	model.eval()
	# TODO: Evaluate the data (from data_loader) using model,
	# TODO: return a List of probabilities
	model.eval()
	probas=[]
	with torch.no_grad():
		for i, (input, target) in enumerate(data_loader):
			if isinstance(input, tuple):
				input = tuple([i.to(device) if type(i) == torch.Tensor else i for i in input])
			else:
				input = input.to(device)
			m1 = model(input)
			m2 = nn.Softmax(dim=1)
			m3 = m2(m1)
			m3 = np.array(m3)
			m4 = list(m3[:, 1])
			probas.append(m4[0])
	return probas


test_prob = predict_mortality(best_model, device, test_loader)
test_id = pickle.load(open(PATH_TEST_IDS, "rb"))
make_kaggle_submission(test_id, test_prob, PATH_OUTPUT)


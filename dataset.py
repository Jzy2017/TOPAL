import os
import torch
from torch.utils.data import Dataset
import numpy as np
import random
import h5py
import cv2
# v i
# o u
# 
class TrainDataSet(Dataset):
	def __init__(self, dataset=None):
		super(TrainDataSet, self).__init__()
		self.source_data = []
		
		
		data = h5py.File(dataset,'r')
		self.data = data['data'][:]
		np.random.shuffle(self.data)


	def __len__(self):
		return self.data.shape[0]

	def __getitem__(self, idx):
		self.id = idx
		traindata = self.data[self.id]
		# traindata = (traindata-0.5)/0.5
		return traindata
			


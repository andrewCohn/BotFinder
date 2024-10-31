"""
TODOC
"""

import datetime

"""
Goal: Serialize the following for use in model training:

MODEL_NAME -> string 
ACCURACY -> float
PRECISION -> float
F1
MICRO F1
MACRO F1
TRAIN_TIME
HYPERPARAMETERS 
AUTHOR
NOTES


A:
model_name (just a string)
performance {train time, test time, [f1/precision/recall]_[bot/human]}
configuration {seed + ALL hyper parameters maybe as JSON encoded dictionary} 
misc {notes, author, etc}
"""


class ModelResults:
	def __init__(self, model_name: str, author: str):
		self.model_name = model_name
		self.author = author

		self.training_start_time = 0
		self.training_stop_time = 0


	def begin_training(self):
		# self.training_start_time = os.time();
		pass


	def write(self, destination_file: str):
		pass


res = ModelResults("model", "b")

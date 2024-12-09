"""
A file including the @ModelResults class.
"""

import csv
import os.path
from datetime import datetime
from sklearn.metrics import classification_report

class ModelResults:
	"""
	Class responsible for storing and serializing the performance and training 
	information about a pytorch model.
	"""

	def __init__(self, model_name: str, author: str, seed: int):
		"""
		Constructs a new class for recording the performance of a model.
		@model_name: The name of this model being trained/evaluated.
		@author: Name of the models creator. Serialized.
		@seed: The seed that all random number generators have been initalized 
			   to.  
		"""
		self.model_name = model_name
		self.author = author

		self.performance = {}
		self.configuration = {}

		self.training_timestamp = 0
		self.testing_timestamp = 0

		self.duration_training = 0
		self.duration_testing = 0

		self.record_hyperparameter("seed", seed)


	def record_training_start(self):
		"""
		Records the time at which model training begins. Call right before 
		starting training of the model.
		"""
		self.training_timestamp = datetime.now().timestamp()


	def record_training_stop(self):
		"""
		Records the time at which model training completes. Call right after
		training of the model completes.
		"""
		duration = datetime.now().timestamp() - self.training_timestamp
		self.duration_training = duration


	def record_testing_start(self):
		"""
		Records the time at which model testing begins. Call right before 
		starting testing of the model.
		"""
		self.testing_timestamp = datetime.now().timestamp()


	def record_testing_stop(self):
		"""
		Records the time at which model testing completes. Call right after
		testing of the model completes.
		"""
		duration = datetime.now().timestamp() - self.testing_timestamp
		self.duration_testing = duration


	def record_performance(self, y_true, y_pred, target_names):
		"""
		Records the performance of a model after training.

		@y_true: Ground truth labels.
		@y_pred: Predicted labels.
		@target_names: Array of class names.
		"""
		report = classification_report(y_true, y_pred, 
								 target_names=target_names,
								 digits=6,
								 output_dict=True,
								 zero_division=0
								 )
		self.performance = dict(report)
		
	
	def record_hyperparameter(self, name: str, value):
		"""
		Records information about a single hyperparameter.
		
		@name: Name of the hyperparameter. As a string.
		@value: Value of this hyperparameter. Any type.
		"""
		self.configuration[name] = value


	def record_hyperparameters(self, hyperparameters: dict):
		"""
		Records information about a set of single hyperparameter.

		@hyperparameters: Dictionary of hyperparameter names (strings) to their
						  values.
		"""
		self.configuration.update(hyperparameters)


	def write(self, destination_file: str):
		"""
		Writes out this classes data to the disk as a CSV. If the file already
		exists, will append rows to existing file.

		@destination_file: Path to the file to write to.
		"""

		# ensure correct file endings.
		destination_file = destination_file.replace(".csv", "")
		destination_file+=".csv"

		if os.path.isfile(destination_file):
			with open(destination_file, "a", newline = '') as file:
				writer = csv.writer(file)
				writer.writerow(self._get_row())
		else:
			with open(destination_file, "w", newline = '') as file:
				writer = csv.writer(file)
				writer.writerow(self._get_header())
				writer.writerow(self._get_row())
		

	def _get_header(self) -> list:
		"""
		Returns the header of the output CSV file.
		"""
		return [
				"model", "timestamp", "seed", "accuracy", 
			
				"f1_macro", "f1_micro", "precision_macro", "precision_micro", 
				"recall_macro", "recall_micro", 

				"duration_training", "duration_testing", 

				"performance_full", "hyperparameters_full",
				"author", "notes"
			  ]


	def _get_row(self) -> list:
		"""
		Returns the row data for the current state of the class.
		"""
		return	[
				self.model_name,
				datetime.now().timestamp(),
				self.configuration["seed"],
				self.performance["accuracy"],
				self.performance["macro avg"]["f1-score"],
				self.performance["weighted avg"]["f1-score"],
				self.performance["macro avg"]["precision"],
				self.performance["weighted avg"]["precision"],
				self.performance["macro avg"]["recall"],
				self.performance["weighted avg"]["recall"],
				self.duration_training,
				self.duration_testing,
				self.performance,
				self.configuration,
				self.author,
				"",
				]



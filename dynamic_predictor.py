
from lightwood.encoder import DatetimeEncoder
from lightwood.data import cleaner
from lightwood.data import splitter
from lightwood.model import LightGBMMixer
from lightwood.encoder import CategoricalAutoEncoder
from lightwood.model import Nn
from lightwood.encoder import NumericEncoder
from lightwood.ensemble import BestOf
import pandas as pd
from mindsdb_datasources import DataSource
import torch
import numpy as np
from lightwood.helpers.seed import seed

class Predictor():
	def __init__(self):
		seed()
		self.target = 'income'

	def learn(self, data: DataSource) -> None:
		# Build a Graph from the JSON
		# Using eval is a bit ugly and we could replace it with factories, personally I'm against this, as it ads pointless complexity
		self.encoders = {
'age':NumericEncoder()
,'workclass':CategoricalAutoEncoder()
,'fnlwgt':NumericEncoder()
,'education':DatetimeEncoder()
,'educational-num':CategoricalAutoEncoder()
,'marital-status':CategoricalAutoEncoder()
,'occupation':CategoricalAutoEncoder()
,'relationship':CategoricalAutoEncoder()
,'race':CategoricalAutoEncoder()
,'gender':CategoricalAutoEncoder()
,'capital-gain':NumericEncoder()
,'capital-loss':NumericEncoder()
,'hours-per-week':NumericEncoder()
,'native-country':CategoricalAutoEncoder()
}


		# Do all the trainining and the data cleaning/processing
		data = cleaner(data)
		folds = splitter(data, 10)
		nfolds = len(data)

		for col_name, encoder in self.encoders.items():
			if encoder.uses_folds:
				encoder.prepare([x[col_name] for x in folds[0:nfolds]])
			else:
				encoder.prepare(pd.concat(folds[0:nfolds])[col_name])

		encoded_folds = lightwood.encode(self.encoders, folds)

		self.models = [Nn(), LightGBMMixer()]
		for model in self.models:
			model.fit(encoded_data[0:nfolds], folds[0:nfolds])

		self.ensemble = BestOf(self.models, encoded_data[nfolds], data[nfolds])

		# Add back when analysis works
		#self.confidence_model, self.predictor_analysis = model_analyzer(self.ensemble, encoded_data[nfolds], data[nfolds])

	def predict(self, data: DataSource) -> pd.DataFrame:
		encoded_data = lightwood.encode(self.encoders, [data])[0]
		df = self.ensemble.predict(encoded_data)
		return df
	
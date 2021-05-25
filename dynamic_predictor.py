
from lightwood.encoder import DatetimeEncoder
from lightwood.model import LightGBM
from lightwood.data import cleaner
from lightwood.encoder import NumericEncoder
from lightwood.analysis import model_analyzer
from lightwood.data import splitter
from lightwood.model import Nn
from lightwood.encoder import CategoricalAutoEncoder
from lightwood.ensemble import BestOf

class Predictor():
	def __init__(self):
		self.seed()
		self.target = income

	def seed(self):
		torch.manual_seed(66)
		torch.backends.cudnn.deterministic = True
		torch.backends.cudnn.benchmark = False
		np.random.seed(66)
		random.seed(66)

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

		self.model = self.lightwood_config['output']['model']

		# Do all the trainining and the data cleaning/processing
		data = cleaner(data)
		data = splitter(data)
		nfolds = len(data)

		for encoder in self.encoders.values():
			self.encoders.fit(data[0:nfolds])

		encoded_data = lightwood.encode(self.encoders, data)

		self.models = [Nn(), LightGBM()]
		for model in self.models:
			model.fit(encoded_data[0:nfolds], data[0:nfolds])

		self.ensemble = BestOf(self.models)

		self.confidence_model, self.predictor_analysis = model_analyzer(self.ensemble, encoded_data[nfolds], data[nfolds])

	def predict(self, data: DataSource) -> pd.DataFrame:
		encoded_data = lightwood.encode(self.encoders, data)
		df = self.ensemble(encoded_data)
		return df
	
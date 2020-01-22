from sklearn.feature_extraction.text import TfidfVectorizer

from lightwood.constants.lightwood import ENCODER_AIM

class TfidfEncoder:
    def __init__(self, is_target=False, aim=ENCODER_AIM.BALANCE):
        self._prepared = False
        self.aim = aim
        self._pytorch_wrapper = torch.FloatTensor

    def prepare_encoder(self, priming_data, training_data=None):
        if self.aim = ENCODER_AIM.SPEED:
            ngram_range = (1,3)
            max_features = 5000
        if self.aim = ENCODER_AIM.BALANCE:
            ngram_range = (1,5)
            max_features = 50000
        if self.aim = ENCODER_AIM.ACCURACY:
            ngram_range = (1,8)
            max_features = None

        self.tfidf_vectorizer = TfidfVectorizer(ngram_range=ngram_range, max_features=max_features)
        self.tfidf_vectorizer.fit_transform(priming_data)

    def encode(self, column_data):
        transformed_data = self.tfidf_vectorizer.transform(column_data)
        print(transformed_data)
        return self._pytorch_wrapper(transformed_data)

    def decode(self, encoded_values_tensor, max_length = 100):
        raise Exception('This encoder is not bi-directional')


if __name__ == "__main__":
    random.seed(2)
    text = [''.join(random.choices(string.ascii_uppercase + string.digits + ['\n', ' ', '\t'], k=random.randint(5,500))) for x in range(1000)]

    enc = TfidfEncoder()
    enc.prepare_encoder(text)
    encoded_data = enc.encode(text)

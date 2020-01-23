from lightwood.encoders.text.rnn import RnnEncoder
from lightwood.encoders.text.infersent import InferSentEncoder
from lightwood.encoders.text.distilbert import DistilBertEncoder
from lightwood.encoders.text.tfidf import TfidfEncoder

#default = TfidfEncoder
default = DistilBertEncoder
#default = InferSentEncoder

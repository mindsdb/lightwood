import torch
from lightwood.config.config import CONFIG
from pytorch_transformers import DistilBertModel, DistilBertForSequenceClassification, DistilBertTokenizer, DistilBertForQuestionAnswering, GPT2Tokenizer, GPT2LMHeadModel, DistilBertForMaskedLM


class DistilBertEncoder:
    def __init__(self, is_target=False):
        self._tokenizer = None
        self._model = None
        self._pytorch_wrapper = torch.FloatTensor
        self._max_len = None
        self._max_ele = None
        self._prepared = False

    def prepare_encoder(self, priming_data):
        print(CONFIG.CACHE_ENCODED_DATA)
        if self._prepared:
            raise Exception('You can only call "prepare_encoder" once for a given encoder.')

        print('Priming text encoder !')
        self._max_len = min(max([len(x) for x in priming_data]),768)

        self._tokenizer = DistilBertTokenizer.from_pretrained('distilbert-base-uncased')
        #self._model = DistilBertForSequenceClassification.from_pretrained('distilbert-base-uncased')
        self._model = DistilBertModel.from_pretrained('distilbert-base-uncased')

        '''
        for text in priming_data:
            for i in range(0,100):
                input = torch.tensor(self._tokenizer.encode(text[0:512])).unsqueeze(0)

                outputs = self._model(input, masked_lm_labels=input)
                loss, prediction_scores = outputs[:2]
                print(loss)

                outputs = self._model(input)
                print(len(outputs))
                #lhs = outputs[2]

                predicted_token = self._tokenizer.convert_ids_to_tokens(outputs[0])[0]
                print(predicted_token)
                loss.backward()

        tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
        model = GPT2LMHeadModel.from_pretrained('gpt2')

        for text in priming_data:
            input_ids = torch.tensor(tokenizer.encode(text[0:512])).unsqueeze(0)
            outputs = model(input_ids, labels=input_ids)
            print(tokenizer.decode(outputs, clean_up_tokenization_spaces=True, skip_special_tokens=True))
        '''
        print('Primed text encoder !')
        self._prepared = True


    def encode(self, column_data):
        encoded_representation = []
        print('Encoding sequence of length: ' + str(len(column_data)))
        with torch.no_grad():
            for text in column_data:
                if text is None:
                    text = ''
                input = torch.tensor(self._tokenizer.encode(text[:self._max_len], add_special_tokens=True)).unsqueeze(0)
                last_hidden_states = self._model(input)

                embeddings = last_hidden_states[0][:,0,:].numpy()[0]

                encoded_representation.append(embeddings)

        return self._pytorch_wrapper(encoded_representation)

    def decode(self, encoded_values_tensor, max_length = 100):
        # When test is an output... a bit trickier to handle this case, thinking on it
        pass

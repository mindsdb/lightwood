import unittest
from lightwood.encoders.text import InferSentEncoder


class TestInferSentEncoder(unittest.TestCase):
    def test_encode_and_decode(self):
        return

        #TODO: check _download_embeddings_file, it will download files after each run which takes 5-10min
        sentences = ["Everyone really likes the newest benefits",
                    "The Government Executive articles housed on the website are not able to be searched",
                    "Most of Mrinal Sen 's work can be found in European collections . ",
                    "Would you rise up and defeaat all evil lords in the town ? ",
                    None
                    ]

        encoder = InferSentEncoder()
        encoder.prepare_encoder(sentences)
        ret = encoder.encode(sentences)
        print(ret)

        ret = encoder.encode(["And they will fail to raise"])
        print(ret)

        ret = encoder.encode(["Everyone really likes the newest benefits"])
        print(ret)

import unittest

from lightwood.encoder.text.vocab import VocabularyEncoder


class TestVocabularyEncoder(unittest.TestCase):
    def test_encode_decode(self):
        self.maxDiff = None
        sentences = [
            'Consequuntur magni dolores eos qui ratione voluptatem sequi nesciunt. Neque porro quisquam est, qui dolorem ipsum quia dolor sit amet, consectetur, adipisci velit, sed quia non numquam eiu.', # noqa
            'At vero eos et accusamus et iusto odio dignissimos ducimus qui blanditiis praesentium voluptatum deleniti atque corrupti quos dolores et quas molestias excepturi sint occaecati cupiditate non provident', # noqa
            'Itaque earum rerum hic tenetur a sapiente delectus, ut aut reiciendis voluptatibus maiores alias? consequatur aut perferendis doloribus asperiores repellat.'] # noqa

        sentences = [x.lower() for x in sentences]

        encoder = VocabularyEncoder()
        encoder.prepare(sentences)

        for sentence in sentences:
            encoded = encoder.encode([sentence])
            decoded = encoder.decode(encoded)
            self.assertEqual(decoded[0], sentence)

import random
import unittest
from lightwood.encoder.text.short import ShortTextEncoder
from lightwood.helpers.text import tokenize_text

VOCAB = [
    'do', 'not', 'remember', 'men', 'pretty', 'break', 'know', 'an', 'forward', 'whose', 'plant', 'decide', 'fit', 'so',
    'connect', 'house', 'then', 'lot', 'protect', 'children', 'above', 'column', 'far', 'continue', 'which', 'discuss',
    'test', 'self', 'dream', 'prepare', 'toward', 'world', 'cold', 'subtract', 'bat', 'subject', 'evening', 'year',
    'low', 'sign', 'very', 'determine', 'sent', 'tube', 'skill', 'first', 'feed', 'fresh', 'note', 'own', 'ball',
    'shape', 'rail', 'drink', 'property', 'did', 'son', 'yet', 'shell', 'believe', 'him', 'noise', 'take', 'spot',
    'read', 'scale', 'but', 'chart', 'wrote', 'war', 'hundred', 'better', 'ease', 'show', 'suggest', 'describe', 'sat',
    'camp', 'oxygen', 'view', 'smile', 'add', 'whole', 'arrange', 'trade', 'money', 'just', 'repeat', 'truck', 'second',
    'shoe', 'wish', 'as', 'mount', 'ground', 'wrong', 'support', 'lead', 'thin', 'agree', 'slave', 'us', 'wide',
    'white', 'room', 'reach', 'are', 'hot', 'thousand', 'science', 'before', 'want', 'broad', 'soon', 'long', 'felt',
    'danger', 'offer', 'saw', 'most', 'rub', 'basic', 'call', 'cost', 'surprise', 'salt', 'office', 'done', 'any',
    'egg', 'a', 'instrument', 'east', 'oil', 'seed', 'state', 'back', 'shop', 'multiply', 'city', 'top', 'arrive',
    'ocean', 'figure', 'good', 'hat', 'claim', 'say', 'he', 'decimal', 'process', 'point', 'hope', 'hunt', 'went',
    'night', 'paragraph', 'hear', 'require', 'near', 'enter', 'collect', 'were', 'region', 'stream', 'teach', 'favor',
    'lay', 'and', 'compare', 'best', 'learn', 'act', 'practice', 'material', 'mean', 'river', 'soldier', 'control',
    'equal', 'between', 'sudden', 'result', 'spend', 'wire', 'wall', 'experiment', 'receive', 'bed', 'condition',
    'ship', 'people', 'your', 'love', 'thank', 'cut', 'who', 'planet', 'drive', 'position', 'select', 'simple',
    'temperature', 'warm', 'power', 'cotton', 'bring', 'them', 'what', 'let', 'kept', 'cross', 'degree', 'center',
    'they', 'brother', 'busy', 'animal', 'wood', 'shall', 'young', 'came', 'is', 'how', 'last', 'able', 'play', 'event',
    'village', 'two', 'many', 'wheel', 'black', 'usual', 'woman', 'will', 'never', 'trouble', 'tell', 'mind', 'dog',
    'fig', 'dollar', 'cover', 'north', 'bought', 'story', 'street', 'suffix', 'laugh', 'instant', 'town', 'rule',
    'trip', 'go', 'told', 'might', 'idea', 'supply', 'mile', 'cow', 'edge', 'rather', 'garden', 'corn', 'parent',
    'hour', 'country', 'fast', 'ten', 'print', 'locate', 'final', 'coast', 'character', 'radio', 'even', 'end', 'stead',
    'make', 'snow', 'possible', 'correct', 'boat', 'here', 'allow', 'month', 'or', 'should', 'same', 'bell', 'matter',
    'run', 'beauty', 'come', 'spread', 'held', 'consonant', 'part', 'food', 'chance', 'sugar', 'history', 'stood',
    'out', 'steam', 'half', 'been', 'draw', 'insect', 'may', 'name', 'music', 'flower', 'through', 'mass', 'map',
    'eight', 'man', 'cent', 'job', 'energy', 'look', 'dad', 'hill', 'million', 'settle', 'song', 'hit', 'does', 'hold',
    'pair', 'dress', 'side', 'cool', 'day', 'gun', 'page', 'until', 'capital', 'appear', 'voice', 'have', 'cause',
    'minute', 'wing', 'keep', 'bone', 'season', 'some', 'also', 'question', 'feel', 'seem', 'necessary', 'these', 'of',
    'was', 'against', 'window', 'don’t', 'chick', 'valley', 'green', 'probable', 'shore', 'fall', 'particular', 'case',
    'colony', 'land', 'place', 'level', 'bear', 'though', 'root', 'weight', 'branch', 'jump', 'true', 'bread', 'yard',
    'be', 'element', 'miss', 'stretch', 'heard', 'lady', 'over', 'present', 'division', 'verb', 'prove', 'ready',
    'carry', 'poem', 'silent', 'poor', 'die', 'death', 'use', 'train', 'anger', 'help', 'substance', 'shine', 'list',
    'send', 'syllable', 'thus', 'brought', 'big', 'now', 'dictionary', 'space', 'unit', 'soil', 'work', 'object',
    'board', 'roll', 'six', 'wonder', 'no', 'sit', 'clock', 'size', 'once', 'front', 'key', 'either', 'if', 'try',
    'neighbor', 'our', 'hard', 'about', 'famous', 'again', 'especially', 'wait', 'think', 'afraid', 'line', 'track',
    'quick', 'rose', 'like', 'field', 'forest', 'numeral', 'path', 'meant', 'color', 'separate', 'copy', 'nation',
    'third', 'desert', 'behind', 'dead', 'spell', 'record', 'teeth', 'lift', 'pattern', 'mountain', 'island', 'soft',
    'king', 'since', 'round', 'made', 'together', 'real', 'floor', 'travel', 'team', 'wife', 'machine', 'plane', 'fish',
    'general', 'enough', 'special', 'natural', 'value', 'join', 'light', 'tie', 'corner', 'rope', 'piece', 'quotient',
    'to', 'write', 'weather', 'old', 'each', 'least', 'provide', 'while', 'log', 'square', 'turn', 'language', 'gas',
    'body', 'method', 'home', 'similar', 'original', 'period', 'circle', 'finish', 'captain', 'fire', 'week', 'post',
    'fill', 'count', 'range', 'well', 'cloud', 'get', 'dark', 'silver', 'occur', 'burn', 'crowd', 'bird', 'double', 'I',
    'would', 'this', 'band', 'quart', 'table', 'rock', 'found', 'friend', 'sight', 'deep', 'dry', 'blood', 'touch',
    'fear', 'finger', 'plan', 'guide', 'hot', 'after', 'hair', 'tree', 'race', 'noon', 'effect', 'wild', 'took', 'hand',
    'give', 'clear', 'noun', 'please', 'do', 'art', 'stay', 'fly', 'whether', 'sell', 'lone', 'from', 'too', 'paint',
    'tire', 'loud', 'divide', 'complete', 'charge', 'left', 'milk', 'spoke', 'base', 'free', 'her', 'human', 'iron',
    'choose', 'continent', 'strange', 'segment', 'summer', 'bit', 'build', 'course', 'type', 'steel', 'press', 'great',
    'those', 'search', 'dear', 'pitch', 'perhaps', 'grand', 'industry', 'quite', 'up', 'term', 'sentence', 'high',
    'shout', 'down', 'we', 'all', 'huge', 'walk', 'solve', 'excite', 'mark', 'pick', 'three', 'other', 'rest', 'law',
    'wind', 'difficult', 'gold', 'populate', 'proper', 'knew', 'one', 'fun', 'seven', 'happen', 'cell', 'throw',
    'motion', 'atom', 'expect', 'live', 'rain', 'stone', 'grass', 'sleep', 'early', 'short', 'always', 'gentle',
    'father', 'cook', 'mouth', 'the', 'inch', 'ago', 'store', 'smell', 'observe', 'magnet', 'leave', 'heart', 'little',
    'written', 'sharp', 'box', 'talk', 'broke', 'score', 'wave', 'bar', 'off', 'century', 'fruit', 'class', 'card',
    'way', 'meat', 'late', 'surface', 'bottom', 'family', 'wash', 'guess', 'catch', 'red', 'fact', 'move', 'visit',
    'port', 'set', 'eat', 'thing', 'when', 'start', 'fair', 'example', 'head', 'four', 'grow', 'earth', 'win', 'gone',
    'where', 'nothing', 'open', 'quiet', 'by', 'can', 'clean', 'group', 'ear', 'moment', 'game', 'close', 'morning',
    'reply', 'straight', 'nature', 'often', 'develop', 'west', 'thick', 'twenty', 'feet', 'led', 'total', 'she', 'slip',
    'create', 'pull', 'system', 'need', 'party', 'sail', 'doctor', 'length', 'has', 'change', 'consider', 'study',
    'rise', 'star', 'operate', 'certain', 'electric', 'leg', 'sheet', 'kind', 'major', 'chair', 'born', 'chord',
    'order', 'ride', 'could', 'word', 'modern', 'face', 'find', 'push', 'me', 'horse', 'differ', 'ever', 'nose',
    'else', 'on', 'spring', 'solution', 'molecule', 'door', 'right', 'enemy', 'symbol', 'paper', 'during', 'sure',
    'tool', 'fight', 'joy', 'stick', 'yes', 'notice', 'station', 'area', 'tall', 'string', 'design', 'tone', 'sky',
    'indicate', 'foot', 'pose', 'success', 'mine', 'air', 'engine', 'listen', 'distant', 'tail', 'invent', 'at']


def generate_sentences(min_, max_, vocab_size):
    vocab_sample = random.sample(VOCAB, vocab_size)
    return [' '.join(random.sample(vocab_sample, random.randint(min_, max_))) for _ in range(200)]


class TestShortTextEncoder(unittest.TestCase):
    def test_get_tokens(self):
        sentences = ['hello, world!', ' !hello! world!!,..#', '#hello!world']
        for sent in sentences:
            assert tokenize_text(sent) == ['hello', 'world']

        assert tokenize_text("don't wouldn't") == ['do', 'not', 'would', 'not']

    def test_smallvocab_target_auto_mode(self):
        priming_data = generate_sentences(2, 6, vocab_size=99)
        test_data = random.sample(priming_data, len(priming_data) // 5)

        enc = ShortTextEncoder(is_target=True)
        enc.prepare(priming_data)

        assert enc.is_target is True

        # _combine is expected to be 'concat' when is_target is True
        assert enc._mode == 'concat'

        encoded_data = enc.encode(test_data)
        decoded_data = enc.decode(encoded_data)

        assert len(test_data) == len(encoded_data) == len(decoded_data)

        for x_sent, y_sent in zip(
            [' '.join(tokenize_text(x)) for x in test_data],
            [' '.join(x) for x in decoded_data]
        ):
            assert x_sent == y_sent

    def test_non_smallvocab_target_auto_mode(self):
        priming_data = generate_sentences(2, 6, vocab_size=800)
        test_data = random.sample(priming_data, len(priming_data) // 5)

        enc = ShortTextEncoder(is_target=True)
        enc.prepare(priming_data)

        assert enc.is_target is True

        # _combine is expected to be 'concat' when is_target is True
        assert enc._mode == 'concat'

        encoded_data = enc.encode(test_data)
        decoded_data = enc.decode(encoded_data)

        assert len(test_data) == len(encoded_data) == len(decoded_data)

        for x_sent, y_sent in zip(
            [' '.join(tokenize_text(x)) for x in test_data],
            [' '.join(x) for x in decoded_data]
        ):
            assert x_sent == y_sent

    def test_smallvocab_non_target_auto_mode(self):
        priming_data = generate_sentences(2, 6, vocab_size=50)
        test_data = random.sample(priming_data, len(priming_data) // 5)

        enc = ShortTextEncoder(is_target=False)
        enc.prepare(priming_data)

        assert enc.is_target is False

        # _combine is expected to be 'mean' when is_target is False
        assert enc._mode == 'mean'

        encoded_data = enc.encode(test_data)

        assert len(test_data) == len(encoded_data)

        with self.assertRaises(ValueError):
            enc.decode(encoded_data)

    def test_non_smallvocab_non_target_auto_mode(self):
        priming_data = generate_sentences(2, 6, vocab_size=101)
        test_data = random.sample(priming_data, len(priming_data) // 5)

        enc = ShortTextEncoder(is_target=False)
        enc.prepare(priming_data)

        assert enc.is_target is False

        # _combine is expected to be 'mean' when is_target is False
        assert enc._mode == 'mean'

        encoded_data = enc.encode(test_data)

        assert len(test_data) == len(encoded_data)

        with self.assertRaises(ValueError):
            enc.decode(encoded_data)

    def test_smallvocab_non_target_manual_mode(self):
        priming_data = generate_sentences(2, 6, vocab_size=99)
        test_data = random.sample(priming_data, len(priming_data) // 5)

        enc = ShortTextEncoder(is_target=False, mode='concat')
        enc.prepare(priming_data)

        assert enc.is_target is False
        assert enc._mode == 'concat'

        encoded_data = enc.encode(test_data)
        decoded_data = enc.decode(encoded_data)

        assert len(test_data) == len(encoded_data) == len(decoded_data)

        for x_sent, y_sent in zip(
            [' '.join(tokenize_text(x)) for x in test_data],
            [' '.join(x) for x in decoded_data]
        ):
            assert x_sent == y_sent

    def test_non_smallvocab_non_target_manual_mode(self):
        priming_data = generate_sentences(2, 6, vocab_size=101)
        test_data = random.sample(priming_data, len(priming_data) // 5)

        enc = ShortTextEncoder(is_target=False, mode='concat')
        enc.prepare(priming_data)

        assert enc.is_target is False
        assert enc._mode == 'concat'

        encoded_data = enc.encode(test_data)
        decoded_data = enc.decode(encoded_data)

        assert len(test_data) == len(encoded_data) == len(decoded_data)

        for x_sent, y_sent in zip(
            [' '.join(tokenize_text(x)) for x in test_data],
            [' '.join(x) for x in decoded_data]
        ):
            assert x_sent == y_sent

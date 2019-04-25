import logging

from lightwood.encoders.image.helpers.nn import NnEncoderHelper


class NnAutoEncoder:

    def __init__(self, images):
        self._model = NnEncoderHelper(images)

    def encode(self, images):
        """
          Encode all the images from the list of paths(to images)

        :param images: List of images paths
        :return: a torch.floatTensor
        """
        if not self._model:
            logging.error("No model to encode, please train the model")

        return self._model.encode(images)

    def decode(self, encoded_values_tensor):
        """
         Decoded the encoded list of image tensors and write the decoded images to "decoded" folder

        :param encoded_values_tensor: List of encoded images tensors
        :return: a torch.floatTensor
        """
        if not self._model:
            logging.error("No model to decode, please train the model")
        return self._model.decode(encoded_values_tensor)

    def train(self, images):
        """
        :param images: List of images paths
        """
        self._model = NnEncoderHelper(images)


# only run the test if this file is called from debugger
if __name__ == "__main__":
    images = ['test_data/cat.jpg', 'test_data/cat2.jpg', 'test_data/catdog.jpg']
    encoder = NnAutoEncoder(images)

    images = ['test_data/cat.jpg', 'test_data/cat2.jpg']
    encoded_data = encoder.encode(images)
    print(encoded_data)

    # decoded images will be stored under decoded folder
    decoded_data = encoder.decode(encoded_data)
    print(decoded_data)



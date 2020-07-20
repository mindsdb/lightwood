import logging
import torch
from lightwood.helpers.device import get_devices
from lightwood.encoders.time_series import RnnEncoder
from lightwood.encoders.time_series.helpers.rnn_helpers import tensor_from_series


def test_ts(encoder, queries, answers, params):
    """ Minimal testing suite for time series encoder-decoder.
    :param encoder: RnnEncoder instance to test
    :param queries: list of multi-dimensional time series to encode, predict, and decode with
                    [[[data_ts1_dim_1], [data_ts1_dim_2]], ...] with data_ts_dim_i a string
    :param answers: list of correct predictions for each query [[ans_query_1], ...], where
                    ans_query_i is a list of numbers
    :param params: dictionary with configuration parameters
    :return:
    """
    forecasts, decoded = list(), list()

    # predict and decode
    for query in queries:
        encoded_data, preds = encoder.encode([query], get_next_count=params['pred_qty'])
        decoded_data = encoder.decode(encoded_data).tolist()

        if params['ts_n_dims'] > 1:
            decoded_data = decoded_data[0]

        preds = torch.reshape(preds, (1, params['ts_n_dims']))
        forecasts.append(preds.tolist())
        decoded.append(decoded_data)

    # evaluate query error
    for i in range(len(queries)):
        logging.info("\t[Query {}]".format(i))
        query = queries[i]
        preds = forecasts[i]
        dec = [elt[0:len(query[0][0].split(" "))] for elt in decoded[i]]  # truncate to original query length
        ans = answers[i]
        aggregate = [preds, dec]

        # check prediction
        pred_test = True
        for answer, pred in zip(ans, preds[-1]):  # compare last predicted timestep with answer
            if abs(pred - answer) > params['margin']:
                pred_test = False
                break

        # check reconstruction
        dec_test = True
        for q, v in zip(query, dec):
            fq = map(float, q[0].split(" "))
            for p, t in zip(fq, v):
                if abs(p - t) > params['margin']:
                    dec_test = False
                    break

        tests = [pred_test, dec_test]
        tests_info = ['Prediction for query [{}]:'.format(query), 'Reconstruction for query [{}]: '.format(query)]

        for info, value, passed in zip(tests_info, aggregate, tests):
            logging.info("{}[{}] {}\n{}{}".format("\t" * 2, "Pass" if passed else "Fail", info, "\t" * 6,
                                                  [[round(elt, 2) for elt in dimn] for dimn in value]))


# only run the test if this file is called from debugger
if __name__ == "__main__":
    logging.basicConfig(level=logging.DEBUG)

    # test 1: padding function
    series = [['1', '2', '3 '], ['2', '3'], ['3', '4', '5', '6'], [' 4', '5', '6']]
    target = [[1.0, 2.0, 3.0, 4.0, 0.0], [2.0, 3.0, 4.0, 5.0, 0.0], [3.0, 0.0, 5.0, 6.0, 0.0]]
    result = tensor_from_series(series, get_devices()[0], n_dims=5, pad_value=0.0, max_len=3).tolist()[0]
    assert result == target

    # test 2: overfit single multi dimensional time series
    logging.info(" [Test] Multi-dimensional time series overfit")
    series = [[['1', '2', '3', '4', '5', '6'], ['2', '3', '4', '5', '6', '7'],
               ['3', '4', '5', '6', '7', '8'], ['4', '5', '6', '7', '8', '9']]]
    data = 100 * series
    n_dims = max([len(q) for q in data])

    params = {'max_ts': 6,
              'hidden_size': 10,
              'batch_size': 1,
              'dropout': 0.1,
              'ts_n_dims': n_dims,
              'train_iters': 20,
              'margin': 0.5,  # error tolerance
              'feedback_fn': lambda x: logging.info(x),
              'pred_qty': 1}

    encoder = RnnEncoder(encoded_vector_size=params['hidden_size'], train_iters=params['train_iters'],
                         ts_n_dims=params['ts_n_dims'], max_timesteps=params['max_ts'])

    encoder.prepare_encoder(data, feedback_hoop_function=params['feedback_fn'], batch_size=params['batch_size'])

    queries = [[['1', '2', '3'], ['2', '3', '4'], ['3', '4', '5'], ['4', '5', '6']]]
    answers = [[4, 5, 6, 7]]
    test_ts(encoder, queries, answers, params)

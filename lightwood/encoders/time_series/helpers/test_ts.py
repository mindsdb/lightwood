import logging
from lightwood.encoders.time_series.rnn import RnnEncoder


def test_ts(encoder, queries, answers, params):
    """ Minimal testing suite for time series encoder-decoder.
    :param encoder: RnnEncoder instance to test
    :param queries: list of multi-dimensional time series to encode, predict, and decode with
                    [[[data_ts1_dim1], [data_ts1_dim2]], ...]
    :param params: dictionary with configuration parameters
    :return:
    """
    forecasts, decoded = list(), list()

    # predict and decode
    for query in queries:
        encoded_data, preds = encoder.encode([query], get_next_count=params['pred_qty'])
        decoded_data = encoder.decode(encoded_data).tolist()[0]

        # shape preds to (timesteps, n_dims)
        preds = preds.squeeze()
        if len(list(preds.shape)) == 1:
            preds = preds.unsqueeze(dim=0)

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

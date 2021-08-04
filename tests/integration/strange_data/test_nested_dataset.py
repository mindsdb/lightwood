from copy import deepcopy
import unittest
import pandas as pd
from mindsdb_datasources.datasources.file_ds import FileDS
from lightwood.api import predictor_from_problem


class TestNestedDataset(unittest.TestCase):
    def test_nested_json(self):
        sample_json = {
            "Airport": {
                "Code": "ATL",
                "Name": "Atlanta, GA: Hartsfield-Jackson Atlanta International"
            },
            "Time": {
                "Label": "2003/06",
                "Month": 6,
                "Month Name": "June",
                "Year": 2003
            },
            "Statistics": {
                "# of Delays": {
                    "Carrier": 1009,
                    "Late Aircraft": 1275,
                    "National Aviation System": 3217,
                    "Security": 17,
                    "Weather": 328
                },
                "Carriers": {
                    "Names": "American Airlines Inc.,JetBlue Airways,Continental Air Lines Inc.,Delta Air Lines Inc.,Atlantic Southeast Airlines,AirTran Airways Corporation,America West Airlines Inc.,Northwest Airlines Inc.,ExpressJet Airlines Inc.,United Air Lines Inc.,US Airways Inc.",
                    "Total": 11
                },
                "Flights": {
                    "Cancelled": 216,
                    "Delayed": 5843,
                    "Diverted": 27,
                    "On Time": 23974,
                    "Total": 30060
                },
                "Minutes Delayed": {
                    "Carrier": 61606,
                    "Late Aircraft": 68335,
                    "National Aviation System": 118831,
                    "Security": 518,
                    "Total": 268764,
                    "Weather": 19474
                }
            }
        }

        expected_columns = ['Statistics.Minutes Delayed.Security', 'Statistics.# of Delays.Security',
                            'Statistics.# of Delays.Weather', 'Statistics.Minutes Delayed.Late Aircraft',
                            'Statistics.Flights.Total', 'Statistics.Flights.Diverted', 'Time.Month Name',
                            'Statistics.Flights.Delayed', 'Time.Month', 'Statistics.# of Delays.Carrier',
                            'Statistics.Flights.On Time', 'Time.Label', 'Airport.Name',
                            'Statistics.Minutes Delayed.National Aviation System', 'Airport.Code',
                            'Statistics.Carriers.Total', 'Statistics.Minutes Delayed.Weather', 'Time.Year',
                            'Statistics.# of Delays.National Aviation System', 'Statistics.Flights.Cancelled',
                            'Statistics.# of Delays.Late Aircraft', 'Statistics.Carriers.Names',
                            'Statistics.Minutes Delayed.Total', 'Statistics.Minutes Delayed.Carrier']

        ds = FileDS('https://raw.githubusercontent.com/mindsdb/benchmarks/main/datasets/airline_delays/data.json')
        pred = predictor_from_problem(ds, {'target': 'Statistics.Flights.Delayed'})

        for col in expected_columns:
            assert col in ds.df.columns
        pred.learn(ds.df)

        model_data = pred.model_analysis
        print(model_data)
        # TODO: Run some check here once we define the predictor analysis

        predictions = pred.predict(pd.json_normalize(sample_json))
        for v in predictions['prediction']:
            assert isinstance(v, int)

        predictions = pred.predict(
            FileDS('https://raw.githubusercontent.com/mindsdb/benchmarks/main/datasets/airline_delays/data.json').df)
        for v in predictions['prediction']:
            assert isinstance(v, int)

        predictions = pred.predict(pd.json_normalize(sample_json))
        for v in predictions['prediction']:
            assert isinstance(v, int)

        missing_json = deepcopy(sample_json)
        del missing_json['Statistics']['Minutes Delayed']['Weather']
        del missing_json['Time']

        extra_json = deepcopy(missing_json)
        extra_json['A'] = 'bsdbsd'
        extra_json['B'] = {
            'C': {
                'D': 453,
                'E': [1, 2, 3, 4]
            },
            'F': [0, 0, 0]
        }
        extra_json['X'] = [1, 2, 3, 5, 6, 7, None, None, 'bar']

        dot_json = pd.json_normalize(sample_json)
        extra_json = pd.json_normalize(extra_json)
        missing_json = pd.json_normalize(missing_json)

        predictions = pred.predict(dot_json)
        for v in predictions['prediction']:
            assert isinstance(v, int)

        predictions = pred.predict(extra_json)
        for v in predictions['prediction']:
            assert isinstance(v, int)

        predictions = pred.predict(missing_json)
        for v in predictions['prediction']:
            assert isinstance(v, int)

        predictions = pred.predict(pd.DataFrame([missing_json, missing_json, extra_json, sample_json]))
        for v in predictions['prediction']:
            assert isinstance(v, int)

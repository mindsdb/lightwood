The `api` functionalities should not require any unit tests since by definition they should be exhaustively tested as part of the integration testing.

If the `api` has a functionality that is not used as part of our integration testing, this means we need better integration test coverage *or* that said part should not be part of lightwood's API.
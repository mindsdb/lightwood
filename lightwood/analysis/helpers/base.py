from typing import Tuple, Dict, Optional

import pandas as pd


class BaseAnalysisBlock:
    """Class to be inherited by any analysis/explainer block."""
    def __init__(self,
                 deps: Optional[list] = None
                 ):

        self.is_prepared = False
        self.dependencies = deps  # can be parallelized when there are no dependencies

    def analyze(self, info: Dict[str, object]) -> Dict[str, object]:
        # @TODO: figure out signature, how to pass only required args
        """
        This method is called during the analysis phase. Receives and returns
        a dictionary to which any information computed here should be added.
        """
        raise NotImplementedError

    def explain(self) -> Tuple[pd.DataFrame, Dict[str, object]]:
        # @TODO: figure out signature, how to pass only required args
        """
        This method is called during model inference. Additional explanations can be
        at an instance level (row-wise) or global. For the former, return a data frame
        with any new insights. For the latter, a dictionary is required.

        Depending on the nature of the block, this method might demand `self.is_prepared==True`.
        """
        raise NotImplementedError

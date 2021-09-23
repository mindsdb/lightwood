from typing import Tuple, List, Dict, Optional

import pandas as pd


class BaseAnalysisBlock:
    """Class to be inherited by any analysis/explainer block."""
    def __init__(self,
                 deps: Optional[List] = []
                 ):

        self.dependencies = deps  # can be parallelized when there are no dependencies

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        """
        This method should be called once during the analysis phase, or not called at all.
        It computes any information that the block may either output once during analysis, or need later during
        inference when `.explain()` is called.

        :param info: Dictionary where any new information or objects are added.
        """
        raise NotImplementedError

    def explain(self, insights: pd.DataFrame, **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        This method is called during model inference. Additional explanations can be
        at an instance level (row-wise) or global. For the former, return a data frame
        with any new insights. For the latter, a dictionary is required.

        Depending on the nature of the block, this method might demand `self.is_prepared==True`.

        :param insights: dataframe with previously computed row-level explanations.
        :returns:
            - insights: modified input dataframe with any new row insights added here.
            - global_insights: dictionary with any explanations that concern all predicted instances.
        """
        raise NotImplementedError

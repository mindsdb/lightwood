from typing import Tuple, Dict, Optional

import pandas as pd
from lightwood.helpers.log import log


class BaseAnalysisBlock:
    """Class to be inherited by any analysis/explainer block."""
    def __init__(self,
                 deps: Optional[Tuple] = ()
                 ):

        self.dependencies = deps  # can be parallelized when there are no dependencies @TODO enforce

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        """
        This method should be called once during the analysis phase, or not called at all.
        It computes any information that the block may either output to the model analysis object,
        or use at inference time when `.explain()` is called (in this case, make sure all needed
        objects are added to the runtime analyzer so that `.explain()` can access them).

        :param info: Dictionary where any new information or objects are added. The next analysis block will use
        the output of the previous block as a starting point.
        :param kwargs: Dictionary with named variables from either the core analysis or the rest of the prediction
        pipeline.
        """
        log.info(f"{self.__class__.__name__}.analyze() has not been implemented, no modifications will be done to the model analysis.")  # noqa
        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """
        This method should be called once during the explaining phase at inference time, or not called at all.
        Additional explanations can be at an instance level (row-wise) or global.
        For the former, return a data frame with any new insights. For the latter, a dictionary is required.

        :param row_insights: dataframe with previously computed row-level explanations.
        :param global_insights: dict() with any explanations that concern all predicted instances or the model itself.

        :returns:
            - row_insights: modified input dataframe with any new row insights added here.
            - global_insights: dict() with any explanations that concern all predicted instances or the model itself.
        """
        log.info(f"{self.__class__.__name__}.explain() has not been implemented, no modifications will be done to the data insights.")  # noqa
        return row_insights, global_insights

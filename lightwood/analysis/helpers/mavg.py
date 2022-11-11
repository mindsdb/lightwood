from typing import Optional, Tuple, Dict
import pandas as pd
# import numpy as np

# from sklearn.preprocessing import OrdinalEncoder

from lightwood.analysis.base import BaseAnalysisBlock
# from lightwood.helpers.log import log
# from lightwood.helpers.parallelism import get_nr_procs


class MAvg(BaseAnalysisBlock):
    """
    Wrapper analysis block for the 'Robbie'...
    """

    def __init__(self, deps: Optional[Tuple] = ()):
        super().__init__(deps=deps)

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        return row_insights, global_insights

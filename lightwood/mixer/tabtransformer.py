from typing import Dict, Optional

import torch
from tab_transformer_pytorch import TabTransformer

from lightwood.helpers.device import get_device_from_name
from lightwood.data.encoded_ds import EncodedDs
from lightwood.encoder.base import BaseEncoder
from lightwood.mixer.neural import Neural


class TabTransformerMixer(Neural):
    def __init__(
            self,
            stop_after: float,
            target: str,
            dtype_dict: Dict[str, str],
            target_encoder: BaseEncoder,
            fit_on_dev: bool,
            search_hyperparameters: bool,
            train_args: Optional[dict] = None
    ):
        """
        This mixer trains a TabTransformer network (FT variant), using concatenated encoder outputs for each dataset feature as input, to predict the encoded target column representation as output.
        
        Training logic is based on the Neural mixer, please refer to it for more details on each input parameter.
        """  # noqa
        self.train_args = train_args if train_args else {}
        super().__init__(
            stop_after,
            target,
            dtype_dict,
            target_encoder,
            'FTTransformer',
            False,  # fit_on_dev
            search_hyperparameters,
            n_epochs=self.train_args.get('n_epochs', None)
        )
        self.lr = self.train_args.get('lr')
        self.stable = False  # still experimental

    def _init_net(self, ds: EncodedDs):
        self.net_class = TabTransformer
        self.model = TabTransformer(
            categories=(),                                                      # unused, everything is numerical by now
            num_continuous=len(ds[0][0]),
            dim=self.train_args.get('dim', 32),
            dim_out=self.train_args.get('dim_out', len(ds[0][1])),
            depth=self.train_args.get('depth', 6),
            heads=self.train_args.get('heads', 8),
            attn_dropout=self.train_args.get('attn_dropout', 0.1),              # post-attention dropout
            ff_dropout=self.train_args.get('ff_dropout', 0.1),                  # feed forward dropout
            mlp_hidden_mults=self.train_args.get('mlp_hidden_mults', (4, 2)),   # relative multiples of each hidden dimension of the last mlp to logits  # noqa
            # mlp_act=self.train_args.get('mlp_act', nn.ReLU()),                # TODO: import str from nn activations
        )
        self.model.device = get_device_from_name('')
        self.model.to(self.model.device)

    def _net_call(self, x: torch.Tensor) -> torch.Tensor:
        x = torch.unsqueeze(x, 0) if len(x.shape) < 2 else x
        return self.model(torch.Tensor(), x)

    def fit(self, train_data: EncodedDs, dev_data: EncodedDs) -> None:
        """ Skip the usual partial_fit call at the end. """  # noqa
        self._fit(train_data, dev_data)

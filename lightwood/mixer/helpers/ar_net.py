import torch
from torch import nn
from lightwood.mixer.helpers.default_net import DefaultNet
from lightwood.helpers.torch import LightwoodAutocast


class ArNet(DefaultNet):
    """
    DefaultNet variant that adds a secondary stream (simple linear layer) with constrained
    weights to learn autoregressive coefficients for numerical time series targets.
    """

    def __init__(self,
                 encoder_span: dict,  # contains index span for each encoder
                 target_name: str,
                 input_size: int = None,
                 output_size: int = None,
                 shape: list = None,
                 max_params: int = 3e7,
                 num_hidden: int = 1,
                 dropout: float = 0) -> None:

        self.ar_net = None
        super().__init__(input_size=input_size,
                         output_size=output_size,
                         shape=shape,
                         max_params=max_params,
                         num_hidden=num_hidden,
                         dropout=dropout
                         )
        self.target = target_name
        self.encoder_span = encoder_span
        self.ar_column = f'__mdb_ts_previous_{self.target}'
        self.ar_idxs = list(*[range(idx[0], idx[1]) for col, idx in encoder_span.items() if col == self.ar_column])
        dims = [(len(self.ar_idxs), output_size)]
        linears = [nn.Linear(in_features=inf, out_features=outf) for inf, outf in dims]
        self.ar_net = nn.Sequential(*linears)
        self.ar_net.to(self.device)

    def to(self, device=None, available_devices=None):
        if self.ar_net:
            self.ar_net.to(device)
        return super().to(device)

    def forward(self, input):
        with LightwoodAutocast():
            if len(input.shape) == 1:
                input = input.unsqueeze(0)

            residual_output = self.net(input)
            ar_output = self.ar_net(input[:, self.ar_idxs])
            if self.ar_net.training:
                self.ar_net._modules['0'].weight = nn.Parameter(torch.clamp(self.ar_net._modules['0'].weight,
                                                                            0.0,
                                                                            0.999))  # force unit root
        return ar_output + residual_output

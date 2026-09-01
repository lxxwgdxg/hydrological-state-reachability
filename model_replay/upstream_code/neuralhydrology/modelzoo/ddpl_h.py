from typing import Dict, Tuple
import torch
import torch.nn as nn
from neuralhydrology.modelzoo.basemodel import BaseModel
from neuralhydrology.modelzoo.inputlayer import InputLayer
from neuralhydrology.utils.config import Config

"""
At the very beginning, note that our code is based on the publicly available codebase - NeuralHydrology
by ( Kratzert et al. (2022) [1]_.), on which we developed the DDPL models
`GitHub repository <https://github.com/neuralhydrology/neuralhydrology>‘).
In addition, ddpl is structurally based on the work of LSTM ([2]_.) and MC-LSTM ([3]_.)
References:
    ----------
    .. [1] Kratzert F, Gauch M, Nearing G, et al. NeuralHydrology---A Python library for Deep 
    Learning research in hydrology[J]. Journal of Open Source Software, 2022, 7(71): 4050.
    .. [2] Hochreiter S, Schmidhuber J. Long short-term memory[J]. Neural computation, 1997, 9(8): 1735-1780.
    .. [3] Hoedt P J, Kratzert F, Klotz D, et al. Mc-lstm: Mass-conserving lstm[C]//International conference 
    on machine learning. PMLR, 2021: 4275-4286.
"""


# This is potentially an operation that performs routing on the final predictions, but we didn't end up employing it
# def routing_uh_conv(q, uh):
#     # routing for every hidden layers --------
#     # batch_size = q.shape[1]
#     # hidden_size = q.shape[-1]
#     # q = torch.relu(q.permute([2, 1, 0])).reshape([1, batch_size * hidden_size, -1])  # [1, 32*256, 10] let it >0
#     # uh = uh.permute([2, 1, 0]).reshape([batch_size * hidden_size, 1, -1])  # [32*256, 1, 10] let it >0
#     # q_aim = torch.nn.functional.conv1d(q, torch.flip(uh, [2]), groups=batch_size * hidden_size, padding=0, stride=1,
#     #                                    bias=None)  # [1,256*32,1]
#     # return q_aim.reshape([-1, hidden_size, batch_size]).permute([0, 2, 1])  # [1,256,32]
#
#     # q--[10, 256, 1]; uh--[10, 256, 1];
#     batch_size = q.shape[1]
#     q = q.permute([2, 1, 0])  # [1, 256, 10]
#     uh = uh.permute([1, 2, 0])  # [256, 1, 10]
#     q_aim = torch.nn.functional.conv1d(q, torch.flip(uh, [2]), groups=batch_size, padding=0, stride=1, bias=None)
#     return q_aim
#
#
# def uh_gamma(n, k, t_lag):
#     # n, k ; [256,1]
#
#     n = torch.repeat_interleave(n.unsqueeze(0), repeats=t_lag, dim=0)  # [10, 256, 1]
#     k = torch.repeat_interleave(k.unsqueeze(0), repeats=t_lag, dim=0)  # [10, 256, 1]
#
#     # n--(0+0.1,3); n--(0+0.1,7)
#     n_new = torch.clamp(n * 3, min=0.1)
#     k_new = torch.clamp(k * 7, min=0.1)
#
#     # The local integral is equal to the median multiplied by the length of the interval(1day)
#     t = torch.arange(0.5, t_lag * 1.0).view([t_lag, 1, 1]).repeat([1, n.shape[1], n.shape[2]]).to(n.device)
#     uh = 1/((n_new.lgamma().exp()) * (k_new ** n_new)) * (t ** (n_new - 1)) * torch.exp(-t / k_new)
#     uh = torch.nn.functional.normalize(uh, p=1, dim=0)
#     # (10, 256, 1)
#     return uh


class DDPL_H(BaseModel):
    """
     DDPL_H: Data-driven process learning paradigm for catchment hydrological modeling
     Designed by He, et al [4]_..

     References
     ----------
     .. [4] He, et al, Data-driven process learning paradigm in geoscientific modeling：
     leveraging the local-to-global learning effects, 2023.
     """

    def __init__(self, cfg: Config):
        super(DDPL_H, self).__init__(cfg=cfg)

        self._n_mass_vars = len(cfg.mass_inputs)
        if self._n_mass_vars > 1:
            raise ValueError("Currently, not support many mass inputs")
        elif self._n_mass_vars == 0:
            raise ValueError("No mass input specified. Specify mass input variable using `mass_inputs`")

        if cfg.hidden_size < 1:
            raise ValueError("At least hidden size 1 is required for a mass cell.")

        self.embedding_net = InputLayer(cfg)

        n_aux_inputs = self.embedding_net.statics_output_size + self.embedding_net.dynamics_output_size
        self.ddpl_h = _ddplcell(mass_input_size=self._n_mass_vars,
                                    aux_input_size=n_aux_inputs,
                                    hidden_size=cfg.hidden_size,
                                    cfg=cfg)

        # self.routing_n = _routing_gate_sp(in_features=n_aux_inputs - 5, b_val=0)

        # self.routing_k = _routing_gate_sp(in_features=n_aux_inputs - 5, b_val=0)

    def forward(self, data: Dict[str, torch.Tensor], _scaler) -> Dict[str, torch.Tensor]:
        """Perform a forward pass on the ddpl_h model.

        Parameters
        ----------
        data : Dict[str, torch.Tensor]
            Dictionary, containing input features as key-value pairs.
        _scaler : Dict
            scaler
        Returns
        -------
        Dict[str, torch.Tensor]
            Model outputs and intermediate states as a dictionary.
        """
        # possibly pass static inputs through embedding layers and concatenate with dynamics
        x_d = self.embedding_net(data, concatenate_output=True)

        # the basedataset stores the mass input at the beginning
        x_m = x_d[:, :, :self._n_mass_vars]
        x_a = x_d[:, :, self._n_mass_vars:]

        # perform forward pass through the model cell
        m_out, et, m_out_ss, m_out_s, snow, Tt, Smax = self.ddpl_h(x_m, x_a, _scaler)
        # do routing for every hidden layers
        # t_lag = 10
        # rt_n = self.routing_n(x_a[0, :, 5:])  # (256,32)  same all time steps for static characteristics
        # rt_k = self.routing_k(x_a[0, :, 5:])  # (256,32)
        # UH = uh_gamma(rt_n, rt_k, t_lag)  # (10, 256, 32)
        # Q_aim = routing_uh_conv(m_out[-t_lag:, :, :], UH)  # [1,256,32]
        # output = torch.cat([m_out[:-1, :, :], Q_aim], dim=0)  # [365,256,32]
        # output_new = output.sum(dim=-1, keepdim=True)  # (365,256,1)

        # exclude trash cell from model predictions
        output = m_out.sum(dim=-1, keepdim=True)

        # do routing for considering the lagged influence of predictors in t_lag days on current hydrological responses
        # t_lag = 10
        # rt_n = self.routing_n(x_a[0, :, 5:])  # (256,1)  same all time steps for static characteristics
        # rt_k = self.routing_k(x_a[0, :, 5:])  # (256,1)
        # UH = uh_gamma(rt_n, rt_k, t_lag)  # (10, 256, 1)
        # Q_aim = routing_uh_conv(output[-t_lag:, :, :], UH)  # [1,256,1]
        # output_new = torch.cat([output[:-1, :, :], Q_aim], dim=0)

        ss_flow = m_out_ss.sum(dim=-1, keepdim=True)
        base_flow = m_out_s.sum(dim=-1, keepdim=True)
        sn = snow.sum(dim=-1, keepdim=True)

        # return {'y_hat': output.transpose(0, 1), 'm_out': m_out.transpose(0, 1), 'c': c.transpose(0, 1)}
        return {'y_hat': output.transpose(0, 1), 'et': et.transpose(0, 1), 'ss_flow': ss_flow.transpose(0, 1), 'base_flow': base_flow.transpose(0, 1),
                'snow': sn.transpose(0, 1), 'Tt': Tt.transpose(0, 1), 'Smax': Smax.transpose(0, 1)}


class _ddplcell(nn.Module):

    def __init__(self, mass_input_size: int, aux_input_size: int, hidden_size: int, cfg: Config):

        super(_ddplcell, self).__init__()
        self.cfg = cfg
        self._hidden_size = hidden_size

        gate_inputs = aux_input_size + hidden_size + mass_input_size

        # initialize gates
        self.output_gate = _Gate(in_features=gate_inputs, out_features=hidden_size)
        self.ddf_gate = _ddf_Gate(in_features=gate_inputs, out_features=hidden_size, b_val=0,
                                  normalizer="normalized_relu")   # '0' makes more sense after experiment
        self.Tt_gate = _Tt_gate(in_features=aux_input_size + mass_input_size, out_features=hidden_size)

        self.input_gate = _NormalizedGate(in_features=gate_inputs,
                                          out_shape=(mass_input_size, hidden_size),
                                          normalizer="normalized_sigmoid")

        self.input_gate_s = _NormalizedGate(in_features=gate_inputs,
                                            out_shape=(mass_input_size, hidden_size),
                                            normalizer="normalized_sigmoid")

        self.redistribution = _NormalizedGate(in_features=gate_inputs,
                                              out_shape=(hidden_size, hidden_size),
                                              normalizer="normalized_relu")

        self.redistribution_s = _NormalizedGate(in_features=gate_inputs,
                                                out_shape=(hidden_size, hidden_size),
                                                normalizer="normalized_relu")

        self.SMmax_gate = _SMmaxGate_nt(in_features=aux_input_size - 5,
                                        out_shape=(mass_input_size, hidden_size),
                                        b_val=3)

        self.SMfc_gate = _SMfcGate_nt(in_features=aux_input_size - 5,
                                      out_shape=(mass_input_size, hidden_size), b_val=1)

        self.bfout_gate = _bfout_gate(in_features=aux_input_size - 5,
                                      out_features=hidden_size, b_val=-3)  # '-3' makes more sense after experiment

        self.et_gate = _ddf_Gate(in_features=gate_inputs, out_features=hidden_size, b_val=-3,
                                 normalizer="normalized_relu")  # '-3' produce better results after experiment

        self._reset_parameters()

    def _reset_parameters(self):
        if self.cfg.initial_forget_bias is not None:
            nn.init.constant_(self.output_gate.fc.bias, val=self.cfg.initial_forget_bias)

    def _comparison_g(self, x):

        """
        A smooth approximation of Heaviside step function
            if x < 0: heaviside(x) ~= 0
            if x > 0: heaviside(x) ~= 1
        """
        tanh = nn.Tanh()
        return (tanh(x * 1e6) + 1) / 2

    def forward(self, x_m: torch.Tensor, x_a: torch.Tensor, _scaler) -> Tuple[torch.Tensor,
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Perform forward pass on the ddpl cell.

        Parameters
        ----------
        x_m : torch.Tensor
            Mass input that will be conserved by the network.
        x_a : torch.Tensor
            Auxiliary inputs.
        _scaler : dict

        Returns
        -------
        Tuple[torch.Tensor, torch.Tensor]
            Outgoing mass and memory cells per time step of shape [sequence length, batch size, hidden size]

        """
        _, batch_size, _ = x_m.size()
        ct = x_m.new_zeros((batch_size, self._hidden_size))
        ct_s = x_m.new_zeros((batch_size, self._hidden_size))

        m_out, et, m_out_ss, m_out_s, snow, Tt, Smax = [], [], [], [], [], [], []
        for xt_m, xt_a in zip(x_m, x_a):
            mt_out, ct, ct_s, mt_out_ss, mt_out_s, Ttt, ett, Stmax = self._step(xt_m, xt_a, ct, ct_s, _scaler)

            m_out.append(mt_out)
            et.append(ett)
            m_out_ss.append(mt_out_ss)
            m_out_s.append(mt_out_s)
            snow.append(ct_s)
            Smax.append(Stmax)
            Tt.append(Ttt)

        m_out, et, m_out_ss, m_out_s, snow, Tt, Smax = torch.stack(m_out), torch.stack(et), torch.stack(m_out_ss), \
                                                       torch.stack(m_out_s), torch.stack(snow), torch.stack(Tt), torch.stack(Smax)

        return m_out, et, m_out_ss, m_out_s, snow, Tt, Smax

    def _step(self, xt_m, xt_a, c, c_s, _scaler):
        """ Make a single time step in the ddpl_h. """

        Tmax_scaler = torch.tensor(_scaler["xarray_feature_scale"]["tmax(C)"].values)
        Tmax_center = torch.tensor(_scaler["xarray_feature_center"]["tmax(C)"].values)
        Tmin_scaler = torch.tensor(_scaler["xarray_feature_scale"]["tmin(C)"].values)
        Tmin_center = torch.tensor(_scaler["xarray_feature_center"]["tmin(C)"].values)
        T = torch.empty_like(xt_a[:, 1:3])
        T[:, 0] = xt_a[:, 1] * Tmax_scaler + Tmax_center
        T[:, 1] = xt_a[:, 2] * Tmin_scaler + Tmin_center
        T = torch.mean(T, dim=-1, keepdim=True)

        atv = nn.ReLU()
        Tt = self.Tt_gate(torch.cat([xt_m, xt_a], dim=-1))   # (256,1)

        Pr = torch.mul(xt_m, self._comparison_g(T - Tt))
        ps = xt_m - Pr

        # Normalize each basin state independently. This is the sample-separable
        # form already retained as a dormant alternative in the archived source.
        features = torch.cat([Pr, xt_a, c / (c.norm(p=1, dim=-1, keepdim=True) + 1e-6)], dim=-1)
        features_s = torch.cat([ps, xt_a, c_s / (c_s.norm(p=1, dim=-1, keepdim=True) + 1e-6)], dim=-1)

        # compute gate activations
        i = self.input_gate(features)   # (256,1,32)
        i_s = self.input_gate_s(features_s)

        S = self.SMmax_gate(xt_a[:, 5:])  # (256,1,32)
        Sfc = self.SMfc_gate(xt_a[:, 5:], S)

        r = self.redistribution(features)  # (256,32,32)
        r_s = self.redistribution_s(features_s)

        o = self.output_gate(features)  # (256,32)
        o_bf = self.bfout_gate(xt_a[:, 5:])
        o_ddf = self.ddf_gate(features_s)

        et = self.et_gate(features)

        m_sys_s = torch.matmul(c_s.unsqueeze(-2), r_s) + torch.matmul(ps.unsqueeze(-2), i_s)  # (256,1,32)
        m_out_melt = torch.minimum(torch.mul(o_ddf, torch.repeat_interleave(atv(T - Tt), repeats=self._hidden_size, dim=-1)
                                             ), m_sys_s.squeeze(-2))
        # m_out_melt = torch.mul(o_ddf, torch.repeat_interleave(differ_1, repeats=self._hidden_size, dim=-1))
        m_new_s = m_sys_s.squeeze(-2) - m_out_melt

        # for soil water system
        m_in = torch.matmul(Pr.unsqueeze(-2), i) + m_out_melt.unsqueeze(-2)
        m_sys = torch.matmul(c.unsqueeze(-2), r)

        m_out_f = torch.mul(m_in, self._comparison_g(m_sys - S))

        m_sys = m_sys + m_in - m_out_f

        m_out_ss = torch.mul(o, atv(m_sys - Sfc).squeeze(-2))
        # m_out_ss[:, 0] = 0

        m_out_bf = torch.mul(o_bf, m_sys.squeeze(-2))
        # m_out_bf[:, 0] = 0

        # Decide whether to force a restriction on et to ensure positive system quality,
        # or for the model to learn spontaneously, but may or may not result in a negative quality output
        # m_sys_re = m_sys.squeeze(-2) - m_out_ss - m_out_bf
        # et = torch.clamp(torch.minimum(m_sys_re, et), min=0)
        # m_new = m_sys_re - et

        m_new = m_sys.squeeze(-2) - m_out_ss - m_out_bf - et

        m_out = m_out_f.squeeze(-2) + m_out_ss + m_out_bf

        # return the outgoing mass and subtract this value from the cell states.
        # here we just return the learned Smax parameter, the Sfc also can be returned but should change other files
        return m_out, m_new, m_new_s, m_out_ss, m_out_bf, Tt, et.sum(dim=-1, keepdim=True), S.squeeze(-2)


class _Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sigmoid(self.fc(x))


class _ddf_Gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int, b_val: int, normalizer: str):
        super(_ddf_Gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.b_val = b_val
        # this can allow non-closed item to be negative
        if normalizer == "normalized_prelu":
            self.activation = nn.PReLU()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(f"Unknown normalizer {normalizer}. Must be one of {'normalized_prelu', 'normalized_relu'}")

        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        if self.b_val == 0:
            nn.init.zeros_(self.fc.bias)
        else:
            nn.init.constant_(self.fc.bias, val=self.b_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.activation(self.fc(x))


class _Tt_gate(nn.Module):
    """Utility class to implement a standard sigmoid gate"""

    def __init__(self, in_features: int, out_features: int):
        super(_Tt_gate, self).__init__()
        self.out_features = out_features
        self.fc = nn.Linear(in_features=in_features, out_features=1)
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        #  we scale it belongs to (0,4)
        return torch.clamp(self.fc(x), min=0, max=4)


class _NormalizedGate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], normalizer: str):
        super(_NormalizedGate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape

        if normalizer == "normalized_sigmoid":
            self.activation = nn.Sigmoid()
        elif normalizer == "normalized_relu":
            self.activation = nn.ReLU()
        else:
            raise ValueError(
                f"Unknown normalizer {normalizer}. Must be one of {'normalized_sigmoid', 'normalized_relu'}")
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Perform forward pass through the normalized gate"""
        h = self.fc(x).view(-1, *self.out_shape)
        return torch.nn.functional.normalize(self.activation(h), p=1, dim=-1)


class _SMmaxGate_nt(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], b_val: int):
        super(_SMmaxGate_nt, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape
        self.activation = nn.ReLU()
        self.b_val = b_val
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        if self.b_val == 0:
            nn.init.zeros_(self.fc.bias)
        else:
            nn.init.constant_(self.fc.bias, val=self.b_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, *self.out_shape)
        # return torch.clamp(self.activation(h), max=1600)
        return self.activation(h)


class _SMfcGate_nt(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_shape: Tuple[int, int], b_val: int):
        super(_SMfcGate_nt, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_shape[0] * out_shape[1])
        self.out_shape = out_shape
        self.activation = nn.ReLU()
        self.b_val = b_val
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        if self.b_val == 0:
            nn.init.zeros_(self.fc.bias)
        else:
            nn.init.constant_(self.fc.bias, val=self.b_val)

    def forward(self, x: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
        h = self.fc(x).view(-1, *self.out_shape)
        # return torch.clamp(self.activation(h), max=1600)
        return torch.minimum(self.activation(h), S)


class _bfout_gate(nn.Module):
    """Utility class to implement a gate with normalised activation function"""

    def __init__(self, in_features: int, out_features: int, b_val: int) -> object:
        super(_bfout_gate, self).__init__()
        self.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.b_val = b_val
        self._reset_parameters()

    def _reset_parameters(self):
        nn.init.orthogonal_(self.fc.weight)
        if self.b_val == 0:
            nn.init.zeros_(self.fc.bias)
        else:
            nn.init.constant_(self.fc.bias, val=self.b_val)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # return torch.sigmoid(self.fc(x)) * 0.5
        return torch.sigmoid(self.fc(x))


# class _routing_gate_sp(nn.Module):
#     """Utility class to implement a standard sigmoid gate"""
#
#     def __init__(self, in_features: int, b_val: int):
#         super(_routing_gate_sp, self).__init__()
#         self.fc = nn.Linear(in_features=in_features, out_features=1)
#         self.b_val = b_val
#         self._reset_parameters()
#
#     def _reset_parameters(self):
#         nn.init.orthogonal_(self.fc.weight)
#         # -1 let a small init values to forcus on today
#         if self.b_val == 0:
#             nn.init.zeros_(self.fc.bias)
#         else:
#             nn.init.constant_(self.fc.bias, val=self.b_val)
#         # nn.init.zeros_(self.fc1.bias)
#         # nn.init.zeros_(self.fc2.bias)
#
#     def forward(self, x: torch.Tensor) -> torch.Tensor:
#         """Perform forward pass through the normalised gate"""
#         # return self.fc(x)
#         # return torch.repeat_interleave(self.fc(x).unsqueeze(-1), repeats=self.out_features, dim=-1)
#         return torch.sigmoid(self.fc(x))

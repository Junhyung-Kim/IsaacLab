# Copyright (c) 2018-2023, NVIDIA Corporation
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# 3. Neither the name of the copyright holder nor the names of its
#    contributors may be used to endorse or promote products derived from
#    this software without specific prior written permission.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from rl_games.algos_torch import torch_ext
from rl_games.algos_torch import layers
from rl_games.algos_torch import network_builder

import torch
import torch.nn as nn
import numpy as np

DISC_LOGIT_INIT_SCALE = 1.0


class AMPBuilder(network_builder.A2CBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(network_builder.A2CBuilder.Network):
        def __init__(self, params, **kwargs):
            self.lipschitz_constant_disc = 1.0 # hyper parameter for spectral normalization
            super().__init__(params, **kwargs)
            if self.is_continuous:
                if (self.space_config['fixed_sigma']):
                    ############### JY edit for linear change sigma ###############
                    self.sigma_init = self.space_config['sigma_init']['val']
                    self.sigma_last = self.space_config['sigma_last']['val']
                    ############### end of JY edit for linear change sigma ########
                    actions_num = kwargs.get('actions_num')
                    sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                    self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=False, dtype=torch.float32), requires_grad=False)
                    sigma_init(self.sigma)
                # else:
                #     sigma_init = self.init_factory.create(**self.space_config['sigma_init'])
                #     self.sigma_last = self.space_config['sigma_last']['val']
                #     actions_num = kwargs.get('actions_num')
                #     self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
                #     sigma_init(self.sigma)
                    
            amp_input_shape = kwargs.get('amp_input_shape')
            self._build_disc(amp_input_shape)
            print(f"_disc_mlp[] device: {next(self._disc_mlp.parameters()).device}")

            return

        def load(self, params):
            super().load(params)

            self._disc_units = params['disc']['units']
            self._disc_activation = params['disc']['activation']
            self._disc_initializer = params['disc']['initializer']

            if 'is_cnn' not in params['disc']:
                self._is_cnn = False
            else:
                self._is_cnn = params['disc']['is_cnn']
                if self._is_cnn:
                    self._disc_cnn_config = params['disc']['cnn']

            return
        
        def forward(self, obs_dict):
            obs = obs_dict['obs']
            states = obs_dict.get('rnn_states', None)
            dones = obs_dict.get('dones', None)
            bptt_len = obs_dict.get('bptt_len', 0) 

            if self.has_cnn:
                # for obs shape 4
                # input expected shape (B, W, H, C)
                # convert to (B, C, W, H)
                if self.permute_input and len(obs.shape) == 4:
                    obs = obs.permute((0, 3, 1, 2))

            if self.separate:
                a_out = c_out = obs
                a_out = self.actor_cnn(a_out)
                a_out = a_out.contiguous().view(a_out.size(0), -1)

                c_out = self.critic_cnn(c_out)
                c_out = c_out.contiguous().view(c_out.size(0), -1)     

                a_out = self.actor_mlp(a_out)
                c_out = self.critic_mlp(c_out)
                            
                value = self.value_act(self.value(c_out))

                if self.is_continuous:
                    if self.fixed_sigma:
                        mu = self.mu_act(self.mu(a_out))
                        sigma = mu * 0.0 + self.sigma_act(self.sigma)
                    else:
                        mu = self.mu_act(self.mu(a_out))
                        sigma = self.sigma_act(self.sigma(a_out))
                        # sigma = self.sigma_act(self.sigma)
                        # sigma = torch.clamp(sigma, self.sigma_last, torch.inf)
                        # scale = torch.norm(sigma.clone().detach(),p=2)
                        # mu = self.mu_act(self.mu(a_out, scale))
                    return mu, sigma, value, states
            else:
                return super().forward(obs_dict)

        def eval_critic(self, obs, history_obs=None):
            c_out = self.critic_cnn(obs)
            c_out = c_out.contiguous().view(c_out.size(0), -1)
            c_out = self.critic_mlp(c_out)              
            value = self.value_act(self.value(c_out))
            return value

        def eval_disc(self, amp_obs):
            disc_mlp_out = self._disc_mlp(amp_obs)
            disc_logits = self._disc_logits(disc_mlp_out)
            return disc_logits

        def get_disc_logit_weights(self):
            return torch.flatten(self._disc_logits.weight)

        def get_disc_weights(self):
            weights = []
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    weights.append(torch.flatten(m.weight))

            weights.append(torch.flatten(self._disc_logits.weight))
            return weights

        def _build_disc(self, input_shape):
            self._disc_mlp = nn.Sequential()
            
            mlp_args = {
                'input_size' : input_shape[0], 
                'units' : self._disc_units, 
                'activation' : self._disc_activation, 
                'dense_func' : torch.nn.Linear,
                # 'dense_func' : sn_linear
            }
            self._disc_mlp = self._build_mlp(**mlp_args)
            print(f"_disc_mlp[] device: {next(self._disc_mlp.parameters()).device}")
            
            mlp_out_size = self._disc_units[-1]
            self._disc_logits = torch.nn.Linear(mlp_out_size, 1)
            # self._disc_logits = sn_linear(mlp_out_size, 1)

            mlp_init = self.init_factory.create(**self._disc_initializer)
            for m in self._disc_mlp.modules():
                if isinstance(m, nn.Linear):
                    mlp_init(m.weight)
                    if getattr(m, "bias", None) is not None:
                        torch.nn.init.zeros_(m.bias) 

            torch.nn.init.uniform_(self._disc_logits.weight, -DISC_LOGIT_INIT_SCALE, DISC_LOGIT_INIT_SCALE)
            torch.nn.init.zeros_(self._disc_logits.bias) 

            return

    def build(self, name, **kwargs):
        net = AMPBuilder.Network(self.params, **kwargs)
        return net
    
import torch.nn.utils as nn_utils
def sn_scaled_linear(input_size, unit, lipschitz_constant=1.0):
    layer = nn_utils.spectral_norm(nn.Linear(input_size, unit)) #, eps=1e-6)
    return ScaledLinear(layer, lipschitz_constant)

def sn_linear(input_size, unit):
    return nn_utils.spectral_norm(nn.Linear(input_size, unit))#, eps=1e-6)

class ScaledLinear(nn.Module):
    """Wrapper to scale the output of spectral norm layers"""
    def __init__(self, layer, lipschitz_constant):
        super().__init__()
        self.layer = layer
        self.lipschitz_constant = lipschitz_constant  # Scaling factor

    # def forward(self, x, scale):
        # scale = scale.to(x.device)
        # return nn.functional.linear(x, scale * self.layer.weight.to(x.device), self.layer.bias.to(scale.device))
    def forward(self, x):
        return self.lipschitz_constant * self.layer(x)  

    # def lipschitz_update(self, progress_remaining):
        # if (progress_remaining > 0.5):
            # self.lipschitz_constant = 1 - 1.6 * (1 - progress_remaining)
            # self.lipschitz_constant = 1.0
        # else:
            # self.lipschitz_constant = 0.2
    def lipschitz_update(self, sigma):
        self.lipschitz_constant = sigma

    @property
    def weight(self):
        """Expose the weight of the underlying layer"""
        return self.layer.weight  

    @property
    def bias(self):
        """Expose the bias of the underlying layer (if exists)"""
        return self.layer.bias  

class AMPwithSNBuilder(AMPBuilder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        return

    class Network(AMPBuilder.Network):
        def __init__(self, params, **kwargs):
            self.fixed_lipschitz = params['fixed_lipschitz']
            self.lipschitz_constant = params['lipschitz_constant'] # hyper parameter for spectral normalization
            super().__init__(params, **kwargs)
            mlp_args = {
                'input_size' : self.critic_mlp[0].in_features, 
                'units' : self.units, 
                'activation' : self.activation, 
                'norm_func_name' : self.normalization,
                # 'dense_func' : torch.nn.Linear,
                'dense_func' : sn_linear,
                'd2rl' : self.is_d2rl,
                'norm_only_first_layer' : self.norm_only_first_layer
            }
            # self.critic_mlp = self._build_mlp(**mlp_args)
            self.actor_mlp = self._build_mlp(**mlp_args)

            out_size = self.mu.in_features
            actions_num = self.mu.out_features  
            self.mu = sn_scaled_linear(out_size, actions_num, self.lipschitz_constant)
            # self.mu = sn_linear(out_size, actions_num)

            # if self.fixed_sigma:
                # self.sigma = nn.Parameter(torch.zeros(actions_num, requires_grad=True, dtype=torch.float32), requires_grad=True)
            # else:
                # self.sigma = nn_utils.spectral_norm(torch.nn.Linear(out_size, actions_num))
            return

        # def _build_value_layer(self, input_size, output_size):
            # return nn_utils.spectral_norm(torch.nn.Linear(input_size, output_size))
        
    def build(self, name, **kwargs):
        net = AMPwithSNBuilder.Network(self.params, **kwargs)
        return net
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

import torch
import torch.nn as nn
from rl_games.algos_torch.models import ModelA2CContinuousLogStd
from rl_games.algos_torch.running_mean_std import RunningMeanStd, RunningMeanStdObs


class ModelAMPContinuous(ModelA2CContinuousLogStd):
    def __init__(self, network):
        super().__init__(network)
        return

    def build(self, config):


        obs_shape = config['input_shape']
        encoder_shape = config.get('history_obs_shape', None)
        normalize_value = config.get('normalize_value', False)
        normalize_input = config.get('normalize_input', False)
        value_size = config.get('value_size', 1)

        if encoder_shape is not None:
            config['input_shape'] = (obs_shape[0] + encoder_shape[0],)    
        net = self.network_builder.build('amp', **config)
        for name, _ in net.named_parameters():
            print(name)
            
        return self.Network(net, obs_shape=obs_shape, encoder_shape=encoder_shape,
            normalize_value=normalize_value, normalize_input=normalize_input, value_size=value_size)


    class Network(ModelA2CContinuousLogStd.Network):
        # def __init__(self, a2c_network, **kwargs):
        #     super().__init__(a2c_network, **kwargs)
        #     return
        def __init__(self, a2c_network, obs_shape, encoder_shape, normalize_value, normalize_input, value_size):
            nn.Module.__init__(self)
            if encoder_shape is not None:
                self.obs_shape = (obs_shape[0] + encoder_shape[0],)
            else:
                self.obs_shape = obs_shape
            self.normalize_value = normalize_value
            self.normalize_input = normalize_input
            self.value_size = value_size

            if self.normalize_value:
                self.value_mean_std = RunningMeanStd((self.value_size,))
            if self.normalize_input:
                if isinstance(obs_shape, dict):
                    self.running_mean_std = RunningMeanStdObs(obs_shape)
                else:
                    self.running_mean_std = RunningMeanStd(obs_shape)
            self.a2c_network = a2c_network
            return
        
        def forward(self, input_dict):
            is_train = input_dict.get('is_train', True)
            # result = super().forward(input_dict)
            # for lipschitz constraint policy
            prev_actions = input_dict.get('prev_actions', None)
            mu, logstd, value, states = self.a2c_network(input_dict)
            sigma = torch.exp(logstd)
            distr = torch.distributions.Normal(mu, sigma, validate_args=False)
            if is_train:
                entropy = distr.entropy().sum(dim=-1)
                prev_neglogp = self.neglogp(prev_actions, mu, sigma, logstd)
                result = {
                    'prev_neglogp' : torch.squeeze(prev_neglogp),
                    'values' : value,
                    'entropy' : entropy,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }                
            else:
                selected_action = distr.sample()
                neglogp = self.neglogp(selected_action, mu, sigma, logstd)
                result = {
                    'neglogpacs' : torch.squeeze(neglogp),
                    'values' : self.denorm_value(value),
                    'actions' : selected_action,
                    'rnn_states' : states,
                    'mus' : mu,
                    'sigmas' : sigma
                }

            if (is_train):
                amp_obs = input_dict['amp_obs']
                disc_agent_logit = self.a2c_network.eval_disc(amp_obs)
                result["disc_agent_logit"] = disc_agent_logit

                amp_obs_replay = input_dict['amp_obs_replay']
                disc_agent_replay_logit = self.a2c_network.eval_disc(amp_obs_replay)
                result["disc_agent_replay_logit"] = disc_agent_replay_logit

                amp_demo_obs = input_dict['amp_obs_demo']
                disc_demo_logit = self.a2c_network.eval_disc(amp_demo_obs)
                result["disc_demo_logit"] = disc_demo_logit

            return result
        
        def update_action_noise(self, progress_remaining):
            # if (progress_remaining > 0.5):
            #     progress_remaining_biased = 1.0
            # else:
            #     progress_remaining_biased = 2*progress_remaining
            
            if (progress_remaining > 0.5):
                progress_remaining_biased = 2*progress_remaining - 1
            else:
                progress_remaining_biased = 0.0
            
            self.a2c_network.sigma[:] = self.a2c_network.sigma_init * progress_remaining_biased + self.a2c_network.sigma_last * (1-progress_remaining_biased)
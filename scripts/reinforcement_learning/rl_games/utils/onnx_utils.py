from rl_games.torch_runner import Runner
from rl_games.torch_runner import _restore, _override_sigma
import torch 

class OnnxRunner(Runner):
    def __init__(self, algo_observer=None):
        Runner.__init__(self, algo_observer)
        
    def run_play(self, args):
        print('Started to play')
        player = self.create_player()
        _restore(player, args)
        _override_sigma(player, args)

        import rl_games.algos_torch.flatten as flatten

        device = torch.device('cpu')
        inputs = {
            'obs' : torch.zeros((1,) + player.obs_shape).to(player.device),
            # 'goal_sequence' : torch.zeros((1,) + player.num_goal_sequence).to(player.device),
            # 'history_obs' : torch.zeros((1,) + player._history_obs_shape).to(player.device),
            'rnn_states' : player.states,
        }

        with torch.no_grad():
            adapter = flatten.TracingAdapter(ModelWrapper(player.model), inputs, allow_non_tensor=True)
            traced = torch.jit.trace(adapter, adapter.flattened_inputs, check_trace=False)

        # export only the file name from the path and exclude the extension
        onnx_name = args['checkpoint'].split('/')[-1]
        onnx_name = onnx_name.split('.')[0]
        torch.onnx.export(traced, adapter.flattened_inputs, "result/"+onnx_name+".onnx", verbose=False, input_names=['history_obs', 'obs'], output_names=['mu','log_std', 'value'])

        player.run()

class ModelWrapper(torch.nn.Module):
    '''
    Main idea is to ignore outputs which we don't need from model
    '''
    def __init__(self, model):
        torch.nn.Module.__init__(self)
        self._model = model
        
        
    def forward(self,input_dict):
        input_dict['obs'] = self._model.norm_obs(input_dict['obs'])
        # input_dict['history_obs'] = self._model._hist_obs_mean_std(input_dict['history_obs'])
        '''
        just model export doesn't work. Looks like onnx issue with torch distributions
        thats why we are exporting only neural network
        '''
        mu, logstd, value, states = self._model.a2c_network(input_dict)
        # mu, value, states = self._model.a2c_network(input_dict)
        value = self._model.denorm_value(value)
        # #input_dict['is_train'] = False
        # return output_dict['logits'], output_dict['values']
        return mu, logstd, value
        # return self._model.a2c_network(input_dict)
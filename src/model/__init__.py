from .model import *

Model_Config = {
    'sac': SAC_Model,
    'curl': CURL_Model,
    'sacae': SACAE_Model,
    'rad': SAC_Model,
    'drq': SAC_Model
}

def make_model(agent_obs_shape, action_shape, args, device):
    Model = Model_Config[args.agent]
     
    model = Model(obs_shape = agent_obs_shape, 
                  action_shape = action_shape, 
                  hidden_dim = args.hidden_dim,
                  encoder_feature_dim = args.encoder_feature_dim,
                  log_std_min = args.actor_log_std_min,
                  log_std_max = args.actor_log_std_max,
                  num_layers = args.num_layers,
                  num_filters = args.num_filters,
                  device = device)
    
    return model
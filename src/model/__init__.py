from .model import *

def make_model(agent_obs_shape, action_shape, args, device):    
    if args.agent == 'curl':
        model = CURL_Model(obs_shape = agent_obs_shape, 
                           action_shape = action_shape, 
                           hidden_dim = args.hidden_dim,
                           encoder_feature_dim = args.encoder_feature_dim,
                           log_std_min = args.actor_log_std_min,
                           log_std_max = args.actor_log_std_max,
                           num_layers = args.num_layers,
                           num_filters = args.num_filters,
                           device = device)
    else:
        print('unsupported agent')
        model = None
    
    return model
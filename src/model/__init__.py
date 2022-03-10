from .model import *

Model_Config = {
    'sac': SAC_Model,
    'curl': CURL_Model,
    'sacae': SACAE_Model,
    'rad': SAC_Model,
    'drq': SAC_Model,
    'atc': ATC_Model
}

def make_model(agent_obs_shape, action_shape, args, device):
    Model = Model_Config[args.agent]
     
    if Model not in [ATC_Model]:
        model = Model(obs_shape = agent_obs_shape, 
                      action_shape = action_shape, 
                      hidden_dim = args.hidden_dim,
                      encoder_feature_dim = args.encoder_feature_dim,
                      log_std_min = args.actor_log_std_min,
                      log_std_max = args.actor_log_std_max,
                      num_layers = args.num_layers,
                      num_filters = args.num_filters,
                      device = device)
    else:
        model = Model(obs_shape = agent_obs_shape, 
                      action_shape = action_shape, 
                      hidden_dim = args.hidden_dim,
                      encoder_feature_dim = args.encoder_feature_dim,
                      log_std_min = args.actor_log_std_min,
                      log_std_max = args.actor_log_std_max,
                      num_layers = args.num_layers,
                      num_filters = args.num_filters,
                      device = device,
                      atc_encoder_feature_dim = args.atc_encoder_feature_dim,
                      atc_hidden_feature_dim = args.atc_hidden_feature_dim)
    
    return model
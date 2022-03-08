from .curl import *
from .sac_ae import *
from .sac import *

Agent_Config = {
    'curl': CURL,
    'sacae': SACAE,
    'sac': SAC
}

def make_agent(model, device, action_shape, args):
    Agent = Agent_Config[args.agent]
    return Agent(model, device, action_shape, args)

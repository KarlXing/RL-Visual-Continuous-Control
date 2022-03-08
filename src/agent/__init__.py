from .curl import *
from .sac_ae import *


def make_agent(model, device, action_shape, args):
    if args.agent == 'curl':
        agent = CURL(model, device, action_shape, args)
    elif args.agent == 'sacae':
        agent = SACAE(model, device, action_shape, args)
    else:
        return None 
    
    return agent
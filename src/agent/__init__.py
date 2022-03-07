from ast import arg
from pyexpat import model
from .curl import *


def make_agent(model, device, action_shape, args):
    if args.agent == 'curl':
        agent = CURL(model, device, action_shape, args)
    else:
        return None 
    
    return agent
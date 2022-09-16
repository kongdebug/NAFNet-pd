import paddle
import os
from collections import OrderedDict

def freeze(model):
    for p in model.parameters():
        p.stop_gradient = True

def unfreeze(model):
    for p in model.parameters():
        p.stop_gradient = False

def is_frozen(model):
    x = [p.stop_gradient for p in model.parameters()]
    return all(x)

def save_checkpoint(model_dir, state, session):
    epoch = state['epoch']
    model_out_path = os.path.join(model_dir,"model_epoch_{}_{}.pdparams".format(epoch,session))
    paddle.save(state, model_out_path)

def load_checkpoint(model, weights):
    checkpoint = paddle.load(weights)
    try:
        model.set_state_dict(checkpoint["state_dict"])
    except:
        state_dict = checkpoint["state_dict"]
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            name = k[7:] # remove `Layer.`
            new_state_dict[name] = v
        model.set_state_dict(new_state_dict)


def load_checkpoint_multigpu(model, weights):
    checkpoint = paddle.load(weights)
    state_dict = checkpoint["state_dict"]
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `Layer.`
        new_state_dict[name] = v
    model.set_state_dict(new_state_dict)

def set_start_epoch(weights):
    checkpoint = paddle.load(weights)
    epoch = checkpoint["epoch"]
    return epoch

def load_optim(optimizer, weights):
    checkpoint = paddle.load(weights)
    optimizer.set_state_dict(checkpoint['optimizer'])
    # for p in optimizer.param_groups: lr = p['lr']
    # return lr

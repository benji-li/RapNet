import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import dataprep
import model

def fb_props(rnn,optimizer,criterion,inp,target,hidden,gpu_avail):
    rnn.zero_grad()
    if gpu_avail:
        inp,target = inp.cuda(),target.cuda()
    
    hidden = tuple([x.data for x in hidden])
    
    output,hidden = rnn.forward(inp,hidden)
    loss = criterion(output.squeeze(),target)
    loss.backward()
    nn.utils.clip_grad_norm_(rnn.parameters(),5)
    optimizer.step()

    return loss.item(),hidden


def training_loop(rnn,batch_size,optimizer,criterion,n_epochs):
    pass
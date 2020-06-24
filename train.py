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
    gpu_avail=False
    if torch.cuda_is_available():
        rnn.cuda()
        gpu_avail=True
    losses=[]
    rnn.train()
    for epoch in range(n_epochs):
        hidden = rnn.init_hidden(batch_size,gpu_avail)
        for batch, (inputs,targets) in enumerate(train_loader,1):
            #ensure it's a full batch before props
            n_batches = len(train_loader.dataset)//batch_size
            if(batch > n_batches):
                break
            loss,hidden = fb_props(rnn,optimizer,criterion,inputs,targets,hidden,gpu_avail)
            losses.append(loss)
            
            if batch%100==0:
                print('Epoch {}/{} \t Loss: {}'.format(epoch+1,n_epochs,np.average(losses)))
                losses=[]
    return rnn
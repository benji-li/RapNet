import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import time
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


def training_loop(rnn,batch_size,optimizer,criterion,n_epochs,gpu_avail):
    losses=[]
    rnn.train()
    t_start = time.time()
    for epoch in range(n_epochs):
        hidden = rnn.init_hidden_weights(batch_size,gpu_avail)
        for batch, (inputs,targets) in enumerate(train_loader,1):
            #ensure it's a full batch before props
            n_batches = len(train_loader.dataset)//batch_size
            if(batch > n_batches):
                break
            loss,hidden = fb_props(rnn,optimizer,criterion,inputs,targets,hidden,gpu_avail)
            losses.append(loss)
            
            if batch%100==0:
                t_end = time.time()
                print('Epoch {}/{} \t Loss: {} \t Progress: {}% \t Time Elapsed: {} minutes'.format(
                    epoch+1,
                    n_epochs,
                    np.average(losses),
                    (epoch*n_batches+batch)/(n_epochs*n_batches)*100,
                    (t_end-t_start)/60))
                losses=[]
    return rnn

data_dir = './data/lyrics.txt'
seq_length = 32
batch_size = 128

#prepping and loading the data
v_to_i,i_to_v,text_nums = dataprep.data_processor(data_dir)
train_loader = dataprep.data_batcher(text_nums,seq_length,batch_size)

#set some hyperparameters
epochs = 5
learning_rate = 0.001
vocab_size = output_size = len(v_to_i)
embedding_dim = 256
hidden_dim = 500
num_layers = 2
dropout = 0.5

net = model.rnn(vocab_size,output_size,embedding_dim,hidden_dim,num_layers,dropout)
print(net)

#check for a gpu
if torch.cuda.is_available():
    print('GPU is available!')
    gpu_avail=True
    net.cuda()
else:
    print('GPU not found, will train on CPU!')
    gpu_avail=False

optimizer = optim.Adam(net.parameters(),lr=learning_rate)
criterion = nn.CrossEntropyLoss()

trained_model = training_loop(net,batch_size,optimizer,criterion,epochs,gpu_avail)

torch.save(trained_model.state_dict(),'trained_model.pt')
print('model successfully trained and saved!')

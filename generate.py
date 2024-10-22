import model
import torch
import torch.nn.functional as F
import dataprep
import numpy as np

data_dir = './data/lyrics.txt'
v_to_i,i_to_v,text_nums = dataprep.data_processor(data_dir)

hypers = model.HyperParams(len(v_to_i),len(v_to_i))
seq_length = 32

trained_net = model.Rnn(hypers.vocab_size,hypers.output_size,hypers.embedding_dim,hypers.hidden_dim,hypers.num_layers,hypers.dropout)
trained_net.load_state_dict(torch.load('./trained_model.pt'))
trained_net.eval()

gen_length = 500
start_word = "wire"
start_word_ind = v_to_i[start_word]
generated = [start_word]

init_seq = np.full((1,seq_length),v_to_i['<newline>'])
init_seq[-1][-1] = start_word_ind

gpu_avail = torch.cuda.is_available()
for x in range(gen_length):
    if gpu_avail:
        init_seq = torch.LongTensor(init_seq).cuda()
    else:
        init_seq = torch.LongTensor(init_seq)
    hidden = trained_net.init_hidden_weights(init_seq.size(0),gpu_avail)
    output, _ = trained_net.forward(init_seq,hidden)
    probs = F.softmax(output,dim=1).data

    top_k = 10
    probs,top_inds = probs.topk(top_k)
    top_inds = top_inds.numpy().squeeze()
    probs =  probs.numpy().squeeze()
    chosen_word_ind = np.random.choice(top_inds,p=probs/probs.sum())
    generated.append(i_to_v[chosen_word_ind])

    init_seq = np.roll(init_seq,-1,1)
    init_seq[-1][-1] = chosen_word_ind
generated = ' '.join(generated)
generated = dataprep.punctuation_handler(generated,for_gen=True)

try:
    f = open("./generated_samples/generated_lyrics.txt","a")
except:
    f = open("./generated_samples/generated_lyrics.txt","w")
f.write(generated)
f.close()
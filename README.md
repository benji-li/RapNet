# YeNet
Deep learning generated rap lyrics trained on a dataset of any musician's discography. I used [PyTorch](pytorch.org) to create a word-level recurrent neural network utilizing Long Short-Term Memory [(LSTM)](https://towardsdatascience.com/illustrated-guide-to-lstms-and-gru-s-a-step-by-step-explanation-44e9eb85bf21) cells
that can generate lyrics upon provided a "primer" word as indicated by the user.

## Generated Samples:
Here are some of my favourite bars out of the many the network generated. I guess they almost make sense?
```
 you know the heart is the whole thing 
 you know i love that shit 
 i need the new girl ? 
 don't be like you 
 they say , " i know what , but she can't have to do 
 you don't really wanna be in a few girl 
 i can't let me go and the game 
 
 i know it's corny , we gon' make it 
 i don't know what i was to the world 
 
  you know how it feel like ? " 
 now i'm in my club , i don't want to do it 
 but i don't need my new slaves 
```
More lyrics, including the ones above, can be found in generated_samples > generated_lyrics.txt

## Try the Program!
Ensure the following libraries are installed along with a distribution of Python3:
* NumPy
* PyTorch

I recommend using [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/) to install these packages, as well as running the project in a virtual environment.

Run train.py to instantiate and train a custom model, or simply run generate.py to try out the pretrained model in this repo.

In terms of training a custom model, simply set some hyperparameters in the HyperParams dataclass in model.py. 

*NOTE: Having an available GPU will speed up the training process substantially, otherwise a decent-performing model will take over an hour to train.*

### Generate Your Own Dataset!
Using genius_lyrics.py, one can easily create a dataset of lyrics from an artist of choice. Using the lyricsgenius wrapper for the [Genius API](https://docs.genius.com/), simply follow this [link](https://genius.com/api-clients) that will allow you to generate an access key. Then, simply specify an artist name and max_songs parameter to create your dataset!



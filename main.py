# -*- coding: utf-8 -*-

from flask import Flask,render_template,url_for,request
import torch
import helper
import numpy as np
import torch.nn as nn



app=Flask(__name__)

@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])

def predict():
    
    


    # Check for a GPU
    train_on_gpu = torch.cuda.is_available()
    if not train_on_gpu:
        print('No GPU found. Please use a GPU to train your neural network.')
    _, vocab_to_int, int_to_vocab, token_dict = helper.load_preprocess()
    
    
    class RNN(nn.Module):
        
        def __init__(self, vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5):
            """
            Initialize the PyTorch RNN Module
            :param vocab_size: The number of input dimensions of the neural network (the size of the vocabulary)
            :param output_size: The number of output dimensions of the neural network
            :param embedding_dim: The size of embeddings, should you choose to use them        
            :param hidden_dim: The size of the hidden layer outputs
            :param dropout: dropout to add in between LSTM/GRU layers
            """
            super(RNN, self).__init__()
            # TODO: Implement function
            
            # set class variables
            self.output_size = output_size
            self.n_layers = n_layers
            self.hidden_dim = hidden_dim
            
            # define model layers
            
            # embedding and LSTM layers
            self.embed = nn.Embedding(vocab_size, embedding_dim)
            self.lstm = nn.LSTM(embedding_dim, hidden_dim, n_layers, dropout=dropout, batch_first=True)
                    
            # linear layer
            self.fc = nn.Linear(hidden_dim, output_size)
        
        
        def forward(self, nn_input, hidden):
            """
            Forward propagation of the neural network
            :param nn_input: The input to the neural network
            :param hidden: The hidden state        
            :return: Two Tensors, the output of the neural network and the latest hidden state
            """
            # TODO: Implement function   
            batch_size = nn_input.size(0)
    
            # embeddings and lstm_out
            embeds = self.embed(nn_input)
            lstm_out, hidden = self.lstm(embeds, hidden)
            # stack up lstm outputs
            lstm_out = lstm_out.contiguous().view(-1, self.hidden_dim)
            
            # dropout and fully-connected layer
            output = self.fc(lstm_out)
            
            # reshape to be batch_size first
            output = output.view(batch_size, -1, self.output_size)
            out = output[:, -1] # get last batch of labels       
            # return one batch of output word scores and the hidden state
            return out, hidden
           
        
        
        
        def init_hidden(self, batch_size):
            '''
            Initialize the hidden state of an LSTM/GRU
            :param batch_size: The batch_size of the hidden state
            :return: hidden state of dims (n_layers, batch_size, hidden_dim)
            '''
            # Implement function
            
            # initialize hidden state with zero weights, and move to GPU if available
          # initialize hidden state with zero weights, and move to GPU if available
            weight = next(self.parameters()).data
            
            if (train_on_gpu):
                hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda(),
                          weight.new(self.n_layers, batch_size, self.hidden_dim).zero_().cuda())
            else:
                hidden = (weight.new(self.n_layers, batch_size, self.hidden_dim).zero_(),
                          weight.new(self.n_layers, batch_size, self.hidden_dim).zero_())
            
            return hidden
    
    
    # Model parameters
    # Vocab size
    vocab_size = len(vocab_to_int)
    # Output size
    output_size = vocab_size
    # Embedding Dimension
    embedding_dim = 256
    # Hidden Dimension
    hidden_dim = 256
    # Number of RNN Layers
    n_layers = 2
    sequence_length =8   
    
    trained_rnn = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, dropout=0.5)
    
    
    
    
    trained_rnn.load_state_dict(torch.load('trained_rnn1.pt',map_location=torch.device('cpu')))
    
    """
    DON'T MODIFY ANYTHING IN THIS CELL THAT IS BELOW THIS LINE
    """
    import torch.nn.functional as F
    
    def generate(rnn, prime_id, int_to_vocab, token_dict, pad_value, predict_len=100):
        """
        Generate text using the neural network
        :param decoder: The PyTorch Module that holds the trained neural network
        :param prime_id: The word id to start the first prediction
        :param int_to_vocab: Dict of word id keys to word values
        :param token_dict: Dict of puncuation tokens keys to puncuation values
        :param pad_value: The value used to pad a sequence
        :param predict_len: The length of text to generate
        :return: The generated text
        """
        rnn.eval()
        
        # create a sequence (batch_size=1) with the prime_id
        current_seq = np.full((1, sequence_length), pad_value)
        current_seq[-1][-1] = prime_id
        predicted = [int_to_vocab[prime_id]]
        
        for _ in range(predict_len):
            if train_on_gpu:
                current_seq = torch.LongTensor(current_seq).cuda()
            else:
                current_seq = torch.LongTensor(current_seq)
            
            # initialize the hidden state
            hidden = rnn.init_hidden(current_seq.size(0))
            
            # get the output of the rnn
            output, _ = rnn(current_seq, hidden)
            
            # get the next word probabilities
            p = F.softmax(output, dim=1).data
            if(train_on_gpu):
                p = p.cpu() # move to cpu
             
            # use top_k sampling to get the index of the next word
            top_k = 5
            p, top_i = p.topk(top_k)
            top_i = top_i.numpy().squeeze()
            
            # select the likely next word index with some element of randomness
            p = p.numpy().squeeze()
            word_i = np.random.choice(top_i, p=p/p.sum())
            
            # retrieve that word from the dictionary
            word = int_to_vocab[word_i]
            predicted.append(word)     
            
            # the generated word becomes the next "current sequence" and the cycle can continue
            current_seq = np.roll(current_seq, -1, 1)
            current_seq[-1][-1] = word_i
        
        gen_sentences = ' '.join(predicted)
        
        # Replace punctuation tokens
        for key, token in token_dict.items():
            ending = ' ' if key in ['\n', '(', '"'] else ''
            gen_sentences = gen_sentences.replace(' ' + token.lower(), key)
        gen_sentences = gen_sentences.replace('\n ', '\n')
        gen_sentences = gen_sentences.replace('( ', '(')
        
        # return all the sentences
        return gen_sentences
    
    if(request.method == 'POST'):
        message = request.form['message']
        gen_length = 400 # modify the length to your preference
        prime_word = message # name for starting the script
        pad_word = helper.SPECIAL_WORDS['PADDING']
        generated_script = generate(trained_rnn, vocab_to_int[prime_word + ':'], int_to_vocab, token_dict, vocab_to_int[pad_word], gen_length)
		
        return render_template('result.html',prediction = generated_script)

    
if __name__ == '__main__':
	app.run(debug=True)

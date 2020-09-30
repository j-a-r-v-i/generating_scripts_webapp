# -*- coding: utf-8 -*-

import torch.nn as nn
import torch
train_on_gpu = torch.cuda.is_available()
if not train_on_gpu:
    print('No GPU found. Please use a GPU to train your neural network.')
class RNN(nn.Module):
        # Check for a GPU
        # Check for a GPU
        
    
    
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
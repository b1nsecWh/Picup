import torch
import torch.nn as nn
import torch.nn.functional as F


class Text(nn.Module):

    def __init__(self,
        output_size,                    
        vocab_size,                
        embedding_dim,          
        input_kernel_dim=5,      
        input_kernel_sizes=(1,2),
        input_dropout=0.5,      
        ):               
        super(Text,self).__init__()

        self.embed      =   nn.Embedding(vocab_size,embedding_dim) if vocab_size else None
        self.convs1      =   nn.ModuleList([nn.Conv2d(1,input_kernel_dim,(k,embedding_dim)) for k in input_kernel_sizes])
        self.dropout1    =   nn.Dropout(input_dropout) 

        self.fc         =   nn.Linear(len(input_kernel_sizes)*input_kernel_dim,output_size)
        self.sigmod     =   nn.Sigmoid()

    
    def forward_channel_input(self,x):
        x = self.embed(x)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.convs1]
        
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x] 
        x = torch.cat(x,1) 

        y = self.dropout1(x)

        return y

    def forward(self,x):
        x_input=x.unsqueeze(1)
        x=self.forward_channel_input(x_input)

        x = self.fc(x)
        y = self.sigmod(x)
        return y

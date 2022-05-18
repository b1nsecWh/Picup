import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelAttention1d(nn.Module):
    '''Channel Attention 1-Dimension  '''
    def __init__(self,in_planes,ratio=4):
        super(ChannelAttention1d, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        self.fc = nn.Sequential(nn.Conv1d(in_planes,in_planes//ratio,1,bias=False),
                                nn.ReLU(),
                                nn.Conv1d(in_planes//ratio,in_planes,1,bias=True))
        
        self.sigmoid      =   nn.Sigmoid()
    
    def forward(self,x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SqatialAttention1d(nn.Module):
    ''' Sqatial Attention 1-Dimension'''
    def __init__(self,kernel_size=2):
        super(SqatialAttention1d, self).__init__()

        self.conv   = nn.Conv1d(2,1,1, bias=False)
        self.sigmid = nn.Sigmoid()

    def forward(self,x):
        avg_out     = torch.mean(x,dim=1,keepdim=True)
        max_out,_   = torch.max(x,dim=1,keepdim=True)

        x=torch.cat([avg_out,max_out],dim=1)
        x=self.conv(x)

        return self.sigmid(x) 


class ChannelAttention2d(nn.Module):
    ''' Channel Attention 2-Dimension '''
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention2d, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
           
        self.fc = nn.Sequential(nn.Conv2d(in_planes, in_planes // 10, 1, bias=False),
                               nn.ReLU(),
                               nn.Conv2d(in_planes // 10, in_planes, 1, bias=False))
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out)

class SpatialAttention2d(nn.Module):
    ''' Sqatial Attention 2-Dimension'''
    def __init__(self, kernel_size=7):
        super(SpatialAttention2d, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        x = self.conv1(x)
        return self.sigmoid(x)


class Text_Lenet5_CNN(nn.Module):
    def __init__(self,
        output_size,                        
        text_vocab_size,                    

        text_embedding_dim,
        file_embedding_dim,
        
        text_kernel_dim=20,
        text_kernel_sizes=(1,2,4),
        
        text_dropout=0.2,
        file_dropout=0.2,
        ):
        super(Text_Lenet5_CNN,self).__init__()


        self.embed           =  nn.Embedding(text_vocab_size+10,text_embedding_dim)
        self.conv_text       =  nn.ModuleList([nn.Conv2d(1,text_kernel_dim,(k,text_embedding_dim)) for k in text_kernel_sizes])
        
        self.convs_file_1       = nn.Sequential(
            nn.Conv2d(1,5,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2),
        )
        self.convs_file_2         = nn.Sequential(
            nn.Conv2d(5,10,5,1,2),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        
        self.dropout1    =   nn.Dropout(text_dropout) 
        self.dropout2    =   nn.Dropout(file_dropout)
        

        w = len(text_kernel_sizes)*text_kernel_dim+ 10*int((int((file_embedding_dim-2)/2+1)-2)/2+1)**2
        self.fc1        =   nn.Linear(w,w//4)
        self.fc2        =   nn.Linear(w//4,w//16)
        self.fc3        =   nn.Linear(w//16,output_size)
        self.sigmod     =   nn.Sigmoid()

    def forward_channel_text(self,x):

        x = self.embed(x)

        x = [F.relu(conv(x)).squeeze(3) for conv in self.conv_text]
        x = [F.max_pool1d(line,line.size(2)).squeeze(2) for line in x]
        x = torch.cat(x,1)
        y = self.dropout1(x)

        return y

    def forward_channel_file(self,x):
        x = self.convs_file_1(x)
        x = self.convs_file_2(x)
        x = x.view(x.size(0),-1)
        y = self.dropout2(x)

        return y
    
    def forward(self,x):

        x_text,x_file=x[0].unsqueeze(1),x[1]

        x_text = self.forward_channel_text(x_text)
        x_file = self.forward_channel_file(x_file)

        x=torch.cat([x_text,x_file],1)

        x = self.fc1(x)
        x = self.fc2(x)
        x = self.fc3(x)
        y = self.sigmod(x)

        return y

class TextCNN(nn.Module):
    def __init__(self,
        output_size,
        embedding_dim,
        vocab_size=None,
        input_kernel_dim=5,
        input_kernel_sizes=(1,2,4),
        input_dropout=0.5,
        ):               
        super(TextCNN,self).__init__()

        self.embed      =   nn.Embedding(vocab_size,embedding_dim) if vocab_size else None
        
        self.convs1      =   nn.ModuleList([nn.Conv2d(1,input_kernel_dim,(k,embedding_dim)) for k in input_kernel_sizes])
        self.dropout1    =   nn.Dropout(input_dropout) 

        self.fc         =   nn.Linear(len(input_kernel_sizes)*input_kernel_dim,output_size)
        self.sigmod     =   nn.Sigmoid()

    
    def forward_channel_input(self,x):
        if self.embed:
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

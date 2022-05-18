

import torch
import torch.nn as nn
from models import Text_Lenet5_CNN,TextCNN
from dataset import Argv_File_Dataset,Text_Dataset
from trainer import ChannelCNNTrainer
from torch.utils.data import DataLoader

def Argv_File_Train(bin_name,
                    file_encode_w=32,
                    lr=0.001,
                    betas=(0.9,0.99),
                    eposhes=4,
                    argv_embed_size=10,
                    text_kernel_dim=20,
                    text_dropout=0.2,
                    file_dropout=0.5
                    ):
 
    trainer = ChannelCNNTrainer(bin_name,epochs=eposhes)
    dataset = Argv_File_Dataset(bin_name,file_encode_w)

    # prepare model
    trainer.setDataloader(DataLoader(dataset,batch_size=1, shuffle=True))
    trainer.setModel(
        Text_Lenet5_CNN(
            dataset.standard_api_size,
            dataset.vocab_size,
            argv_embed_size,
            file_encode_w,
            text_kernel_dim=text_kernel_dim,
            text_dropout=text_dropout,
            file_dropout=file_dropout
            ))
    trainer.setOptimizer(torch.optim.Adam(trainer.model.parameters(),lr=lr,betas=betas))
    trainer.setCriterion(nn.BCELoss())

    # tain model
    trainer.train()
    trainer.show_train_plot(save=True)

    # save model
    trainer.save_model()

    # test model
    dataset = Argv_File_Dataset(bin_name,file_encode_w,type='val')
    trainer.setDataloader(DataLoader(dataset,batch_size=1))
    trainer.evaluation()

def Text_Train( bin_name,
                lr=0.001,
                betas=(0.9,0.99),
                eposhes=2,
                embedding="fasttext",
                vocab_size=None,                 
                input_kernel_dim=6,               
                input_kernel_sizes=(1,2,4),       
                input_dropout=0.3, 
                ):

 
    trainer =ChannelCNNTrainer(bin_name,epochs=eposhes)
    dataset =Text_Dataset(bin_name)

    # prepare model
    trainer.setDataloader(DataLoader(dataset,batch_size=1, shuffle=True))
    trainer.setModel(
        TextCNN(
            dataset.standard_api_size,
            dataset.fasttext.get_dimension(),
            vocab_size=vocab_size,
            input_kernel_dim=input_kernel_dim,               
            input_kernel_sizes=input_kernel_sizes,       
            input_dropout=input_dropout, 
            ))
    trainer.setOptimizer(torch.optim.Adam(trainer.model.parameters(),lr=lr,betas=betas))
    trainer.setCriterion(nn.BCELoss())
    
    # train model
    trainer.train()
    trainer.show_train_plot(save=True)

    # save model
    trainer.save_model()

    # test model
    dataset =Text_Dataset(bin_name,type='val')
    trainer.setDataloader(DataLoader(dataset,batch_size=1))
    trainer.evaluation()


if __name__ == "__main__":

    #Argv_File_Train('objdump',eposhes=5, lr=0.0005)
    #Argv_File_Train('readelf',eposhes=5, lr=0.0005)
    #Argv_File_Train('nm',eposhes=5, lr=0.0005)
    #Text_Train("nginx",eposhes=3, lr=0.0008)
    #Text_Train("lighttpd",eposhes=2, lr=0.0005)
    #Text_Train("redis",eposhes=2, lr=0.001)
    Text_Train("memcached",eposhes=2, lr=0.001)
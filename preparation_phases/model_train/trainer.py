import time
import csv
import torch
import torch.nn as nn
import math,time,csv
import matplotlib.pyplot as plt

def timeSince(since):
    now = time.time()
    s = now - since
    m = math.floor(s / 60)
    s -= m * 60
    return '%dm %ds' % (m, s)

class CommonTrainer():
    ''' Triainer farther class '''

    model=optimizer=criterion=None
    all_losses,all_acces=[],[]
    prediction_acc=[]

    def __init__(self,
                 bin_name,
                 epochs=1000,  
                 print_every=1,  
                 plot_every=10,  
                 threshold=0.5
                 ):
        self.bin_name = bin_name
        self.epochs = epochs
        self.print_every = print_every
        self.plot_every = plot_every
        self.threshold = threshold
    
    def setDataloader(self, dataloader):
        self.dataloader = dataloader

    def setModel(self, model):
        self.model = model

    def setOptimizer(self, optimizer):
        self.optimizer = optimizer

    def setCriterion(self, criterion):
        self.criterion = criterion

    def output_accuracy(self,output,target):
        _,api_len=output.size()
        sum=0
        for i in range(api_len):
            if (output[0][i]>self.threshold)==target[0][i]:
                sum+=1
        acc=sum/api_len

        return acc
    
    def train(self):
        ''' need to reload '''
        pass
    
    def show_train_plot(self,save=False):
        ''' show train plot '''
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax1.plot(self.all_losses)
        ax2 = fig.add_subplot(212)
        ax2.plot([max(self.all_acces) for _ in range(len(self.all_acces))],color='red',linestyle=':',label="max:{}".format(max(self.all_acces))) 
        ax2.plot([min(self.all_acces[int(6*len(self.all_acces)/10):]) for _ in range(len(self.all_acces))],color='green',linestyle=':',label="min:{}".format(min(self.all_acces[int(6*len(self.all_acces)/10):])))
        
        tmp=self.all_acces[int(6*len(self.all_acces)/10):]
        avg=sum(tmp)/len(tmp)
        ax2.plot([ avg for _ in range(len(self.all_acces))],color='yellow',linestyle=':',label="avg:{}".format(avg))
        

        ax2.plot(self.all_acces)
        ax2.legend()
        if save:
            plt.savefig("./imgs/{}_{}_{}.png".format(self.bin_name,self.model.__class__.__name__,time.strftime("%Y_%m_%d_%H_%M")))
        plt.show()
    
    def save_model(self, path=None):
        ''' save model '''
        if not path:
            path="./dist/models/{}_{}_{}".format(self.bin_name,self.model.__class__.__name__,time.strftime("%Y_%m_%d_%H_%M"))

        torch.save(self.model, path)
        print("[*] model save to =>{}".format(path))

    def load_model(self, path):
        ''' load model '''
        self.model=torch.load(path)
        print("[*] load model save from <={}".format(path))
    

    def evaluation(self, threshold=0.5):
        ''' test model '''
        print("==============[   evaluation     threshold:{}    ]================".format(threshold))
        output_path = "./validation/{}_evaluation_{}_{}.csv".format(self.bin_name,threshold, time.strftime("%Y_%m_%d_%H_%M"))

        with open(output_path, 'w', newline="") as f:
            csv_w = csv.writer(f)
            csv_w.writerow(['index', 'correct', 'Type I error', 'Type  II  error'])

            index=0

            acc_all,err_all=[],[]

            for step, (X, y) in enumerate(self.dataloader):
                prediction=self.model(X)
                batch_size,api_size=prediction.size()
                for i in range(batch_size):
                    correct, IError, IIEroor,T,F = 0, 0, 0, 0,0
                    for k in range(api_size):
                        if y[i][k] == 1:
                            T+=1
                        else:
                            F+=1 

                        if prediction[i][k] >= threshold:
                            if y[i][k] == 1:
                                correct += 1
                            else:
                                IIEroor += 1
                        else:
                            if y[i][k] == 0:
                                correct += 1
                            else:
                                IError += 1
                    length = api_size

                    correct, IError, IIEroor = correct / length, IError / T, IIEroor / F
                    index += 1
                    acc_all.append(correct)
                    err_all.append(IError)
                    # #print("[{}]  correct:{} IError:{:.3} IIError:{:.3}".format(index, correct, IError, IIEroor))
                    csv_w.writerow([index, correct, IError, IIEroor])

            print("ACC:",sum(acc_all)/len(acc_all))
            print("ERR:",sum(err_all)/len(err_all))

class ChannelCNNTrainer(CommonTrainer):
    def __init__(self,
                 bin_name,
                 epochs=1000,  
                 print_every=1,  
                 plot_every=10,  
                 threshold=0.5 
                 ):
        super(ChannelCNNTrainer,self).__init__(bin_name,epochs,print_every,plot_every,threshold)

    def train(self):
        start = time.time()

        for epoch in range(self.epochs):
            for step, (X,y) in enumerate(self.dataloader):
                loss,acc=0,0
                output=self.model(X)

                loss  = self.criterion(output,y) 
                acc   = self.output_accuracy(output,y)    
                if acc<0.9: print(step,",".join([str(int(x)) for x in y[0]]))
                self.optimizer.zero_grad()            
                loss.backward()                       
                self.optimizer.step()                 

                if step % self.print_every == 0:
                    print('%d  (%s) %.4f  Acc: %.4f ' % (step,timeSince(start),loss,acc))

                if step % self.plot_every == 0:
                    self.all_losses.append(loss.item())
                    self.all_acces.append(acc)
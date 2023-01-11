import torch
import torchvision.datasets as dataset
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transform
import torch.optim as optim
bach_size = 64
learning_rate = 0.001;epochs = 3
input_size = 28; num_layer = 2; num_hiden_size = 256; class_number=10
training_data=dataset.MNIST(root='dataset/', train=True,download=True,transform=transform.ToTensor())
training_loader=DataLoader(training_data, batch_size=bach_size,shuffle=True)
test_data=dataset.MNIST(root='dataset/', train=False,download=True,transform=transform.ToTensor())
test_loader=DataLoader(test_data, batch_size=bach_size,shuffle=True)


class RNN_(nn.Module):
    def __init__(self,input_size, num_layer, num_hiden_size, class_number):
        super(RNN_, self).__init__()
        self.input_size = input_size
        self.num_layer = num_layer
        self.num_hiden_size = num_hiden_size
        self.class_number = class_number
        #self.rnn=nn.RNN(input_size=self.input_size,num_layers=self.num_layer,hidden_size=self.num_hiden_size,batch_first=True)
        #self.gru=nn.GRU(input_size=self.input_size,num_layers=self.num_layer,hidden_size=self.num_hiden_size,batch_first=True)
        self.lstm = nn.LSTM(input_size=self.input_size, num_layers=self.num_layer, hidden_size=self.num_hiden_size, batch_first=True, bidirectional=True)
        self.fc=nn.Linear(num_hiden_size*2,self.class_number)
    def forward(self,x):
        # h0=torch.zeros(self.num_layer,x.shape[0],self.num_hiden_size)
        # c0=torch.zeros(self.num_layer,x.shape[0],self.num_hiden_size)
        h0=torch.zeros(2*self.num_layer,x.shape[0], self.num_hiden_size)
        c0=torch.zeros(2*self.num_layer,x.shape[0], self.num_hiden_size)
        #out,_=self.rnn(x,h0)
        #out, _ = self.gru(x, h0)
        out, _ = self.lstm(x, (h0,c0))
        # out=out.reshape(out.shape[0],-1)
        print(out.shape)
        out=self.fc(out[:,-1,:])
        return out
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model=RNN_(input_size,num_layer,num_hiden_size,class_number).to(device=device)
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
Criterion=nn.CrossEntropyLoss()
for epoch in range(epochs):
    for _, (data,target) in enumerate(training_loader):
        data=data.squeeze(1)
        data=data.to(device=device)
        target=target.to(device=device)
        optimizer.zero_grad()
        prediction=model(data)
        loss=Criterion(prediction,target)
        loss.backward()
        optimizer.step()
def accuracy(model,data):
    num_correct=0
    num_sample=0
    for _,(data,target) in enumerate(data):
        data=data.squeeze(1)
        data=data.to(device=device)
        target=target.to(device=device)
        source=model(data)
        _,prediction_=source.max(1)
        num_correct+=(prediction_==target).sum()
        num_sample+=prediction_.shape[0]
    print(f'got {num_correct}/{num_sample} with accutacy of {float(num_correct/num_sample)*100:2f}')

accuracy(model,training_loader)
accuracy(model,test_loader)
import torch
import torch.nn as nn
import torch.functional as F
import torch.utils.data as dataset
import  torchvision
import torchvision.transforms as transform
import torch.optim as optim
from torch.utils.data import dataloader


class NN(nn.Module):
    def __init__(self,input_size,class_size):
        super(NN,self).__init__()
        self.f1=nn.Linear(input_size,50)
        self.f2=nn.Linear(50,class_size)
    def forward(self,x):

        x=torch.nn.functional.relu(self.f1(x))
        x=self.f2(x)
        return x
# device
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#hyperparameter
batch_size=64
learning_rate=10**(-3)
input_size=784
class_size=10
epoch=10
# dataset
data_training=torchvision.datasets.MNIST(root='dataset/',train=True,transform=transform.ToTensor(),download=True)
training_loader=dataset.DataLoader(dataset=data_training,batch_size=batch_size,shuffle=True)
data_testing=torchvision.datasets.MNIST(root='dataset/',train=False,transform=transform.ToTensor(),download=True)
testing_loader=dataset.DataLoader(dataset=data_testing,batch_size=batch_size,shuffle=True)
# model=NN(input_=input_size,output_=class_size).to(device)
model=NN(input_size=input_size,class_size=class_size).to(device)

#loss,optimazer
Criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)

for epoch_ in range(epoch):
    for index_, (data,target) in enumerate(training_loader):
        data=data.to(device=device)
        target=target.to(device=device)
        data=data.reshape(data.shape[0],-1)
        source=model(data)
        loss=Criterion(source,target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
def check_arracy(model,load_):
    num_correct=0
    num_samples=0
    model.eval()
    with torch.no_grad():
        for _,(data,target) in enumerate(load_):
            data=data.to(device=device)
            target=target.to(device=device)
            data=data.reshape(data.shape[0],-1)
            num_samples+=data.shape[0]
            source_=model(data)
            _,prediction=source_.max(1)
            num_correct+=(prediction==target).sum()
        print(f'got {num_correct}/{num_samples} with accuracy {float(num_correct)/float(num_samples)*100:.2f}')
        model.train()
check_arracy(model,training_loader)
check_arracy(model,testing_loader)













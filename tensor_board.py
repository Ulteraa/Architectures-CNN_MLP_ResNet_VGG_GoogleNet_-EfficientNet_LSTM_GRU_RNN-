import torch
import torch.nn as nn
import torch.functional as F
import torchvision.datasets
import torchvision.datasets as dataset
import torch.optim as optim
import torchvision.transforms as transform
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
class CNN(nn.Module):
    def __init__(self,input_size=1,class_number=10):
        super(CNN, self).__init__()
        self.conv1=nn.Conv2d(in_channels=input_size,out_channels=8,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.pool=nn.MaxPool2d(kernel_size=(2,2),stride=(2,2))
        self.conv2=nn.Conv2d(in_channels=8,out_channels=16,kernel_size=(3,3),stride=(1,1),padding=(1,1))
        self.fc=nn.Linear(16*7*7,class_number)
    def forward(self,x):
        x=nn.functional.relu(self.conv1(x))
        x=self.pool(x)
        x=nn.functional.relu(self.conv2(x))
        x=self.pool(x)
        x=x.reshape(x.shape[0],-1)
        x=self.fc(x)
        return  x
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
#hyperparameter
batch_size=64
learning_rate=0.001
epochs=1
model=CNN().to(device=device)
writer=SummaryWriter()
#loss and optimizer
Criterion=nn.CrossEntropyLoss()
optimizer=optim.Adam(model.parameters(),lr=learning_rate)
dataset_training=torchvision.datasets.MNIST(root='dataset/',train=True,transform=transform.ToTensor(),download=True)
train_loader=DataLoader(dataset=dataset_training,batch_size=batch_size,shuffle=True)
dataset_testing=torchvision.datasets.MNIST(root='dataset/',train=False,transform=transform.ToTensor(),download=True)
test_loader=DataLoader(dataset=dataset_testing,batch_size=batch_size,shuffle=True)
step=0
class_=['0','1','2','3','4','5','6','7','8','9']
for epoch in range(epochs):
    for index_, (data,target) in enumerate(train_loader):

        data=data.to(device=device)
        target=target.to(device=device)
        features=data.reshape(data.shape[0],-1)

        prediction=model(data)
        pre_,lables=prediction.max(1)
        optimizer.zero_grad()
        loss=Criterion(prediction,target)
        class_lables=[class_[i] for i in lables]
        loss.backward()
        optimizer.step()
        image_grid_=torchvision.utils.make_grid(data)
        writer.add_scalar('loss',loss,global_step=step)
        writer.add_image('image',image_grid_)
        writer.add_histogram('weight',model.fc.weight,global_step=step)

        writer.add_embedding(features,metadata=class_lables,label_img=data,global_step=step)
        step+=1


def accuracy(model, data):
    correct_sample = 0
    sample_number = 0
    model.eval()
    for _, (data_, target) in enumerate(data):
        data_ = data_.to(device=device)
        target = target.to(device=device)
        prediction_ = model(data_)
        _, m = prediction_.max(1)
        correct_sample += (m == target).sum()
        sample_number += prediction_.shape[0]
    print(f'got {correct_sample}/{sample_number} with accuracy {float(correct_sample / sample_number) * 100:.2f}')


accuracy(model, train_loader)
accuracy(model, test_loader)

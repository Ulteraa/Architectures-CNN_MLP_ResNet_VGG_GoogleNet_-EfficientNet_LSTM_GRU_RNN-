import torch
import pandas
import numpy as np
# print(torch.__version__)


device='cuda' if torch.cuda.is_available() else 'cpu'
tensor_=torch.tensor([[1,2,3],[4,5,6]],dtype=float,device=device,requires_grad=True)
tensor=torch.empty(size=(3,2))
# x=torch.zeros_like(tensor)
tensor=torch.normal(mean=0.0,std=10,size=(3,2))
tensor=torch.ones((3,4))
tensor=torch.eye(2)
tensor=torch.arange(start=0,end=10)
tensor=torch.linspace(start=0,end=1,steps=10)
tensor=torch.empty((4,5)).normal_(mean=0.0,std=1)
tensor=torch.empty((3,4)).uniform_(0,1)
array_=np.ones((10,10))
tensor=torch.from_numpy(array_)
back_to_array=tensor.numpy()
############## operation
tensor_1=torch.tensor([[1,2,3,4],[4,5,6,7]])
tensor_2=torch.tensor([[10,20,30,40],[40,50,60,70]])
tensor=tensor_2+tensor_1
z=torch.add(tensor_1,tensor_2)
z=tensor_2-tensor_1
z=torch.sub(tensor_2,tensor_1)
z=tensor_1/tensor_2
z=tensor_2*tensor_1
z=tensor_2/2
z=torch.true_divide(tensor_2,2)
# tensor_1.add_(tensor_2)
# print(tensor_1.pow(2.1))
bach=9
x=torch.rand(bach,3,3)
y=torch.rand(bach,3,3)
# z=torch.bmm(x,y)
s=torch.sum(tensor_1,dim=1)
# print(torch.min(tensor_1,dim=1))
# print(torch.argmax(tensor_1,dim=0))
z=torch.clamp(tensor_1,min=2,max=3)
x=torch.arange(10)
# print(x[x>2 | x<8])
x=torch.tensor([1,2,1,1,2,3,4,5,1])
x=x.unsqueeze(0).unsqueeze(1)
print(x.shape)
# print(torch.where(x>2,x,2*x))
# print(x.unique())
# new_tensor=torch.tensor([1,2,2,3,56,6,7,8,10])
# new=new_tensor.view(3,3)
# new_=new_tensor.reshape((3,3))
# print(tensor_1.shape,tensor_2.shape)
# print(torch.cat((tensor_1,tensor_2),dim=1))
# print(tensor_1.view(-1))
# print(new_)
# # torch.transpose(new_)
# print(new_.transpose())
# print(torch.any(z))
# print(torch.all(z))
# index=[0,1]
# print(tensor_1[index,0])
# # # print(x)
# # z=x.mm(y)
#
# print(s)
# # print(tensor_1)



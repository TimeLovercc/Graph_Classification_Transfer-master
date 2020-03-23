import torch
from torch.autograd import Variable


#数据准备
x11 = torch.linspace(1,10,10)
x12 = torch.rand(10) * 10
x1 = torch.cat((x11,-x11,-x11,x11))
x2 = torch.cat((x12,x12,-x12,-x12))
y = torch.ones(1,10)
y = torch.cat((y,-y,y,-y),1)


a = Variable(torch.rand(1),requires_grad=True)
b = Variable(torch.rand(1),requires_grad=True)
c = Variable(torch.rand(1),requires_grad=True)

learning = 0.0001

for i in range(10000):
    if (a.grad is not None) and (b.grad is not None) and (c.grad is not None):
        a.grad.data.zero_()
        b.grad.data.zero_()
        c.grad.data.zero_()

    y1 = x1 + x2 + a.expand_as(x1)
    y2 = x1 + x2 + b.expand_as(x1)
    y3 = torch.abs(y1 + y2) + c.expand_as(x1)

    loss = torch.mean((y3 - y)**2)
    loss.backward()
    a.data.add_(-learning*a.grad.data)
    b.data.add_(-learning*b.grad.data)
    b.data.add_(-learning * b.grad.data)

print(a.data,b.data,c.data)


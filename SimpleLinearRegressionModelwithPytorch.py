#import> required pakage

import numpy as np
import matplotlib.pyplot as plt
import torch
import torch.nn as nn


X = torch.linspace(1,50,50).reshape(-1,1)
#print(X)
torch.manual_seed(71)
e = torch.randint(-8,9,(50,1), dtype=torch.float)
#print(e)
y= 2*X+1 + e
#print(y)
print(y.shape)
plt.figure()
plt.scatter(X.numpy(), y.numpy())
plt.show()


# general model

class Model(nn.Module):
    def __init__(self,in_features, out_features):
        super().__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, X):
        y_pred = self.linear(X)
        return y_pred

torch.manual_seed(59)
model = Model(1,1)
print(model.linear.weight)
print(model.linear.bias)

for name,param in model.named_parameters():
    print(name, param.item())




#Model execution without loss and optimization calculation
x1 = np.linspace(0.0,50.0,50)
print(x1)
w1= .1059
b1= .9637
y1 = w1*x1 + b1
print(y1)

plt.figure()
plt.subplot(2,1,1)
plt.scatter(X.numpy(), y.numpy())
plt.plot(x1,y1, 'r')
plt.title('Model predict without loss and optimization calculation ')
#plt.show()



#model execute with loss and iptimization calculation

criterion =nn.MSELoss ()
optimizer = torch.optim.SGD(model.parameters(),lr= 0.001)
epochs =50
losses =[]
for i in range (epochs):
    i = i+1
    # predicting on the forwardpass
    y_pred = model.forward(X)
    #calculate our loss
    loss = criterion (y_pred, y)
    #record the error
    losses.append(loss)
    print (f'epoch {i} loss : {loss.item()} weight: {model.linear.weight.item()} bias: {model.linear.bias.item()}')
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

x = np.linspace(0.0, 50.0, 50)
current_weight = model.linear.weight.item()
current_bias = model.linear.bias.item()

predicted_y = current_weight * x + current_bias
print(predicted_y)

plt.subplot(2,1,2)
plt.scatter(X.numpy(), y.numpy())
plt.plot(x, predicted_y, "r")
plt.title('Model predict with loss and optimization calculation ')
plt.show()

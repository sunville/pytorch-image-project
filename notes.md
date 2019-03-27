# BASIC STRUCTURE

```python
net.zero_grad()
output = net(input)
error = criterion(output,label)
error.backward()
optimizer.step()
```
#### Flow Graph
Pytorch construct a flow graph while calculation.\
It feed forward when you call forward().\
It backpropagation when you call backward(). Then the graph will be deleted. Unless retain_graph=True.\
It learns (descent gradient) when you call step().

#### nn.zero_grad()
Empty the gradients at each node. Otherwise, the gradient will accumulate.

#### torch.autograd.backward()
Computes the sum of gradients of given tensors w.r.t. graph leaves.\
Just compute gradients!!!!

#### optimizer.step()
Do the gradient descent.

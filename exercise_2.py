import torch
import torch.nn as nn
import matplotlib.pyplot as plt
import pandas as pd

# 1: Load the dataset as x and y tensors
data = pd.read_csv("polynomial_data.csv")
x = torch.tensor(data['x'].values, dtype=torch.float32).view(-1, 1)
y = torch.tensor(data['y'].values, dtype=torch.float32).view(-1, 1)


# 2: Define the class of the cubic polynomial model
class CubicModel(nn.Module):
    def __init__(self):
        super(CubicModel, self).__init__()
        self.a = nn.Parameter(torch.randn(1, requires_grad=True))
        self.b = nn.Parameter(torch.randn(1, requires_grad=True))
        self.c = nn.Parameter(torch.randn(1, requires_grad=True))
        self.d = nn.Parameter(torch.randn(1, requires_grad=True))

    def forward(self, x_input):
        return self.a * x_input ** 3 + self.b * x_input ** 2 + self.c * x_input + self.d


# 3: Instantiate the model, define loss function and optimizer
model = CubicModel()

# I decided to start with very basic MSE loss, and it appeared to work nicely
criterion = nn.MSELoss()

# I tried different values for learning rate to achieve accuracy close to the analytical approach
# I started with 0,01, and it seemed to learn a bit too slowly
# in the end, something around 15 gives coefficient values very close to analytical approach
# However, loss is still noticeably higher compared to analytical approach
optimizer = torch.optim.Adam(model.parameters(), lr=2)

# 4: Train the model and also print loss every 100 epochs
# I tried different numbers of epochs, 3000 appeared to be the most optimal one
epochs = 8900
for epoch in range(epochs):
    y_pred = model(x)
    loss = criterion(y_pred, y)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    if (epoch + 1) % 100 == 0:
        print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss.item()}")

# 5: Extract learned coefficients
a, b, c, d = model.a.item(), model.b.item(), model.c.item(), model.d.item()
print(f"Estimated coefficients: a = {a}, b = {b}, c = {c}, d = {d}")

# 6: Visualize the curve fitting
plt.scatter(x.numpy(), y.numpy(), color='blue', label='Original data')
x_sorted, _ = torch.sort(x, dim=0)
y_fitted = model(x_sorted).detach().numpy()
plt.plot(x_sorted.numpy(), y_fitted, color='red', label='Fitted curve')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Cubic Polynomial Fit")
plt.show()
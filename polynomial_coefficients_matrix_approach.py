import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

# 1: Load data
data = pd.read_csv("inputs/polynomial_data.csv")
x = data['x'].values
y = data['y'].values

# 2: Parse input data into Vandermonde matrix
degree = 3  # Degree of the polynomial
vandermonde_matrix = np.vstack([x**i for i in range(degree + 1)]).T

# 3: Compute coefficients using the OLS formula
gram_matrix = np.linalg.inv(vandermonde_matrix.T @ vandermonde_matrix)
moment_matrix = vandermonde_matrix.T @ y
coefficients = gram_matrix @ moment_matrix  # gives us the vector of estimated coefficients

# Since in the matrix representation of our polynomial coefficients go from β0(d) to β4(a),
# I ensure their correct assignment to a, b, c, d
a, b, c, d = coefficients[3], coefficients[2], coefficients[1], coefficients[0]
print(f"Estimated coefficients: a = {a}, b = {b}, c = {c}, d = {d}")

# 4: Generate predictions and compare them to input data using MSE
y_pred = vandermonde_matrix.dot(coefficients)
mse = np.mean((y - y_pred)**2)
print("Mean Squared Error MSE:", mse)

# 5: Visualize the curve fitting
plt.scatter(x, y, color='blue', label='Original data')
plt.plot(np.sort(x), y_pred[np.argsort(x)], color='red', label='Fitted curve')
plt.xlabel("x")
plt.ylabel("y")
plt.legend()
plt.title("Polynomial Regression Fit")
plt.show()


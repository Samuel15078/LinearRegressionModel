import pandas as pd
import matplotlib.pyplot as plt

# Load dataset
data = pd.read_csv('HoursScores.csv')

# Define loss function (optional for debugging)
def loss_function(m, b, points):
    total_error = 0
    for i in range(len(points)):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        total_error += (y - (m * x + b)) ** 2
    return total_error / float(len(points))

# Gradient descent function
def gradient_descent(m_now, b_now, points, L):
    m_gradient = 0
    b_gradient = 0
    n = len(points)

    for i in range(n):
        x = points.iloc[i].Hours
        y = points.iloc[i].Scores
        m_gradient += -(2/n) * x * (y - (m_now * x + b_now))
        b_gradient += -(2/n) * (y - (m_now * x + b_now))

    m = m_now - m_gradient * L
    b = b_now - b_gradient * L
    return m, b

# Initialize variables
m = 0
b = 0
L = 0.0001
epochs = 300

# Train model
for i in range(epochs):
    if i % 50 == 0:
        print(f"Epoch: {i}, Loss: {loss_function(m, b, data)}")
    m, b = gradient_descent(m, b, data, L)

# Print final values
print(f"\nTrained model parameters:\nm = {m:.4f}, b = {b:.4f}")

# Plotting the result
plt.scatter(data.Hours, data.Scores, color="black")
plt.plot(list(range(0, 10)), [m * x + b for x in range(0, 10)], color="red")
plt.xlabel("Hours Studied")
plt.ylabel("Score")
plt.title("Linear Regression Fit")
plt.show()

# --- User Input and Prediction ---
try:
    user_input = float(input("Enter hours studied: "))
    predicted_score = m * user_input + b
    print(f"Predicted score: {predicted_score:.2f}")
except ValueError:
    print("Please enter a valid numeric value.")

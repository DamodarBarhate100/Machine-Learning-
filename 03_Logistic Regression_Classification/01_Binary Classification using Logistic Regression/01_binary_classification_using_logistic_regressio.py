import numpy as np
import matplotlib.pyplot as plt
# X = [Annual Income ($), Credit Score]
X_train = np.array([
    [25000, 400], [30000, 550], [35000, 450], [40000, 500], 
    [45000, 550], [50000, 520], [55000, 480], [60000, 600],
    [65000, 650], [70000, 750], [75000, 800], [85000, 680], 
    [90000, 720], [100000, 780], [120000, 820]
])

# y = Approved (1) or Denied (0)
y_train = np.array([0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1])


x_mean = X_train.mean(axis=0)
x_std = X_train.std(axis=0)
x_scaled = (X_train-x_mean)/x_std

def sigmoid(z):
    return 1/(1+np.exp(-z))

def logistic_regression(x,y):
    m = len(y)
    x_with_bias = np.column_stack((np.ones(shape=m),x))
    n = x_with_bias.shape[1]
    theta_j = np.zeros(shape=n)
    learning_rate = 0.5
    iterations = 1000
    for i in range(iterations):
        y_predicted  = sigmoid(theta_j@x_with_bias.T)
        errors = y - y_predicted
        epsilon = 1e-15 # A very small number
        # as we know log 0 is undefined so adding a very small number to log so that i can't be zero
        l_theta_cost = np.sum((y*np.log(y_predicted+epsilon)) + ((1-y) * np.log(1-y_predicted+epsilon)))

        gradient = errors.T @ x_with_bias
        # The division by m (the number of examples) is standard practice to help stabilize the learning rate.
        theta_j = theta_j + learning_rate * gradient/m

        if (i%20==0):
            print("cost:{}  theta_j:{}  i:{}".format(l_theta_cost,theta_j,i))


    return theta_j

theta = logistic_regression(x_scaled,y_train)
print("\nFinal Parameters are:")
print("intercept (b):",theta[0])
print("Weight 1:",theta[1])
print("Weight 2:",theta[2])



# predicting the same values using X_train dataset
def predict(theta,x_scaled,threshold=0.5):
    x_scaled_with_bias = np.column_stack((np.ones(shape=x_scaled.shape[0]),x_scaled))
    result = sigmoid(theta@x_scaled_with_bias.T)
    # it will convert it into boolean and then astype will convert them into 0 and 1
    return (result >= threshold).astype(int)


probabilities = predict(theta,x_scaled)
print("\nPredicted the value 0 and 1 using the threshold of 0.5:")
print(probabilities)



# 1. Get the range of x values (Income) from your data
# here we take the minimum and maximum point on x axis and put it into array
# and  then -0.5 from minimum to stretch in negative direction and added +0.5 to stretch into positive direction
x_values = np.array([np.min(x_scaled[:, 0]) - 0.5, np.max(x_scaled[:, 0]) + 0.5])

# as we are plotting the data into 2 d we have to x1 is on x axis and x2 is on y axis 
# x2 = -(theta_0 + theta_1 * x1) / theta_2
y_values = -(theta[0] + theta[1] * x_values) / theta[2]


# # # Visualizind data
loan_approved = x_scaled[y_train==1]
loan_not_approved = x_scaled[y_train==0]

plt.scatter(loan_approved[:,0],loan_approved[:,1],color='green',marker='o',label='loan approved')
plt.scatter(loan_not_approved[:,0],loan_not_approved[:,1],color='salmon',marker='x',label='loan not approved')
plt.plot(x_values, y_values, label='Decision Boundary', c='blue')
plt.xlabel("Loan Approved data")
plt.ylabel("Loan Not Approved data")
plt.grid(True,linestyle='--',alpha=0.3)
plt.legend()
plt.tight_layout()
plt.show()
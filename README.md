
# Ethereum Price Prediction Using MLP(Multilayer perceptron) Regression Models



### Problem Statement

This study aims to forecast Ethereum (ETH) prices in Indian Rupees (INR) using MLP 
Regression, emphasizing dimension and feature engineering methodologies. The primary 
challenge is to develop accurate predictive models that anticipate Ethereum price movements, 
including changes in price signs, to support informed investment decisions.

### Data

A single dataset for Ethereum (ETH) prices in Indian Rupees (INR), 
collected from Kaggle. The dataset spans from January 2018 to July 2021 and contains essential 
attributes such as date, open price, high price, low price, close price, adjusted close price, and 
trading volume.

### Feature engineering

The selected features used in our model include 'Year', 'Month', 'Day', 'Open', 'High', and 'Low'. 
These features represent temporal aspects (year, month, day) as well as daily price fluctuations 
(open, high, low) of Ethereum

```python
import Component from 'my-project'

function App() {
  return <Component />
}
```

The 'Date' column in the dataset contains timestamps representing the date and time when each 
observation was recorded. Extracting the year, month, and day from the 'Date' column allows 
the model to capture potential seasonal or yearly patterns in Ethereum price movements. 

```python
import Component from 'my-project'

function App() {
  return <Component />
}
```

### Model Implementation: 
- The dataset is split into training and testing sets, with 80% allocated for training and 20% for testing.  
Notably, the split was done sequentially to preserve the temporal sequence of the data. Random splitting may bias the evaluation of model performance, especially if there are systematic changes in the data over time. 

- To ensure model stability and convergence, features are standardized using the StandardScaler to achieve zero mean and unit variance. 

- The MLP regressor is instantiated with specific hyperparameters, in our implementation, two hidden layers are configured with 100 and 50 neurons, and the "ReLU" activation function is utilized. 

- The "Adam" solver is employed for optimization, with a maximum of 1000 iterations allowed for model convergence.

- The MLP regressor is trained on the scaled training data to learn the relationship between the input features and the target variable. 

- Predictions are subsequently made on the scaled test set. 


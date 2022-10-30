# Car-resale-value-predictor
# Objective
The main objective of this project is to predict resale value of a car by training a machine learning model on historical data.
# About the dataset
The dataset is downloaded from Kaggle, it has 4340 records. Selling_price is the target variable for us in the dataset. There were no null values in the dataset. There are 8 columns in the dataset. 

# Data pre-processing 
From the feature kms driven the outliers were removed from our dataset using IQR, after removing finally we have 4230 records in our dataset.

The name of the cars are encoded with the help of Label encoder. Since there are 1491 different type of car names if we use pandas get dummies it is adding about 1490 new columns in our dataset which is becoming computationally quite heavy. 

The other categorical varaiables are encoded with the help of get_dummies of pandas. 

MinMax scaler is used to scale the varaiables in range of 0 to 1. 

# Conclusion and Methodology

First I tried to fit a linear regressor model where I got a testing accuracy of 46.044 percent. 

Since our target varaiable selling_price has a highly skewed distribution we applied a log transformation on the target, which increased our testing accuracy to 58.044 percent for a linear regression which is an increase of about 10 percent from our previous testing accuracy. 

An unoptimized random forest regressor is giving a testing accuracy of 48.96 percent. So I tried optimizing our random forest regressor in two steps. First i tried to optimize the depth of the tree and got a training accuracy of 96.16 percent and a testing accuracy of 76.85 percent at a depth of 12. From the difference of the testing accuracy and the training accuracy it was quite evident that my model tends to overfit. So I chose the hyperparameter min_sample_leaf as 5 and got a traning accuracy of 81.94 percent and a testing accuracy of 77.13 percent at a depth of 18. 

# Kaggel---Bike-Share-Demand

This was a competition on Kaggel to Forecast use of a city bikeshare system.
The dataset provides hourly rental data spanning two years. For this competition, the training set is comprised of the first 19 days of each month, while the test set is the 20th to the end of the month. 
The Objective is to predict the total count of bikes rented during each hour covered by the test set, using only information available prior to the rental period.

## Data Fields
datetime - hourly date + timestamp  
season -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
holiday - whether the day is considered a holiday
workingday - whether the day is neither a weekend nor holiday
weather - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog 
temp - temperature in Celsius
atemp - "feels like" temperature in Celsius
humidity - relative humidity
windspeed - wind speed
casual - number of non-registered user rentals initiated
registered - number of registered user rentals initiated
count - number of total rentals

## Data Analysis & Model Building
Looking at the data initially, everything seemed good with all the columns in Numerical Datatypes and one column with datetime stamp. No Null/Nan or duplicated values were present in the dataset. However few columns were actually categorical and having them in numerical (int/float) format will create an incorrect influence while training the model. Hence it was converted into categorical and further encoded to make them readable for the ML model. 

The target variable "Count" was a combination of "Casual" + "Registered", which must the user type. With all the columns rightly pre-processed, the data was ready to be trained for the regression model.

The approach here was to train and test the model separately for the "Casual" & "Registered" and combine the result to form the final submission "Count". The models were trained with Random Forest Regressor and Light Gradient Booting Method Regressor with Voting Ensemble technique. Basically, A voting ensemble is an ensemble machine learning technique that combines the predictions from multiple models, thus involves calculating the average of the predictions from the models, in case of regressions.

### The results obtained from this approach was satisfactorily good, with Kaggel public leader board score of 0.55

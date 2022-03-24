# Predict_Credit_Risk
Building a machine learning model that attempts to predict whether a loan will become high risk or not. Used sklearn, and pandas. 

## Machine Learning Steps

<p>
In the given 2019 and 2020 Q1 data frames, not all of the column values were numeric. None numeric columns lead to an issue because if the column values are not numeric,  it fails to meet the standards of machine learning models. As we can see in the train data frame, we have seven columns that fail to hold numeric values only. 
</p>

![2019 Train Table](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/train_df_aka_2019.png)

![2020 Q1 Test Table](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/test_df_aka_2020_q1.png)

<p>
Both data frames have over fifty columns. The following code will help break down the columns with object values only. Object values are the same thing as string values. 
</p>

<p>
To convert the non-numeric columns is apply the get_dummies() method.
</p>

#### <u> pd.get_dummies() </u>

<p>
Here is an example of how pd.get_dummies() work.
</p> 

![Get_Dummies() Part 1](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/example_get_dummies_part_1.png) <p> The following data frame is named <i> preview_get_dummies </i> displays two non-numeric columns. </p>

<p>
Then apply the get_dummies() code: pd.get_dummies(<i> preview_get_dummies </i>)
</p>

![Get_Dummies() Part 2](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/example_get_dummies_part_2.png)

<p>
Now the <i> preview_get_dummies </i> data frame has numeric columns which means it meets the standards of machine learning models.
</p>

<p>
After converting the non-numeric columns into numeric columns, the data frame needs to have an x_value and y_value, especially in machine learning models. The x_value and y_value for this case will be the cleaned-up data frames. The y_value will hold the 'loan_status' object values. The x_values will have the whole data frame with numeric values only. The x_value should drop the 'loan_status' column or the model will fail. 
</p>

![Y_values List](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/y_values_list.png)

![X values List](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/x_values_list.png)

<p>
Once the x and y values are assigned to train and test, the x values must be in the same shape. If the train and test shape are not equal, the whole model fails. The data frame of the test had 91 columns. The data frame of the train had 92 columns. Therefore, the two data frames cannot be used for the models. To solve this issue is running a for a loop by its columns. If its columns are not found in its data frame,  it would have a column that holds numeric values of 0. The image below will display the code. 
</p>

![Code Fill Missing Columns](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/code_to_fill_missing_columns.png)

### LogisticRegression Model() and RandomForestClassifier() without StandardScaler


<p>
The logistic regression model needs four values for fit and score. The fit values will end up as the train. Score values will end up as the test. The solver will be equal to 'lbfgs' and random_state will be 1. The random state sets seed to the random generator. Depending on the random state will determine the train-test splits. If the random_state is not set, the outcome runs differently. LBFGS stands for "Limited-memory Broyden–Fletcher–Goldfarb–Shanno Algorithm" and is one of the algorithms in the Scikit-Learn library. 
</p>

![Logistic Code Model Code](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/logitic_regression_model_code.png) 

<p>
After applying the logistic code, the train values need to be fitted under the logistic code. The fit method is equal to or the same as training. Once trained, the model can be used to make predictions. It will try to find coefficients that will fit the equation defined via the algorithm being used. Again, applying the fit method will allow the model to use the predict method. 
</p>

<p>
The code after fit method will be the scoring method. The score method takes a feature matrix X_test and the expected target values y_test. It will predict the x test while comparing it with the Y test and return the accuracy. The accuracy can be called the R squared score which is the regression estimator. The accuracy outcome is 0.5168013611229264. An accuracy of 0.52 is poor and the model is not accurate. 
</p>

![Random Forest Classifier Code](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/random_forest_classifier_code.png)

<p>
Random forest classifier is another model that decision trees based on a random selection of data samples. It will get predictions from every tree and selects the most promising solution through votes. It uses the fit and score method also. The outcome of the score is 0.6424925563589962. Random forest classifier accuracy is stronger and higher than the logistic regression model, but still weak. 
</p>

### LogisticRegression Model() and RandomForestClassifier() with StandardScaler

![Standard Scaler Code](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/standard_scaler_code.png)

<p>
To get better accuracy for our model, apply the scaler method. The point of StandardScaler is transforming the data to have a mean value of 0 and a standard deviation of 1. Applying the StandardScaler should improve the accuracy of the LogisticRegression Model() and RandomForestClassifier().
</p>

![Logistic Regression Model with Scaler](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/logitic_regression_model_code_with_scaler.png)

<p>
The LogisticRegression Model() will have the same code except X_train and X_test will be replaced by X_train_scaled and X_test_scaled. After replacing the test and train values, the accuracy improved from 0.5168013611229264 to 0.767333049766057. The score increased by 22. 
</p>

![Random Forest Classifier with Scaler](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/random_forest_classifier_code_with_scaler.png) Random Forest Classifier_code_witH_scaler

<p>
The RandomForestClassifier Model() will have the same code except X_train and X_test will be replaced by X_train_scaled and X_test_scaled. After replacing the test and train values, the accuracy decreased from 0.6424925563589962 to 0.6339855380689069. The standardscaler did not improve the RandomForestClassifier. 
</p>


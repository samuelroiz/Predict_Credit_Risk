# Predict_Credit_Risk
<p>
Built a machine learning model that attempts to predict whether a loan will become high risk or not for a company named LendingClub. LendingClub is a peer-to-peer lending services company that allows individual investors to partially fund personal loans as well as buy and sell notes backing the loans on a secondary market. LendingClub offers their previous data through an API. Utilized the given data from LendingClub to create machine learning models to classify the risk level of loans. Compared the Logistic Regression model and Random Forest Classifier with sklearn and pandas. 
</p>





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
  Then apply the get_dummies() code: <b> pd.get_dummies(<i> preview_get_dummies </i>) </b>
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

#### How does the Standard scaler work and what is it doing to the data? 

<p> Standard Scaler has a formula, let's assume the formula is <b> z = (x - u) / s </b> where <b> z </b> is the scaled data, <b> x </b> is to be the scaled data, <b> u </b> is the average of the training samples, and <b> s </b> is the standard deviation of the training samples. The average or mean is the sum of all data points divided by the number of data points. In this case, <b> u </b> is going to add up all of the training samples divided by several training samples. The standard deviation formula starts by taking the given values and placing them in one column. Square each value and place them in a second column. Find the sum of all values in the first column and square it. The value is divided by the number of data points in first the column and called this number <b> i </b>. Find the sum of all values in the created second column. Once find the value, subtract it by <b> i </b>. The outcome will be divided by the number of data points minus one. Value leads to the <b> variance </b> of the sample and data. Finally, the variance will be square rooted leading to the value standard deviation of the data. 
</p>

#### Example of Standard Scaler
<p>
To demonstrate the algorithm and how it functions, consider the data set {1,2,3,4,5}. The data set consists of 5 one dimensional data points and each data point has one feature. Now apply the standard scaler() to the data. The data set becomes {−1.41,−0.71,0.,0.71,1.41}.
</p>

<p>
  This is how the math works behind the Standard Scaler. <br>
  &nbsp;The steps to apply StandardScaler to the data set [1,2,3,4,5]: <br>
  &nbsp;<ol>
    <li>Calculate the mean of the feature vector X:</li> <br>
    <b> mean = (1 + 2 + 3 + 4 + 5) / 5 = 3 </b> <br> <br>
    <li> Calculate the standard deviation of the feature vector X: </li> <br> 
    <b> std_dev = sqrt( ((1 - 3)^2 + (2 - 3)^2 + (3 - 3)^2 + (4 - 3)^2 + (5 - 3)^2) / 5 ) = 1.4142 </b> <br> <br>
    <li> Subtract the mean from each element of X: </li> <br>
    <b> X = [1 - 3, 2 - 3, 3 - 3, 4 - 3, 5 - 3] = [-2, -1, 0, 1, 2] </b> <br> <br> 
    <li> Divide each element of X by the standard deviation: </li> <br>
    <b> Z = [-2 / 1.4142, -1 / 1.4142, 0 / 1.4142, 1 / 1.4142, 2 / 1.4142] = [-1.4142, -0.7071, 0, 0.7071, 1.4142] </b> <br> <br>
    </ol>
 </p>
 <p>  So the standardized feature vector <b> Z is [-1.4142, -0.7071, 0, 0.7071, 1.4142]</b>. </p>
 <p> Note that when applying StandardScaler to a data set, the <b> fit </b> method is used to calculate the mean and standard deviation, and the <b> transform </b> method is used to apply the scaling to the data. In this case, since we only have one feature, we could use the <b> fit_transform method </b> to combine these two steps: </p>
 
<p> 
   <b> from sklearn.preprocessing import StandardScaler <br> 
   import numpy as np <br> 
   X = np.array([1, 2, 3, 4, 5]).reshape(-1, 1) <br>
  scaler = StandardScaler() <br>
  X_std = scaler.fit_transform(X) <br> 
  print(X_std) </b> 
</p>

<p> This will produce the output: </p>
<p> <b>
[[-1.41421356] <br>
 [-0.70710678] <br>
 [ 0.        ] <br>
 [ 0.70710678] <br>
  [ 1.41421356]] </b>
</p>

<p>
  The following example takes all of the data points and converts them to a closer range of 0 to 1. Standard scaler helps prevent outliers and keep the data closer to each other rather than gaps. 
 </p>
 <p>
  Given a feature vector X of size n: <br>
  &nbsp;<b> X = [x1, x2, ..., xn] </b>  <br>
  <br>
 The StandardScaler calculates the standardized feature vector Z as follows:

<ol>
  <li>Calculate the mean of the feature vector X: </li><br>
    &nbsp;<b> mean = (x1 + x2 + ... + xn) / n </b> <br> <br>
  <li>Calculate the standard deviation of the feature vector X:</li> <br> 
    &nbsp;<b> std_dev = sqrt( ((x1 - mean)^2 + (x2 - mean)^2 + ... + (xn - mean)^2) / n ) </b> <br> <br>
  <li>Subtract the mean from each element of X:</li> <br> <br>
    &nbsp;<b> X = [x1 - mean, x2 - mean, ..., xn - mean] </b> <br> <br>
  <li> Divide each element of X by the standard deviation: </li> <br>
    &nbsp;<b> Z = [ (x1 - mean) / std_dev, (x2 - mean) / std_dev, ..., (xn - mean) / std_dev ] </b> <br> <br>
 </ol> 
The resulting standardized feature vector Z has a mean of zero and a standard deviation of one, which is useful for ensuring that different features are on the same scale and for improving the performance of certain machine learning algorithms.
 </p>

![Standard Scaler Code](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/standard_scaler_code.png)

<p>
To get better accuracy for our model, apply the scaler method. The point of StandardScaler is transforming the data to have a mean value of 0 and a standard deviation of 1. Applying the StandardScaler should improve the accuracy of the LogisticRegression Model() and RandomForestClassifier().
</p>

![Logistic Regression Model with Scaler](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/logitic_regression_model_code_with_scaler.png)

<p>
The LogisticRegression Model() will have the same code except X_train and X_test will be replaced by X_train_scaled and X_test_scaled. After replacing the test and train values, the accuracy improved from 0.5168013611229264 to 0.767333049766057. The score increased by 22. 
</p>

![Random Forest Classifier with Scaler](https://github.com/samuelroiz/Predict_Credit_Risk/blob/main/Images/random_forest_classifier_code_with_scaler.png) 

<p>
The RandomForestClassifier Model() will have the same code except X_train and X_test will be replaced by X_train_scaled and X_test_scaled. After replacing the test and train values, the accuracy decreased from 0.6424925563589962 to 0.6339855380689069. The standardscaler did not improve the RandomForestClassifier. 
</p>

### LogisticRegression Model() and RandomForestClassifier() Comparison

<p>
  In this specific model, the standard scaler improves the Logistic Regression Model. It improved from 0.5168013611229264 to 0.767333049766057 and an improvement for the Logistic Regression model. Applying the standard scaler worked for the logistic regression model is because the data has outliers and gaps from each data point. Outliers and gaps decrease the accuracy of the model. However, applying the standard scaler does not always help the model's accuracy. 
</p>

<pre>
  StandardScaler can potentially improve or weaken the accuracy of a model, depending on the nature of the data and the modeling algorithm being used.

In general, StandardScaler can help improve model accuracy by ensuring that all features are on the same scale. This is important for modeling algorithms that are sensitive to the scale of the features, such as those based on distances or gradients (e.g., k-nearest neighbors, gradient descent-based algorithms). By scaling the features to have zero mean and unit variance, StandardScaler can help these algorithms converge more quickly and produce better results.

On the other hand, StandardScaler may not be helpful or may even hurt model accuracy if the data is already well-scaled and the algorithm being used is not sensitive to feature scaling. In some cases, scaling the data can introduce noise or outliers that can negatively impact model performance. Additionally, if the data has a highly skewed distribution, scaling may not be appropriate and other techniques such as normalization or log transformation may be more effective.

Ultimately, the effect of StandardScaler on model accuracy will depend on the specific dataset and algorithm being used, so it's important to evaluate the impact of feature scaling on model performance in a systematic way.
</pre>



## Contributing

Please read [CONTRIBUTING.md](https://gist.github.com/samuelroiz/1af49ec9eea365bc845ba04c5071a976) for details on our code of conduct, and the process for submitting pull requests to us.

## Versioning

We use [SemVer](http://semver.org/) for versioning. For the versions available, see the [tags on this repository](https://github.com/your/project/tags). 

## Authors

* **Samuel Roiz** - *Data clean, Analyzed Data, Math Model* - [Profile](https://github.com/samuelroiz)

See also the list of [contributors](https://github.com/samuelroiz) who participated in this project.

## License

This project is licensed under the MIT License - see the [LICENSE.md](https://gist.github.com/samuelroiz/1af49ec9eea365bc845ba04c5071a976) file for details

## Acknowledgments

* LendingClub 
* USC Data Visualization
* CSUN Mathematics


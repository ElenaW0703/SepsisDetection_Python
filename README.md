# SepsisDetection_Python
Only cleaned datasets uploaded due to large size
### Introduction
Sepsis is a significant public health concern. According to WHO and CDC's data, nearly 1.7 million people in the U.S. develop sepsis and 270,000 people die from sepsis each year. In addition, U.S. hospitals spend more than 24 billion dollars on sepesis each year (13% U.S. healthcare expenses). Detection and treatement are critical for improving sepsis outcome. Taking advantage of the public data from https://physionet.org/content/challenge-2019/1.0.0/, which is sourced from ICU patients in three separate hospital systems, we tried to develop an automated, mechine-learning algorithm for the detection of sepsis to improve the Sepsis diagnosis.

### Data
The original data are in packed in `psc` files and can be download directly from the website mentioned above. Training dataset A includes 20,336 subjects’ psv files, while test dataset B has 20,000 subjects’ psc files. For simplicity, we directly load the cleaned datasets in the reports. The detailed reading-in process can be found in `AllCode` file. 

1. Outcome variable: `SepsisLabel`  
    The outcome variable is the `SepsisLabel`. For sepsis patients, it equals to \
    1 if t>= t_sepsis - 6 and 0 if t<t_sepsis-6, while for non-sepsis patients, `SepsisLabel` is 0
    
2. Covariates:  
    Both datasets totally have 40 covariates. This includes:
    * 8 vital physical signs: Heart rate (beats/min), Pulse oximetry (%), Temperature (degree), etc;
    * 26 laboratory values: Measure of excess bicarbonate (mmol/L), Calcium(mg/dL), Hematocrit (%), etc
    * 6 demographical features: Age, Gender, Administrative identifier for ICU unit, etc.
    
### Data Cleaning
After combined `psv` files into two large datasets, we conducted further data cleaning work. 

First, We removed hospital-level variables `Unit1`, `Unit2`, `HospAdmTime`, and `ICULOS` since the information in these columns is not related to individual-level prediction. Another column of `ETCO2` was also dropped in both dataset A and dataset B because it only contains `NA` value.  

Then, we spent time in the imputation of missing values. We assumed that each patient has unique characteristic and every value has ongoing time specified. Thus, primarily, the `NAs` were imputed based on the last or next recordsthe missing values, meaning that if there is a value in previous or next time period, then impute with the previous or the next one. Otherwise, we replaced the mean for continuous variables and mode for categorical variables if the values are missing. 

Finally, considering the great benefit of early detection of Sepsis, we extracted the initial situation of patients, i.e. using the laboratory results and demographical features of the first time when patients visited hospitals. This gave us 20,336 records in training set and 20,000 data in test set. Notice that the final datasets are imbalanced: percentages of $0$ (do not develop sepsis) in the `SepsisLabel` in datasets A and B are around 91% and 94%, respectively. They will be balanced before modeling process using the SMOTE method later. 

### EDA
We did the following visualizations to help us get a general understanding of the data:
1. Distribution Plots of each covariate 
2. Correlation acorss covariates
3. PCA for training data
4. UMAP for training data

### Model Implementation
#### Data Pre-processing
For convenience, we treated normalized dataset A as training set and normalized dataset B as test set. The outcome and covariates were separated and denoted as `x` and `y`, respectively.
#### Balancing Data
The original data is highly imbalanced: only 1,790 participants were test positive of Sepsis, which the remaining 18,546 participants were labeled 0 for non-positive. This largely influences the models in classifing the minority class-diagnosed Sepsis, which is more important.  Therefore, we used Synthetic Minority Oversampling Technique (SMOTE) to balance data before building models. Particularly in our case, since one of the covariate `Gender` is binary, we applied the `SMOTENC` function from `imblearn.over_sampling` module and set the `categorical_features` argument to specify it. After balancing on our training sets, we obtained 18,546 zero-labeled records and 18,546 ones of Sepsis.
#### Variable Selection
Fitting a LASSO regression, variable selection was conducted before applying classification models. Coefficients of three covariates are relatively small (<0.005): `Chloride(mmol/L)`, `Troponin I (ng/mL)` and `Hematocrit (%)`. Therefore, the above features were dropped, and the final datasets for modeling were obtained.
#### Model 1: Logistic Regression
Logistic regression was used on the balanced dataset. The stochastic gradient descent (SGD) was used to the regularized linear methods to help build an estimator for classification and regression problems. The SGD classifier works well with large-scale datasets and is an efficient easy-to-implement method. Compared with the accuracy, the model with `l2` norm and 0.011 `alpha` has best performance and was selected.
#### Model 2: KNN
The second model we use is the K-nearest neighbor on the balanced dataset. The best number of `k` selected via cross validation is 16 with an accuracy score of 0.84. 
#### Model3: Naive Bayes
Naive bayes  was used on the balanced dataset as well. The tuning parameter is `var_smoothing`, which is the portion of the largest variance of all features that is added to variances for calculation stability. The best value selected via cross validation is 1.53448 with an accuracy score of 0.746.
#### Model 4: Random Forest
The fourth model we applied is Random forest. Based on the balanced dataset, we tuned parameters using 5-fold cross-validation method. The optimal model with the maximum depth of the tree being 50, results in an 96.29% training accuracy. However, when we applied it to test data, the random forest performs poorly in predicting Sepsis patients, which leads to the result that the test accuracy is only about 0.77.
#### Model 5: XGBoost
Finally, the extreme gradient boosting was used on the balanced dataset. Because `XGBClassifier` has a built-in parameter `scale_pos_weight`, which controls the balance of positive and negative weights for an imbalance dataset, using the balanced dataset from SMOTE is not necessary. The `scale_pos_weight` was set to be the number of negative results divided by the number of positive results as default. Other parameters were tuned via `GridSearchCV` function, and the optimized parameters are: `eta` = 0.8, `max_depth` = 3, `n_estimators` = 150. 
#### Neural Network for Classification with Tensorflow
A neural network for classification was also used on the balanced dataset, but it did not achieve a large improvement. The model consists of 3 layers with activation function of sigmoid. The accuracy score isn’t very different from those of the models above, and the model did not improve as the number of epochs increased, meaning the model was not able to converge. Therefore, we did not present the models and results here in the report, but more details can be found in `AllCode` file.
#### Discussion & Conclusions
Please see more details in Report and Code

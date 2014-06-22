Practical Machine Learning Prediction Assignment
========================================================

## About the Data

Using devices such as Jawbone Up, Nike FuelBand, and Fitbit it is now possible to collect a large amount of data about personal activity relatively inexpensively. These type of devices are part of the quantified self movement – a group of enthusiasts who take measurements about themselves regularly to improve their health, to find patterns in their behavior, or because they are tech geeks. One thing that people regularly do is quantify how much of a particular activity they do, but they rarely quantify how well they do it. In this project, your goal will be to use data from accelerometers on the belt, forearm, arm, and dumbell of 6 participants. They were asked to perform barbell lifts correctly and incorrectly in 5 different ways. More information is available from the website here: http://groupware.les.inf.puc-rio.br/har (see the section on the Weight Lifting Exercise Dataset).

## Obtaining Data

```r
setwd("~/Desktop/data_science/08_ml/assign/")
# training set
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-training.csv", 
    "pml-training.csv", method = "curl")
# testing set
download.file("https://d396qusza40orc.cloudfront.net/predmachlearn/pml-testing.csv", 
    "pml-testing.csv", method = "curl")
download_date <- date()
```


This data as obtained Sun Jun 22 00:29:12 2014.

## Summary

We would like to predict the behavior while doing the curl. The five classes of the `classe` variable are:
* **A** Correct curl
* **B** Throwing elbows to the front
* **C** Lifting the dumbbell halfway
* **D** Lowering the dumbbell halfway
* **E** Moving hips forward

## Load Data and Nessisary Packages

```r
library(caret)
```

```
## Loading required package: lattice
## Loading required package: ggplot2
```

```r
training <- read.csv("pml-training.csv")
testing <- read.csv("pml-testing.csv")
```


## Cleaning data

Some of these variables have large numbers of missing values `NA` and empty strings `""` which can't be all that informative.


```r
training[training == ""] = NA  # replace all empty strings with NA
testing[testing == ""] = NA
remove = vector()
for (i in names(training)) {
    if (sum(is.na(training[i])) > 0) {
        remove <- c(remove, i)
    }
}
training <- training[!(names(training) %in% remove)]
testing <- testing[!(names(training) %in% remove)]
```


Next we will remove columsn like name, id, time, which are unlikely to have much predictive power.


```r
training <- training[, -c(1:7)]
testing <- testing[, -c(1:7)]
```


# Prediction

Here we'll use a random forest algorithm and divide the training data into 70% training set and 30% validation set.


```r
set.seed(12345)
inTrain <- createDataPartition(y = training$classe, p = 0.7, list = FALSE)[, 
    1]
train <- training[inTrain, ]
test <- training[-inTrain, ]
```


Train the model using training set and predict using validation set to evaluate the estimated prediction error using a confusion matrix.

```r
set.seed(54321)
fit <- train(classe ~ ., data = train, method = "rf", ntree = 10)
```

```
## Loading required package: randomForest
## randomForest 4.6-7
## Type rfNews() to see new features/changes/bug fixes.
```

```r

# predict on the test set to evaluate the error
pred <- predict(fit, newdata = test)
confusionMatrix(test$classe, pred)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1669    3    2    0    0
##          B   12 1117    8    0    2
##          C    0   12 1002   12    0
##          D    1    7   20  935    1
##          E    0    3    4    7 1068
## 
## Overall Statistics
##                                        
##                Accuracy : 0.984        
##                  95% CI : (0.98, 0.987)
##     No Information Rate : 0.286        
##     P-Value [Acc > NIR] : <2e-16       
##                                        
##                   Kappa : 0.98         
##  Mcnemar's Test P-Value : NA           
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity             0.992    0.978    0.967    0.980    0.997
## Specificity             0.999    0.995    0.995    0.994    0.997
## Pos Pred Value          0.997    0.981    0.977    0.970    0.987
## Neg Pred Value          0.997    0.995    0.993    0.996    0.999
## Prevalence              0.286    0.194    0.176    0.162    0.182
## Detection Rate          0.284    0.190    0.170    0.159    0.181
## Detection Prevalence    0.284    0.194    0.174    0.164    0.184
## Balanced Accuracy       0.996    0.987    0.981    0.987    0.997
```

```r
est_acc <- round(confusionMatrix(test$classe, pred)$overall["Accuracy"] * 100, 
    2)
```


The extimated accuracy of the models is about 98.4%.

Finally, we will train the model on the full training set `training` and predict on the official test set `testing`.

```r
final_fit <- train(classe ~ ., data = training, method = "rf", ntree = 10)
final_predictions <- predict(final_fit, newdata = testing)
```


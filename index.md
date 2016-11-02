# HAR Project
Patrick Conroy  
11/1/2016  


## Introduction

We analyze the Weight Lifting Exercise Dataset from the Human Activity recognition project at http://groupware.les.inf.puc-rio.br/har. We attempt to generate a model that predicts the "classe" variable, which represents the quality of execution of a specified weightlifting exercise. We are given two datasets, a training and a testing set. We use the training set to train the model and estimate its accuracy. We use the testing set to respond to a quiz as part of the Coursera machine learning class.

## Loading and Cleaning the Data

Load required libraries and the testing and training sets.

```r
library(caret)
```

```
## Loading required package: lattice
```

```
## Loading required package: ggplot2
```

```r
testing <- read.csv(file = "pml-testing.csv")
training <- read.csv(file = "pml-training.csv")
```

We are supplied a training set that includes the "classe" variable -- the dependent variable, and a testing set where the "classe" variable is replaced with a "problem_id" for use with a quiz.

To predict the accuracy of the model, we will need to split the given training data into our own training and validation data.

First, we note that there are a large number of columns in the testing data that are all NAs. We therefore remove those columns in both the training and testing data. We also remove the first seven columns (X [an index], various timestamps, user id, and *_window varables), which have nothing to do with the sensor data.


```r
set.seed(314)
seeds <- vector(mode = "list", length = 31) # tenfold validation repeated three times + 1
for(i in 1:30) seeds[[i]] <- sample.int(1000, 3)
seeds[[31]] <- sample.int(1000, 1) # setting up seeds for reproduceable parallel processing.
obsCount <- colSums(!is.na(testing))
cleanTesting <- testing[,obsCount != 0]
cleanTraining <- training[,obsCount != 0]
cleanTesting <- cleanTesting[,-(1:7)]
cleanTraining <- cleanTraining[, -(1:7)]
sum(is.na(cleanTesting))
```

```
## [1] 0
```

```r
sum(is.na(cleanTraining)) ## No more NAs in the data.
```

```
## [1] 0
```

```r
inTrain <- createDataPartition(cleanTraining$classe, p = 0.75, list = FALSE)
trainingSet <- cleanTraining[inTrain,]
validationSet <- cleanTraining[-inTrain,]
```

## Create the Model

We will use repeated k-fold cross-validation on a random forest model. We cache this block because it takes a long time to run, even with doParallel. We use the the doParallel package with caret to speed processing on a multi-core processor.


```r
library(doParallel)
```

```
## Loading required package: foreach
```

```
## Loading required package: iterators
```

```
## Loading required package: parallel
```

```r
cl <- makeCluster(detectCores())
train_control <- trainControl(method = "repeatedcv", number = 10, seeds = seeds, repeats = 3)
registerDoParallel(cl)
model <- train(classe ~., data = trainingSet, trControl = train_control, method = "rf")
```

```
## Loading required package: randomForest
```

```
## randomForest 4.6-12
```

```
## Type rfNews() to see new features/changes/bug fixes.
```

```
## 
## Attaching package: 'randomForest'
```

```
## The following object is masked from 'package:ggplot2':
## 
##     margin
```

```r
stopCluster(cl)
```


```r
model$finalModel
```

```
## 
## Call:
##  randomForest(x = x, y = y, mtry = param$mtry) 
##                Type of random forest: classification
##                      Number of trees: 500
## No. of variables tried at each split: 27
## 
##         OOB estimate of  error rate: 0.59%
## Confusion matrix:
##      A    B    C    D    E  class.error
## A 4182    1    1    0    1 0.0007168459
## B   23 2820    4    1    0 0.0098314607
## C    0   14 2546    7    0 0.0081807557
## D    0    1   18 2391    2 0.0087064677
## E    0    1    7    6 2692 0.0051736881
```

```r
predVal <- predict(model, validationSet)
confusionMatrix(validationSet$class, predVal)
```

```
## Confusion Matrix and Statistics
## 
##           Reference
## Prediction    A    B    C    D    E
##          A 1393    2    0    0    0
##          B    3  944    2    0    0
##          C    0    3  848    4    0
##          D    0    0    4  799    1
##          E    0    0    3    2  896
## 
## Overall Statistics
##                                           
##                Accuracy : 0.9951          
##                  95% CI : (0.9927, 0.9969)
##     No Information Rate : 0.2847          
##     P-Value [Acc > NIR] : < 2.2e-16       
##                                           
##                   Kappa : 0.9938          
##  Mcnemar's Test P-Value : NA              
## 
## Statistics by Class:
## 
##                      Class: A Class: B Class: C Class: D Class: E
## Sensitivity            0.9979   0.9947   0.9895   0.9925   0.9989
## Specificity            0.9994   0.9987   0.9983   0.9988   0.9988
## Pos Pred Value         0.9986   0.9947   0.9918   0.9938   0.9945
## Neg Pred Value         0.9991   0.9987   0.9978   0.9985   0.9998
## Prevalence             0.2847   0.1935   0.1748   0.1642   0.1829
## Detection Rate         0.2841   0.1925   0.1729   0.1629   0.1827
## Detection Prevalence   0.2845   0.1935   0.1743   0.1639   0.1837
## Balanced Accuracy      0.9986   0.9967   0.9939   0.9957   0.9988
```

```r
predTest <- predict(model, cleanTesting)
predTest
```

```
##  [1] B A B A A E D B A A B C B A E E A B B B
## Levels: A B C D E
```
Checking the model against the reserved validation data from the training set shows an accuracy of 99.5% with a Kappa of .9938. The estimated out-of-sample error rate is approximately 0.5%.

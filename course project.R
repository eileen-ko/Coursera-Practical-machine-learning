#Practical Machine Learning Course Project
#load packages
library(caret)
library(rpart)
library(rpart.plot)
library(rattle)
library(dplyr)

set.seed(113311)

#import data
pml.training <- read.csv("~/Documents/Coursera R/Pratical Machine Learning/Course Project/pml-training.csv")
pml.testing <- read.csv("~/Documents/Coursera R/Pratical Machine Learning/Course Project/pml-testing.csv")

#take a brief look at the dataset and the number of NA
str(pml.training)
sum(is.na(pml.training))

#creat a partition with the training dataset
inTrain <- createDataPartition(y=pml.training$classe, p=0.75, list=FALSE)
training <- pml.training[inTrain,]
testing <- pml.training[-inTrain,]

# remove variables that are unbalanced/near zero variance
nzv <- nearZeroVar(training)
filteredtraining <- training[, -nzv]
filteredtesting <- testing[, -nzv]

# remove variables that are mostly NA
removena <- sapply(filteredtraining, function(x) mean(is.na(x))) > 0.95
TrainSet <- filteredtraining[, removena==FALSE]
TestSet <- filteredtesting[, removena==FALSE]

#Take a look at the training and testing datasets
dim(TrainSet)
names(TrainSet)
TrainSet <- TrainSet[, -(1:6)]
TestSet <- TestSet[, -(1:6)]
names(TrainSet[53])

#find out the highly correlated variables 
cor <- cor(TrainSet[ ,-53])
which(cor >0.85, arr.ind = TRUE)

names(TrainSet[c(1, 4, 8, 9, 11)])

#creat a new dataset omitted the above correlated variables
TrainSet <- TrainSet[-c(1, 4, 8, 9, 11)]
TestSet <- TestSet[-c(1, 4, 8, 9, 11)]
rm(list = "TestSet")

#Fit models
#Classification trees
FitDT <- rpart(classe ~ ., data= TrainSet, method="class")
fancyRpartPlot(FitDT)

#validate the model by using TestSet
predictFitDT <- predict(FitDT, TestSet, type = "class")
DecisionTree <- confusionMatrix(predictFitDT, TestSet$classe)
DecisionTree

#random forrest
FitRf <- train(TrainSet$classe~ .,data=TrainSet, method="rf",prox=TRUE)


controlRF <- trainControl(method="cv", number=3, verboseIter=FALSE)
FitRf <- train(classe ~ ., data=TrainSet, 
                          method="rf", trControl=controlRF)
FitRf$finalModel

#Use the TestSet to predict classe~
predictRf <- predict(FitRf, newdata=TestSet)
confRf <- confusionMatrix(predictRf, TestSet$classe)
confRf

confRf$table
confRf$byClass

#Boosting trees
controlGBM <- trainControl(method = "repeatedcv", number = 5, repeats = 1)
modGBM  <- train(classe ~ ., data=TrainSet, method = "gbm", trControl = controlGBM, verbose = FALSE)
modGBM$finalModel
print(modGBM)

#validate the accuracy with GBM
predictGBM <- predict(modGBM, newdata=TestSet)
cmGBM <- confusionMatrix(predictGBM, TestSet$classe)
cmGBM


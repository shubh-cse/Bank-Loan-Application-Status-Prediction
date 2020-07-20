####################################Bank Loan Default Case########################################################

#remove all the objects stored
rm(list=ls())

#set working directory
setwd("C:/Users/Shubh Gupta/Desktop/DataScience/Project/Bank-loan")

#check working directory
getwd()

#importing libraries
library(gridExtra)
library(grid)
library(ggplot2)
library(lattice)
library(usdm)
library(pROC)
library(caret)
library(rpart)
library(DataCombine)
library(ROSE)
library(e1071)
library(xgboost)
library(randomForest)

#loading the dataset
data=read.csv('bank-loan.csv')

#size of the dataset
dim(data)

#describing the dataset
summary(data)

#exploring the dataset
head(data)

##########################################Distribution of Target Variable#########################################

#count of both the classes in the target variable
table(data$default)

###########################################Observation#########################################################
#It is an imbalanced dataset.
#The number of customers that will have a default status are comparatively less as compared to those who will 
#have a non-default status.

########################################Missing Value Analysis###################################################

#function to calculate the missing values

findMissingValue =function(df)
{
  missing_val = data.frame(apply(df,2,function(x){sum(is.na(x))}))
  return(missing_val)
}

#checking missing value in the dataset
findMissingValue(data)

###########################################Observation#########################################################
#In all the 9 variables, we have 150 missing values in the default predictor which is our target or dependent 
#variable. Since imputing data in a target variable could result into negative impact on our model, so we will 
#drop observations associated with these 150 missing values.

###############################################Creating new dataset############################################

#now we will create new dataset with 700 observations
data_new=data[1:700,]

#size of the new dataset
dim(data_new)

#describing the new dataset
summary(data_new)

#exploring the new dataset
head(data_new)

##################################Missing Value Analysis on the new dataset#######################################

#checking missing values in the new dataset
findMissingValue(data_new)

###########################################Observation###########################################################
#Now we have a new dataset with 700 observations and 9 variables with no missing values.

##############################Splitting the new dataset into train and test#######################################

#dividing the data into train and test using stratified sampling
set.seed(1234)
index = createDataPartition(data_new$default, p = .80, list = FALSE)
train = data_new[ index,]
test  = data_new[-index,]

#size of train data
dim(train)

#exploring the train data
head(train)

#size of test data
dim(test)

#exploring the test data
head(test)

###########################################Observation#########################################################
#Through splitting the data, we have now 560 observations in the train data and 140 observations in the test data.

#splitting the dependent and independent variables from the train dataset
independent_var=(colnames(train)!='default')
X=train[,independent_var]
Y=train$default

#checking the shape of independent variables
dim(X)

#checking the shape of dependent variable
dim(Y)

#exploring the independent variable
head(X)

#exploring the dependent variable
head(Y)

###########################################Multicollinearity Analysis###########################################

#checking the correlation between independent variables
cor=vifcor(X)
print(cor)

###########################################Observation#########################################################
#No variable from the 8 input variables has collinearity problem.

####################################Distribution of Independent Variables#####################################

#function to check the distribution of independent variables
plot_distribution=function(X)
{
  variblename =colnames(X)
  temp=1
  for(i in seq(8,dim(X)[2],8))
  {
    plot_helper(temp,i ,variblename)
    temp=i+1
  }
}
plot_helper=function(start ,stop, variblename)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(4,3))
  for (i in variblename[start:stop])
  {
    plot(density(X[[i]]) ,main=i )
  }
}

#checking the distribution of independent variables
plot_distribution(X)

###########################################Observations#########################################################
#From the distribution plots, we can conclude that none of the independent variables are normally distributed.
#All the eight predictors are positive or right skewed.

############################################Outlier Analysis###################################################

#function to plot a boxplot for checking outliers in the dataset
plot_boxplot=function(X)
{
  variblename =colnames(X)
  temp=1
  for(i in seq(8,dim(X)[2],8))
  {
    plot_helper(temp,i ,variblename)
    temp=i+1
  }
}
plot_helper=function(start ,stop, variblename)
{ 
  par(mar=c(2,2,2,2))
  par(mfrow=c(4,3))
  for (i in variblename[start:stop])
  {
    boxplot(X[[i]] ,main=i)
  }
}

#outlier analysis on the independent variables
plot_boxplot(X)

###########################################Observation#########################################################
#data points below lower fense and above upper fense will be declared as outliers.

############################################Replacing outliers with NA#########################################

#function to replace outliers in the dataset with NA
fill_outlier_with_na=function(df)
{
  cnames=colnames(df)
  for(i in cnames)
  {
    val = df[,i][df[,i] %in% boxplot.stats(df[,i])$out]
    df[,i][df[,i] %in% val] = NA
  }
  return (df)
}

#replacing outliers from the data
X=fill_outlier_with_na(X)

#count of oulier
sum(is.na(X))

###########################################Observation#########################################################
#Total number of ouliers are 179

###########################################Mean Imputation#######################################################

#function to replace NA with the mean of that particular variable
fill_outlier_with_mean=function(df)
{
  cnames=colnames(df)
  for(i in cnames)
  {
    
    df[is.na(df[,i]), i] <- mean(df[,i], na.rm = TRUE)
  }
  return (df)
}

#mean imputation data
X=fill_outlier_with_mean(X)

#count of NA values in data after imputation
sum(is.na(X))

###########################################Observation#########################################################
#Total number of NA values in the data after mean imputation is 0

###########################################Standardization#######################################################

#standardization is done to scale all the variables in the same range
#standardization=(x-mean(x)/sd(x)

#function to perform standardization
standardization=function(df)
{
  cnames =colnames(df)
  for( i in   cnames ){
    df[,i]=(df[,i] -mean(df[,i] ,na.rm=T))/sd(df[,i])
  }
  return(df)
}

#standardization of the data
X=standardization(X)

##########################################Model Training#######################################################
#We will use three models for training the dataset :-
#1. Logistic Regression
#2. Decision Tree
#3. Random Forest

###############################Generic function to calculate various classification metrics#####################

#function to check performance of classification models
metric_fun=function(conf_matrix)
{
  model_parm =list()
  tpr=conf_matrix[1,1]
  fnr=conf_matrix[1,2]
  fpr=conf_matrix[2,1]
  tnr=conf_matrix[2,2]
  p=(tpr)/(tpr+fpr)
  r=(tpr)/(tpr+fnr)
  s=(tnr)/(tnr+fpr)
  f1=2*((p*r)/(p+r))
  print(paste("accuracy",(tpr+tnr)/(tpr+tnr+fpr+fnr)))
  print(paste("precision",p))
  print(paste("recall",r))
  print(paste("specificity",s))
  print(paste("fpr",(fpr)/(fpr+tnr)))
  print(paste("fnr",(fnr)/(fnr+tpr)))
  print(paste("f1",f1))
}

#############################################Logistic Regression################################################

logit=glm(formula=Y~.,data=X,family='binomial')
summary(logit)
y_prob=predict(logit,test[-9],type='response')
y_pred=ifelse(y_prob >0.5, 1, 0)
conf_matrix= table(test[,9],y_pred)
metric_fun(conf_matrix)
roc=roc(test[,9],y_prob)
print(roc)
plot(roc ,main="Logistic Regression roc-auc curve")

###################################################Observation#################################################
# Accuracy --> 37.85%
# Precision --> 100%
# Recall --> 13%
# Specificity --> 100%
# FPR --> 0%
# FNR --> 87%
# F1 Score --> 23.01%
# AUC --> 77.76%

#############################################Decision Tree#####################################################

dt_model=rpart(Y~.,data=X)
summary(dt_model)
y_prob=predict(dt_model,test[-9])
y_pred=ifelse(y_prob>0.5,1,0)
conf_matrix=table(test[,9],y_pred)
metric_fun(conf_matrix)
roc=roc(test[,9],y_prob )
print(roc)
plot(roc,main="Decision Tree roc-auc curve")

###################################################Observation#################################################
# Accuracy --> 32.14%
# Precision --> 100%
# Recall --> 5%
# Specificity --> 100%
# FPR --> 0%
# FNR --> 95%
# F1 Score --> 9.52%
# AUC --> 52.5%

#############################################Random Forest#######################################################

rf_model = randomForest(default ~ ., train, importance = TRUE, ntree = 500)
summary(rf_model)
y_prob =predict(rf_model,as.matrix(test[,-9]))
y_pred = ifelse(y_prob >0.5, 1, 0)
conf_matrix= table(test[,9] , y_pred)
metric_fun(conf_matrix)
roc=roc(test[,9], y_prob )
print(roc)
plot(roc ,main="Random Forest roc-auc curve")

###################################################Observation#################################################
# Accuracy --> 79.28%
# Precision --> 82.56%
# Recall --> 90%
# Specificity --> 52.5%
# FPR --> 47.5%
# FNR --> 10%
# F1 Score --> 86.12%
# AUC --> 82.15%

##################################################Model Selection#############################################
#The model should be selected based on the following parameters :-
#1. High Accuracy
#2. High F1 Score
#3. High AUC Score
#4. High Recall
#5. High Precision
#6. High Specificity
#7. Low FPR
#8. Low FNR

#################################################Freezed Model################################################
#We will freeze Random Forest as our final model based on the above parameters.
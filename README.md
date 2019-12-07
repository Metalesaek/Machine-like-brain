# Machine-like-brain

---
title: "deep learning model"
author: "Dr.metales"
date: "12/5/2019"
output:
  html_document: default
  pdf_document: default
---

```{r setup, include=TRUE}
knitr::opts_chunk$set(echo = TRUE, warning=FALSE,error=FALSE,message=FALSE)
```
## Prepare the data

Deep learning models are becoming the most important predictive models used by well known companies in the world. In this paper we will use the deep learning  model to predict the titanic data set (kaggle competition). 

Let's call this data.

```{r}
library(tidyverse)
data <- read_csv("train.csv")

```

Then we will call **keras** package for deep learning models, and **caret** for randomly spliting  the data and creating the confusion matrix.

```{r}
library(keras)
library(caret)
```
The first step in modeling is to clean and prepare the data. the following code shows the structure of this data. 

```{r}
glimpse(data)
```

Using this data we want to predict the variable **Survived** using the remaining variables.We see that some variables have unique values such as **PassengerId**,**Name**, and **ticket**. Thus,they cannot be used as predictors. the same note applies to the vraiable **Cabin** with the additional problem of missing values. these variables will be removed as follows:

```{r}
mydata<-data[,-c(1,4,9,11)]
head(mydata)
```

As we see some variables should be of factor type such as **Pclass** (which is now double), **Sex** (character), and "Embarked** (character). thus, we convert them to factor type:

```{r}
mydata$Pclass<-as.factor(mydata$Pclass)
mydata$Embarked<-as.factor(mydata$Embarked)
mydata$Sex<-as.factor(mydata$Sex)
glimpse(mydata)

```

Now let's get some summary about this data

```{r}
summary(mydata)
```

We have only two variables that have missing values, **Age** with large number 177 , followed by **Embarked** with 2 missing values.
 To deal with this issue we have two options:
 
* the first and easy one is to remove the entire rows that have any missing value but with the cost of may losing valuable informations specially when we have large number of missing values which is in our case.

* the second option is to impute this missing values using the other complete cases, for instance we can replace a missing value of a peticular column by the mean of this column (for numeric variable) or we use multinomial method to predict the categorical variables.

fortunately , there is a usefull package called **mice** which will do this imputation for us. However, applying this imputation on the entire data would lead us to fall on a problem called **train-test comtamination** ,which means that when we split the data , the missing values of the training set are imputed using cases in the test set, and this violates a crucial concept in machine learning for model evaluation, the test set should never be seen by the model during the training process.

To avoid this problem we apply the imputation seperatly on the training set and on the test set. 
So let's partition the data using **caret** package function.


## Partition the data & impute the missing values.

we randomly split the data into two sets , 80% of samples will be used in the training process and the remaining 20% will be kept as test set.  

```{r}
set.seed(1234)
index<-createDataPartition(mydata$Survived,p=0.8,list=FALSE)
train<-mydata[index,]
test<-mydata[-index,]

```

Now we are  ready to impute the missing values for both train and test set.

```{r}

library(mice)
impute_train<-mice(train,m=1,seed = 1111)
train<-complete(impute_train,1)

impute_test<-mice(test,m=1,seed = 1111)
test<-complete(impute_test,1)
```

### Convert data into a normalized matrix.

in deep learning all the variables should of numeric type, so first we convert the factors to integer type and recode the levels in order to start from 0, then we convert the data into matrix. After that we pull out the target variable into a separate vector, and finally we normalize our matrix. 

We do this transformation for both sets (train and test).


```{r}
train$Embarked<-as.integer(train$Embarked)-1
train$Sex<-as.integer(train$Sex)-1
train$Pclass<-as.integer(train$Pclass)-1

test$Embarked<-as.integer(test$Embarked)-1
test$Sex<-as.integer(test$Sex)-1
test$Pclass<-as.integer(test$Pclass)-1
glimpse(test)
```

we convert the tow sets into matrix form. (we also remove the column names)

```{r}
trained<-as.matrix(train)
dimnames(trained)<-NULL

tested<-as.matrix(test)
dimnames(tested)<-NULL
str(trained)
```

Now we pull out the target variabele

```{r}
trainy<-trained[,1]
testy<-tested[,1]
trainx<-trained[,-1]
testx<-tested[,-1]

```

## Apply one hot encoding on the target variable 

```{r}
trainlabel<-to_categorical(trainy)
testlabel<-to_categorical(testy)
```

The final step now is normalizing the matrices (trainx and testx)

```{r}
trainx<-normalize(trainx)
testx<-normalize(testx)
summary(testx)
```

## Train the model.

Now it is time time to buil our model. Th first step is to define the model type and the number of layers that will be used with the prespecified parameters.
We will choose a simple model with one hidden layer with 10 unites (nodes). Since we have 7 predictors the input_shape will be 7, and the activation function is **relu** which is the most used one, but for the output layer we choose sigmoid function since we have binary classification.

# Creat the model

```{r}
model <- keras_model_sequential()

model %>%
    layer_dense(units=10,activation = "relu",
              kernel_initializer = "he_normal",input_shape =c(7))%>%
    layer_dense(units=2,activation = "sigmoid")

summary(model)  

```

We have in total 102 parameters to estimate, since we have 7 inputs and 10 nodes and 10 biases, so the parameters number of the hidden layer is 80 (7*10+10).By the same way get the parameters number of the output layer.   

# Compile the model

In the compile function (from keras) we spacify the loss function, the optimizer and the metric type that will be used. In our case we use the **binary crossentropy**, the optimizer is the popular one **adam** and for the metric we use **accuracy**.  


```{r}
model %>%
  compile(loss="binary_crossentropy",
          optimizer="adam",
          metric="accuracy")

```

# Execute the model

Now it is time to run our model and we can follow the dynamic evolution of the process in the plot windwo on the right lower corner of the screen. and you can also plot the model in a static way.
for our model we choose 100 epochs (iterations), for the stochastic  gradient we use 20 samples at each iteration, and we hold out 20% of the training data to asses the model. 

```{r}
#history<- model %>%
# fit (trainx,trainlabel,epoch=100,batch_size=20,validation_split=0.2)

```
## Note : to make the model reproducible (since the weights initialization is done randomely), I save it after it has been run, and after that we preceed the code by **#** symbol to prevent it to be executed again. If you would rerun the code remove this symbol.     

From the last iteration we see that the loss is about 0.4261 and the accuracy is 79.37%.

It should be noted here that since the accuracy lines are more or less closer to each other and running togather in the same direction  we do not have to be worry about overfiting.

We can save this model (or save only the wiehts) and we load it again to continue our analysis)

```{r}
# save_model_hdf5(model,"simplemodel.h5")
model<-load_model_hdf5("simplemodel.h5")
```

Here also to save your model remove the **#** from the shunck


## The model evaluation

```{r}

model %>% 
  evaluate(testx, testlabel)

```
 The accuracy rate of the model using the test set is 81.46% which is higher than that of the training set (79.37%). 

## prediction and confusion matrix

we get the prediction on the test set as follows.

```{r}
pred<-predict_classes(model,testx)
head(pred)

```
Using the **caret** package we get the confusion matrix
 
```{r}
confusionMatrix(as.factor(pred),as.factor(testy))
```

Here also we have a good accuracy rate ** 80.9% ** . If we are lucky , this rate may be improved further by increasing the number of nodes or the number of hidden layers but with careful, otherwise you get an overfitting results.   


## tune the model by increasing the number of nodes to 40 nodes

By using the same trik by saving our optimal model 

```{r}
model1 <- keras_model_sequential()

model1 %>%
    layer_dense(units=40,activation = "relu",
              kernel_initializer = "he_normal",input_shape =c(7))%>%
    layer_dense(units=2,activation = "sigmoid")

model1 %>%
  compile(loss="binary_crossentropy",
          optimizer="adam",
          metric="accuracy")


# history1<- model1 %>%
#  fit (trainx,trainlabel,epoch=200,batch_size=40,validation_split=0.2)

```
 We save and load the new model.


with this new model we get a good improvment with the accuracy in the validation set at the last iteration of  about 83.57% .

```{r}
# save_model_hdf5(model1,"simplemodel1.h5")
model1<-load_model_hdf5("simplemodel1.h5")
```


Let's check the accuracy for the test set.

```{r}
pred<-predict_classes(model1,testx)
confusionMatrix(as.factor(pred),as.factor(testy))
```


we get also a large improvment for the test set which is now more than 82%. 


Now we finetune the model by increasing the number of layers This time we will add one hidden layer with 10 , and we use the ** he_uniform** as **kernel_initializer** 


```{r}
model2 <- keras_model_sequential()

model2 %>%
    layer_dense(units=40,activation = "relu",
              kernel_initializer = "he_uniform",input_shape =c(7))%>%
    layer_dense(units=10,activation = "relu")%>%
    layer_dense(units=2,activation = "sigmoid")

model2 %>%
  compile(loss="binary_crossentropy",
          optimizer="adam",
          metric="accuracy")


# history2 <- model2 %>%
#  fit (trainx,trainlabel,epoch=200,batch_size=50,validation_split=0.2)

```

we save this model as we did with the previous ones.

```{r}
# save_model_hdf5(model2,"simplemodel2.h5")
model2<-load_model_hdf5("simplemodel2.h5")

```


```{r}
pred<-predict_classes(model2,testx)
confusionMatrix(as.factor(pred),as.factor(testy))

```

using this model our test accuracy has been improved further from 82.02% to 83.15%. 

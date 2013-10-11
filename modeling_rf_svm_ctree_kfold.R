require(verification)
require(randomForest)

setwd("C:/Users/IBM_ADMIN/Documents/Work/Reading/kaggle/titanic")
train_all <- read.csv("train.csv")

train <- data.frame(Survived=as.factor(train_all$Survived), Pclass=train_all$Pclass, Sex=train_all$Sex, Age=train_all$Age, SibSp=train_all$SibSp, Parch=train_all$Parch, Embarked=as.integer(train_all$Embarked))

train$Age[is.na(train$Age)] <- median(na.omit(train$Age))

train$Embarked[train$Embarked==2] <- 1
train$Embarked[train$Embarked==3] <- 2
train$Embarked[train$Embarked==4] <- 3

train$Embarked=as.factor(train$Embarked)


k <- 10
n <- floor(nrow(train)/k)


#Cross-Validation for RandomForest
err.vect <- rep(NA, k)
for(i in 1:k){
  s1 <- ((i-1)*n+1)
  s2 <- (i*n)
  subset <- s1:s2
  cv.train <- train[-subset,]
  cv.test <- train[subset,]
  
  fit <- randomForest(cv.train[,-1],y = as.factor(cv.train[,1]), ntree=1800)
  
  prediction <- predict(fit, newdata = cv.test[,-1],type="prob")[,2]
  
  err.vect[i] <- roc.area(as.numeric(as.character(cv.test[,1])), prediction)$A
  print(paste("RandomForest AUC for fold", i,":", err.vect[i]))
}
print(paste("RandomForest Mean AUC for fold:", mean(err.vect)))

#Cross-Validation for SVM

library(kernlab)
err.vect <- rep(NA, k)
for(i in 1:k){
  s1 <- ((i-1)*n+1)
  s2 <- (i*n)
  subset <- s1:s2
  cv.train <- train[-subset,]
  cv.test <- train[subset,]
  
  fit <- ksvm(Survived~Pclass+Sex+Age,data=cv.train,type="C-bsvc",kernel='vanilladot',C=100,scaled=c())
  
  prediction <- predict(fit, newdata = cv.test[,-1],type="decision")[,1]
  
  err.vect[i] <- roc.area(as.numeric(as.character(cv.test[,1])), prediction)$A
  print(paste("SVM AUC for fold", i,":", err.vect[i]))
}
print(paste("SVM Mean AUC for fold:", mean(err.vect)))

#Cross-Validation for ctree
library(party)
err.vect <- rep(NA, k)
for(i in 1:k){
  s1 <- ((i-1)*n+1)
  s2 <- (i*n)
  subset <- s1:s2
  cv.train <- train[-subset,]
  cv.test <- train[subset,]
  
  fit <- ctree(Survived~Pclass+Sex+Age,data=cv.train)
  
  prediction <- 1- unlist(predict(fit, newdata = cv.test[,-1], type="prob"), use.names=F)[seq(1,nrow(cv.test[,-1])*2,2)]
  
  err.vect[i] <- roc.area(as.numeric(as.character(cv.test[,1])), prediction)$A
  print(paste("Ctree AUC for fold", i,":", err.vect[i]))
}
print(paste("Ctree Mean AUC for fold:", mean(err.vect)))
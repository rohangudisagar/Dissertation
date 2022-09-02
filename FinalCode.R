rm(list=ls())

# ********************************************************************************************************
library(mice)
library(leaps)
library(Hmisc)
library(corrplot)
library(moments)
library(randomForest)
library(caret)
library(glmnet)
library(pROC)
library(stats)
library(Boruta)
library(MASS)
library(neuralnet)
library(class)
library(e1071)
library(nnet)
library(kknn)
library(MLmetrics)
library(mccr)
library(MLeval)

# ********************************************************************************************************
# Data Manipulation and handling

lung = read.csv("~/Documents/OneDrive - University College Cork/Project/Dataset/lung.csv", header = T)
ncol(lung)
nrow(lung)
dim(lung)

# Remove useless data
lung[,c(1,2,3,4,5,12,13,14,15)] = NULL
lung$min_HIST = NULL
lung$max_HIST = NULL
lung$range_HIST = NULL
lung$surind = NULL
lung$surtim = NULL
names(lung)

# Classification types
table(lung$stg)
# "1"  "1A" "1B" "2A" "2B" "3A" "3B" "4" 

# Handle NA values
summary(lung)
rem_data = lung[,c(2,3)]
lung_NA = subset(lung, select = -c(stg,sex))
ncol(lung_NA)

imputed_Data <- mice(lung_NA,m = 5,method = 'cart',seed = 500,maxit = 50)
newData <- complete(imputed_Data,2)

lung_data = cbind(rem_data, newData) # Final Dataframe
ncol(lung_data)
cbind(lung_data$age, lung_NA$age)

# Finding data types of each feature
col_names = names(lung_data)
dt = matrix(1:2, nrow = 1, ncol = 2)
for(i in 1:143){
  c = cbind(col_names[i], class(lung_data[,i]))
  dt = rbind(dt,c)
}
dt = dt[-1,]

lung_data_num = lung_data
lung_data_num[3:143] <- lapply(lung_data_num[3:143], as.numeric)
lung_data_num$sex <- as.numeric(lung_data_num$sex)

# Scale the data: Standardizing the dataset
# lung_data_num[,3:143] = as.data.frame(scale(lung_data_num[,3:143]))


summary(lung_data_num)

y = lung_data_num$stg
y = as.factor(y)
table(y)
# levels(y) = c("1","1","1","2","2","3","3","4")
# levels(y) = c("0","0","0","0","0","1","1","1")
levels(y) = c("early","early","early","early","early","late","late","late")
y


x = lung_data_num

x$stg = NULL

summary(x)


# Find Correlations
cor = cor(x)
dim(cor)
count = 0
corrMat = matrix(1:3, nrow = 1, ncol = 3)
for(i in 1:59){
  for(j in (i+1):60){
    if(abs(cor[i,j]) > 0.9){
      x_w = cbind(rownames(cor)[i], colnames(cor)[j], cor[i,j])
      corrMat = rbind(corrMat, x_w)
    }
  }
}
corrDF = as.data.frame(corrMat[-1,])
corrDF
names(x)
corrplot(cor, method="color", sig.level = 0.05)
?corrplot
length(findCorrelation(cor))
sort(findCorrelation(cor))
x = x[,-c(4,6,8,11,12,13,14,15,16,17,18,19,20,22,25,26,27,28,30,37,38,39,40,41,42,43,44,45,48,51,52,53,57,58,59,60,61,65,66,67,72,75,76,77,80,81,84,86,87,90,92,93,94,97,98,99,100,101,102,103,105,107,111,112,113,114,116,118,119,120,121,122,123,124,126,128,130,131,135,137,138,139)]
ncol(x)



# E.O. Data Manipulation and handling
# ********************************************************************************************************
# Feature Selection

# RFE

set.seed(0420)
folds = 100
control = rfeControl(functions = rfFuncs, 
                     method = "boot", 
                     # repeats = 10,
                     number = folds,
)



result_rfe1 <- rfe(
  x = x,
  y = y,
  sizes = c(1:60),
  rfeControl = control)
?rfe
selected_features = predictors(result_rfe1)

ggplot(data = result_rfe1, metric = "Accuracy") + theme_bw()
ggplot(data = result_rfe1, metric = "Kappa") + theme_bw()
varImp(result_rfe1)
xrfe = x[selected_features]
ncol(xrfe)
data.rfe = cbind(xrfe,y)
names(data.rfe)
boxplot(result_rfe1$results[,2])

# Boruta
set.seed(0420)
n = nrow(x)
bSel = numeric(0)
for(i in 1:100){
  i.train = sample(1:n,n,replace = TRUE)
  ytrain = y[i.train]
  xtrain = x[i.train,]
  lung.boruta = Boruta(ytrain~., data = xtrain, ntree = 500)
  final.boruta <- TentativeRoughFix(lung.boruta)
  selAtt = getSelectedAttributes(final.boruta, withTentative = FALSE)
  bSel = append(bSel, selAtt)
}
table(bSel)
plot(table(bSel), ylab = "Bootstrap Selections", xlab = "Features")
abline(h = 90, col = "blue")
xb_boot_binomial = x[,c("entropy_GLCM", "tlg.seg", "run.entropy_GLSZM", "energy", "volume", "info.corr.1_GLCM")]
data.boruta = cbind(xb_boot_binomial,y)



# Lasso


set.seed(0420)
n = nrow(x)
sel = matrix(0, nrow=K, ncol=ncol(x))
colnames(sel) = names(x)
K = 100
for(k in 1:K){
  i.train = sample(1:n,n,replace = TRUE)
  cv_lasso.lung = cv.glmnet(as.matrix(x),as.numeric(y),alpha = 1,subset = i.train)
  best_lambda <- cv_lasso.lung$lambda.min
  lasso.lung <- glmnet(as.matrix(x),as.numeric(y), alpha = 1, lambda = best_lambda, subset = i.train)
  isel = which(coef(lasso.lung)[-1] != 0)
  sel[k,isel] = 1
}
apply(sel,2,mean)*100
plot(apply(sel,2,mean)*100, pch = 20)
abline(h = 90, col = "blue")
xlass_binomial = x[,c("gradb1", "entropy_GLCM", "info.corr.1_GLCM","run.entropy_GLSZM")] 
data.lasso = cbind(xlass_binomial,y)


# E.O.Feature Selection
# ********************************************************************************************************
# Scaling

ncol(data.boruta)
data.boruta[,1:6] = as.data.frame(scale(data.boruta[,1:6]))
ncol(data.rfe)
data.rfe[,1:56] = as.data.frame(scale(data.rfe[,1:56]))
ncol(data.lasso)
data.lasso[,1:4] = as.data.frame(scale(data.lasso[,1:4]))

# E.O. Scaling
# ********************************************************************************************************
# SVM 

# Boruta features

ncol(data.boruta)

set.seed(0420)

itrain = createDataPartition(data.boruta$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)


dat_train = data.boruta[itrain,]
dat_test = data.boruta[-itrain,]

table(dat_train$y)
table(dat_test$y)



fitControl <- trainControl(
  method = "boot",
  number = 100,
  # repeats = 50,
  classProbs = TRUE,
  summaryFunction=twoClassSummary,
  savePredictions = TRUE
)


cv_svm_r = train(y~., data=dat_train, method="svmRadial", trControl=fitControl, tuneLength=5, 
                 metric = "ROC"
                 )

svm_bor_ROC = (cv_svm_r$resample[,1])*100
svm_bor_Sens = (cv_svm_r$resample[,2])*100
svm_bor_Spec = (cv_svm_r$resample[,3])*100

svm.pred <- predict(cv_svm_r, newdata = dat_test[,-ncol(dat_test)])
mat = confusionMatrix(svm.pred, dat_test[,ncol(dat_test)])

(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], svm.pred))

bor_acc_SVM = (cv_svm_r$resample[,1])*100




cv_svm_r_full = train(y~., data=data.boruta, method="svmRadial", trControl=fitControl, tuneLength=5, 
                      # metric = "ROC"
                      )


res_bor_acc = evalm(cv_svm_r_full)


svm_bor_ROC_full = (cv_svm_r_full$resample[,1])*100
(meanROC_SVM_bor = mean(svm_bor_ROC_full))
(sdROC_SVM_bor = sd(svm_bor_ROC_full))

svm_bor_Sens_full = (cv_svm_r_full$resample[,2])*100
(meanSens_SVM_bor = mean(svm_bor_Sens_full))
(sdSens_SVM_bor = sd(svm_bor_Sens_full))

svm_bor_Spec_full = (cv_svm_r_full$resample[,3])*100
(meanSpec_SVM_bor = mean(svm_bor_Spec_full))
(sdSpec_SVM_bor = sd(svm_bor_Spec_full))

bor_acc_SVM_full = (cv_svm_r_full$resample[,1])*100
(meanAcc_SVM_bor = mean(bor_acc_SVM_full))
(sdAcc_SVM_bor = sd(bor_acc_SVM_full))




# RFE Features

ncol(data.rfe)

set.seed(0420)

itrain = createDataPartition(data.rfe$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)


dat_train = data.rfe[itrain,]
dat_test = data.rfe[-itrain,]

table(dat_train$y)
table(dat_test$y)



fitControl <- trainControl(
  method = "boot",
  number = 100,
  # repeats = 50,
  classProbs = TRUE,
  summaryFunction=twoClassSummary,
  savePredictions = TRUE
)


cv_svm_r = train(y~., data=dat_train, method="svmRadial", trControl=fitControl, tuneLength=5, 
                 metric = "ROC"
                 )

svm_rfe_ROC = (cv_svm_r$resample[,1])*100
svm_rfe_Sens = (cv_svm_r$resample[,2])*100
svm_rfe_Spec = (cv_svm_r$resample[,3])*100

svm.pred <- predict(cv_svm_r, newdata = dat_test[,-ncol(dat_test)])
mat = confusionMatrix(svm.pred, dat_test[,ncol(dat_test)])

(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], svm.pred))

rfe_acc_SVM = (cv_svm_r$resample[,1])*100


cv_svm_r_full = train(y~., data=data.rfe, method="svmRadial", trControl=fitControl, tuneLength=5, 
                      # metric = "ROC"
                      )

res_rfe_acc = evalm(cv_svm_r_full)

svm_rfe_ROC_full = (cv_svm_r_full$resample[,1])*100
(meanROC_SVM_rfe = mean(svm_rfe_ROC_full))
(sdROC_SVM_rfe = sd(svm_rfe_ROC_full))

svm_rfe_Sens_full = (cv_svm_r_full$resample[,2])*100
(meanSens_SVM_rfe = mean(svm_rfe_Sens_full))
(sdSens_SVM_rfe = sd(svm_rfe_Sens_full))

svm_rfe_Spec_full = (cv_svm_r_full$resample[,3])*100
(meanSpec_SVM_rfe = mean(svm_rfe_Spec_full))
(sdSpec_SVM_rfe = sd(svm_rfe_Spec_full))


rfe_acc_SVM_full = (cv_svm_r_full$resample[,1])*100
(meanAcc_SVM_rfe = mean(rfe_acc_SVM_full))
(sdAcc_SVM_rfe = sd(rfe_acc_SVM_full))

# Lasso Features


ncol(data.lasso)

set.seed(0420)

itrain = createDataPartition(data.lasso$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)


dat_train = data.lasso[itrain,]
dat_test = data.lasso[-itrain,]

table(dat_train$y)
table(dat_test$y)



fitControl <- trainControl(
  method = "boot",
  number = 100,
  # repeats = 50,
  classProbs = TRUE,
  summaryFunction=twoClassSummary,
  savePredictions = TRUE
)


cv_svm_r = train(y~., data=dat_train, method="svmRadial", trControl=fitControl, tuneLength=5, 
                 metric = "ROC"
                 )

svm_lasso_ROC = (cv_svm_r$resample[,1])*100
svm_lasso_Sens = (cv_svm_r$resample[,2])*100
svm_lasso_Spec = (cv_svm_r$resample[,3])*100

svm.pred <- predict(cv_svm_r, newdata = dat_test[,-ncol(dat_test)])
mat = confusionMatrix(svm.pred, dat_test[,ncol(dat_test)])

(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], svm.pred))
lass_acc_SVM = (cv_svm_r$resample[,1])*100



cv_svm_r_full = train(y~., data=data.lasso, method="svmRadial", trControl=fitControl, tuneLength=5, 
                      # metric = "ROC"
                      )

res_las_acc = evalm(cv_svm_r_full)

svm_lasso_ROC_full = (cv_svm_r_full$resample[,1])*100
(meanROC_SVM_lass = mean(svm_lasso_ROC_full))
(sdROC_SVM_lass = sd(svm_lasso_ROC_full))


svm_lasso_Sens_full = (cv_svm_r_full$resample[,2])*100
(meanSens_SVM_lass = mean(svm_lasso_Sens_full))
(sdSens_SVM_lass = sd(svm_lasso_Sens_full))

svm_lasso_Spec_full = (cv_svm_r_full$resample[,3])*100
(meanSpec_SVM_lass = mean(svm_lasso_Spec_full))
(sdSpec_SVM_lass = sd(svm_lasso_Spec_full))

lass_acc_SVM_full = (cv_svm_r_full$resample[,1])*100
(meanAcc_SVM_lass = mean(lass_acc_SVM_full))
(sdAcc_SVM_lass = sd(lass_acc_SVM_full))


# Summary from SVM

boxplot(svm_bor_ROC, svm_rfe_ROC, svm_lasso_ROC)
boxplot(svm_bor_Sens, svm_rfe_Sens, svm_lasso_Sens)
boxplot(svm_bor_Spec, svm_rfe_Spec, svm_lasso_Spec)



acc_svm = c(77.78,62.96,77.78)
sen_svm = c(54.55,27.27,63.64)
spec_svm = c(93.75,87.5, 87.5)
f1_svm = c(66.67,83.33,37.5,73.68, 70, 82.35)


boxplot(svm_bor_ROC_full, svm_rfe_ROC_full, svm_lasso_ROC_full)
boxplot(svm_bor_Sens_full, svm_rfe_Sens_full, svm_lasso_Sens_full)
boxplot(svm_bor_Spec_full, svm_rfe_Spec_full, svm_lasso_Spec_full)


boxplot(bor_acc_SVM,rfe_acc_SVM,lass_acc_SVM)
boxplot(bor_acc_SVM_full,rfe_acc_SVM_full,lass_acc_SVM_full)


res_bor_acc$roc
res_bor_acc$proc

res_rfe_acc$roc
res_rfe_acc$proc

res_las_acc$roc
res_las_acc$proc




# E.O. SVM 
# ********************************************************************************************************
# KNN

# Boruta features

ncol(data.boruta)


set.seed(0420)
itrain = createDataPartition(data.boruta$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.boruta[itrain,]
dat_test = data.boruta[-itrain,]

knnGrid <-  expand.grid(k = c(1,3,5,7,9,11,13,15))

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,  
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


knnFit <- train(y~ ., # formula
                data = dat_train, # train data   
                method = "knn", # method for caret see https://topepo.github.io/caret/available-models.html for list of models 
                trControl = fitControl, 
                tuneGrid = knnGrid,
                ## Specify which metric to optimize
                # metric = "ROC"
                )

knn_bor_ROC = (knnFit$resample[,1])*100
knn_bor_Sens = (knnFit$resample[,2])*100
knn_bor_Spec = (knnFit$resample[,3])*100


plot(knnFit)
knnFit$finalModel
pred_class <- predict(knnFit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(knnFit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]

(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

bor_acc_KNN = (knnFit$resample[,1])*100


knnFit_full <- train(y~ ., # formula
                data = data.boruta, # train data   
                method = "knn", # method for caret see https://topepo.github.io/caret/available-models.html for list of models 
                trControl = fitControl, 
                tuneGrid = knnGrid,
                ## Specify which metric to optimize
                # metric = "ROC"
                )

res_bor_knn = evalm(knnFit_full)




knn_bor_ROC_full = (knnFit_full$resample[,1])*100
(meanROC_KNN_bor = mean(knn_bor_ROC_full))
(sdROC_KNN_bor = sd(knn_bor_ROC_full))

knn_bor_Sens_full = (knnFit_full$resample[,2])*100
(meanSens_KNN_bor = mean(knn_bor_Sens_full))
(sdSens_KNN_bor = sd(knn_bor_Sens_full))

knn_bor_Spec_full = (knnFit_full$resample[,3])*100
(meanSpec_KNN_bor = mean(knn_bor_Spec_full))
(sdSpec_KNN_bor = sd(knn_bor_Spec_full))

bor_acc_KNN_full = (knnFit_full$resample[,1])*100
(meanAcc_KNN_bor = mean(bor_acc_KNN_full))
(sdAcc_KNN_bor = sd(bor_acc_KNN_full))

# RFE features

ncol(data.rfe)


set.seed(0420)
itrain = createDataPartition(data.rfe$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.rfe[itrain,]
dat_test = data.rfe[-itrain,]

knnGrid <-  expand.grid(k = c(1,3,5,7,9,11,13,15))

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,  
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


knnFit <- train(y~ ., # formula
                data = dat_train, # train data   
                method = "knn", # method for caret see https://topepo.github.io/caret/available-models.html for list of models 
                trControl = fitControl, 
                tuneGrid = knnGrid,
                ## Specify which metric to optimize
                # metric = "ROC"
                )

knn_rfe_ROC = (knnFit$resample[,1])*100
knn_rfe_Sens = (knnFit$resample[,2])*100
knn_rfe_Spec = (knnFit$resample[,3])*100

plot(knnFit)
knnFit$finalModel
pred_class <- predict(knnFit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(knnFit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]

(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

rfe_acc_KNN = (knnFit$resample[,1])*100



knnFit_full <- train(y~ ., # formula
                data = data.rfe, # train data   
                method = "knn", # method for caret see https://topepo.github.io/caret/available-models.html for list of models 
                trControl = fitControl, 
                tuneGrid = knnGrid,
                ## Specify which metric to optimize
                # metric = "ROC"
                )


res_rfe_knn = evalm(knnFit_full)

knn_rfe_ROC_full = (knnFit_full$resample[,1])*100
(meanROC_KNN_rfe = mean(knn_rfe_ROC_full))
(sdROC_KNN_rfe = sd(knn_rfe_ROC_full))

knn_rfe_Sens_full = (knnFit_full$resample[,2])*100
(meanSens_KNN_rfe = mean(knn_rfe_Sens_full))
(sdSens_KNN_rfe = sd(knn_rfe_Sens_full))

knn_rfe_Spec_full = (knnFit_full$resample[,3])*100
(meanSpec_KNN_rfe = mean(knn_rfe_Spec_full))
(sdSpec_KNN_rfe = sd(knn_rfe_Spec_full))

rfe_acc_KNN_full = (knnFit_full$resample[,1])*100
(meanAcc_KNN_rfe = mean(rfe_acc_KNN_full))
(sdAcc_KNN_rfe = sd(rfe_acc_KNN_full))

# Lasso features

ncol(data.lasso)


set.seed(0420)
itrain = createDataPartition(data.lasso$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.lasso[itrain,]
dat_test = data.lasso[-itrain,]

knnGrid <-  expand.grid(k = c(1,3,5,7,9,11,13,15))

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,  
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


knnFit <- train(y~ ., # formula
                data = dat_train, # train data   
                method = "knn", # method for caret see https://topepo.github.io/caret/available-models.html for list of models 
                trControl = fitControl, 
                tuneGrid = knnGrid,
                ## Specify which metric to optimize
                # metric = "ROC"
                )

knn_lasso_ROC = (knnFit$resample[,1])*100
knn_lasso_Sens = (knnFit$resample[,2])*100
knn_lasso_Spec = (knnFit$resample[,3])*100


plot(knnFit)
knnFit$finalModel
pred_class <- predict(knnFit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(knnFit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]

(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

las_acc_KNN = (knnFit$resample[,1])*100

knnFit_full <- train(y~ ., # formula
                data = data.lasso, # train data   
                method = "knn", # method for caret see https://topepo.github.io/caret/available-models.html for list of models 
                trControl = fitControl, 
                tuneGrid = knnGrid,
                ## Specify which metric to optimize
                # metric = "ROC"
                )

res_las_knn = evalm(knnFit_full)


knn_lasso_ROC_full = (knnFit_full$resample[,1])*100
(meanROC_KNN_las = mean(knn_lasso_ROC_full))
(sdROC_KNN_las = sd(knn_lasso_ROC_full))

knn_lasso_Sens_full = (knnFit_full$resample[,2])*100
(meanSens_KNN_las = mean(knn_lasso_Sens_full))
(sdSens_KNN_las = sd(knn_lasso_Sens_full))

knn_lasso_Spec_full = (knnFit_full$resample[,3])*100
(meanSpec_KNN_las = mean(knn_lasso_Spec_full))
(sdSpec_KNN_las = sd(knn_lasso_Spec_full))

las_acc_KNN_full = (knnFit_full$resample[,1])*100
(meanAcc_KNN_las = mean(las_acc_KNN_full))
(sdAcc_KNN_las = sd(las_acc_KNN_full))

# Summary from KNN

boxplot(knn_bor_ROC, knn_rfe_ROC, knn_lasso_ROC)
boxplot(knn_bor_Sens, knn_rfe_Sens, knn_lasso_Sens)
boxplot(knn_bor_Spec, knn_rfe_Spec, knn_lasso_Spec)

acc_knn = c(77.78, 74.07, 77.78)
sen_knn = c(54.55, 36.36, 54.55)
spec_knn = c(93.75, 100, 93.75)
f1_knn = c(66.67, 83.33, 53.33, 82.05, 66.67, 83.33)


boxplot(knn_bor_ROC_full, knn_rfe_ROC_full, knn_lasso_ROC_full)
boxplot(knn_bor_Sens_full, knn_rfe_Sens_full, knn_lasso_Sens_full)
boxplot(knn_bor_Spec_full, knn_rfe_Spec_full, knn_lasso_Spec_full)



boxplot(bor_acc_KNN,rfe_acc_KNN,las_acc_KNN)
boxplot(bor_acc_KNN_full,rfe_acc_KNN_full,las_acc_KNN_full)


res_bor_knn$roc
res_rfe_knn$roc
res_las_knn$roc

# E.O. KNN
# ********************************************************************************************************
# LDA


# Boruta features 

ncol(data.boruta)

set.seed(0420)
itrain = createDataPartition(data.boruta$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.boruta[itrain,]
dat_test = data.boruta[-itrain,]

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = TRUE
)


lda.fit = train(y ~ ., data=dat_train, method="lda",
                trControl = fitControl, 
                # metric = "ROC"
                )


lda_bor_ROC = (lda.fit$resample[,1])*100
lda_bor_Sens = (lda.fit$resample[,2])*100
lda_bor_Spec = (lda.fit$resample[,3])*100


pred_class <- predict(lda.fit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(lda.fit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

bor_acc_LDA = (lda.fit$resample[,1])*100

lda.fit_full = train(y ~ ., data=data.boruta, method="lda",
                trControl = fitControl, 
                # metric = "ROC"
                )


res_bor_lda = evalm(lda.fit_full)


lda_bor_ROC_full = (lda.fit_full$resample[,1])*100
(meanROC_LDA_bor = mean(lda_bor_ROC_full))
(sdROC_LDA_bor = sd(lda_bor_ROC_full))

lda_bor_Sens_full = (lda.fit_full$resample[,2])*100
(meanSens_LDA_bor = mean(lda_bor_Sens_full))
(sdSens_LDA_bor = sd(lda_bor_Sens_full))

lda_bor_Spec_full = (lda.fit_full$resample[,3])*100
(meanSpec_LDA_bor = mean(lda_bor_Spec_full))
(sdSpec_LDA_bor = sd(lda_bor_Spec_full))

bor_acc_LDA_full = (lda.fit_full$resample[,1])*100
(meanAcc_LDA_bor = mean(bor_acc_LDA_full))
(sdAcc_LDA_bor = sd(bor_acc_LDA_full))

# RFE features 

ncol(data.rfe)

set.seed(0420)
itrain = createDataPartition(data.rfe$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.rfe[itrain,]
dat_test = data.rfe[-itrain,]

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


lda.fit = train(y ~ ., data=dat_train, method="lda",
                trControl = fitControl, 
                # metric = "ROC"
                )



lda_rfe_ROC = (lda.fit$resample[,1])*100
lda_rfe_Sens = (lda.fit$resample[,2])*100
lda_rfe_Spec = (lda.fit$resample[,3])*100


pred_class <- predict(lda.fit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(lda.fit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

rfe_acc_LDA = (lda.fit$resample[,1])*100

lda.fit_full = train(y ~ ., data=data.rfe, method="lda",
                trControl = fitControl, 
                # metric = "ROC"
                )

res_rfe_lda = evalm(lda.fit_full)


lda_rfe_ROC_full = (lda.fit_full$resample[,1])*100
(meanROC_LDA_rfe = mean(lda_rfe_ROC_full))
(sdROC_LDA_rfe = sd(lda_rfe_ROC_full))

lda_rfe_Sens_full = (lda.fit_full$resample[,2])*100
(meanSens_LDA_rfe = mean(lda_rfe_Sens_full))
(sdSens_LDA_rfe = sd(lda_rfe_Sens_full))

lda_rfe_Spec_full = (lda.fit_full$resample[,3])*100
(meanSpec_LDA_rfe = mean(lda_rfe_Spec_full))
(sdSpec_LDA_rfe = sd(lda_rfe_Spec_full))

rfe_acc_LDA_full = (lda.fit_full$resample[,1])*100
(meanAcc_LDA_rfe = mean(rfe_acc_LDA_full))
(sdAcc_LDA_rfe = sd(rfe_acc_LDA_full))

# Lasso features 

ncol(data.lasso)

set.seed(0420)
itrain = createDataPartition(data.lasso$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.lasso[itrain,]
dat_test = data.lasso[-itrain,]

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


lda.fit = train(y ~ ., data=dat_train, method="lda",
                trControl = fitControl,
                # metric = "ROC"
                )


lda_lasso_ROC = (lda.fit$resample[,1])*100
lda_lasso_Sens = (lda.fit$resample[,2])*100
lda_lasso_Spec = (lda.fit$resample[,3])*100


pred_class <- predict(lda.fit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(lda.fit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

las_acc_LDA = (lda.fit$resample[,1])*100

lda.fit_full = train(y ~ ., data=data.lasso, method="lda",
                trControl = fitControl, 
                # metric = "ROC"
                )

res_las_lda = evalm(lda.fit_full)



lda_lasso_ROC_full = (lda.fit_full$resample[,1])*100
(meanROC_LDA_las = mean(lda_lasso_ROC_full))
(sdROC_LDA_las = sd(lda_lasso_ROC_full))

lda_lasso_Sens_full = (lda.fit_full$resample[,2])*100
(meanSens_LDA_las = mean(lda_lasso_Sens_full))
(sdSens_LDA_las = sd(lda_lasso_Sens_full))

lda_lasso_Spec_full = (lda.fit_full$resample[,3])*100
(meanSpec_LDA_las = mean(lda_lasso_Spec_full))
(sdSpec_LDA_las = sd(lda_lasso_Spec_full))

las_acc_LDA_full = (lda.fit_full$resample[,1])*100
(meanAcc_LDA_las = mean(las_acc_LDA_full))
(sdAcc_LDA_las = sd(las_acc_LDA_full))

# Summary of LDA


boxplot(lda_bor_ROC, lda_rfe_ROC, lda_lasso_ROC)
boxplot(lda_bor_Sens, lda_rfe_Sens, lda_lasso_Sens)
boxplot(lda_bor_Spec, lda_rfe_Spec, lda_lasso_Spec)

acc_lda = c(74.07, 66.67, 70.37)
sen_lda = c(54.55, 63.64, 45.45)
spec_lda = c(87.5, 68.75, 87.5)
f1_lda = c(63.16, 80,60.87, 70.97,55.56, 77.78)


boxplot(lda_bor_ROC_full, lda_rfe_ROC_full, lda_lasso_ROC_full)
boxplot(lda_bor_Sens_full, lda_rfe_Sens_full, lda_lasso_Sens_full)
boxplot(lda_bor_Spec_full, lda_rfe_Spec_full, lda_lasso_Spec_full)


boxplot(bor_acc_LDA,rfe_acc_LDA,las_acc_LDA)
boxplot(bor_acc_LDA_full,rfe_acc_LDA_full,las_acc_LDA_full)


res_bor_lda$roc
res_rfe_lda$roc
res_las_lda$roc


# E.O. LDA
# ********************************************************************************************************
# QDA

# Boruta Features

ncol(data.boruta)


set.seed(0420)
itrain = createDataPartition(data.boruta$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.boruta[itrain,]
dat_test = data.boruta[-itrain,]

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


qda.fit = train(y ~ ., data=dat_train, method="qda",
                trControl = fitControl, 
                # metric = "ROC"
                )



qda_bor_ROC = (qda.fit$resample[,1])*100
qda_bor_Sens = (qda.fit$resample[,2])*100
qda_bor_Spec = (qda.fit$resample[,3])*100

summary(qda.fit)

pred_class <- predict(qda.fit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(qda.fit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

bor_res_QDA = (qda.fit$resample[,1])*100

qda.fit_full = train(y ~ ., data=data.boruta, method="qda",
                trControl = fitControl, 
                # metric = "ROC"
                )

res_bor_qda = evalm(qda.fit_full)

qda_bor_ROC_full = (qda.fit_full$resample[,1])*100
(meanROC_QDA_bor = mean(qda_bor_ROC_full))
(sdROC_QDA_bor = sd(qda_bor_ROC_full))

qda_bor_Sens_full = (qda.fit_full$resample[,2])*100
(meanSens_QDA_bor = mean(qda_bor_Sens_full))
(sdSens_QDA_bor = sd(qda_bor_Sens_full))

qda_bor_Spec_full= (qda.fit_full$resample[,3])*100
(meanSpec_QDA_bor = mean(qda_bor_Spec_full))
(sdSpec_QDA_bor = sd(qda_bor_Spec_full))

bor_res_QDA_full = (qda.fit_full$resample[,1])*100
(meanAcc_QDA_bor = mean(bor_res_QDA_full))
(sdAcc_QDA_bor = sd(bor_res_QDA_full))


# Lasso Features

ncol(data.lasso)


set.seed(0420)
itrain = createDataPartition(data.lasso$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.lasso[itrain,]
dat_test = data.lasso[-itrain,]

fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 50,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)


qda.fit = train(y ~ ., data=dat_train, method="qda",
                trControl = fitControl, 
                # metric = "ROC"
                )



qda_lasso_ROC = (qda.fit$resample[,1])*100
qda_lasso_Sens = (qda.fit$resample[,2])*100
qda_lasso_Spec = (qda.fit$resample[,3])*100


summary(qda.fit)

pred_class <- predict(qda.fit, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(qda.fit, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat$overall[1]
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

las_res_QDA = (qda.fit$resample[,1])*100


qda.fit_full = train(y ~ ., data=data.lasso, method="qda",
                trControl = fitControl, 
                # metric = "ROC"
                )

res_las_qda = evalm(qda.fit_full)

qda_lasso_ROC_full = (qda.fit_full$resample[,1])*100
(meanROC_QDA_las = mean(qda_lasso_ROC_full))
(sdROC_QDA_las = sd(qda_lasso_ROC_full))

qda_lasso_Sens_full = (qda.fit_full$resample[,2])*100
(meanSens_QDA_las = mean(qda_lasso_Sens_full))
(sdSens_QDA_las = sd(qda_lasso_Sens_full))

qda_lasso_Spec_full = (qda.fit_full$resample[,3])*100
(meanSpec_QDA_las = mean(qda_lasso_Spec_full))
(sdSpec_QDA_las = sd(qda_lasso_Spec_full))

las_res_QDA_full = (qda.fit_full$resample[,1])*100
(meanAcc_QDA_las = mean(las_res_QDA_full))
(sdAcc_QDA_las = sd(las_res_QDA_full))

# Summary of QDA

boxplot(qda_bor_ROC, qda_lasso_ROC)
boxplot(qda_bor_Sens, qda_lasso_Sens)
boxplot(qda_bor_Spec, qda_lasso_Spec)



acc_qda = c(66.67,0,70.37)
sen_qda = c(63.64,0,54.55)
spec_qda = c(68.75,0,81.25)
f1_qda = c(60.87, 70.97,0,0,60,76.47)


boxplot(qda_bor_ROC_full, qda_lasso_ROC_full)
boxplot(qda_bor_Sens_full, qda_lasso_Sens_full)
boxplot(qda_bor_Spec_full, qda_lasso_Spec_full)


boxplot(bor_res_QDA, las_res_QDA)
boxplot(bor_res_QDA_full, las_res_QDA_full)

res_bor_qda$roc
res_las_qda$roc


# E.O. QDA
# ********************************************************************************************************
# Weighted Logistic regression


# Boruta Features

ncol(data.boruta)

set.seed(0420)
itrain = createDataPartition(data.boruta$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.boruta[itrain,]
dat_test = data.boruta[-itrain,]


tbt = table(data.boruta$y)
ws = rev(as.numeric(tbt/sum(tbt)))

w = numeric(nrow(dat_train))
l = levels(dat_train$y)
w[which(dat_train$y == l[1])] = ws[1]
w[which(dat_train$y == l[2])] = ws[2]



fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 5,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)



mod = train(dat_train[,-ncol(dat_train)], dat_train$y ,family = "binomial", method = "glm", 
            trControl = fitControl, weights = w,
            # metric = "ROC"
            )



LR_bor_ROC = (mod$resample[,1])*100
LR_bor_Sens = (mod$resample[,2])*100
LR_bor_Spec = (mod$resample[,3])*100



pred_class <- predict(mod, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(mod, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

bor_res_LR = (mod$resample[,1])*100





tbt = table(data.boruta$y)
ws = rev(as.numeric(tbt/sum(tbt)))

w = numeric(nrow(data.boruta))
l = levels(data.boruta$y)
w[which(data.boruta$y == l[1])] = ws[1]
w[which(data.boruta$y == l[2])] = ws[2]


mod_full = train(data.boruta[,-ncol(data.boruta)], data.boruta$y ,family = "binomial", method = "glm", 
            trControl = fitControl, weights = w, 
            # metric = "ROC"
            )

res_bor_lr = evalm(mod_full)

LR_bor_ROC_full = (mod_full$resample[,1])*100
(meanROC_LR_bor = mean(LR_bor_ROC_full))
(sdROC_LR_bor = sd(LR_bor_ROC_full))

LR_bor_Sens_full = (mod_full$resample[,2])*100
(meanSens_LR_bor = mean(LR_bor_Sens_full))
(sdSens_LR_bor = sd(LR_bor_Sens_full))

LR_bor_Spec_full = (mod_full$resample[,3])*100
(meanSpec_LR_bor = mean(LR_bor_Spec_full))
(sdSpec_LR_bor = sd(LR_bor_Spec_full))

bor_res_LR_full = (mod_full$resample[,1])*100
(meanAcc_LR_bor = mean(bor_res_LR_full))
(sdAcc_LR_bor = sd(bor_res_LR_full))

# RFE features


ncol(data.rfe)

set.seed(0420)
itrain = createDataPartition(data.rfe$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.rfe[itrain,]
dat_test = data.rfe[-itrain,]


tbt = table(data.rfe$y)
ws = rev(as.numeric(tbt/sum(tbt)))

w = numeric(nrow(dat_train))
l = levels(dat_train$y)
w[which(dat_train$y == l[1])] = ws[1]
w[which(dat_train$y == l[2])] = ws[2]



fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 5,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)



mod = train(dat_train[,-ncol(dat_train)], dat_train$y ,family = "binomial", method = "glm", 
            trControl = fitControl, weights = w, 
            # metric = "ROC"
            )





LR_rfe_ROC = (mod$resample[,1])*100
LR_rfe_Sens = (mod$resample[,2])*100
LR_rfe_Spec = (mod$resample[,3])*100



pred_class <- predict(mod, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(mod, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

rfe_res_LR = (mod$resample[,1])*100


tbt = table(data.rfe$y)
ws = rev(as.numeric(tbt/sum(tbt)))

w = numeric(nrow(data.rfe))
l = levels(data.rfe$y)
w[which(data.rfe$y == l[1])] = ws[1]
w[which(data.rfe$y == l[2])] = ws[2]


mod_full = train(data.rfe[,-ncol(data.rfe)], data.rfe$y ,family = "binomial", method = "glm", 
                 trControl = fitControl, weights = w, 
                 # metric = "ROC"
                 )

res_rfe_lr = evalm(mod_full)

LR_rfe_ROC_full = (mod_full$resample[,1])*100
(meanROC_LR_rfe = mean(LR_rfe_ROC_full))
(sdROC_LR_rfe = sd(LR_rfe_ROC_full))

LR_rfe_Sens_full = (mod_full$resample[,2])*100
(meanSens_LR_rfe = mean(LR_rfe_Sens_full))
(sdSens_LR_rfe = sd(LR_rfe_Sens_full))

LR_rfe_Spec_full = (mod_full$resample[,3])*100
(meanSpec_LR_rfe = mean(LR_rfe_Spec_full))
(sdSpec_LR_rfe = sd(LR_rfe_Spec_full))

rfe_res_LR_full = (mod_full$resample[,1])*100
(meanAcc_LR_rfe = mean(rfe_res_LR_full))
(sdAcc_LR_rfe = sd(rfe_res_LR_full))

# Lasso features


ncol(data.lasso)

set.seed(0420)
itrain = createDataPartition(data.lasso$y, p = .70, list = FALSE, times = 1)[,1]
length(itrain)

dat_train = data.lasso[itrain,]
dat_test = data.lasso[-itrain,]


tbt = table(data.lasso$y)
ws = rev(as.numeric(tbt/sum(tbt)))

w = numeric(nrow(dat_train))
l = levels(dat_train$y)
w[which(dat_train$y == l[1])] = ws[1]
w[which(dat_train$y == l[2])] = ws[2]



fitControl <- trainControl(method = "boot",
                           number = 100, 
                           # repeats = 5,
                           classProbs = TRUE,
                           summaryFunction=twoClassSummary,
                           savePredictions = T
)



mod = train(dat_train[,-ncol(dat_train)], dat_train$y ,family = "binomial", method = "glm", 
            trControl = fitControl, weights = w,
            # metric = "ROC"
            )


LR_lasso_ROC = (mod$resample[,1])*100
LR_lasso_Sens = (mod$resample[,2])*100
LR_lasso_Spec = (mod$resample[,3])*100



pred_class <- predict(mod, dat_test[,-ncol(dat_test)],'raw')
probs <- predict(mod, dat_test[-ncol(dat_test)],'prob')
mat = confusionMatrix(pred_class, dat_test[,ncol(dat_test)])
mat
(precision <- diag(mat$table) / rowSums(mat$table))
(recall <- (diag(mat$table) / colSums(mat$table)))
(f1 = (2*precision*recall)/(precision + recall))
(accuracy = mat$overall[1]*100)
(mccr(dat_test[,ncol(dat_test)], pred_class))

las_res_LR = (mod$resample[,1])*100



tbt = table(data.lasso$y)
ws = rev(as.numeric(tbt/sum(tbt)))

w = numeric(nrow(data.lasso))
l = levels(data.lasso$y)
w[which(data.lasso$y == l[1])] = ws[1]
w[which(data.lasso$y == l[2])] = ws[2]


mod_full = train(data.lasso[,-ncol(data.lasso)], data.lasso$y ,family = "binomial", method = "glm", 
                 trControl = fitControl, weights = w, 
                 # metric = "ROC"
                 )


res_las_lr = evalm(mod_full)

LR_lasso_ROC_full = (mod_full$resample[,1])*100
(meanROC_LR_las = mean(LR_lasso_ROC_full))
(sdROC_LR_las = sd(LR_lasso_ROC_full))

LR_lasso_Sens_full = (mod_full$resample[,2])*100
(meanSens_LR_las = mean(LR_lasso_Sens_full))
(sdSens_LR_las = sd(LR_lasso_Sens_full))

LR_lasso_Spec_full = (mod_full$resample[,3])*100
(meanSpec_LR_las = mean(LR_lasso_Spec_full))
(sdSpec_LR_las = sd(LR_lasso_Spec_full))

las_res_LR_full = (mod_full$resample[,1])*100
(meanAcc_LR_las = mean(las_res_LR_full))
(sdAcc_LR_las = sd(las_res_LR_full))




# Summary

boxplot(LR_bor_ROC,LR_rfe_ROC ,LR_lasso_ROC)
boxplot(LR_bor_Sens,LR_rfe_Sens ,LR_lasso_Sens)
boxplot(LR_bor_Spec,LR_rfe_Spec ,LR_lasso_Spec)



acc_LR = c(70.37, 70.37, 74.07)
sen_LR = c(54.55, 81.82, 54.55)
spec_LR = c(81.25, 62.5, 87.5)
f1_LR = c(60,76.47, 69.23,71.42, 63.15, 80)



boxplot(bor_res_LR, rfe_res_LR, las_res_LR)
boxplot(bor_res_LR_full, rfe_res_LR_full, las_res_LR_full)


res_bor_lr$roc
res_rfe_lr$roc
res_las_lr$roc



# E.O. Weighted Logistic regression
# ********************************************************************************************************
# Plots

# 70% Dataset

# ROC

boxplot(svm_bor_ROC, svm_rfe_ROC, svm_lasso_ROC, 
        knn_bor_ROC, knn_rfe_ROC, knn_lasso_ROC,
        lda_bor_ROC, lda_rfe_ROC, lda_lasso_ROC,
        qda_bor_ROC, qda_lasso_ROC,
        LR_bor_ROC, LR_rfe_ROC, LR_lasso_ROC,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "ROC", ylim = c(0,100)
)

abline(h = c(20,30,40,50,60,70,80,90), lty = 2)
boxplot(svm_bor_ROC, svm_rfe_ROC, svm_lasso_ROC, 
        knn_bor_ROC, knn_rfe_ROC, knn_lasso_ROC,
        lda_bor_ROC, lda_rfe_ROC, lda_lasso_ROC,
        qda_bor_ROC, qda_lasso_ROC,
        LR_bor_ROC, LR_rfe_ROC, LR_lasso_ROC,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "ROC", add = T,  ylim = c(0,100)
)
legend(6, 15, legend = c("SVM", "KNN","LDA","QDA", "WKNN"), col = c(7,2,3,4,5), lty = 1, cex = 1)

# Sensitivity

boxplot(svm_bor_Sens, svm_rfe_Sens, svm_lasso_Sens, 
        knn_bor_Sens, knn_rfe_Sens, knn_lasso_Sens,
        lda_bor_Sens, lda_rfe_Sens, lda_lasso_Sens,
        qda_bor_Sens, qda_lasso_Sens,
        LR_bor_Sens, LR_rfe_Sens, LR_lasso_Sens,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Sensitivity"
)

abline(h = c(20,30,40,50,60,70,80,90), lty = 2)

boxplot(svm_bor_Sens, svm_rfe_Sens, svm_lasso_Sens, 
        knn_bor_Sens, knn_rfe_Sens, knn_lasso_Sens,
        lda_bor_Sens, lda_rfe_Sens, lda_lasso_Sens,
        qda_bor_Sens, qda_lasso_Sens,
        LR_bor_Sens, LR_rfe_Sens, LR_lasso_Sens,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Sensitivity", add = T
)

# Specificity



boxplot(svm_bor_Spec, svm_rfe_Spec, svm_lasso_Spec, 
        knn_bor_Spec, knn_rfe_Spec, knn_lasso_Spec,
        lda_bor_Spec, lda_rfe_Spec, lda_lasso_Spec,
        qda_bor_Spec, qda_lasso_Spec,
        LR_bor_Spec, LR_rfe_Spec, LR_lasso_Spec,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Specificity", ylim = c(0,100)
)

abline(h = c(20,30,40,50,60,70,80,90), lty = 2)
boxplot(svm_bor_Spec, svm_rfe_Spec, svm_lasso_Spec, 
        knn_bor_Spec, knn_rfe_Spec, knn_lasso_Spec,
        lda_bor_Spec, lda_rfe_Spec, lda_lasso_Spec,
        qda_bor_Spec, qda_lasso_Spec,
        LR_bor_Spec, LR_rfe_Spec, LR_lasso_Spec,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Specificity", add = T, ylim = c(0,100)
)


# Test Accuracy

boxplot(bor_acc_SVM, rfe_acc_SVM, lass_acc_SVM, 
        bor_acc_KNN, rfe_acc_KNN, las_acc_KNN,
        bor_acc_LDA, rfe_acc_LDA, las_acc_LDA,
        bor_res_QDA, las_res_QDA,
        bor_res_LR, rfe_res_LR, las_res_LR,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Accuracy", ylim = c(0,100)
)

abline(h = c(20,30,40,50,60,70,80,90), lty = 2)
boxplot(bor_acc_SVM, rfe_acc_SVM, lass_acc_SVM, 
        bor_acc_KNN, rfe_acc_KNN, las_acc_KNN,
        bor_acc_LDA, rfe_acc_LDA, las_acc_LDA,
        bor_res_QDA, las_res_QDA,
        bor_res_LR, rfe_res_LR, las_res_LR,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Accuracy", ylim = c(0,100), add = T
)
legend(6, 15, legend = c("SVM", "KNN","LDA","QDA", "WKNN"), col = c(7,2,3,4,5), lty = 1, cex = 1)



# Full Dataset



# ROC

boxplot(svm_bor_ROC_full, svm_rfe_ROC_full, svm_lasso_ROC_full, 
        knn_bor_ROC_full, knn_rfe_ROC_full, knn_lasso_ROC_full,
        lda_bor_ROC_full, lda_rfe_ROC_full, lda_lasso_ROC_full,
        qda_bor_ROC_full, qda_lasso_ROC_full,
        LR_bor_ROC_full, LR_rfe_ROC_full, LR_lasso_ROC_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "ROC", ylim = c(0,100)
)

abline(h = c(20,30,40,50,60,70,80,90), lty = 2)
boxplot(svm_bor_ROC_full, svm_rfe_ROC_full, svm_lasso_ROC_full, 
        knn_bor_ROC_full, knn_rfe_ROC_full, knn_lasso_ROC_full,
        lda_bor_ROC_full, lda_rfe_ROC_full, lda_lasso_ROC_full,
        qda_bor_ROC_full, qda_lasso_ROC_full,
        LR_bor_ROC_full, LR_rfe_ROC_full, LR_lasso_ROC_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "ROC", ylim = c(0,100), add = T, main = "Full Dataset"
)
legend(6, 15, legend = c("SVM", "KNN","LDA","QDA", "WKNN"), col = c(7,2,3,4,5), lty = 1, cex = 1)

# Sensitivity

boxplot(svm_bor_Sens_full, svm_rfe_Sens_full, svm_lasso_Sens_full, 
        knn_bor_Sens_full, knn_rfe_Sens_full, knn_lasso_Sens_full,
        lda_bor_Sens_full, lda_rfe_Sens_full, lda_lasso_Sens_full,
        qda_bor_Sens_full, qda_lasso_Sens_full,
        LR_bor_Sens_full, LR_rfe_Sens_full, LR_lasso_Sens_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Sensitivity", ylim = c(0,100)
)

abline(h = c(10,20,30,40,50,60,70,80,90), lty = 2)

boxplot(svm_bor_Sens_full, svm_rfe_Sens_full, svm_lasso_Sens_full, 
        knn_bor_Sens_full, knn_rfe_Sens_full, knn_lasso_Sens_full,
        lda_bor_Sens_full, lda_rfe_Sens_full, lda_lasso_Sens_full,
        qda_bor_Sens_full, qda_lasso_Sens_full,
        LR_bor_Sens_full, LR_rfe_Sens_full, LR_lasso_Sens_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Sensitivity", ylim = c(0,100), add = T 
)

# Specificity



boxplot(svm_bor_Spec_full, svm_rfe_Spec_full, svm_lasso_Spec_full, 
        knn_bor_Spec_full, knn_rfe_Spec_full, knn_lasso_Spec_full,
        lda_bor_Spec_full, lda_rfe_Spec_full, lda_lasso_Spec_full,
        qda_bor_Spec_full, qda_lasso_Spec_full,
        LR_bor_Spec_full, LR_rfe_Spec_full, LR_lasso_Spec_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Specificity", ylim = c(0,100)
)

abline(h = c(10,20,30,40,50,60,70,80,90), lty = 2)
boxplot(svm_bor_Spec_full, svm_rfe_Spec_full, svm_lasso_Spec_full, 
        knn_bor_Spec_full, knn_rfe_Spec_full, knn_lasso_Spec_full,
        lda_bor_Spec_full, lda_rfe_Spec_full, lda_lasso_Spec_full,
        qda_bor_Spec_full, qda_lasso_Spec_full,
        LR_bor_Spec_full, LR_rfe_Spec_full, LR_lasso_Spec_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor",  "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Specificity", ylim = c(0,100), add = T
)


# Test Accuracy

boxplot(bor_acc_SVM_full, rfe_acc_SVM_full, lass_acc_SVM_full, 
        bor_acc_KNN_full, rfe_acc_KNN_full, las_acc_KNN_full,
        bor_acc_LDA_full, rfe_acc_LDA_full, las_acc_LDA_full,
        bor_res_QDA_full, las_res_QDA_full,
        bor_res_LR_full, rfe_res_LR_full, las_res_LR_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Accuracy", ylim = c(0,100)
)

abline(h = c(20,30,40,50,60,70,80,90), lty = 2)
boxplot(bor_acc_SVM_full, rfe_acc_SVM_full, lass_acc_SVM_full, 
        bor_acc_KNN_full, rfe_acc_KNN_full, las_acc_KNN_full,
        bor_acc_LDA_full, rfe_acc_LDA_full, las_acc_LDA_full,
        bor_res_QDA_full, las_res_QDA_full,
        bor_res_LR_full, rfe_res_LR_full, las_res_LR_full,
        col = c(7,7,7,2,2,2,3,3,3,4,4,5,5,5),
        names = c("SVM - Bor", "SVM - RFE", "SVM - Las",
                  "KNN - Bor", "KNN - RFE", "KNN - Las",
                  "LDA - Bor", "LDA - RFE", "LDA - Las",
                  "QDA - Bor", "QDA - Las",
                  "LR - Bor", "LR - RFE", "LR - Las"),
        ylab = "Accuracy", ylim = c(0,100), add = TRUE
)
legend(6, 15, legend = c("SVM", "KNN","LDA","QDA", "WKNN"), col = c(7,2,3,4,5), lty = 1, cex = 1)






70.25+1.96*7.3
70.25-1.96*7.3

67.6+1.96*7.3
67.6-1.96*7.3







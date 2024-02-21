rm(list = ls())
library(survival)
library(tidyverse)
library(simsurv)
library(MASS)
library(deepTL)
library(survivalmodels)
library(randomForestSRC)
library(survivalsvm)
library(reticulate)
library(Hmisc)
library(xgboost)
library(survivalsvm)
library(pheatmap)
library(timeROC)


print("Load successfully")



args1 = commandArgs(TRUE)
#args1=c(4)
Folder_args = as.numeric(args1[1])
#start.seed 
seedused= as.numeric(args1[1])
#= (start.seed+42) %% 11 + 43
Fold_Name = "/nas/longleaf/home/xiaoninz/PermFit"
print(seedused)

impfile = paste(Fold_Name,"/","original_importance","_",seedused,".csv",sep = "")
msefile = paste(Fold_Name,"/","original_mse","_",seedused,".csv",sep = "")
prefile = paste(Fold_Name,"/","original_prediction","_",seedused,".csv",sep = "")
#impfile = "/nas/longleaf/home/bzou/perdna/result/impDNAKRAT3C106.csv";
#msefile = "/nas/longleaf/home/bzou/perdna/result/mseDNAKRAT3C106_seeds.csv";
#prefile = "/nas/longleaf/home/bzou/perdna/result/preDNAKRAT3C106.csv";



pvacut = 0.1

dat = read.csv("../df.csv",head=T,row.names=1)
#dat = read.csv("df.csv",head=T,row.names=1)

print("read file successfully")
dat$Mean_Opioid_Daily_Push <- log(dat$Mean_Opioid_Daily_Push+1)

dimsize = dim(dat)[2]
y = dat$Mean_Opioid_Daily_Push
x = as.matrix(subset(dat,select = -Mean_Opioid_Daily_Push))
dimsizx = dim(x)[2]
print(dim(x))

# 
if (seedused == 1)  {
  newx = x
  newy = y
  posct = dat[,"Mean_Opioid_Daily_Push"]
}
if (seedused > 1)  {
 set.seed(20220301*seedused)
index = sample(1:length(y),length(y),replace=T)
 newx = x[index,]
 newy = y[index]
 posct = dat[index,"Mean_Opioid_Daily_Push"]
}





#### Functions for Evaluation
mse = function(y, x) mean((y - x)**2)
pcc = function(y, x) cor(x,y,method="pearson")


#### 0.0 Categorical feature list ####
pathwaylist = list(INSURANCE=67:68,ETHNICITY=69:72,MARITAL_STATUS=74:77,ADMIT_TYPE=78:79,SURGERY=80:85,ADMIT_LOC=86:89)
# pathwaylist
conlist = c(1:66,73)
#GENDER_F=33,


#### 0.1 Hyper-parameters ####
n_ensemble = 100  #100
n_perm = 100  #100
new_perm = 200 #200  #100
fold_num = 10  #10
new_fold = 10 
num_sample = 400 #400 # 1/10 original size 
n_epoch = 1000  #1000
n_tree = 1000 #1000
node_size = 5 
esCtrl = list(n.hidden = c(50, 40, 30, 20), activate = "relu", l1.reg = 10**-4, early.stop.det = 1000, n.batch = 30, 
              n.epoch = n_epoch, learning.rate.adaptive = "adam", plot = FALSE)
print(c("esCtrl",esCtrl))




#### 0.2 Random Shuffle
if (seedused == 1)  set.seed(20220301)
shuffle = sample(length(y))
oy = rep(0,length(posct))


#### 2. Cross-Validation ####
validate = shuffle[1:num_sample]
oy[validate] = posct[validate]
trainx = newx[-validate, ]
trainy = newy[-validate]
trainlen = length(trainy)
validatx = newx[validate,]
validaty = newy[validate]


numeric_col <- grep("PainScore|AGE|LOS.|post",colnames(x))
trainx[,numeric_col] <- scale(trainx[,numeric_col])
validatx[,numeric_col] <- scale(validatx[,numeric_col])


dnn_obj = importDnnet(x = trainx, y = trainy)
pred = matrix(NA, num_sample, 9)

print("Cross Validation Split Successfully")




### Baseline Model: Generalized Multiple Linear Regression
train = data.frame(y = trainy, trainx)
valid = data.frame(y=validaty,validatx)
fit2 <- lm(y~.,data = train)
pred[, 9] <-  predict(fit2,valid)



print("Baseline LM")








#### base DNN ####
dnn_mod = ensemble_dnnet(dnn_obj, n_ensemble, esCtrl, verbose = 0)
### pred[validate, 1] = predict(dnn_mod, x[validate, ])
pred[, 1] = predict(dnn_mod, validatx)
nshuffle = sample(trainlen)



print("DNN_mod successfully")



#### PermFIT ####
#### 2.2 PermFIT-DNN ####
npermfit_dnn = permfit(train = dnn_obj, k_fold = new_fold, n.ensemble = n_ensemble, n_perm = new_perm, pathway_list = pathwaylist, method = "ensemble_dnnet", shuffle = nshuffle, esCtrl = esCtrl, verbose = 0) 
print("npermfit_dnn finished")

#### 2.3 PermFIT-SVM ####
npermfit_svm = permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "svm", shuffle = nshuffle, n.ensemble = n_ensemble) 
print("npermfit_svm finished")

#### 2.4 PermFIT-RF ####
npermfit_rf = permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "random_forest", shuffle = nshuffle, n.ensemble = n_ensemble, ntree = n_tree, nodesize = node_size) 
print("npermfit_rf finished")



#### 2.5 PermFIT-XGB ####
parms = list(booster="gbtree", objective="reg:linear", eta=0.3, gamma=0, max_depth=5, min_child_weight=1, subsample=1, colsample_bytree=1) 
npermfit_xgb = permfit(train = dnn_obj, k_fold = new_fold, n_perm = new_perm, pathway_list = pathwaylist, method = "xgboost", shuffle = nshuffle, params = parms) 
print("permfit_xgb successfully")




ndnn_feature = which(npermfit_dnn@importance$importance_pval <= pvacut) 
ndnn_feature = intersect(ndnn_feature,conlist) 
dnn_cat = which(npermfit_dnn@block_importance$importance_pval <= pvacut) 
dcatlen = length(dnn_cat) 
if (dcatlen > 0)  {
  for (i in 1:dcatlen)  ndnn_feature = c(ndnn_feature,pathwaylist[[dnn_cat[i]]]) 
}


nsvm_feature = which(npermfit_svm@importance$importance_pval <= pvacut) 
nsvm_feature = intersect(nsvm_feature,conlist) 
svm_cat = which(npermfit_svm@block_importance$importance_pval <= pvacut) 
scatlen = length(svm_cat) 
if (scatlen > 0)  {
  for (i in 1:scatlen)  nsvm_feature = c(nsvm_feature,pathwaylist[[svm_cat[i]]]) 
}



nrf_feature = which(npermfit_rf@importance$importance_pval <= pvacut) 
nrf_feature = intersect(nrf_feature,conlist) 
rf_cat = which(npermfit_rf@block_importance$importance_pval <= pvacut) 
rcatlen = length(rf_cat) 
if (rcatlen > 0)  {
  for (i in 1:rcatlen)  nrf_feature = c(nrf_feature,pathwaylist[[rf_cat[i]]]) 
}



nxgb_feature = which(npermfit_xgb@importance$importance_pval <= pvacut) 
nxgb_feature = intersect(nxgb_feature,conlist) 
xgb_cat = which(npermfit_xgb@block_importance$importance_pval <= pvacut) 
xcatlen = length(xgb_cat) 
if (xcatlen > 0)  {
  for (i in 1:xcatlen)  nxgb_feature = c(nxgb_feature,pathwaylist[[xgb_cat[i]]]) 
}



print("permfit finished")



#### 2.2 PermFIT-DNN ####
dnn_mod = ensemble_dnnet(importDnnet(x = as.matrix(trainx[, ndnn_feature]), y = trainy), n_ensemble, esCtrl, verbose = 0) 
# pred[validate, 2] = predict(dnn_mod, as.matrix(newx[validate, ndnn_feature])) 
pred[, 2] = predict(dnn_mod, as.matrix(validatx[, ndnn_feature])) 

#### 2.4 SVM ####
svm_mod = tune.svm(trainx, trainy, gamma = 10**(-(0:4)), cost = 10**(0:4/2), 
                   tunecontrol = tune.control(cross = fold_num)) 
svm_mod = svm(trainx, trainy, gamma = svm_mod$best.parameters$gamma, cost = svm_mod$best.parameters$cost) 
# pred[validate, 3] = predict(svm_mod, as.matrix(newx[validate, ])) 
pred[, 3] = predict(svm_mod, as.matrix(validatx)) 

#### 2.5 PermFIT-SVM ####
svm_mod = tune.svm(as.matrix(trainx[, nsvm_feature]), trainy, gamma = 10**(-(0:4)), cost = 10**(0:4/2), 
                   tunecontrol = tune.control(cross = fold_num)) 
svm_mod = svm(as.matrix(trainx[, nsvm_feature]), trainy, gamma = svm_mod$best.parameters$gamma, cost = svm_mod$best.parameters$cost) 
# pred[validate, 4] = predict(svm_mod, as.matrix(newx[validate, nsvm_feature])) 
pred[, 4] = predict(svm_mod, as.matrix(validatx[, nsvm_feature])) 

#### 2.7 RF ####
rf_mod = randomForest(trainx, trainy, ntree = n_tree, nodesize = node_size, importance = TRUE) 
# pred[validate, 5] = predict(rf_mod, newx[validate, ]) 
pred[, 5] = predict(rf_mod, validatx) 

#### 2.8 PermFIT-RF ####
rf_mod = randomForest(as.matrix(trainx[, nrf_feature]), trainy, ntree = n_tree, nodesize = node_size, importance = TRUE) 
# pred[validate, 6] = predict(rf_mod, as.matrix(newx[validate, nrf_feature])) 
pred[, 6] = predict(rf_mod, as.matrix(validatx[, nrf_feature])) 

#### 2.13 XGBoost ####
xgbtrain = xgb.DMatrix(data=trainx,label=trainy) 
xgbtest = xgb.DMatrix(data=validatx,label=validaty) 
xgb_mod = xgb.train(params=parms, data=xgbtrain, nrounds=5, watchlist=list(train=xgbtrain), print_every_n=NULL, maximize=F, eval_metric="rmse") 
# pred[validate, 7] = predict(xgb_mod,xgbtest) 
pred[, 7] = predict(xgb_mod,xgbtest) 

#### 2.14 PermFIT-XGB ####
xgbtrain = xgb.DMatrix(data = trainx[, nxgb_feature], label = trainy)
xgbtest = xgb.DMatrix(data = validatx[, nxgb_feature], label = validaty)
xgb_mod = xgb.train(
  params = parms,
  data = xgbtrain,
  nrounds = 5,
  watchlist = list(train = xgbtrain),
  print_every_n = NULL,
  maximize = F,
  eval_metric = "rmse"
)
# pred[validate, 8] = predict(xgb_mod,xgbtest)
pred[, 8] = predict(xgb_mod, xgbtest)

# }

colnames(pred)<-c("DNN","PF-DNN","SVM","PF-SVM","RF","PF-RF","XGB","PF-XGB","GLM")

print("comparison successfully")

#### 3. Summary

#if (seedused == 1)  {
  
## 3.1 Importance scores and p-values
dfimp = data.frame(var_name = c(colnames(x)[conlist],names(pathwaylist)),
                   'PermFIT-DNN-IMP' = paste0(round(c(npermfit_dnn@importance$importance[conlist],npermfit_dnn@block_importance$importance), 5)),
                   'PermFIT-DNN-PVL' = paste0(round(c(npermfit_dnn@importance$importance_pval[conlist],npermfit_dnn@block_importance$importance_pval), 5)),
                   'PermFIT-SVM-IMP' = paste0(round(c(npermfit_svm@importance$importance[conlist],npermfit_svm@block_importance$importance), 5)),
                   'PermFIT-SVM-PVL' = paste0(round(c(npermfit_svm@importance$importance_pval[conlist],npermfit_svm@block_importance$importance_pval), 5)),
                   'PermFIT-RF-IMP' = paste0(round(c(npermfit_rf@importance$importance[conlist],npermfit_rf@block_importance$importance), 5)),
                   'PermFIT-RF-PVL' = paste0(round(c(npermfit_rf@importance$importance_pval[conlist],npermfit_rf@block_importance$importance_pval), 5)),
                   'PermFIT-XGB-IMP' = paste0(round(c(npermfit_xgb@importance$importance[conlist],npermfit_xgb@block_importance$importance), 5)),
                   'PermFIT-XGB-PVL' = paste0(round(c(npermfit_xgb@importance$importance_pval[conlist],npermfit_xgb@block_importance$importance_pval), 5))) 

## 3.3 Observed and predicted
dfpre = data.frame('PosCT'=oy[validate],'DNN'=pred[,1],'PermFIT-DNN'=pred[,2],'SVM'=pred[,3],'PermFIT-SVM'=pred[,4],'RF'=pred[,5],'PermFIT-RF'=pred[,6],'XGBoost'=pred[,7],'PermFIT-XGB'=pred[,8]) 


write.csv(dfimp,file=impfile) 
write.csv(dfpre,file=prefile) 
#}
#          
## Note in the paper, PermFIT is repeated 100 times.
## 3.2 Performace
dfmse = data.frame(Method = c("DNN", "PermFIT-DNN", "SVM", "PermFIT-SVM", "RF", "PermFIT-RF", "XGBoost", "PermFIT-XGB","GLM"),
                   MPSE = round(apply(pred, 2, function(x) mse(y = newy[validate], x)),3), PCC = round(apply(pred, 2, function(x) pcc(y = newy[validate], x)),3)) 

## Performance is evaluated via 10-fold CV, randomly repeated for 100 times
# msefile <- paste()
write.csv(dfmse,file=msefile)
            #,append=F,row.names=F,col.names=T,sep=",") 














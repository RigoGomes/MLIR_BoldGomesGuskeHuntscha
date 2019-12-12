#################################################
# Problem Set 2
# Philip Bold - 461075
# Johannes Huntscha - 466310
# Ricardo Gomes - 421962
# Phillip Guske - 465343
#################################################

#install.packages(c("ISLR", "ggplot2", "randomForest", "gbm", "tree", "e1071", "mlr"))


###############################################
# Task 1
###############################################

library(ISLR)
head(Carseats)
summary(Carseats)


# Subtask 1

  set.seed(815)
  half_auto = sample(nrow(Carseats), size=nrow(Carseats)/2, replace = FALSE )
  half_auto
  tr_auto = Carseats[half_auto,]
  te_auto = Carseats[-half_auto,]
  summary(tr_auto)
  summary(te_auto)
  # both contain 200 observations each


# Subtask 2

  library(tree)
  attach(Carseats)
  
  
  tree.carseats = tree(Sales~., Carseats, subset=half_auto)
  summary(tree.carseats)
  
  # The variables used to grow the tree are "ShelveLoc", "Price", "Age", "Income", "CompPrice" and "Advertising".
  # There is a total of 18 terminal nodes.
  # The residual mean deviance of a regression tree can be compared to the sum of squared errors.
  # It can be computed by dividing the Total residual deviance by the Number of observations, adjusted by the number of terminal nodes.
  # In this case, it is equal to  2.147.
  
  plot(tree.carseats)
  text(tree.carseats, pretty=0, cex=0.75)
  
  # ShelveLoc seems to be the most important variable as it represents the first internal node.
  # The 2nd most important seems to be Price, since on both of the first subtrees the internal node is given by different levels of Price.
  # The following internal nodes consider Age and CompPrice.
  # In total 6 variables have been used to construct the tree.
  
  # Test error rate:
  
  yhat= predict(tree.carseats, newdata = Carseats[-half_auto,])
  carseats.test = Carseats[-half_auto, "Sales"]
  plot(yhat, carseats.test, xlim = c(0, 16), ylim = c(0, 16))
  abline (0, 1)
  mean((yhat - carseats.test) ^2)
  
  # The MSE equals 4.518836. 
  # As we compute a regression tree and not a classification tree, we cannot compute an error rate as we do not look at a binary (true/false) classification. 
  # Therefore we use the MSE as a proxy for the test error. 
  

# Subtask 3

  set.seed(815)
  cv.carseats = cv.tree(tree.carseats, FUN = prune.tree)
  cv.carseats
  plot(cv.carseats)
  plot(cv.carseats$size, cv.carseats$dev, type= "b")
  # By looking at the deviance values, size=18 seems to have the lowest deviance. The next best value is size=10.
  # By looking at the plotted graph, size=18 seems to have the lowest deviance. The next best value is size=10.
  
  # Sidenote:
  # We only have 17 values for deviance and k.
  # A value for size=16 is missing (see graph). 
  # There might be no possibility to prune the tree such that it only has 16 terminal nodes.
  
  # First best solution (size=18):
  
  prune.carseats18 = prune.tree(tree.carseats, best = 18)
  plot(prune.carseats18)
  text(prune.carseats18, pretty=0, cex=0.75)
  
  # MSE of original (unpruned) tree:
  yhat= predict(tree.carseats, newdata = Carseats[-half_auto,])
  carseats.test = Carseats[-half_auto, "Sales"]
  plot(yhat, carseats.test, xlim = c(0, 16), ylim = c(0, 16))
  abline (0, 1)
  mean((yhat - carseats.test) ^2)
  
  # MSE of the "pruned" tree with size=18:
  yhat.pruned18= predict(prune.carseats18, newdata = Carseats[-half_auto,])
  plot(yhat.pruned18, carseats.test, xlim = c(0, 16), ylim = c(0, 16))
  abline (0, 1)
  mean((yhat.pruned18 - carseats.test) ^2)
  
  # The values are the same as the pruning size of 18 equals the original size. 
  # The MSE value for the unpruned tree is the best possible MSE value.
  
  # Second-best solution (size=10):
  
  prune.carseats10 = prune.tree(tree.carseats, best = 10)
  plot(prune.carseats10)
  text(prune.carseats10, pretty=0, cex=0.75)
  
  # MSE of the pruned tree with size=10:
  yhat.pruned10= predict(prune.carseats10, newdata = Carseats[-half_auto,])
  plot(yhat.pruned10, carseats.test, xlim = c(0, 16), ylim = c(0, 16))
  abline (0, 1)
  mean((yhat.pruned10 - carseats.test) ^2)
  # MSE:4.590036
  
  # The Test MSE of the pruned tree (size=10) is larger than the Test MSE of the original tree.
  # Therefore pruning the tree does not improve the test MSE.
  

# Subtask 4
  
  library(mlr) 
  lrn.carseats <- makeLearner("regr.rpart")
  traintask.carseats <- makeRegrTask(data = tr_auto, target = "Sales")
  set.seed(111)
  resample.res <- resample(lrn.carseats, traintask.carseats, resampling = cv10, measures = mse, show.info = FALSE)
  resample.res$measures.test
  
  min(resample.res$measures.test[,2])
  max(resample.res$measures.test[,2])
  
  # The code shows that the MSEs are not consistent at all. The values vary from around 3.76 (fifth iteration) to around 6.94 (first iteration).
  # This could be due to the low number of overall observations and the fact that we (randomly) use half of them (200) to train the tree model.
  # As in cross-validation, the results highly depend on how the random procedure splits the data set into test and training data (i.e. which observations fall into which data set).
  # Therefore we should not rely too much on our results from subtask 3 as it could only be due to chance that size=18 (and size=10) deliver the best Test MSE results.

# Subtask 5

  library(randomForest)
  
  set.seed(815)
  bag.carseats = randomForest(Sales~., data = Carseats, subset = half_auto, mtry= 10, importance = TRUE)
  bag.carseats
  
  set.seed(815)
  bag.carseats2 = randomForest(Sales~., data = Carseats, subset = half_auto, mtry= 10, importance = TRUE, ntree=1000)
  bag.carseats2
  
  # mtry=10 as all available predictors are used for each split of the tree (m=p). 
  # m=p indicates that we perform bagging.
  # ntree equals the number of trees built in the bagging procedure. 
  # The mean of squared residuals and the percentage of variance explained vary slightly when we change the number of trees built.
  
  yhat.bag= predict(bag.carseats, newdata = Carseats[-half_auto,])
  plot(yhat.bag, carseats.test, xlim = c(0, 16), ylim = c(0, 16))
  abline (0, 1)
  
  mean((yhat.bag - carseats.test) ^2)
  # The Test MSE after performing the bagging procedure and splitting the tree 500 times equals 2.718382.
  
  mean((yhat.pruned18 - carseats.test) ^2) #For comparison
  # The Test MSE of the previous 'pruned' tree was much larger.
  
  # Sidenote:
  # MSE with ntree=1000
  yhat.bag2= predict(bag.carseats2, newdata = Carseats[-half_auto,])
  plot(yhat.bag2, carseats.test, xlim = c(0, 16), ylim = c(0, 16))
  abline (0, 1)
  
  mean((yhat.bag2 - carseats.test) ^2)
  # The Test MSE is even better wenn we split the tree 1000 times: 2.713138.
  
  
  importance(bag.carseats)
  # %IncMSE stands for the mean decrease of accuracy in predictions on the out of bag samples when a given variable is excluded from the model.
  # IncNodePurity is a measure of the total decrease in node impurity that results from splits over that variable.
  # The node impurity for regression trees is measured by the training RSS.
  # The higher the measure, the more important the variable.
  
  # In this case, the three most important variables are:
  # ShelveLoc, Price and CompPrice according to both the %IncMSE and the IncNodePurity measure.
  
  # The importance of certain variables can also be seen in the following plots:
  varImpPlot(bag.carseats)



# Subtask 6

  # If we do not define mtry, it is automatically put to m=p/3 for regression trees and m=sqrt(p) for classification trees. 
  # In this case m is equal to p/3 as we build regression trees.
  # In our example, the default option uses 3 variables at each split (~10/3).
  # As we intend to determine the mtry resulting in the lowest MSE, we deploy the below loop:
  
  mse.rF = double(10)
  
  for(i in 1:10 ){
    set.seed(815)
    rf.carseats = randomForest(Sales ~ . , data = Carseats, subset = half_auto, mtry = i,  importance = TRUE)
    yhat.rf = predict(rf.carseats, newdata = Carseats[-half_auto,])
    mse.rF[i] = mean((yhat.rf - carseats.test)^2)
  }
  
  which.min(mse.rF) #6
  mse.rF[which.min(mse.rF)] #2.586396 
  
  # Testing all possible values for m [1;10], m=6 yields the lowest Test MSE.
  # Considering 6 predictors at each split, the Test MSE decreases slightly to 2.586396 and is now even lower than the Test MSE of the Bagging approach.
  # The optimal mtry is different to the default value for mtry (m=p/3). This could be dependent on the level of correlation between the predictors.
  
  set.seed(815)
  rf.carseats6 = randomForest(Sales~., data = Carseats, subset = half_auto, mtry=6, importance = TRUE)
  rf.carseats6
  
  importance(rf.carseats6)
  varImpPlot(rf.carseats6)
  
  # For the RF[6] (mtry=6), the trees' most important variables are ShelveLoc, Price and CompPrice (%IncMS) and Age (IncNodePurity).
  # Bagging claims ShelveLoc, Price and CompPrice the most important variables according to both measures.
  


# Subtask 7

  library(gbm)
  
  set.seed(815)
  boost.carseats = gbm(Sales~., data = Carseats[half_auto,], distribution= "gaussian")
  
  # distribution= "gaussian" as we want to fit a regression tree. Would be "bernoulli" if we wanted to fit a classification tree.
  # The default option for n.trees is 100, i.e. in total 100 trees are fitted.
  # The default option for interaction.depth is 1 implying an additive model that only involves one variable for each tree.
  # The default option for shrinkage is 0.1. Small values for the shrinkage parameter tend to require a large value for n.trees to achieve good performance.
  
  summary (boost.carseats)
  # If we only use the default values for n.trees, interaction.depth and shrinkage, the most important variables are ShelveLoc, Price and Age as in the RF example.
  
  plot(boost.carseats ,i="ShelveLoc")
  plot(boost.carseats ,i="Price")
  plot(boost.carseats ,i="Age")
  
  # The partial dependence plots indicate a negative impact of rising Price and Age on Sales as well as a better impact of Good ShelveLoc on Sales than for Medium ShelveLoc and Bad ShelveLoc.
  
  yhat.boost = predict(boost.carseats , newdata = Carseats[-half_auto,], n.trees =100)
  mean((yhat.boost - carseats.test)^2)
  # If we only use the default values for n.trees, interaction.depth and shrinkage, the Test MSE decreases sharply to 1.918701.
  # This means an improvement over the previous models.
  
  # The optimal values for n.trees, interaction.depth and shrinkage are hard to determine because of interdependencies. 
  
  # Different values for n.trees, interaction.depth and shrinkage:
  # Tested different combinations, best "data-mined" values for seed=815 provided below:
  
  set.seed(815)
  boost.carseats2 = gbm(Sales~., data = Carseats[half_auto,], distribution= "gaussian" , n.trees = 200 , interaction.depth = 1, shrinkage = 0.1)
  yhat.boost2 = predict(boost.carseats2 , newdata = Carseats[-half_auto,], n.trees =200)
  mean((yhat.boost2 - carseats.test)^2)
  # We reach a value of 1.672576 for the Test MSE which is much smaller than the RF Test MSE.
  # Problem here might be Data Mining/Hacking as we use our test data set to determine the optimal values used in our training data set.


###############################################
# Task 2
###############################################

load(url("https://web.stanford.edu/~hastie/ElemStatLearn/datasets/ESL.mixture.rda"))
# names(ESL.mixture)
# prob gives probabilites for each class when the true density functions are known
# px1 and px2 are coordinates in x1 (length 69) and x2 (length 99) where class probabilites are calculated

rm(x,y)
attach(ESL.mixture)
dat=data.frame(y=factor(y),x)
xgrid=expand.grid(X1=px1,X2=px2)
par(pty="s")
plot(xgrid, pch=20,cex=.2)
points(x,col=y+1,pch=20)
contour(px1,px2,matrix(prob,69,99),level=0.5,add=TRUE,col="blue",lwd=2) #optimal boundary


# Subtask 1

# Bayes classifier:
  # The Bayes classifier assigns a test observation with a certain predictor vector (e.g. "x0") to the class (e.g. "j") for which the conditional probability Pr(Y = j|X = x0) is largest.
  # If there are only two possible responses, the Bayes classifier assigns an observation to a certain class if its conditional prob. exceeds 50% (or to the other class if the conditional prob. is smaller than 50%).

# Bayes decision boundary:
  # In a scatter plot that contains both the observations and their corresponding class as well as the prediction area where -in the case of 2 classes- the Bayes classifier's cond. prob. is larger or smaller than 50%,
  # the Bayes decision boundary separates these prediction areas. Speaking of conditional prob. in a 2-class world, the cond. prob. of an observation belonging to class 1 (and 2) is exactly 50% where the decision boundary is.
  # If a class 1 observation is "on the wrong side of the Bayes decision boundary" (i.e. not in the right prediction area), it is wrongly assigned to a class.

# Bayes error rate:
  # The Bayes error rate represents the lowest possible test error rate produced by the Bayes classifier.
  # The Overall Bayes error rate is equal to 1-E(max Pr(Y = j|X)).
  # The Bayes error rate can take values from 0 to 1.
  # The larger the Bayes error rate is, the more observations are assigned to the wrong class.

  # In the scatter plot, we can see the Bayes decision boundary which is represented by a blue line which separates the class=red from the class=black predicton areas.
  # The prediction areas are determined by the Bayes classifier. Where the cond. prob. for an observation being class=red exceeds 50%, the Bayes classifier assigns this observation to the red class.
  # The Bayes error rate cannot be directly seen in the graph itself, but it is represented by red dots being in the black area and the other way round. The more dots are in the wrong prediction area, the larger the Bayes error rate is.
  # If the Bayes error rate was equal to 0, every observation would be correctly classified. Then there would be no black dot on the red side and vice versa.

#Subtask 2

  # In general, the Bayes decision boundary always produces the best possible classification for the underlying data set. 
  # Therefore, if we know the true Bayes decision boundary, we do not need the test set as we have already achieved the best possible classification.
  # However, if we want to compare the performance of the Bayes decision boundary against other methods or want to compute the Bayes error rate, we still need the test set.
  
  
  
  # Code:
  
  library(e1071)
  # support vector classifier
  svcfits=tune(svm,factor(y)~.,data=dat,scale=FALSE,kernel="linear",ranges=list(cost=c(1e-2,1e-1,1,5,10)))
  summary(svcfits)
  svcfit=svm(factor(y)~.,data=dat,scale=FALSE,kernel="linear",cost=0.01)
  
  # support vector machine with radial kernel
  set.seed(4268)
  svmfits=tune(svm,factor(y)~.,data=dat,scale=FALSE,kernel="radial",ranges=list(cost=c(1e-2,1e-1,1,5,10),gamma=c(0.01,1,5,10)))
  summary(svmfits)
  svmfit=svm(factor(y)~.,data=dat,scale=FALSE,kernel="radial",cost=1,gamma=5)
  
  # the same as in a - the Bayes boundary
  par(pty="s")
  plot(xgrid, pch=20,cex=.2)
  points(x,col=y+1,pch=20)
  contour(px1,px2,matrix(prob,69,99),level=0.5,add=TRUE,col="blue",lwd=2) #optimal boundary
  
  # decision boundaries from svc and svm added
  svcfunc=predict(svcfit,xgrid,decision.values=TRUE)
  svcfunc=attributes(svcfunc)$decision
  contour(px1,px2,matrix(svcfunc,69,99),level=0,add=TRUE,col="red") #svc boundary
  
  svmfunc=predict(svmfit,xgrid,decision.values=TRUE)
  svmfunc=attributes(svmfunc)$decision
  contour(px1,px2,matrix(svmfunc,69,99),level=0,add=TRUE,col="orange") #svm boundary
  

#Subtask 3

  # A support vector classifier represents a linear decision boundary which maximizes the distance between the decision boundary and the two different classes.
  # For unseparable data, the SVC includes classifier boundaries where the decision boundary (Hyperplane, red line) is placed in the middle.
  # In the area between the two classifier boundaries, false classifications are tolerated.
  
  # In contrast to the SVC, the SVM is used when linear boundaries (as in this example) fail.
  # The SVM allows for a feature expansion by including polynomials or -as an easier approach- kernels. 
  # A SVM can be built by combining a SVC and a nonlinear kernel.
  # In this example, a radial kernel is used to account for nonlinearities in the data (orange line).


#Subtask 4

  # Parameters for the SVC:
    # c, which is a regularization parameter / determines the number and severity of margin violations we tolerate.
    # Can also be interpreted as a budget.
    # The smaller c is, the smaller the area between the two classifier boundaries is.
    # The smaller c is, the more strongly the decision boundary reacts on an addition of new points in the classifier boundary area (=Overfitting).
    # In this example, the optimal c is determined by 10-fold cross-validation out of 5 possible values for c.
    # The optimal value for c is 0.01 in this example.
  
  # Parameters for the SVM:
    # c (see above).
    # Optimal value for c is again determined by 10-fold CV, in this example it is 1.
    # Gamma is a positive constant which influences the radial kernel used in the SVM.
    # Gamma determines the influence two points have on each other. The larger gamma, the smaller the influence.
    # The larger gamma gets the more "spiky" the hypersurface will be. The smaller gamma gets, the flatter the hypersurface will be.
    # Thus, a large gamma will give you low bias and high variance while a small gamma will give you higher bias and low variance.
    # So a large gamma could lead to overfitting.
    # In this example, gamma is also determined by 10-fold CV. Out of 4 possible values for gamma, 5 is chosen.


#Subtask 5

  # The error rate of the SVM is equal to 0.16 in this example.
  # As we do not have a concrete value for the error rate of the Bayes decision boundary, we can only compare the decision boundaries in the plotted graph themselves.
  # The SVM and the Bayes classifier seem to perform quite similar.
  # The SVM seems to do a better job in separating the two classes where they are really close (in the middle of the plot).
  # However, the SVM also generates an "enclave" at the top left which does not make sense. ALthough there is no observation inside it at the moment, it seems like "red area" at a first glance.
  # When there are no observations as in the down left corner, the SVM and the Bayes classifier would classify the observations differently.
  # The SVM might be a bit more "overfit" to the data.
 

###############################################
# Task 3
###############################################

# Subtask 1

  data("USArrests")
  
  states =row.names (USArrests)
  states
  list(USArrests)
  apply(USArrests, 2, mean)
  
  
  hc.complete = hclust(dist(USArrests), method = "complete")
  plot(hc.complete, main = "Complete Linkage", xlab ="", sub ="", cex =.9)


# Subtask 2

  cutree(hc.complete, 5)
  
  hc.out = hclust(dist(USArrests))
  hc.clusters = cutree (hc.out, 5)
  table (hc.clusters)
  
  # The smallest cluster is cluster 4 with 2 observations.
  
  plot(hc.out)
  abline (h=75 , col = "red")
  # At a height of 75, there are 5 distinct clusters.
  
  hc.out
  
  list(hc.clusters)
  # Cluster 4 contains the states of Florida and North Carolina.
  

#Subtask 3
  
  # First approach: 
  # Variables are scaled to standard deviation one, but not centered to mean zero.
  
  USArrests_sc = scale (USArrests, center = FALSE , scale =TRUE)
  # The variables are now scaled to have standard deviation one.
  hc.scaled = hclust (dist(USArrests_sc), method ="complete")
  plot(hc.scaled, main ="Hierarchical Clustering with
       Scaled Observations", cex=.9)
  abline (h=1.08 , col ="red")
  
  cutree(hc.scaled, 5)
  
  hc.clusters2 = cutree (hc.scaled, 5)
  table (hc.clusters2)
  
  # The first cluster is now the smallest cluster. It contains 6 observations.
  # Cluster 1 contains the following states: Alabama, Georgia, Louisiana, Mississippi, North Carolina and South Carolina.
  # The height of the dendrogram is different now as all variables are scaled to have a standard deviation of 1.
  # Now, at a height of 1.08, there are 5 distinct clusters.
  # Furthermore, now states are added much later in the process (=more height in the dendrogram) than before.
  # Not scaling could lead to undesired effects since we are essentially overweighting certain variables.
  # Scaling dissolves unit discrepancies.
  



  # Second (alternative) approach:
  # Variables are scaled to standard deviation one and centered to mean zero.
  
  USArrests_sc2 = scale (USArrests, center = TRUE , scale =TRUE)
  # The variables are now scaled to have standard deviation one and centered to mean zero.
  hc.scaled2 = hclust (dist(USArrests_sc2), method ="complete")
  plot(hc.scaled2, main ="Hierarchical Clustering with Scaled
       and Centered Observations", cex=.9)
  abline (h=3.15 , col ="red")
  
  cutree(hc.scaled2, 5)
  
  hc.clusters3 = cutree (hc.scaled2, 5)
  table (hc.clusters3)
  # The second cluster is now the smallest cluster. It contains one observation (Alaska).
  # Now, at a height of 3.15, there are 5 distinct clusters.
  # Further explications similar to explications under the first approach.
  


# Subtask 4

  # First approach: 
  # Variables are scaled to standard deviation one, but not centered to mean zero.
  
  plot(hc.scaled, main ="Hierarchical Clustering with
       Scaled Observations", cex=.9)
  abline (h=0.07927176 , col ="red")
  
  # Looking at the dendrogram with scaled observations, we can see that Iowa and New Hampshire are the most similar ones as they merge at the lowest height (see red line).
  
  distance1 = dist(USArrests_sc, method = "euclidian", upper = TRUE)
  distance1
  min(distance1)
  # The Euclidian distance between these states is 0.07927176.
  
  
  # Second (alternative) approach:
  # Variables are scaled to standard deviation one and centered to mean zero.
  
  plot(hc.scaled2, main ="Hierarchical Clustering with Scaled
       and Centered Observations", cex=.9)
  abline (h=0.2058539 , col ="red")
  
  # Looking at the dendrogram with scaled observations, we can see that Iowa and New Hampshire are the most similar ones as they merge at the lowest height (see red line).
  
  distance2 = dist(USArrests_sc2, method = "euclidian", upper = TRUE)
  distance2
  min(distance2)
  # The Euclidian distance between these states is 0.2058539.

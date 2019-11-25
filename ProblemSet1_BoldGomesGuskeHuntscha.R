#################################################
# Problem Set 1
# Philip Bold - 461075
# Johannes Huntscha - 466310
# Ricardo Gomes - 421962
# Phillip Guske - 465343
#################################################


# Loading required packages
  library(GLMsData)
  library(ggplot2)
  library(Hmisc)
  library(class)
  library(caret)
  library(MASS)
  library(colorspace)
  library(ISLR)
  library(boot)
  library(leaps)
  library(Metrics)
  library(dplyr)
  library(glmnet)


#################################################
#Task1: Multiple linear regression
#################################################

library(GLMsData)
data("lungcap")
lungcap$Htcm=lungcap$Ht*2.54
lung_model = lm(log(FEV) ~ Age + Htcm + Gender + Smoke, data=lungcap)
summary(lung_model)

summary(lungcap)
summary(lungcap$Htcm)

#1.1. Write down the equation for the fitted lung_model.
    
    # General equation: log(FEV)=Beta1+Beta2*Age+Beta3*Htcm+Delta1*Gender+Delta2*Smoke+epsilon
    # Beta for cardinal variables, delta for dummy variables
    # log(FEV)=-1.943998+0.023387*Age+0.016849*Htcm+0.029319*Gender-0.046067*Smoke

#1.2. Why is log(FEV) used as response instead of FEV? 
    
    # Initial simple plotting:
    plot(lungcap$Age, lungcap$FEV)
    plot(lungcap$Htcm, lungcap$FEV)
    plot(lungcap$Gender, lungcap$FEV)
    plot(lungcap$Smoke, lungcap$FEV)    

    plot(lungcap$Age, log(lungcap$FEV))
    plot(lungcap$Htcm, log(lungcap$FEV))
    plot(lungcap$Gender, log(lungcap$FEV))
    plot(lungcap$Smoke, log(lungcap$FEV))
    
    # Further scrutiny of Htcm and Age:
      # Plotting FEV against Htcm
      ggplot(lungcap, aes(lungcap$Htcm, lungcap$FEV))+
        geom_point()+
        geom_smooth(se = FALSE, col = "red", size = 0.5, method = "lm")+
        geom_smooth(se = FALSE, col = "blue", size = 0.5, method = "auto")
      # Plotting the logarithm of FEV against Htcm
      ggplot(lungcap, aes(lungcap$Htcm, log(lungcap$FEV)))+
        geom_point()+
        geom_smooth(se = FALSE, col = "red", size = 0.5, method = "lm")+
        geom_smooth(se = FALSE, col = "blue", size = 0.5, method = "auto")
    
      # Plotting FEV against Age
      ggplot(lungcap, aes(lungcap$Age, lungcap$FEV))+
        geom_point()+
        geom_smooth(se = FALSE, col = "red", size = 0.5, method = "lm")+
        geom_smooth(se = FALSE, col = "blue", size = 0.5, method = "auto")
      # Plotting the logarithm of FEV against Age
      ggplot(lungcap, aes(log(lungcap$Age), lungcap$FEV))+
        geom_point()+
        geom_smooth(se = FALSE, col = "red", size = 0.5, method = "lm")+
        geom_smooth(se = FALSE, col = "blue", size = 0.5, method = "auto")
      
      
    # Plotting FEV against Htcm reveals that there is a non-linear relationship: the marginal slope increases with each unit of Htcm.
    # As a non-linear relationship can't be modelled by a multiple linear regression model, log(FEV) is used as response instead of FEV.
    # Taking the log of FEV linearizes the relationship which can therefore be captured by a multiple linear regression model

#1.3. Explain what the following in the summary-outputs mean. 
    
    summary(lung_model)
    
    # Estimate:
      # Estimated regression coefficients which describe the influence of a certain IV on the DV.
      # The influence is given by the value of the Estimate. If all other variables are kept constant, the Estimate represents the change in the DV if the IV is increased by one unit.
      # In case of binary variables (Gender, Smoke), the estimate is the change in the DV if the IV changes compared to the reference level (e.g. Smoke=0/"No Smoker" is the reference level, therefore the influence of smoking (Smoke=0 -> Smoke=1) on log(FEV) is negative).
      # Age (continuous): If Age is increased by 1 unit (one year), log(FEV) increases by 0.023387 units, keeping all other variables constant.
      # Smoke (dummy): If a person smokes (Smoke=1), log(FEV) decreases by 0.046067 units, keeping all other variables constant.
      
    # Std.Error (SE):
      # The standard error is an estimate of the standard deviation of the coefficient, i.e. the amount it varies across observations. 
      # A coefficient's SE can be used to identify the range in which the true parameter of a regression coefficient should be with a certain probability.
      # E.g. regression coefficient estimate +- 2*SE covers the true parameter with a prob. of around 95%.
      # Age (continuous): The true impact of a 1 unit change in age on log(FEV) should be between (0.016691 <= x <= 0.030083) with a 95% probability
      # Therefore, we can say that with (at least) 95% certainty the true impact of Age on log(FEV) is positive
      # Smoke (dummy): The true impact of smoking (Smoke=1) on log(FEV) should be between (-0.087887 <= x <= -0.004247) with a 95% probability
      # Therefore, we can say that with (at least) 95% certainty the true impact of Smoke on log(FEV) is negative.
      
    # Residual standard error:
      # The Residual SE is the SE of the error term/residual.
      # It is the average amount that the response will deviate from the true regression line.
      # Calculation: RSE = sqrt(RSS/(n − 2))
      # In our example, the actual log(FEV) can deviate from the true regression line by approximately 0.1455, on average.
      # The smaller the Residual SE is, the better the regression line fits the data.
      # When the residual standard error is exactly 0 then the model fits the data perfectly (maybe problem of overfitting).
    
    # F-statistic:
      # Calculation: F = ((TSS − RSS)/p) / (RSS/(n − p − 1))
      # Used to test the null hypothesis that all of the regression coefficients are equal to 0 versus the alternative hypothesis that at least one regression coefficient is not equal to 0.
      # The larger the value for the F-statistic (i.e. the lower the corresponding p-value), the more likely it is that not all regression coefficients are equal to 0.
      # The F-statistic of the lung_model is really large (694.6) and the corresponding p-value (2.2e-16) really low. Hence, we reject H0.
      # Therefore we can say that with a very high probability, there is at least one regression coefficient (IV) that has a significant impact on log(FEV) (= is not equal to 0)

#1.4. What is the proportion of variability explained by the fitted lung_model?
    
    # The proportion of variability explained by the fitted lung_model is represented by Multiple R-Squared and Adjusted R-Squared.
    # Calculation: R² = 1-RSS/TSS, Adj. R² = 1− (RSS/(n − d − 1)) / (TSS/(n − 1)).
    # The larger the R² value, the larger the proportion of the variability in the response that can be explained by the regression.
    # In this example, 81.06% of the variability can be explained by the model, which is a pretty large proportion.
    # Furthermore, the Adj. R² takes into account the number of variables used to estimate the model. It favours simpler models and therefore punishes models with many estimated (insignificant) coefficients.
    # Value for the Adj. R²: 80.95%

#1.5. Consider a 14-year-old male. He is 175cm tall and not smoking. What is your best guess for his log(FEV)?
    
    # A prediction for log(FEV) can be constructed  by plugging in the values for the specific variables in the regression formula.
    # log(FEV)=-1.943998+0.023387*Age+0.016849*Htcm+0.029319*GenderM-0.046067*Smoke
    # log(FEV)=-1.943998+0.023387*14+0.016849*175+0.029319*"M"-0.046067*0
    
    data.boy = data.frame(Age=14, Htcm=175, Gender="M", Smoke=0)
    data.boy
    prediction = predict(lung_model, newdata = data.boy)
    prediction
    # The prediction for log(FEV) is 1.361271.

    # Computation of prediction interval:
    logfev.pi = predict(lung_model, newdata = data.boy, level = 0.95, interval = "prediction")
    logfev.pi
    # Transformation of prediction interval to FEV (for better interpretability):
    fev.pi = exp(logfev.pi)
    fev.pi

    # The prediction for the FEV is around 3.9 litres.
    # The prediction interval ranges from around 2.93 litres to around 5.20 litres, which is a rather wide interval.
    # Therefore the prediction interval is not very useful at a first glance.
    # To evaluate the usefulness thoroughly, we have to take a look at the distribution of FEV itself.
    
    if (!require("Hmisc")) install.packages("Hmisc")
    library(Hmisc)
    describe(lungcap$FEV)
    # Taking a look at the descriptive statistics of FEV, we see that the prediction (3.9 litres) is larger than the 90% percentile of the variable's distribution.
    # This relativizes our earlier statement and makes our (large) prediction interval a bit more useful as it seems to be in a tail region. 
    
#1.6 Redo the multiple linear regression, but add an interaction term between Age and Smoke
    # We include an interaction term in the regression model in order to accommodate non-additive relationships.
    # We can assess whether the effect of a change in an IV on the DV depends on the value of another IV.
    lung_model2 = lm(log(FEV) ~ Age + Htcm + Gender + Smoke + Age:Smoke, data=lungcap)
    summary(lung_model2)
    
    summary(lung_model2)
    summary(lung_model)
    # The interaction term captures the additional effect of one additional year of age on log(FEV) if a person smokes.
    # The interaction of age and smoke has a negative impact on log(LEV), namely a decrease of log(LEV) by -0.0116659 for every year a smoker (relative to a non-smoker) ages.
    # However, the interaction term is not statistically significant (not even at the 5%-level; p-value=0.16863). Therefore, the interpretation is not reliable. 
    # The interaction term seems to capture some of the effect that Smoke had on log(FEV). Therefore Smoke loses its significance (at the 5%- and the 10%-level).
    # The statistical significance of Age does not change.
    
    # The impact of Smoke on log(FEV) is now positive (although not being statistically significant).
    # This seems odd at first as we might think that smoking should generally have a negative impact on log(FEV).
    # Taking a closer look at the data and especially at both Smoke and Age:Smoke, we see that the total impact of smoking on log(FEV) becomes only negative from around 10 years of age onwards.
    # Therefore the difference for log(FEV) between smokers and non-smokers increases with the age of a person. 
    



#################################################
#Task 2: Classification (Logit/LDA/KNN)
#################################################

  setwd() #set working directory such that tennis dataset is in the same folder
  getwd()
  library(class)# needed for knn later on
  library(caret)# needed for confusion matrices
  library(ggplot2)
  t_raw = read.csv("tennisdata.csv")
  tn = na.omit(data.frame(y=as.factor(t_raw$Result),
                          x1=t_raw$ACE.1-t_raw$UFE.1-t_raw$DBF.1,
                          x2=t_raw$ACE.2-t_raw$UFE.2-t_raw$DBF.2))
  set.seed(4268) # for reproducibility
  tr = sample.int(nrow(tn),nrow(tn)/2)
  trte=rep(1,nrow(tn))
  trte[tr]=0
  tennis=data.frame(tn,"istest"=as.factor(trte))
  ggplot(data=tennis,aes(x=x1,y=x2,colour=y,group=istest,shape=istest))+
    geom_point()+
    theme_minimal()
  
  
#2.1 Get a general overview of the data frame M. How many observations are there?
    
    head(tennis,10)
    summary(tn)
    # There are a total of 392+395=787 observations
    
    # What are the median values of 1 and 2?
        median(tn$x1)
        median(tn$x2)
        #The median for x1 is -24, for x2 it is -24 as well.

    # Display the matrix of pairwise scatterplots and briefly comment on the relationship between y and x1 resp. y and x2.
        plot(tn)
        plot(tn$x1, tn$y)
        plot(tn$x2, tn$y)
        #y-axis is set between 1 and 2, not between 0 and 1. Therefore we will use the ggplot() function (see below).
        
        ggplot(data=tn, mapping = aes(x=tn$x1, y=tn$y)) +
          xlab("x1")+
          ylab("y")+
          geom_point(pch=20, col="blue")
        ggplot(data=tn, mapping = aes(x=tn$x2, y=tn$y)) +
          xlab("x2")+
          ylab("y")+
          geom_point(pch=20, col="blue")
        
        # The larger (smaller) the value for x1 is, the more (less) likely it is that player 1 wins (y=1).
        # The larger (smaller) the value for x2 is, the more (less) likely it is that player 2 wins (y=0).
        # Positive relationship between player quality and the probability to win.

        
#2.2 Perform a logistic regression with the full data set and print the numerical results.
    
    str(tn)
    tennis_model = glm(y ~ x1 + x2, data=tn, family = "binomial")
    summary(tennis_model)
    # Both coefficients for x1 and x2 are highly significant at the 0.1%-level (p-value <2e-16).
    
    # What is the predicted probability that player 1 wins the first match? 
        glm.probs = predict(tennis_model, newdata = tn, type = "response")
        glm.probs[1]
        #The predicted probability that player 1 wins is 38.29%.
        #Based on this, we would think that player 1 is more likely to lose.
      
    # What is the actual result?
        tn[1,]
        #The actual result is that player 1 loses (y=0).
        #Our prediction is right in this case.
    
      
#2.3 Compute the confusion matrix
    
    # Actual outcomes
        actual_tennis = tn$y
        summary(actual_tennis)
        
    # Predicted outcomes
        predicted_tennis = ifelse(tennis_model$fitted.values>0.5, 1, 0)
        summary(predicted_tennis)
    
    # Confusion matrix
        table(predicted_tennis, actual_tennis) 

        # n = 301+108+91+287 = 787
        # Sensitivity = True Pos. / (True Pos.+False Neg.)
        # Specificity = True Neg. / (True Neg.+False Pos.)
        
        # Assumption: 1) True Pos.: Player 1 is predicted to win & wins (both outcomes = 1)
        #             2) True Neg.: Player 1 is predicted to fail & fails (both outcomes = 0)
        
        # Sensitivity:
            287/(287+108)
            # Sensitivity = 0.7266
        
        # Specificity:
            301/(301+91)
            # Specificity = 0.7679
          
        # Null Error Rate: How often we would be wrong if we always predicted the majority class (in this case failure of player 1)?
            # Assumption: the majority class depends on the majority according to the prediction, not according to the actual results.
            (108+287)/787
            # Null error rate = 0.5019
        
        # Accuracy: correct predictions divided by total observations
            (301+287)/787
            # Accuracy = 0.7471
            # The model predicts around 74.71% of the matches correctly.
            # The accuracy is much higher than (1-null error rate) which means that our logistic regression model seems to be a pretty good classifier although it still fails to predict the correct outcome of some matches.

        
#2.4 Logistic regression with training data set
    
    # Define variable that contains training data
        train_data= tn[tr,] 
        summary(train_data)
    # Define variable that contains test data
        test_data=tn[-tr,]
        summary(test_data)
    
    # Regression and Prediction
        tennis_model_training = glm(y ~ x1 + x2, data=train_data, family="binomial")
        summary(tennis_model_training)
        
        predicted_tennis_test = ifelse(predict(tennis_model_training, newdata = test_data, type="response")>0.5, 1, 0)
        summary(predicted_tennis_test)
        
        actual_tennis_test = test_data$y
    
    # Confusion matrix 
        table(predicted_tennis_test, actual_tennis_test)
        
        # Misclassification rate:
            # Miscl. rate = (False Pos. + False Neg.) / n
            (50+42)/(149+42+50+153)
            # Miscl. rate = 0.2335
            # Miscl. is smaller than in the full sample (see subtask 3: (1-accuracy)).

        
#2.5 Perform a linear discriminant analysis (LDA) with only the training data set and compute the confusion matrix for the test data set. What is the misclassification error?
   
    library(MASS)
    tennis_model_LDA = lda(y ~ x1 + x2, data=train_data)
    tennis_model_LDA
    
    predicted_tennis_test_LDA = predict(tennis_model_LDA, newdata=test_data)
    summary(predicted_tennis_test_LDA$class)
    summary(actual_tennis_test)
    
    names(predicted_tennis_test_LDA)
    
    # Confusion matrix
        table(predicted_tennis_test_LDA$class, actual_tennis_test)
        
        # Misclassification rate:
            #Miscl. rate = (False Pos. + False Neg.) / n
            (51+42)/(148+42+51+153)
            #Miscl. rate = 0.2360
            #Miscl. rate is smaller than in subtask 3 but larger than in subtask 4.
        
        
#2.6 According to LDA, how many matches can I predict with a probability larger than 80%
    
    count_LDA = ifelse(predicted_tennis_test_LDA$posterior>0.8, 1, 0)
    count_LDA
    sum(count_LDA)
    #There are in total 106 matches in the test dataset that I can predict with a prob. larger than 80%.

    
#2.7
    
    ks = 1:30
    yhat = sapply(ks, function(k){
      class::knn(train=tn[tr,-1], cl=tn[tr,1], test=tn[,-1], k = k) #test both train and test
    })
    train.e = colMeans(tn[tr,1]!=yhat[tr,])
    test.e = colMeans(tn[-tr,1]!=yhat[-tr,])
    
    train.e #error rates training set
    test.e #error rates test set
    
    set.seed(0)
    ks = 1:30 # Choose K from 1 to 30.
    idx = createFolds(tn[tr,1], k=5) # Divide the training data into 5 folds.
    # "Sapply" is a more efficient for-loop.
    # We loop over each fold and each value in "ks"
    # and compute error rates for each combination.
    cv = sapply(ks, function(k){
      sapply(seq_along(idx), function(j) {
        yhat = class::knn(train=tn[tr[ -idx[[j]] ], -1],
                          cl=tn[tr[ -idx[[j]] ], 1],
                          test=tn[tr[ idx[[j]] ], -1], k = k)
        mean(tn[tr[ idx[[j]] ], 1] != yhat)
      })
    })
    
    # What is the meaning of the stored numbers
        cv
        # Meaning of the stored numbers:
        # The numbers correspond to the CV train error rates for each of the 5 folds (rows) and each of the 30 possible values for K (columns).
    
    # Compute the average cv.e and the standard error cv.se of the average CV error over all five folds.
        cv.e = colMeans(cv)
        cv.se = apply(cv, 2, sd)/sqrt(5)
        cv.e

    # Which K corresponds to the smallest CV error?  
        which.min(cv.e)
        min(cv.e)
        # The minimum CV train error rate corresponds to K=29. The CV train error rate is cv.e=0.2669912
        cv.se[29]
        mean(cv.se)
        # We have to keep in mind that the CV standard error (se) for K=29 is higher than the mean CV se for all Ks. This lowers the reliability of the corresponding CV error rate value (in K=29).


#2.8 Plot the misclassification errors using the code below. 
      
      library(colorspace)
      co = rainbow_hcl(3)
      par(mar=c(4,4,1,1)+.1, mgp = c(3, 1, 0))
      plot(ks, cv.e, type="o", pch = 16, ylim = c(0, 0.7), col = co[2],
           xlab = "Number of neighbors", ylab="Misclassification error")
      arrows(ks, cv.e-cv.se, ks, cv.e+cv.se, angle=90, length=.03, code=3, col=co[2])
      lines(ks, train.e, type="o", pch = 16, ylim = c(0.5, 0.7), col = co[3])
      lines(ks, test.e, type="o", pch = 16, ylim = c(0.5, 0.7), col = co[1])
      legend("topright", legend = c("Test", "5-fold CV", "Training"), lty = 1, col=co)
      
      # Bias-Variance Tradeoff
      # The larger K, the lower the flexibility of the model. As a consequence, bias increases & (training) variance decreases.
      # With very low K, the model is likely to be overfit. It performs very good with the training set, but also very badly with the test set.
      # This problem leads to a low bias but a very high variance for the test data (see K=1).
      # With higher K, there is more bias but therefore also less variance.

      
#2.9 Run the code below and briefly explain the graph that it creates.
      
      k = 30
      size = 100
      xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
      grid = expand.grid(xnew[,1], xnew[,2])
      grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
      np = 300
      par(mar=rep(2,4), mgp = c(1, 1, 0))
      contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
              xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
              main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
      points(grid, pch=".", cex=1, col=grid.yhat)
      points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
      legend("topleft", c("Player 1 wins", "Player 2 wins"),
             col=c("red", "black"), pch=1)
      box()
    
      # The graph plots the predicted areas of player 1 winning (red area) and player 2 winning (grey area) against the true outcomes of the matches between player 1 and player 2.
      # The true outcomes are represented by black (player 1 winning) and red dots (player 2 winning).
      # When there is a black dot in the red area (and vice versa), the KNN-prediction with K=30 misclassifies the observation.
      # As K=30, the 30 nearest observations are taken into account when deciding whether player 1 or player 2 should be predicted to win.
      # The black line is the decision boundary between player 1 winning (prediction) & player 2 winning (prediction), although there are some "enclaves", too.

      
#2.10 Run the code again, but choose k=1, k=50 and k=300 in the first line. Compare the emerging graphs.
      
      # k=1:
          k = 1
          size = 100
          xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
          grid = expand.grid(xnew[,1], xnew[,2])
          grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
          np = 300
          par(mar=rep(2,4), mgp = c(1, 1, 0))
          contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
                  xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
                  main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
          points(grid, pch=".", cex=1, col=grid.yhat)
          points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
          legend("topleft", c("Player 1 wins", "Player 2 wins"),
                 col=c("red", "black"), pch=1)
          box()
      
      # k=50:
          k = 50
          size = 100
          xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
          grid = expand.grid(xnew[,1], xnew[,2])
          grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
          np = 300
          par(mar=rep(2,4), mgp = c(1, 1, 0))
          contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
                  xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
                  main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
          points(grid, pch=".", cex=1, col=grid.yhat)
          points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
          legend("topleft", c("Player 1 wins", "Player 2 wins"),
                 col=c("red", "black"), pch=1)
          box()
      
      # k=300:
          k = 300
          size = 100
          xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
          grid = expand.grid(xnew[,1], xnew[,2])
          grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
          np = 300
          par(mar=rep(2,4), mgp = c(1, 1, 0))
          contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
                  xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
                  main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
          points(grid, pch=".", cex=1, col=grid.yhat)
          points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
          legend("topleft", c("Player 1 wins", "Player 2 wins"),
                 col=c("red", "black"), pch=1)
          box()
      
      # k=500:
          k = 500
          size = 100
          xnew = apply(tn[tr,-1], 2, function(X) seq(min(X), max(X), length.out=size))
          grid = expand.grid(xnew[,1], xnew[,2])
          grid.yhat = knn(tn[tr,-1], tn[tr,1], k=k, test=grid)
          np = 300
          par(mar=rep(2,4), mgp = c(1, 1, 0))
          contour(xnew[,1], xnew[,2], z = matrix(grid.yhat, size), levels=.5,
                  xlab=expression("x"[1]), ylab=expression("x"[2]), axes=FALSE,
                  main = paste0(k,"-nearest neighbors"), cex=1.2, labels="")
          points(grid, pch=".", cex=1, col=grid.yhat)
          points(tn[1:np,-1], col=factor(tn[1:np,1]), pch = 1, lwd = 1.5)
          legend("topleft", c("Player 1 wins", "Player 2 wins"),
                 col=c("red", "black"), pch=1)
          box()
      
      # The smaller k, the more flexible the model and the decision boundary. This means that there are much more "enclaves" and no clear boundary.
      # The problem is that k=1 might work well with the training data (bc of its flexibility) but not with the test data.
      # Small bias for small k but large variance (Bias-Variance Tradeoff & problem of overfitting).
      # The higher the k, the less flexible the model and the more "linear" the decision boundary.
      # For K=500, the problem is that there are less than 500 observations in the training data in total;
      # Therefore, the KNN classifier with K=500 automatically always predicts player 1 to win as he wins more matches in the training dataset.
      
      summary(tn[tr,]) #player 1 wins 200 of the 393 matches in the training dataset.

      
      
      
#################################################
#Task 3: Cross-Validation and Bootstrapping
#################################################

library(ISLR)
head(Carseats)

#3.1 Fit a multiple linear regression model lm.fit_cs that uses Price, Urban, and US to predict Sales. Print the results.
      
      lm.fit_cs = lm(Sales ~ Price + Urban + US, data = Carseats)
      lm.fit_cs
      summary(lm.fit_cs)
      
      contrasts(Carseats$US)
      contrasts(Carseats$Urban)
      # The intercept, price and US are highly significant at the 0.1% level.
      # Urban is not significant at all (p-value of 0.936), therefore the estimate should not be interpreted.
      # A higher Price influences sales negatively whereas an origin in the US (US=Yes) influences sales positively. The intercept is positive as well.

      
#3.2
      #3.2a Split the sample set into a training set and a validation set, each encompassing half of the data. Use the random seed 2019.
          
          set.seed(2019)
          half_auto = sample(nrow(Carseats), size=nrow(Carseats)/2, replace = FALSE )
          half_auto
          tr_auto = Carseats[half_auto,]
          te_auto = Carseats[-half_auto,]
          summary(tr_auto)
          summary(te_auto)
          #both contain 200 observations each

          
      #3.2b Fit a multiple linear regression model lm.fit_cstr using only the training observations. Briefly compare the results with lm.fit_cs
          
          lm.fit_cstr = lm(Sales ~ Price + Urban + US, data = tr_auto)
          summary(lm.fit_cstr)
          summary(lm.fit_cs)
          
          # Comparison to 3.1.
            # Intercept:
                # Remains significant at the 0.1% level (no change in p-value).
                # Estimate is only slightly higher than for the complete dataset. SE is higher.
            # Price:
                # Remains significant at the 0.1% level (slight change in p-value).
                # Estimate is slightly higher than for the complete dataset. SE is higher.
            # Urban:
                # Remains insignificant.
                # Estimate is much lower, SE higher and p-value much lower. 
                # But should still not be interpreted as not statistically significant.
            # US:
                # Is now not significant at the 0.1% level anymore, but at the 1% level (p-value increases).
                # Estimate is now slighty lower, SE higher than for the full dataset.
            # SEs are higher presumably because of the smaller number of observations in the training dataset (as dividing by the sqrt of n (formula for SE) suggests)
  
          
      #3.2c Predict the response for all 400 observations and calculate the mean squared error (MSE) of the observations in the validation set.
          
          library(Metrics)
          
          set.seed(2019)
          pred.=predict(lm.fit_cs,Carseats)
          pred.
          # Prediction for all 400 observations using the full model (lm.fit_cs).
          
          pred2=predict(lm.fit_cstr, Carseats)
          pred2
          # Prediction for all 400 observations using the training model (lm.fit_cstr).
          
          mean((Carseats$Sales-pred2)[-half_auto]^2)
          mse(Carseats$Sales[-half_auto], pred2[-half_auto])
          # Calculation of the MSE for the validation set, using the training model (lm.fit_cstr).
          # MSE=6.030194
  
          
      #3.2d How does your answer to c. change if you use the random seeds 2018 or 2020 instead of 2019 to split the data set?
          
          # set.seed(2018):
            set.seed(2018)
            attach(Carseats)
            half_auto2 = sample(nrow(Carseats), size=nrow(Carseats)/2, replace = FALSE )
            lm.fit_cstr_2018= lm(Sales ~ Price + Urban + US, subset = half_auto2)
            pred2=predict(lm.fit_cstr_2018, Carseats )
            pred2
            # Prediction for all 400 observations using the training model (lm.fit_cstr_2018).
            
            mean((Carseats$Sales-pred2)[-half_auto2]^2)
            mse(Carseats$Sales[-half_auto2], pred2[-half_auto2])
            # Calculation of the MSE for the validation set, using the training model (lm.fit_cstr_2018).
            # MSE=5.88007
            # Setting a different random seed results in a lower MSE, i.e. a lower average of the squares of errors.
          
          # set.seed(2020):
            set.seed(2020)
            attach(Carseats)
            half_auto3 = sample(nrow(Carseats), size=nrow(Carseats)/2, replace = FALSE )
            lm.fit_cstr_2020 = lm(Sales ~ Price + Urban + US, subset = half_auto3)
            pred3=predict(lm.fit_cstr_2020, Carseats)
            pred3
            # Prediction for all 400 observations using the training model (lm.fit_cstr_2020).
            
            mean((Carseats$Sales-pred3)[-half_auto3]^2)
            mse(Carseats$Sales[-half_auto3], pred3[-half_auto3])
            # Calculation of the MSE for the validation set, using the training model (lm.fit_cstr).
            # MSE=6.042666
          
          
      #3.2e Compute the LOOCV estimate for the MSE using the cv.glm function.
          library(boot)
          glm.fit_cs = glm(Sales ~ Price + Urban + US, data = Carseats)
          loocv.err = cv.glm(Carseats, glm.fit_cs)
          names(loocv.err)
          
          loocv.err$delta
          # The LOOCV Estimate for the MSE lies between 6.168444 and 6.168591.

          
#3.3 Use the regsubsets function to find the best subset consisting of three variables to predict Sales, excluding ShelveLoc. 
      
      library(leaps)
      best_subset = regsubsets(Sales ~ . -ShelveLoc, data = Carseats, nvmax = 3)
      
      summary(best_subset)
      # The best subset includes the variables "Price", "CompPrice" and "Advertising"-
      # Among all possible three predictor model combinations (nvmax=3), they have the largest predictive power over Sales/ explain the largest part of its variance/highest R?
      
      glm.fit_cs_best_subset = glm(Sales ~ Price + CompPrice + Advertising, data = Carseats)
      loocv.err2 = cv.glm(Carseats, glm.fit_cs_best_subset)
      loocv.err2$delta
      # The LOOCV Estimate for the MSE is around 4.411 (lower than the model's MSE in task 3.2). This confirms that the 3 chosen predictors have higher predictive power over sales (than the ones in the previous task).

      
#3.4. 
      
      # Compute the mean of the Sales variable. 
          mean_Sales = mean(Carseats$Sales)
          # The mean of Sales is 7.496325
          
      # Create 20 bootstrapped replicates of the data, using random seed 1. Then apply the mean function to each bootstrapped replicate. Enter the command 20 times and write down the range of the mean, i.e. lowest and the largest number.
          set.seed(1)
          attach(Carseats)
          Boot_1 = sample(Sales, 400, replace=T)
          Boot_2 = sample(Sales, 400, replace=T)
          Boot_3 = sample(Sales, 400, replace=T)
          Boot_4 = sample(Sales, 400, replace=T)
          Boot_5 = sample(Sales, 400, replace=T)
          Boot_6 = sample(Sales, 400, replace=T)
          Boot_7 = sample(Sales, 400, replace=T)
          Boot_8 = sample(Sales, 400, replace=T)
          Boot_9 = sample(Sales, 400, replace=T)
          Boot_10 = sample(Sales, 400, replace=T)
          Boot_11 = sample(Sales, 400, replace=T)
          Boot_12 = sample(Sales, 400, replace=T)
          Boot_13 = sample(Sales, 400, replace=T)
          Boot_14 = sample(Sales, 400, replace=T)
          Boot_15 = sample(Sales, 400, replace=T)
          Boot_16 = sample(Sales, 400, replace=T)
          Boot_17 = sample(Sales, 400, replace=T)
          Boot_18 = sample(Sales, 400, replace=T)
          Boot_19 = sample(Sales, 400, replace=T)
          Boot_20 = sample(Sales, 400, replace=T)
          
          Boot_mean=(c (mean(Boot_1), mean(Boot_2), mean(Boot_3), mean(Boot_4), mean(Boot_5), mean(Boot_6), mean(Boot_7), mean(Boot_8), mean(Boot_9), mean(Boot_10), mean(Boot_11), mean(Boot_12), mean(Boot_13), mean(Boot_14), mean(Boot_15), mean(Boot_16), mean(Boot_17), mean(Boot_18), mean(Boot_19), mean(Boot_20)))
          min(Boot_mean) #lowest mean = 7.1183
          max(Boot_mean) #largest mean = 7.7362
          
          # Alternative/easier approach:
              library(boot)
              set.seed(1)
              mean_sales_boot = replicate(20, mean(sample(Carseats$Sales, size = 400, replace = TRUE)))
              min(mean_sales_boot)
              max(mean_sales_boot)

          # The range of the population mean for both approaches is [7.1183; 7.7362].
     
               
#3.5 Repeat the above analysis by using boot().
    
    # boot function
          set.seed(1)
          boot_function = function(Carseats, Sales) {
            return(mean(Carseats[Sales]))}
          
          mean_sales_boot2 = boot(Carseats$Sales, boot_function, R=20)
          mean_sales_boot2
          boot.ci(mean_sales_boot2, conf = 0.95, type = "norm")
          #With 95% probability, the true parameter of Sales lies within the range of 7.341 and 7.797 (Assumption: Normal distribution).

      
      
      
#################################################
#Task 4: Linear Model Selection and Regularization
#################################################

library(ggplot2)
set.seed(1007) # for reproducibility
sm = sample.int(nrow(diamonds),nrow(diamonds)/10)
smbi=rep(1,nrow(diamonds))
smbi[sm]=0
diamonds2=data.frame(diamonds,"issmall"=as.factor(smbi))
small = (diamonds2$issmall==0)
diamondssm=diamonds2[small,]

#4.1 Get an overview of the diamondssm data.
    
    # What are three highest prices in the dataset? How many carats do those diamonds weigh?
        summary(diamondssm)
        library(dplyr)
        top_n(diamondssm, 3, price)
        # The highest 3 prices are 18795 (2.00 carat), 18779 (2.06 carat) and 18759 (2.00 carat)
    
    # What is the mean weight?
        mean(diamondssm$carat)
        # The mean weight is 0.8002299 carat.
    
    # Which colour is the most prevalent? 
        summary(diamondssm)
        #color=G is most prevalent with 1101 appearances.
    
    # Plot price against carat as well as their logged forms against each other.
        plot(diamondssm$carat, diamondssm$price) #there seems to be a non-linear relationship
        plot(log(diamondssm$carat), log(diamondssm$price)) #relationship is linearized
        # We should replace price and carat by logprice and logcarat for following regressions.
    
    diamondssm=data.frame(diamondssm,"logprice"=log(diamondssm$price), "logcarat"=log(diamondssm$carat))
    diamonds3 <- subset(diamondssm, select = -c(price, carat, issmall))
    
 
#4.2 Perform forward and backward stepwise selection to choose the best subset of predictors for logprice, using the diamonds3 data set. Compare the two results. Using adjusted R? as criterion, how large is the best subset from the backward stepwise selection?
    
    head(diamonds3)
    library(leaps)
    
    # Forward stepwise selection:
      regfit.fwd = regsubsets (diamonds3$logprice~., data=diamonds3, nvmax=23, method ="forward")
      reg_fwd_summary = summary (regfit.fwd)
      names(reg_fwd_summary)
      
      reg_fwd_summary$adjr2 # gives us the Adj. R?s of all models built by the forward stepwise selection.
      plot(reg_fwd_summary$adjr2, xlab="No. of Variables", ylab="Adj. R?", type = "b") # plots the Adj. R?s.
      
      which.max(reg_fwd_summary$adjr2)
      # The model containing 17 variables has the highest Adj. R2?.
      
      reg_fwd_summary$adjr2[17]
      # Maximum Adj. R?=0.9829226
      
      reg_fwd_summary$which #Shows us the variables that each model built by the forward stepwise selection contains.
    
    # Backward stepwise selection:
      regfit.bwd = regsubsets (diamonds3$logprice~., data=diamonds3, nvmax=23, method ="backward")
      reg_bwd_summary = summary (regfit.bwd)
      names(reg_bwd_summary)
      
      reg_bwd_summary$adjr2 # gives us the Adj. R?s of all models built by the backward stepwise selection.
      plot(reg_bwd_summary$adjr2, xlab="No. of Variables", ylab="Adj. R?", type = "b") # plots the Adj. R?s.
      
      which.max(reg_bwd_summary$adjr2)
      # The model containing 17 variables has the highest Adj. R2?.
      
      reg_bwd_summary$adjr2[17]
      # Maximum Adj. R?=0.9829226
      
      reg_bwd_summary$which # Shows us the variables that each model built by the backward stepwise selection contains.
    
    # There is a total of 23 variables
    # The forward selection picks logcarat as first variable.  Then it follows: clarity.L,color.L,clarity.Q,cut.L, color.Q, clarity.c, clarity^4, clarity^7, clarity^5, cut.Q, x, color.C, color^4, cut.C, depth, z, y,     table, cut^4, color^5, color^6, clarity^6
    # The backward selection picks logcarat as first variable. Then it follows: clarity.L,color.L,clarity.Q,cut.L, color.Q, clarity.c, clarity^4, clarity^7, clarity^5, cut.Q, x, color.C, color^4, cut.C, depth, z, cut^4, table, y,     color^5, color^6, clarity^6
    # The 18th and 20th iterations result in different variables (y and cut^4 for the forward selection vs. cut^4 and y (swapped) for the backward selection)

    # Measured by Adj. R2, both the forward and the backward stepwise selection suggest a model containing 17 variables.
    # Both -best- models contain the following variables: logcarat, clarity.L,color.L,clarity.Q,cut.L, color.Q, clarity.c, clarity^4, clarity^7, clarity^5, cut.Q, x, color.C, color^4, cut.C, depth, z.
    # For both Models: Adj. R2=0.9829226 (as they select identical variables up to the 17th iteration).

      
#4.3 What are the main differences between using adjusted R2 for model selection and using cross-validation (with mean squared test error MSE)?
   
    # The Adj. R2 is an indirect approach to estimate the test error. We make an adjustment to the training error (depending on the model size) to account for the bias due to overfitting.
    # The Adj. R2 statistic pays a price for the inclusion of unnecessary variables in the model. 
    
    # On the other hand, cross-validation estimates the test error directly.
    
    # Comparing the CV to the Adj. R2:
    # The CV does not require an estimate for the error variance sigma. Therefore it can also be used for datasets that contain more predictors than observations (p>n), whereas the Adj. R? cannot be computed in this case (limited degrees of freedom).
    # Furthermore, it can also be used when the df of the model or its error variance sigma cannot be estimated.

      
#4.4. Use lasso and ridge regression on the data to predict the logprice. Briefly describe the results and compare them to a simple multiple linear regression.
    
      library(glmnet)
    
    # We need to define x and y in the following way to be able to use the cv.glmnet() function later on:
        x = model.matrix(diamonds3$logprice~., diamonds3)[,-1]
        head(x)
        y = diamonds3$logprice
        
    # Split the sample into a train and a test set:
        set.seed (1007) #seed as given in the task
        train = sample (1: nrow(x), nrow (x)/2)
        test = (-train)
        y.test = y[test]
        
    ####Ridge####
        # We use CV to choose the optimal value for lambda:
            # Build the baseline Ridge model (later needed for prediction):
                ridge.mod = glmnet(x[train,],y[train], alpha =0)
                plot(ridge.mod)
                
            # CV Ridge Regression with 10 folds:
                cv.out_ridge =cv.glmnet(x[train,],y[train], alpha =0)
                plot(cv.out_ridge)
                
            # Identify the value of lambda that results in the smallest crossvalidation error:
                bestlam_ridge =cv.out_ridge$lambda.min
                bestlam_ridge
                #The best value for lambda is 0.09840529
        
        # Predicting the values for the test set using the baseline ridge model and the optimal lambda:
            ridge.pred= predict(ridge.mod, s= bestlam_ridge, newx=x[test,])
            mse(y.test, ridge.pred)
            # The MSE of the Ridge model using the optimal value for lambda is 0.03114731.
        
        # Refit Ridge regression model on the full data set:
            out_ridge = glmnet(x,y,alpha =0)
        
        # Predict the values using the optimal value for lambda chosen by CV; showing the coefficient estimates
            ridge.coef = predict(out_ridge, type ="coefficients", s= bestlam_ridge)[1:24,]
            ridge.coef
        
        # The larger the tuning parameter lambda is, the smaller the predictor coefficients are (i.e. higher penalty for large predictor coefficients).
        # Ridge regression therefore is a shrinkage method.
        # In this case the optimal tuning parameter is lambda=0.09840529.
        # Ridge: MSE=0.03114731
        
        # Lasso regression does -in contrast to ridge regression- have the possibility to eliminate predictors from the model.
        # This is due to another penalty function (l1 vs. l2 norm).
        # The lasso therefore is not only a shrinkage method, but also performs variable selection just like forward/backward stepwise selection does.
        
    
    ####Lasso####
        # We use CV to choose the optimal value for lambda:
            # Build the baseline Lasso model (later needed for prediction):
                lasso.mod = glmnet(x[train,],y[train], alpha =1)
                plot(lasso.mod)
                # Baseline Lasso model already looks different to the baseline ridge model as there are variables exluded from the model.
            
            # CV Lasso Regression with 10 folds:
                cv.out_lasso =cv.glmnet(x[train,],y[train ], alpha =1)
                plot(cv.out_lasso)
                
            # Identify the value of lambda that results in the smallest crossvalidation error:
                bestlam_lasso =cv.out_lasso$lambda.min
                bestlam_lasso
                # The best value for lambda is 0.000917731.
        
        # Predicting the values for the test set using the baseline ridge model and the optimal lambda:
            lasso.pred= predict (lasso.mod ,s= bestlam_lasso ,newx =x[test ,])
            mse(y.test, lasso.pred)
            # The MSE of the Lasso model using the optimal value for lambda is 0.01859911.
        
        # Refit Lasso regression model on the full data set:
            out_lasso = glmnet (x,y,alpha =1)
        
        # Predict the values using the optimal value for lambda chosen by CV; showing the coefficient estimates
            lasso.coef = predict (out_lasso, type ="coefficients",s= bestlam_lasso)[1:24,]
            lasso.coef
        
        # Some results are shrunk to 0
        # This results in a sparser model than based on the ridge regression.
        # The Lasso model contains 17 variables. Remind: Fwd & bwd stepwise selection models also contained 17 variables.
        # In this case the optimal tuning parameter is lambda=0.000917731.
        # Lasso: MSE= 0.01859911.
        
        # For this dataset, Lasso regression performs better than Ridge regression. We expect some variables to be unrelated to the response (i.e. not important considering the explanatory power of the model).
        
    
    ###Linear model###
        train1 = diamonds3[train,]  # generating training data set for linear regression
        linear_model = lm(train1$logprice ~ ., train1)
        summary(linear_model)
        # The linear model still contains all 23 variables (as expected).
        # The coefficients of the insignificant variables are larger than in the ridge and lasso regressions. 
        
        head(diamonds3[-train])
        test1 = diamonds3[test,]#generating test data set for linear regression
        linear.pred = predict(linear_model, newdata=test1, type="response")
        mse(y.test, linear.pred)
        # Linear model: MSE=0.01844196
        # The MSE for the linear model is smaller than the one for the LASSO and ridge regression, i.e. the linear model has a higher predictive power than the models resulting from LASSO or ridge. 
        # This is possibly due to the fact that our number of n is much bigger than p. Since this is the case, there can be a lot of variability in the fit which can result in overfitting and very poor predictive ability of Lasso and Ridge regression. 
        




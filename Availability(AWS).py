# -*- coding: utf-8 -*-
"""
Spyder Editor

This is a temporary script file.

"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import statsmodels.formula.api as smf
#from sklearn.model_selection import train_test_split
from sklearn.cross_validation import train_test_split

def MAPE(actual, predicted):
    percent_error = (actual - predicted)*100/actual
    percent_error = percent_error[np.isfinite(percent_error)]
    abs_percent_error = abs(percent_error)
    mean_ape = np.mean(abs_percent_error)
    med_ape = np.median(abs_percent_error)
    err = pd.Series([mean_ape,med_ape],
                    index = ["Mean APE", "Median APE"])
    return(err)
availability = pd.read_csv("C:\\Users\\user\\Desktop\\greens\\Availability.csv")

## Step 1:
# DV: Availability
# IDVs: Bid price, Spot Price (ordinal categorical variable)

## Step 2
plt.scatter(availability["Bid"], availability["Availability"])
##also: sns.relplot(x = 'Bid', y = 'Availability', data = availability)

sns.scatterplot("Bid","Availability", data = availability, size = 'Spotprice')

availability["Bid"].corr(availability["Availability"]) # 0.62; decent positive correlation
availability["Spotprice"].corr(availability["Availability"])  #-0.17; weak negative correlation

## Step 3, 4
avail_train, avail_test = train_test_split(availability, test_size = 0.3,
                                           random_state = 1234)
""-------------------------------------""
## Simple Linear
avail_simplelin_model = smf.ols("Availability ~ Bid", data=avail_train).fit()
avail_simplelin_model.summary()
# Rsquared: 0.406  --> Not so great
# p value is close to 0 indicating the variable is statistically significant
avail_fitted_simplelin = avail_simplelin_model.predict(avail_train)
plt.scatter(avail_train["Bid"], avail_train["Availability"])
plt.scatter(avail_train["Bid"], avail_fitted_simplelin, c = "red")

pred_avail_simplelin = avail_simplelin_model.predict(avail_test)
MAPE(avail_test["Availability"],pred_avail_simplelin)
#Mean APE      341.634558%
#Median APE     26.311299%
""-------------------------------------""
## Simple Non Linear
availability["Bid_sq"]=availability["Bid"]**2

## Step 3, 4
avail_train_sq, avail_test_sq = train_test_split(availability, test_size = 0.3,
                                           random_state = 1234)

## Simple Non-Linear
avail_Nonlin_model_sq = smf.ols("Availability ~ Bid+Bid_sq", data=avail_train_sq).fit()
avail_Nonlin_model_sq.summary()
##R-squared:0.632
# p value is close to 0 indicating the variable is statistically significant

avail_fitted_Nonlin=avail_Nonlin_model_sq.predict(avail_train_sq)

plt.scatter(avail_train["Bid"], avail_train["Availability"])
plt.scatter(avail_train["Bid"], avail_fitted_simplelin, c = "red")
plt.scatter(avail_train["Bid"], avail_fitted_Nonlin, c = "red")

pred_avail_Nonlin = avail_Nonlin_model_sq.predict(avail_test_sq)
MAPE(avail_test_sq["Availability"],pred_avail_Nonlin)
##Mean APE      157.721798--Better than previous but still worst
#Median APE     19.165843
""-------------------------------------""
## Multiple Linear/ Non Linear
## Multiple Linear
    
availability.corr()
availability["Bid"].corr(availability["Availability"])

avail_train, avail_test = train_test_split(availability, test_size = 0.3,
                                           random_state = 1234)

avail_Multi_linear = smf.ols("Availability ~ Bid+Spotprice", data=avail_train).fit()
avail_Multi_linear.summary() ##0.622

avail_fitted_Multilin=avail_Multi_linear.predict(avail_train)

plt.scatter(avail_train["Bid"], avail_train["Availability"])
plt.scatter(avail_train["Bid"], avail_fitted_Multilin, c = "red")

pred_avail_Multilin = avail_Multi_linear.predict(avail_test)
MAPE(avail_test["Availability"],pred_avail_Multilin)
##Mean APE      243.422291
##Median APE     26.912156
""-------------------------------------""
## Multiple Non-Linear
availability.corr()##Bid and Bid_sq have strong positive correlation??

avail_train, avail_test = train_test_split(availability, test_size = 0.3,
                                           random_state = 1234)

avail_Multi_Nonlinear = smf.ols("Availability ~ Bid+Spotprice+Bid_sq", data=avail_train).fit()
avail_Multi_Nonlinear.summary() ##0.791-->good adj R compared to other models.

avail_fitted_MultiNonlin=avail_Multi_Nonlinear.predict(avail_train)

plt.scatter(avail_train["Bid"], avail_train["Availability"])
plt.scatter(avail_train["Bid"], avail_fitted_Multilin, c = "red")
plt.scatter(avail_train["Bid"], avail_fitted_MultiNonlin, c = "red")

red_avail_MultiNonlin=avail_Multi_Nonlinear.predict(avail_train)
MAPE(avail_train["Availability"],red_avail_MultiNonlin)
#Mean APE      348.974342
#Median APE     17.252421

pred_avail_MultiNonlin = avail_Multi_Nonlinear.predict(avail_test)
MAPE(avail_test["Availability"],pred_avail_MultiNonlin)
##Mean APE      184.806567
##Median APE     16.797305

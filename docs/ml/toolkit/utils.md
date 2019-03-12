---
author: Conor McCarthy
date: October 2018
keywords: confusion matrix, machine-learning, ml, utilities, interpolation, filling, statistics, kdb+, q
---

# <i class="fa fa-share-alt"></i> Utilities 

The utility functions cover procedures and analyses common to many machine-learning applications.

<i class="fa fa-github"></i>
[KxSystems/ml](https://github.com/kxsystems/ml/)

## Loading

Load the utilities library using:

```q
q)\l ml/ml.q
q).ml.loadfile`:util/init.q
```

## Functions

Functions are divided into two namespaces:

-    `.ml`: statistical analysis and vector creation
-    `.ml.util`: manipulation and transformation of data

These namespaces will be extended over time to wider functionality.

Function                                      | Output
:---------------------------------------------|:-------------------------------------------------
[`.ml.accuracy`](utils.md#mlaccuracy)         | Accuracy of classification results 
[`.ml.arange`](utils.md#mlarange)             | Evenly-spaced values within a range 
[`.ml.confdict`](utils.md#mlconfdict)         | True/false positives and negatives as dictionary
[`.ml.confmat`](utils.md#mlconfmat)           | Confusion matrix 
[`.ml.corrmat`](utils.md#mlcorrmat)           | Correlation of a table
[`.ml.crm`](utils.md#mlcrm)                   | Correlation of a matrix
[`.ml.crossentropy`](utils.md#mlcrossentropy) | Categorical cross entropy 
[`.ml.cvm`](utils.md#mlcvm)                   | Covariance of matrix
[`.ml.describe`](utils.md#mldescribe)         | Descriptive information about a table
[`.ml.eye`](utils.md#mleye)                   | Identity matrix
[`.ml.f1score`](utils.md#f1score)	      | F1-score for classification results
[`.ml.fbscore`](utils.md#fbscore)             | Fbeta-score for classfication results
[`.ml.kfshuff`](utils.md#mlkfshuff)	      | Non repeating indices from target in K-Folds
[`.ml.kfxval`](utils.md#mlkfxval)	      | K-Fold cross validation
[`.ml.linspace`](utils.md#mllinspace)         | List of evenly-spaced values 
[`.ml.logloss`](utils.md#mllogloss)           | Logarithmic loss
[`.ml.mae`](utils.md#mlmae)                   | Mean absolute error 
[`.ml.mape`](utils.md#mlmase)                 | Mean absolute percentage error
[`.ml.matcorr`](utils.md#mlmatcorr)           | Matthews-correlation coefficient
[`.ml.mse`](utils.md#mlmse)                   | Mean squared error 
[`.ml.percentile`](utils.md#mlpercentile)     | Percentile calculation for an array
[`.ml.precision`](utils.md#mlprecision)       | Precision of a binary classifier 
[`.ml.r2score`](utils.md#mlr2score)	      | R2-score for regression model validation
[`.ml.range`](utils.md#mlrange)               | Range of values 
[`.ml.roc`](utils.md#mlroc)                   | X- and Y-axis values for a ROC curve 
[`.ml.rocaucscore`](utils.md#mlrocaucscore)   | Area under a ROC curve 
[`.ml.rmse`](utils.md#mlrmse)		      | Root mean squared error for a regressor
[`.ml.rmsle`](utils.md#mlrmsle)               | Root mean squared log error 
[`.ml.sensitivity`](utils.md#mlsensitivity)   | Sensitivity of a binary classifier 
[`.ml.shape`](utils.md#mlshape)               | Shape of a matrix 
[`.ml.smape`](utils.md#mlsmape)               | Symmetric mean absolute percentage error
[`.ml.specificity`](utils.md#mlspecificity)   | Specificity of a binary classifier 
[`.ml.sse`](utils.md#mlsse)                   | Sum squared error 
[`.ml.tscore`](utils.md#mltscore)             | One-sample t-test score 
[`.ml.tscoreeq`](utils.md#mltscoreeq)         | T-test for independent samples with equal variances and equal sample size


Function                                                           | Output
:------------------------------------------------------------------|:-------------------------------------------------
[`.ml.util.classreport`](utils.md#mlclassreport)		   | Statistical information about classification results
[`.ml.util.combs`](utils.md#mlcombs)                               | Unique combinations of a vector or matrix
[`.ml.util.df2tab`](utils.md#mlutildf2tab)                         | q table from a Pandas dataframe 
[`.ml.util.dropconstant`](utils.md#mlutildropconstant)             | Columns with zero variance removed
[`.ml.util.filltab`](utils.md#mlutilfilltab)		           | Tailored filling of null values for a simple matrix
[`.ml.util.freqencode`](util.md#mlutilsfreqencode)		   | Encoded frequency of individual category occurences
[`.ml.util.infreplace`](util.md#mlutilsinfreplace)		   | +/- infinities replaced with data min/max
[`.ml.util.lexiencode`](utils.md#mlutilslexiencode)		   | Categorical labels of features based on their lexigraphical order
[`.ml.util.minmaxscaler`](utils.md#mlutilminmaxscaler)             | Data scaled between 0-1 
[`.ml.util.nullencode`](utils.md#mlutilnullencode)		   | Nulls filled with avg/med/min/max of the column
[`.ml.util.onehot`](utils.md#mlutilonehot)                         | One-hot encoding 
[`.ml.util.polytab`](utils.md#mlutilpolytab)                       | Polynomial features of degree n from a table 
[`.ml.util.stdscaler`](utils.md#mlutilstdscaler)                   | Standard scaler transform-based representation of a table 
[`.ml.util.tab2df`](utils.md#mlutiltab2df)                         | Pandas dataframe from a q table 
[`.ml.util.timespantrainsform`](utils.md#mlutiltimespantransform)  | Timespan converted into its constituent parts
[`.ml.util.traintestsplit`](utils.md#mlutiltraintestsplit)         | Training and testing sets 
[`.ml.util.traintestsplitseed`](utils.md#mlutiltraintestsplitseed) | Training and testing sets with a seed 


## `.ml.accuracy`

_Accuracy of classification results_

Syntax: `.ml.accuracy[x;y]`

Where 

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the accuracy of predictions made.

```q
q).ml.accuracy[1000?0b;1000?0b] / binary classifier
0.482
q).ml.accuracy[1000?10;1000?10] / non-binary classifier
0.108
```


## `.ml.arange`

_Evenly-spaced values within a range_

Syntax: `.ml.arange[x;y;z]`

Where

-   `x`, `y`, and `z` are numeric atoms

returns a vector of evenly-spaced values between `x` (inclusive) and `y` (non-inclusive) in steps of length `z`.

```q
q).ml.arange[1;10;1]
1 2 3 4 5 6 7 8 9
q).ml.arange[6.25;10.5;.05]
6.25 6.3 6.35 6.4 6.45 6.5 6.55 6.6 6.65 6.7 6.75 6.8 6.85 6.9 6.95 7 7.05 7...
```

## `.ml.confdict`

_True/false positives and negatives as dictionary_

Syntax: `.ml.confdict[x;y]`

Where

-   `x` is a vector of (binary) predicted values
-   `y` is a vector of (binary) true values

returns a dictionary giving the count of true positives (`tp`), true negatives (`tn`), false positives (`fp`) and false negatives (`fn`).

```q
q).ml.confdict[100?"AB";100?"AB"]
tp| 25
fn| 25
fp| 21
tn| 29
```


## `.ml.confmat`

_Confusion matrix_

Syntax: `.ml.confmat[x;y]`

Where

-   `x` is a vector of predicted labels
-   `y` is a vector of the true labels

returns a dictionary representing a confusion matrix with counts.

```q
q).ml.confmat[100?0b;100?0b] / binary classifier
0| 26 27
1| 22 25
q).ml.confmat[100?5;100?5]   / non-binary classifier
0| 5 3 4 5 2
1| 6 5 3 5 6
2| 5 6 5 2 4
3| 5 3 5 2 1
4| 3 4 6 2 3
```

## `.ml.corrmat`

_Correlation of a table_

Syntax: `ml.corrmat[x]`

Where

-   `x` is a table

returns the confusion matrix of a table

```q
q)show t1:([]5?10;5?10;5?10)
x x1 x2
-------
4 1  4 
0 2  4 
2 8  3 
1 5  8 
2 5  9 
q).ml.corrmat[t1]
  | x          x1         x2        
- | --------------------------------
x | 1          -0.1700753 -0.1497195
x1| -0.1700753 1          0.1133737 
x2| -0.1497195 0.1133737  1    
```

## `.ml.crm`

_Correlation of a matrix_

Syntax: `.ml.crm[x]`

Where 

-  `x` is a matrix

returns the correlation matrix

```q
q)show mat:(5?10;5?10;5?10)
8 2 5 0 6
4 0 4 9 6
0 2 9 0 0
q).ml.crm[mat]
1          -0.2048455 0.0562181 
-0.2048455 1          -0.2848782
0.0562181  -0.2848782 1         
```
## `.ml.crossentropy`

_Categorical cross entropy_

Syntax: `.ml.crossentropy[x;y]`

Where

-   `x` is a vector of indices representing class labels
-   `y` is a list of vectors representing the probability of belonging to each class

returns the categorical cross entropy for each class.

```q
q)p%:sum each p:1000 5#5000?1f
q)g:(first idesc@)each p / good labels
q)b:(first iasc@)each p  / bad labels
q).ml.crossentropy[g;p]
1.07453
q).ml.crossentropy[b;p]
3.187829
```
## `.ml.cvm`

_Covariance of a matrix_

Syntax: `.ml.cvm[x]`

Where

-  `x` is a matrix

Returns the covariance matrix

```q
q)show mat:(5?10;5?10;5?10)
3 6 0 8 4
1 7 4 7 6
6 9 9 8 7
q).ml.cvm[mat]
7.36 4   0.04
4    5.2 1.6 
0.04 1.6 1.36
```

## `.ml.describe`

_Descriptive information about a table_

Syntax: `.ml.describe[x]`

Where

-   `x` is a table

returns a dictionary description of aggregate values (count, mean, standard deviation and quartiles) for each numeric column.

```q
q)n:1000
q)tab:([]sym:n?`4;x:n?10000f;x1:1+til n;x2:reverse til n;x3:n?100f)
q).ml.describe tab
     | x        x1       x2       x3       
-----| ------------------------------------
count| 1000     1000     1000     1000     
mean | 4953.491 500.5    499.5    49.77201 
std  | 2890.066 288.8194 288.8194 28.91279 
min  | 7.908894 1        0        0.1122762
q1   | 2491.828 250.75   249.75   24.38531 
q2   | 5000.222 500.5    499.5    49.96016 
q3   | 7453.287 750.25   749.25   74.98685 
max  | 9994.308 1000     999      99.98165 
```


## `.ml.eye`

_Identity matrix_

Syntax: `.ml.eye[x]`

Where

-  `x` is an integer atom

returns an identity matrix of height/width `x`.  

```q
q).ml.eye 5
1 0 0 0 0
0 1 0 0 0
0 0 1 0 0
0 0 0 1 0
0 0 0 0 1
```

## `.ml.f1score`

_F-1 score for classification results_

Syntax: `.ml.f1score[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values
-   `z` is the positive class

returns the F-1 score between predicted and true values.

```q
q)xr:1000?5
q)yr:1000?5
q).ml.f1score[xr;yr;4]
0.1980676
q)xb:1000?0b
q)yb:1000?0b
q).ml.f1score[xb;yb;0b]
0.4655532
```

## `.ml.fbscore`

_F-beta score for classification results_

Syntax: `.ml.fbscore[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values
-   `z` is the positive class
-   `b` is the value of beta

returns the F-beta score between predicted and true values.

```q
q)xr:1000?5
q)yr:1000?5
q).ml.fbscore[xr;yr;4;.5]
0.2254098
q)xb:1000?0b
q)yb:1000?0b
q).ml.fbscore[xb;yb;1b;.5]
0.5191595
```

## `.ml.kfoldx`

_K-Fold cross validation_

Syntax: `.ml.kfoldx[x;y;i;fn]`

Where

-   `x` is the data matrix
-   `y` is the target vector
-   `i` are the indices for the K-fold validation using `.ml.kfsplit`
-   `fn` is the model which is being passed to the function for cross validation

returns the cross validated score for an applied machine-learning algorithm.
```q
n:10000
q)xg:(n?100f;asc n?100f)        / 'good values' for linear regressor
q)yg:asc n?100f
q)reg:.p.import[`sklearn.linear_model][`:LinearRegression][]
q)reg1:.p.import[`sklearn.linear_model][`:SGDRegressor][]
q)reg2:.p.import[`sklearn.linear_model][`:ElasticNet][]
q)reg3:.p.import[`sklearn.neighbors][`:KNeighborsRegressor][]
q)folds:10                      / number of folds for data
q)i:.ml.kfshuff[yg;folds]
q).ml.kfoldx[xg;yg;i]each(reg;reg1;reg2;reg3)
0.9998536 -1.24663e+24 0.9998393 0.9999997
q)yb:n?100f                     / 'bad values' for linear regression
q)xb:(n?100f;n?100f)
q).ml.kfoldx[xb;yb;i]each(reg;reg1;reg2;reg3)
-0.009119423 -7.726559e+21 -0.009119348 -0.2275681
```
!!! note
        This is an aliased version of the function `.ml.xval.kfoldx` which is contained in the `.ml.xval` namespace.

## `.ml.kfshuff`

_Randomized non repeating indices for K-fold cross validation_

Syntax: `.ml.kfshuff[x;y]`

Where

-   `x` is the target vector
-   `y` is the number of 'folds' which the data is to be split into

returns randomized non repeating indices associated with each of K folds.

```q
q)yg:asc 1000?100f
q).ml.kfshuff[yg;5]
647 755 790 152 948 434 583 536 156 637 699 159 315 698 41  345 565 680 775 6..
118 402 34  601 833 877 762 703 129 294 593 634 192 939 545 98  641 266 910 4..
795 69  664 393 519 722 616 55  132 802 448 140 361 194 977 97  247 74  733 6..
633 430 346 267 102 201 123 295 487 418 606 108 154 899 398 932 994 643 944 5..
919 354 119 478 954 567 497 848 665 471 406 541 307 82  984 198 134 622 550 9..
```
!!! note
        This is an aliased version of the function `.ml.xval.kfshuff` which is contained in the `.ml.xval` namespace.


## `.ml.linspace`

_List of evenly-spaced values_

Syntax: `.ml.linspace[x;y;z]`

Where 

-   `x` and `y` are numeric atoms
-   `z` is an int atom

returns a vector of `z` evenly-spaced values between `x` (inclusive) and `y` (inclusive).

```q
q).ml.linspace[10;20;9]
10 11.25 12.5 13.75 15 16.25 17.5 18.75 20
q).ml.linspace[0.5;15.25;12]
0.5 1.840909 3.181818 4.522727 5.863636 7.204545 8.545455 9.886364 11.22727 1..
```


## `.ml.logloss`

_Logarithmic loss_

Syntax: `.ml.logloss[x;y]`

Where

-   `x` is a vector of class labels 0/1
-   `y` is a list of vectors representing the probability of belonging to each class

returns the total logarithmic loss.

```q
q)v:1000?2
q)g:g,'1-g:1-(.5*v)+1000?.5 / good predictions
q)b:b,'1-b:1000?.5          / bad predictions
q).ml.logloss[v;g]
0.3005881
q).ml.logloss[v;b]
1.032062
```

## `.ml.mae`

_Mean absolute error_

Syntax: `.ml.mae[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the mean absolute error between predicted and true values.

```q
q).ml.mae[100?0b;100?0b]
45f
q).ml.mae[100?5;100?5]
173f
```

## `.ml.mape`

_Mean absolute percentage error_

Syntax: `.ml.mape[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the mean absolute percentage error between predicted and true values. All values must be floats.

```q
q)mape[100?5.0;100?5.0]
660.9362
```

## `.ml.matcorr`

_Matthews-correlation coefficient_

Syntax: `.ml.matcorr[x;y]`

Where

-   `x` is a vector of predicted labels
-   `y` is a vector of the true labels

returns the Matthews-correlation coefficient between predicted and true values.

```q
q).ml.matcorr[100?0b;100?0b]
0.1256775
q).ml.matcorr[100?5;100?5]
7.880334e-06
```

## `.ml.mse`

_Mean squared error_

Syntax: `.ml.mse[x;y]`

Where 

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the mean squared error between predicted and true values.

```q
q).ml.mse[asc 100?1f;asc 100?1f]
0.0004801384
q).ml.mse[asc 100?1f;desc 100?1f]
0.3202164
```


## `.ml.percentile`

_Percentile calculation for an array_

Syntax: `.ml.percentile[x;y]`

Where

-  `x` is a numerical array
-  `y` is the percentile of interest

returns the value below which `y` percent of the of the observations within the array are found.

```q
q).ml.percentile[10000?1f;.2]
0.2030272
q).ml.percentile[10000?1f;.6]
0.5916521
```


## `.ml.precision`

_Precision of a binary classifier_

Syntax: `.ml.precision[x;y;z]`

Where

-   `x` is a vector of (binary) predicted values
-   `y` is a vector of (binary) true values
-   `z` is the binary value defined to be 'true'

returns a measure of the precision.

```q
q).ml.precision[1000?0b;1000?0b;1b]
0.5020161
q).ml.precision[1000?"AB";1000?"AB";"B"]
0.499002
```

## `.ml.r2score`

_R2-score for regression model validation_

Syntax: `.ml.r2score[x;y]`

Where

-   `x` are predicted continuous values
-   `y` are true continuous values

returns the R2-score between the true and predicted values. Values close to 1 indicate good prediction, while negative values indicate poor predictors of the system behaviour.

```q
q)xg:asc 1000?50f           / 'good values'
q)yg:asc 1000?50f
q).ml.r2score[xg;yg]
0.9966209
q)xb:asc 1000?50f           / 'bad values'
q)yb:desc 1000?50f
-2.981791
```

## `.ml.range`

_Range of values_

Syntax: `.ml.range[x]`

Where

-   `x` is a vector of numeric values

returns the range of its values.

```q
q).ml.range 1000?100000f
99742.37
```


## `.ml.roc`

_X- and Y-axis values for a ROC curve_

Syntax: `.ml.roc[x;y]`

Where

-   `x` is the label associated with a prediction
-   `y` is the probability that a prediction belongs to the positive class

returns the co-ordinates of the true and false positive values associated with the ROC curve.

```q
q)v:raze reverse 50?'til[20+1]xprev\:20#1b
q)p:asc count[v]?1f
q).ml.roc[v;p]
0           0           0           0           0           0          0     ..
0.001937984 0.003875969 0.005813953 0.007751938 0.009689922 0.01162791 0.0135..
```


## `.ml.rocaucscore`

_Area under a ROC curve_

Syntax: `.ml.rocaucscore[x;y]`

Where

-   `x` is the label associated with a prediction
-   `y` is the probability that a prediction belongs to the positive class

returns the area under the ROC curve.

```q
q)v:raze reverse 50?'til[20+1]xprev\:20#1b
q)p:asc count[v]?1f
q).ml.rocaucscore[v;p]
0.8696362
```

## `.ml.rmse`

_Root mean squared error for a regressor_

Syntax: `.ml.rmse[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the root mean squared error for a regression model between predicted and true values.

```q
q)n:10000
q)xg:asc n?100f
q)yg:asc n?100f
q).ml.rmse[xg;yg]
0.5321886
q)xb:n?100f
q).ml.rmse[xb;yg]
41.03232
```

## `.ml.rmsle`

_Root mean squared log error_

Syntax: `.ml.rmsle[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the root mean squared log error between predicted and true values.

```q
q).ml.rmsle[100?0b;100?0b]
0.5187039
q).ml.rmsle[100?5;100?5]
0.8465022
```

## `.ml.sensitivity`

_Sensitivity of a binary classifier_

Syntax: `.ml.sensitivity[x;y;z]`

Where

-   `x` is a vector of (binary) predicted values
-   `y` is a vector of (binary) true values
-   `z` is the binary value defined to be 'true'

returns a measure of the sensitivity.

```q
q).ml.sensitivity[1000?0b;1000?0b;1b]
0.4876033
q).ml.sensitivity[1000?`class1`class2;1000?`class1`class2;`class1]
0.5326923
```


## `.ml.shape`

_Shape of a matrix_

Syntax: `.ml.shape[x]`

Where

-   `x` is an object

returns its shape as a list of dimensions.

```q
q).ml.shape 10
`long$()
q).ml.shape enlist 10
,1
q).ml.shape til 10
,10
q).ml.shape enlist til 10
1 10
q).ml.shape 2 5#til 10
2 5
q).ml.shape 2 3 4#til 24
2 3 4
q).ml.shape ([]c1:til 10;c2:0)
10 2
```

!!! warning "Behavior of `.ml.shape` is undefined for ragged/jagged arrays."

## `.ml.smape`

_Symmetric mean absolute percentage error_

Syntax: `.ml.smape[x;y]`

Where

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the symmetric-mean absolute percentage between predicted and true values.

```q
q)smape[100?0b;100?0b]
92f
q)smape[100?5;100?5]
105.781
```

## `.ml.specificity`

_Specificity of a binary classifier_

Syntax: `.ml.specificity[x;y;z]`

Where

-   `x` is a vector of (binary) predicted values
-   `y` is a vector of (binary) true values
-   `z` is the binary value defined to be 'true'

returns a measure of the specificity.

```q
q).ml.specificity[1000?0b;1000?0b;1b]
0.5426829
q).ml.specificity[1000?100 200;1000?100 200;200]
0.4676409
```


## `.ml.sse`

_Sum squared error_

Syntax: `.ml.mse[x;y]`

Where 

-   `x` is a vector of predicted values
-   `y` is a vector of true values

returns the sum squared error between predicted and true values.

```q
q).ml.sse[asc 100?1f;asc 100?1f]
0.4833875
q).ml.sse[asc 100?1f;desc 100?1f]
32.06403
```


## `.ml.tscore`

_One-sample t-test score_

Syntax: `.ml.tscore[x;y]`

Where

-   `x` is a set of samples from a distribution
-   `y` is the population mean

returns the one sample t-score for a distribution with less than 30 samples.

```q
q)x:25?100f
q)y:15
q).ml.tscore[x;y]
7.634824
```

!!! note "Above 30 samples a one-sample t-score is not statistically significant."


## `.ml.tscoreeq`

_T-test for independent samples with equal variances and equal sample size_

Syntax: `.ml.tscoreeq[x;y]`

Where

-   `x` and `y` are independent sample sets with equal variance and sample size

returns their t-test score.

```q
q)x:10+(-.5+avg each 0N 20#1000?1f)*sqrt 20*12 / ~N(10,1)
q)y:20+(-.5+avg each 0N 20#1000?1f)*sqrt 20*12 / ~N(20,1)
q)(avg;dev)@\:/:(x;y) / check dist
9.87473  1.010691 
19.84484 0.9437789
q).ml.tscoreeq[x;y]
50.46957
```

## `.ml.util.classreport`

_Statistical information about classification results_

Syntax: `.ml.util.classreport[x;y]`

Where

* `x` is a vector of predicted values.
* `y` is a vector of true values.

returns the accuracy, precision and f1 scores and the support (number of occurrences) of each class.

```q
q)n:1000
q)xr:n?5
q)yr:n?5
q).ml.util.classreport[xr;yr]
class     precision recall f1_score support
-------------------------------------------
0         0.2       0.23   0.22     179
1         0.22      0.22   0.22     193
2         0.21      0.21   0.21     192
3         0.19      0.19   0.19     218
4         0.21      0.17   0.19     218
avg/total 0.206     0.204  0.206    1000

q)xb:n?0b
q)yb:n?0b
q).ml.util.classreport[xb;yb]
class     precision recall f1_score support
-------------------------------------------
0         0.51      0.51   0.51     496
1         0.52      0.52   0.52     504
avg/total 0.515     0.515  0.515    1000

q)xc:n?`A`B`C
q)yc:n?`A`B`C
q).ml.util.classreport[xc;yc]
class     precision recall f1_score support
-------------------------------------------
A         0.34      0.33   0.33     336
B         0.33      0.36   0.35     331
C         0.32      0.3    0.31     333
avg/total 0.33      0.33   0.33     1000
```

## `.ml.util.combs`

_Unique combinations of vector or matrix_

Syntax: `.ml.util.combs[x]`

Where 

-  `x` is a vector or matrix

Returns the unique combination of values from the data

```q
q)show l:5?10
4 1 7 1 1
q).ml.util.combs[3;l]
7 1 4
1 1 4
1 7 4
1 7 1
1 1 4
1 7 4
1 7 1
1 1 4
1 1 1
1 1 7
```

## `.ml.util.df2tab`

_q table from a Pandas dataframe_

Syntax: `.ml.util.df2tab[x]`

Where

-   `x` is an embedPy representation of a Pandas DataFrame

returns it as a q table.

```q
q)p)import pandas as pd
q)print t:.p.eval"pd.DataFrame({'fcol':[.1,.2,.3,.4,.5],'jcol':[10,20,30,40,50]})"
   fcol  jcol
0   0.1    10
1   0.2    20
2   0.3    30
3   0.4    40
4   0.5    50
q).ml.util.df2tab t
fcol jcol
---------
0.1  10  
0.2  20  
0.3  30  
0.4  40  
0.5  50  
q)print kt:t[`:set_index]`jcol
      fcol
jcol      
10     0.1
20     0.2
30     0.3
40     0.4
50     0.5
q).ml.util.df2tab kt
jcol| fcol
----| ----
10  | 0.1 
20  | 0.2 
30  | 0.3 
40  | 0.4 
50  | 0.5 
```

!!! note "DataFrame indices are mapped to q key columns."


## `.ml.util.dropconstant`

_Columns with zero variance removed_

Syntax: `.ml.util.dropconstant[x]`

Where

-   `x` is a numerical table

returns `x` without columns of zero variance.

```q
q)5#tab:([]1000?100f;1000#10;1000#0N)
x        x1 x2
--------------
95.25017 10   
42.09728 10   
98.80532 10   
54.5461  10   
51.7746  10   
q)5#.ml.util.dropconstant tab
x       
--------
95.25017
42.09728
98.80532
54.5461 
51.7746 
```


## `.ml.util.filltab`

_Tailored filling of null values for a simple matrix_

Syntax: `.ml.util.filltab[t;gcol;tcol;dict]`

Where

-   `t` is a table
-   `gcol` is a grouping column for the fill
-   `tcol` is a time column in the data
-   `dict` is a dictionary defining fill behaviour

returns a table with columns filled according to assignment of keys in the dictionary, `dict`. The function defaults to forward-filling nulls, followed by back-filling. However, changes to the default dictionary allow for zero, median, mean or linear interpolation to be applied on individual columns.

```q
q)show tab:([]sym:10?`A`B;time:asc 10?0t;@[10?1f;0 1 8 9;:;0n];@[10?1f;4 5;:;0n];@[10?1f;2*til 5;:;0n])
sym time         x          x1         x2        
-------------------------------------------------
B   00:34:58.497            0.3252897            
A   03:19:24.024            0.5564216  0.2726145 
A   06:27:11.361 0.3867108  0.1692341            
B   10:47:00.412 0.5624072  0.2069839  0.8841147 
A   12:59:24.858 0.3319393                       
B   17:34:47.917 0.2318218             0.2439706 
A   18:56:14.238 0.02746107 0.8516542            
B   19:28:11.045 0.9505891  0.07314201 0.1275351 
B   22:28:15.345            0.2221491            
A   22:58:06.856            0.2407035  0.04661209
q).ml.util.filltab[tab;`sym;`time;()!()] / default fill
sym time         x          x1         x2        
-------------------------------------------------
B   00:34:58.497 0.5624072  0.3252897  0.8841147 
A   03:19:24.024 0.3867108  0.5564216  0.2726145 
A   06:27:11.361 0.3867108  0.1692341  0.2726145 
B   10:47:00.412 0.5624072  0.2069839  0.8841147 
A   12:59:24.858 0.3319393  0.1692341  0.2726145 
B   17:34:47.917 0.2318218  0.2069839  0.2439706 
A   18:56:14.238 0.02746107 0.8516542  0.2726145 
B   19:28:11.045 0.9505891  0.07314201 0.1275351 
B   22:28:15.345 0.9505891  0.2221491  0.1275351 
A   22:58:06.856 0.02746107 0.2407035  0.04661209
q).ml.util.filltab[tab;`sym;`time;`linear`median`mean!(`x;`x1;`x2)] / interpolations fills
sym time         x          x1         x2        
-------------------------------------------------
B   00:34:58.497 0.8518633  0.3252897  0.4185401 
A   03:19:24.024 0.434668   0.5564216  0.2726145 
A   06:27:11.361 0.3867108  0.1692341  0.1596133 
B   10:47:00.412 0.5624072  0.2069839  0.8841147 
A   12:59:24.858 0.3319393  0.2407035  0.1596133 
B   17:34:47.917 0.2318218  0.2069839  0.2439706 
A   18:56:14.238 0.02746107 0.8516542  0.1596133 
B   19:28:11.045 0.9505891  0.07314201 0.1275351 
B   22:28:15.345 1.316885   0.2221491  0.4185401 
A   22:58:06.856 -0.1277062 0.2407035  0.04661209

```

## `.ml.util.freqencode`

_Encoded frequency of individual category occurences_

Syntax:.`.ml.util.frequencode[x]`

Where

-   `x` is a table containing at least one column with categorical features (symbols)

returns table with frequency of occurrance of individual categories.

```q
q)tab:([]10?`a`b`c;10?10f)
x x1      
----------
b 7.667049
b 2.975384
b 8.607676
a 7.35515 
c 3.384913
c 8.934151
c 3.731562
b 6.22616 
c 2.714665
a 7.288787
q).ml.util.freqencode[tab]
x1         freq_x
-----------------
9.602129   0.6   
2.256037   0.4   
7.922876   0.6   
0.04380018 0.6   
3.442662   0.6   
7.116088   0.4   
0.9901482  0.4   
8.60422    0.6   
6.491715   0.4   
6.482049   0.6   
```


## `.ml.util.infreplace`

_+/- infinities replaced with data min/max_

Syntax: `.ml.util.infreplace[x]`

Where

-   `x` is a dictionary/table/list of numeric values

returns the data with positive/negative infinities replaced by max/min values for the given key.

```q
q)show d:`A`B`C!(5 6 9 0w;10 -0w 0 50;0w 1 2 3)
A| 5  6   9 0w
B| 10 -0w 0 50
C| 0w 1   2 3 
q).ml.util.infreplace d`A
5 6 9 9f
q).ml.util.infreplace d
A| 5  6 9 9 
B| 10 0 0 50
C| 3  1 2 3 
q).ml.util.infreplace flip d
A B  C
------
5 10 3
6 0  1
9 0  2
9 50 3
```

## `.ml.util.lexiencode`

_Categorical labels of features based on their lexigraphical order_

Syntax:`.ml.util.lexiencode[tab]`

Where

-   `tab` is a table containing at least one column with letters

returns table with lexigraphical order of letters column.

```q
q)show tab:([]10?10f;10?`a`b`c;10?10f)
x         x1 x2      
---------------------
2.652528  b  2.194196
3.49562   a  3.997433
6.02218   c  9.08391 
0.5352858 a  3.560436
2.101436  a  7.084245
2.812222  b  7.724299
5.215879  c  8.157121
2.991163  c  2.096742
8.250197  a  7.881085
0.4501006 b  4.31398 
q).ml.util.lexiencode[tab]
x         x2       lexi_label_x1
--------------------------------
2.652528  2.194196 1            
3.49562   3.997433 0            
6.02218   9.08391  2            
0.5352858 3.560436 0            
2.101436  7.084245 0            
2.812222  7.724299 1            
5.215879  8.157121 2            
2.991163  2.096742 2            
8.250197  7.881085 0            
0.4501006 4.31398  1 
```


## `.ml.util.minmaxscaler`

_Data scaled between 0-1_

Syntax: `.ml.util.minmaxscaler[x]`

Where

-   `x` is a numerical table, matrix or list

returns a min-max scaled representation with values scaled between 0 and 1.

```q
q)n:5
q)show tab:([]n?100f;n?1000;n?100f;n?50f)
x        x1  x2       x3
------------------------------
12.48134 837 42.83142 7.597138
18.019   591 77.97026 46.69185
98.8875  860 73.44471 28.72854
30.70513 599 80.56178 39.70485
42.17381 187 75.26142 38.26483

q).ml.util.minmaxscaler[tab]
x         x1        x2        x3
---------------------------------------
0          0.9658247 0         0        
0.06408864 0.6002972 0.9313147 1        
1          1         0.8113701 0.5405182
0.2109084  0.6121842 1         0.8212801
0.3436384  0         0.85952   0.7844459

q)show mat:value flip tab
12.48134 18.019   98.8875  30.70513 42.17381
837      591      860      599      187     
42.83142 77.97026 73.44471 80.56178 75.26142
7.597138 46.69185 28.72854 39.70485 38.26483

q).ml.util.minmaxscaler[mat]
0         0.06408864 1         0.2109084 0.3436384
0.9658247 0.6002972  1         0.6121842 0        
0         0.9313147  0.8113701 1         0.85952  
0         1          0.5405182 0.8212801 0.7844459

q)list:100?100
q).ml.util.minmaxscaler[list]
0.7835052 0.2886598 0.5463918 0.443299 1 0.09278351 0.1030928 0 0.9175258 0.9..
```

## `.ml.nullencode`

_Nulls filled with avg/med/min/max of the column_

Syntax: `.ml.util.nullencode[t;y]`

Where

-   `t` is a table containing nulls and y is the function to be applied to nulls

returns table with nulls filled and additional columns that encode the null positions of each original column.

```q
q)show tab:([]@[10?1f;0 1 8 9;:;0n];@[10?1f;4 5;:;0n];@[10?1f;2*til 5;:;0n])
x         x1        x2        
------------------------------
          0.3238152           
          0.8616775 0.8640424 
0.2062779 0.032353            
0.7802666 0.3826807 0.8002228 
0.4809723                     
0.940638            0.5370027 
0.3497468 0.2885991           
0.3165283 0.2879356 0.5639193 
          0.7708107           
          0.783687  0.08766932

q).ml.util.nullencode[tab;avg]
x         x1        x2         null_x null_x1 null_x2
-----------------------------------------------------
0.512405  0.3238152 0.5705713  1      0       1      
0.512405  0.8616775 0.8640424  1      0       0      
0.2062779 0.032353  0.5705713  0      0       1      
0.7802666 0.3826807 0.8002228  0      0       0      
0.4809723 0.4664448 0.5705713  0      1       1      
0.940638  0.4664448 0.5370027  0      1       0      
0.3497468 0.2885991 0.5705713  0      0       1      
0.3165283 0.2879356 0.5639193  0      0       0      
0.512405  0.7708107 0.5705713  1      0       1      
0.512405  0.783687  0.08766932 1      0       0 
```

## `.ml.util.onehot`

_One-hot encoding_

Syntax: `.ml.util.onehot[x]`

Where

-   `x` is a list of symbols or a table containing symbols

returns its one-hot encoded representation.

```q
q)x:`a`a`b`b`c`a
q).ml.util.onehot[x]
1 0 0
1 0 0
0 1 0
0 1 0
0 0 1
1 0 0

q)show tab:([]10?`a`b`c;10?10f;10?10f)
x x1        x2        
----------------------
a 7.310299  5.697305  
b 0.8842384 1.714374  
b 8.203719  0.09300471
a 8.749888  6.349401  
b 7.73655   8.607533  
a 6.01945   3.617256  
b 7.03047   0.9167078 
a 7.830783  7.658206  
b 5.559878  2.115546  
c 6.325994  1.445261  
q)q).ml.util.onehot[tab]
x1        x2         x_a x_b x_c
--------------------------------
7.310299  5.697305   1   0   0  
0.8842384 1.714374   0   1   0  
8.203719  0.09300471 0   1   0  
8.749888  6.349401   1   0   0  
7.73655   8.607533   0   1   0  
6.01945   3.617256   1   0   0  
7.03047   0.9167078  0   1   0  
7.830783  7.658206   1   0   0  
5.559878  2.115546   0   1   0  
6.325994  1.445261   0   0   1 
```


## `.ml.util.polytab`

_Polynomial features of degree n from a table_

Syntax: `.ml.util.polytab[t;n]`

Where

-   `t` is a table of numerical values
-   `n` is the order of the polynomial feature being created

returns the polynomial derived features of degree `n` in the form of a table.

```q
q)n:100
q)5#tab:([]val:sin .001*til n;til n;n?100f;n?1000f;n?10)
val          x  x1       x2       x3
--------------------------------------
0            0  52.26479 990.7741 1
0.0009999998 1  2.740491 52.11973 1
0.001999999  2  42.23458 967.8194 8
0.002999996  3  23.54108 337.9137 0

q)5#.ml.util.polytab[tab;2]
val_x        val_x1      val_x2     val_x3       x_x1     x_x2     x_x3 x1_x2..
-----------------------------------------------------------------------------..
0            0           0          0            0        0        0    51782..
0.0009999998 0.002740491 0.05211972 0.0009999998 2.740491 52.11973 1    142.8..
0.003999997  0.08446911  1.935637   0.01599999   84.46917 1935.639 16   40875..
0.008999987  0.07062314  1.01374    0            70.62325 1013.741 0    7954...
0.01599996   0.2569871   2.354285   0.01999995   256.9878 2354.291 20   37814..

q)5#.ml.util.polytab[tab;3]
val_x_x1    val_x_x2   val_x_x3     val_x1_x2 val_x1_x3   val_x2_x3  x_x1_x2 ..
-----------------------------------------------------------------------------..
0           0          0            0         0           0          0       ..
0.002740491 0.05211972 0.0009999998 0.1428336 0.002740491 0.05211972 142.8337..
0.1689382   3.871275   0.03199998   81.75084  0.6757529   15.4851    81750.9 ..
0.2118694   3.041219   0            23.86453  0           0          23864.57..
1.027948    9.41714    0.07999979   151.2556  1.284935    11.77143   151256  ..

/this can be integrated with the original data via the syntax
q)5#newtab:tab^.ml.util.polytab[tab;2]^.ml.util.polytab[tab;3]

val          x  x1       x2       x3 val_x        val_x1    ..
------------------------------------------------------------..
0            0  52.26479 990.7741 1  0            0         ..
0.0009999998 1  2.740491 52.11973 1  0.0009999998 0.00274049..
0.001999999  2  42.23458 967.8194 8  0.003999997  0.08446911..
0.002999996  3  23.54108 337.9137 0  0.008999987  0.07062314..
0.003999989  4  64.24694 588.5728 5  0.01599996   0.2569871 ..
```


## `.ml.util.stdscaler`

_Standard scaler transform-based representation_

Syntax: `.ml.util.stdscaler[x]`

Where

-   `x` is a numerical table, matrix or list

returns a table where each column has undergone a standard scaling given as:

 `(x-avg x)%dev x`

```q
q)n:5
q)show tab:([]n?100f;n?1000;n?100f;n?50f)
x        x1  x2       x3
------------------------------
12.48134 837 42.83142 7.597138
18.019   591 77.97026 46.69185
98.8875  860 73.44471 28.72854
30.70513 599 80.56178 39.70485
42.17381 187 75.26142 38.26483

q).ml.util.stdscaler[tab]
x          x1         x2         x3
---------------------------------------
-0.9029555 0.9173914   -1.969172 -1.813095 
-0.7241963 -0.09826245 0.5763785 1.068269  
1.886294   1.012351    0.2485354 -0.2556652
-0.3146793 -0.06523305 0.764115  0.5533119 
0.0555375  -1.766247   0.380143  0.4471792 

q)show mat:value flip tab
12.48134 18.019   98.8875  30.70513 42.17381
837      591      860      599      187     
42.83142 77.97026 73.44471 80.56178 75.26142
7.597138 46.69185 28.72854 39.70485 38.26483

q).ml.util.stdscaler[mat]
-0.9029555 -0.7241963  1.886294   -0.3146793  0.0555375
0.9173914  -0.09826245 1.012351   -0.06523305 -1.766247
-1.969172  0.5763785   0.2485354  0.764115    0.380143 
-1.813095  1.068269    -0.2556652 0.5533119   0.4471792

q)list:100?100
q).ml.util.stdscaler[list]
0.8394957 -0.7121328 0.09600701 -0.2272489 1.518333 -1.326319 -1.293993 -1.61..
```


## `.ml.util.tab2df`

_Pandas dataframe from a q table_

Syntax: `.ml.util.tab2df[x]`

Where

-   `x` is a table

returns it as a Pandas dataframe.

```q
q)n:5
q)table:([]x:n?10000f;x1:1+til n;x2:reverse til n;x3:n?100f) / q table for input
x        x1 x2 x3
-----------------------
2631.44  1  4  78.71917
1118.109 2  3  80.09356
3250.627 3  2  16.71013
q)show pdf:.ml.util.tab2df[table] / convert to pandas dataframe and show it is an embedPy object
{[f;x]embedPy[f;x]}[foreign]enlist
q)print pdf / display the python form of the dataframe
             x  x1  x2         x3
0  2631.439704   1   4  78.719172
1  1118.109056   2   3  80.093563
2  3250.627243   3   2  16.710134
```

## `.ml.util.timespantransform`

_Timespan converted into its constituent parts_

Syntax: `.ml.util.timespantransform[tab]`

Where

-   `tab` is the table onto which this calculation is to be computed

returns a table, with the timespan column removed, converted into columns containing the consituent parts, representing the information contained in the timespan.

!!!note
        At a fundamental level this works by looking for columns in the dataset of type "p". It should also be noted here that if the function is passed a table without a timespan, then it will return the original table.
```q
q)show t:([]asc 100?0p;100?1f;100?`1;100?10)
x                             x1         x2 x3
----------------------------------------------
2000.01.18D20:12:15.589928180 0.3813679  e  8
2000.01.29D20:12:45.571620315 0.03112646 o  1
2000.02.06D08:20:37.705269456 0.4194691  k  9
2000.03.01D20:24:37.530120313 0.1862869  d  5
2000.04.03D02:38:08.821253926 0.7959531  e  4
q).ml.util.timespantransform[t]
yr_x qtr_x mm_x dom_x dow_x wd_x hr_x mn_x sec_x x1         x2 x3
-----------------------------------------------------------------
2000 1     1    18    3     1    20   12   15    0.3813679  e  8
2000 1     1    29    0     0    20   12   45    0.03112646 o  1
2000 1     2    6     1     0    8    20   37    0.4194691  k  9
2000 1     3    1     4     1    20   24   37    0.1862869  d  5
2000 2     4    3     2     1    2    38   8     0.7959531  e  4
```



## `.ml.util.traintestsplit`

_Training and testing sets_

Syntax: `.ml.util.traintestsplit[x;y;sz]`

Where

-   `x` is a matrix
-   `y` is a boolean vector of the same count as `x`
-   `sz` is a numeric atom in the range 0-1, representing the percentage of data in the testing set

returns a dictionary containing the data matrix `x` and target `y`, split into a train/test sets according to `sz`.

```q
q)x:(30 20)#1000?10f
q)y:rand each 30#0b
q).ml.util.traintestsplit[x;y;.2] / split the data such that 20% is contained in the test set
xtrain| (2.02852 2.374546 1.083376 2.59378 6.698505 6.675959 4.120228 2.63468..
ytrain| 110010100101111001110000b
xtest | (8.379916 8.986609 7.06074 2.067817 5.468488 4.103195 0.1590803 0.259..
ytest | 000001b
```


## `.ml.util.traintestsplitseed`

_Training and testing sets with a seed_

Syntax: `.ml.util.traintestsplitseed[x;y;sz;seed]`

Where

-   `x` is a matrix
-   `y` is a boolean vector of the same count as `x`
-   `sz` is a numeric atom in the range 0-1, representing the percentage of data in the testing set
-   `seed` is a numeric atom

returns a dictionary containing the data matrix `x` and target `y`, split into a train/test sets according to `sz`. This version of `.ml.util.traintestsplit` can be given a random seed, `seed`, allowing the same data split to be achieved repeatedly.

```q
q)x:(30 20)#1000?10f
q)y:rand each 30#0b
q).ml.util.traintestsplitseed[x;y;.2;42]
xtrain| (8.752061 6.82448 3.992896 2.465234 8.599461 2.452222 6.070236 6.8686..
ytrain| 000011101000001110111101b
xtest | (4.204472 7.137387 1.163132 9.893949 4.504886 5.465625 8.298632 0.049..
ytest | 000001b
q).ml.util.traintestsplitseed[x;y;.2;42] /show that the splitting on repetition is the same
xtrain| (8.752061 6.82448 3.992896 2.465234 8.599461 2.452222 6.070236 6.8686..
ytrain| 000011101000001110111101b
xtest | (4.204472 7.137387 1.163132 9.893949 4.504886 5.465625 8.298632 0.049..
ytest | 000001b
```



---
author: Conor McCarthy
date: August 2018
keywords: machine-learning, ml, feature extraction, feature selection, time-series forecasting, interpolation
---

# <i class="fa fa-share-alt"></i> FRESH: a feature-extraction and feature-significance toolkit


<i class="fa fa-github"></i>
[KxSystems/ml](https://github.com/kxsystems/ml)

Feature extraction and selection are vital components of many machine-learning pipelines. Here we outline an implementation of the [FRESH](https://arxiv.org/pdf/1610.07717v3.pdf) (FeatuRe Extraction and Scalable Hypothesis testing) algorithm.

Feature extraction is the process of building derived, aggregate features from a time-series dataset. The features created are designed to characterize the underlying time-series in a way that is easier to interpret, providing a more suitable input to machine-learning algorithms.

Following feature extraction, statistical-significance tests between feature and target vectors can be applied. This allows selection of only those features with relevance (in the form of p-values) above a given threshold.

Feature selection can improve the accuracy of a machine-learning algorithm by:

-   Simplifying the models used.
-   Shortening the training time needed.
-   Avoiding the curse of dimensionality.
-   Reducing variance in the dataset to reduce overfitting.


Notebooks showing examples of the FRESH algorithm, used in different applications, can be found at <i class="fa fa-github"></i>[KxSystems/ml/fresh/notebooks](https://github.com/kxsystems/ml/tree/master/fresh/notebooks).


## Loading

Load the FRESH library using:

```q
q)\l ml/ml.q
q).ml.loadfile`:fresh/init.q
```


## Data formatting

Data passed to the feature extraction procedure should contain an identifying (ID) column, which groups the time-series into subsets. The ID column can be inherent to the data or derived for a specific use-case, e.g. applying a sliding window to the dataset.

Null values in the data should be replaced with derived values most appropriate to the column.

Data-types supported by the feature-extraction procedure are boolean, int, real, long, short and float. Other datatypes should not be passed to the extraction procedure.

In particular, data should not contain text (symbols or strings), other than the ID column. If a text-based feature is thought to be important, one-hot encoding can be used to convert to numerical values.

!!! note
    A range of formatting functions (e.g. null-filling and one-hot encoding) are supplied in the [Utilities section](utils.md) of the toolkit.


## Calculated features

Feature-extraction functions are defined in the script `fresh.q` and found within the `.ml.fresh` namespace.

Function                               | Output 
:--------------------------------------|:--------------------------------------------
absenergy[x]                           | Absolute sum of differences between successive datapoints
aggautocorr[x]                         | Aggregation (mean/var etc.) of an auto-correlation over all possible lags, from 1 to the length of the dataset
augfuller[x]                           | Hypothesis test to check for unit root in time-series dataset
autocorr[x;lag]                        | Auto-correlation over specified lags
binnedentropy [x;n bins]               | System entropy of data, binned into equidistant bins
c3[x;lag]                              | Measure of the non-linearity of the time-series
changequant[x;ql;qh;isabs]             | Aggregated value of successive changes within corridor specified by `ql `(lower quantile) and `qh` (upper quantile)
cidce[x;isabs]                         | Measure of time-series complexity based on peaks and troughs in the dataset
countabovemean[x]                      | Number of points in the dataset with a value greater than the mean
fftaggreg[x]                           | Spectral centroid (mean), variance, skew, and kurtosis of the absolute Fourier-transform spectrum
fftcoeff[x;coeff]                      | Fast-Fourier transform coefficients given real inputs and extract real, imaginary, absolute and angular components
hasdup[x]                              | If the time-series contains any duplicate values
hasdupmax[x]                           | Boolean stating if duplicate of maximum value exists in the dataset
indexmassquantile[x;q]                 | Relative index `i`, where `q`% of the time-series x values mass lies left of `i`
kurtosis[x]                            | Adjusted G2 Fisher-Pearson kurtosis
lintrend[x]                            | Slope, intercept, r-value, p-value and standard error associated with the time-series
longstrikelmean[x]                     | Length of the longest subsequence in `x` less than the mean of `x`
meanchange[x]                          | Mean over the absolute difference between subsequent time-series values
mean2dercentral[x]                     | Mean value of the central approximation of the second derivative of the time-series
numcrossingm[x;m]                      | Number of crossings in the dataset over a value `m`. Crossing is defined as sequential values either side of `m`, where the first is less than `m` and the second is greater, or vice-versa
numcwtpeaks[x;width]                   | Peaks in the time-series following data smoothing via application of a Ricker wavelet
numpeaks[x;support]                    | Number of peaks with a specified support in a time-series `x`
partautocorrelation[x;lag]             | Partial autocorrelation of the time-series at a specified lag
perrecurtoalldata[x]                   | (Count of values occurring more than once)÷(count different values)
perrecurtoallval[x]                    | (Count of values occurring more than once)÷(count data)
ratiobeyondrsigma[x;r]                 | Ratio of values more than `r*dev[x]` from the mean of `x`
ratiovalnumtserieslength[x]            | (Number of unique values)÷(count values)
spktwelch[x;coeff]                     | Cross power spectral density of the time-series at different tunable frequencies
symmetriclooking[x]                    | If the data ‘appears’ symmetric
treverseasymstat[x;lag]                | Measure of the asymmetry of the time-series based on lags applied to the data
vargtstdev[x]                          | If the variance of the dataset is larger than the standard deviation

!!! note

Feature-extraction functions are not, typically, called individually. A detailed explanation of each operation is therefore excluded.


## Feature extraction

Feature-extraction involves applying a set of aggregations to subsets of the initial input data, with the goal of obtaining information that is more informative than the raw time-series. 

The `.ml.fresh.createfeatures` function applies a set of aggregation functions to derive features. There are 57 such functions available in the `.ml.fresh.feat` namespace, though users may select a subset based on requirements.

Syntax: `.ml.fresh.createfeatures[table;aggs;cnames;funcs]`

Where

-   `table` is the input data (table).
-   `aggs` is the id column name (symbol).
-   `cnames` are the column names (symbols) on which extracted features will be calculated (these columns should contain only numerical values).
-   `funcs` is the dictionary of functions to be applied to the table (a subset of `.ml.fresh.feat`).

This returns a table keyed by ID column and containing the features extracted from the subset of the data identified by the ID.

```q 
q)m:30;n:100
q)show tab:([]date:raze m#'"d"$til n;time:(m*n)#"t"$til m;col1:50*1+(m*n)?20;col2:(m*n)?1f)
date       time         col1 col2      
---------------------------------------
2000.01.01 00:00:00.000 1000 0.3927524 
2000.01.01 00:00:00.001 350  0.5170911 
2000.01.01 00:00:00.002 950  0.5159796 
2000.01.01 00:00:00.003 550  0.4066642 
2000.01.01 00:00:00.004 450  0.1780839 
2000.01.01 00:00:00.005 400  0.3017723 
2000.01.01 00:00:00.006 400  0.785033  
2000.01.01 00:00:00.007 500  0.5347096 
2000.01.01 00:00:00.008 600  0.7111716 
2000.01.01 00:00:00.009 250  0.411597  
2000.01.01 00:00:00.010 50   0.4931835 
2000.01.01 00:00:00.011 400  0.5785203 
2000.01.01 00:00:00.012 800  0.08388858
2000.01.01 00:00:00.013 950  0.1959907 
2000.01.01 00:00:00.014 1000 0.375638  
2000.01.01 00:00:00.015 650  0.6137452 
2000.01.01 00:00:00.016 50   0.5294808 
2000.01.01 00:00:00.017 600  0.6916099 
2000.01.01 00:00:00.018 900  0.2296615 
2000.01.01 00:00:00.019 300  0.6919531 
..
q)dict:.ml.i.dict
q)show features:.ml.fresh.createfeatures[tab;`date;2_ cols tab;dict]
date      | absenergy_col1 absenergy_col2 abssumchange_col1 abssumchange_col2..
----------| -----------------------------------------------------------------..
2000.01.01| 1.156e+07      9.245956       8700              7.711325         ..
2000.01.02| 1.1225e+07     8.645625       11350             9.036386         ..
2000.01.03| 9910000        10.8401        9800              9.830704         ..
2000.01.04| 1.0535e+07     7.900601       7350              10.21271         ..
2000.01.05| 7830000        8.739328       6900              11.02193         ..
2000.01.06| 9150000        9.530337       8150              11.38859         ..
2000.01.07| 1.296e+07      11.36589       9500              10.8551          ..
2000.01.08| 1.11175e+07    12.97225       8800              11.70683         ..
2000.01.09| 1.183e+07      10.99597       11600             9.372777         ..
2000.01.10| 1.076e+07      8.8356         11600             9.923837         ..
2000.01.11| 7640000        11.77406       11400             9.307188         ..
2000.01.12| 1.13e+07       9.965319       9150              9.232088         ..
2000.01.13| 1.0195e+07     9.743622       6400              8.435915         ..
2000.01.14| 1.077e+07      9.934516       10400             8.685272         ..
2000.01.15| 1.033e+07      12.23959       11750             7.666534         ..
2000.01.16| 7602500        10.44816       8800              8.319819         ..
2000.01.17| 1.10275e+07    11.49045       7850              9.464308         ..
2000.01.18| 8942500        8.282222       9500              6.880915         ..
2000.01.19| 7775000        12.43864       10200             9.068333         ..
2000.01.20| 1.20175e+07    14.59714       9400              9.383993         ..
..
```

!!!warning
        It is important to note that the time needed to calculate all the features when including the hyperparameterised functions can be extremely long. This is owed to an excess of 800 features being calculated per column, the functions being applied per ID and the complexity of some of the functions being applied. As such millions of calculations may need to be completed. Execution times can range from minutes to hours based on data sizes and data complexity.


## Feature significance

Statistical-significance tests can be applied to the derived features, to determine how useful each feature is in predicting a target vector. The specific significance test applied, depends on the characteristics of the feature and target. The following table outlines the test applied in each case.

feature type       | target type       | significance test 
:------------------|:------------------|:------------------
Binary             | Real              | Kolmogorov-Smirnov
Binary             | Binary            | Fisher-Exact      
Real               | Real              | Kendall Tau-b     
Real               | Binary            | Kolmogorov-Smirnov

1. The Benjamini-Hochberg-Yekutieli (BHY) procedure: determines if the feature meets a defined False Discovery Rate (FDR) level (set at 5% by default).
2. K-best features: choose the features which have the lowest p-values and thus K-most important features to prediction of the target vector.
3. Percentile based selection: set a percentage threshold for p-values below which features are selected.
 
Each of these procedures are accessed as shown in the below examples, where data being used is the output of the create features procedure, completed in the previous section.

### Benjamini Hochberg

The selection of features based on the Benjamini-Hochberg procedure is completed using the following function:

Syntax: `.ml.fresh.benjhochfeat[table;targets]`

Where

-   `table` is the value section of the table produced by the feature-creation procedure
-   `targets` is a list of target values corresponding to the IDs 

returns a list of the features deemed statistically significant.

```q
q)target:value exec avg col2+.001*col2 by date from tab / combination of col avgs
q)show sigfeats:.ml.fresh.benjhochfeat[value features;target]  / threshold defaulted to 5%
`mean_col2`sumval_col2`absenergy_col2`c3_1_col2`c3_2_col2`med_col2`quantile_0..
q)count 2_cols tab      / number of raw features
2
q)count 1_cols features / number of extracted features
260
q)count sigfeats        / number of selected features
21
```

### K-best feature Selection

In line with the other two procedures, this selection method makes a comparison between the features and targets and calculates p-values. In this case, the user defines the number of features that will be taken as output. These are the K-best features which best reject the null hypothesis.

Syntax: `.ml.fresh.ksigfeat[table;targets;k]`

Where

-  `table` is the value section of the table, produced by the feature creation procedure
-  `targets` is a list of target values corresponding to the IDs
-  `k` is the number of features to be chosen

returns a list of the K-best features.

```q
q)show sigfeats:.ml.fresh.ksigfeat[value features;target;2]  / find the best 2 features
`mean_col2`sumval_col2
q)count sigfeats        / number of selected features
2
```
###Percentile based Feature Selection

Syntax: `.ml.fresh.percentilesigfeat[table;targets;p]`

Where

- `table` is the value section of the table produced by the feature-creation procedure
- `targets` is a list of target values corresponding to the IDs
- `p` is the percentage threshold for p-values below which features are selected

returns a list of the features deemed statistically significant.

```q
q)show sigfeats:.ml.fresh.percentilesigfeat[value features;target;.05]  / set the percentile to be 5%
`absenergy_col2`mean_col2`med_col2`sumval_col2`c3_1_col2`c3_2_col2`c3_3_col2`..
q)count sigfeats        / number of selected features
8
```



## Fine tuning

### Parameter dictionary

Hyperparameters for a number of the functions are contained in the dictionary `.ml.fresh.paramdict` (defined in the script `paramdict.q`). The default dictionary can be modified by users to suit their use cases better.


### User-defined functions

The aggregation functions contained in this library are a small subset of the functions that could be applied in a feature-extraction pipeline. Users can add their own functions by following the template outlined within `fresh.q`.


##Peaching Data

When running multiple slave threads on large datasets, `peach` can be used when creating multiple features, as `.ml.fresh.createfeatures` is a computationally expensive function. This is done by loading `distrib_fresh.q`.

```q
$ q -s -4
q)\l distrib_fresh.q
q)m:30;n:100
q)show tab:([]date:raze m#'"d"$til n;col1:50*1+(m*n)?20;col2:(m*n)?1f;col3:501+(m*n)?20;
  col4:(m*n)?1f;col5:501+(m*n)?20;col6:(m*n)?1f;col7:501+(m*n)?20;col8:(m*n)?1f)
date       col1 col2       col3 col4      col5 col6       col7 col8      
-------------------------------------------------------------------------
2000.01.01 550  0.1668385  506  0.9828995 513  0.377193   510  0.3128276 
2000.01.01 200  0.8142935  509  0.6742066 509  0.5825595  513  0.7769104 
2000.01.01 100  0.406557   519  0.8448589 510  0.5055341  507  0.4385946 
2000.01.01 500  0.3976431  517  0.589426  509  0.03360174 519  0.4270084 
2000.01.01 850  0.1678661  514  0.6153198 501  0.4381629  515  0.08262207
2000.01.01 900  0.9123221  507  0.7526229 516  0.212853   512  0.2900495 
2000.01.01 550  0.1994978  519  0.5585663 516  0.9083837  518  0.50071   
2000.01.01 1000 0.5864429  512  0.9486947 513  0.6865164  509  0.61231   
2000.01.01 1000 0.251346   508  0.2394833 512  0.03327776 511  0.5744397 
2000.01.01 350  0.2218106  514  0.5546665 503  0.1300084  508  0.5358078 
2000.01.01 350  0.9309829  514  0.2490826 515  0.1692926  507  0.2077364 
2000.01.01 750  0.1268818  508  0.1526625 505  0.2228439  508  0.9892383 
2000.01.01 600  0.03473344 509  0.4186494 509  0.2436332  512  0.8610513 
2000.01.01 450  0.9681282  512  0.7933249 503  0.5810057  504  0.2795273 
2000.01.01 500  0.6101471  515  0.4722181 506  0.8917662  518  0.2734299 
2000.01.01 450  0.7355978  514  0.7174564 514  0.5869287  518  0.3045341 
2000.01.01 250  0.5817326  513  0.7864528 511  0.1290175  515  0.9413026 
2000.01.01 100  0.2461874  511  0.230709  506  0.8149661  520  0.07071531
2000.01.01 550  0.5526173  506  0.104398  502  0.205728   506  0.5804658 
2000.01.01 1000 0.663168   502  0.1531194 517  0.6852129  509  0.05064922
q)\ts .ml.fresh.peachcreatefeatures[tab;`date;1]
51
q)\t .ml.fresh.createfeatures[tab;`date;1_cols tab;0b]
95
```

---
layout: post
title: "Evaluating a Linear Regression Model"
subtitle: "Outlier or Caitlin Clark? [Part 8]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, linear regression]
share-title: "Evaluating a Linear Regression Model: Outlier or Caitlin Clark? [Part 8]" 
share-description: Interested in evaluating the performance of a linear regression model on your own data? Learn how to evaluate a linear regression machine learning model in the latest installment of this data science series that is perfect for beginner data scientists and Python enthusiasts.
thumbnail-img: /assets/img/posts/2024-11-27-basketball-evaluate-ols-model/thumbnail.png
share-img: /assets/img/posts/2024-11-27-basketball-evaluate-ols-model/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll wrap up the basics of machine learning by examining how to evaluate the performance of our linear regression model. This is the eighth part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

<div id="toc"></div>

# Getting Started
First, let's take a look at an overview of this data science project. If you're already familiar with it, feel free to skip to the [next section](#basics-of-machine-learning).

## Project Overview

As a reminder, the dataset we'll be using in this project contains individual basketball player statistics (such as total points scored and blocks made) for the 2023-2024 NCAA women's basketball season. Here's a brief description of each major step of this project: 

![the steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/project_steps.png "the steps for this data science project")

1. **Data Acquisition** - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. **Data Cleaning** - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. **Data Preprocessing** - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. **Feature Engineering** - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.
5. **Data Exploration** - This step focuses on analyzing and visualizing the dataset to uncover patterns, relationships, and general trends and is a helpful preliminary step before deeper analysis.
6. **Creating Visualizations** - This step involves identifying the relationships between various parameters (such as height and blocked shots) and generating meaningful visualizations (such as bar charts, scatterplots, and candlestick charts).
7. **Machine Learning** - This step focuses on selecting, training, and evaluating a machine learning model. For this project, the model will identify the combination of individual player statistics that correlates with optimal performance.

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Due to the size of this project, today we'll cover part of the seventh step: evaluating a machine learning model.

## Dependencies
Since this is the eighth installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first article of the series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. 

## Import Packages
You'll want to have the latest version of [Python](https://www.python.org/) installed with the following packages: 
  - [pandas](https://pandas.pydata.org/docs/)
  - [requests](https://requests.readthedocs.io/en/latest/)
  - [json](https://docs.python.org/3/library/json.html)
  - [os](https://docs.python.org/3/library/os.html)
  - [numpy](https://numpy.org/doc/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
  - [plotly](https://plotly.com/)
  - [scikit-learn](https://scikit-learn.org/stable/index.html)
  - [joblib](https://joblib.readthedocs.io/en/stable/)
  - [statsmodels](https://www.statsmodels.org/stable/index.html)
  
For today's machine learning sgement specifically, we'll want to import a few of these libraries: 


```python
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
```

## Import Data
In the [previous part](/2024-09-13-basketball-train-ols/) of this series, we created our training and testing splits from the `player_data` dataframe. If you want to follow along with the code samples in this article, it's recommended to import the testing splits before proceeding.

*Note: To reduce confusion, the variable names in this article are slightly different than in the previous article. Since the model initially trained used the full set of features, variable names for that model will be appended with `_full`. Since the alternate model trained used fewer features (`FIELD_GOALS_MADE`, `TWO_POINTS_MADE`, and `POINTS` were all removed), variable names for that model will be appended with `_few` instead of `_alt`. For example, `X_test` is now `X_test_full` and `X_test_alt` is now `X_test_few`.* 


```python
X_test_full = pd.read_csv('X_test_full.csv')
X_test_full.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>MINUTES_PLAYED</th>
      <th>FIELD_GOALS_MADE</th>
      <th>THREE_POINTS_MADE</th>
      <th>TWO_POINTS_MADE</th>
      <th>FREE_THROWS_MADE</th>
      <th>TOTAL_REBOUNDS</th>
      <th>ASSISTS</th>
      <th>TURNOVERS</th>
      <th>STEALS</th>
      <th>BLOCKS</th>
      <th>FOULS</th>
      <th>POINTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71</td>
      <td>821</td>
      <td>184</td>
      <td>43</td>
      <td>141</td>
      <td>80</td>
      <td>155</td>
      <td>36</td>
      <td>90</td>
      <td>51</td>
      <td>5</td>
      <td>46</td>
      <td>491</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71</td>
      <td>840</td>
      <td>61</td>
      <td>7</td>
      <td>54</td>
      <td>45</td>
      <td>221</td>
      <td>40</td>
      <td>41</td>
      <td>34</td>
      <td>9</td>
      <td>63</td>
      <td>174</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68</td>
      <td>961</td>
      <td>100</td>
      <td>36</td>
      <td>64</td>
      <td>75</td>
      <td>120</td>
      <td>107</td>
      <td>84</td>
      <td>27</td>
      <td>1</td>
      <td>79</td>
      <td>311</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73</td>
      <td>1060</td>
      <td>231</td>
      <td>77</td>
      <td>154</td>
      <td>105</td>
      <td>167</td>
      <td>59</td>
      <td>105</td>
      <td>26</td>
      <td>45</td>
      <td>56</td>
      <td>644</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>814</td>
      <td>112</td>
      <td>2</td>
      <td>110</td>
      <td>42</td>
      <td>208</td>
      <td>33</td>
      <td>60</td>
      <td>33</td>
      <td>24</td>
      <td>37</td>
      <td>268</td>
    </tr>
  </tbody>
</table>
</div>




```python
X_test_few = pd.read_csv('X_test_few.csv')
X_test_few.head()
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>Height</th>
      <th>MINUTES_PLAYED</th>
      <th>THREE_POINTS_MADE</th>
      <th>FREE_THROWS_MADE</th>
      <th>TOTAL_REBOUNDS</th>
      <th>ASSISTS</th>
      <th>TURNOVERS</th>
      <th>STEALS</th>
      <th>BLOCKS</th>
      <th>FOULS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>71</td>
      <td>821</td>
      <td>43</td>
      <td>80</td>
      <td>155</td>
      <td>36</td>
      <td>90</td>
      <td>51</td>
      <td>5</td>
      <td>46</td>
    </tr>
    <tr>
      <th>1</th>
      <td>71</td>
      <td>840</td>
      <td>7</td>
      <td>45</td>
      <td>221</td>
      <td>40</td>
      <td>41</td>
      <td>34</td>
      <td>9</td>
      <td>63</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68</td>
      <td>961</td>
      <td>36</td>
      <td>75</td>
      <td>120</td>
      <td>107</td>
      <td>84</td>
      <td>27</td>
      <td>1</td>
      <td>79</td>
    </tr>
    <tr>
      <th>3</th>
      <td>73</td>
      <td>1060</td>
      <td>77</td>
      <td>105</td>
      <td>167</td>
      <td>59</td>
      <td>105</td>
      <td>26</td>
      <td>45</td>
      <td>56</td>
    </tr>
    <tr>
      <th>4</th>
      <td>74</td>
      <td>814</td>
      <td>2</td>
      <td>42</td>
      <td>208</td>
      <td>33</td>
      <td>60</td>
      <td>33</td>
      <td>24</td>
      <td>37</td>
    </tr>
  </tbody>
</table>
</div>



As a reminder, we previously created testing splits named `y_test` and `y_test_alt` for the target `FANTASY_POINTS` variable. However, those two splits contained identical data, so we'll use a single testing split called `y_actual` for the target variable for both models instead. 


```python
y_actual = np.genfromtxt('y_actual.csv', delimiter=',', skip_header=True)
y_actual
```




    array([ 753. ,  544.2,  587.5,  969.9,  621.1,  594.4,  554.5,  808. ,
            884.2,  838.2,  901.6,  797.2, 1314.6,  474.1,  552.4,  687. ,
            514.9,  499. ,  886.9,  189.9,  697.1,  325.8,  465.8,  569.9,
            793.6,  691.4,  590.6,  661.2,  920. ,  643.1,  557.4,  634.1,
            562. ,  542.6,  848.8,  283. , 1218.9,  698.5,  476.1,  694. ,
            675.5,  638.8,  634.4,  646.9,  696.2,  611.3,  777.1,  335.3,
            430.7,  664.6,  604.9,  534.5,  860.9,  655.1,  478.8,  584. ,
            636.9,  787.2,  375.1,  622.7,  465.6,  545.4,  712.7,  398. ,
            538.5,  742.9,  559. ,  476.5,  395. ,  463.3,  568.3,  890.3,
            619. ,  582.4,  705.7,  690.6, 1027.6,  602.5,  540.3,  560.9,
            423.4,  653.3, 1171.8,  868.5,  526.8,  730. ,  834. ,  547.4,
            719.2,  765.3,  676.5,  826.8,  845. ,  361. ,  723.3,  372.7,
            876.9,  570.1,  708.8,  720.2,  780.5,  901.9,  489.8,  583.7,
            702. ,  769.6,  557.1,  595.5,  417.6,  799.9,  727.5,  960.4,
            430.6,  659.7,  499.6,  327.8,  870.2,  806.4,  550.4,  396.3,
            521.2,  447.3,  809.9,  561.6,  680.2,  446.6,  332.9,  495.2,
            823. ,  820.7,  706.4,  811.6, 1119. ,  329. ,  783.7,  787.9,
            737.3,  494.5,  508.3,  478. , 1182.3,  672.5,  733.2,  733.1,
            615.6,  559.6,  807.1,  728.8,  751.1,  864.1,  543.3,  737.3,
            986.7,  494.9,  639.8,  597.6,  612.5,  572.7,  709.4,  487.6,
            523.5,  484.3,  686.7,  815.9,  699.4,  614. ,  651.1,  576. ,
            832.7,  802. ,  974.1,  365.3,  656.1,  578.1,  444.2,  813.7,
            670.3,  746. ,  714.4,  473.9,  635.3,  435.9,  635.1,  773.5,
            412.3,  723.1,  464. ,  760.4,  532. ,  723.9,  514.2,  790.7,
            392.3,  649.4,  814.3,  951.3,  336.1,  714.6,  602.2,  429.6,
            652.1,  698.3,  577.1,  708.4,  966.5,  770.1,  638.1,  641.9,
            671.8, 1267.4,  757.2,  908.6,  646.3,  797.9,  758.8,  624. ,
            639.1,  769. ,  451.1,  643.5,  734.2,  545.7,  603.6,  858.6])



## Import Models
In the [previous article](/2024-09-13-basketball-train-ols/) of this series, we trained two machine learning models. If you want to follow along with the code samples in this article, it's recommended to import both of those models before proceeding.


```python
model_full = joblib.load('model_full.sav')
model_few = joblib.load('model_few.sav')
```

## Set Graph Preferences
This is an entirely optional step to configure the aesthetics of the graphs and charts. You can import a custom color scheme or set colors individually. In this case, we’ll define a list of custom colors (`graph_colors`) and configure both Matplotlib and Seaborn to use these colors for subsequent plots.


```python
graph_colors = ['#615EEA', '#339E7A', '#ff8c6c']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=graph_colors)
sns.set_palette(graph_colors)
```

We can also set the overall style for the matplotlib plots using a style sheet. You can print the list of available style sheets and view examples of these style sheets on [matplotlib’s website](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).


```python
plt.style.use('seaborn-v0_8-white')
```

## Basics of Machine Learning

Before we get into training a model, let’s briefly revisit a few basics of machine learning. If you are already familiar with these concepts, feel free to skip to the [next section](#Model-Training). [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a branch of artificial intelligence that focuses on creating algorithms and statistical models that allow computer systems to "learn" how to improve their performance on a specific task through experience. In the context of our basketball statistics project, machine learning can be particularly useful for predicting player performance, classifying player position, and identifying similar players.

Key concepts in machine learning that we'll encounter include:

1. **Model** - The system that learns patterns from data and can be used to make predictions on previously unseen data. Machine learning models are often of a specific type (Linear or Logistic Regression, Random Forests, Support Vector Machines, Neural Networks, etc.). Today's model is a Linear Regression model. 
2. **Training Data** - The subset of our data used to train the model.
3. **Testing Data** - A separate subset of data used to evaluate the model's performance. 
4. **Features** - The input variable(s) used to make predictions. These are sometimes referred to as the independent variable(s) or the predictor(s). For this project, these are various player statistics like three points made and assists. 
5. **Target Variable** - The variable we're trying to predict or optimize. This is sometimes referred to as the dependent variable(s), as it depends on the independent variable(s). In today's project, this is Fantasy Points.
6. **Parameters** - The values that the model learns during training, such as coefficients in linear regression. These parameters define how the model transforms input features into predictions.
7. **Hyperparameters** - The configuration settings for the model that are set before training begins. These are not learned from the data but are specified by the data scientist. Examples include learning rate, number of iterations, or regularization strength. Hyperparameters can significantly affect model performance and are often tuned to optimize the model. 
    - *Note*: The model we’ll be using today is straightforward and doesn’t typically have hyperparameters in the traditional sense. However, it’s still important to know the difference between parameters and hyperparameters since many models will have hyperparameters. 
8. **Residuals** - The differences between the observed values and the predicted values from the model. Residuals help assess how well the model fits the data and can reveal patterns or issues in the model's predictions.
9. **Model Evaluation** - Metrics used to assess how well our model is performing. For a Linear Regression model, this will include metrics like Mean Squared Error (MSE) and the R-squared value.

We’ll use most of these terms throughout this article, so it’s best to familiarize yourself with them now. Hyperparameters and additional machine learning concepts will be explored in more detail in future articles (please [let me know](/workwithme/) if that is something you are interested in!). 

Note: Our focus in this article is on classic machine learning models designed for tabular data. We won't be covering models built specifically for natural language processing, image recognition, or video analysis. However, it's worth mentioning that many problems in these domains often get transformed into tabular data problems, so some of the principles we discuss here may still apply in those contexts. With all of that out of the way, let’s move on to evaluating our machine learning model.

# Generate Predictions
To evaluate the model's performance, we need to compare the values that the model predicts to the actual values (sometimes referred to as the "ground truth" values). Models that predict values close to the actual values perform better, and models that predict values far from the actual values perform worse. There are various evaluation metrics we can calculate using these predictions to quantify how well a model performs, but the first step is generating the predictions for each model. 

To generate predictions, we'll apply each trained model to the testing data split. We'll use the testing data split instead of the training data split to ensure that the model is evaluated on data that it hasn't seen during training. (For a refresher on *why* we split the dataset into training and testing subsets, see the [previous article](/2024-09-13-basketbal-train-ols/#create-training-and-testing-splits)).

In real-world settings, you'll likely want to use an optimized function like [sklearn's LinearRegression.predict()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) to generate predictions. Since this is a learning project, we'll generate the predictions in three ways: 
1. Manually
2. Using [np.dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
3. Using [LinearRegression.predict()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html)

Note: Since we're primarily working with NumPy arrays, we'll be using the optimized NumPy functions (`np.square()`, `np.mean()`) in most cases instead of the built-in Python syntax (`sum()`, `**2`). For example, `np.square()` is optimized for NumPy arrays and is generally faster for element-wise squaring of large arrays. The `** 2` operator works for both scalars and arrays but may be less efficient for large NumPy arrays. You're welcome to use either, but it's generally recommended to use the NumPy functions.

## Calculate Predictions Manually

Even if it might not be the most efficient method, we can manully calculate predictions. To understand how this works, let's start by looking at the model equation from the previous article: 


    FANTASY_POINTS = 1.3779112764770156e-14*Height
     + 1.1934897514720433e-15*MINUTES_PLAYED
     + 1.6666666666666634*FIELD_GOALS_MADE
     + 1.333333333333333*THREE_POINTS_MADE
     + 0.33333333333333315*TWO_POINTS_MADE
     + 1.0000000000000009*FREE_THROWS_MADE
     + 1.1999999999999982*TOTAL_REBOUNDS
     + 1.499999999999999*ASSISTS
     + -0.9999999999999992*TURNOVERS
     + 1.999999999999999*STEALS
     + 2.000000000000001*BLOCKS
     + -1.201296007113939e-15*FOULS
     + -7.233796894823286e-16*POINTS + -6.821210263296962e-13


As a reminder, this equation was assembled using the `.coef_`, `.feature_names_in_`, and `.intercept_` [attributes](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html). 

Now that we have the model equation, let's pull out a single row of data to use to calculate the model's prediction: 


```python
X_row = X_test_full.iloc[0][model_full.feature_names_in_]
X_row
```




    Height                71
    MINUTES_PLAYED       821
    FIELD_GOALS_MADE     184
    THREE_POINTS_MADE     43
    TWO_POINTS_MADE      141
    FREE_THROWS_MADE      80
    TOTAL_REBOUNDS       155
    ASSISTS               36
    TURNOVERS             90
    STEALS                51
    BLOCKS                 5
    FOULS                 46
    POINTS               491
    Name: 0, dtype: int64



We can manually calculate the model's prediction for this row by starting with the y-intercept and then adding the product of the feature and its corresponding coefficient:


```python
row_pred = model_full.intercept_
for feature, coef in zip(X_row, model_full.coef_):
    row_pred += feature * coef
row_pred
```




    753.0



Great! So for a player with the given `Height`, `MINUTES_PLAYED`, `FIELD_GOALS_MADE`, etc. (stored in `X_row`), the linear regression model (stored in `model_full`) predicts their Fantasy Points would be `753`.

## Calculate Predictions with .dot()

You might recognize the previous manual calculation used a [dot product](https://en.wikipedia.org/wiki/Dot_product), so we can also use NumPy's [.dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html) function to shorten this calculation:


```python
row_pred = model_full.intercept_ + np.dot(X_row, model_full.coef_)
row_pred
```




    753.0



This is the exact same value as the calculation in the previous step, so let's move on to the next method. 

## Calculate Predictions with .predict()

Now that we've calculated the prediction manually, let's see what it would look like by using [sklearn's LinearRegression.predict()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html#sklearn.linear_model.LinearRegression.predict) function: 


```python
row_pred = model_full.predict(X_test_full.iloc[[0]])
row_pred[0]
```




    753.0



This predicts the same value as the previous two methods. Feel free to test out each of these methods on other rows, but we'll move on to generating predictions for the entire test dataset:


```python
y_pred_full = model_full.predict(X_test_full)
y_pred_full[:5]
```




    array([753. , 544.2, 587.5, 969.9, 621.1])




```python
y_pred_few = model_few.predict(X_test_few)
y_pred_few[:5]
```




    array([689.05242335, 629.09942281, 640.65898995, 934.3165614 ,
           617.45450647])



We now have our predictions for each model and can see that the predicted values differ slightly between the two models. 

## Create Baseline Model
Before evaluating how well these models perform, let's create one more "model" as a baseline. This baseline model will simply predict the mean of the target variable in the training data and will serve as a simple reference point. By comparing the performance of the linear regression model to this naive approach, we can determine if the model is capturing meaningful patterns in the data and offering improvements beyond just predicting the average. 

To create predictions for this baseline model, we can create an array of the same size and type as our predictions or actual values using [NumPy's .full_like()](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy-full-like) function: 


```python
y_pred_base = np.full_like(y_actual, np.mean(y_actual))
y_pred_base[:5]
```




    array([658.009375, 658.009375, 658.009375, 658.009375, 658.009375])



Now we have all three of our predictions and can compare those to the actual values to evaluate how well each of the three models performs.

## Floating Point Errors
If you're not familiar with [floating-point errors](https://en.wikipedia.org/wiki/Floating-point_arithmetic#Accuracy_problems), this is a good opportunity to learn about them. As you might remember from the previous article, one of our models (`model_full`) predicts essentially the same vales as `y_actual` (the "correct" values). However, due to floating-point errors, these values will not evaluate as exactly identical. 

For example, let's look at the first value of `y_pred_full` and compare it to the first value of `y_actual`: 


```python
y_actual[0]
```




    753.0




```python
y_pred_full[0]
```




    753.0000000000001



If you calculated `y_pred_full[0]` by hand using the simplified model equation, you would get exactly `753.0`. However, due to how floating-point arithmetic and representation work in computing, certain coefficients are stored as `2.000000000000001` instead of `2` or `1.333333333333333` instead of \\(1.\bar{3}\\) (due to finite precision). This then introduces tiny variations in values that are not strictly equal in a programmatic sense, even if practically we might consider `753.0` and `753.0000000000001` as equal. 

If we do a direct equality comparison, for example: 


```python
y_actual[0] == y_pred_full[0]
```




    False



In cases like this where floating-point precision issues could be a factor, we can use [Numpy's .isclose()](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) function with the default tolerances: 


```python
np.isclose(y_pred_full[0], y_actual[0])
```




    True



*Note: The default tolerances are not suitable for numbers with magnitudes much smaller than one. In those cases, specify different `rtol` and `atol` limits.* 

*Note: This function also assumes that the second array is the "correct" reference value, which works fine in this case. However, for a more generalized, symmetric comparison, you might want to use [math.isclose()](https://numpy.org/doc/stable/reference/generated/numpy.isclose.html) instead. 

We can also use [NumPy's .allclose()](https://numpy.org/doc/stable/reference/generated/numpy.allclose.html) function to compare the entire arrays, instead of individual elements: 


```python
np.allclose(y_pred_full, y_actual)
```




    True



We can also apply this to the predictions to the other model: 


```python
np.allclose(y_pred_few, y_actual)
```




    False



This means that we can expect the predictions from `model_full` to be almost perfect (since they are all close to the actual values) and the predictions from `model_few` to be imperfect. We're now ready to move on to robustly evaluating each model's predictions! 

# Evaluate the Model
After training our linear regression model, the next crucial step is to evaluate its performance. This evaluation process helps us understand how well our model is doing, identify any issues, and determine if it's ready for real-world application or if it needs further refinement. In this section, we'll explore various metrics and techniques to assess our model's accuracy and reliability.

## Evaluation Metric Definitions

Let's start with a quick overview of each evaluation metric we'll be exploring today. 
- **[R-squared (R²)](https://en.wikipedia.org/wiki/Coefficient_of_determination)** - This measures the proportion of variance explained by the model. It gives a good first glance at how much of the variability in the target (Fantasy Points in this case) is explained by the model. It's also referred to as the coefficient of determination.
- **[Adjusted R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2)** - This is similar to R-squared, but is adjusted for the number of predictors to account for overfitting. This is often useful when comparing multiple models with different numbers of features (as is the case between `model_full` and `model_few`).
- **[Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error)** - This is the average of the squared differences between predicted and actual values. It indicates the model's prediction accuracy, penalizing larger errors more heavily. 
- **[Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root_mean_square_deviation)** - This is the square root of MSE. It provides a more interpretable measure of prediction accuracy than MSE since it is in the same units and scale as the target variable. It helps understand the magnitude of the average prediction error. 
- **[Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error)** - This is the average absolute difference between predicted and actual values. It is also a measure of the model's prediction accuracy, but it penalizes errors equally (instead of penalizing larger errors like MSE) and is less sensitive to outliers as a result. 
- **[Residuals](https://en.wikipedia.org/wiki/Errors_and_residuals)** - These are the difference between the predicted values and the actual values. These help assess the accuracy of the model by potentially revealing patterns or biases in the model. These are usually plotted and analyzed visually. 

*Note that there are additional metrics that can be used to evaluate and compare Linear Regression models (such as [Variance Inflation Factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) and [F-test](https://en.wikipedia.org/wiki/F-test)), but the metrics covered today are commonly used and will serve as a good starting point.*

Similar to how we generated predictions manually and then with scikit-learn's function, we'll also calculate each of these evaluation metrics using multiple methods. 

## Define Variables
Before jumping into the evaluation metrics, let's define a few helpful variables: 

- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)
- \\(\bar{y}\\) is the mean of the actual values (`mean(y_actual)`)
- \\(n\\) is the number of data points (`len(y_actual)`)

The variables for \\(y_i\\) and \\(\hat{y}\\) are already defined, so let's calculate \\(n\\) and \\(\bar{y}\\) next. 

### Calulate \\(n\\)
This step is pretty straightforward, since the number of data points in this context can be calculated by looking at the length of the testing data. You could use any of the variables from the testing dataset (`X_test_full`, `y_pred_few`, etc.), but we'll simply use `y_actual`: 


```python
n = len(y_actual)
n
```




    224



### Calculate the Mean
The [mean](https://en.wikipedia.org/wiki/Mean) will be used to calculate a few of the evaluation metrics, so let's take this opportunity to calculate it. 

#### Equation
Let's start by looking at the equation for the arithmetic mean: 

$$
\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$

where:
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\bar{y}\\) is the mean of the actual values (`mean(y_actual)`)
- \\(n\\) is the number of data points (`len(y_actual)`)

*Note: If you'd like a refresher on \\(\Sigma{}\\) notation, [the summation Wikipedia page](https://en.wikipedia.org/wiki/Summation#Capital-sigma_notation) is a good resource.*

#### Calculate Manually
Since the mean is the sum of \\(y_i\\) divided by \\(n\\), this is simple to calculate manually: 


```python
y_mean = np.sum(y_actual) / n
y_mean
```




    658.009375



#### Calculate with NumPy
We can also use the more efficient [NumPy .mean()](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html) function: 


```python
y_mean = np.mean(y_actual)
y_mean
```




    658.009375



Both of these methods produce the same value, so we can move on to the next variable. 

### Calculate Residuals
The last variable we'll calculate before getting into the evaluation metrics is the [residuals](https://en.wikipedia.org/wiki/Errors_and_residuals). In this context, the residuals is the difference between \\(y_i\\) and \\(\hat{y}\\). 

#### Equation

$$
\text{residuals} = y_i - \hat{y}
$$

where:
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)

#### Calculate Manually
We can calculate the residuals for each model: 


```python
residuals_full = y_actual - y_pred_full
residuals_full[:5]
```




    array([-1.13686838e-13,  0.00000000e+00, -1.13686838e-13, -2.27373675e-13,
           -2.27373675e-13])




```python
residuals_few = y_actual - y_pred_few
residuals_few[:5]
```




    array([ 63.94757665, -84.89942281, -53.15898995,  35.5834386 ,
             3.64549353])




```python
residuals_base = y_actual - y_pred_base
residuals_base[:5]
```




    array([  94.990625, -113.809375,  -70.509375,  311.890625,  -36.909375])



#### Evaluation
We'll be plotting and analyzing these residuals in a later step, so for now we'll use these variables to calculate a few of the evaluation metrics.

<div class="email-subscription-container"></div>

## \\(R^2\\)
\\(R^2\\), also known as the [coefficient of determination](https://en.wikipedia.org/wiki/Coefficient_of_determination), is a useful metric for evaluating regression models. It represents the proportion of variance in the dependent variable (Fantasy Points) that is predictable from the independent variable(s) (Height, Points, Steals, etc.). 

As a proportion, \\(R^2\\) values usually range from `0` to `1`, with higher values indicating better fit. For example: 

- `0` indicates that the model explains **none** of the variability of the target variable.
  - This means the model's predictions are no better than simply predicting the mean of the target variable.
  - This suggests a poor fit or that there is no useful relationship between the target and the independent variables.
- `1` indicates that the model explains **all** the variability of the target variable.
  - This means the model's predictions perfectly match the target variable. 
  - This suggests either an ideal fit or that the model is overfit. 
- `0.5` indicates that the model explains **some** of the variability of the target variable. 
  - This means the model's predictions somewhat match the target variable and performs better than predicting the mean.
  - Values between `0` and `1` are common in real-world scenarios. \\(R^2\\) values closer to `1` indicate a better fit than values closer to `0`.

In other words, R-squared provides a measure of how well observed outcomes are replicated by the model, based on the proportion of total variation of outcomes explained by the model. 

If this is your first time learning about \\(R^2\\), [Newcastle University](https://www.ncl.ac.uk/webtemplate/ask-assets/external/maths-resources/statistics/regression-and-correlation/coefficient-of-determination-r-squared.html) has an excellent step-by-step walkthrough of how to calculate \\(R^2\\) by hand with visuals.

### Equation
Let's start by looking at the equation for \\(R^2\\):

$$
R^2 = 1 - \frac{\text{RSS}}{\text{TSS}}
$$

where: 
- \\(\text{RSS}\\) is the [residual sum of squares](https://en.wikipedia.org/wiki/Residual_sum_of_squares)
- \\(\text{TSS}\\) is the [total sum of squares](https://en.wikipedia.org/wiki/Total_sum_of_squares)

\\(\text{RSS}\\) can be calculated by: 

$$
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where: 
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)
- \\(n\\) is the number of data points (`len(y_actual)`)

Referring to the equation for \\(\text{residuals}\\) from earlier in this article, the equation for \\(\text{RSS}\\) can also be written as:

$$
\text{RSS} = \sum_{i=1}^{n} (\text{residuals}_\text{model})^2
$$

\\(\text{residuals}_\text{model}\\) in the equation above represents the residuals for each model, so there will be one \\(\text{RSS}\\) for `residuals_full`, another for `residuals_few`, and a third for `residuals_base`. 


\\(\text{TSS}\\) can be calculated by: 

$$
\text{TSS} = \sum_{i=1}^{n} (y_i - \bar{y})^2
$$

where:
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\bar{y}\\) is the mean of the actual values (`mean(y_actual)`)
- \\(n\\) is the number of data points (`len(y_actual)`)

Using the \\(\text{residuals}\\) equation from earlier, we can make a substition for \\(\text{TSS}\\) as well: 

$$
\text{TSS} = \sum_{i=1}^{n} (\text{residuals}_\text{base})^2
$$

### Calculate Manually
The \\(\text{TSS}\\) will be the same for each of our three models, so let's start by calculating that: 


```python
tss = np.sum(np.square(residuals_base))
```

Next, we can calculate the \\(\text{RSS}\\) for each of the model's residuals: 


```python
rss_full = np.sum(np.square(residuals_full))
rss_full
```




    8.766983582068369e-24




```python
rss_few = np.sum(np.square(residuals_few))
rss_few
```




    580028.3960045542




```python
rss_base = np.sum(np.square(residuals_base))
rss_base
```




    7316716.2503125



Lastly, we can use each \\(\text{RSS}\\) and the \\(\text{TSS}\\) to calculate the \\(R^2\\) for each model: 


```python
r2_full = 1 - (rss_full / tss)
r2_full
```




    1.0




```python
r2_few = 1 - (rss_few / tss)
r2_few
```




    0.920725585609558




```python
r2_base = 1 - (rss_base / tss)
r2_base
```




    0.0



### Calculate with scikit-learn
We can also calculate \\(R^2\\) using [scikit-learn's .r2_score()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) function for each model: 


```python
r2_full = r2_score(y_actual, y_pred_full)
r2_full
```




    1.0




```python
r2_few = r2_score(y_actual, y_pred_few)
r2_few
```




    0.920725585609558




```python
r2_base = r2_score(y_actual, y_pred_base)
r2_base
```




    0.0



These values exactly match what we calculated manually, so we're ready to move on to evaluating the \\(R^2\\) results. 

### Evaluation
Now that we have our results for each model, let's take a look at how each model performs: 

- `model_full`: 1.0
- `model_few`:  0.92...
- `model_base`: 0

As mentioned earlier, a higher \\(R^2\\) generally indicates a better fit for the model, with `1` being a perfect fit and `0` being a poor fit that performs no better than predicting the mean. Since we already know that `model_full` correctly predicts the `y_actual` for each data point, it makes sense that \\(R^2\\) is `1.0` for this model. On the other end, it also makes sense that `model_base` has a \\(R^2\\) of `0`, since this model is predicting the mean for each observation. `model_few` has a \\(R^2\\) of `0.92...`, which is relatively close to `1`, so this model has a fairly good fit. 

## Adjusted \\(R^2\\)
[Adjusted \\(R^2\\)](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2) is a modification to the standard \\(R^2\\) that we just calculated that adjusts for the number of predictors (Height, Points, Steals, etc.) in the model. Standard \\(R^2\\) will always increase as you add more predictors (even if they aren't improving the model), which can make the results a bit misleading for models with many predictors. Adjusted  \\(R^2\\) penalizes the addition of unnecessary predictors, so it provides a more accurate measure of the model's performance when there are multiple predictors. This also makes it quite useful for comparing models with different numbers of predictors.

Adjusted \\(R^2\\) is similar to standard \\(R^2\\) in that it values closer to `1` indicate a good fit, and values closer to (or below) `0` indicate a poor fit. Adjusted \\(r^2\\) can also be below zero in cases of poorly fitted models or when \\(p\\) is much greater than \\(n\\).

### Equation
Let's start by looking at the equation for Adjusted \\(R^2\\): 

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)
$$

where 
- \\(R^2\\) is the regular coefficient of determination
- \\(n\\) is the number of data points
- \\(p\\) is the number of predictors (independent variables)

### Calculate Manually
We already have variables for \\(R^2\\) and \\(n\\), so let's begin the manual calculation by defining \\(p\\). Scikit-learn's LinearRegression models have an attribute called `n_features_in_` that returns the number of features seen when fitting the model, so we can use that: 


```python
p_full = model_full.n_features_in_
p_full
```




    13




```python
p_few = model_few.n_features_in_
p_few
```




    10



The baseline model always predicts the same value (the mean of the target variable), so it doesn't use any features to make predictions. This means we can set \\(p\\) to `0` for this model: 


```python
p_base = 0
p_base
```




    0




```python
adj_r2_full = 1 - ((1 - r2_full) * (n - 1)) / (n - p_full - 1)
adj_r2_full
```




    1.0




```python
adj_r2_few = 1 - ((1 - r2_few) * (n - 1)) / (n - p_few - 1)
adj_r2_few
```




    0.9170037821170489




```python
adj_r2_base = 1 - ((1 - r2_base) * (n - 1)) / (n - p_base - 1)
adj_r2_base
```




    0.0



### Calculate with scikit-learn

At time of writing, there isn't a built-in function in scikit-learn to calculate Adjusted \\(R^2\\), so we'll move on to the next metric.

### Evaluation
There is no difference between Adjusted \\(R^{2}\\) and standard \\(R^{2}\\) for `model_full` or `model_base`. However, the adjusted \\(R^{2}\\) is slightly lower than standard \\(R^{2}\\) for `model_few`.

## Mean Squared Error (MSE)

[Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error) is another common metric for evaluating regression models. It calculates the average of the squared differences between predicted values (\\(\hat{y}\\)) and actual values (\\(y\\)). Since MSE squares the errors, it can be more sensitive to outliers and less interpretable than other metrics, so it's particularly useful when you want to heavily penalize large prediction errors. 

- `0` indicates a the model makes perfect predictions
- Values close to `0` indicate a better fit
- Larger values indicate a worse fit, but there is no upper bound


### Equation
Let's start by looking at the equation for MSE:


$$
\text{MSE} = \frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

where: 
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)
- \\(n\\) is the number of data points (`len(y_actual)`)

You might notice that part of this equation is the same as the calculation for \\(RSS\\) that we used to compute \\(R^2\\) earlier: 

$$
\text{RSS} = \sum_{i=1}^{n} (y_i - \hat{y}_i)^2
$$

We can substitute that in to rewrite this equation as: 

$$
\text{MSE} = \frac{\text{RSS}}{n}
$$

### Calculate Manually
Since we already calculated \\(RSS\\) in a previous step, we could re-use that value, but let's calculate MSE from scratch for each model for completeness.


```python
mse_full = np.sum(np.square(residuals_full)) / n
mse_full
```




    3.9138319562805214e-26




```python
mse_few = np.sum(np.square(residuals_few)) / n
mse_few
```




    2589.4124821631885




```python
mse_base = np.sum(np.square(residuals_base)) / n
mse_base
```




    32663.91183175223



### Calculate Manually (Alternate Method)
Now let's calculate MSE for each model using the \\(RSS\\) values computed previously. 


```python
mse_full = rss_full / n
mse_full
```




    3.9138319562805214e-26




```python
mse_few = rss_few / n
mse_few
```




    2589.4124821631885




```python
mse_base = rss_base / n
mse_base
```




    32663.91183175223



### Calculate with scikit-learn
Now that we've finished the manual calculations, we can also use [scikit-learn's mean_squared_error()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) function. 


```python
mse_full = mean_squared_error(y_actual, y_pred_full)
mse_full
```




    3.9138319562805214e-26




```python
mse_few = mean_squared_error(y_actual, y_pred_few)
mse_few
```




    2589.4124821631885




```python
mse_base = mean_squared_error(y_actual, y_pred_base)
mse_base
```




    32663.91183175223



### Evaluation
We ended up with the same values for all three models with each calculation method, so let's evaluate the results. 

- `mse_full`: \\(3.91... \times 10^{-26}\\)
- `mse_few` : \\(2589.412...\\)
- `mse_base`: \\(32663.911...\\)

Note that all of these results are in the units of fantasy points-squared. As mentioned earlier, a MSE closer to `0` is better, so it makes sense that the `model_full` performs the best. `mse_few` is in between the values of `mse_full` and `mse_base`, with `mse_base` being over 10x larger than `mse_few`. The results are somewhat similar to that of \\(R^2\\), but a bit less interpretable, so let's move on to the next metric.

## Root Mean Squared Error (RMSE) 
[Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root_mean_square_deviation) is the square root of the MSE and is helpful for determining the average magnitude of the error. It's similar to MSE in many ways, including being sensitive to outliers. However, RMSE is often preferred over MSE because it's in the same units as the target variable, making it easier to interpret.

### Equation
The full equation for RMSE is: 

$$
\text{RMSE} = \sqrt{\frac{1}{n} \sum_{i=1}^{n} (y_i - \hat{y}_i)^2}
$$

where: 
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)
- \\(n\\) is the number of data points (`len(y_actual)`)

Since RMSE is the square root of MSE, this equation can also be written as:

$$
\text{RMSE} = \sqrt{\text{MSE}}
$$

### Calculate Manually
Since we have already calculated MSE, we can get RMSE simply by taking the square root of MSE. 


```python
rmse_full = np.sqrt(mse_full)
rmse_full
```




    1.9783407078358676e-13




```python
rmse_few = np.sqrt(mse_few)
rmse_few
```




    50.88627007517046




```python
rmse_base = np.sqrt(mse_base)
rmse_base
```




    180.7316016410861



### Calculate with scikit-learn
Scikit-learn also provides the [root_mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html) function to calculate RMSE directly that we can apply to the predictions from each model. 


```python
rmse_full = root_mean_squared_error(y_actual, y_pred_full)
rmse_full
```




    1.9783407078358676e-13




```python
rmse_few = root_mean_squared_error(y_actual, y_pred_few)
rmse_few
```




    50.88627007517046




```python
rmse_base = root_mean_squared_error(y_actual, y_pred_base)
rmse_base
```




    180.7316016410861



### Evaluation
For RMSE, values closer to zero are better, so it's no surprise that the RMSE for `model_full` is almost zero. `rmse_few` is closer to zero than `rmse_base`, but we can also evaluate these quantities within the context of the target values. 


```python
print(f'y_actual values range from {np.amin(y_actual)} to {np.amax(y_actual)}')
```

    y_actual values range from 189.9 to 1314.6



```python
np.mean(y_actual)
```




    658.009375



In this case, the target variable and its mean are on the order of hundreds, so a RMSE of 50.8 for `model_few` seems fairly good, while the RMSE of nearly 200 for `model_base` is quite poor. 

## Mean Absolute Error (MAE)
[Mean Absolute Error (MAE)]() measures the average magnitude of errors in a set of predictions, without considering their direction. It treats errors equally, making it less sensitive to outliers than MSE or RMSE. Similar to RMSE, it uses the same units as the target variable, making it easier to interpret than MSE. 

- `0` indicates a the model makes perfect predictions
- Values close to `0` indicate a better fit
- Larger values indicate a worse fit, but there is no upper bound

### Equation
Let's take a look at the equation for MAE:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |y_i - \hat{y}_i|
$$

where: 
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)
- \\(n\\) is the number of data points (`len(y_actual)`)

Since \\(y_i - \hat{y}_i\\) is the same as the \\(\text{residuals}\\) calculated earlier in this article, the equation for \\(\text{MAE}\\) can also be written as:

$$
\text{MAE} = \frac{1}{n} \sum_{i=1}^{n} |\text{residuals}_\text{model}|
$$

### Calculate Manually
Since we already have the residuals for each model, we can get MAE by summing the absolute value of the residuals and then dividing by the number of data points. 


```python
mae_full = np.sum(np.abs(residuals_full)) / n
mae_full
```




    1.6634312974669488e-13




```python
mae_few = np.sum(np.abs(residuals_few)) / n
mae_few
```




    39.83435011538076




```python
mae_base = np.sum(np.abs(residuals_base)) / n
mae_base
```




    139.73130580357142



### Calculate with scikit-learn
We can also use scikit-learn's [mean_absolute_error()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) function. 


```python
mae_full = mean_absolute_error(y_actual, y_pred_full)
mae_full
```




    1.6634312974669488e-13




```python
mae_few = mean_absolute_error(y_actual, y_pred_few)
mae_few
```




    39.83435011538076




```python
mae_base = mean_absolute_error(y_actual, y_pred_base)
mae_base
```




    139.73130580357142



### Evaluation
Similar to MSE and RMSE, the lower the MAE, the better the model's fit. `mae_full` is still quite close to zero, and `mae_few` is much better than `mae_base`, so both of those models perform better than the baseline model. We can use the same context used for RMSE (where `y_actual` ranges from ~190 to ~1,300 with a mean of ~658) to further confirm that the baseline model performs quite poorly, while `mae_few` performs reasonably well. 

<div class="email-subscription-container"></div>

## Evaluation Metric Results
For convenience, we can also summarize the results of all of these evaluation metrics in a single table: 

| Method | Model | \\(R^2\\) | Adjusted \\(R^2\\) | MSE | RMSE | MAE |
|---|---|---|---|---|---|---|
| Mean | `model_base` | 0 | 0 | 32,663.91 | 180.73 | 139.73 |
| OLS | `model_full` | 1.0 | 1.0 | 3.91 \\(x10^{-26}\\) | 1.97 \\(x10^{-13}\\) | 1.66 \\(x10^{-13}\\) |
| OLS | `model_few` | 0.921 | 0.917 | 2,589.41 | 50.88 | 39.83 |
|  |  | *Closer to 1.0 is better* | *Closer to 1.0 is better* | *Lower is better* | *Lower is better* | *Lower is better* |

## Residuals Plots
As a reminder, a [residual](https://en.wikipedia.org/wiki/Errors_and_residuals) is the difference between an observed value and its corresponding predicted value (\\(y_i - \hat{y_i}\\)). We calculated the residuals in a previous step, so now we're ready to plot and evaluate them. Plotting residuals is a useful visual way to evaluate the assumptions and identify potential issues with the fit of a regression model that metrics alone might miss.

When reviewing these residual plots, we'll primarily be checking for whether or not thehre's an issue with the model. For models with a good fit, these plots should have an even, random distribution of residuals around the horizontal line (zero on the y-axis) without any outliers. If there are any clear patterns (curves or clusters) in the residuals plot, that can suggest that the model is not capturing some aspect of the data, such as omitted variables, non-linearity issues, or [heteroskedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity). 

### Evaluating Scatterplot of Residuals
Let's start by creating a scatterplot of the residuals versus the predicted values for each model. 


```python
plt.scatter(y_pred_full, residuals_full)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot for model_full')
plt.show()
```


    
![png](/assets/img/posts/2024-11-27-basketball-evaluate-ols-model/output_145_0.png)
    


The residuals for `model_full` at first glance show striations (clustering along horizontal bands). However, the scale for the y-axis is \\(\times 10^{-13}\\), so all of these residuals are quite close to zero. We covered in a previous section that the residuals for this model are essentially due to floating-point errors, so these residuals are negligable when compared to the scale of the predicted values. 


```python
plt.scatter(y_pred_few, residuals_few)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot for model_few')
plt.show()
```


    
![png](/assets/img/posts/2024-11-27-basketball-evaluate-ols-model/output_147_0.png)
    


The residuals plot for `model_few` shows a more even distribution without any outliers, so there aren't any major issues. 


```python
plt.scatter(y_pred_base, residuals_base)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot for model_base')
plt.show()
```


    
![png](/assets/img/posts/2024-11-27-basketball-evaluate-ols-model/output_149_0.png)
    


The residuals plot for `model_base` looks clearly different from the other two and is not evenly distributed around zero. This plot indicates an issue with the underlying model. Since this model just predicts the mean of the target variable (and uses zero predictors), the pattern shown in this residuals plot is likely due to omitting variables. 

For excellent examples of residuals plots that show good and poor model fits, I highly recommend Penn State's page on [Identifying Specific Problems using Residuals Plots](https://online.stat.psu.edu/stat462/node/120/). 

### Evaluating Histogram of Residuals
A histogram is another useful chart for evaluating residuals. This helps check the normality of residuals and will ideally show a bell-shape (a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution)). We can plot this using [seaborn's .histplot() function](https://seaborn.pydata.org/generated/seaborn.histplot.html).


```python
sns.histplot(residuals_full, bins=15, kde=True)
plt.title('Histogram of Residuals for model_full')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/assets/img/posts/2024-11-27-basketball-evaluate-ols-model/output_152_0.png)
    


The histogram of residuals for `model_full` is roughly bell-shaped (minus the gaps in data due to the striations showin in the previous graph), although it is shifted to the left (below zero). 


```python
sns.histplot(residuals_few, bins=15, kde=True)
plt.title('Histogram of Residuals for model_few')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/assets/img/posts/2024-11-27-basketball-evaluate-ols-model/output_154_0.png)
    


The histogram of residuals for `model_few` matches the normal distribution quite well and is evenly centered around zero. 


```python
sns.histplot(residuals_base, bins=15, kde=True)
plt.title('Histogram of Residuals for model_base')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/assets/img/posts/2024-11-27-basketball-evaluate-ols-model/output_156_0.png)
    


The histogram of residuals for `model_base` is quite normal, minus the "tail" on the right (above zero). Overall, the residuals are normally distributed. 

For more examples of these plots, Penn State has an excellent page on the [Normal Probability Plot of Residuals](https://online.stat.psu.edu/stat462/node/122/), including examples of residuals that are not normally distributed. 

## Feature Importance
As the last piece of model evaluation for today, we'll take a look at the feature importance of each model. To do this, we'll examine differences between the coefficients of the two models (`model_full` and `model_few`). As a reminder, `model_base` simply predicts the mean of the target variable without using any features. This means that all of the feature coefficients are set to zero for `model_base`. Since there is no meaningful comparison to make with `model_base`, we'll focus on comparing the relative feature importance of `model_full` and `model_few`. 

To compare the feature importance between the two models, let's start by identifying the coefficients for each model. For both models, the coefficients can be accessed with the `.coef` attribute and the corresponding feature names can be accessed with the `.feature_names_in` attribute.


```python
coef_full = pd.Series(data=model_full.coef_, index=model_full.feature_names_in_)
coef_full
```




    Height              -3.352062e-15
    MINUTES_PLAYED      -2.775558e-17
    FIELD_GOALS_MADE     1.666667e+00
    THREE_POINTS_MADE    1.333333e+00
    TWO_POINTS_MADE      3.333333e-01
    FREE_THROWS_MADE     1.000000e+00
    TOTAL_REBOUNDS       1.200000e+00
    ASSISTS              1.500000e+00
    TURNOVERS           -1.000000e+00
    STEALS               2.000000e+00
    BLOCKS               2.000000e+00
    FOULS               -1.280226e-15
    POINTS               1.319604e-14
    dtype: float64



There are a few coefficients in `model_full` that are quite close to zero, so let's set the coefficients smaller than \\(\times 10^{-10}\\) to zero.


```python
coef_full[abs(coef_full) < 1e-10] = 0
coef_full
```




    Height               0.000000
    MINUTES_PLAYED       0.000000
    FIELD_GOALS_MADE     1.666667
    THREE_POINTS_MADE    1.333333
    TWO_POINTS_MADE      0.333333
    FREE_THROWS_MADE     1.000000
    TOTAL_REBOUNDS       1.200000
    ASSISTS              1.500000
    TURNOVERS           -1.000000
    STEALS               2.000000
    BLOCKS               2.000000
    FOULS                0.000000
    POINTS               0.000000
    dtype: float64



Now we can assemble the coefficients for `model_few`. 


```python
coef_few = pd.Series(data=model_few.coef_, index=model_few.feature_names_in_)
coef_few
```




    Height               2.453177
    MINUTES_PLAYED       0.103874
    THREE_POINTS_MADE    2.203725
    FREE_THROWS_MADE     2.391730
    TOTAL_REBOUNDS       1.521916
    ASSISTS              1.323135
    TURNOVERS           -0.570632
    STEALS               2.239326
    BLOCKS               2.481790
    FOULS               -0.261209
    dtype: float64



`model_few` has fewer coefficients than `model_full`, so let's put the coefficients for each model and the difference between the two into a [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) for easy comparison. 


```python
df = pd.DataFrame({'model_full': coef_full, 'model_few': coef_few}).fillna(0)
df['difference'] = df.model_few - df.model_full
df
```




<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>model_full</th>
      <th>model_few</th>
      <th>difference</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>ASSISTS</th>
      <td>1.500000</td>
      <td>1.323135</td>
      <td>-0.176865</td>
    </tr>
    <tr>
      <th>BLOCKS</th>
      <td>2.000000</td>
      <td>2.481790</td>
      <td>0.481790</td>
    </tr>
    <tr>
      <th>FIELD_GOALS_MADE</th>
      <td>1.666667</td>
      <td>0.000000</td>
      <td>-1.666667</td>
    </tr>
    <tr>
      <th>FOULS</th>
      <td>0.000000</td>
      <td>-0.261209</td>
      <td>-0.261209</td>
    </tr>
    <tr>
      <th>FREE_THROWS_MADE</th>
      <td>1.000000</td>
      <td>2.391730</td>
      <td>1.391730</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>0.000000</td>
      <td>2.453177</td>
      <td>2.453177</td>
    </tr>
    <tr>
      <th>MINUTES_PLAYED</th>
      <td>0.000000</td>
      <td>0.103874</td>
      <td>0.103874</td>
    </tr>
    <tr>
      <th>POINTS</th>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>STEALS</th>
      <td>2.000000</td>
      <td>2.239326</td>
      <td>0.239326</td>
    </tr>
    <tr>
      <th>THREE_POINTS_MADE</th>
      <td>1.333333</td>
      <td>2.203725</td>
      <td>0.870391</td>
    </tr>
    <tr>
      <th>TOTAL_REBOUNDS</th>
      <td>1.200000</td>
      <td>1.521916</td>
      <td>0.321916</td>
    </tr>
    <tr>
      <th>TURNOVERS</th>
      <td>-1.000000</td>
      <td>-0.570632</td>
      <td>0.429368</td>
    </tr>
    <tr>
      <th>TWO_POINTS_MADE</th>
      <td>0.333333</td>
      <td>0.000000</td>
      <td>-0.333333</td>
    </tr>
  </tbody>
</table>
</div>



As a reminder, `model_few` has fewer parameters than `model_full`, because `FIELD_GOALS_MADE`, `TWO_POINTS_MADE`, and `POINTS` were removed from the feature set to create `model_few`. Interestingly, we can see that `model_full` deemed the Points feature completely unnecessary. `model_few` weights Blocks, Free Throws Made, Height, Three Points Made, Steals, and Total Rebounds as more important than `model_full`. `model_full` weights Assists and Turnovers (plus Field Goals Made and Two Points Made of course) as more important than `model_few`. The importance of Fouls and Minutes Played is similar between the two models. 

# Wrap Up
In today's guide, we covered common methods to evaluate the performance of our OLS linear regression model. This is the final installment in this series! We might revisit this dataset again in the future with a different type of machine learning model (please [let me know](https://www.pineconedata.com/workwithme/) if you're interested in that).

Also, all of the code snippets in today's guide are available in a Jupyter Notebook in the [ncaa-basketball-stats](https://github.com/pineconedata/ncaa-basketball-stats) repository on [GitHub](https://github.com/pineconedata/).

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/) 
8. [Evaluating the Machine Learning Model](/2024-11-27-basketball-evaluate-ols-model) (Today's Guide)

<div class="email-subscription-container"></div>
<div id="sources"></div>

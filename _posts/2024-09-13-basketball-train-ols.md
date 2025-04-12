---
layout: post
title: "Training a Linear Regression Model"
subtitle: "Outlier or Caitlin Clark? [Part 7]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, linear regression]
share-title: "Training a Linear Regression Model: Outlier or Caitlin Clark? [Part 7]" 
share-description: Interested in training a linear regression model on your own data? Learn how to use Python, scikit-learn, and pandas to create train-test splits, train the model, and analyse the model equation in the latest installment of this data science series!
thumbnail-img: /assets/img/posts/2024-09-13-basketball-train-ols/thumbnail.jpg
share-img: /assets/img/posts/2024-09-13-basketball-train-ols/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll expand on the basics of machine learning and examine how to train a linear regression machine learning model. This is the seventh part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

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

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Due to the size of this project, today we'll cover part of the seventh step: training a machine learning model.

## Dependencies
Since this is the seventh installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first article of the series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. 

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
  - [statsmodels](https://www.statsmodels.org/stable/index.html)
  
For today's machine learning sgement specifically, we'll want to import a few of these libraries: 


```python
import pandas as pd
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
```

## Import Data
In [Part 3](/2024-05-30-basketball-feature_engineering/) of this series, we engineered new features for our dataset, which is stored in a dataframe named `player_data`. No changes have been made to the underlying dataset in the intermediary articles. If you want to follow along with the code examples in this article, it's recommended to import the `player_data` dataframe before proceeding. 


```python
player_data = pd.read_excel('player_data_engineered.xlsx')
player_data.head()
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
      <th>PLAYER_NAME</th>
      <th>Team</th>
      <th>Class</th>
      <th>Height</th>
      <th>Position</th>
      <th>PLAYER_ID</th>
      <th>TEAM_NAME</th>
      <th>GAMES</th>
      <th>MINUTES_PLAYED</th>
      <th>FIELD_GOALS_MADE</th>
      <th>...</th>
      <th>Conference</th>
      <th>MINUTES_PER_GAME</th>
      <th>FOULS_PER_GAME</th>
      <th>POINTS_PER_GAME</th>
      <th>ASSISTS_PER_GAME</th>
      <th>STEALS_PER_GAME</th>
      <th>BLOCKS_PER_GAME</th>
      <th>REBOUNDS_PER_GAME</th>
      <th>ASSIST_TO_TURNOVER</th>
      <th>FANTASY_POINTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kiara Jackson</td>
      <td>UNLV (Mountain West)</td>
      <td>Junior</td>
      <td>67</td>
      <td>Guard</td>
      <td>ncaaw.p.67149</td>
      <td>UNLV</td>
      <td>29</td>
      <td>895</td>
      <td>128</td>
      <td>...</td>
      <td>Mountain West</td>
      <td>30.862069</td>
      <td>1.620690</td>
      <td>11.137931</td>
      <td>4.655172</td>
      <td>1.068966</td>
      <td>0.172414</td>
      <td>4.448276</td>
      <td>3.214286</td>
      <td>710.3</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Raven Johnson</td>
      <td>South Carolina (SEC)</td>
      <td>Sophomore</td>
      <td>68</td>
      <td>Guard</td>
      <td>ncaaw.p.67515</td>
      <td>South Carolina</td>
      <td>30</td>
      <td>823</td>
      <td>98</td>
      <td>...</td>
      <td>SEC</td>
      <td>27.433333</td>
      <td>1.133333</td>
      <td>8.100000</td>
      <td>4.933333</td>
      <td>2.000000</td>
      <td>0.166667</td>
      <td>5.366667</td>
      <td>2.792453</td>
      <td>735.2</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gina Marxen</td>
      <td>Montana (Big Sky)</td>
      <td>Senior</td>
      <td>68</td>
      <td>Guard</td>
      <td>ncaaw.p.57909</td>
      <td>Montana</td>
      <td>29</td>
      <td>778</td>
      <td>88</td>
      <td>...</td>
      <td>Big Sky</td>
      <td>26.827586</td>
      <td>0.896552</td>
      <td>10.241379</td>
      <td>3.827586</td>
      <td>0.551724</td>
      <td>0.068966</td>
      <td>2.068966</td>
      <td>2.921053</td>
      <td>533.5</td>
    </tr>
    <tr>
      <th>3</th>
      <td>McKenna Hofschild</td>
      <td>Colorado St. (Mountain West)</td>
      <td>Senior</td>
      <td>62</td>
      <td>Guard</td>
      <td>ncaaw.p.60402</td>
      <td>Colorado St.</td>
      <td>29</td>
      <td>1046</td>
      <td>231</td>
      <td>...</td>
      <td>Mountain West</td>
      <td>36.068966</td>
      <td>1.172414</td>
      <td>22.551724</td>
      <td>7.275862</td>
      <td>1.241379</td>
      <td>0.137931</td>
      <td>3.965517</td>
      <td>2.971831</td>
      <td>1117.5</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kaylah Ivey</td>
      <td>Boston College (ACC)</td>
      <td>Junior</td>
      <td>68</td>
      <td>Guard</td>
      <td>ncaaw.p.64531</td>
      <td>Boston Coll.</td>
      <td>33</td>
      <td>995</td>
      <td>47</td>
      <td>...</td>
      <td>ACC</td>
      <td>30.151515</td>
      <td>1.454545</td>
      <td>4.333333</td>
      <td>5.636364</td>
      <td>1.090909</td>
      <td>0.030303</td>
      <td>1.727273</td>
      <td>2.906250</td>
      <td>500.4</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 40 columns</p>
</div>



## Basics of Machine Learning
Before we get into training a model, let’s briefly revisit a few basics of machine learning. If you are already familiar with these concepts, feel free to skip to the [next section](#Model-Training). [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a branch of artificial intelligence that focuses on creating algorithms and statistical models that allow computer systems to "learn" how to improve their performance on a specific task through experience. In the context of our basketball statistics project, machine learning can be particularly useful for predicting player performance, classifying player position, and identifying similar players.

Key concepts in machine learning that we'll encounter include:

1. **Model** - The system that learns patterns from data and can be used to make predictions on previously unseen data. Machine learning models are often of a specific type (Linear or Logistic Regression, Random Forests, Support Vector Machines, Neural Networks, etc.). Today's model is a Linear Regression model. 
2. **Training Data** - The subset of our data used to train the model.
3. **Testing Data** - A separate subset of data used to evaluate the model's performance.
4. **Features** - The input variables used to make predictions. These are sometimes referred to as the independent variable(s) or the predictor(s). For this project, these are various player statistics like three points made and assists.
5. **Target Variable** - The variable we're trying to predict or optimize. This is sometimes referred to as the dependent variable(s), as it depends on the independent variable(s). In today's project, this is Fantasy Points.
6. **Parameters** - The values that the model learns during training, such as coefficients in linear regression. These parameters define how the model transforms input features into predictions.
7. **Hyperparameters** - The configuration settings for the model that are set before training begins. These are not learned from the data but are specified by the data scientist. Examples include learning rate, number of iterations, or regularization strength. Hyperparameters can significantly affect model performance and are often tuned to optimize the model. 
    - *Note*: The model we’ll be using today is straightforward and doesn’t typically have hyperparameters in the traditional sense. However, it’s still important to know the difference between parameters and hyperparameters since many models will have hyperparameters. 
8. **Residuals** - The differences between the observed values and the predicted values from the model. Residuals help assess how well the model fits the data and can reveal patterns or issues in the model's predictions.
9. **Model Evaluation** - Metrics used to assess how well our model is performing. For a Linear Regression model, this will include metrics like Mean Squared Error (MSE) and the R-squared value.

We’ll use primarily the first six terms throughout this article, so it’s best to familiarize yourself with them now. The other concepts will be explored in more detail in future articles (please [let me know](/workwithme/) if that is something you are interested in!). 

Note: Our focus in this article is on classic machine learning models designed for tabular data. We won't be covering models built specifically for natural language processing, image recognition, or video analysis. However, it's worth mentioning that many problems in these domains often get transformed into tabular data problems, so some of the principles we discuss here may still apply in those contexts. With all of that out of the way, let’s move on to training the machine learning model.


# Model Training
Now that we've covered the basics of machine learning, we're ready to move on to the exciting part: training our [Ordinary Least Squares (OLS)](https://en.wikipedia.org/wiki/Ordinary_least_squares) linear regression model! This process involves several key steps that will help us build a robust and accurate predictive model for our basketball player statistics.

## Define the Variables 
As a reminder from the [previous article](/2024-08-12-basketball-select-ml-ols/), we defined the `target` and `feature` variables as: 

```python
target = 'FANTASY_POINTS'
features = ['Height', 'MINUTES_PLAYED', 'FIELD_GOALS_MADE', 'THREE_POINTS_MADE',
            'TWO_POINTS_MADE', 'FREE_THROWS_MADE', 'TOTAL_REBOUNDS', 'ASSISTS',
            'TURNOVERS', 'STEALS', 'BLOCKS', 'FOULS', 'POINTS']
```

We'll label the features (independent variables) as `X` and the target (dependent) variable as `y` for conciseness.


```python
X = player_data[features]
y = player_data[target]
```

Let's take a quick look at the values in `X`:

```python
X
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
      <td>67</td>
      <td>895</td>
      <td>128</td>
      <td>28</td>
      <td>100</td>
      <td>39</td>
      <td>129</td>
      <td>135</td>
      <td>42</td>
      <td>31</td>
      <td>5</td>
      <td>47</td>
      <td>323</td>
    </tr>
    <tr>
      <th>1</th>
      <td>68</td>
      <td>823</td>
      <td>98</td>
      <td>20</td>
      <td>78</td>
      <td>27</td>
      <td>161</td>
      <td>148</td>
      <td>53</td>
      <td>60</td>
      <td>5</td>
      <td>34</td>
      <td>243</td>
    </tr>
    <tr>
      <th>2</th>
      <td>68</td>
      <td>778</td>
      <td>88</td>
      <td>58</td>
      <td>30</td>
      <td>63</td>
      <td>60</td>
      <td>111</td>
      <td>38</td>
      <td>16</td>
      <td>2</td>
      <td>26</td>
      <td>297</td>
    </tr>
    <tr>
      <th>3</th>
      <td>62</td>
      <td>1046</td>
      <td>231</td>
      <td>55</td>
      <td>176</td>
      <td>137</td>
      <td>115</td>
      <td>211</td>
      <td>71</td>
      <td>36</td>
      <td>4</td>
      <td>34</td>
      <td>654</td>
    </tr>
    <tr>
      <th>4</th>
      <td>68</td>
      <td>995</td>
      <td>47</td>
      <td>32</td>
      <td>15</td>
      <td>17</td>
      <td>57</td>
      <td>186</td>
      <td>64</td>
      <td>36</td>
      <td>1</td>
      <td>48</td>
      <td>143</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>895</th>
      <td>66</td>
      <td>742</td>
      <td>92</td>
      <td>53</td>
      <td>39</td>
      <td>45</td>
      <td>113</td>
      <td>73</td>
      <td>66</td>
      <td>45</td>
      <td>2</td>
      <td>54</td>
      <td>282</td>
    </tr>
    <tr>
      <th>896</th>
      <td>73</td>
      <td>815</td>
      <td>108</td>
      <td>58</td>
      <td>50</td>
      <td>26</td>
      <td>140</td>
      <td>34</td>
      <td>46</td>
      <td>19</td>
      <td>19</td>
      <td>51</td>
      <td>300</td>
    </tr>
    <tr>
      <th>897</th>
      <td>71</td>
      <td>774</td>
      <td>102</td>
      <td>56</td>
      <td>46</td>
      <td>67</td>
      <td>176</td>
      <td>29</td>
      <td>48</td>
      <td>29</td>
      <td>3</td>
      <td>68</td>
      <td>327</td>
    </tr>
    <tr>
      <th>898</th>
      <td>71</td>
      <td>848</td>
      <td>127</td>
      <td>54</td>
      <td>73</td>
      <td>76</td>
      <td>123</td>
      <td>71</td>
      <td>90</td>
      <td>35</td>
      <td>9</td>
      <td>94</td>
      <td>384</td>
    </tr>
    <tr>
      <th>899</th>
      <td>67</td>
      <td>872</td>
      <td>133</td>
      <td>55</td>
      <td>78</td>
      <td>44</td>
      <td>127</td>
      <td>29</td>
      <td>41</td>
      <td>37</td>
      <td>2</td>
      <td>52</td>
      <td>365</td>
    </tr>
  </tbody>
</table>
<p>900 rows × 13 columns</p>
</div>

Let's check the values in `y` as well:


```python
y
```




    0       710.3
    1       735.2
    2       533.5
    3      1117.5
    4       500.4
            ...  
    895     555.1
    896     549.0
    897     597.7
    898     636.1
    899     597.9
    Name: FANTASY_POINTS, Length: 900, dtype: float64

These look great and match with the values we saw in the previous article, so we can move on to the next step.

## Create Training and Testing Splits
Now that we have our variables defined, we can create the training and testing splits. This involves dividing our dataset into two parts: a set for training and a set for testing. The train set will be used to train the model, and the test set will be used exclusively for testing and evaluating the model after training. 

At this point, you might wonder: *Why don't we just use all of the data for training?* There are several reasons for this:
1. **Model Evaluation** - Having a test set allows us to evaluate how well our model performs on unseen data, giving us a more realistic estimate of its performance in real-world scenarios.
2. **Preventing Overfitting** - By keeping a portion of our data separate for testing, we can detect if our model is overfitting to the training data. Overfitting occurs when a model learns the training data too well, including its noise and peculiarities, leading to poor generalization on new data.
3. **Validating Model Generalization** - The test set helps us validate whether our model can generalize well to new, unseen data. This is crucial for ensuring that our model will perform reliably when deployed in practice.
4. **Hyperparameter Tuning** - While we don't have any hyperparameters to tune today, splitting the data is also essential for hyperparameter tuning techniques like cross-validation.

To create our training and test splits, we'll use the [`train_test_split` function from scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html). This function allows us to easily split our data while maintaining the proportion of samples for each class. Here's how we can implement it:


```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
```

The `_train` splits (`X_train` and `y_train`) include the data for training the model and the `_test` splits (`X_test` and `y_test`) include the data for testing the model. By using these separate splits, we know that our model is trained on one subset of the data and evaluated on a completely separate subset, so we can essentially set aside the `_test` splits until we're ready to evaluate the performance of the model.

We can take a look at the first few rows of the training split for the features using `X_train`: 


```python
X_train.head(5)
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
      <th>200</th>
      <td>68</td>
      <td>669</td>
      <td>53</td>
      <td>10</td>
      <td>43</td>
      <td>75</td>
      <td>73</td>
      <td>111</td>
      <td>89</td>
      <td>25</td>
      <td>3</td>
      <td>50</td>
      <td>191</td>
    </tr>
    <tr>
      <th>300</th>
      <td>73</td>
      <td>577</td>
      <td>103</td>
      <td>0</td>
      <td>103</td>
      <td>46</td>
      <td>170</td>
      <td>29</td>
      <td>56</td>
      <td>30</td>
      <td>37</td>
      <td>81</td>
      <td>252</td>
    </tr>
    <tr>
      <th>336</th>
      <td>74</td>
      <td>791</td>
      <td>108</td>
      <td>6</td>
      <td>102</td>
      <td>38</td>
      <td>213</td>
      <td>19</td>
      <td>56</td>
      <td>33</td>
      <td>40</td>
      <td>97</td>
      <td>260</td>
    </tr>
    <tr>
      <th>727</th>
      <td>70</td>
      <td>591</td>
      <td>84</td>
      <td>0</td>
      <td>84</td>
      <td>39</td>
      <td>210</td>
      <td>12</td>
      <td>60</td>
      <td>20</td>
      <td>21</td>
      <td>93</td>
      <td>207</td>
    </tr>
    <tr>
      <th>403</th>
      <td>74</td>
      <td>303</td>
      <td>49</td>
      <td>2</td>
      <td>47</td>
      <td>16</td>
      <td>125</td>
      <td>8</td>
      <td>23</td>
      <td>10</td>
      <td>26</td>
      <td>52</td>
      <td>116</td>
    </tr>
  </tbody>
</table>
</div>



### Reproducibility
You might notice that if you run the `train_test_split()` for a second time, there are different rows of data included in each split. Here's an example of re-running the exact same code:


```python
X_train, X_test, y_train, y_test = train_test_split(X, y)
X_train.head(5)
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
      <th>397</th>
      <td>74</td>
      <td>596</td>
      <td>80</td>
      <td>4</td>
      <td>76</td>
      <td>63</td>
      <td>192</td>
      <td>28</td>
      <td>48</td>
      <td>16</td>
      <td>32</td>
      <td>69</td>
      <td>227</td>
    </tr>
    <tr>
      <th>501</th>
      <td>73</td>
      <td>829</td>
      <td>199</td>
      <td>4</td>
      <td>195</td>
      <td>65</td>
      <td>301</td>
      <td>53</td>
      <td>73</td>
      <td>42</td>
      <td>7</td>
      <td>96</td>
      <td>467</td>
    </tr>
    <tr>
      <th>107</th>
      <td>69</td>
      <td>1072</td>
      <td>128</td>
      <td>42</td>
      <td>86</td>
      <td>75</td>
      <td>187</td>
      <td>137</td>
      <td>86</td>
      <td>111</td>
      <td>7</td>
      <td>80</td>
      <td>373</td>
    </tr>
    <tr>
      <th>462</th>
      <td>75</td>
      <td>619</td>
      <td>59</td>
      <td>0</td>
      <td>59</td>
      <td>54</td>
      <td>126</td>
      <td>18</td>
      <td>57</td>
      <td>29</td>
      <td>26</td>
      <td>53</td>
      <td>172</td>
    </tr>
    <tr>
      <th>263</th>
      <td>78</td>
      <td>995</td>
      <td>278</td>
      <td>2</td>
      <td>276</td>
      <td>103</td>
      <td>330</td>
      <td>52</td>
      <td>62</td>
      <td>17</td>
      <td>60</td>
      <td>50</td>
      <td>661</td>
    </tr>
  </tbody>
</table>
</div>


This happens because the data is shuffled before splitting, and the shuffling is not guaranteed to be reproducible by default. This can mean that the model is trained and tested on different datasets each time that you run it. That can be a good thing, but it might be better to have reproducible results for initial creation and evaluation of the model (especially if you want to follow along with this guide). 

We can ensure reproducibility of the splits by controlling the shuffling with the `random_state` parameter: 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=314)
X_train.head(5)
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
      <th>239</th>
      <td>68</td>
      <td>1037</td>
      <td>258</td>
      <td>50</td>
      <td>208</td>
      <td>92</td>
      <td>152</td>
      <td>96</td>
      <td>87</td>
      <td>32</td>
      <td>3</td>
      <td>59</td>
      <td>658</td>
    </tr>
    <tr>
      <th>638</th>
      <td>70</td>
      <td>961</td>
      <td>109</td>
      <td>44</td>
      <td>65</td>
      <td>66</td>
      <td>136</td>
      <td>87</td>
      <td>75</td>
      <td>63</td>
      <td>11</td>
      <td>71</td>
      <td>328</td>
    </tr>
    <tr>
      <th>848</th>
      <td>69</td>
      <td>908</td>
      <td>142</td>
      <td>72</td>
      <td>70</td>
      <td>58</td>
      <td>78</td>
      <td>61</td>
      <td>71</td>
      <td>20</td>
      <td>5</td>
      <td>71</td>
      <td>414</td>
    </tr>
    <tr>
      <th>260</th>
      <td>76</td>
      <td>913</td>
      <td>112</td>
      <td>28</td>
      <td>84</td>
      <td>59</td>
      <td>200</td>
      <td>40</td>
      <td>64</td>
      <td>23</td>
      <td>73</td>
      <td>94</td>
      <td>311</td>
    </tr>
    <tr>
      <th>745</th>
      <td>66</td>
      <td>713</td>
      <td>146</td>
      <td>36</td>
      <td>110</td>
      <td>57</td>
      <td>124</td>
      <td>105</td>
      <td>96</td>
      <td>72</td>
      <td>2</td>
      <td>93</td>
      <td>385</td>
    </tr>
  </tbody>
</table>
</div>


Now, no matter how many times you run the snippet, you should get the same rows of data in the train and test splits every time. 

### Dataset Proportions
Another parameter that is commonly specified is the `test_size` (or, less commonly, the `train_size`) to specify a proportion of the dataset to include in the test or train split, respectively. According to the [documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html), the default `test_size` (assuming `train_size` is not set) is `0.25`, so one-quarter of the data will be included in the test dataset and three-quarters of the data will be included in the train dataset. 

We can verify this on our own splits: 


```python
print(f'Test data split proportion: {len(X_test) / len(X)}')
print(f'Train data split proportion: {len(X_train) / len(X)}')
```

    Test data split proportion: 0.25
    Train data split proportion: 0.75


If you want to change those proportions, then you can use either the `test_size` or the `train_size` parameter. For example, if you want the test split to be 20% of the data instead of 25%, you would run: 


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42, test_size=0.2)
print(f'Test data split proportion: {len(X_test) / len(X)}')
print(f'Train data split proportion: {len(X_train) / len(X)}')
```

    Test data split proportion: 0.2
    Train data split proportion: 0.8


We'll be using the default 25% test size, so we can remove the `test_size` parameter for today.


```python
X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=314)
print(f'Test data split proportion: {len(X_test) / len(X)}')
print(f'Train data split proportion: {len(X_train) / len(X)}')
```

    Test data split proportion: 0.25
    Train data split proportion: 0.75

<div class="email-subscription-container"></div>

### A Note on Dataframes versus numPy Arrays
You might notice that we're using dataframes directly (`player_data[features]`) in the `train_test_split`, but some code samples will use numPy arrays instead (`player_data[features].to_numpy()`). Both approaches can work, but they have different implications depending on how you intend to use the data. Let's quickly review the pros and cons of each approach.

#### Using pandas DataFrames or Series
This entails passing `X` and `y` as DataFrames or Series directly to the `train_test_split` function. 


```python
print(f'dtype of X: {type(X)}')
print(f'dtype of y: {type(y)}')
```

    dtype of X: <class 'pandas.core.frame.DataFrame'>
    dtype of y: <class 'pandas.core.series.Series'>


Some advantages of this approach are:
- Retains column names, which can be useful for model interpretation and debugging.
- Works seamlessly with scikit-learn, which can handle DataFrames directly.
- Makes it easier to perform certain operations, like feature selection and transformation.
Some disadvantages of this approach are: 
- Slightly more overhead than working with numpy arrays, but generally negligible.

#### Using numPy arrays
This entails passing `X` and `y` as numPy arrays to the `train_test_split` function. 


```python
print(f'dtype of X: {type(X.to_numpy())}')
print(f'dtype of y: {type(y.to_numpy())}')
```

    dtype of X: <class 'numpy.ndarray'>
    dtype of y: <class 'numpy.ndarray'>


Some advantages of this approach are:
- Can be slightly faster for certain operations because numpy arrays are lower-level structures.
- If you need to work with libraries that require numpy arrays (although most scikit-learn functions accept DataFrames as well).
Some disadvantages of this approach are: 
- You lose the column names and index information, which can make it harder to trace errors or interpret results later.
- Not necessary for most scikit-learn functions, which work fine with DataFrames.

It's generally more convenient to use DataFrames directly unless you have a specific reason to convert to numpy arrays. This way, you retain all the metadata that can be useful during data analysis and model interpretation.

In summary, you can choose either method based on your preference, but it's generally more convenient to use DataFrames directly unless you have a specific reason to convert to numpy arrays. If you have a specific scenario where a numpy array is required, then use `.to_numpy()`. For today, we'll move on with the DataFrames approach.

## Train the Model
Now that we have our data split into training and test sets, we're ready to train our linear regression model. We'll use [scikit-learn's LinearRegression class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LinearRegression.html) (which uses ordinary least squares) for this purpose. 

We can start by initializing the model using an instance of the `LinearRegression` class with the default parameters: 


```python
linear_reg_model = LinearRegression()
```

We can then use the `fit()` method to actually train our model on our data. This method takes two arguments: `X_train` (the training split of our features) and `y_train` (the training split of our target variable).


```python
linear_reg_model.fit(X_train, y_train)
```

During the training process, the model uses the training set to learn the optimal parameters for each feature that minimize the difference between the predicted values of the target variable and the actual values of the target variable. Once the training is complete, our `linear_reg_model` object will contain the learned parameters (coefficients and intercept) that can be used to make predictions on new data. In the future, we can use `X_test` to predict what the model thinks `y` is, and then compare that output `y` to the actual `y` values stored in `y_test` to evaluate the model performance. For now, let's take a look at the model parameters directly.

## Print the Model Equation
While not strictly necessary, it can be helpful to circle back to the linear regression equation mentioned in an [earlier section](#model-characteristics) by printing the final equation of our trained model. For a linear regression model, the learned parameters are the coefficients and intercept, which can be used to assemble the model equation. 

The model coefficients can be printed with the `coef_` property: 


```python
linear_reg_model.coef_
```




    array([ 1.37791128e-14,  1.19348975e-15,  1.66666667e+00,  1.33333333e+00,
            3.33333333e-01,  1.00000000e+00,  1.20000000e+00,  1.50000000e+00,
           -1.00000000e+00,  2.00000000e+00,  2.00000000e+00, -1.20129601e-15,
           -7.23379689e-16])



The model intercept can be printed with the `intercept_` property: 


```python
linear_reg_model.intercept_
```




    -6.821210263296962e-13



We can print the feature names with the `feature_names_in_` property:


```python
linear_reg_model.feature_names_in_
```




    array(['Height', 'MINUTES_PLAYED', 'FIELD_GOALS_MADE',
           'THREE_POINTS_MADE', 'TWO_POINTS_MADE', 'FREE_THROWS_MADE',
           'TOTAL_REBOUNDS', 'ASSISTS', 'TURNOVERS', 'STEALS', 'BLOCKS',
           'FOULS', 'POINTS'], dtype=object)



This isn't very readable though, so with a bit of effort we can [map each coefficient to the corresponding feature name](https://stackoverflow.com/questions/34649969/how-to-find-the-features-names-of-the-coefficients-using-scikit-linear-regressio). We can start by creating a dictionary of the coefficients: 


```python
coef_series = pd.Series(data=linear_reg_model.coef_, index=linear_reg_model.feature_names_in_)
coef_series
```




    Height               1.377911e-14
    MINUTES_PLAYED       1.193490e-15
    FIELD_GOALS_MADE     1.666667e+00
    THREE_POINTS_MADE    1.333333e+00
    TWO_POINTS_MADE      3.333333e-01
    FREE_THROWS_MADE     1.000000e+00
    TOTAL_REBOUNDS       1.200000e+00
    ASSISTS              1.500000e+00
    TURNOVERS           -1.000000e+00
    STEALS               2.000000e+00
    BLOCKS               2.000000e+00
    FOULS               -1.201296e-15
    POINTS              -7.233797e-16
    dtype: float64



This makes the coefficients far more readable! We can print this data as a string in the format of `coefficient`*`feature_name` (with newlines for formatting) to then use in the model equation:


```python
coef_string = "\n + ".join(f"{coef}*{feat}" for feat, coef in coef_series.items())
print(coef_string)
```

    1.3779112764770156e-14*Height
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
     + -7.233796894823286e-16*POINTS


Let's use this coefficient string to assemble the final model equation:


```python
print(f'{target} = {coef_string} + {linear_reg_model.intercept_} + error')
```

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
     + -7.233796894823286e-16*POINTS + -6.821210263296962e-13 + error


### Analyze the Model Equation
Now that we have the final model equation, we can see that multiple variables have a coefficient that is very close to zero (`e-15` or smaller values). If we remove those values using `abs(coef) > 0.0001`, this equation simplifies to: 


```python
coef_series_simple = coef_series[abs(coef_series) > 0.0001]
coef_string_simple = "\n\t\t + ".join(f"{coef:.4f} * {feat}" for feat, coef in coef_series_simple.items())
print(f'{target} = {coef_string_simple} + {linear_reg_model.intercept_} + error')
```

    FANTASY_POINTS = 1.6667 * FIELD_GOALS_MADE
		 + 1.3333 * THREE_POINTS_MADE
		 + 0.3333 * TWO_POINTS_MADE
		 + 1.0000 * FREE_THROWS_MADE
		 + 1.2000 * TOTAL_REBOUNDS
		 + 1.5000 * ASSISTS
		 + -1.0000 * TURNOVERS
		 + 2.0000 * STEALS
		 + 2.0000 * BLOCKS + 2.2737367544323206e-13 + error


Excellent! We can compare this to the original equation for Fantasy Points:

```
FANTASY_POINTS =  3   * THREE_POINTS_MADE + 
                  2   * TWO_POINTS_MADE + 
                  1   * FREE_THROWS_MADE + 
                  1.2 * TOTAL_REBOUNDS + 
                  1.5 * ASSISTS + 
                  2   * BLOCKS + 
                  2   * STEALS + 
                 -1   * TURNOVERS
```

The model estimated some of the coefficients to be the same values as the known equation. We included more parameters in the machine learning model than were in the original equation and most of those extra parameters were estimated to have a coefficient around zero. However, we can see that the coefficient was not zero for one of those extra parameters, `FIELD_GOALS_MADE`, and that the coefficients for `THREE_POINTS_MADE` and `TWO_POINTS_MADE` do not match between the estimated and original equations. Here's a quick table to summarize the differences:

|-| FIELD_GOALS_MADE | THREE_POINTS_MADE | TWO_POINTS_MADE | 
|-|-|-|-|
| Estimated Coefficient | 1.6 | 1.3 | 0.3 | 
| Original Coefficient | *Does Not Exist* | 3 | 2 |

*Note: the fractions were rounded to one-tenth for readability but the values are actually \\(1 \frac{2}{3}\\) instead of 1.6, \\(1 \frac{1}{3}\\) instead of 1.3, \\(\frac{1}{3}\\) instead of 0.3, etc.*

This already seems pretty close, but we can take it a step further by revisiting the [definition of field goals](https://en.wikipedia.org/wiki/Field_goal_(basketball)). The `FIELD_GOALS_MADE` variable is actually the sum of `TWO_POINTS_MADE` and `THREE_POINTS_MADE` (but not `FREE_THROWS_MADE`). We can sanity check this in our dataset as well: 


```python
check = player_data['FIELD_GOALS_MADE'] == player_data['TWO_POINTS_MADE'] + player_data['THREE_POINTS_MADE']
print(f'True count: {check.sum()} rows')
print(f'False count: {(~check).sum()} rows')
```

    True count: 900 rows
    False count: 0 rows


So, we can write this relationship as the equation: 

```
FIELD_GOALS_MADE = TWO_POINTS_MADE + THREE_POINTS_MADE
```

We can then substitute this equation into the model's equation: 

```
FANTASY_POINTS =  1.6 * (TWO_POINTS_MADE + THREE_POINTS_MADE) + 
                  1.3 * THREE_POINTS_MADE + 
                  0.3 * TWO_POINTS_MADE + 
                  1   * FREE_THROWS_MADE + 
                  1.2 * TOTAL_REBOUNDS + 
                  1.5 * ASSISTS + 
                  2   * BLOCKS + 
                  2   * STEALS + 
                 -1   * TURNOVERS
```

A quick distribution of the coefficient turns this into: 

```
FANTASY_POINTS =  1.6 * THREE_POINTS_MADE + 
                  1.3 * THREE_POINTS_MADE + 
                  1.6 * TWO_POINTS_MADE +
                  0.3 * TWO_POINTS_MADE + 
                  1   * FREE_THROWS_MADE + 
                  1.2 * TOTAL_REBOUNDS + 
                  1.5 * ASSISTS + 
                  2   * BLOCKS + 
                  2   * STEALS + 
                 -1   * TURNOVERS
```

Using the actual fractional notation of these coefficients (instead of the rounded values shown above), this simplifies down into: 

```
FANTASY_POINTS =  3   * THREE_POINTS_MADE + 
                  2   * TWO_POINTS_MADE +
                  1   * FREE_THROWS_MADE + 
                  1.2 * TOTAL_REBOUNDS + 
                  1.5 * ASSISTS + 
                  2   * BLOCKS + 
                  2   * STEALS + 
                 -1   * TURNOVERS
```

This means that the model estimated approximately the same equation as the original fantasy points calculation, with the addition of a few dependent variables with coefficients close to zero and an intercept value close to zero. As a reminder, some models will be sufficiently complex that it might be difficult to output and effectively analyze the estimated equation, but it provides a lot of value in this case.

## Alternate Training
Since we have ended up with essentially the same equation as the original fantasy points calculation, we can logically expect our model to perform pretty well in the next phase of model evaluation. So, we can also train an alternate model with a few of the features removed for comparison. First, let's create an alternate version of `X` with all three of the features with high correlation coefficients removed: 


```python
X_alt = player_data[features].drop(columns=['FIELD_GOALS_MADE', 'TWO_POINTS_MADE', 'POINTS'])
```

*Note: this is more features than you would likely want to remove in a real-world scenario, but removing too many features will give us an opportunity to compare a less-than-perfect model to a perfect model in the model evaluation phase.*

Our target variable `y` is unchanged, so we can create alternate training and test splits using this `X_alt`: 


```python
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(X_alt, y, random_state=314)
```

We can now train an alternate model using these new training splits: 


```python
ols_alt = LinearRegression()
ols_alt.fit(X_train_alt, y_train_alt)
```

We can print the model equation for this alternate model as well: 


```python
coef_series_alt = pd.Series(data=ols_alt.coef_, index=ols_alt.feature_names_in_)
coef_series_alt = coef_series_alt[abs(coef_series_alt) > 0.0001]
coef_string_alt = "\n\t\t + ".join(f"{coef:.4f} * {feat}" for feat, coef in coef_series_alt.items())
print(f'{target} = {coef_string_alt} + {ols_alt.intercept_} + error')
```

    FANTASY_POINTS = 1.4457 * Height
    		 + 0.1001 * MINUTES_PLAYED
    		 + 2.2441 * THREE_POINTS_MADE
    		 + 2.3925 * FREE_THROWS_MADE
    		 + 1.5098 * TOTAL_REBOUNDS
    		 + 1.2482 * ASSISTS
    		 + 2.1495 * STEALS
    		 + 2.5698 * BLOCKS + -131.85858178609237 + error
             
    FANTASY_POINTS = 2.4532 * Height
		 + 0.1039 * MINUTES_PLAYED
		 + 2.2037 * THREE_POINTS_MADE
		 + 2.3917 * FREE_THROWS_MADE
		 + 1.5219 * TOTAL_REBOUNDS
		 + 1.3231 * ASSISTS
		 + -0.5706 * TURNOVERS
		 + 2.2393 * STEALS
		 + 2.4818 * BLOCKS
		 + -0.2612 * FOULS + -203.27425560271263 + error


We can see that the model coefficients and the y-intercept are substantially different from the model we originally trained. We won't know if this alternate model performs as well as the original one until we evaluate each model in the next article. 

# Export Data & Models
If you're going to use a new Jupyter notebook / Python script for the next part of this series, then it's a good idea to export the testing dataset. 

```python
X_test.to_csv('X_test_full.csv', index=False)
X_test_alt.to_csv('X_test_few.csv', index=False)
y_test.to_csv('y_actual.csv', index=False)
```

While it's not strictly necessary to export small, simple models like these, it's often helpful for checkpointing and collaboration. There are multiple ways to export machine learning models detailed in [scikit-learn's model persistence](https://scikit-learn.org/stable/model_persistence.html) page, including the popular [pickle](https://docs.python.org/3/library/pickle.html#module-pickle) library, but for today we'll use [joblib](https://joblib.readthedocs.io/en/latest/index.html#module-joblib). 

```python
joblib.dump(linear_reg_model, 'model_full.sav')
joblib.dump(ols_alt, 'model_few.sav')
```

# Wrap Up
In today's guide, we covered how to train the selected machine learning model, including how to properly split our dataset into train and test subsets. In the next part, we'll focus on how to evaluate the model's performance.

Also, all of the code snippets in today's guide are available in a Jupyter Notebook in the [ncaa-basketball-stats](https://github.com/pineconedata/ncaa-basketball-stats) repository on [GitHub](https://github.com/pineconedata/).

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/) (Today's Guide)
8. [Evaluating the Machine Learning Model](/2024-11-27-basketball-evaluate-ols-model/)

<div class="email-subscription-container"></div>
<div id="sources"></div>

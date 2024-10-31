---
layout: post
title: "Training a Linear Regression Model"
subtitle: "Outlier or Caitlin Clark? [Part 5]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, linear regression]
share-title: "Training a Linear Regression Model: Outlier or Caitlin Clark? [Part 5]" 
share-description: Interested in training a linear regression model on your own data? Learn how to select and train a linear regression machine learning model in the latest installment of this data science series that is perfect for beginner data scientists and Python enthusiasts.
thumbnail-img: /assets/img/posts/2024-09-13-basketball-train-ols/thumbnail.jpg
share-img: /assets/img/posts/2024-09-13-basketball-train-ols/social.png
readtime: true
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
after-content: post-subscribe.html
---

Today we'll cover the basics of machine learning and examine how to train a linear regression machine learning model. This is the fifth part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

As a reminder, the dataset we'll be using in this project contains individual basketball player statistics (such as total points scored and blocks made) for the 2023-2024 NCAA women's basketball season. Here's a brief description of each major step of this project: 

![the steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_project_steps.png "the steps for this data science project")

1. **Data Acquisition** - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. **Data Cleaning** - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. **Data Preprocessing** - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. **Feature Engineering** - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.
5. **Creating Visualizations** - This step involves identifying the relationships between various parameters (such as height and blocked shots) and generating meaningful visualizations (such as bar charts, scatterplots, and candlestick charts).
6. **Machine Learning** - This step focuses on training a machine learning model to identify the combination of individual player statistics that correlates with optimal performance. 

We'll use Python along with the popular [scikit-learn](https://scikit-learn.org/stable/index.html) and [statsmodels](https://www.statsmodels.org/stable/index.html) libraries to train and evaluate the model. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, create meaningful visualizations, and train a basic machine learning model. Since we already gathered the raw data from online sources in [Part 1](/2024-04-11-basketball-data-acquisition/), cleaned that data in [Part 2](/2024-05-02-basketball-data-cleaning-preprocessing/), engineered new features in [Part 3](/2024-05-30-basketball-feature_engineering/), and explored visualizations in [Part 4](/2024-07-29-basketball-visualizations/), we're ready to move on to training a machine learning model.

<div id="toc"></div>

# Getting Started
Since this is the fifth installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first article of the series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. 

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
  - [statsmodels](https://www.statsmodels.org/stable/index.html)
  
For today's machine learning sgement specifically, we'll want to import a few of these libraries: 


```python
import pandas as pd
import numpy as np
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import matplotlib.pyplot as plt
from statsmodels.api import OLS
```

## Import Data
In [Part 3](/2024-05-30-basketball-feature_engineering/) of this series, we engineered new features for our dataset, which is stored in a dataframe named `player_data`. No changes were made to the underlying data set in [Part 4](/2024-07-29-basketball-visualizations/) of this series, since that part focused on creating visualizations from the data set. If you want to follow along with the code examples in this article, it's recommended to import the `player_data` dataframe before proceeding. 


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
Before we get into training a model, let’s briefly cover a few basics of machine learning. [Machine learning](https://en.wikipedia.org/wiki/Machine_learning) is a branch of artificial intelligence that focuses on creating algorithms and statistical models that allow computer systems to "learn" how to improve their performance on a specific task through experience. In the context of our basketball statistics project, machine learning can be particularly useful for predicting player performance, classifying player position, and identifying similar players.

Key concepts in machine learning that we'll encounter include:

1. **Model**: The system that learns patterns from data and can be used to make predictions on previously unseen data. Machine learning models are often of a specific type (Linear or Logistic Regression, Random Forests, Support Vector Machines, Neural Networks, etc.). 
2. **Training Data:** The subset of our data used to train the model.
3. **Testing Data:** A separate subset of data used to evaluate the model's performance.
4. **Features:** The input variables used to make predictions. This is sometimes referred to as the independent variable(s). In our case, these could be various player statistics like three points made or assists. 
5. **Target Variable:** The variable we're trying to predict or optimize, such as points scored or fantasy points. This is sometimes referred to as the dependent variable(s), as it depends on the independent variable(s). 
6. **Parameters:** The values that the model learns during training, such as coefficients in linear regression. These parameters define how the model transforms input features into predictions.
7. **Hyperparameters:** The configuration settings for the model that are set before training begins. These are not learned from the data but are specified by the data scientist. Examples include learning rate, number of iterations, or regularization strength. Hyperparameters can significantly affect model performance and are often tuned to optimize the model. 
    - *Note*: The model we’ll be using today is straightforward and doesn’t typically have hyperparameters in the traditional sense. However, it’s still important to know the difference between parameters and hyperparameters since many models will have hyperparameters. 
8. **Residuals:** The differences between the observed values and the predicted values from the model. Residuals help assess how well the model fits the data and can reveal patterns or issues in the model's predictions.
9. **Model Evaluation:** Metrics used to assess how well our model is performing. For a Linear Regression model, this will include metrics like Mean Squared Error (MSE) and the R-squared value.

We’ll use primarily the first six terms throughout this article, so it’s best to familiarize yourself with them now. The other concepts will be explored in more detail in future articles (please [let me know](/workwithme/) if that is something you are interested in!). 

It's important to note that our focus in this article is on classic machine learning models designed for tabular data. We won't be covering models built specifically for natural language processing, image recognition, or video analysis. However, it's worth mentioning that many problems in these domains often get transformed into tabular data problems, so some of the principles we discuss here may still apply in those contexts. With all of that out of the way, let’s move on to defining the problem and selecting an appropriate machine learning model.

# Select a Model
Before we choose a model, it's a good idea to clearly define our objective to help us ensure we're using an appropriate model for our task. This step sets the foundation for our entire machine learning process and helps guide our decision-making throughout the project.

## Define the Objective
The goal of a machine learning project in commercial settings will often be determined by a desired business outcome. However, for a hobby project like this, we have the freedom to pick the objective. So, for today’s machine learning model, we’ll focus on training the model to predict a target variable based on one or more input features (such as field goals, blocks, assists, etc.). Let's choose the target variable and set of features as well. 

### Define the Target Variable
The target variable has a massive impact on the machine learning model, including what type (regression, classification, clustering, etc.) of machine learning model is appropriate. For today, let's choose one of the numerical columns from [Part 4](/2024-07-29-basketball-visualizations/): 
```
numerical_columns = ['Height', 'MINUTES_PLAYED', 'FIELD_GOALS_MADE', 
                     'THREE_POINTS_MADE', 'TWO_POINTS_MADE', 'FREE_THROWS_MADE',
                     'TOTAL_REBOUNDS', 'ASSISTS', 'TURNOVERS', 
                     'STEALS', 'BLOCKS', 'FOULS', 'POINTS', 'FANTASY_POINTS']
```
We could choose the `POINTS` variable, but that would end up focusing on primarily offensive players in the model. Defensive players could be prioritized by choosing the `BLOCKS` variable. To include all types of players, let's instead use `FANTASY_POINTS` as the target variable.  


```python
target = 'FANTASY_POINTS'
```

### Define the Features
Next up, let's define the features that the model should use. Typically, this will involve performing feature selection and feature engineering on the dataset, but we've already essentially done that in [Part 3](/2024-05-30-basketball-feature_engineering/) and [Part 4](/2024-07-29-basketball-visualizations/). So, we'll use the list of numerical columns with the target variable (`FANTASY_POINTS`) removed. 


```python
features = ['Height', 'MINUTES_PLAYED', 'FIELD_GOALS_MADE', 'THREE_POINTS_MADE',
            'TWO_POINTS_MADE', 'FREE_THROWS_MADE', 'TOTAL_REBOUNDS', 'ASSISTS',
            'TURNOVERS', 'STEALS', 'BLOCKS', 'FOULS', 'POINTS']
```

### Final Objective
In summary, today's objective is to train a machine learning model to predict `FANTASY_POINTS` (the target variable) based on `Height`, `MINUTES_PLAYED`, `FIELD_GOALS_MADE`, `THREE_POINTS_MADE`, `TWO_POINTS_MADE`, `FREE_THROWS_MADE`, `TOTAL_REBOUNDS`, `ASSISTS`, `TURNOVERS`, `STEALS`, `BLOCKS`, `FOULS`, and `POINTS` (the features).

### A Solved Problem
If you've been following this series from the beginning, you might remember that we actually calculated `FANTASY_POINTS` from some of these variables at the end of [Part 3](/2024-05-30-basketball-feature_engineering/). The equation we used was: 

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

So this is essentially a solved problem and training a machine learning model is technically unnecessary. However, there are some benefits of using a known relationship as a starting point for learning about machine learning and training your first model:

1. **Verification of results** - Since we know the exact formula used to calculate the target variable, we can easily verify if our model is working correctly. This provides a clear benchmark for our model's performance.
2. **Understanding feature importance** - By comparing the coefficients yur model learns to the actual coefficients used in the fantasy points calculation, we can gain insights into how well the model identifies important features.
3. **Concept reinforcement** - Working with a known relationship helps reinforce key machine learning concepts like feature selection, model training, and evaluation in a controlled environment.
4. **Confidence building** - Successfully predicting a known relationship can also boost confidence in applying machine learning techniques to more complex, unknown relationships in the future.

While using a known relationship is a great learning tool, it's important to remember that in real-world scenarios, the relationships between variables are often unknown or more complex. This exercise serves as a stepping stone to tackling more challenging predictive tasks. To simulate a bit of this complexity and to make the future article on model evaluation more valuable, we'll also train an alternate version of the model after removing some of the features. 

## Identify Appropriate Models
The first part of training any machine learning model is selecting the model to use. This might sound obvious, but selecting the "best" model for your problem depends on a variety of factors. We'll likely explore this step in more detail in a future article ([send me a note](/workwithme/) if you would be interested in that), but for today we'll go through [scikit-learn's model flowchart](https://scikit-learn.org/stable/machine_learning_map.html) to find an appropriate model. 

![scikit-learn's algorithm flowchat](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)

Let's start with the first question in the flowchart: "Are there more than 50 samples?" We can answer this by checking the number of rows in our dataframe.


```python
len(player_data)
```




    900



This confirms that there are more than 50 rows in our dataframe, so we can follow the "yes" path to the next question: "Are we predicting a category"? We'll be predicting `FANTASY_POINTS`, which we can check the datatype of using `.dtypes`: 


```python
player_data['FANTASY_POINTS'].dtypes
```




    dtype('float64')



So in this case our target variable contains `float64` values. Note that in many cases, the data type might be listed as `object`, such as if numeric data is stored as strings or if there are multiple datatypes in the column. In those cases, it can also be a good idea to look at a sample of the target variable data: 


```python
player_data['FANTASY_POINTS'].sample(10)
```




    611    776.6
    554    901.9
    201    624.2
    386    875.9
    791    611.0
    628    563.9
    19     474.0
    853    668.6
    175    410.0
    741    534.0
    Name: FANTASY_POINTS, dtype: float64



Since our target variable contains entirely numeric data, we can answer "no" to "predicting a category?". Note that in some cases you might treat numeric data as categorical data or you might want to separate the numeric data into bins, but for today we are not predicing a category. 

Next up: "Are we predicing a quantity?" In this case, the answer is "yes", since `FANTASY_POINTS` is indeed a quantity. Answering "yes" to this guides us to the [regression](https://en.wikipedia.org/wiki/Regression_analysis) category of models. Regression analysis is a statistical  method used to model the relationship between a continuous numeric dependent variable and one or more independent variables. It aims to predict the value of the dependent variable based on the independent variables. Regression analysis models are a staple in the industry and are often a good starting point due to their simplicity and explainability of results.

The next question is: "Are there fewer than 100,000 samples?" Looking back to the output of `len(player_data)` above, we know that we do have fewer than 100k samples, so we can go down the "yes" path. 

At this point, we have ended up at the question: "Should only a few features be important?" Answering "yes" will take us to [Lasso](https://en.wikipedia.org/wiki/Lasso_(statistics)) and answering "no" will take us to [RidgeRegression](https://en.wikipedia.org/wiki/Ridge_regression). However, both Lasso and Ridge regression in this context are meant to be improvements upon [Ordinary Least Squares (OLS)](https://en.wikipedia.org/wiki/Ordinary_least_squares) linear regression. To clarify, [linear regression](https://en.wikipedia.org/wiki/Linear_regression) is a specific type of regression analysis where the relationship between the dependent variable and the independent variables is assumed to be linear. It finds the best-fitting straight line (called the regression line) through the data points to make predictions. So, instead of using either lasso or ridge regression today, we'll explore the classic OLS linear regression model. 

## Model Characteristics
Ordinary Least Squares (OLS) is the most basic form of linear regression. It aims to minimize the sum of the squared differences between observed and predicted values (the residuals). As a type of linear regression, it assumes a linear relationship between the independent and dependent variables. It can be sensitive to outliers, which can skew the results significantly. It also does not offer regularization, which means it can overfit when there are many predictors or when predictors are highly correlated. However, it is simple and explainable, so it offers a good starting point. 

As mentioned earlier, linear regression assumes that the relationship between the independent variables (features) and the dependent variable (target) can be described by a straight line. This line (also known as the regression line) is represented by an equation of the form:

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
$$

Where:

- \\(y\\) is the predicted value of the target variable
- \\(x_1, x_2, ..., x_n\\) are the input features 
- \\(b_0\\) is the y-intercept (the value of \\(y\\) when all \\(x\\) are zero)
- \\(b_1, b_2, ..., b_n\\) are the coefficients that represent the change in \\(y\\) for a one-unit change in the corresponding \\(x\\), holding all other \\(x\\) constant

We can also rewrite this equation for today's project as: 

$$
FantasyPoints = \beta_0 + \beta_1Height + \beta_2MinutesPlayed + ... + \beta_nPoints
$$

The goal of linear regression is to find the values of \\(b_0\\) and \\(b_1, b_2, ..., b_n\\) that minimize the difference between the predicted \\(FantasyPoints\\) values and the actual \\(FantasyPoints\\) values in our dataset.

*Note: This is a slightly simplified explanation of the underlying mathematics. I highly recommend reading the [Wikipedia page](https://en.wikipedia.org/wiki/Linear_regression#Formulation), or other suitable sources, for a more nuanced understanding of the process.*

Now that we conceptually understand a bit more about how this model works, we can take a quick look at the common assumptions of the model and whether or not those assumptions are satisfied. 

## Verify Assumptions of the Model
Next up, we should verify the underlying assumptions of the machine learning model are satisfied by our particular problem and situation. This step might be tempting to skip, but it can save hours of time in the future and can help ensure your model is generalized. The basic [assumptions of linear regression models](https://en.wikipedia.org/wiki/Linear_regression#Assumptions) generally are: 
 - **Linearity** - The relationship between the target variable and the features is linear.
 - **No Multicollinearity** - The features are not too highly correlated with each other.
 - **Weak Exogeneity** - The features are treated as fixed values, not random variables, and are free from measurement errors. 
 - **Independence of errors** - Residuals are independent of and unrelated to one another. 
 - **Zero Mean of Residuals** - The mean of the residuals is zero or close to zero.
 - **Constant Variance (Homoskedasticity)** - Residuals have constant variance across all levels of the independent variables. 
 
*Note: These are simplified summaries of each assumption. We'll go through each one in a bit more detail later in this article and in future articles, but once again I highly recommend reading a supplemental source for a deeper understanding. Some suitable sources include [Wikipedia](https://en.wikipedia.org/wiki/Linear_regression#Assumptions), [Stanford](https://web.stanford.edu/class/stats191/markdown/Chapter8/Simple_Linear_Regression_Assumptions.html), and [Statistics by Jim](https://statisticsbyjim.com/regression/ols-linear-regression-assumptions/).*

You might notice that the first three assumptions pertain to the features and the last three assumptions pertain to the residuals. Residuals are the difference between the predicted target variable and the actual target variable. This means that we must be able to generate predictions and therefore have trained the model before evaluating those assumptions. So we can go through each of the first three assumptions (linearity, no multicollinearity, and weak exogeneity) today and will leave the other three for a future article.  

### Linearity
The first assumption we'll look at is whether the relationship between the independent and dependent variables is linear. Looking at the basic OLS linear regression equation from the [earlier section](#model-characteristics), we can see that the coefficients (parameters), random error, and variables are linear. Linear regression models can model curvature to some extent (in the variables, not the parameters), but for today we'll focus on a strictly linear relationship. 

There are a few common methods to detect linearity: 

1. **Scatterplot** - We can use a simple scatterplot to evaluate the relationship between the target variable and the features. For projects with only one feature, we would only need one scatterplot. Since today's project has multiple features, we'll need one scatterplot for each feature. 
2. **LOWESS Curve** - This is the same idea as the scatterplot, but adds in a [LOWESS curve](https://en.wikipedia.org/wiki/Local_regression) to more robustly evaluate the linearity assumption. 
3. **Residuals Plot** - If the residuals plot shows curves or trends, it could mean that the linearity assumption is not satisfied. 

Since we're looking for a strictly linear relationship today, we can start with the same [`pairplot()` function](https://seaborn.pydata.org/generated/seaborn.pairplot.html) that we did in [Part 4](/2024-07-29-basketball-visualizations/). Since we're only looking at the relationship between the target variable (`FANTASY_POINTS`) and each feature, we can make a slight modification to show only the target variable plots.


```python
pairplot = sns.pairplot(data=player_data, x_vars=features, y_vars=target)
```


    
![png](/assets/img/posts/2024-09-13-basketball-train-ols/output_25_0.png)
    


A LOWESS (Locally Weighted Scatterplot Smoothing) curve is a type of [local regression](https://en.wikipedia.org/wiki/Local_regression) that fits a smooth curve through a scatter plot to visualize relationships between variables. It's particularly useful for identifying non-linear patterns in data by fitting simple models to localized subsets. LOWESS curves are flexible and can capture complex relationships, making them a helpful visual aid for determining the linearity of each pairing.

We can use Seaborn's [`regplot()` function](https://seaborn.pydata.org/generated/seaborn.regplot.html) with the `lowess=True` parameter to add a LOWESS curve to each scatterplot: 


```python
def loess_reg(x, y, **kwargs):
    sns.regplot(x=x, y=y, lowess=True, line_kws={'color': 'red'}, **kwargs)


pairplot = sns.pairplot(data=player_data, x_vars=features, y_vars=target)
pairplot.map(loess_reg)
```




    <seaborn.axisgrid.PairGrid at 0x74cd86773280>




    
![png](/assets/img/posts/2024-09-13-basketball-train-ols/output_27_1.png)
    


### No Multicollinearity
The next assumption that we'll examine is the lack of multicollinearity. Multicollinearity occurs when the features in a regression model are highly correlated with each other. This can make it challenging to determine individual variable effects, lead to unstable coefficient estimates, and increase standard errors. As a result, some variables may appear statistically insignificant when they should be significant. 

There are a few common ways to detect multicollinearity; we'll look at the top two: 
1. **Correlation Coefficient** - We can check the correlation coefficient between each feature. 
2. **Variance Inflation Factor** - We can check the [variance inflation factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) after training the model.

To check the correlation coefficient, we can use the same method as in [Part 3](https://www.pineconedata.com/2024-07-29-basketball-visualizations/#generate-correlation-matrix). 
We can check the correlation coefficients before training the model using a correlation matrix. As a reminder, a correlation matrix displays the correlation coefficients between all pairs of variables, with values ranging from -1 to 1. Strong correlations (when the absolute value of the correlation coefficient is typically above 0.8 or 0.9) suggest potential multicollinearity.


```python
plt.figure(figsize=(12, 10))
correlation_matrix = player_data[features].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
```




    <Axes: >




    
![png](/assets/img/posts/2024-09-13-basketball-train-ols/output_29_1.png)
    


From this chart, we can see that the variables with the strongest correlations are: 
 - `FIELD_GOALS_MADE` and `TWO_POINTS_MADE` with a correlation coefficient of `0.86`
 - `FIELD_GOALS_MADE` and `POINTS` with a correlation coefficient of `0.97`

When you encounter variable pairings with strong correlations in your correlation matrix, there are several ways to deal with them:

1. **Remove one of the correlated variables** - If two variables are highly correlated, they might provide similar information. Keeping both may lead to redundancy and potential issues with multicollinearity. We could choose the variable that we believe is more important or relevant to your analysis.
2. **Combine correlated variables** - We might create a new feature that combines the information from both correlated variables. This could be done through a variety of methods such as  by creating a composite score.
3. **Use regularization techniques** - Methods like Ridge Regression (L2 regularization) or Lasso Regression (L1 regularization) can help mitigate the effects of multicollinearity by adding a penalty term to the model that discourages large coefficients. We could also apply regularization to the features separately, before training the OLS model.
4. **Collect more data** - Sometimes, multicollinearity can be reduced by increasing the sample size, which may help differentiate the effects of correlated variables. This is more situational and will not apply in all scenarios.

Since we're interested in seeing how the model behaves with a simple application of OLS linear regression, we'll proceed without removing, combining, or regularizing any of the features. However, we should keep this information in mind in the future since we might want to retrain the model without one or more of the features. 

### Weak Exogeneity
Weak exogeneity is another crucial assumption in linear regression that we need to verify. This assumption essentially means that the predictor variables (our independent variables) can be treated as fixed values, rather than random variables. In other words, we assume that our predictor variables are not influenced by the dependent variable or by external factors that also affect the dependent variable.

Practically speaking, weak exogeneity implies that our predictor variables are error-free, meaning they are not contaminated with measurement errors. While this assumption may not always be realistic in many real-world settings, it's an important simplification that allows us to use standard linear regression techniques.

It's worth noting that dropping this assumption leads to significantly more complex models known as errors-in-variables models. These models account for measurement errors in the predictor variables but are considerably more challenging to implement and interpret.

There is no direct statistical test for weak exogeneity, so we'll treat this as more of a logical check than a mathematical one. For our basketball player statistics model, weak exogeneity would mean that the statistics we're using as predictors (such as minutes played, field goals attempted, etc.) are not themselves influenced by the player's fantasy points or by unmeasured factors that also affect fantasy points. This makes logical sense based on our domain knowledge and understanding of how the data was collected, so we'll consider this assumption satisfied. 

# Train the Model
Now that we've covered the basics of machine learning and verified the suitability of our chosen model, we're ready to move on to the exciting part: training our linear regression model! This process involves several key steps that will help us build a robust and accurate predictive model for our basketball player statistics.

## Define the Variables 
As a reminder, in an earlier section we defined the features and target variable. We'll label the features (independent variables) as `X` and the target (dependent) variable as `y` for conciseness.


```python
X = player_data[features]
y = player_data[target]
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



## Create Training and Testing Splits
Now that we have our variables defined, we can create the training and testing splits. This involves dividing our dataset into two parts: a set for training and a set for testing. The train set will be used to train the model, and the test set will be used exclusively for testing and evaluating the model after training. 

Why wouldn't we just use all of the data for training?
1. **Model Evaluation** - The test set allows us to evaluate how well our model performs on unseen data, giving us a more realistic estimate of its performance in real-world scenarios.
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
You might notice that if you run the `train_test_split()` for a second time, there are different rows of data included in each split. This is because the data is shuffled before splitting, and the shuffling is not guaranteed to be reproducible by default.


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



This can mean that the model is trained and tested on different datasets each time that you run it. That's often a good thing, but it can be better to have reproducible results for initial creation and evaluation of the model (especially if you want to follow along with this guide). 

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




<style>#sk-container-id-1 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-1 {
  color: var(--sklearn-color-text);
}

#sk-container-id-1 pre {
  padding: 0;
}

#sk-container-id-1 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-1 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-1 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-1 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-1 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-1 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-1 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-1 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-1 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-1 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-1 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-1 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-1 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-1 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-1 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-1 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-1 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-1 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-1 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-1 div.sk-label label.sk-toggleable__label,
#sk-container-id-1 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-1 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-1 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-1 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-1 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-1 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-1 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-1 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-1 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-1 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-1 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-1 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-1 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>



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
coef_series_simple = coef_series[coef_series > 0.0001]
coef_string_simple = "\n\t\t + ".join(f"{coef:.4f} * {feat}" for feat, coef in coef_series_simple.items())
print(f'{target} = {coef_string_simple} + {linear_reg_model.intercept_} + error')
```

    FANTASY_POINTS = 1.6667 * FIELD_GOALS_MADE
    		 + 1.3333 * THREE_POINTS_MADE
    		 + 0.3333 * TWO_POINTS_MADE
    		 + 1.0000 * FREE_THROWS_MADE
    		 + 1.2000 * TOTAL_REBOUNDS
    		 + 1.5000 * ASSISTS
    		 + 2.0000 * STEALS
    		 + 2.0000 * BLOCKS + -6.821210263296962e-13 + error


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
Since we have ended up with essentially the same equation as the original fantasy points calculation, we can logically expect our model to perform pretty well in the next phase of model evaluation. So, we can also train an alternate model with a few of the features removed for comparison. First, let's remove all three of the features with high correlation coefficients: 


```python
X_alt = player_data[features].drop(columns=['FIELD_GOALS_MADE', 'TWO_POINTS_MADE', 'POINTS'])
```

*Note: this is more features than you would likely want to remove in a real-world scenario, but removing too many features will give us an opportunity to compare a less-than-perfect model to a perfect model in the model evaluation phase.*

Our target variable is unchanged, so we can create alternate training and test splits: 


```python
X_train_alt, X_test_alt, y_train_alt, y_test_alt = train_test_split(X_alt, y, random_state=314)
```

We can now train an alternate model using these new test splits: 


```python
ols_alt = LinearRegression()
ols_alt.fit(X_train_alt, y_train_alt)
```




<style>#sk-container-id-2 {
  /* Definition of color scheme common for light and dark mode */
  --sklearn-color-text: black;
  --sklearn-color-line: gray;
  /* Definition of color scheme for unfitted estimators */
  --sklearn-color-unfitted-level-0: #fff5e6;
  --sklearn-color-unfitted-level-1: #f6e4d2;
  --sklearn-color-unfitted-level-2: #ffe0b3;
  --sklearn-color-unfitted-level-3: chocolate;
  /* Definition of color scheme for fitted estimators */
  --sklearn-color-fitted-level-0: #f0f8ff;
  --sklearn-color-fitted-level-1: #d4ebff;
  --sklearn-color-fitted-level-2: #b3dbfd;
  --sklearn-color-fitted-level-3: cornflowerblue;

  /* Specific color for light theme */
  --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, white)));
  --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, black)));
  --sklearn-color-icon: #696969;

  @media (prefers-color-scheme: dark) {
    /* Redefinition of color scheme for dark theme */
    --sklearn-color-text-on-default-background: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-background: var(--sg-background-color, var(--theme-background, var(--jp-layout-color0, #111)));
    --sklearn-color-border-box: var(--sg-text-color, var(--theme-code-foreground, var(--jp-content-font-color1, white)));
    --sklearn-color-icon: #878787;
  }
}

#sk-container-id-2 {
  color: var(--sklearn-color-text);
}

#sk-container-id-2 pre {
  padding: 0;
}

#sk-container-id-2 input.sk-hidden--visually {
  border: 0;
  clip: rect(1px 1px 1px 1px);
  clip: rect(1px, 1px, 1px, 1px);
  height: 1px;
  margin: -1px;
  overflow: hidden;
  padding: 0;
  position: absolute;
  width: 1px;
}

#sk-container-id-2 div.sk-dashed-wrapped {
  border: 1px dashed var(--sklearn-color-line);
  margin: 0 0.4em 0.5em 0.4em;
  box-sizing: border-box;
  padding-bottom: 0.4em;
  background-color: var(--sklearn-color-background);
}

#sk-container-id-2 div.sk-container {
  /* jupyter's `normalize.less` sets `[hidden] { display: none; }`
     but bootstrap.min.css set `[hidden] { display: none !important; }`
     so we also need the `!important` here to be able to override the
     default hidden behavior on the sphinx rendered scikit-learn.org.
     See: https://github.com/scikit-learn/scikit-learn/issues/21755 */
  display: inline-block !important;
  position: relative;
}

#sk-container-id-2 div.sk-text-repr-fallback {
  display: none;
}

div.sk-parallel-item,
div.sk-serial,
div.sk-item {
  /* draw centered vertical line to link estimators */
  background-image: linear-gradient(var(--sklearn-color-text-on-default-background), var(--sklearn-color-text-on-default-background));
  background-size: 2px 100%;
  background-repeat: no-repeat;
  background-position: center center;
}

/* Parallel-specific style estimator block */

#sk-container-id-2 div.sk-parallel-item::after {
  content: "";
  width: 100%;
  border-bottom: 2px solid var(--sklearn-color-text-on-default-background);
  flex-grow: 1;
}

#sk-container-id-2 div.sk-parallel {
  display: flex;
  align-items: stretch;
  justify-content: center;
  background-color: var(--sklearn-color-background);
  position: relative;
}

#sk-container-id-2 div.sk-parallel-item {
  display: flex;
  flex-direction: column;
}

#sk-container-id-2 div.sk-parallel-item:first-child::after {
  align-self: flex-end;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:last-child::after {
  align-self: flex-start;
  width: 50%;
}

#sk-container-id-2 div.sk-parallel-item:only-child::after {
  width: 0;
}

/* Serial-specific style estimator block */

#sk-container-id-2 div.sk-serial {
  display: flex;
  flex-direction: column;
  align-items: center;
  background-color: var(--sklearn-color-background);
  padding-right: 1em;
  padding-left: 1em;
}


/* Toggleable style: style used for estimator/Pipeline/ColumnTransformer box that is
clickable and can be expanded/collapsed.
- Pipeline and ColumnTransformer use this feature and define the default style
- Estimators will overwrite some part of the style using the `sk-estimator` class
*/

/* Pipeline and ColumnTransformer style (default) */

#sk-container-id-2 div.sk-toggleable {
  /* Default theme specific background. It is overwritten whether we have a
  specific estimator or a Pipeline/ColumnTransformer */
  background-color: var(--sklearn-color-background);
}

/* Toggleable label */
#sk-container-id-2 label.sk-toggleable__label {
  cursor: pointer;
  display: block;
  width: 100%;
  margin-bottom: 0;
  padding: 0.5em;
  box-sizing: border-box;
  text-align: center;
}

#sk-container-id-2 label.sk-toggleable__label-arrow:before {
  /* Arrow on the left of the label */
  content: "▸";
  float: left;
  margin-right: 0.25em;
  color: var(--sklearn-color-icon);
}

#sk-container-id-2 label.sk-toggleable__label-arrow:hover:before {
  color: var(--sklearn-color-text);
}

/* Toggleable content - dropdown */

#sk-container-id-2 div.sk-toggleable__content {
  max-height: 0;
  max-width: 0;
  overflow: hidden;
  text-align: left;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content pre {
  margin: 0.2em;
  border-radius: 0.25em;
  color: var(--sklearn-color-text);
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-toggleable__content.fitted pre {
  /* unfitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

#sk-container-id-2 input.sk-toggleable__control:checked~div.sk-toggleable__content {
  /* Expand drop-down */
  max-height: 200px;
  max-width: 100%;
  overflow: auto;
}

#sk-container-id-2 input.sk-toggleable__control:checked~label.sk-toggleable__label-arrow:before {
  content: "▾";
}

/* Pipeline/ColumnTransformer-specific style */

#sk-container-id-2 div.sk-label input.sk-toggleable__control:checked~label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-label.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator-specific style */

/* Colorize estimator box */
#sk-container-id-2 div.sk-estimator input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted input.sk-toggleable__control:checked~label.sk-toggleable__label {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

#sk-container-id-2 div.sk-label label.sk-toggleable__label,
#sk-container-id-2 div.sk-label label {
  /* The background is the default theme color */
  color: var(--sklearn-color-text-on-default-background);
}

/* On hover, darken the color of the background */
#sk-container-id-2 div.sk-label:hover label.sk-toggleable__label {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-unfitted-level-2);
}

/* Label box, darken color on hover, fitted */
#sk-container-id-2 div.sk-label.fitted:hover label.sk-toggleable__label.fitted {
  color: var(--sklearn-color-text);
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Estimator label */

#sk-container-id-2 div.sk-label label {
  font-family: monospace;
  font-weight: bold;
  display: inline-block;
  line-height: 1.2em;
}

#sk-container-id-2 div.sk-label-container {
  text-align: center;
}

/* Estimator-specific */
#sk-container-id-2 div.sk-estimator {
  font-family: monospace;
  border: 1px dotted var(--sklearn-color-border-box);
  border-radius: 0.25em;
  box-sizing: border-box;
  margin-bottom: 0.5em;
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-0);
}

#sk-container-id-2 div.sk-estimator.fitted {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-0);
}

/* on hover */
#sk-container-id-2 div.sk-estimator:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-2);
}

#sk-container-id-2 div.sk-estimator.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-2);
}

/* Specification for estimator info (e.g. "i" and "?") */

/* Common style for "i" and "?" */

.sk-estimator-doc-link,
a:link.sk-estimator-doc-link,
a:visited.sk-estimator-doc-link {
  float: right;
  font-size: smaller;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1em;
  height: 1em;
  width: 1em;
  text-decoration: none !important;
  margin-left: 1ex;
  /* unfitted */
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
  color: var(--sklearn-color-unfitted-level-1);
}

.sk-estimator-doc-link.fitted,
a:link.sk-estimator-doc-link.fitted,
a:visited.sk-estimator-doc-link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
div.sk-estimator:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover,
div.sk-label-container:hover .sk-estimator-doc-link:hover,
.sk-estimator-doc-link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

div.sk-estimator.fitted:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover,
div.sk-label-container:hover .sk-estimator-doc-link.fitted:hover,
.sk-estimator-doc-link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

/* Span, style for the box shown on hovering the info icon */
.sk-estimator-doc-link span {
  display: none;
  z-index: 9999;
  position: relative;
  font-weight: normal;
  right: .2ex;
  padding: .5ex;
  margin: .5ex;
  width: min-content;
  min-width: 20ex;
  max-width: 50ex;
  color: var(--sklearn-color-text);
  box-shadow: 2pt 2pt 4pt #999;
  /* unfitted */
  background: var(--sklearn-color-unfitted-level-0);
  border: .5pt solid var(--sklearn-color-unfitted-level-3);
}

.sk-estimator-doc-link.fitted span {
  /* fitted */
  background: var(--sklearn-color-fitted-level-0);
  border: var(--sklearn-color-fitted-level-3);
}

.sk-estimator-doc-link:hover span {
  display: block;
}

/* "?"-specific style due to the `<a>` HTML tag */

#sk-container-id-2 a.estimator_doc_link {
  float: right;
  font-size: 1rem;
  line-height: 1em;
  font-family: monospace;
  background-color: var(--sklearn-color-background);
  border-radius: 1rem;
  height: 1rem;
  width: 1rem;
  text-decoration: none;
  /* unfitted */
  color: var(--sklearn-color-unfitted-level-1);
  border: var(--sklearn-color-unfitted-level-1) 1pt solid;
}

#sk-container-id-2 a.estimator_doc_link.fitted {
  /* fitted */
  border: var(--sklearn-color-fitted-level-1) 1pt solid;
  color: var(--sklearn-color-fitted-level-1);
}

/* On hover */
#sk-container-id-2 a.estimator_doc_link:hover {
  /* unfitted */
  background-color: var(--sklearn-color-unfitted-level-3);
  color: var(--sklearn-color-background);
  text-decoration: none;
}

#sk-container-id-2 a.estimator_doc_link.fitted:hover {
  /* fitted */
  background-color: var(--sklearn-color-fitted-level-3);
}
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>LinearRegression()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;LinearRegression<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.LinearRegression.html">?<span>Documentation for LinearRegression</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>LinearRegression()</pre></div> </div></div></div></div>



We can print the model equation for this alternate model as well: 


```python
coef_series_alt = pd.Series(data=ols_alt.coef_, index=ols_alt.feature_names_in_)
coef_series_alt = coef_series_alt[coef_series_alt > 0.0001]
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


We can see that the model coefficients and the y-intercept are substantially different from the model we originally trained. We won't know if this alternate model performs as well as the original one until we evaluate each model in the next article. 

# Wrap Up
In this series, we've built a new dataset by acquiring and then combining the NCAA women's basketball player information dataset with the Yahoo Sports player statistics dataset. We laid the groundwork for data analysis by cleaning and preprocessing the combined player data, and then expanded upon it by engineering a few new features. In the previous part, we took a closer look at the underlying data in each column and created visualizations to identify the relationship between various parameters. In today's article, we learned how to select an appropriate machine learning model, properly split our data set into train and test subsets, and trained the model. In the next section, we'll move on to evaluating the model's performance.

<div id="sources"></div>

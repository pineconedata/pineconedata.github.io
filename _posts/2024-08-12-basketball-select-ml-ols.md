---
layout: post
title: "Selecting a Machine Learning Model"
subtitle: "Outlier or Caitlin Clark? [Part 6]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, linear regression]
share-title: "Selecting a Machine Learning Model: Outlier or Caitlin Clark? [Part 6]" 
share-description: Interested in learning how to select an approriate machine learning model? Learn the intricacies of how to select a machine learning model for your dataset in the latest installment of this data science series that is perfect for beginner data scientists and Python enthusiasts.
thumbnail-img: "https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg"
share-img: /assets/img/posts/2024-09-13-basketball-train-ols/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll cover the basics of machine learning and examine how to select an appropriate machine learning model. This is the sixth part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

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
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
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

<div class="email-subscription-container"></div>

### Final Objective
In summary, our ultimate objective is to train a machine learning model to predict `FANTASY_POINTS` (the target variable) based on `Height`, `MINUTES_PLAYED`, `FIELD_GOALS_MADE`, `THREE_POINTS_MADE`, `TWO_POINTS_MADE`, `FREE_THROWS_MADE`, `TOTAL_REBOUNDS`, `ASSISTS`, `TURNOVERS`, `STEALS`, `BLOCKS`, `FOULS`, and `POINTS` (the features).

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

# Wrap Up
In this series, we've built a new dataset by acquiring and then combining the NCAA women's basketball player information dataset with the Yahoo Sports player statistics dataset. We laid the groundwork for data analysis by cleaning and preprocessing the combined player data, and then expanded upon it by engineering a few new features. In the previous part, we took a closer look at the underlying data in each column and created visualizations to identify the relationship between various parameters. In today's article, we learned how to select an appropriate machine learning model, properly split our data set into train and test subsets, and trained the model. In the next section, we'll move on to evaluating the model's performance.

<div class="email-subscription-container"></div>
<div id="sources"></div>

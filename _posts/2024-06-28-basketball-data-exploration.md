---
layout: post
title: "Data Exploration"
subtitle: "Outlier or Caitlin Clark? [Part 4]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, data visualization]
thumbnail-img: /assets/img/posts/2024-06-28-basketball-data-exploration/compass.jpg
share-title: "Data Exploration: Outlier or Caitlin Clark? [Part 4]" 
share-description: Interested in exploring and selecting appropriate features for your data? Learn how to determine the relationship between metrics, explore correlation matrices, and select meaningful features in the latest installment of this data science series that is perfect for beginner data scientists and Python enthusiasts.
share-img: /assets/img/posts/2024-06-28-basketball-data-exploration/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll demonstrate how to determine the relationship between metrics and select features. This is the fourth part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

<div id="toc"></div>

# Getting Started
First, let's take a look at an overview of this data science project. If you're already familiar with it, feel free to skip to the [next section](#identify-relationships-between-variables).

## Project Overview

As a reminder, the dataset we'll be using in this project contains individual basketball player statistics (such as total points scored and blocks made) for the 2023-2024 NCAA women's basketball season. Here's a brief description of each major step that we'll go through for this project: 

![the steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/project_steps.png "the steps for this data science project")

1. **Data Acquisition** - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. **Data Cleaning** - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. **Data Preprocessing** - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. **Feature Engineering** - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.
5. **Data Exploration** - This step focuses on analyzing and visualizing the dataset to uncover patterns, relationships, and general trends and is a helpful preliminary step before deeper analysis.
6. **Creating Visualizations** - This step involves identifying the relationships between various parameters (such as height and blocked shots) and generating meaningful visualizations (such as bar charts, scatterplots, and candlestick charts).
7. **Machine Learning** - This step focuses on selecting, training, and evaluating a machine learning model. For this project, the model will identify the combination of individual player statistics that correlates with optimal performance. 

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Due to the size of this project, today we'll cover the fifth step: exploratory data analysis.

## Dependencies
Since this is the fourth installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first article of the series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. 

You'll want to have the latest version of [Python](https://www.python.org/) installed with the following packages: 
  - [pandas](https://pandas.pydata.org/docs/)
  - [requests](https://requests.readthedocs.io/en/latest/)
  - [json](https://docs.python.org/3/library/json.html)
  - [os](https://docs.python.org/3/library/os.html)
  - [numpy](https://numpy.org/doc/)
  - [matplotlib](https://matplotlib.org/)
  - [seaborn](https://seaborn.pydata.org/)
  - [plotly](https://plotly.com/)
  - [scipy](https://scipy.org/)
  
For today's guide specifically, we'll want to import the following packages: 


```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.io as pio
```

Since I'm creating these graphs using [JupyterLab](https://jupyter.org/), I'll also make sure the `jupyterlab-plotly` extension is installed and will specify that plotly charts should display as an iframe. 


```python
pio.renderers.default = 'iframe'
```

## Import Data
In [Part 3](/2024-05-30-basketball-feature_engineering/) of this series, we engineered new features for our dataset, which is stored in a dataframe named `player_data`. If you want to follow along with the code examples in this article, it's recommended to import the `player_data` dataframe before proceeding. 


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


That's all the setup steps needed, so we're ready to move on to identifying the relationships between various variables in our dataset.

# Identify Relationships between Variables
In this section, we'll explore the dataset to understand relationships between various parameters. We'll use the [pandas `describe()` function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) for a statistical summary, create a correlation matrix to visualize variable relationships, and generate a pairwise plot for a detailed view. This analysis will help us identify patterns and help us select meaningful visualizations for this data science project.

## Describe the Dataset
The first step is to get a statistical summary of our dataset. The [`describe()` function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.describe.html) in pandas provides a high-level statistical summary for each numerical column of our dataset, including: 

1. **Count** - The number of non-null entries in a column.
2. **Mean**- The average value of all entries in a column.
3. **Standard Deviation** - A measure of the amount of variation or dispersion of the entries in a column.
4. **Minimum** - The smallest entry in a column.
5. **25th percentile** - The value below which a quarter of the entries in a column fall.
6. **Median** - The middle value in a column when the entries are sorted in ascending or descending order.
7. **75th percentile** - The value below which three quarters of the entries in a column fall.
8. **Maximum** - The largest entry in a column.


```python
player_data.describe()
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
      <th>GAMES</th>
      <th>MINUTES_PLAYED</th>
      <th>FIELD_GOALS_MADE</th>
      <th>FIELD_GOAL_ATTEMPTS</th>
      <th>FIELD_GOAL_PERCENTAGE</th>
      <th>THREE_POINTS_MADE</th>
      <th>THREE_POINT_ATTEMPTS</th>
      <th>THREE_POINT_PERCENTAGE</th>
      <th>FREE_THROWS_MADE</th>
      <th>...</th>
      <th>TWO_POINT_PERCENTAGE</th>
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
      <th>count</th>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>841.000000</td>
      <td>900.00000</td>
      <td>...</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
      <td>900.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>70.643333</td>
      <td>29.402222</td>
      <td>841.816667</td>
      <td>121.420000</td>
      <td>279.767778</td>
      <td>43.482444</td>
      <td>27.165556</td>
      <td>81.780000</td>
      <td>28.875505</td>
      <td>62.66000</td>
      <td>...</td>
      <td>47.129786</td>
      <td>28.611697</td>
      <td>2.215790</td>
      <td>11.306896</td>
      <td>2.334072</td>
      <td>1.300121</td>
      <td>0.591105</td>
      <td>5.203494</td>
      <td>1.059605</td>
      <td>664.424111</td>
    </tr>
    <tr>
      <th>std</th>
      <td>3.460328</td>
      <td>2.172842</td>
      <td>161.470966</td>
      <td>47.015549</td>
      <td>101.044999</td>
      <td>6.988500</td>
      <td>24.575016</td>
      <td>66.603741</td>
      <td>12.276598</td>
      <td>33.83484</td>
      <td>...</td>
      <td>7.335796</td>
      <td>5.005458</td>
      <td>0.591091</td>
      <td>4.269809</td>
      <td>1.365048</td>
      <td>0.616440</td>
      <td>0.587401</td>
      <td>2.233181</td>
      <td>0.569156</td>
      <td>185.177943</td>
    </tr>
    <tr>
      <th>min</th>
      <td>62.000000</td>
      <td>18.000000</td>
      <td>223.000000</td>
      <td>19.000000</td>
      <td>42.000000</td>
      <td>22.600000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>0.000000</td>
      <td>3.00000</td>
      <td>...</td>
      <td>23.255814</td>
      <td>8.259259</td>
      <td>0.413793</td>
      <td>1.766667</td>
      <td>0.043478</td>
      <td>0.096774</td>
      <td>0.000000</td>
      <td>1.294118</td>
      <td>0.040000</td>
      <td>189.900000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>68.000000</td>
      <td>29.000000</td>
      <td>748.250000</td>
      <td>89.000000</td>
      <td>208.750000</td>
      <td>38.775000</td>
      <td>4.000000</td>
      <td>18.000000</td>
      <td>24.700000</td>
      <td>38.00000</td>
      <td>...</td>
      <td>42.343026</td>
      <td>26.000000</td>
      <td>1.800000</td>
      <td>8.341954</td>
      <td>1.250000</td>
      <td>0.833333</td>
      <td>0.148148</td>
      <td>3.412197</td>
      <td>0.643591</td>
      <td>544.425000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>71.000000</td>
      <td>30.000000</td>
      <td>860.000000</td>
      <td>119.000000</td>
      <td>278.000000</td>
      <td>42.850000</td>
      <td>22.500000</td>
      <td>75.000000</td>
      <td>31.400000</td>
      <td>57.00000</td>
      <td>...</td>
      <td>46.963563</td>
      <td>29.360753</td>
      <td>2.200000</td>
      <td>11.266963</td>
      <td>2.095262</td>
      <td>1.233333</td>
      <td>0.375000</td>
      <td>4.745370</td>
      <td>0.968990</td>
      <td>651.100000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>73.000000</td>
      <td>31.000000</td>
      <td>963.000000</td>
      <td>151.000000</td>
      <td>345.000000</td>
      <td>47.500000</td>
      <td>44.000000</td>
      <td>129.250000</td>
      <td>36.200000</td>
      <td>83.00000</td>
      <td>...</td>
      <td>51.761573</td>
      <td>31.976562</td>
      <td>2.633333</td>
      <td>14.146205</td>
      <td>3.210598</td>
      <td>1.666667</td>
      <td>0.933333</td>
      <td>6.879464</td>
      <td>1.362177</td>
      <td>774.075000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>80.000000</td>
      <td>34.000000</td>
      <td>1230.000000</td>
      <td>332.000000</td>
      <td>719.000000</td>
      <td>66.700000</td>
      <td>168.000000</td>
      <td>437.000000</td>
      <td>100.000000</td>
      <td>203.00000</td>
      <td>...</td>
      <td>68.224299</td>
      <td>38.437500</td>
      <td>3.781250</td>
      <td>31.875000</td>
      <td>8.812500</td>
      <td>4.677419</td>
      <td>3.433333</td>
      <td>15.312500</td>
      <td>3.214286</td>
      <td>1716.800000</td>
    </tr>
  </tbody>
</table>
<p>8 rows × 33 columns</p>
</div>



This gives an understanding of the distribution of data and characteristics of each column. This helps us identify any outliers or missing data, as well as assess how spread out the data is. It's often recommended to take a few minutes to scan through the statistics for each column to get a better understanding of each one and to quickly check for any issues. For example, you might notice that the `count` for the `THREE_POINT_PERCENTAGE` column is lower than the other columns. If you've read [Part 2](https://www.pineconedata.com/2024-05-02-basketball-data-cleaning-preprocessing/#handle-missing-three-point-percentages) of this series, you might remember that some rows are missing a three point percentage in cases where a player had zero three point goals attempted, so it makes sense that the `count` of non-null `THREE_POINT_PERCENTAGE` rows is `841` instead of `900`. 

## Feature Selection
Next, it’d be helpful to generate a few charts to explore the relationships between the various player statistics. However, including too many metrics can slow down the plot generation process, so let's limit the number of numerical columns. (To be clear, this step is entirely optional and it is possible to generate a plot with all of these columns.)

But how do we choose the right columns? This is often an entire step of a data science project and is referred to as [feature selection](https://en.wikipedia.org/wiki/Feature_selection). There are plenty of feature selection methods, but identifying which features are best often depends on your specific use case. For example,  if you’re developing a machine learning model that optimizes for defensive players, you might want to include features such as steals, blocks, and rebounds. However, if you’re optimizing for offensive players, then you might focus on features like points and assists. Other features, such as turnovers and fouls, might be included in both cases. 

For today’s purpose, we don’t have a specific use-case in mind and are instead more focused on exploring the dataset and creating interesting visualizations. So, let’s make an educated guess on some features that might be similar enough to choose a few of them. For example, for each points metric (two-point goals, three-point goals, total field goals, and free throws) there are three columns (goals made, goals attempted, and goal percentage). 


```python
player_data[['PLAYER_NAME', 'TWO_POINTS_MADE', 'TWO_POINT_ATTEMPTS', 'TWO_POINT_PERCENTAGE']].head()
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
      <th>TWO_POINTS_MADE</th>
      <th>TWO_POINT_ATTEMPTS</th>
      <th>TWO_POINT_PERCENTAGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kiara Jackson</td>
      <td>100</td>
      <td>222</td>
      <td>45.045045</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Raven Johnson</td>
      <td>78</td>
      <td>157</td>
      <td>49.681529</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gina Marxen</td>
      <td>30</td>
      <td>79</td>
      <td>37.974684</td>
    </tr>
    <tr>
      <th>3</th>
      <td>McKenna Hofschild</td>
      <td>176</td>
      <td>360</td>
      <td>48.888889</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kaylah Ivey</td>
      <td>15</td>
      <td>55</td>
      <td>27.272727</td>
    </tr>
  </tbody>
</table>
</div>



These columns are distinct and we already suspect that they are interrelated. The goal percentage is directly calculated by dividing goals made by goals attempted, so we likely don’t need to include that for each metric. The number of goals made is likely related to the number of goals attempted (you cannot score a goal without attempting it), so we could use just one of those two columns as a proxy in today’s visualizations. You can use either, but since goals made is directly used in the calculation for fantasy points, let’s go with that one. 

In summary, we can collapse the goals made, goals attempted, and goal percentage columns down into just the goals made columns. Using similar logic, we can include total rebounds (excluding offensive and defensive rebounds), minutes played (instead of games played), and remove certain calculated columns (like the per-game metrics and assist-to-turnover ratio). This dramatically reduces the number of numerical columns for these initial exploratory plots while still preserving critical features.

Here's the final list of numerical columns we'll use for the first few visualizations: 


```python
numerical_columns = ['Height', 'MINUTES_PLAYED', 'FIELD_GOALS_MADE', 
                     'THREE_POINTS_MADE', 'TWO_POINTS_MADE', 'FREE_THROWS_MADE',
                     'TOTAL_REBOUNDS', 'ASSISTS', 'TURNOVERS', 
                     'STEALS', 'BLOCKS', 'FOULS', 'POINTS', 'FANTASY_POINTS']
```

<div class="email-subscription-container"></div>

## Correlation vs Causation
Note that [correlation does not imply causation](https://en.wikipedia.org/wiki/Correlation_does_not_imply_causation). Just because there is a statistical association between two variables does not mean that a change in one variable actually *causes* a change in the other variable. There is an entire website containing [spurious correlations](https://tylervigen.com/spurious-correlations), but here's one example of two variables that are strongly correlated, but are unlikely to have a cause-and-effect relationship: the divorce rate in Maine and the per capita consumption of margarine. 

![correlation does not imply causation](https://tylervigen.com/spurious/correlation/image/5920_per-capita-consumption-of-margarine_correlates-with_the-divorce-rate-in-maine.svg "correlation does not imply causation")

So keep in mind that correlation is not the same as causation. That said, knowing which variables are correlated with each other is still useful for this project. By examining the correlation matrix, we can identify which statistics tend to increase or decrease together, which can give us insights about the data. This can be particularly useful for feature selection in machine learning models, as it helps to avoid situations where multiple features are highly correlated with each other ([multicollinearity](https://en.wikipedia.org/wiki/Multicollinearity)).

## Generate Correlation Matrix

A [correlation matrix](https://en.wikipedia.org/wiki/Correlation#Correlation_matrices) is a table showing the correlation coefficients between many variables. Each cell in the table shows the correlation between two variables. The value is in the range of `-1` to `1` and each cell color indicates the strength and direction of the correlation between two variables. If two variables have a high correlation, it means that when one variable changes, the other variable tends to also change in a specific direction. Values closer to zero indicate a weak correlation, where a change in one variable does not correlate with a change in the other variable. 

### Correlation Matrix of Two Features

Before creating the full correlation matrix with all of the numerical columns, let's take a quick look at a correlation matrix that only includes two metrics that we already suspect are correlated: field goals made and field goals attempted. We'll use the [pandas `corr()` method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.corr.html) to create the correlation matrix itself and then use a [Seaborn heatmap](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to visualize it.


```python
plt.figure(figsize=(12, 10))
correlation_matrix = player_data[['FIELD_GOALS_MADE', 'FIELD_GOAL_ATTEMPTS']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Player Statistics')
plt.show()
```


    
![png](/assets/img/posts/2024-06-28-basketball-data-exploration/output_28_0.png)
    


The simplest example of a strong correlation in the correlation matrix is the one-to-one relationship, where the values across one diagonal show the correlation of a variable with itself, so the correlation coefficient is exactly 1. Here we can see that the correlation coefficient between field goals attempted and field goals made is `0.91`. This is pretty close to `1.00`, so these two variables are strongly correlated. As one value goes up, the other is also likely to go up, and vice versa. 

You might also notice that we only need half of this matrix - either the half above the diagonal line of `1.00` values or the half below the diagonal line. The diagonal line of correlation coefficients show the one to one relationship between each metric and itself - for example, the top-left square shows the relationship between `FIELD_GOALS_MADE` and `FIELD_GOALS_MADE` (itself). The correlation coefficients in the bottom-left and the top-right squares are identical, since both of them show the relationship between `FIELD_GOAL_ATTEMPTS` and `FIELD_GOALS_MADE`. To see this point illustrated a bit more clearly, we can add two more pairings of similar metrics: two pointers made and two pointers attempted. 

### Correlation Matrix of Four Features
Let's add the two pointers made and two pointers attempted to our previous correlation matrix.

```python
plt.figure(figsize=(12, 10))
correlation_matrix = player_data[['FIELD_GOALS_MADE', 'FIELD_GOAL_ATTEMPTS', 'TWO_POINTS_MADE', 'TWO_POINT_ATTEMPTS']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Player Statistics')
plt.show()
```
    
![png](/assets/img/posts/2024-06-28-basketball-data-exploration/output_30_0.png)


Just like the previous matrix, we can see that there's a diagonal line of `1.00` values and the values are mirrored across that diagonal line of `1.00` values. The correlation coefficient between two pointers made and two pointers attempted is even stronger (`0.95`) than field goals made and attempted, so these two values are also strongly correlated. 

### Correlation Matrix of All Selected Features

Now that we understand a bit more about correlation matrices, we're ready to create the full chart. 


```python
correlation_matrix = player_data[numerical_columns].corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=0.5)
plt.title('Correlation Heatmap of Player Statistics')
plt.show()
```


    
![png](/assets/img/posts/2024-06-28-basketball-data-exploration/output_32_0.png)
    


This is a really useful visual that we'll be referring back to when we decide which variable pairings to use for visualizations.  

## Generate Scatterplot Matrix
To explore the relationships between variables, we can create a scatterplot matrix as well (also known as a pairwise plot). A scatterplot matrix is a collection of scatterplots organized into a grid. Each scatterplot shows the relationship between a pair of variables. The diagonal cells show a histogram of the variable corresponding to that row/column. By organizing these scatterplots into a matrix, we can easily compare multiple variables and observe potential correlations between different pairs of variables at a glance. 

This method is particularly useful because it allows us to visualize the relationships between two variables, similar to the correlation matrix, but provides more detail by showing a scatterplot instead of a single number (the correlation coefficient). This makes it easier to visualize the interaction between variables. Just like the correlation matrix, this is useful for feature selection and deciding which variables to include in subsequent data analysis or machine learning models. 

We can create a scatterplot matrix with our smaller list of numerical columns using [Seaborn's `pairplot()` function](https://seaborn.pydata.org/generated/seaborn.pairplot.html). 


```python
sns.pairplot(player_data[numerical_columns])
plt.show()
```


    
![png](/assets/img/posts/2024-06-28-basketball-data-exploration/output_35_0.png)
    


Looking at this chart, you can see why we reduced the number of numerical columns. Scatterplot matrices can get quite large with too many variables, so it can be helpful to focus on a few variables at first and individually analyze additional variables later. For example, the scatterplot matrix shows a dense linear relationship between `POINTS` and `FIELD_GOALS_MADE` and this matches the `0.97` correlation coefficient from the previous chart. Just like the correlation matrix, we can refer to back to this scatterplot matrix to quickly check the relationship between variables. 

# Wrap Up
In today's guide, we took a closer look at the underlying data in each column and created visualizations to identify the relationship between various parameters. Data exploration depends greatly on your individual project, so it's likely to look a bit different for each dataset. This step is generally best as an informal, free-form exploration of your data without being too focused on the finer details like axis titles or color scheme. In the next article, we'll cover generating meaningful visualizations, including a variety of charts and graphs.

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/) (Today's Guide)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/)
8. [Evaluating the Machine Learning Model](/)

<div class="email-subscription-container"></div>
<div id="sources"></div>

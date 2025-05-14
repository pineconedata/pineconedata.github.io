---
layout: post
title: "Ridge vs. OLS Linear Regression Models"
subtitle: "Outlier or Caitlin Clark? [Part 9]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, linear regression]
share-title: "Ridge vs. OLS Linear Regression Models: Outlier or Caitlin Clark? [Part 9]" 
share-description: Curious about the difference between Ridge Regression and Ordinary Least Squares Linear Regression? Join me in our latest blog post to learn how to train and evaluate Ridge Regression models using Python's Scikit-learn library. 
thumbnail-img: /assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/thumbnail.png
share-img: /assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll revisit our *Outlier or Caitlin Clark?* data science project by examining the differences between Ridge Regression and our previously-trained Ordinary Least Squares (OLS) linear regression model. This is the ninth part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. This article will have some overlap and assume knowledge from the [previous articles](#articles-in-this-series), so it's recommended to check out [selecting a machine learning model](/2024-08-12-basketball-select-ml-ols/), [training an OLS model](/2024-09-13-basketball-train-ols/), and [evaluating an OLS model](/2024-11-27-basketball-evaluate-ols-model/) as well. 

<div id="toc"></div>

# Getting Started
First, let's take a look at an overview of this data science project. If you're already familiar with it, feel free to skip to the [next section](#select-the-model). 

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

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Due to the size of this project, today we'll cover part of the seventh step: machine learning. A refresher on the [basics of machine learning](/2024-11-27-basketball-evaluate-ols-model/#basics-of-machine-learning) is also available in a previous article.

## Dependencies
Since this is the ninth installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first article of the series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. 

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
import pandas as pd
import numpy as np
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error, root_mean_squared_error
import seaborn as sns
import matplotlib.pyplot as plt
```

## Import Data
In a [previous part](/2024-09-13-basketball-train-ols/) of this series, we created our training and testing splits from the `player_data` dataframe. If you want to follow along with the code samples in this article, it's recommended to import the testing splits before proceeding.

*Note: To reduce confusion, the variable names in this article are slightly different than in a previous article. Models and datasets using the full set of features will have variable names appended with `_full`. Since the alternate models are trained using fewer features, those variable names will be appended with `_few`. For example, `X_test` of the full dataset is `X_test_full` and `X_test` with fewer features is `X_test_few`.* 


```python
X_train_full = pd.read_csv('X_train_full.csv')
X_train_few = pd.read_csv('X_train_few.csv')
```


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




```python
y_train_full = np.genfromtxt('y_train_full.csv', delimiter=',', skip_header=True)
y_train_few = np.genfromtxt('y_train_few.csv', delimiter=',', skip_header=True)
```

As a reminder, we previously created testing splits for the target `FANTASY_POINTS` variable. However, those two splits contained identical data, so we'll use a single testing split called `y_actual` for the target variable. 


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

# Select the Model
The first part of training any machine learning model is selecting the model to use. This might sound obvious, but selecting the “best” model for your problem depends on a variety of factors. We’ll likely explore this step in more detail in future articles ([let me know](/workwithme/) if you would be interested in that), but for today let's revisit [scikit-learn’s model flowchart](https://scikit-learn.org/stable/machine_learning_map.html) to find an appropriate model.

![scikit-learn's algorithm flowchat](https://scikit-learn.org/stable/_downloads/b82bf6cd7438a351f19fac60fbc0d927/ml_map.svg)

Just like in a [previous article](/2024-08-12-basketball-select-ml-ols/#identify-appropriate-models) where we ended up selecting a Ordinary Least Squares linear regression model, we can go through this flowchart until we get to the "few features should be important" split of the **regression** section. This time, let's explore the "No" branch of that split to see how a Ridge Regression model works.

## Model Characterstics 
[Ridge Regression](https://en.wikipedia.org/wiki/Ridge_regression), also referred to as L2 Regularization is an extension of the Ordinary Least Squares (OLS) method. Both OLS and Ridge Regression aim to find the best fit line through data points to make predictions. However, the main difference is that Ridge regression is a type of [regularization](https://en.wikipedia.org/wiki/Regularization_(mathematics)). 

While OLS minimizes the residual sum of squares without any constraints, Ridge regression adds a penalty term (L2 regularization) to the loss function. This penalty term is proportional to the absolute size of the coefficients and is designed to shrink large coefficient values asymptotically towards zero. One important characteristic of Ridge regression is that the coefficients never actually reach zero. This suppression of coefficients helps reduce overfitting caused by multi-collinearity, makes model parameters more interpretable, and improves generalization performance. 

In simplified terms, Ridge Regression strikes a balance between [bias and variance](https://en.wikipedia.org/wiki/Bias%E2%80%93variance_tradeoff) by constraining the magnitude of coefficients in a linear regression model. This constraint prevents coefficients from growing too large and offers a stable, predictive model suitable for complex datasets with multiple predictor variables that may not be independent of each other.

### Model Equation

We can represent the regression line of a Ridge regression model with an equation similar to the one for OLS: 
$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n + \lambda(\beta_1^2 + \beta_2^2 + ... + \beta_n^2)
$$

where:

- \\(y\\) is the predicted value of the target variable
- \\(x_1, x_2, ..., x_n\\) are the input features 
- \\(b_0\\) is the y-intercept (the value of \\(y\\) when all \\(x\\) are zero)
- \\(b_1, b_2, ..., b_n\\) are the coefficients that represent the change in \\(y\\) for a one-unit change in the corresponding \\(x\\), holding all other \\(x\\) constant
- \\(\lambda\\) is the regularization [hyperparameter](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning)) (the L2 regularization penalty on the coefficient values)

From this equation, we know the goal of Ridge regression is still to find the values of \\(b_0\\) and \\(b_1, b_2, ..., b_n\\) that minimize the difference between the predicted \\(\text{FantasyPoints}\\) (`y_pred`) values and the actual \\(\text{FantasyPoints}\\) (`y_actual`) values in our dataset. However, Ridge regression adds an additional goal to minimize the values of \\(b_1, b_2, ..., b_n\\) themselves in order to reduce the penalty term. 

*Note: This is a slightly simplified explanation of the underlying mathematics. I highly recommend reading the [Wikipedia page](https://en.wikipedia.org/wiki/Ridge_regression#Overview), or other suitable sources, for a more nuanced understanding of the process. I'd also recommend [this YouTube video](https://www.youtube.com/watch?v=Q81RR3yKn30) for a visual example of how the regularization parameter works.*

### Regularization Hyperparameter
You might notice that if we set \\(\lambda\\) (the regularization hyperparameter) to zero, then the equation for Ridge regression would simplify down to the equation for the OLS model that we [previously covered](/2024-08-12-basketball-select-ml-ols/#model-characteristics): 

$$
y = \beta_0 + \beta_1x_1 + \beta_2x_2 + ... + \beta_nx_n
$$

So there's not much point in setting \\(\lambda\\) to `0`, but how do we know what to set \\(\lambda\\) to? The best value for \\(\lambda\\) can be determined in a few different ways that could be the topic of it's own article, so [please let me know](/workwithme/) if you would be interested in that! For today, we'll use the default value set by the [scikit-learn Ridge() class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html), which is `1`. 

Now that we conceptually understand a bit more about how this model works, let's try training a Ridge regression model on our dataset. 

# Train the Models
Now that we have our data split into training and test sets, we’re ready to train our model. We’ll use scikit-learn’s [Ridge class](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html) for this purpose. We can start by initializing the model with the default parameters:


```python
model_full = Ridge()

# this is equivalent to running:
# model_full = Ridge(alpha=1)
```

*Note:* The default parameters of Ridge() include setting \\(\lambda\\) to `1` using the `alpha` parameter. We could (and probably will) explore how to identify the best value of \\(\lambda\\) in a future article, but for today we'll use the default value. If you want to try out a different value, feel free to run `model_full = Ridge(alpha=YOUR_LAMBDA_VALUE)` instead. 

<div class="email-subscription-container"></div>

We can then use the [fit() method](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge.fit) to actually train our model on our data. This method takes two arguments: `X_train` (the training split of our features) and `y_train` (the training split of our target variable). For a model trained on the full set of features, this will be `X_train_full` and `y_train_full` respectively.


```python
model_full.fit(X_train_full, y_train_full)
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
</style><div id="sk-container-id-1" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Ridge()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-1" type="checkbox" checked><label for="sk-estimator-id-1" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Ridge<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.Ridge.html">?<span>Documentation for Ridge</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Ridge()</pre></div> </div></div></div></div>



We can follow this same process to train a model on the dataset with fewer features:


```python
model_few = Ridge()
model_few.fit(X_train_few, y_train_few)
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
</style><div id="sk-container-id-2" class="sk-top-container"><div class="sk-text-repr-fallback"><pre>Ridge()</pre><b>In a Jupyter environment, please rerun this cell to show the HTML representation or trust the notebook. <br />On GitHub, the HTML representation is unable to render, please try loading this page with nbviewer.org.</b></div><div class="sk-container" hidden><div class="sk-item"><div class="sk-estimator fitted sk-toggleable"><input class="sk-toggleable__control sk-hidden--visually" id="sk-estimator-id-2" type="checkbox" checked><label for="sk-estimator-id-2" class="sk-toggleable__label fitted sk-toggleable__label-arrow fitted">&nbsp;&nbsp;Ridge<a class="sk-estimator-doc-link fitted" rel="noreferrer" target="_blank" href="https://scikit-learn.org/1.4/modules/generated/sklearn.linear_model.Ridge.html">?<span>Documentation for Ridge</span></a><span class="sk-estimator-doc-link fitted">i<span>Fitted</span></span></label><div class="sk-toggleable__content fitted"><pre>Ridge()</pre></div> </div></div></div></div>



Just like with training the OLS model, we're using the training set to learn the optimal parameters for each feature that minimize the difference between the predicted values of the target variable and the actual values of the target variable. However, ridge regression adds additional logic to minimize the optimal parameters for each feature as well. Once the training is complete, the model will contain the learned parameters (coefficients and intercept) that can be used to make predictions on new data. In the next step, we can use `X_test` to predict what the model thinks y is, and then compare that output y (`y_pred`) to the actual y values (`y_actual`) to evaluate the model performance. 

# Generate Predictions
To evaluate the model's performance, we'll compare the values that the model predicts to the actual values (sometimes referred to as the "ground truth" values). Models that predict values close to the actual values perform better, and models that predict values far from the actual values perform worse. There are various evaluation metrics we can calculate using these predictions to quantify how well a model performs, but the first step is generating the predictions for each model. 

To generate predictions, we'll apply each trained model to the testing data split. We'll use the testing data split instead of the training data split to ensure that the model is evaluated on data that it hasn't seen during training. (For a refresher on *why* we split the dataset into training and testing subsets, see a [previous article](/2024-09-13-basketbal-train-ols/#create-training-and-testing-splits)) in this series.

In a [previous article](/2024-11-27-basketball-evaluate-ols-model/#generate-predictions), we generated predictions in three ways: 

1. Manually
2. Using [np.dot()](https://numpy.org/doc/stable/reference/generated/numpy.dot.html)
3. Using [Ridge.predict()](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge.predict)

You could follow the same steps to generate predictions manually and using `np.dot()` for this model as well, but for today we'll jump straight into predictions using sklearn's [.predict() method](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge.predict).

## Calculate Predictions with .predict()

Just like with the OLS LinearRegression model, sklearn's Ridge model also has a [.predict() function](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.Ridge.html#sklearn.linear_model.Ridge.predict) to calculate predictions. 


```python
y_pred_full = model_full.predict(X_test_full)
y_pred_full[:5]
```




    array([752.99104584, 544.20411972, 587.49782753, 969.89164701,
           621.09740207])




```python
y_pred_few = model_few.predict(X_test_few)
y_pred_few[:5]
```




    array([689.05084667, 629.10060781, 640.65984357, 934.31552165,
           617.45344465])



We now have our predictions for both models and can see that the predicted values noticeably differ between the two models. 

## Create Baseline Model
Before evaluating how well these models perform, let's create one more "model" as a baseline. This baseline model will simply predict the mean of the target variable in the training data and will serve as a simple reference point. (If you read the [previous article](/2024-11-27-basketball-evaluate-ols-model/) in this series, this is exactly the same baseline model.) By comparing the performance of the ridge regression model to this naive approach, we can determine if the model is capturing meaningful patterns in the data and offering improvements beyond just predicting the average. 

To create predictions for this baseline model, we can create an array of the same size and type as our predictions or actual values using [NumPy's .full_like()](https://numpy.org/doc/stable/reference/generated/numpy.full_like.html#numpy-full-like) function: 


```python
y_pred_base = np.full_like(y_actual, np.mean(y_actual))
y_pred_base[:5]
```




    array([658.009375, 658.009375, 658.009375, 658.009375, 658.009375])



Now we have all three of our predictions and can compare those to the actual values to evaluate how well each of the three models performs.

# Evaluate the Model
After training our ridge regression models, the next crucial step is to evaluate performance. This evaluation process helps us understand how well our model is doing, identify any issues, and determine if it's ready for real-world application or if it needs further refinement. In this section, we'll explore various metrics and techniques to assess our model's accuracy and reliability.

## Evaluation Metric Definitions

Let's start with a quick overview of each evaluation metric we'll be exploring today. 
- **[R-squared (R²)](https://en.wikipedia.org/wiki/Coefficient_of_determination)** - This measures the proportion of variance explained by the model. It gives a good first glance at how much of the variability in the target (Fantasy Points in this case) is explained by the model. It's also referred to as the coefficient of determination.
- **[Adjusted R-squared](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2)** - This is similar to R-squared, but is adjusted for the number of predictors to account for overfitting. This is often useful when comparing multiple models with different numbers of features (as is the case between `model_full` and `model_few`).
- **[Mean Squared Error (MSE)](https://en.wikipedia.org/wiki/Mean_squared_error)** - This is the average of the squared differences between predicted and actual values. It indicates the model's prediction accuracy, penalizing larger errors more heavily. 
- **[Root Mean Squared Error (RMSE)](https://en.wikipedia.org/wiki/Root_mean_square_deviation)** - This is the square root of MSE. It provides a more interpretable measure of prediction accuracy than MSE since it is in the same units and scale as the target variable. It helps understand the magnitude of the average prediction error. 
- **[Mean Absolute Error (MAE)](https://en.wikipedia.org/wiki/Mean_absolute_error)** - This is the average absolute difference between predicted and actual values. It is also a measure of the model's prediction accuracy, but it penalizes errors equally (instead of penalizing larger errors like MSE) and is less sensitive to outliers as a result. 
- **[Residuals](https://en.wikipedia.org/wiki/Errors_and_residuals)** - These are the difference between the predicted values and the actual values. These help assess the accuracy of the model by potentially revealing patterns or biases in the model. These are usually plotted and analyzed visually. 

*Note that there are additional metrics that can be used to evaluate and compare regression models (such as [Variance Inflation Factor](https://en.wikipedia.org/wiki/Variance_inflation_factor) and [F-test](https://en.wikipedia.org/wiki/F-test)), but the metrics covered today are commonly used and will serve as a good starting point.*

We could calculate most of these evaluation metrics using multiple methods (like in a [previous article](/2024-11-27-basketball-evaluate-ols-model//2024-11-27-basketball-evaluate-ols-model/)), but for today we'll use the convenient scikit-learn functions wherever available.

## Define Variables
Before jumping into the evaluation metrics, let's define a few helpful variables. In the equations in this section, the following variables will be used:

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
The [mean](https://en.wikipedia.org/wiki/Mean) will be used to calculate a few of the evaluation metrics, so let's take this opportunity to calculate it. Let's start by looking at the equation for the arithmetic mean: 

$$
\bar{y} = \frac{1}{n} \sum_{i=1}^{n} y_i
$$

where:
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\bar{y}\\) is the mean of the actual values (`mean(y_actual)`)
- \\(n\\) is the number of data points (`len(y_actual)`)

*Note: If you'd like a refresher on \\(\Sigma{}\\) notation, [the summation Wikipedia page](https://en.wikipedia.org/wiki/Summation#Capital-sigma_notation) is a good resource.*

We can use the [NumPy .mean()](https://numpy.org/doc/2.2/reference/generated/numpy.mean.html) function to calculate the mean directly: 


```python
y_mean = np.mean(y_actual)
y_mean
```




    658.009375



### Calculate Residuals
The last variable we'll calculate before getting into the evaluation metrics is the [residuals](https://en.wikipedia.org/wiki/Errors_and_residuals). In this context, the residuals refer to the difference between \\(y_i\\) and \\(\hat{y}\\). 

#### Equation
$$
\text{residuals} = y_i - \hat{y}
$$

where:
- \\(y_i\\) are the actual values (`y_actual`)
- \\(\hat{y}\\) are the predicted values (`y_pred_full`, `y_pred_few`, `y_pred_base`)

#### Calculate
We can calculate the residuals for each model: 


```python
residuals_full = y_actual - y_pred_full
residuals_full[:5]
```




    array([ 0.00895416, -0.00411972,  0.00217247,  0.00835299,  0.00259793])




```python
residuals_few = y_actual - y_pred_few
residuals_few[:5]
```




    array([ 63.94915333, -84.90060781, -53.15984357,  35.58447835,
             3.64655535])




```python
residuals_base = y_actual - y_pred_base
residuals_base[:5]
```




    array([  94.990625, -113.809375,  -70.509375,  311.890625,  -36.909375])



#### Evaluation
We'll be plotting and analyzing these residuals in a later step, so for now we'll use the residuals and other variables to calculate a few of the evaluation metrics.

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

### Calculate
We can calculate \\(R^2\\) using [scikit-learn's r2_score() function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.r2_score.html) function for each model: 


```python
r2_full = r2_score(y_actual, y_pred_full)
r2_full
```




    0.9999999995121926




```python
r2_few = r2_score(y_actual, y_pred_few)
r2_few
```




    0.9207256674767885




```python
r2_base = r2_score(y_actual, y_pred_base)
r2_base
```




    0.0



### Evaluation
Now that we have our results for each model, let's take a look at how each model performs. As mentioned earlier, a higher \\(R^2\\) generally indicates a better fit for the model, with `1` being a perfect fit and `0` being a poor fit that performs no better than predicting the mean. 
In this case, `model_full` has a \\(R^2\\) value of almost, but not quite, `1.0`, so it's predictions are quite close to the ideal. On the other end, it makes sense that `model_base` has a \\(R^2\\) of `0`, since this model is predicting the mean for each observation. `model_few` has a \\(R^2\\) of `0.92...`, which is relatively close to `1`, so this model also predicted values close to the actual values, but did not perform quite as well as `model_full`.

## Adjusted \\(R^2\\)
[Adjusted \\(R^2\\)](https://en.wikipedia.org/wiki/Coefficient_of_determination#Adjusted_R2) is a modification to the standard \\(R^2\\) that we just calculated that adjusts for the number of predictors (Height, Points, Steals, etc.) in the model. Standard \\(R^2\\) will always increase as you add more predictors (even if they aren't improving the model), which can make the results a bit misleading for models with many predictors. Adjusted  \\(R^2\\) penalizes the addition of unnecessary predictors, so it provides a more accurate measure of the model's performance when there are multiple predictors. This also makes it quite useful for comparing models with different numbers of predictors.

Adjusted \\(R^2\\) is similar to standard \\(R^2\\) in that it values closer to `1` indicate a good fit, and values closer to (or below) `0` indicate a poor fit. Adjusted \\(R^2\\) can also be below zero in cases of poorly fitted models or when \\(p\\) is much greater than \\(n\\).

### Equation
Let's start by looking at the equation for Adjusted \\(R^2\\): 

$$
\text{Adjusted } R^2 = 1 - \left( \frac{(1 - R^2)(n - 1)}{n - p - 1} \right)
$$

where 
- \\(R^2\\) is the regular coefficient of determination
- \\(n\\) is the number of data points
- \\(p\\) is the number of predictors (independent variables)

### Calculate
We already have variables for \\(R^2\\) and \\(n\\), so let's begin the manual calculation by defining \\(p\\). Scikit-learn's Ridge regression models have an attribute called `n_features_in_` that returns the number of features seen when fitting the model, so we can use that: 


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



Now that we know both \\(p\\) and \\(n\\), we can calculate the Adjusted \\(R^2\\) for each model. At time of writing, there isn't a built-in function in scikit-learn to calculate Adjusted \\(R^2\\), so we'll calculate it manually using the standard \\(R^2\\) calculated in the previous step. 


```python
adj_r2_full = 1 - ((1 - r2_full) * (n - 1)) / (n - p_full - 1)
adj_r2_full
```




    0.999999999481995




```python
adj_r2_few = 1 - ((1 - r2_few) * (n - 1)) / (n - p_few - 1)
adj_r2_few
```




    0.9170038678278114




```python
adj_r2_base = 1 - ((1 - r2_base) * (n - 1)) / (n - p_base - 1)
adj_r2_base
```




    0.0



### Evaluation
Evaluating Adjusted \\(R^2\\) follows the same logic as the standard \\(R^2\\). We can see that the Adjusted \\(R^2\\) for `model_base` is still `0`. Adjusted \\(R^2\\) is slightly lower than standard \\(R^2\\) for both `model_full` and `model_few`, which indicates that both models perform quite well, with `model_full` doing slightly better than `model_few`.

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

### Calculate
We can use [scikit-learn's mean_squared_error() function](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_squared_error.html) to calculate MSE directly. 


```python
mse_full = mean_squared_error(y_actual, y_pred_full)
mse_full
```




    1.5933696074432595e-05




```python
mse_few = mean_squared_error(y_actual, y_pred_few)
mse_few
```




    2589.40980805919




```python
mse_base = mean_squared_error(y_actual, y_pred_base)
mse_base
```




    32663.91183175223



### Evaluation
An important note about MSE is that the units are \\(\text{FantasyPoints}^2\\) As mentioned earlier, a MSE closer to `0` is better, so it makes sense that the `model_full` performs the best. `mse_few` is in between the values of `mse_full` and `mse_base`, with `mse_base` being over 10x larger than `mse_few`. The results are somewhat similar to that of \\(R^2\\), but a bit less interpretable, so let's move on to the next metric.

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

### Calculate
Scikit-learn provides the [root_mean_squared_error](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.root_mean_squared_error.html) function that we can apply to the predictions from each model to calculate RMSE. 


```python
rmse_full = root_mean_squared_error(y_actual, y_pred_full)
rmse_full
```




    0.0039917034051182455




```python
rmse_few = root_mean_squared_error(y_actual, y_pred_few)
rmse_few
```




    50.88624379986393




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



In this case, the target variable and its mean are on the order of hundreds, so a RMSE of 50.8 for `model_few` seems fairly good, while the RMSE of 180 for `model_base` is quite poor. 

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

### Calculate
We can use scikit-learn's [mean_absolute_error()](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.mean_absolute_error.html) function with each model's predictions. 


```python
mae_full = mean_absolute_error(y_actual, y_pred_full)
mae_full
```




    0.0031245476904033153




```python
mae_few = mean_absolute_error(y_actual, y_pred_few)
mae_few
```




    39.83424851367819




```python
mae_base = mean_absolute_error(y_actual, y_pred_base)
mae_base
```




    139.73130580357142



### Evaluation
Similar to MSE and RMSE, the lower the MAE, the better the model's fit. `mae_full` is still quite close to zero, and `mae_few` is much better than `mae_base`, so both of those models perform better than the baseline model. We can use the same context used for RMSE (where `y_actual` ranges from `~190` to `~1,300` with a mean of `~658`) to further confirm that the baseline model performs quite poorly, while `mae_few` performs reasonably well.

<div class="email-subscription-container"></div>

## Evaluation Metrics Results

For convenience, we can also summarize the results of all of these evaluation metrics in a single table:

| Method | Model | \\(R^2\\) | Adjusted \\(R^2\\) | MSE | RMSE | MAE |
|---|---|---|---|---|---|---|
| Mean | `model_base` | 0 | 0 | 32,663.91 | 180.73 | 139.73 |
| OLS | `model_full` | 1.0 | 1.0 | 3.91 \\(x10^{-26}\\) | 1.97 \\(x10^{-13}\\) | 1.66 \\(x10^{-13}\\) |
| OLS | `model_few` | 0.921 | 0.917 | 2,589.41 | 50.88 | 39.83 |
| Ridge | `model_full` | 0.999 | 0.999 | 1.59 \\(x10^{-5}\\) | 0.004 | 0.003 |
| Ridge | `model_few` | 0.921 | 0.917 | 2,589.41 | 50.88 | 39.83 |
|  |  | *Closer to 1.0 is better* | *Closer to 1.0 is better* | *Lower is better* | *Lower is better* | *Lower is better* |


In this table, we can see that the evaluation metrics for the Ridge regression model are essentially the same as the OLS model. However, this dataset we're using to train the models today reflects an ideal situation that is often not present in real-world scenarios. Ridge regression remains an excellent choice for situations where reducing overfitting (and improving generalization) is important.

## Residuals Plots
As a reminder, a [residual](https://en.wikipedia.org/wiki/Errors_and_residuals) is the difference between an observed value and its corresponding predicted value (\\(y_i - \hat{y_i}\\)). We calculated the residuals in a previous step, so let's wrap up by plotting and reviewing the residuals. Plotting residuals is a useful visual way to evaluate the assumptions and identify potential issues with the fit of a regression model that metrics alone might miss.

When reviewing these residual plots, we'll primarily be checking for whether or not there's an issue with the model. For models with a good fit, these plots should have an even, random distribution of residuals around the horizontal line (zero on the y-axis) without any outliers. If there are any clear patterns (curves or clusters) in the residuals plot, that can suggest that the model is not capturing some aspect of the data, such as omitted variables, non-linearity issues, or [heteroskedasticity](https://en.wikipedia.org/wiki/Homoscedasticity_and_heteroscedasticity). 

### Evaluating Scatterplot of Residuals
Let's start by creating a scatterplot of the residuals versus the predicted values for each model. 


```python
plt.scatter(y_pred_full, residuals_full)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot for model_full')
plt.show()
```


    
![png](/assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/output_97_0.png)
    


The residuals for `model_full` at first glance shows a slight upward trend. However, the scale for the y-axis is \\(\times 10^{-2}\\), so all of these residuals are quite small compared to our y values. 


```python
plt.scatter(y_pred_few, residuals_few)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot for model_few')
plt.show()
```


![png](/assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/output_99_0.png)
    

The residuals plot for `model_few` shows a more even distribution, so there aren't any major issues. Note that the scale of the y-axis is \\(\times 10^2\\), so these residuals are considerably larger than for `model_full`.


```python
plt.scatter(y_pred_base, residuals_base)
plt.xlabel('Predicted values')
plt.ylabel('Residuals')
plt.title('Residual Plot for model_base')
plt.show()
```


    
![png](/assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/output_101_0.png)
    


The residuals plot for `model_base` looks clearly different from the other two and is not evenly distributed around zero. This plot indicates an issue with the underlying model. Since this model just predicts the mean of the target variable (and uses zero predictors), the pattern shown in this residuals plot is clearly showing an issue (omitted variables). 

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


    
![png](/assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/output_104_0.png)
    


The histogram of residuals for `model_full` is roughly bell-shaped with a very slight shift to the right (above zero).


```python
sns.histplot(residuals_few, bins=15, kde=True)
plt.title('Histogram of Residuals for model_few')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/output_106_0.png)
    


The histogram of residuals for `model_few` matches the normal distribution quite well and is evenly centered around zero. 


```python
sns.histplot(residuals_base, bins=15, kde=True)
plt.title('Histogram of Residuals for model_base')
plt.xlabel('Residuals')
plt.ylabel('Frequency')
plt.show()
```


    
![png](/assets/img/posts/2025-04-04-ridge-regression-vs-ols-linear-regression-models/output_108_0.png)
    


The histogram of residuals for `model_base` is quite normal, minus the small "tail" on the right (above zero). Overall, the residuals are normally distributed. However, the residuals are quite large (non-negligable) compared to the scale of the y values, so `model_base` still does not perform well even though the histogram of residuals is otherwise normal.

For more examples of these plots, Penn State has an excellent page on the [Normal Probability Plot of Residuals](https://online.stat.psu.edu/stat462/node/122/), including examples of residuals that are not normally distributed. 

## Feature Importance
As the last piece of model evaluation for today, we'll take a look at the feature importance of each model. To do this, we'll examine differences between the coefficients of the two models (`model_full` and `model_few`). As a reminder, `model_base` simply predicts the mean of the target variable without using any features. This means that all of the feature coefficients are set to zero for `model_base`. Since there is no meaningful comparison to make with `model_base`, we'll focus on comparing the relative feature importance of `model_full` and `model_few`. 

To compare the feature importance between the two models, let's start by identifying the coefficients for each model. For both models, the coefficients can be accessed with the `.coef` attribute and the corresponding feature names can be accessed with the `.feature_names_in` attribute.


```python
coef_full = pd.Series(data=model_full.coef_, index=model_full.feature_names_in_)
coef_full
```




    Height               0.000004
    MINUTES_PLAYED       0.000005
    FIELD_GOALS_MADE     1.461186
    THREE_POINTS_MADE    1.168972
    TWO_POINTS_MADE      0.292214
    FREE_THROWS_MADE     0.876717
    TOTAL_REBOUNDS       1.200012
    ASSISTS              1.500007
    TURNOVERS           -1.000151
    STEALS               2.000006
    BLOCKS               2.000033
    FOULS                0.000070
    POINTS               0.123281
    dtype: float64



Now we can assemble the coefficients for `model_few`. 


```python
coef_few = pd.Series(data=model_few.coef_, index=model_few.feature_names_in_)
coef_few
```




    Height               2.452227
    MINUTES_PLAYED       0.103872
    THREE_POINTS_MADE    2.203715
    FREE_THROWS_MADE     2.391725
    TOTAL_REBOUNDS       1.521931
    ASSISTS              1.323122
    TURNOVERS           -0.570635
    STEALS               2.239279
    BLOCKS               2.481845
    FOULS               -0.261200
    dtype: float64



`model_few` has fewer coefficients than `model_full`, so let's put the coefficients for each model and the difference between the two into a [DataFrame](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.html) for easy comparison. 


```python
df = pd.DataFrame({'model_full': coef_full, 'model_few': coef_few})
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
      <td>1.500007</td>
      <td>1.323122</td>
      <td>-0.176886</td>
    </tr>
    <tr>
      <th>BLOCKS</th>
      <td>2.000033</td>
      <td>2.481845</td>
      <td>0.481813</td>
    </tr>
    <tr>
      <th>FIELD_GOALS_MADE</th>
      <td>1.461186</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>FOULS</th>
      <td>0.000070</td>
      <td>-0.261200</td>
      <td>-0.261270</td>
    </tr>
    <tr>
      <th>FREE_THROWS_MADE</th>
      <td>0.876717</td>
      <td>2.391725</td>
      <td>1.515008</td>
    </tr>
    <tr>
      <th>Height</th>
      <td>0.000004</td>
      <td>2.452227</td>
      <td>2.452223</td>
    </tr>
    <tr>
      <th>MINUTES_PLAYED</th>
      <td>0.000005</td>
      <td>0.103872</td>
      <td>0.103867</td>
    </tr>
    <tr>
      <th>POINTS</th>
      <td>0.123281</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>STEALS</th>
      <td>2.000006</td>
      <td>2.239279</td>
      <td>0.239273</td>
    </tr>
    <tr>
      <th>THREE_POINTS_MADE</th>
      <td>1.168972</td>
      <td>2.203715</td>
      <td>1.034743</td>
    </tr>
    <tr>
      <th>TOTAL_REBOUNDS</th>
      <td>1.200012</td>
      <td>1.521931</td>
      <td>0.321919</td>
    </tr>
    <tr>
      <th>TURNOVERS</th>
      <td>-1.000151</td>
      <td>-0.570635</td>
      <td>0.429516</td>
    </tr>
    <tr>
      <th>TWO_POINTS_MADE</th>
      <td>0.292214</td>
      <td>NaN</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



As a reminder, `model_few` has fewer parameters than `model_full`, because `FIELD_GOALS_MADE`, `TWO_POINTS_MADE`, and `POINTS` were removed from the feature set to create `model_few`. Certain features (`MINUTES_PLAYED`, `Height`, `FOULS`) of `model_full` are fairly close to zero but are not actually zero; this matches the expectations of Ridge regression models that we covered in the [#model-characteristics] section. `model_few` weights  

# Wrap Up
In today's guide, we covered the primary differences between Ridge and OLS regression models. The previous article was meant to be the last installment in the series, but we've already returned for a quick look at Ridge regression models. So, please [let me know](https://www.pineconedata.com/workwithme/) if you'd like to use this dataset (or another dataset) to explore another type of machine learning model in the future!

As a reminder, all of the code snippets in today's guide are available in a Jupyter Notebook in the [ncaa-basketball-stats](https://github.com/pineconedata/ncaa-basketball-stats) repository on [GitHub](https://github.com/pineconedata/).

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/) 
8. [Evaluating the Machine Learning Model](/2024-11-27-basketball-evaluate-ols-model/)
9. [Ridge vs OLS Linear Regression Models]() (Today's Guide)

<div class="email-subscription-container"></div>
<div id="sources"></div>

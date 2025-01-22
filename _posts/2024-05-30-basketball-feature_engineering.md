---
layout: post
title: "Feature Engineering"
subtitle: "Outlier or Caitlin Clark? [Part 3]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, scipy]
thumbnail-img: /assets/img/posts/2024-05-30-basketball-feature-engineering/engineer.jpg
share-title: "Feature Engineering: Outlier or Caitlin Clark? [Part 3]" 
share-description: Interested in engineering new features for your dataset? Learn how to derive new features, calculate additional metrics, and extract textual data in the latest installment of this data science series that is perfect for beginner data scientists and Python enthusiasts.
share-img: /assets/img/posts/2024-05-30-basketball-feature-engineering/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll explore how to derive new features from existing columns by calculating additional metrics and extracting textual data. This is the third part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, machine learning, and creating visualizations. 


<div id="toc"></div>

# Getting Started
First, let's take a look at an overview of this data science project. If you're already familiar with it, feel free to skip to the [next section](#feature-engineering).

## Project Overview
As a reminder, the dataset we'll be using in this project contains individual basketball player statistics (such as total points scored and blocks made) for the 2023-2024 NCAA women's basketball season. Here's a brief description of each major step that we'll go through for this project: 

![the steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/project_steps.png "the steps for this data science project")

1. **Data Acquisition** - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. **Data Cleaning** - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. **Data Preprocessing** - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. **Feature Engineering** - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.
5. **Data Exploration** - This step focuses on analyzing and visualizing the dataset to uncover patterns, relationships, and general trends and is a helpful preliminary step before deeper analysis.
6. **Creating Visualizations** - This step involves identifying the relationships between various parameters (such as height and blocked shots) and generating meaningful visualizations (such as bar charts, scatterplots, and candlestick charts).
5. **Machine Learning** - This step focuses on selecting, training, and evaluating a machine learning model. For this project, the model will identify the combination of individual player statistics that correlates with optimal performance. 

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Since we already gathered the raw data from online sources in [Part 1](/2024-04-11-basketball-data-acquisition/) and cleaned that data in [Part 2](2024-05-02-basketball-data-cleaning-preprocessing/), we're ready to move on to feature engineering.

## Dependencies
Since this is the third installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first post in this series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. In summary, you'll want to have [Python](https://www.python.org/) installed with the following packages: 
  - [pandas](https://pandas.pydata.org/docs/)
  - [requests](https://requests.readthedocs.io/en/latest/)
  - [json](https://docs.python.org/3/library/json.html)
  - [os](https://docs.python.org/3/library/os.html)
  - [numpy](https://numpy.org/doc/)
  
For today's guide specifically, we'll want to import the following packages: 
```python
import pandas as pd
import numpy as np
```

## Import Data
  
In [Part 2](2024-05-02-basketball-data-cleaning-preprocessing/) of this series, we cleaned and preprocessed the values in our dataset, which is stored in a dataframe named `player_data`. If you want to follow along with the code examples in this article, it's recommended to import the `player_data` dataframe before proceeding. 


```python
player_data = pd.read_excel('player_data_clean.xlsx')
```

# Feature Engineering
In this section, we'll create new features (columns) by extracting data and calculating additional metrics (derived column creation) from columns that already exist in our dataset. For example, we can calculate the per-game averages by using the number of total games played and another statistic (such as total three-point baskets made). By the end of this process, we should have all the columns we want when we start creating visualizations and training machine learning models.

## Calculate Two-Point Basket Metrics
This dataset contains the statistics on both field goals (the combination of two-point and three-point baskets) and three-point baskets specifically, so the first set of columns we can derive are two-point basket statistics. 


```python
# Calculate two-pointers made
player_data['TWO_POINTS_MADE'] = player_data['FIELD_GOALS_MADE'] - player_data['THREE_POINTS_MADE']

# Calculate two-point attempts
player_data['TWO_POINT_ATTEMPTS'] = player_data['FIELD_GOAL_ATTEMPTS'] - player_data['THREE_POINT_ATTEMPTS']

# Calculate two-point percentage
player_data['TWO_POINT_PERCENTAGE'] = (player_data['TWO_POINTS_MADE'] / player_data['TWO_POINT_ATTEMPTS']) * 100

player_data.dtypes
```




    Unnamed: 0                  int64
    PLAYER_NAME                object
    Team                       object
    Class                      object
    Height                      int64
    Position                   object
    PLAYER_ID                  object
    TEAM_NAME                  object
    GAMES                       int64
    MINUTES_PLAYED              int64
    FIELD_GOALS_MADE            int64
    FIELD_GOAL_ATTEMPTS         int64
    FIELD_GOAL_PERCENTAGE     float64
    THREE_POINTS_MADE           int64
    THREE_POINT_ATTEMPTS        int64
    THREE_POINT_PERCENTAGE    float64
    FREE_THROWS_MADE            int64
    FREE_THROW_ATTEMPTS         int64
    FREE_THROW_PERCENTAGE     float64
    OFFENSIVE_REBOUNDS          int64
    DEFENSIVE_REBOUNDS          int64
    TOTAL_REBOUNDS              int64
    ASSISTS                     int64
    TURNOVERS                   int64
    STEALS                      int64
    BLOCKS                      int64
    FOULS                       int64
    POINTS                      int64
    TWO_POINTS_MADE             int64
    TWO_POINT_ATTEMPTS          int64
    TWO_POINT_PERCENTAGE      float64
    dtype: object



Just like the three-point basket statistics, we can see that the two-pointers made and attempted are integers whereas the two-point percentage is stored as a float. 

## Extract Conference from Team Name
This dataset includes two fields for the team name: one that came from the player information dataset and the other that came from the player statistics dataset. Let's look at a sample of each.


```python
player_data[['TEAM_NAME', 'Team']].sample(10)
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
      <th>TEAM_NAME</th>
      <th>Team</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>727</th>
      <td>Old Dominion</td>
      <td>Old Dominion (Sun Belt)</td>
    </tr>
    <tr>
      <th>856</th>
      <td>Missouri</td>
      <td>Missouri (SEC)</td>
    </tr>
    <tr>
      <th>867</th>
      <td>LIU Brooklyn</td>
      <td>LIU (NEC)</td>
    </tr>
    <tr>
      <th>94</th>
      <td>Arkansas</td>
      <td>Arkansas St. (Sun Belt)</td>
    </tr>
    <tr>
      <th>715</th>
      <td>Rutgers</td>
      <td>Rutgers (Big Ten)</td>
    </tr>
    <tr>
      <th>500</th>
      <td>S.F. Austin</td>
      <td>SFA (WAC)</td>
    </tr>
    <tr>
      <th>676</th>
      <td>Gonzaga</td>
      <td>Gonzaga (WCC)</td>
    </tr>
    <tr>
      <th>150</th>
      <td>Wisconsin</td>
      <td>Wisconsin (Big Ten)</td>
    </tr>
    <tr>
      <th>666</th>
      <td>Holy Cross</td>
      <td>Holy Cross (Patriot)</td>
    </tr>
    <tr>
      <th>427</th>
      <td>Miami (FL)</td>
      <td>Notre Dame (ACC)</td>
    </tr>
  </tbody>
</table>
</div>



So the TEAM_NAME column from the player statistics dataset contains strictly the name of the team, but the Team column from the player information dataset contains both the team name and the [basketball conference](https://en.wikipedia.org/wiki/NCAA_Division_I_women%27s_basketball_conference_tournaments) in parenthesis. If we only had the TEAM_NAME column and wanted to know the conference, then we could pull in another data source to map team names to conference names. However, since the conference is listed in the TEAM column, we can split this information into a separate column as a feature. 

There are multiple ways to do this, so let's start with the most obvious one. It looks like the conference name is enclosed in parentheses after the team name. We could extract the conference name by splitting the 'Team' column on the opening and closing parentheses and selecting the last element of the resulting list.


```python
player_data['Team'].str.split('\(', expand=True)[1].str.split('\)', expand=True)[0]
```




    0      Mountain West
    1                SEC
    2            Big Sky
    3      Mountain West
    4                ACC
               ...      
    895             MAAC
    896              AAC
    897            SoCon
    898             SWAC
    899             ASUN
    Name: 0, Length: 900, dtype: object



However, depending on the results of your previous sample, you might notice an issue with this. Certain teams have two sets of parenthesis instead of just one. Here are two examples. 


```python
player_data.loc[[125, 824], 'Team']
```




    125       LMU (CA) (WCC)
    824     Miami (FL) (ACC)
    Name: Team, dtype: object



For these two examples, the extracted Conference name would be "CA" and "FL" instead of the proper "WCC" and "ACC". 


```python
player_data['Team'].str.split('\(', expand=True)[1].str.split('\)', expand=True)[0].iloc[[125, 824]]
```




    125    CA
    824    FL
    Name: 0, dtype: object



We could modify the previous code to select only the values in the right-most set of parenthesis, but switching from the `.split()` method to using a [regular expression](https://en.wikipedia.org/wiki/Regular_expression) (a.k.a. "regex") might be a more robust solution. Regular expressions provide more flexibility in pattern matching and are quite efficient, so let's build a pattern to select the proper value.

In this regular expression, we can use the following parts: 
1. `\(` matches on an opening parenthesis. Parenthesis in regular expressions are used for capturing groups, so the `\` escapes the parenthesis and matches the literal value.
2. `([^)]+)` matches matches one or more characters that are not a closing parentheses (any text inside the parentheses). The `^` symbol negates the set, so this pattern matches anything except a right parenthesis. The `+` quantifier means "one or more" of these characters. This ensures we capture the entire conference name instead of just the first letter. 
3. `\)` matches on a closing parenthesis. Just like the opening parenthesis, this needs to be escaped.
4. `$` searches for matches from the end of the string first instead of from the start of the string. 

This pattern should capture the text within the last pair of parentheses at the end of the string. Let's preview what this logic would extract as the conference name.


```python
player_data['Team'].str.extract(r'\(([^)]+)\)$')
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Mountain West</td>
    </tr>
    <tr>
      <th>1</th>
      <td>SEC</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Big Sky</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Mountain West</td>
    </tr>
    <tr>
      <th>4</th>
      <td>ACC</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
    </tr>
    <tr>
      <th>895</th>
      <td>MAAC</td>
    </tr>
    <tr>
      <th>896</th>
      <td>AAC</td>
    </tr>
    <tr>
      <th>897</th>
      <td>SoCon</td>
    </tr>
    <tr>
      <th>898</th>
      <td>SWAC</td>
    </tr>
    <tr>
      <th>899</th>
      <td>ASUN</td>
    </tr>
  </tbody>
</table>
<p>900 rows × 1 columns</p>
</div>



We can double-check that this regex pattern will pull from the proper set of parentheses for the values with multiple parentheses. 


```python
player_data['Team'].str.extract(r'\(([^)]+)\)$').iloc[[125, 824]]
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
      <th>0</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>125</th>
      <td>WCC</td>
    </tr>
    <tr>
      <th>824</th>
      <td>ACC</td>
    </tr>
  </tbody>
</table>
</div>



This is perfect, so let's go ahead and create a new column for this data. 


```python
player_data['Conference'] = player_data['Team'].str.extract(r'\(([^)]+)\)$')
```

We can look at the unique values in this new column to verify that we have not extracted any incorrect data. 


```python
sorted(player_data['Conference'].unique())
```




    ['AAC',
     'ACC',
     'ASUN',
     'America East',
     'Atlantic 10',
     'Big 12',
     'Big East',
     'Big Sky',
     'Big South',
     'Big Ten',
     'Big West',
     'CAA',
     'CUSA',
     'DI Independent',
     'Horizon',
     'Ivy League',
     'MAAC',
     'MAC',
     'MEAC',
     'MVC',
     'Mountain West',
     'NEC',
     'OVC',
     'Pac-12',
     'Patriot',
     'SEC',
     'SWAC',
     'SoCon',
     'Southland',
     'Summit League',
     'Sun Belt',
     'WAC',
     'WCC']



These values closely match the [list of conferences](https://en.wikipedia.org/wiki/NCAA_Division_I_women%27s_basketball_conference_tournaments) (with a few minor syntax differences such as "DI Independent" for independent schools) so this feature is complete. The rest of the features we'll be generating today will be based on straightforward calculations using existing columns.

<div class="email-subscription-container"></div>

## Calculate per-Game Metrics
The next set of columns we can calculate are per-game metrics. Each metric can be divided by the total number of games in the season to get the per-game average of that metric. For example, to determine the average points-per-game, we can divide the total points by the total games played. We could do this for almost every one of the numeric columns in this dataset, but let's focus on the number of minutes played, fouls, and the big five player statistics. 


```python
player_data['MINUTES_PER_GAME'] = player_data['MINUTES_PLAYED'] / player_data['GAMES']
player_data['FOULS_PER_GAME'] = player_data['FOULS'] / player_data['GAMES']
player_data['POINTS_PER_GAME'] = player_data['POINTS'] / player_data['GAMES']
player_data['ASSISTS_PER_GAME'] = player_data['ASSISTS'] / player_data['GAMES']
player_data['STEALS_PER_GAME'] = player_data['STEALS'] / player_data['GAMES']
player_data['BLOCKS_PER_GAME'] = player_data['BLOCKS'] / player_data['GAMES']
player_data['REBOUNDS_PER_GAME'] = player_data['TOTAL_REBOUNDS'] / player_data['GAMES']

player_data[['PLAYER_NAME', 'MINUTES_PER_GAME', 'FOULS_PER_GAME', 'POINTS_PER_GAME', 'ASSISTS_PER_GAME', 'STEALS_PER_GAME', 'BLOCKS_PER_GAME', 'REBOUNDS_PER_GAME']].sample(5)
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
      <th>MINUTES_PER_GAME</th>
      <th>FOULS_PER_GAME</th>
      <th>POINTS_PER_GAME</th>
      <th>ASSISTS_PER_GAME</th>
      <th>STEALS_PER_GAME</th>
      <th>BLOCKS_PER_GAME</th>
      <th>REBOUNDS_PER_GAME</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>865</th>
      <td>Dena Jarrells</td>
      <td>27.625000</td>
      <td>1.437500</td>
      <td>10.625000</td>
      <td>3.500000</td>
      <td>0.687500</td>
      <td>0.062500</td>
      <td>2.562500</td>
    </tr>
    <tr>
      <th>566</th>
      <td>Jasmine Gayles</td>
      <td>34.566667</td>
      <td>1.800000</td>
      <td>16.800000</td>
      <td>2.466667</td>
      <td>0.933333</td>
      <td>0.033333</td>
      <td>4.033333</td>
    </tr>
    <tr>
      <th>172</th>
      <td>Elena Rodriguez</td>
      <td>27.173913</td>
      <td>2.739130</td>
      <td>10.652174</td>
      <td>3.347826</td>
      <td>1.173913</td>
      <td>0.695652</td>
      <td>6.695652</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Caitlin Staley</td>
      <td>13.689655</td>
      <td>2.482759</td>
      <td>3.310345</td>
      <td>0.310345</td>
      <td>0.310345</td>
      <td>1.379310</td>
      <td>2.517241</td>
    </tr>
    <tr>
      <th>76</th>
      <td>Bria Sanders-Woods</td>
      <td>25.068966</td>
      <td>2.482759</td>
      <td>7.758621</td>
      <td>4.206897</td>
      <td>1.241379</td>
      <td>0.206897</td>
      <td>2.103448</td>
    </tr>
  </tbody>
</table>
</div>



These look great, so we can move on to the next feature. 

## Calculate Assist-to-Turnover Ratio
Another metric often used in basketball performance analysis is the assist-to-turnover ratio, so let's add a column for that. This can be calculated by dividing the number of assists by the number of turnovers for each player. 


```python
player_data['ASSIST_TO_TURNOVER'] = player_data['ASSISTS'] / player_data['TURNOVERS']

player_data[['PLAYER_NAME', 'ASSISTS', 'TURNOVERS', 'ASSIST_TO_TURNOVER']].sample(5)
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
      <th>ASSISTS</th>
      <th>TURNOVERS</th>
      <th>ASSIST_TO_TURNOVER</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>766</th>
      <td>Kylee Mabry</td>
      <td>35</td>
      <td>61</td>
      <td>0.573770</td>
    </tr>
    <tr>
      <th>678</th>
      <td>Quay Miller</td>
      <td>42</td>
      <td>72</td>
      <td>0.583333</td>
    </tr>
    <tr>
      <th>825</th>
      <td>MiLaysia Fulwiley</td>
      <td>70</td>
      <td>54</td>
      <td>1.296296</td>
    </tr>
    <tr>
      <th>294</th>
      <td>JuJu Watkins</td>
      <td>96</td>
      <td>120</td>
      <td>0.800000</td>
    </tr>
    <tr>
      <th>588</th>
      <td>Jasmine Shavers</td>
      <td>50</td>
      <td>80</td>
      <td>0.625000</td>
    </tr>
  </tbody>
</table>
</div>



That's all we need to do for the assist-to-turnover ratio, so let's create the final feature.

## Calculate Fantasy Points
The last column we'll calculate today is a metric called [Fantasy Points](https://sportsdata.io/developers/fantasy-scoring-system/nba). There are multiple ways to calculate this metric, but today we'll go with a weighted sum of the major metrics. Notice that, in this method, nearly all of the metrics contribute positively towards the Fantasy Points, but turnovers contributes negatively. This means players with higher points, rebounds, assists, blocks, and steals are rewarded, whereas players with high turnovers are slightly penalized. 


```python
player_data['FANTASY_POINTS'] = (player_data['THREE_POINTS_MADE'] * 3) + \
                                (player_data['TWO_POINTS_MADE'] * 2) + \
                                (player_data['FREE_THROWS_MADE'] * 1) + \
                                (player_data['TOTAL_REBOUNDS'] * 1.2) + \
                                (player_data['ASSISTS'] * 1.5) + \
                                (player_data['BLOCKS'] * 2) + \
                                (player_data['STEALS'] * 2) + \
                                (player_data['TURNOVERS'] * -1)

player_data[['PLAYER_NAME', 'FANTASY_POINTS']].sample(5)
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
      <th>FANTASY_POINTS</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>152</th>
      <td>Benthe Versteeg</td>
      <td>751.1</td>
    </tr>
    <tr>
      <th>266</th>
      <td>Rayah Marshall</td>
      <td>803.0</td>
    </tr>
    <tr>
      <th>501</th>
      <td>Lex Therien</td>
      <td>932.7</td>
    </tr>
    <tr>
      <th>411</th>
      <td>Ashlee Locke</td>
      <td>395.0</td>
    </tr>
    <tr>
      <th>145</th>
      <td>Last-Tear Poa</td>
      <td>393.3</td>
    </tr>
  </tbody>
</table>
</div>



This completes the feature engineering section of this project! Let's take a look at our final set of columns and data types before we move on to data visualization and machine learning. 


```python
player_data.dtypes
```




    Unnamed: 0                  int64
    PLAYER_NAME                object
    Team                       object
    Class                      object
    Height                      int64
    Position                   object
    PLAYER_ID                  object
    TEAM_NAME                  object
    GAMES                       int64
    MINUTES_PLAYED              int64
    FIELD_GOALS_MADE            int64
    FIELD_GOAL_ATTEMPTS         int64
    FIELD_GOAL_PERCENTAGE     float64
    THREE_POINTS_MADE           int64
    THREE_POINT_ATTEMPTS        int64
    THREE_POINT_PERCENTAGE    float64
    FREE_THROWS_MADE            int64
    FREE_THROW_ATTEMPTS         int64
    FREE_THROW_PERCENTAGE     float64
    OFFENSIVE_REBOUNDS          int64
    DEFENSIVE_REBOUNDS          int64
    TOTAL_REBOUNDS              int64
    ASSISTS                     int64
    TURNOVERS                   int64
    STEALS                      int64
    BLOCKS                      int64
    FOULS                       int64
    POINTS                      int64
    TWO_POINTS_MADE             int64
    TWO_POINT_ATTEMPTS          int64
    TWO_POINT_PERCENTAGE      float64
    Conference                 object
    MINUTES_PER_GAME          float64
    FOULS_PER_GAME            float64
    POINTS_PER_GAME           float64
    ASSISTS_PER_GAME          float64
    STEALS_PER_GAME           float64
    BLOCKS_PER_GAME           float64
    REBOUNDS_PER_GAME         float64
    ASSIST_TO_TURNOVER        float64
    FANTASY_POINTS            float64
    dtype: object



If you're going to use a new Jupyter notebook / Python script for the next part of this series, then it's a good idea to export this final dataset. As a reminder, you can use the `to_csv()` method instead of `.to_excel()` if you prefer. 


```python
player_data.to_excel('player_data_engineered.xlsx', index=False)
```

# Wrap up 
In today's guide, we expanded upon our dataset by engineering a few new features. In the next part, we'll take a closer look at the underlying data in each column and create visualizations to identify the relationship between various parameters.

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/) (Today's Guide)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/)
8. [Evaluating the Machine Learning Model](/)

<div class="email-subscription-container"></div>
<div id="sources"></div>

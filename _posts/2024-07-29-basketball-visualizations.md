---
layout: post
title: "Data Visualizations"
subtitle: "Outlier or Caitlin Clark? [Part 5]"
tags:  [Python, data science, pandas, machine learning, scikit-learn, data visualization]
thumbnail-img: /assets/img/posts/2024-07-29-basketball-visualizations/histogram.png
share-title: "Data Visualizations: Outlier or Caitlin Clark? [Part 5]" 
share-description: Interested in creating charts and graphs for your data? Learn how to select meaningful charts and generate helpful graphs in the latest installment of this data science series that is perfect for beginner data scientists and Python enthusiasts.
share-img: /assets/img/posts/2024-07-29-basketball-visualizations/social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll demonstrate how to generate meaningful visualizations. This is the fifth part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

<div id="toc"></div>

# Getting Started

First, let's take a look at an overview of this data science project. If you're already familiar with it, feel free to skip to the [next section](#generate-visualizations).

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

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Due to the size of this project, today we'll cover the sixth step: creating visualizations.

## Dependencies
Since this is the fifth installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [first article of the series](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. 

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
import scipy
```

Since I'm creating these graphs using [JupyterLab](https://jupyter.org/), I'll also make sure the `jupyterlab-plotly` extension is installed and will specify that plotly charts should display as an iframe. 


```python
pio.renderers.default = 'iframe'
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



## Note about Graphing Packages
The example visualizations in this article use multiple graphing packages: [seaborn](https://seaborn.pydata.org/), [plotly](https://plotly.com/python/), [matplotlib](https://matplotlib.org/), and the [pandas DataFrame plot() method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.plot.html) (which uses matplotlib under the hood by default). Each of these graphing packages has strengths and weaknesses that could be the subject of an entire article. We could create all of our visualizations with just one of those three graphing packages (seaborn, matplotlib, plotly), but we'll use a little bit of all three today. 

Here's a quick example of a scatterplot created with the same data using each graphing package. 


```python
plt.scatter(x=player_data['FIELD_GOALS_MADE'], y=player_data['POINTS'])
plt.title('Matplotlib Scatterplot')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_8_0.png)
    



```python
player_data.plot.scatter(x='FIELD_GOALS_MADE', y='POINTS', title='pandas Scatterplot (matplotlib backend)')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_9_0.png)
    



```python
sns.scatterplot(player_data, x='FIELD_GOALS_MADE', y='POINTS').set_title('Seaborn Scatterplot')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_10_0.png)
    



```python
fig = px.scatter(player_data, x='FIELD_GOALS_MADE', y='POINTS', title='Plotly Scatterplot')
fig.show()
```


![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_plotly.PNG)



We can see that each graph is displaying the same dataset, but the syntax and default formatting for each one differs slightly. 

## Set Graph Preferences
This is an entirely optional step to configure the aesthetics of the graphs and charts. You can import a custom color scheme or set colors individually. In this case, we'll define a list of custom colors (`graph_colors`) and configure both Matplotlib and Seaborn to use these colors for subsequent plots.


```python
graph_colors = ['#615EEA', '#339E7A', '#ff8c6c']
plt.rcParams['axes.prop_cycle'] = plt.cycler(color=graph_colors)
sns.set_palette(graph_colors)
```

We can also set the overall style for the matplotlib plots using a style sheet. You can print the list of available style sheets and view examples of these style sheets [on matplotlib's website](https://matplotlib.org/stable/gallery/style_sheets/style_sheets_reference.html).


```python
print(plt.style.available)
```

    ['Solarize_Light2', '_classic_test_patch', '_mpl-gallery', '_mpl-gallery-nogrid', 'bmh', 'classic', 'dark_background', 'fast', 'fivethirtyeight', 'ggplot', 'grayscale', 'seaborn-v0_8', 'seaborn-v0_8-bright', 'seaborn-v0_8-colorblind', 'seaborn-v0_8-dark', 'seaborn-v0_8-dark-palette', 'seaborn-v0_8-darkgrid', 'seaborn-v0_8-deep', 'seaborn-v0_8-muted', 'seaborn-v0_8-notebook', 'seaborn-v0_8-paper', 'seaborn-v0_8-pastel', 'seaborn-v0_8-poster', 'seaborn-v0_8-talk', 'seaborn-v0_8-ticks', 'seaborn-v0_8-white', 'seaborn-v0_8-whitegrid', 'tableau-colorblind10']



```python
plt.style.use('seaborn-v0_8-white')
```

That's all the setup steps needed, so we're ready to move on to generating visualizations.

# Generate Visualizations
We explored the relationships between our variables a bit in the previous article in this series, so let's use those selected features to create a few visualizations. We'll cover scatterplots, stacked bar charts, heatmaps, violin plots, joint plots, and histograms.

## Scatterplot of Field Goals Made vs Points
First, let's plot a scatterplot of field goals made versus points. This will illustrate the relationship between the number of field goals a player makes and the total points they score.


```python
plt.scatter(player_data['FIELD_GOALS_MADE'], player_data['POINTS'], label='Player')
plt.xlabel('Field Goals Made')
plt.ylabel('Points')
plt.title('Field Goals Made vs Points')
plt.legend()
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_39_0.png)
    


### Add Line of Best Fit to Scatterplot
Since this looks like a linear relationship, we can also add a line of best fit to the scatterplot, which is a straight line that attempts to closely match the trend of the data on the scatter plot. This line may pass through some of the points, none of the points, or all of the points. This will give us a visual representation (as well as a mathematical formula) of the correlation between the number of field goals made and the total points. We can calculate the line of best fit using scipy's [linregress](https://docs.scipy.org/doc//scipy-1.10.1/reference/generated/scipy.stats.linregress.html) function: 


```python
slope, intercept, r_value, p_value, std_err = scipy.stats.linregress(player_data['FIELD_GOALS_MADE'], player_data['POINTS'])
x_values = np.array([min(player_data['FIELD_GOALS_MADE']), max(player_data['FIELD_GOALS_MADE'])])
y_values = slope * x_values + intercept
print(f"Equation of the trend line: y = {slope:.2f}x + {intercept:.2f}")
```

    Equation of the trend line: y = 2.65x + 10.96


Now that we've calculated the line of best fit, we can add it to the scatterplot: 


```python
plt.scatter(player_data['FIELD_GOALS_MADE'], player_data['POINTS'], label='Player')
plt.plot(x_values, y_values, color='#ff8c6c', label='Line of Best Fit')
plt.xlabel('Field Goals Made')
plt.ylabel('Points')
plt.title('Field Goals Made vs Points')
plt.legend()
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_43_0.png)
    


In this scatterplot, we can see a dense clump of players up until around 225 field goals made, and then a more spread out distribution of players with between 225 and 300 field goals made. There's also one player who scored more than 300 field goals in the season, represented as the single outlier in the far upper right corner of the graph. Let's identify that top player.

### View Top Players by Points
To see which players made the highest number of field goals in the season, we can use the [pandas nlargest() function](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.nlargest.html). By passing the arguments `5` and `FIELD_GOALS_MADE`, we can specify that we want to select the five rows with the highest value in the `FIELD_GOALS_MADE` column. 


```python
player_data.nlargest(5, 'FIELD_GOALS_MADE')[['PLAYER_NAME', 'TEAM_NAME', 'Class', 'Position', 'FIELD_GOALS_MADE']]
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
      <th>TEAM_NAME</th>
      <th>Class</th>
      <th>Position</th>
      <th>FIELD_GOALS_MADE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>Caitlin Clark</td>
      <td>Iowa</td>
      <td>Senior</td>
      <td>Guard</td>
      <td>332</td>
    </tr>
    <tr>
      <th>263</th>
      <td>Elizabeth Kitley</td>
      <td>Virginia Tech</td>
      <td>Senior</td>
      <td>Center</td>
      <td>278</td>
    </tr>
    <tr>
      <th>294</th>
      <td>JuJu Watkins</td>
      <td>Indiana</td>
      <td>Freshman</td>
      <td>Guard</td>
      <td>270</td>
    </tr>
    <tr>
      <th>136</th>
      <td>Lucy Olsen</td>
      <td>Iowa</td>
      <td>Junior</td>
      <td>Guard</td>
      <td>268</td>
    </tr>
    <tr>
      <th>239</th>
      <td>Chellia Watson</td>
      <td>Buffalo</td>
      <td>Senior</td>
      <td>Guard</td>
      <td>258</td>
    </tr>
  </tbody>
</table>
</div>



From this table, we can see that the player who made the most field goals in the season was Caitlin Clark. She scored 50+ additional field goals than Elizabeth Kitley, the player with the second highest number of field goals made. 

## Scatterplot of Minutes Played vs Points
Next, let's create a scatterplot to explored the relationship between the total minutes played and the total points scored by the player. As a reminder, the minutes played measures the time on the game clock that the player was one of the five teammates on the court. This does not include time on the bench or when the game clock is stopped (such as for free throws or halftime). 


```python
player_data.plot(x='MINUTES_PLAYED', y='POINTS', kind='scatter', title='Minutes Played vs Points')
plt.xlabel('Minutes Played')
plt.ylabel('Points Scored')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_49_0.png)
    


## Box Plot of Height by Position

Next, let’s take a look at a categorical value (position) compared to the player’s height. A box plot (a.k.a. box-and-whisker plot, candlestick chart) allows us to compare the height distribution of players across different positions in one visualization. Each box represents the [interquartile range](https://en.wikipedia.org/wiki/Interquartile_range) (the middle 50% of the height distribution for each position) with the line in the box indicating the [median](https://en.wikipedia.org/wiki/Median) height. The length of the whiskers can vary depending on the method used, with the simplest method being to extend the whiskers to the highest and lowest data points. [Seaborn’s box plot documentation](https://seaborn.pydata.org/tutorial/categorical.html#boxplots) mentions that the whiskers extend to the highest and lowest data points (height values) within 1.5 times the interquartile range of the upper and lower quartile, and any points beyond that are considered outliers.


```python
# Get the unique values in the "Position" column and sort alphabetically
position_order = sorted(player_data['Position'].unique())

# Create a box plot
plt.figure(figsize=(10, 6))
sns.boxplot(x='Position', y='Height', data=player_data, order=position_order)
plt.title('Height Distribution by Position')
plt.xlabel('Position')
plt.ylabel('Height (inches)')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_52_0.png)
    


From the boxes on this plot, we can see that Centers are generally the tallest and Guards are generally the shortest players in this dataset. Forwards, on the other hand, have a smaller interquartile range in between Centers and Guards, and as a result have more outliers drawn on the plot. 

## Box Plot of Three Points Made by Conference
We can create another box plot that shows the number of three point field goals each player made by `Conference`. This gives us a high level overview of the distribution of successful three-pointers for players in each `Conference`. 


```python
plt.figure(figsize=(12, 8))
sns.boxplot(x="Conference", y="THREE_POINTS_MADE", data=player_data, order=sorted(player_data['Conference'].unique()))
plt.xticks(rotation=90)
plt.title('Three Points Made by Conference')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_55_0.png)
    


This chart can also highlight issues with how data is distributed for each category. For example, the box-and-whiskers for the `DI Independent` conference is quite collapsed compared to the other conferences. This could indicate a lack of data for that conference, which we can verify using the panda's [groupby()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) and [size()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.size.html) functions.


```python
player_data[player_data['THREE_POINTS_MADE'] > 0].groupby('Conference').size()
```




    Conference
    AAC               35
    ACC               37
    ASUN              16
    America East      12
    Atlantic 10       35
    Big 12            40
    Big East          28
    Big Sky           20
    Big South         15
    Big Ten           40
    Big West          24
    CAA               28
    CUSA              19
    DI Independent     1
    Horizon           22
    Ivy League        15
    MAAC              26
    MAC               23
    MEAC              10
    MVC               29
    Mountain West     25
    NEC               12
    OVC               22
    Pac-12            33
    Patriot           16
    SEC               41
    SWAC              18
    SoCon             11
    Southland         19
    Summit League     18
    Sun Belt          33
    WAC               26
    WCC               24
    dtype: int64



We can see that the DI Independent conference only has one data point where the number of `THREE_POINTS_MADE` is greater than zero, which explains why the box in the chart above looks more like a single point. 

Also we can see that the Big Ten conference has an outlier with significantly more `THREE_POINTS_MADE` compared to any other data point. We can verify who that player is with the panda's [groupby()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html) and [max()](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.max.html) functions.


```python
player_data.sort_values(by='THREE_POINTS_MADE', ascending=False)[['PLAYER_NAME', 'Conference', 'TEAM_NAME', 'THREE_POINTS_MADE']].head()
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
      <th>Conference</th>
      <th>TEAM_NAME</th>
      <th>THREE_POINTS_MADE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>78</th>
      <td>Caitlin Clark</td>
      <td>Big Ten</td>
      <td>Iowa</td>
      <td>168</td>
    </tr>
    <tr>
      <th>93</th>
      <td>Dyaisha Fair</td>
      <td>ACC</td>
      <td>Syracuse</td>
      <td>107</td>
    </tr>
    <tr>
      <th>647</th>
      <td>Aaliyah Nye</td>
      <td>SEC</td>
      <td>Alabama</td>
      <td>104</td>
    </tr>
    <tr>
      <th>639</th>
      <td>Sara Scalia</td>
      <td>Big Ten</td>
      <td>Indiana</td>
      <td>95</td>
    </tr>
    <tr>
      <th>648</th>
      <td>Maaliya Owens</td>
      <td>OVC</td>
      <td>Tennessee Tech</td>
      <td>94</td>
    </tr>
  </tbody>
</table>
</div>

<div class="email-subscription-container"></div>

## Stacked Bar Chart of Rebound Type by Position

Next, let's take a look at how rebound type is distributed among different positions using a stacked bar chart. A stacked bar chart allows us to see the proportion of each rebound type (offensive and defensive) for each position. The height of each bar represents the total number of rebounds, while the colors represent the percentage distribution between offensive and defensive rebounds.


```python
# Group by "Position" and sum the rebound columns
grouped_rebounds = player_data.groupby('Position').agg({
    'OFFENSIVE_REBOUNDS': 'sum',
    'DEFENSIVE_REBOUNDS': 'sum',
    'TOTAL_REBOUNDS': 'sum'
})

# Create a stacked bar plot
ax = grouped_rebounds.plot(x=None, y=['OFFENSIVE_REBOUNDS', 'DEFENSIVE_REBOUNDS'], kind="bar", rot=0, stacked=True)
ax.set_title('Total Rebounds by Position')
ax.set_xlabel('Position')
ax.set_ylabel('Total Rebounds')
ax.legend(title='Rebound Type')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_62_0.png)
    


We can check the totals for each position and type of rebound by printing `grouped_rebounds`. 


```python
grouped_rebounds
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
      <th>OFFENSIVE_REBOUNDS</th>
      <th>DEFENSIVE_REBOUNDS</th>
      <th>TOTAL_REBOUNDS</th>
    </tr>
    <tr>
      <th>Position</th>
      <th></th>
      <th></th>
      <th></th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>Center</th>
      <td>3949</td>
      <td>7695</td>
      <td>11644</td>
    </tr>
    <tr>
      <th>Forward</th>
      <td>18305</td>
      <td>37903</td>
      <td>56208</td>
    </tr>
    <tr>
      <th>Guard</th>
      <td>16555</td>
      <td>53039</td>
      <td>69594</td>
    </tr>
  </tbody>
</table>
</div>



From the stacked bar chart, we can see that Centers have the lowest number of rebounds, both offensively and defensively. Guards have slightly more total rebounds than Forwards, and all three positions have more defensive rebounds than offensive rebounds. 

## Heatmap of Player Assists and Steals by Height

Next, let's create a heatmap using [Seaborn's `heatmap()` function](https://seaborn.pydata.org/generated/seaborn.heatmap.html) to visualize the relationship between a player's height and their number of steals and assists. Heatmaps are great for identifying patterns and trends in large datasets. To make patterns easier to see, we grouped assists and steals into bins. Once the data is binned, we can configure the heatmap to show assists on the x-axis, height on the y-axis, and steals as the color. 


```python
# Define bins
assist_bins = np.arange(0, player_data['ASSISTS'].max() + 1, 10)
player_data['Assist_Bin'] = pd.cut(player_data['ASSISTS'], bins=assist_bins)

# Create heatmap
plt.figure(figsize=(10, 8))
heatmap_data = player_data.pivot_table(index='Assist_Bin', columns='Height', values='STEALS', aggfunc='sum', observed=False)
sns.heatmap(heatmap_data, cmap='Greens', linewidths=0.5, cbar_kws={'label': 'Steals'})
plt.title('Steals and Assists by Height')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_68_0.png)
    


From this graph, we can see a general upward trend between height and assists. However, the number of steals is more clumped, with the highest number of steals in three separate spots.

## Violin Plot of Points by Class

A violin plot is particularly useful for visualizing the distribution of data points. It combines aspects of a box plot and a kernel density plot to show the distribution shape and spread of the data. Similar to the box plot, the violin plot shows the same interquartile range and median, but with the distribution (smoothed using the [kernel density estimation](https://en.wikipedia.org/wiki/Kernel_density_estimation) method) plotted on either side. The width of each "violin" at each point, for example, would indicate how many players of that class scored that many total points in the season.

Let's use Seaborn's [violinplot() function](https://seaborn.pydata.org/generated/seaborn.violinplot.html) to visualize the distribution of points scored by players across different classes (academic years).


```python
plt.figure(figsize=(12, 8))
sns.violinplot(x="Class", y="POINTS", data=player_data, order=['Freshman', 'Sophomore', 'Junior', 'Senior'])
plt.title('Distribution of Points Scored by Class')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_72_0.png)
    


From this plot, we can see that the shape of the distribution for each class is slightly different, but each one has a peak between 200 and 400 points. The Freshman and Senior distributions both have long "tails" at the top, but the top "tail" for the Senior distribution is both longer and thinner. Since it's so thin, perhaps the long upper tail for Seniors is being created by one player who scored substantially more points than any other Senior. Let's explore this by re-creating the violin plots but exclude Caitlin Clark from the dataset.


```python
violin_data = player_data.loc[player_data['PLAYER_NAME'] != 'Caitlin Clark']

plt.figure(figsize=(12, 8))
sns.violinplot(x="Class", y="POINTS", data=violin_data, order=['Freshman', 'Sophomore', 'Junior', 'Senior'])
plt.title('Distribution of Points Scored by Class excluding Caitlin Clark')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_74_0.png)
    


From this new plot, we can see that the rest of the distributions are the same, but the long upper tail for Seniors has entirely disappeared. Unlike box plots, Seaborn violin plots do not represent outliers are separate dots, so that one data point (Caitlin Clark) was being represented in the distribution as a long upper tail. 

## Histogram of Fouls per Game

We can look at the distribution of average fouls per game using Seaborn's [histplot() function](https://seaborn.pydata.org/generated/seaborn.histplot.html). The `bins` parameter lets us specify how many bins we want the data grouped into, and the `kde` parameter allows us to add a [Kernel Density Estimate (KDE)](https://en.wikipedia.org/wiki/Kernel_density_estimation) line to the graph. The KDE line shows a smoothed estimate of the data's distribution. 


```python
plt.figure(figsize=(10, 8))
sns.histplot(player_data['FOULS_PER_GAME'], bins=30, kde=True)
plt.title('Distribution of Fouls per Game')
plt.xlabel('Number of Fouls per Game')
plt.ylabel('Count')
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_78_0.png)
    


From this chart, we can see that most players averaged between 1.5 and 3.0 fouls per game over the entire season. The players at the left-most end of the graph averaged around half a foul per game (or one foul every other game) and the players at the right-most end of the graph averaged nearly four fouls per game (or nearly fouling out every single game). The peak of this distribution is between 2.0 and 2.5 fouls per game. 

## Joint Plot of Fantasy Points vs Turnovers

In this section, we'll create a joint plot to visualize the relationship between turnovers and fantasy points. This joint plot combines a scatter plot (including a regression line) with histograms to show the distribution and correlation of two variables. This will help us understand how the number of turnovers impacts the fantasy points scored by a player.


```python
sns.jointplot(x='TURNOVERS', y='FANTASY_POINTS', data=player_data, kind='reg', scatter_kws={'alpha': 0.5})
plt.subplots_adjust(top=0.9)
plt.suptitle('Fantasy Points vs Turnovers', fontsize=16)
plt.show()
```


    
![png](/assets/img/posts/2024-07-29-basketball-visualizations/output_82_0.png)
    


From the joint plot, we can see the scatter of data points representing the relationship between turnovers and fantasy points. The regression line helps us identify any potential correlation between these two variables.

# Wrap Up
In this article, we focused on generating a variety of visualizations for our dataset. Today's visualizations were primarily an exploration of various chart types and options that can be helpful. Which charts you want to include in your report will depend on your dataset, but it's generally a good idea to try out multiple chart types and variable pairings before deciding on which ones are best. In the next article, we'll go over how to select the right machine learning model for a given problem. 

Also, all of the code snippets in today's guide are available in a Jupyter Notebook in the [ncaa-basketball-stats](https://github.com/pineconedata/ncaa-basketball-stats) repository on [GitHub](https://github.com/pineconedata/).

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/) (Today's Guide)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/)
8. [Evaluating the Machine Learning Model](/2024-11-27-basketball-evaluate-ols-model/)


<div class="email-subscription-container"></div>
<div id="sources"></div>

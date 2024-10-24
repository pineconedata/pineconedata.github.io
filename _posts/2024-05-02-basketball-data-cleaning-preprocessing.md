---
layout: post
title: "Data Cleaning and Preprocessing"
subtitle: "Outlier or Caitlin Clark? [Part 2]"
tags:  [Python, data science, pandas, API]
thumbnail-img: /assets/img/posts/2024-05-02-basketball-data-cleaning/clean.jpg
share-title: "Outlier or Caitlin Clark? A Data Science Project: Part 1 - Project Setup and Data Acquisition" 
share-description: Interested in the fundamental steps of any data science project? Learn how to thoroughly clean and preprocess your datasets in this comprehensive guide that is perfect for beginner data scientists and Python enthusiasts.
share-img: /assets/img/posts/2024-05-02-basketball-data-cleaning/data-cleaning-social.png
readtime: true
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
after-content: post-subscribe.html
---

Today we'll walk through how to clean and preprocess a dataset to ensure it is ready for analysis. This is the second part of a series that walks through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, machine learning, and creating visualizations. 

As a reminder, the dataset we'll be using in this project contains individual basketball player statistics (such as total points scored and blocks made) for the 2023-2024 NCAA women's basketball season. Here's a brief description of each major step that we'll go through for this project: 

![the steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_project_steps.png "the steps for this data science project")

1. **Data Acquisition** - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. **Data Cleaning** - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. **Data Preprocessing** - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. **Feature Engineering** - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.
5. **Machine Learning** - This step focuses on training a machine learning model to identify the combination of individual player statistics that correlates with optimal performance. 
6. **Creating Visualizations** - This step involves identifying the relationships between various parameters (such as height and blocked shots) and generating meaningful visualizations (such as bar charts, scatterplots, and candlestick charts).

We'll use Python along with the popular [pandas](https://pandas.pydata.org/docs/) and [requests](https://requests.readthedocs.io/en/latest/) libraries to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, create meaningful visualizations, and train a basic machine learning model. Since we already gathered the raw data from online sources in the last part, let's move on to the data cleaning and preprocessing steps. 

<div id="toc"></div>

# Getting Started
Since this is the second installment in the series, you likely already have your environment setup and can skip to the next section. If you're not already set up and you want to follow along on your own machine, it's recommended to read the [previous post](/2024-04-11-basketball-data-acquisition/) or at least review the [Getting Started](/2024-04-11-basketball-data-acquisition/#getting-started) section of that post before continuing. In summary, you'll want to have [Python](https://www.python.org/) installed with the following packages: 
  - [pandas](https://pandas.pydata.org/docs/)
  - [requests](https://requests.readthedocs.io/en/latest/)
  - [json](https://docs.python.org/3/library/json.html)
  - [os](https://docs.python.org/3/library/os.html)
  - [numpy](https://numpy.org/doc/)
  - [openpyxl](https://openpyxl.readthedocs.io/en/stable/)
  
In [Part 1](/2024-04-11-basketball-data-acquisition/) of this series, we acquired two datasets and combined them into one final dataset, stored in a dataframe named `player_data`. If you want to follow along with the code examples in this article, it's recommended to import the `player_data` dataframe before proceeding. 

```python
player_data = pd.read_excel('player_data_raw.xlsx')
```

# Data Cleaning
Before we can analyze the dataset, we need to ensure it is clean and reliable. In this section, we'll address issues like missing values, incorrect entries, and inconsistencies. This saves you the headache of training a model with unintended values or creating graphs without the full dataset.

## Handle Missing Values
Missing values (such as `None` and `NaN`) can significantly impact the accuracy and validity of statistical analyses and visualizations, so we want to identify and handle (remove, impute, accept) each instance of missing and empty values in the dataset. There are several ways to handle missing values, but let's take a look at the most common methods:
   - **Correction** - In limited circumstances, the missing values may be due to an import error or available in an alternate data source that can supplement your original dataset. This allows you to make corrections to the original dataset and eliminate missing values.
   - **Imputation** - Imputation involves replacing missing values with estimated or predicted values based on other available information in the dataset. Common imputation techniques include mean, median, mode imputation, or more advanced methods such as regression imputation or k-nearest neighbors (KNN) imputation.
   - **Deletion** - Deleting rows or columns with missing values is a straightforward approach, but it should be used carefully as it can lead to the loss of valuable information. Row deletion (also known as listwise deletion) removes entire observations with missing values, while column deletion (variable-wise deletion) removes entire variables with missing values.
   - **Advanced Techniques** - Advanced techniques such as multiple imputation, which generates multiple imputed datasets and combines the results, or sophisticated machine learning algorithms designed to handle missing data directly, offer more robust solutions for handling missing values in complex datasets.

How you handle missing values will depend on a variety of factors (the nature of the missing data, the requirements and objectives of your project, etc.) and should be evaluated on a case-by-case basis. For today, we'll go through each row with missing values and determine the best way to handle them one at a time. 

### Identify Missing Values
Let's begin by looking at any rows that contain at least one missing value (represented as `None` and `NaN` in Python). We'll use the [pandas `isna()` method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.isna.html) for this.


```python
player_data[player_data.isna().any(axis=1)]
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
      <th>FREE_THROW_PERCENTAGE</th>
      <th>OFFENSIVE_REBOUNDS</th>
      <th>DEFENSIVE_REBOUNDS</th>
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
      <th>148</th>
      <td>Ally Becki</td>
      <td>Ball St. (MAC)</td>
      <td>Jr.</td>
      <td>5-8</td>
      <td>NaN</td>
      <td>ncaaw.p.66590</td>
      <td>Ball St.</td>
      <td>31</td>
      <td>972</td>
      <td>143</td>
      <td>...</td>
      <td>75.4</td>
      <td>21</td>
      <td>122</td>
      <td>143</td>
      <td>148</td>
      <td>108</td>
      <td>60</td>
      <td>11</td>
      <td>69</td>
      <td>398</td>
    </tr>
    <tr>
      <th>245</th>
      <td>Caitlin Weimar</td>
      <td>Boston U. (Patriot)</td>
      <td>Sr.</td>
      <td>6-4</td>
      <td>F</td>
      <td>ncaaw.p.64464</td>
      <td>N.C. State</td>
      <td>28</td>
      <td>987</td>
      <td>199</td>
      <td>...</td>
      <td>68.4</td>
      <td>76</td>
      <td>219</td>
      <td>295</td>
      <td>54</td>
      <td>69</td>
      <td>37</td>
      <td>80</td>
      <td>74</td>
      <td>519</td>
    </tr>
    <tr>
      <th>250</th>
      <td>Abby Muse</td>
      <td>Boise St. (Mountain West)</td>
      <td>Sr.</td>
      <td>6-2</td>
      <td>F</td>
      <td>ncaaw.p.64516</td>
      <td>Boise St.</td>
      <td>31</td>
      <td>785</td>
      <td>90</td>
      <td>...</td>
      <td>54.3</td>
      <td>74</td>
      <td>191</td>
      <td>265</td>
      <td>34</td>
      <td>66</td>
      <td>36</td>
      <td>87</td>
      <td>72</td>
      <td>230</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Emily Bowman</td>
      <td>Samford (SoCon)</td>
      <td>So.</td>
      <td>6-5</td>
      <td>C</td>
      <td>ncaaw.p.64719</td>
      <td>Samford</td>
      <td>30</td>
      <td>703</td>
      <td>89</td>
      <td>...</td>
      <td>53.1</td>
      <td>83</td>
      <td>172</td>
      <td>255</td>
      <td>17</td>
      <td>55</td>
      <td>5</td>
      <td>74</td>
      <td>88</td>
      <td>238</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Christina Dalce</td>
      <td>Villanova (Big East)</td>
      <td>Jr.</td>
      <td>6-2</td>
      <td>F</td>
      <td>ncaaw.p.67708</td>
      <td>Villanova</td>
      <td>30</td>
      <td>833</td>
      <td>108</td>
      <td>...</td>
      <td>54.9</td>
      <td>145</td>
      <td>146</td>
      <td>291</td>
      <td>30</td>
      <td>52</td>
      <td>25</td>
      <td>70</td>
      <td>89</td>
      <td>255</td>
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
      <th>726</th>
      <td>Kseniia Kozlova</td>
      <td>James Madison (Sun Belt)</td>
      <td>Sr.</td>
      <td>6-3</td>
      <td>C</td>
      <td>ncaaw.p.64131</td>
      <td>James Madison</td>
      <td>24</td>
      <td>526</td>
      <td>118</td>
      <td>...</td>
      <td>57.1</td>
      <td>74</td>
      <td>97</td>
      <td>171</td>
      <td>19</td>
      <td>51</td>
      <td>8</td>
      <td>8</td>
      <td>55</td>
      <td>284</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Simone Cunningham</td>
      <td>Old Dominion (Sun Belt)</td>
      <td>Jr.</td>
      <td>5-10</td>
      <td>G</td>
      <td>ncaaw.p.113163</td>
      <td>Old Dominion</td>
      <td>30</td>
      <td>591</td>
      <td>84</td>
      <td>...</td>
      <td>56.5</td>
      <td>87</td>
      <td>123</td>
      <td>210</td>
      <td>12</td>
      <td>60</td>
      <td>20</td>
      <td>21</td>
      <td>93</td>
      <td>207</td>
    </tr>
    <tr>
      <th>729</th>
      <td>Otaifo Esenabhalu</td>
      <td>Longwood (Big South)</td>
      <td>Fr.</td>
      <td>6-2</td>
      <td>F</td>
      <td>ncaaw.p.113170</td>
      <td>Longwood</td>
      <td>30</td>
      <td>547</td>
      <td>60</td>
      <td>...</td>
      <td>60.0</td>
      <td>79</td>
      <td>128</td>
      <td>207</td>
      <td>10</td>
      <td>59</td>
      <td>23</td>
      <td>13</td>
      <td>78</td>
      <td>147</td>
    </tr>
    <tr>
      <th>755</th>
      <td>Sedayjha Payne</td>
      <td>Morgan St. (MEAC)</td>
      <td>Sr.</td>
      <td>5-8</td>
      <td>G</td>
      <td>ncaaw.p.113276</td>
      <td>Morgan St.</td>
      <td>29</td>
      <td>634</td>
      <td>80</td>
      <td>...</td>
      <td>52.0</td>
      <td>66</td>
      <td>73</td>
      <td>139</td>
      <td>24</td>
      <td>53</td>
      <td>67</td>
      <td>7</td>
      <td>58</td>
      <td>186</td>
    </tr>
    <tr>
      <th>843</th>
      <td>Madelyn Bischoff</td>
      <td>Ball St. (MAC)</td>
      <td>Jr.</td>
      <td>5-9</td>
      <td>NaN</td>
      <td>ncaaw.p.66600</td>
      <td>Ball St.</td>
      <td>30</td>
      <td>881</td>
      <td>112</td>
      <td>...</td>
      <td>86.6</td>
      <td>11</td>
      <td>60</td>
      <td>71</td>
      <td>37</td>
      <td>36</td>
      <td>20</td>
      <td>6</td>
      <td>38</td>
      <td>359</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 27 columns</p>
</div>



There are quite a few rows and columns in this output, so let's find out which columns contain missing values. 


```python
player_data.columns[player_data.isna().any()].tolist()
```




    ['Position', 'THREE_POINT_PERCENTAGE']



Now that we know which columns to focus on, let's look at just those two columns with the player name and team name.


```python
player_data[player_data.isna().any(axis=1)][['PLAYER_NAME', 'Team', 'Position', 'THREE_POINT_PERCENTAGE']]
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
      <th>Position</th>
      <th>THREE_POINT_PERCENTAGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>148</th>
      <td>Ally Becki</td>
      <td>Ball St. (MAC)</td>
      <td>NaN</td>
      <td>34.8</td>
    </tr>
    <tr>
      <th>245</th>
      <td>Caitlin Weimar</td>
      <td>Boston U. (Patriot)</td>
      <td>F</td>
      <td>None</td>
    </tr>
    <tr>
      <th>250</th>
      <td>Abby Muse</td>
      <td>Boise St. (Mountain West)</td>
      <td>F</td>
      <td>None</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Emily Bowman</td>
      <td>Samford (SoCon)</td>
      <td>C</td>
      <td>None</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Christina Dalce</td>
      <td>Villanova (Big East)</td>
      <td>F</td>
      <td>None</td>
    </tr>
    <tr>
      <th>...</th>
      <td>...</td>
      <td>...</td>
      <td>...</td>
      <td>...</td>
    </tr>
    <tr>
      <th>726</th>
      <td>Kseniia Kozlova</td>
      <td>James Madison (Sun Belt)</td>
      <td>C</td>
      <td>None</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Simone Cunningham</td>
      <td>Old Dominion (Sun Belt)</td>
      <td>G</td>
      <td>None</td>
    </tr>
    <tr>
      <th>729</th>
      <td>Otaifo Esenabhalu</td>
      <td>Longwood (Big South)</td>
      <td>F</td>
      <td>None</td>
    </tr>
    <tr>
      <th>755</th>
      <td>Sedayjha Payne</td>
      <td>Morgan St. (MEAC)</td>
      <td>G</td>
      <td>None</td>
    </tr>
    <tr>
      <th>843</th>
      <td>Madelyn Bischoff</td>
      <td>Ball St. (MAC)</td>
      <td>NaN</td>
      <td>40.3</td>
    </tr>
  </tbody>
</table>
<p>63 rows × 4 columns</p>
</div>



The `NaN` values in each column should be dealt with separately, meaning the `NaN`s in the `Position` column will be handled separately from the `NaN`s in the `THREE_POINT_PERCENTAGE` column. Since the THREE_POINT_PERCENTAGE is a calculated field, let's take a look at that first. 

### Handle Missing Three-Point Percentages
Three-point percentages in basketball are calculated by dividing the number of three-point baskets made by the number of three-point baskets attempted. So, let's add those two columns in to our output.


```python
player_data[player_data['THREE_POINT_PERCENTAGE'].isna()][['PLAYER_NAME', 'Team', 'Position', 'THREE_POINTS_MADE', 'THREE_POINT_ATTEMPTS', 'THREE_POINT_PERCENTAGE']]
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
      <th>Position</th>
      <th>THREE_POINTS_MADE</th>
      <th>THREE_POINT_ATTEMPTS</th>
      <th>THREE_POINT_PERCENTAGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>245</th>
      <td>Caitlin Weimar</td>
      <td>Boston U. (Patriot)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>250</th>
      <td>Abby Muse</td>
      <td>Boise St. (Mountain West)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Emily Bowman</td>
      <td>Samford (SoCon)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Christina Dalce</td>
      <td>Villanova (Big East)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>258</th>
      <td>Emily Saunders</td>
      <td>Youngstown St. (Horizon)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>264</th>
      <td>Lauren Betts</td>
      <td>UCLA (Pac-12)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>265</th>
      <td>Tenin Magassa</td>
      <td>Rhode Island (Atlantic 10)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>275</th>
      <td>Brooklyn Meyer</td>
      <td>South Dakota St. (Summit League)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>282</th>
      <td>Breya Cunningham</td>
      <td>Arizona (Pac-12)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>290</th>
      <td>Rochelle Norris</td>
      <td>Central Mich. (MAC)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>296</th>
      <td>Rita Igbokwe</td>
      <td>Ole Miss (SEC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>297</th>
      <td>Kyndall Golden</td>
      <td>Kennesaw St. (ASUN)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>306</th>
      <td>Kennedy Basham</td>
      <td>Oregon (Pac-12)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>307</th>
      <td>Maria Gakdeng</td>
      <td>North Carolina (ACC)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>311</th>
      <td>Kate Samson</td>
      <td>Navy (Patriot)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>316</th>
      <td>Liatu King</td>
      <td>Pittsburgh (ACC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>318</th>
      <td>Ashlee Lewis</td>
      <td>Cal St. Fullerton (Big West)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>319</th>
      <td>Jillian Archer</td>
      <td>St. John's (NY) (Big East)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>322</th>
      <td>Kylee Watson</td>
      <td>Notre Dame (ACC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>326</th>
      <td>Caitlin Staley</td>
      <td>Western Ky. (CUSA)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>332</th>
      <td>Sacha Washington</td>
      <td>Vanderbilt (SEC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>340</th>
      <td>Adreanna Waddle</td>
      <td>Prairie View (SWAC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>348</th>
      <td>River Baldwin</td>
      <td>NC State (ACC)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>353</th>
      <td>Zyheima Swint</td>
      <td>Hofstra (CAA)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>355</th>
      <td>Phillipina Kyei</td>
      <td>Oregon (Pac-12)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>358</th>
      <td>Jasmin Dixon</td>
      <td>Northwestern St. (Southland)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>361</th>
      <td>Shamarre Hale</td>
      <td>Austin Peay (ASUN)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>366</th>
      <td>Jadyn Donovan</td>
      <td>Duke (ACC)</td>
      <td>G</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>367</th>
      <td>Melyia Grayson</td>
      <td>Southern Miss. (Sun Belt)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>379</th>
      <td>ZiKeyah Carter</td>
      <td>Chicago St. (DI Independent)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>380</th>
      <td>Meghan Downing</td>
      <td>ETSU (SoCon)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>394</th>
      <td>Makayla Minett</td>
      <td>Denver (Summit League)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>401</th>
      <td>Ugonne Onyiah</td>
      <td>California (Pac-12)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>402</th>
      <td>Bella Murekatete</td>
      <td>Washington St. (Pac-12)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>409</th>
      <td>Zoe McCrary</td>
      <td>Col. of Charleston (CAA)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>411</th>
      <td>Ashlee Locke</td>
      <td>Mercer (SoCon)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>414</th>
      <td>Kyra Wood</td>
      <td>Syracuse (ACC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>420</th>
      <td>Laura Bello</td>
      <td>Idaho St. (Big Sky)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>424</th>
      <td>Khalis Cain</td>
      <td>UNC Greensboro (SoCon)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>428</th>
      <td>Hannah Noveroske</td>
      <td>Toledo (MAC)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>431</th>
      <td>Yaubryon Chambers</td>
      <td>Tennessee Tech (OVC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>433</th>
      <td>Sierra McCullough</td>
      <td>Eastern Ky. (ASUN)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>437</th>
      <td>India Howard</td>
      <td>North Ala. (ASUN)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>440</th>
      <td>Kayla Clark</td>
      <td>Bethune-Cookman (SWAC)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>443</th>
      <td>Katlyn Manuel</td>
      <td>ULM (Sun Belt)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>445</th>
      <td>Jordana Reisma</td>
      <td>Cleveland St. (Horizon)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>462</th>
      <td>Carys Roy</td>
      <td>Saint Peter's (MAAC)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>468</th>
      <td>Clarice Akunwafo</td>
      <td>Southern California (Pac-12)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>526</th>
      <td>Carter McCray</td>
      <td>Northern Ky. (Horizon)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>548</th>
      <td>Ajae Petty</td>
      <td>Kentucky (SEC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>664</th>
      <td>Ellie Mitchell</td>
      <td>Princeton (Ivy League)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>682</th>
      <td>Teneisia Brown</td>
      <td>FDU (NEC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>686</th>
      <td>Alancia Ramsey</td>
      <td>Coastal Carolina (Sun Belt)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>715</th>
      <td>Chyna Cornwell</td>
      <td>Rutgers (Big Ten)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>722</th>
      <td>Kennedy Taylor</td>
      <td>Missouri St. (MVC)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>726</th>
      <td>Kseniia Kozlova</td>
      <td>James Madison (Sun Belt)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>727</th>
      <td>Simone Cunningham</td>
      <td>Old Dominion (Sun Belt)</td>
      <td>G</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>729</th>
      <td>Otaifo Esenabhalu</td>
      <td>Longwood (Big South)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
    <tr>
      <th>755</th>
      <td>Sedayjha Payne</td>
      <td>Morgan St. (MEAC)</td>
      <td>G</td>
      <td>0</td>
      <td>0</td>
      <td>None</td>
    </tr>
  </tbody>
</table>
</div>



We can see that for all of the rows where THREE_POINT_PERCENTAGE is `None`, the three-point baskets made and attempted values are zero. If we manually calculated the three-point percentage for these rows, we would get `NaN` results instead of `None` due to the [division by zero](https://en.wikipedia.org/wiki/Division_by_zero). `NaN` and `None` are different values, but we've confirmed that there is no issue with the underlying data. The `None` values will automatically be replaced with `NaN`s later in this process, so we'll leave these values at `None` for now and move on to the `Position` field.

### Handle Missing Positions
Let's see how many rows are missing a Position value. 


```python
player_data[player_data['Position'].isna()][['PLAYER_NAME', 'Team', 'Position']]
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
      <th>Position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>148</th>
      <td>Ally Becki</td>
      <td>Ball St. (MAC)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>359</th>
      <td>Marie Kiefer</td>
      <td>Ball St. (MAC)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>709</th>
      <td>Ava Uhrich</td>
      <td>Southern Utah (WAC)</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>843</th>
      <td>Madelyn Bischoff</td>
      <td>Ball St. (MAC)</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>



It can be helpful to evaluate whether or not each column with missing values should have a missing or null value. In this case, it doesn't make much sense for a basketball player to be missing a position (guard, forward, center), since every player is assigned a standard position on their team. Knowing that, we can assume that our current dataset is missing these values in error, and the entries should be updated. 

We'll be using an external data source to look up the correct positions for each of these players. If there were more rows with a missing Position value, then we might acquire another dataset and combine it with the current one. However, there are only four rows here, so we'll look up the values manually. 

You can use several external data sources, but here's an example using [ESPN's website](https://www.espn.com/womens-college-basketball/). Search for the player's name and school to pull up the player's individual page (for example, [Ally Becki at Ball St.](https://www.espn.com/womens-college-basketball/player/_/id/4900609/ally-becki)). On this page, you can see the player's team, class, height, jersey number, as well as a `F`, `C`, or `G` for the position (in this example, the position is listed as `G` for Guard). 

![Manually look up a player's position](/assets/img/posts/2024-05-02-basketball-data-cleaning/ncaa_wbb_player_data_missing_position.png "Manually look up a player's position")

With that, we can locate and set the correct value of the Position field for each of the four players. 


```python
player_data.loc[player_data['PLAYER_NAME'] == 'Ally Becki', 'Position'] = 'G'
player_data.loc[player_data['PLAYER_NAME'] == 'Marie Kiefer', 'Position'] = 'F'
player_data.loc[player_data['PLAYER_NAME'] == 'Ava Uhrich', 'Position'] = 'F'
player_data.loc[player_data['PLAYER_NAME'] == 'Madelyn Bischoff', 'Position'] = 'G'
```

We can confirm that these changes were applied correctly by re-pulling the rows with a missing Position value.


```python
player_data[player_data['Position'].isna()][['PLAYER_NAME', 'Team', 'Position']]
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
      <th>Position</th>
    </tr>
  </thead>
  <tbody>
  </tbody>
</table>
</div>



There are no results, so we've now handled all of the values caught by the `isna()` method. This is a great start, but you'll often find datasets with missing or incorrect values that are not caught by the `isna()` method. 

## Handle Incorrect Values
Depending on the size of your dataset, the `unique()` method can be a good way to find additional missing values. Let's try that out on the Height column. 

### Handle Incorrect Heights
Using the `unique()` method, we can see one height value that is not valid: `0-0`.


```python
player_data['Height'].unique()
```




    array(['5-7', '5-8', '5-2', '5-9', '5-6', '6-0', '5-10', '5-3', '5-11',
           '5-5', '5-4', '6-2', '6-1', '6-3', '6-4', '6-6', '6-5', '6-7',
           '6-8', '0-0'], dtype=object)



Let's pull all of the rows where the Height is set to `0-0`. 


```python
player_data[player_data['Height'].eq('0-0')]
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
      <th>FREE_THROW_PERCENTAGE</th>
      <th>OFFENSIVE_REBOUNDS</th>
      <th>DEFENSIVE_REBOUNDS</th>
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
      <th>709</th>
      <td>Ava Uhrich</td>
      <td>Southern Utah (WAC)</td>
      <td>Fr.</td>
      <td>0-0</td>
      <td>F</td>
      <td>ncaaw.p.115529</td>
      <td>Southern Utah</td>
      <td>29</td>
      <td>877</td>
      <td>151</td>
      <td>...</td>
      <td>68.4</td>
      <td>65</td>
      <td>149</td>
      <td>214</td>
      <td>47</td>
      <td>55</td>
      <td>22</td>
      <td>19</td>
      <td>76</td>
      <td>383</td>
    </tr>
    <tr>
      <th>823</th>
      <td>Payton Hull</td>
      <td>Abilene Christian (WAC)</td>
      <td>Fr.</td>
      <td>0-0</td>
      <td>G</td>
      <td>ncaaw.p.112709</td>
      <td>Abilene Christian</td>
      <td>29</td>
      <td>837</td>
      <td>155</td>
      <td>...</td>
      <td>75.3</td>
      <td>33</td>
      <td>67</td>
      <td>100</td>
      <td>59</td>
      <td>87</td>
      <td>47</td>
      <td>8</td>
      <td>71</td>
      <td>444</td>
    </tr>
  </tbody>
</table>
<p>2 rows × 27 columns</p>
</div>



There are only two rows with missing heights, so we can manually look up and update these values using the same method as the missing Position values. 


```python
player_data.loc[player_data['PLAYER_NAME'] == 'Ava Uhrich', 'Height'] = '6-0'
player_data.loc[player_data['PLAYER_NAME'] == 'Payton Hull', 'Height'] = '5-11'
```

We can double-check that this worked properly, just like the Position field.


```python
player_data[player_data.eq('0-0').any(axis=1)]
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
      <th>FREE_THROW_PERCENTAGE</th>
      <th>OFFENSIVE_REBOUNDS</th>
      <th>DEFENSIVE_REBOUNDS</th>
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
  </tbody>
</table>
<p>0 rows × 27 columns</p>
</div>



### Handle Incorrect Classes
We can apply the same process used for Height to the Class field. Since there's only one player with an incorrect class, this is a quick update.


```python
player_data['Class'].unique()
player_data[player_data['Class'] == '---']
player_data.loc[player_data['PLAYER_NAME'] == 'Ayanna Khalfani', 'Class'] = 'Sr.'
player_data['Class'].unique()
```




    array(['Jr.', 'So.', 'Sr.', 'Fr.'], dtype=object)



Now that we've handled all of the empty and incorrect values, we're ready to move on to other data preprocessing steps. 

# Data Preprocessing 
The goal of this step is to make sure our dataset is consistent and suitable for analysis. This stepp will change quite a bit depending on the exact project. For this project, we'll be setting column datatypes, converting units, and substituting abbreviated values. For advanced machine learning projects, data preprocessing often includes additional steps like feature scaling, normalizing certain features, and encoding categorical values. 

## Data Type Conversion
Data type conversion is a fundamental step in preparing a dataset for analysis. Specifically, we often encounter situations where numbers are stored as strings or objects instead of their native numerical formats (e.g., integers or floats). In this step, let's take a closer look at the initial data type of each column and identify opportunities to convert these values into more suitable datatypes.


```python
player_data.dtypes
```




    PLAYER_NAME               object
    Team                      object
    Class                     object
    Height                    object
    Position                  object
    PLAYER_ID                 object
    TEAM_NAME                 object
    GAMES                     object
    MINUTES_PLAYED            object
    FIELD_GOALS_MADE          object
    FIELD_GOAL_ATTEMPTS       object
    FIELD_GOAL_PERCENTAGE     object
    THREE_POINTS_MADE         object
    THREE_POINT_ATTEMPTS      object
    THREE_POINT_PERCENTAGE    object
    FREE_THROWS_MADE          object
    FREE_THROW_ATTEMPTS       object
    FREE_THROW_PERCENTAGE     object
    OFFENSIVE_REBOUNDS        object
    DEFENSIVE_REBOUNDS        object
    TOTAL_REBOUNDS            object
    ASSISTS                   object
    TURNOVERS                 object
    STEALS                    object
    BLOCKS                    object
    FOULS                     object
    POINTS                    object
    dtype: object



We can see that all of the columns currently have the `object` data type, which is not ideal for numeric columns like number of minutes played or field goals made. 

### Convert Numeric-only Columns
There are a few reasons to convert columns from the object to numeric data type. The main reason is that most mathematical operations (including numpy methods and sum/averages/etc. taken during aggregation steps) only work on numeric data types. Plotting and visualization libraries also expect numeric data types, so you may not be able to create any charts or graphs (besides categorical charts like histograms) unless specific columns use the numeric data type. Storing numbers as strings can give unexpected results, such as sorting the values `1`, `2`, and `11` in the order `1`, `11`, `2`. For large datasets, it's worth noting that this can also take up more memory and be slower than storing numbers in numeric data types. 

Now that we know why we would want to convert numeric columns to numeric data types, let's start by defining which columns we expect to store only numeric data.


```python
numeric_columns = ['GAMES', 'MINUTES_PLAYED', 'FIELD_GOALS_MADE', 'FIELD_GOAL_ATTEMPTS', 
                  'FIELD_GOAL_PERCENTAGE', 'THREE_POINTS_MADE', 'THREE_POINT_ATTEMPTS', 
                  'THREE_POINT_PERCENTAGE', 'FREE_THROWS_MADE', 'FREE_THROW_ATTEMPTS', 
                  'FREE_THROW_PERCENTAGE', 'OFFENSIVE_REBOUNDS', 'DEFENSIVE_REBOUNDS', 
                  'TOTAL_REBOUNDS', 'ASSISTS', 'TURNOVERS', 'STEALS', 'BLOCKS', 'FOULS', 'POINTS']
```

Now that we have this list of columns, we can use the [pandas `to_numeric()` method](https://pandas.pydata.org/docs/reference/api/pandas.to_numeric.html) to convert these columns to the appropriate numeric data type. We can confirm that worked by re-checking the data type of each column. 


```python
player_data[numeric_columns] = player_data[numeric_columns].apply(pd.to_numeric)
player_data.dtypes
```




    PLAYER_NAME                object
    Team                       object
    Class                      object
    Height                     object
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
    dtype: object



All of the defined numeric columns were properly converted to either integers or floats. The percentage fields are now stored as floats, while the rest of them are integers.

We can also revisit the `None` values in the THREE_POINT_PERCENTAGE column to see if they were impacted by the datatype conversion. 

```python
player_data[player_data['THREE_POINT_PERCENTAGE'].isna()][['PLAYER_NAME', 'Team', 'Position', 'THREE_POINTS_MADE', 'THREE_POINT_ATTEMPTS', 'THREE_POINT_PERCENTAGE']].head()
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
      <th>Position</th>
      <th>THREE_POINTS_MADE</th>
      <th>THREE_POINT_ATTEMPTS</th>
      <th>THREE_POINT_PERCENTAGE</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>245</th>
      <td>Caitlin Weimar</td>
      <td>Boston U. (Patriot)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>250</th>
      <td>Abby Muse</td>
      <td>Boise St. (Mountain West)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>254</th>
      <td>Emily Bowman</td>
      <td>Samford (SoCon)</td>
      <td>C</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>257</th>
      <td>Christina Dalce</td>
      <td>Villanova (Big East)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
    <tr>
      <th>258</th>
      <td>Emily Saunders</td>
      <td>Youngstown St. (Horizon)</td>
      <td>F</td>
      <td>0</td>
      <td>0</td>
      <td>NaN</td>
    </tr>
  </tbody>
</table>
</div>


So the `to_numeric()` method automatically handled the `None` values in the three-point percentage column and converted them to `NaN`s. With the numeric columns converted, we can move on to the columns that should contain only text. 

### Convert Text-only Columns
In pandas, both the object and string data types can be used to represent text. However, there is a slight difference between them:
   - object dtype - This is the default dtype and is a catch-all for any non-numeric data. It can hold any Python object, including strings, lists, dictionaries, etc. When a column contains multiple data types or when pandas cannot guess the data type, it defaults to the object dtype. While this is more flexible, operations on columns with object dtype may be slower compared to columns with the string data type. The [pandas documentation](https://pandas.pydata.org/pandas-docs/stable/user_guide/text.html) outlines reasons why it's better to use the string dtype for storing text-only data. 
   - string dtype - This dtype specifically represents strings. Columns with this dtype contain only string data (not a mixture of string and other dtypes), which improves readability and allows for better dtype-specific operations like the `select_dtypes()` method. Additionally, with the str dtype, you can use some string-specific methods and functions directly on the column without the need for explicit type conversions. 

Let's start by identifying which columns should contain only string data. 


```python
string_columns = ['PLAYER_NAME', 'Team', 'Class', 'Height', 'Position', 'PLAYER_ID', 'TEAM_NAME']
```

We can use the [pandas `astype()` method]() to convert object columns to string columns. However, you should watch out for values with other data types when performing this conversion. For example, any missing values (such as `NaN`) will be converted to the string `"NaN"`. We've already replaced all of the missing values in these columns, so this is no longer a concern. However, we do not want other non-string values (numbers, arrays, dictionaries, etc.) to be converted either, so it's best to check for those before performing the conversion. For this dataset, we do not expect these columns to have any non-string values, but it's a good idea to look at a sample of the data in each column just to be safe. 


```python
player_data[string_columns].sample(10)
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>400</th>
      <td>Precious Johnson</td>
      <td>Duquesne (Atlantic 10)</td>
      <td>Sr.</td>
      <td>6-4</td>
      <td>C</td>
      <td>ncaaw.p.61337</td>
      <td>Duquesne</td>
    </tr>
    <tr>
      <th>385</th>
      <td>Meghan O'Brien</td>
      <td>Lehigh (Patriot)</td>
      <td>Jr.</td>
      <td>6-1</td>
      <td>F</td>
      <td>ncaaw.p.66792</td>
      <td>Lehigh</td>
    </tr>
    <tr>
      <th>770</th>
      <td>Carmen Villalobos</td>
      <td>Cleveland St. (Horizon)</td>
      <td>Sr.</td>
      <td>6-1</td>
      <td>G</td>
      <td>ncaaw.p.61574</td>
      <td>Cleveland St.</td>
    </tr>
    <tr>
      <th>180</th>
      <td>Ashley Hawkins</td>
      <td>Gardner-Webb (Big South)</td>
      <td>Jr.</td>
      <td>5-6</td>
      <td>G</td>
      <td>ncaaw.p.100658</td>
      <td>Gardner-Webb</td>
    </tr>
    <tr>
      <th>821</th>
      <td>Natalie Picton</td>
      <td>Montana St. (Big Sky)</td>
      <td>Fr.</td>
      <td>5-5</td>
      <td>G</td>
      <td>ncaaw.p.112319</td>
      <td>Montana St.</td>
    </tr>
    <tr>
      <th>273</th>
      <td>Jada Tiggett</td>
      <td>N.C. Central (MEAC)</td>
      <td>Fr.</td>
      <td>6-2</td>
      <td>G</td>
      <td>ncaaw.p.113273</td>
      <td>N.C. Central</td>
    </tr>
    <tr>
      <th>710</th>
      <td>Nyla McGill</td>
      <td>Yale (Ivy League)</td>
      <td>Jr.</td>
      <td>5-8</td>
      <td>G</td>
      <td>ncaaw.p.67510</td>
      <td>Yale</td>
    </tr>
    <tr>
      <th>200</th>
      <td>Keshuna Luckett</td>
      <td>Jackson St. (SWAC)</td>
      <td>Sr.</td>
      <td>5-8</td>
      <td>G</td>
      <td>ncaaw.p.61415</td>
      <td>Jackson St.</td>
    </tr>
    <tr>
      <th>577</th>
      <td>Bri McDaniel</td>
      <td>Maryland (Big Ten)</td>
      <td>So.</td>
      <td>5-10</td>
      <td>G</td>
      <td>ncaaw.p.70818</td>
      <td>Maryland</td>
    </tr>
    <tr>
      <th>649</th>
      <td>Halli Poock</td>
      <td>Bradley (MVC)</td>
      <td>Fr.</td>
      <td>5-4</td>
      <td>G</td>
      <td>ncaaw.p.112767</td>
      <td>Murray St.</td>
    </tr>
  </tbody>
</table>
</div>



This looks good, so we'll go ahead and convert the data type for these columns and confirm the change was made. 


```python
player_data[string_columns] = player_data[string_columns].astype('string')
player_data[string_columns].dtypes
```




    PLAYER_NAME    string[python]
    Team           string[python]
    Class          string[python]
    Height         string[python]
    Position       string[python]
    PLAYER_ID      string[python]
    TEAM_NAME      string[python]
    dtype: object



We're finished converting data types, so it's time to move on to value substitution. 

## Value Substitution 
Value substitution (or value mapping) refers to replacing one value with another. One of the most common uses for this is to replace abbreviations with their full values (such as replacing "Fr." with "Freshman"). For machine learning datasets, value substitution can also include mapping categorical values to number or array formats. 

The first value substitution we'll make is swapping the abbreviated position names with the full values. The [Position](https://en.wikipedia.org/wiki/Basketball_positions) field in this dataset refers to where the player typically plays on the court. Here's a diagram of the positions: 

![Diagram of basketball positions](https://upload.wikimedia.org/wikipedia/commons/a/ac/Basketball_Positions.png "Diagram of basketball positions")

Let's see what Position values are currently in our dataset.


```python
player_data['Position'].unique()
```




    <StringArray>
    ['G', 'F', 'C']
    Length: 3, dtype: string



So this data set uses the three types of positions instead of the five specific positions. To make the values more readable, let's create a map between the current single-letter values and the full-length position name. 


```python
position_names = {
    'F': 'Forward',
    'G': 'Guard',
    'C': 'Center'
}
```

Now all we have to do is use the [pandas `replace()` method](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.replace.html) to apply this mapping to the entire series. 


```python
player_data['Position'] = player_data['Position'].replace(position_names)
player_data['Position'].unique()
```




    <StringArray>
    ['Guard', 'Forward', 'Center']
    Length: 3, dtype: string



We can use the same process to substitute the values in the `Class` column as well. 


```python
class_names = {
    'Fr.': 'Freshman',
    'So.': 'Sophomore',
    'Jr.': 'Junior',
    'Sr.': 'Senior'
}
player_data['Class'] = player_data['Class'].replace(class_names)
player_data['Class'].unique()
```




    <StringArray>
    ['Junior', 'Sophomore', 'Senior', 'Freshman']
    Length: 4, dtype: string



There are more substitutions that we could make (such as replacing "St." with "State" in the team name), but those are the only ones we'll make today. Next, let's look at unit conversions.

## Unit Conversion
The only unit conversion we'll be doing today is to convert the feet-inches notation in the `Height` column to the total number of inches. This means that a height of five feet, three inches tall is currently stored as `5-3`, but will be converted to the total number of inches, `63`. We can define a function to convert the individual Height values:


```python
def height_to_inches(height):
    feet, inches = map(int, height.split('-'))
    return feet * 12 + inches
```

This function breaks the values in the `Height` column on the hyphen and store the number of feet and inches in separate variables. It then multiplies the number of feet by 12 (the number of inches per foot) and adds the number of inches to get the total height of the player in inches. 

Now that we have this function, we can apply it to the `Height` column and verify it worked by checking the unique values. 


```python
player_data['Height'] = player_data['Height'].apply(height_to_inches)
player_data['Height'].unique()
```




    array([67, 68, 62, 69, 66, 72, 70, 63, 71, 65, 64, 74, 73, 75, 76, 78, 77,
           79, 80])



That wraps up everything needed for data preprocessing and our dataset is ready for feature engineering and machine learning. 

If you're going to use a new Jupyter notebook / Python script for the next part of this series, then it's a good idea to export this dataset. As a reminder, you can use the `to_csv()` method instead of `.to_excel()` if you prefer. 

```python
player_data.to_excel('player_data_clean.xlsx', index=False)
```

# Wrap up
In today's guide, we laid the groundwork for data analysis by cleaning and preprocessing the combined player data. In the next article, we'll expand upon this dataset by engineering a few new features and training a machine learning model. In the final installment of this series, we'll identify relationships between various parameters and create meaningful visualizations. 

<div id="sources"></div>

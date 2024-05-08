---
layout: post
title: "A Full Guide to Data Science Projects using Basketball"
subtitle: "Part 1 - Project Setup and Data Acquisition"
tags:  [Python, data science, pandas, API]
thumbnail-img: /assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa.jpg
share-title: "A Full Guide to Data Science Projects using Basketball: Part 1 - Project Setup and Data Acquisition" 
share-description: Interested in starting a new data science project? Learn the initial steps to start your project and acquire your datasets in this comprehensive guide that is perfect for beginner data scientists and Python enthusiasts.
share-img: /assets/img/posts/2024-04-11-basketball-data-acquisition/acquisition-social.png
readtime: true
gh-repo: pineconedata/pineconedata.github.io
gh-badge: [star, fork, follow]
after-content: post-subscribe.html
js: /assets/js/table-of-contents.js
published: false
---

In this guide, we'll walk through the process of acquiring, preprocessing, and cleaning a data set. These steps are essential for any data analysis project to make sure the dataset is ready for analysis and visualization. The dataset we'll be using today contains basketball player statistics for the 2023-2024 NCAA women's basketball season. We'll use Python along with the popular pandas and requests libraries to accomplish these tasks efficiently. By the end of this tutorial, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, and eliminate any inconsistencies and errors.

This process will involve four major steps: 
1. Data Acquisition - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. Data Cleaning - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. Data Preprocessing - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. Feature Engineering - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.

Steps 2 and 3 specifically will help mitigate the pitfalls of the "garbage in, garbage out" saying for data analysis - where the quality of your analysis is only as good as the data you start with. 

<div id="toc"></div>

# Getting Started
## Requirements
Before we get started, you should have: 
- A computer with the appropriate access level to install and remove programs.
  - This guide uses a Linux distribution (specifically Ubuntu), but this code can work on any major OS with a few minor changes.
- A reliable internet connection to download the necessary software and make API requests. 
- A text editor or [IDE](https://en.wikipedia.org/wiki/Integrated_development_environmenthttps://en.wikipedia.org/wiki/Integrated_development_environment) to create and edit program files.
- Basic programming knowledge is a plus. If you've never used Python before, then going through the [beginner's guide](https://wiki.python.org/moin/BeginnersGuide) first might be helpful.

## Dependencies
This project depends on having Python, a package manager (such as pip), and the relevant packages (listed below) installed before moving forward.

### Python
This project depends on [Python](https://www.python.org), which is probably already installed on your computer if you're using a common OS. You can verify if Python is installed and which version is currently being used by running:
```bash
$ python3 --version
Python 3.10.12
```
If you get a response that says `command not found` instead of a version number, then Python is not installed. In that case, you can follow [Python's official installation instructions](https://www.python.org/downloads/). You can also try running `python --version` to see if you have Python 2 installed instead of Python 3, but ultimately you should install Python 3 for this project. 

### Package Manager (pip)
You'll also need a package manager to install dependencies. I've used `pip`, but this works with any package manager. You can check if `pip` is installed by running: 
```bash
$ pip --version
pip 24.0 from /home/scoops/.local/lib/python3.10/site-packages/pip (python 3.10)
```
Similar to the command for Python, if you get a response that says `command not found` instead of a version number, then `pip` might not be installed on your device. You should follow the [official instructions](https://pip.pypa.io/en/stable/installation/) to install `pip` (or your package manager of choice) before proceeding. 

### Specific Packages
You should install the following required packages: 
- *pandas* - a powerful data manipulation library, essential for handling structured data. If you're using `pip`, then you can install pandas by running `pip install pandas`. 
- *requests* - a library used for making HTTP requests, which will be useful for fetching data from online sources.
- *json* - a library for encoding and decoding JSON data, facilitating the interchange of data between a Python program and external systems or files.
- *os* - a library providing a portable way of interacting with the operating system, enabling functionalities such as file management, directory manipulation, and environment variables handling.
- *numpy* - a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. 

Once those required packages have been installed, we can start writing the program by importing the packages. 


```python
import pandas as pd
import requests
import json
import os
import numpy as np
```

# Data Acquisition
Now that we've ensured the necessary dependencies are installed, it's time to acquire the data. Before diving into the process of gathering basketball player statistics for the 2023-2024 NCAA women's basketball season, let's briefly look at the two data sets we'll be acquiring and a rough overview of the process to get each one:

1. Player Information Dataset
   - This dataset will include player information such as name, height, position, team name, and class. 
   - To obtain this data, we'll be navigating to the [NCAA website's basketball statistics section](https://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&sportCode=WBB). From there, we selected the desired season, division, and reporting week to access the statistics. After selecting "All Statistics for Individual," we clicked on "Show report CSV" to generate and download the dataset in CSV format.

2. Player Statistics Dataset
   - This dataset will include player statistics and individual results for the latest season, including points scored, field goals made, blocks, steals, assists, etc.
   - To obtain this dataset, we'll be making multiple API requests to the [Yahoo Sports API](https://sports.yahoo.com/ncaaw/stats/individual/?selectedTable=0&leagueStructure=ncaaw.struct.div.1&sortStatId=FREE_THROWS_MADE). Each request will pull the top players for a given statistic (such as points, blocks, assists, etc.) and then those results will be combined into one dataset.

Once we have those two datasets, the next step will be to merge them to have one comprehensive dataset that includes player information (such as height, position, class, etc.) as well as the player's total statistics (such as total points scored, blocks made, steals made, etc.). The final dataset can then be used for analysis and to generate visualizations. 

## Acquiring Player Information Data
As described above, the player information data can be obtained from the [NCAA website's basketball statistics section](https://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&sportCode=WBB). You can write a small web scraper in Python to scrape this data automatically, but today we'll walk through how to manually download the data. Starting on the NCAA's website, the page should look like this: 

![NCAA basketball statistics website](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_info_step1.png "NCAA basketball statistics website")

From this page, we'll select the desired season, which is the most recent 2023-2024 season: 

![select the proper season](/assets/img/posts/2024-04-11-basketball-data-acquisition//ncaa_wbb_player_info_step2.png "select the proper season")

Once you click "view", it should open a new tab with a page that looks like this: 

![select division and statistics](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_info_step3.png "select division and statistics")

We can now select the division (which will be Division 1), the reporting week (which will be the final week of the season), and the statistics (which will be all individual player statistics). With those selections made, we can click "Show Report (CSV - Spreadsheet)" to get the data in CSV format.

![download csv results](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_info_step4.png "download csv results")

With the CSV downloaded, the only step left is to import the data (after removing all blank and duplicate rows). 


```python
player_info = pd.read_csv("2024-04-07 NCAA WBB-Div1 Player Info.csv", encoding='latin1')
player_info.head()
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
      <th>Player</th>
      <th>Team</th>
      <th>Class</th>
      <th>Height</th>
      <th>Position</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kiara Jackson</td>
      <td>UNLV (Mountain West)</td>
      <td>Jr.</td>
      <td>5-7</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Raven Johnson</td>
      <td>South Carolina (SEC)</td>
      <td>So.</td>
      <td>5-8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gina Marxen</td>
      <td>Montana (Big Sky)</td>
      <td>Sr.</td>
      <td>5-8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>McKenna Hofschild</td>
      <td>Colorado St. (Mountain West)</td>
      <td>Sr.</td>
      <td>5-2</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kaylah Ivey</td>
      <td>Boston College (ACC)</td>
      <td>Jr.</td>
      <td>5-8</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
</div>




```python
player_info.shape
```




    (1009, 5)


This dataset contains 1,009 rows and 5 columns. Here's a quick description of each column: 
  - PLAYER_NAME: The name of the basketball player.
  - Team: The name of the basketball team the player is associated with, including the conference they belong to.
  - Class: The classification or academic year of the player (e.g., freshman, sophomore, junior, senior).
  - Height: The height of the player in a standardized format (e.g., feet-inches or centimeters).
  - Position: The playing position of the player on the basketball court (e.g., point guard, shooting guard, small forward, power forward, center).

## Acquiring Player Statistics Data
As mentioned before, the player statistics data can be obtained by making API requests to the [Yahoo Sports API](https://sports.yahoo.com/ncaaw/stats/individual/?selectedTable=0&leagueStructure=ncaaw.struct.div.1&sortStatId=FREE_THROWS_MADE). There are a variety of API endpoints available for sports data, but today we'll be using the Yahoo Sports API since it is free and contains all of the player statistics in one convenient result. 

### Explore the API Endpoint
You can view this data in a table format on Yahoo's website, which can help identify particular columns (such as "G" stands for "Games Played"). 
![Yahoo Sports basketball statistics webpage](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_stats_page.png "Yahoo Sports basketball statistics webpage")

Tools such as [Postman](https://www.postman.com/) can be extremely helpful for fine-tuning the API request and viewing the response data. The "View Code" button allows you to view the API request as Python code directly (in this case, by using the `requests` library specifically) in the right sidebar.

![Postman API request with code sidebar](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_stats_postman.png "Postman API request with code sidebar")

### Write a Function to Request Data
Here's an example of making an API request to the Yahoo Sports data using the Python `requests` library (copied from Postman). 

```python
import requests

url = "https://graphite-secure.sports.yahoo.com/v1/query/shangrila/seasonStatsBasketballTotal?lang=en-US&region=US&tz=America/New_York&ysp_redesign=1&ysp_platform=desktop&season=2023&league=ncaaw&leagueStructure=ncaaw.struct.div.1&count=200&sortStatId=FREE_THROWS_MADE&positionIds=&qualified=TRUE"

payload = {}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
```

Depending on your use case, you might not need to make any modifications to this code. However, for this project, we'll want to request data for multiple statistics, so we will make a few changes to improve usability. First, we'll separate all of the query string parameters into a dictionary to be passed into the `requests.get()` function. Then, we'll put the request into a convenience function with basic error handling. 


```python
def get_data_for_stat(stat_name, season='2023', league='ncaaw', count='500'):
    """
    Retrieve basketball player statistics for a specified statistical category (stat_name) for a given season and league.

    Parameters:
    - stat_name (str): Specifies the statistical category for which data is requested.
    - season (str): Specifies the season for which data is requested (default is '2023').
    - league (str): Specifies the league for which data is requested (default is 'ncaaw' for NCAA Women's Basketball).
    - count (str): Specifies the maximum number of data entries to retrieve (default is '500').

    Returns:
    - dict: JSON response containing player statistics for the specified statistical category.
    """
    url = "https://graphite-secure.sports.yahoo.com/v1/query/shangrila/seasonStatsBasketballTotal"
    params = {
        'lang': 'en-US',
        'region': 'US',
        'tz': 'America/New_York',
        'ysp_redesign': '1',
        'ysp_platform': 'desktop',
        'season': season,
        'league': league,
        'leagueStructure': f'{league}.struct.div.1',
        'count': count,
        'sortStatId': stat_name,
        'positionIds': '',
        'qualified': 'FALSE'
    }
    try:
        # Send GET request to the API endpoint with specified parameters
        response = requests.get(url, params)
        response.raise_for_status()  # Raise exception for 4xx and 5xx status codes
        return response.json()
    except requests.exceptions.RequestException as e:
        # Handle any errors encountered during the API request
        print(f'Error requesting data for {stat_name}: {e}')
        return None
```

This `get_data_for_stat()` function retrieves basketball player statistics for the top players in a specified statistical category (`stat_name`) for a given season (`2023`) and league (`ncaaw`). At a high level, here's an overview of how it works: 

1. Function Arguments
   - stat_name - specifies the statistic to pull the top players (points scored, field goals made, etc.)
   - season - specifies the starting year of the desired season (set to '2023' by default to get data for the 2023-2024 season)
   - league - specifies the league (set to 'ncaaw' for NCAA Women's Basketball by default)
   - count - specifies the maximum number of players to retrieve (set to '500' by default)

2. API Endpoint and Parameters
   - The function begins by setting the base URL for the Yahoo Sports API endpoint for basketball season statistics in the `url` variable.
   - Next, the function uses the `params` variable to store various query string parameters for the API request (such as language, region, time zone, etc.)

3. Making the API Request
   - The function uses the `requests.get()` method to send a GET request to the API endpoint with the specified parameters.
   - It checks the response status code for errors using `response.raise_for_status()`. If the response indicates an error (4xx or 5xx status code), it raises an exception.
   - If the request is successful, the function returns the JSON data obtained from the API response.

4. Error Handling
   - The function includes error handling using a try-except block to catch any exceptions that may occur during the API request process.
   - If an exception occurs (e.g., network issues, invalid response), an error message is printed, and the function returns `None`.

You could further modify this code to fit your needs and add additional parameters to the function arguments (such as language or region), but we'll use this function to retrieve the top players for a given statistical category. Here's an example of retrieving the top 500 players (the default `count` is set to `500` in the function) by total points scored during the season: 


```python
example_response = get_data_for_stat('POINTS')
```

Since the function returns data in `json` format, we can take a look at the structure and identify a few values of interest.
- `example_response['data']['statTypes']` stores the syntax and sort order for all of the available statistics
- `example_response['data']['leagues'][0]['leaders']` stores the result set for our request (the 'leaders' or top players for the given request)

Let's print the available statistic types as well as the first result for the top player by points scored. 


```python
example_response.keys()
example_response['data']['statTypes']
example_response['data']['leagues'][0]['leaders'][0]
```



```
    {'player': {'displayName': 'Caitlin Clark',
      'playerId': 'ncaaw.p.64550',
      'team': {'displayName': 'Iowa',
       'abbreviation': 'IOWA',
       'teamLogo': {'url': 'https://s.yimg.com/iu/api/res/1.2/c1eT0fjpIOp9jIlg5xiq0w--~B/YXBwaWQ9c2hhcmVkO2ZpPWZpbGw7cT0xMDA7aD0xMjg7dz0xMjg-/https://s.yimg.com/cv/apiv2/default/ncaab/20181211/500x500/Iowa.png',
        'height': 128,
        'width': 128}},
      'alias': {'url': 'https://sports.yahoo.com/ncaaw/players/64550/'},
      'playerCutout': None},
     'stats': [{'statId': 'GAMES', 'value': '32'},
      {'statId': 'MINUTES_PLAYED', 'value': '1088'},
      {'statId': 'FIELD_GOALS_MADE', 'value': '332'},
      {'statId': 'FIELD_GOAL_ATTEMPTS', 'value': '719'},
      {'statId': 'FIELD_GOAL_PERCENTAGE', 'value': '46.2'},
      {'statId': 'THREE_POINTS_MADE', 'value': '168'},
      {'statId': 'THREE_POINT_ATTEMPTS', 'value': '437'},
      {'statId': 'THREE_POINT_PERCENTAGE', 'value': '38.4'},
      {'statId': 'FREE_THROWS_MADE', 'value': '188'},
      {'statId': 'FREE_THROW_ATTEMPTS', 'value': '219'},
      {'statId': 'FREE_THROW_PERCENTAGE', 'value': '85.8'},
      {'statId': 'OFFENSIVE_REBOUNDS', 'value': '10'},
      {'statId': 'DEFENSIVE_REBOUNDS', 'value': '224'},
      {'statId': 'TOTAL_REBOUNDS', 'value': '234'},
      {'statId': 'ASSISTS', 'value': '282'},
      {'statId': 'TURNOVERS', 'value': '151'},
      {'statId': 'STEALS', 'value': '55'},
      {'statId': 'BLOCKS', 'value': '17'},
      {'statId': 'FOULS', 'value': '61'},
      {'statId': 'POINTS', 'value': '1020'}]}
```


### Write a Function to Format Data
Now that the API request function is working, let's define another convenience function. This one will extract the relevant data and load it into a dataframe. 


```python
def format_response_data(response_data):
    """
    Process and format the JSON response data obtained from the Yahoo Sports API into a pandas DataFrame.

    Parameters:
    - response_data (dict): JSON response data obtained from the Yahoo Sports API, containing player statistics.

    Returns:
    - DataFrame: Pandas DataFrame containing formatted player statistics.
    """
    if not response_data:
        return None
    try:
        # Extract relevant data from the JSON response
        response_data = response_data['data']['leagues'][0]['leaders']
        data = []
        for item in response_data:
            # Extract player details
            player_details = {
                'PLAYER_NAME': item['player']['displayName'],
                'PLAYER_ID': item['player']['playerId'],
                'TEAM_NAME': item['player']['team']['displayName']
            }
            # Extract player statistics
            player_stats = {stat['statId']: stat['value'] for stat in item['stats']}
            # Combine player details and statistics into a single dictionary
            player_row = {**player_details, **player_stats}
            data.append(player_row)
        # Convert the list of dictionaries into a pandas DataFrame
        return pd.DataFrame(data)
    except KeyError as e:
        # Handle any errors encountered during data formatting
        print(f'Error formatting response data: {e}')
        return None
```

This `format_response_data()` function extracts the player statistics from the Yahoo Sports API response data (`response_data`) and formats it into a pandas DataFrame. Here's a summary of how it works: 

1. Function Arguments
   - response_data - the JSON object containing player statistics from the Yahoo Sports API.
2. Response Data Formatting
   - The function begins by checking if the response_data is empty. If it is, the function returns None.
   - It then attempts to access specific nested fields within the JSON data to extract relevant information.
   - Inside the try block, the function extracts the player details (name, ID, team name) and player statistics from the JSON data.
   - It iterates through each player in the response data, creating a dictionary (player_row) containing player details and statistics.
   - The player details include 'PLAYER_NAME', 'PLAYER_ID', and 'TEAM_NAME', while the player statistics are dynamically extracted from the JSON data.
   - These player details and statistics are appended to a list (data).
   - Finally, the function constructs a pandas DataFrame from the list of dictionaries and returns it.
3. Error Handling
   - The function includes error handling using a try-except block to catch any exceptions that may occur during the data formatting process.
   - If an exception occurs (e.g., missing or unexpected data structure), an error message is printed, and the function returns None.

By using this function in conjunction with the `get_data_for_stat()` function, we can efficiently retrieve, process, and format player statistics from the Yahoo Sports API into a structured DataFrame. 

We can test out this function using the example_response from the previous step. 


```python
example_dataframe = format_response_data(example_response)
example_dataframe.head()
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
      <th>PLAYER_ID</th>
      <th>TEAM_NAME</th>
      <th>GAMES</th>
      <th>MINUTES_PLAYED</th>
      <th>FIELD_GOALS_MADE</th>
      <th>FIELD_GOAL_ATTEMPTS</th>
      <th>FIELD_GOAL_PERCENTAGE</th>
      <th>THREE_POINTS_MADE</th>
      <th>THREE_POINT_ATTEMPTS</th>
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
      <th>0</th>
      <td>Caitlin Clark</td>
      <td>ncaaw.p.64550</td>
      <td>Iowa</td>
      <td>32</td>
      <td>1088</td>
      <td>332</td>
      <td>719</td>
      <td>46.2</td>
      <td>168</td>
      <td>437</td>
      <td>...</td>
      <td>85.8</td>
      <td>10</td>
      <td>224</td>
      <td>234</td>
      <td>282</td>
      <td>151</td>
      <td>55</td>
      <td>17</td>
      <td>61</td>
      <td>1020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JuJu Watkins</td>
      <td>ncaaw.p.112021</td>
      <td>USC</td>
      <td>29</td>
      <td>989</td>
      <td>270</td>
      <td>656</td>
      <td>41.2</td>
      <td>58</td>
      <td>176</td>
      <td>...</td>
      <td>84.6</td>
      <td>52</td>
      <td>161</td>
      <td>213</td>
      <td>96</td>
      <td>120</td>
      <td>72</td>
      <td>45</td>
      <td>78</td>
      <td>801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hannah Hidalgo</td>
      <td>ncaaw.p.112250</td>
      <td>Notre Dame</td>
      <td>31</td>
      <td>1104</td>
      <td>255</td>
      <td>560</td>
      <td>45.5</td>
      <td>45</td>
      <td>132</td>
      <td>...</td>
      <td>78.3</td>
      <td>25</td>
      <td>175</td>
      <td>200</td>
      <td>170</td>
      <td>109</td>
      <td>145</td>
      <td>3</td>
      <td>86</td>
      <td>725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lucy Olsen</td>
      <td>ncaaw.p.67706</td>
      <td>Iowa</td>
      <td>30</td>
      <td>1087</td>
      <td>268</td>
      <td>612</td>
      <td>43.8</td>
      <td>47</td>
      <td>158</td>
      <td>...</td>
      <td>80.9</td>
      <td>30</td>
      <td>114</td>
      <td>144</td>
      <td>115</td>
      <td>73</td>
      <td>57</td>
      <td>18</td>
      <td>72</td>
      <td>697</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ta'Niya Latson</td>
      <td>ncaaw.p.70600</td>
      <td>Florida St.</td>
      <td>32</td>
      <td>991</td>
      <td>249</td>
      <td>566</td>
      <td>44.0</td>
      <td>27</td>
      <td>98</td>
      <td>...</td>
      <td>85.2</td>
      <td>17</td>
      <td>118</td>
      <td>135</td>
      <td>128</td>
      <td>98</td>
      <td>50</td>
      <td>13</td>
      <td>53</td>
      <td>680</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



We can see all of the columns in this dataframe as well:


```python
example_dataframe.columns
```




    Index(['PLAYER_NAME', 'PLAYER_ID', 'TEAM_NAME', 'GAMES', 'MINUTES_PLAYED',
           'FIELD_GOALS_MADE', 'FIELD_GOAL_ATTEMPTS', 'FIELD_GOAL_PERCENTAGE',
           'THREE_POINTS_MADE', 'THREE_POINT_ATTEMPTS', 'THREE_POINT_PERCENTAGE',
           'FREE_THROWS_MADE', 'FREE_THROW_ATTEMPTS', 'FREE_THROW_PERCENTAGE',
           'OFFENSIVE_REBOUNDS', 'DEFENSIVE_REBOUNDS', 'TOTAL_REBOUNDS', 'ASSISTS',
           'TURNOVERS', 'STEALS', 'BLOCKS', 'FOULS', 'POINTS'],
          dtype='object')



Let's quickly verify that the dataframe has the proper number of rows using the `shape()` method. Since we set the `count` to `500` in the API request, there should be
500 rows in this dataframe. 


```python
example_dataframe.shape
```




    (500, 23)



### Write a Function to Format and Request Data
We could stop here and use those two functions to make our API requests, but let's define one more convenient function that will request and format the data in one step.


```python
def get_and_format_data_for_stat(stat_name, season='2023', league='ncaaw'):
    """
    Retrieve basketball player statistics for a specified statistical category (stat_name) for a given season and league
    and format the data into a pandas DataFrame.

    Parameters:
    - stat_name (str): Specifies the statistical category for which data is requested.
    - season (str): Specifies the season for which data is requested (default is '2023').
    - league (str): Specifies the league for which data is requested (default is 'ncaaw' for NCAA Women's Basketball).

    Returns:
    - DataFrame: Pandas DataFrame containing formatted player statistics for the specified statistical category.
    """
    # Retrieve player statistics for the specified statistical category, season, and league
    response_data = get_data_for_stat(stat_name, season, league)
    # Format the retrieved data into a pandas DataFrame
    return format_response_data(response_data)
```

This function combines the other two function calls and passes through the same parameters. In summary, it retrieves player statistics for a specified statistical category (stat_name) for a given season and league using the get_data_for_stat function and then formats the data into a pandas DataFrame using the format_response_data function. Since we'll be sending multiple requests to the API, this function streamlines the entire API request into one function and simplifies the process of generating the player statistics dataset. 

### Request the Data for each Statistic
Now we're ready to request data for each statistic. For this project, we'll pull the list of top players by five statistics: points, assists, rebounds, blocks, and steals. Here's what that looks like: 


```python
# Get and format data for each stat
points_top_players = get_and_format_data_for_stat('POINTS')
assists_top_players = get_and_format_data_for_stat('ASSISTS')
rebounds_top_players = get_and_format_data_for_stat('TOTAL_REBOUNDS')
blocks_top_players = get_and_format_data_for_stat('BLOCKS')
steals_top_players = get_and_format_data_for_stat('STEALS')
```

### Combine each statistic into one dataset
Since these dataframes contain the same columns with different data rows, we'll combine all five into one single dataframe. The same player could show up in each dataframe (such as Caitlin Clark being a top player in both points and assists), so we'll drop any duplicate rows and reset the index in this step as well. 


```python
# Combine the leaders for each stat into one df
player_stats = pd.concat([points_top_players, assists_top_players, rebounds_top_players,
                         blocks_top_players, steals_top_players], ignore_index=True).drop_duplicates()
player_stats.head()
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
      <th>PLAYER_ID</th>
      <th>TEAM_NAME</th>
      <th>GAMES</th>
      <th>MINUTES_PLAYED</th>
      <th>FIELD_GOALS_MADE</th>
      <th>FIELD_GOAL_ATTEMPTS</th>
      <th>FIELD_GOAL_PERCENTAGE</th>
      <th>THREE_POINTS_MADE</th>
      <th>THREE_POINT_ATTEMPTS</th>
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
      <th>0</th>
      <td>Caitlin Clark</td>
      <td>ncaaw.p.64550</td>
      <td>Iowa</td>
      <td>32</td>
      <td>1088</td>
      <td>332</td>
      <td>719</td>
      <td>46.2</td>
      <td>168</td>
      <td>437</td>
      <td>...</td>
      <td>85.8</td>
      <td>10</td>
      <td>224</td>
      <td>234</td>
      <td>282</td>
      <td>151</td>
      <td>55</td>
      <td>17</td>
      <td>61</td>
      <td>1020</td>
    </tr>
    <tr>
      <th>1</th>
      <td>JuJu Watkins</td>
      <td>ncaaw.p.112021</td>
      <td>USC</td>
      <td>29</td>
      <td>989</td>
      <td>270</td>
      <td>656</td>
      <td>41.2</td>
      <td>58</td>
      <td>176</td>
      <td>...</td>
      <td>84.6</td>
      <td>52</td>
      <td>161</td>
      <td>213</td>
      <td>96</td>
      <td>120</td>
      <td>72</td>
      <td>45</td>
      <td>78</td>
      <td>801</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Hannah Hidalgo</td>
      <td>ncaaw.p.112250</td>
      <td>Notre Dame</td>
      <td>31</td>
      <td>1104</td>
      <td>255</td>
      <td>560</td>
      <td>45.5</td>
      <td>45</td>
      <td>132</td>
      <td>...</td>
      <td>78.3</td>
      <td>25</td>
      <td>175</td>
      <td>200</td>
      <td>170</td>
      <td>109</td>
      <td>145</td>
      <td>3</td>
      <td>86</td>
      <td>725</td>
    </tr>
    <tr>
      <th>3</th>
      <td>Lucy Olsen</td>
      <td>ncaaw.p.67706</td>
      <td>Iowa</td>
      <td>30</td>
      <td>1087</td>
      <td>268</td>
      <td>612</td>
      <td>43.8</td>
      <td>47</td>
      <td>158</td>
      <td>...</td>
      <td>80.9</td>
      <td>30</td>
      <td>114</td>
      <td>144</td>
      <td>115</td>
      <td>73</td>
      <td>57</td>
      <td>18</td>
      <td>72</td>
      <td>697</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Ta'Niya Latson</td>
      <td>ncaaw.p.70600</td>
      <td>Florida St.</td>
      <td>32</td>
      <td>991</td>
      <td>249</td>
      <td>566</td>
      <td>44.0</td>
      <td>27</td>
      <td>98</td>
      <td>...</td>
      <td>85.2</td>
      <td>17</td>
      <td>118</td>
      <td>135</td>
      <td>128</td>
      <td>98</td>
      <td>50</td>
      <td>13</td>
      <td>53</td>
      <td>680</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 23 columns</p>
</div>



We can see that this combined dataframe has the same number of columns as the individual dataframes, but we ended up with a dataset of 1392 unique players.


```python
player_stats.shape
```




    (1392, 23)



This dataset contains 1,392 rows and 23 columns (approximately 300 more rows than the previous dataset and nearly 20 additional columns). Here's a short description of each column: 
   - PLAYER_NAME: The name of the basketball player.
   - PLAYER_ID: The unique identifier for each basketball player.
   - TEAM_NAME: The name of the basketball team the player is associated with.
   - GAMES: The number of games the player has played in. This does not include attending a game but not playing, such as when a player is injured.
   - MINUTES_PLAYED: The total number of minutes the player has been on the court. This is measured in the time on the game clock, not in real-time. There are typically 40 minutes of game-time, comprised of four 10-minute quarters. Overtime would be considered extra time. 
   - FIELD_GOALS_MADE: The number of successful field goals made by the player. This includes both two-point and three-point field goals and does not include free throws. 
   - FIELD_GOAL_ATTEMPTS: The total number of field goal attempts by the player.
   - FIELD_GOAL_PERCENTAGE: The percentage of successful field goals made by the player.
   - THREE_POINTS_MADE: The number of successful three-point baskets made by the player.
   - THREE_POINT_ATTEMPTS: The total number of three-point basket attempts by the player.
   - THREE_POINT_PERCENTAGE: The percentage of successful three-point baskets made by the player.
   - FREE_THROWS_MADE: The number of successful free throws made by the player.
   - FREE_THROW_ATTEMPTS: The total number of free throw attempts by the player.
   - FREE_THROW_PERCENTAGE: The percentage of successful free throws made by the player.
   - OFFENSIVE_REBOUNDS: The number of offensive rebounds grabbed by the player.
   - DEFENSIVE_REBOUNDS: The number of defensive rebounds grabbed by the player.
   - TOTAL_REBOUNDS: The total number of rebounds grabbed by the player.
   - ASSISTS: The number of assists made by the player.
   - TURNOVERS: The number of turnovers committed by the player.
   - STEALS: The number of steals made by the player.
   - BLOCKS: The number of baskets blocked by the player.
   - FOULS: The number of fouls committed by the player.
   - POINTS: The total number of points scored by the player.


```python
player_stats.columns
```




    Index(['PLAYER_NAME', 'PLAYER_ID', 'TEAM_NAME', 'GAMES', 'MINUTES_PLAYED',
           'FIELD_GOALS_MADE', 'FIELD_GOAL_ATTEMPTS', 'FIELD_GOAL_PERCENTAGE',
           'THREE_POINTS_MADE', 'THREE_POINT_ATTEMPTS', 'THREE_POINT_PERCENTAGE',
           'FREE_THROWS_MADE', 'FREE_THROW_ATTEMPTS', 'FREE_THROW_PERCENTAGE',
           'OFFENSIVE_REBOUNDS', 'DEFENSIVE_REBOUNDS', 'TOTAL_REBOUNDS', 'ASSISTS',
           'TURNOVERS', 'STEALS', 'BLOCKS', 'FOULS', 'POINTS'],
          dtype='object')



With that, we've finished acquiring both of the datasets needed for this project and can combine them into one combined dataset that includes both the player information and player statistics.

## Combine Player Information and Statistics Datasets
Combining datasets can be easy or difficult, depending on what columns are shared by the two datasets. In this case, the `Player` column in the `player_info` dataframe closely matches the `PLAYER_NAME` column in the `player_stats` dataframe, so we'll start by renaming one of those columns to match the other.


```python
player_info.rename(columns={"Player": "PLAYER_NAME"}, inplace=True)
player_info.head()
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
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>Kiara Jackson</td>
      <td>UNLV (Mountain West)</td>
      <td>Jr.</td>
      <td>5-7</td>
      <td>G</td>
    </tr>
    <tr>
      <th>1</th>
      <td>Raven Johnson</td>
      <td>South Carolina (SEC)</td>
      <td>So.</td>
      <td>5-8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>2</th>
      <td>Gina Marxen</td>
      <td>Montana (Big Sky)</td>
      <td>Sr.</td>
      <td>5-8</td>
      <td>G</td>
    </tr>
    <tr>
      <th>3</th>
      <td>McKenna Hofschild</td>
      <td>Colorado St. (Mountain West)</td>
      <td>Sr.</td>
      <td>5-2</td>
      <td>G</td>
    </tr>
    <tr>
      <th>4</th>
      <td>Kaylah Ivey</td>
      <td>Boston College (ACC)</td>
      <td>Jr.</td>
      <td>5-8</td>
      <td>G</td>
    </tr>
  </tbody>
</table>
</div>



Next, we'll want to use the pandas `merge()` method to merge the two dataframes on the player's name. This final `player_data` dataframe includes columns for the player's information (height, position, class, etc.) as well as their statistics for the season (points, rebounds, blocks, assists, etc.)


```python
player_data = pd.merge(player_info, player_stats, on=['PLAYER_NAME'], how='inner')
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
      <th>0</th>
      <td>Kiara Jackson</td>
      <td>UNLV (Mountain West)</td>
      <td>Jr.</td>
      <td>5-7</td>
      <td>G</td>
      <td>ncaaw.p.67149</td>
      <td>UNLV</td>
      <td>29</td>
      <td>895</td>
      <td>128</td>
      <td>...</td>
      <td>75.0</td>
      <td>27</td>
      <td>102</td>
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
      <td>Raven Johnson</td>
      <td>South Carolina (SEC)</td>
      <td>So.</td>
      <td>5-8</td>
      <td>G</td>
      <td>ncaaw.p.67515</td>
      <td>South Carolina</td>
      <td>30</td>
      <td>823</td>
      <td>98</td>
      <td>...</td>
      <td>64.3</td>
      <td>33</td>
      <td>128</td>
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
      <td>Gina Marxen</td>
      <td>Montana (Big Sky)</td>
      <td>Sr.</td>
      <td>5-8</td>
      <td>G</td>
      <td>ncaaw.p.57909</td>
      <td>Montana</td>
      <td>29</td>
      <td>778</td>
      <td>88</td>
      <td>...</td>
      <td>72.4</td>
      <td>6</td>
      <td>54</td>
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
      <td>McKenna Hofschild</td>
      <td>Colorado St. (Mountain West)</td>
      <td>Sr.</td>
      <td>5-2</td>
      <td>G</td>
      <td>ncaaw.p.60402</td>
      <td>Colorado St.</td>
      <td>29</td>
      <td>1046</td>
      <td>231</td>
      <td>...</td>
      <td>83.5</td>
      <td>6</td>
      <td>109</td>
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
      <td>Kaylah Ivey</td>
      <td>Boston College (ACC)</td>
      <td>Jr.</td>
      <td>5-8</td>
      <td>G</td>
      <td>ncaaw.p.64531</td>
      <td>Boston Coll.</td>
      <td>33</td>
      <td>995</td>
      <td>47</td>
      <td>...</td>
      <td>60.7</td>
      <td>12</td>
      <td>45</td>
      <td>57</td>
      <td>186</td>
      <td>64</td>
      <td>36</td>
      <td>1</td>
      <td>48</td>
      <td>143</td>
    </tr>
  </tbody>
</table>
<p>5 rows × 27 columns</p>
</div>



We've done an inner join to keep rows that exist in both columns. For some projects, you might want to use an outer join or even a right or left join, but an inner join works well for this project. If you're familiar with SQL, the `how` parameter is similar to the different types of SQL JOINs, but you can read about (and see examples of) the different `how` parameters in the [pandas `merge` documentation](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.merge.html). In this case, using an inner join means that we have fewer rows in the final dataset than we had in either of the other two dataframes. 


```python
player_data.shape
```




    (900, 27)



This is enough data for our purposes today, so we're ready to move on to the cleaning and preprocessing steps.

# Wrap up 
In this guide, we built a new dataset by acquiring and then combining the NCAA women's basketball player information dataset with the Yahoo Sports player statistics dataset. In the next part, we'll lay the groundwork for data analysis by cleaning and preprocessing the combined player data, and then expanding upon it by engineering a few new features.

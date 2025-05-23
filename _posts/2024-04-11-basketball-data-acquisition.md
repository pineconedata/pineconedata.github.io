---
layout: post
title: "Project Setup and Data Acquisition"
subtitle: "Outlier or Caitlin Clark? [Part 1]"
tags:  [Python, data science, pandas, API]
thumbnail-img: /assets/img/posts/2024-04-11-basketball-data-acquisition/basketball.jpg
share-title: "Project Setup and Data Acquisition: Outlier or Caitlin Clark? [Part 1]" 
share-description: Interested in starting a new data science project? Discover how to acquire multiple datasets using APIs and online sources, then seamlessly merge those dataframes into one unified dataset using Python's pandas and requests libraries.
share-img: /assets/img/posts/2024-04-11-basketball-data-acquisition/acquisition-social.png
gh-repo: pineconedata/ncaa-basketball-stats
gh-badge: [star, fork, follow]
---

Today we'll go over how to start a new data science project by acquiring the data (using APIs and a CSV export). This is the first part of a series that will walk through the entire process of a data science project - from initial steps like data acquisition, preprocessing, and cleaning to more advanced steps like feature engineering, creating visualizations, and machine learning. 

<div id="toc"></div>

# Getting Started
First, let's take a look at an overview of this data science project. If you're already familiar with it, feel free to skip to the [next section](#identifying-datasets).

## Project Overview
The dataset we'll be using in this project contains individual basketball player statistics (such as total points scored and blocks made) for the 2023-2024 NCAA women's basketball season. Here's a brief description of each major step that we'll go through for this project: 

![the steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/project_steps.png "the steps for this data science project")

1. **Data Acquisition** - This initial step involves obtaining data from two sources: (1) exporting the NCAA's online individual player statistics report and (2) making API requests to the Yahoo Sports endpoint. 
2. **Data Cleaning** - This step focuses on identifying and correcting any errors within the dataset. This includes removing duplicates, correcting inaccuracies, and handling missing data. 
3. **Data Preprocessing** - This step ensures the data is suitable for analysis by converting datatypes, standardizing units, and replacing abbreviations.
4. **Feature Engineering** - This step involves selecting and expanding upon the dataset's features (or columns). This includes calculating additional metrics from existing columns.
5. **Data Exploration** - This step focuses on analyzing and visualizing the dataset to uncover patterns, relationships, and general trends and is a helpful preliminary step before deeper analysis.
6. **Creating Visualizations** - This step involves identifying the relationships between various parameters (such as height and blocked shots) and generating meaningful visualizations (such as bar charts, scatterplots, and candlestick charts).
7. **Machine Learning** - This step focuses on selecting, training, and evaluating a machine learning model. For this project, the model will identify the combination of individual player statistics that correlates with optimal performance. 

We'll use Python along with popular libraries like [pandas](https://pandas.pydata.org/docs/), [numpy](https://numpy.org/doc/), and [scikit-learn](https://scikit-learn.org/) to accomplish these tasks efficiently. By the end of this series, you'll be equipped with the skills needed to gather raw data from online sources, structure it into a usable format, eliminate any inconsistencies and errors, identify relationships between variables, create meaningful visualizations, and train a basic machine learning model. Due to the size of this project, we'll start today with just the first step: data acqusition. 

## Requirements
Next, let's cover what you'll need if you want to follow along with this guide. If you already have a Python environment up and running and are familiar with how to install packages, then feel free to skip to the next section. 

Before starting, you should have: 
- A computer with the appropriate access level to install and remove programs.
  - This guide uses a Linux distribution (specifically Ubuntu), but this code can work on any major OS with a few minor changes.
- A reliable internet connection to download the necessary software and make API requests. 
- A text editor or [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) to create and edit program files.
- Basic programming knowledge is a plus. If you've never used Python before, then going through the [beginner's guide](https://wiki.python.org/moin/BeginnersGuide) first might be helpful.

## Dependencies
This project depends on having Python, a package manager (such as `pip`), and the relevant packages (listed below) installed before moving forward.

### Python
This project depends on [Python](https://www.python.org), which is probably already installed on your computer if you're using a common OS. You can verify if Python is installed and which version is currently being used by running:
```bash
$ python3 --version
Python 3.10.12
```
If you get a response that says `command not found` instead of a version number, then Python is not installed. In that case, you can follow [Python's official installation instructions](https://www.python.org/downloads/). You can also try running `python --version` to see if you have Python 2 installed instead of Python 3, but ultimately you should install Python 3 for this project. 

### Package Manager (pip)
You'll also need a package manager to install dependencies. I've used [`pip`](https://pypi.org/project/pip/), but this works with any package manager. You can check if `pip` is installed by running: 
```bash
$ pip --version
pip 24.0 from /home/scoops/.local/lib/python3.10/site-packages/pip (python 3.10)
```
Similar to the command for Python, if you get a response that says `command not found` instead of a version number, then `pip` might not be installed on your device. You should follow the [official instructions](https://pip.pypa.io/en/stable/installation/) to install `pip` (or your package manager of choice) before proceeding. 

### Specific Packages
You should install the following required packages: 
- [*pandas*](https://pandas.pydata.org/docs/) - a powerful data manipulation library, essential for handling structured data. If you're using `pip`, then you can install pandas by running `pip install pandas`. 
- [*requests*](https://requests.readthedocs.io/en/latest/) - a library used for making HTTP requests, which will be useful for fetching data from online sources.
- [*json*](https://docs.python.org/3/library/json.html) - a module for encoding and decoding JSON data, facilitating the interchange of data between a Python program and external systems or files.
- [*os*](https://docs.python.org/3/library/os.html) - a module providing a portable way of interacting with the operating system, enabling functionalities such as file management, directory manipulation, and environment variables handling.
- [*numpy*](https://numpy.org/doc/) - a fundamental package for scientific computing in Python, providing support for large, multi-dimensional arrays and matrices, along with a collection of mathematical functions to operate on these arrays efficiently. 
- [*openpyxl*](https://openpyxl.readthedocs.io/en/stable/) - a useful library for reading from, writing to, and modifying Excel files. This library is optional for today's project - you can write the dataframes to `.csv` files  (using [`.to_csv()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html)) instead of `.xlsx`  (using [`.to_excel()`](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html)) files to skip this requirement. 

Once those required packages have been installed, we can start writing the program by importing the packages. 


```python
import pandas as pd
import requests
```

Now that we've ensured the necessary dependencies are installed, it's time to acquire the data. 

# Identifying Datasets
Before diving into the process of gathering basketball player statistics for the 2023-2024 NCAA women's basketball season, let's briefly look at the two data sets we'll be acquiring and a rough overview of the process to get each one:

1. Player Information Dataset
   - This dataset will include player information such as name, height, position, team name, and class. 
   - To obtain this data, we'll be navigating to the [NCAA website's basketball statistics section](https://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&sportCode=WBB). From there, we selected the desired season, division, and reporting week to access the statistics. After selecting "All Statistics for Individual," we clicked on "Show report CSV" to generate and download the dataset in CSV format.

2. Player Statistics Dataset
   - This dataset will include player statistics and individual results for the latest season, including points scored, field goals made, blocks, steals, assists, etc.
   - To obtain this dataset, we'll be making multiple API requests to the [Yahoo Sports API](https://sports.yahoo.com/ncaaw/stats/individual/?selectedTable=0&leagueStructure=ncaaw.struct.div.1&sortStatId=FREE_THROWS_MADE). Each request will pull the top players for a given statistic (such as points, blocks, assists, etc.) and then those results will be combined into one dataset.

Once we have those two datasets, the next step will be to merge them to have one comprehensive dataset that includes player information (such as height, position, class, etc.) as well as the player's total statistics (such as total points scored, blocks made, steals made, etc.). The final dataset can then be used for analysis and to generate visualizations. 

# Acquiring Player Information Data
As described above, the player information data can be obtained from the [NCAA website's basketball statistics section](https://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&sportCode=WBB). You can write a small web scraper in Python to scrape this data automatically, but today we'll walk through how to manually download the data. Starting on the NCAA's website, the page should look like this: 

![NCAA basketball statistics website](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_info_step1.png "NCAA basketball statistics website")

From this page, we'll select the desired season, which is the most recent 2023-2024 season: 

![select the proper season](/assets/img/posts/2024-04-11-basketball-data-acquisition//ncaa_wbb_player_info_step2.png "select the proper season")

Once you click "view", it should open a new tab with a page that looks like this: 

![select division and statistics](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_info_step3.png "select division and statistics")

We can now select the division (which will be Division 1), the reporting week (which will be the final week of the season), and the statistics (which will be all individual player statistics). With those selections made, we can click "Show Report (CSV - Spreadsheet)" to get the data in CSV format.

![download csv results](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_info_step4.png "download csv results")

With the CSV downloaded, the only step left is to import the data (after deleting all blank and duplicate rows). 


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

We can see how many rows and columns are in this dataset using the [`.shape()` method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.shape.html). 


```python
player_info.shape
```

    (1009, 5)


This dataset contains 1,009 rows and 5 columns. Here's a quick description of each column: 
  - *Player* - The name of the basketball player.
  - *Team* - The name of the basketball team the player is associated with, including the conference they belong to.
  - *Class* - The academic year of the player (freshman, sophomore, junior, senior).
  - *Height* - The height of the player in feet-inches.
  - *Position* - The playing position of the player on the basketball court (guard, forward, center). 
  
Now that we're done with this dataset, let's save a copy of it. As a reminder, you can use the `to_csv()` method instead of `.to_excel()` if you prefer. 

```python
player_info.to_excel('player_info.xlsx', index=False)
```

# Acquiring Player Statistics Data
As mentioned before, the player statistics data can be obtained by making API requests to the [Yahoo Sports API](https://sports.yahoo.com/ncaaw/stats/individual/?selectedTable=0&leagueStructure=ncaaw.struct.div.1&sortStatId=FREE_THROWS_MADE). There are a variety of API endpoints available for sports data, but today we'll be using the Yahoo Sports API since it is free and contains all of the player statistics in one convenient result. 

## Explore the API Endpoint
You can view this data in a table format on Yahoo's website, which can help identify particular columns (such as "G" stands for "Games Played"). 
![Yahoo Sports basketball statistics webpage](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_stats_page.png "Yahoo Sports basketball statistics webpage")

Tools such as [Postman](https://www.postman.com/) can be extremely helpful for fine-tuning the API request and viewing the response data. The "View Code" button allows you to view the API request as Python code directly (in this case, by using the `requests` library specifically) in the right sidebar.

![Postman API request with code sidebar](/assets/img/posts/2024-04-11-basketball-data-acquisition/ncaa_wbb_player_stats_postman.png "Postman API request with code sidebar")

## Write a Function to Request Data
We can use the Python code copied from Postman as a starting point. At a high level, this code will make a GET request to the Yahoo Sports API endpoint with the parameters specified in the query string, using the given payload and headers. The response data is then printed. 

```python
import requests

url = "https://graphite-secure.sports.yahoo.com/v1/query/shangrila/seasonStatsBasketballTotal?lang=en-US&region=US&tz=America/New_York&ysp_redesign=1&ysp_platform=desktop&season=2023&league=ncaaw&leagueStructure=ncaaw.struct.div.1&count=200&sortStatId=FREE_THROWS_MADE&positionIds=&qualified=TRUE"

payload = {}
headers = {}

response = requests.request("GET", url, headers=headers, data=payload)

print(response.text)
```

Depending on your use case, you might not need to make any modifications to this code. However, for this project, we'll want to request data for multiple statistics (rather than just `FREE_THROWS_MADE`), so we will make a few changes to improve usability: 
1. Separate all of the query string parameters into a dictionary that will be passed into the `requests.get()` function 
2. Add function parameters for the statistic name, season, league, and count
3. Add basic error handling for request exceptions
4. Add a docstring 


```python
def get_data_for_stat(stat_name, season='2023', league='ncaaw', count='500'):
    """
    Retrieve basketball player statistics for a specified statistical category (stat_name) for a given season and league.

    Parameters:
    - stat_name (str): Specifies the name of the statistic to pull (points scored, field goals made, etc.).
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

This `get_data_for_stat()` function retrieves basketball player statistics for the top players in a specified statistical category (`stat_name`) for a given season (`2023`) and league (`ncaaw`). At a high level, the function begins by setting the base URL for the Yahoo Sports API endpoint for basketball season statistics in the `url` variable. Next, it uses the `params` variable to store various query string parameters for the API request (such as language, region, time zone, etc.).

The function then uses the `requests.get()` method to send a GET request to the API endpoint with the specified parameters. It checks the response status code for errors using `response.raise_for_status()`. If the response indicates an error (4xx or 5xx status code), it raises an exception. If the request is successful, the function returns the JSON data obtained from the API response.

You could further modify this code to fit your needs and add additional parameters to the function arguments (such as language or region), but we'll use this function to retrieve the top players for a given statistical category. Here's an example of retrieving the top 500 players (the default `count` is set to `500` in the function) by total points scored during the season: 


```python
example_response = get_data_for_stat('POINTS')
```

Since the function returns data in `json` format, we can take a look at the structure of the response data (using `example_response['data']`) and identify a few values of interest.
- `example_response['data']['statTypes']` stores the syntax and sort order for all of the available statistics
- `example_response['data']['leagues'][0]['leaders']` stores the result set for our request (where 'leaders' refers to the top players for the given request)

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
This is correctly showing Caitlin Clark as the top-scoring player of the 2023-2024 season, so we're ready to move on to the next step. 

## Write a Function to Format Data
Now that the API request function is working, let's define another function. This one will extract the relevant data and load it into a dataframe. 


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

This `format_response_data()` function will take the given `response_data` (from the Yahoo Sports API) and formats it into a pandas DataFrame. At a high level, the function begins by checking if the response_data is empty. If it is, the function returns `None`. It then attempts to access specific nested fields within the JSON data to extract relevant information. Inside the try block, the function extracts the player details (name, ID, team name) and player statistics from the JSON data. It iterates through each player in the response data, creating a dictionary (`player_row`) containing player details and statistics.

The player details include `PLAYER_NAME`, `PLAYER_ID`, and `TEAM_NAME`, while the player statistics are dynamically extracted from the JSON data. These player details and statistics are appended to a list (`data`). Finally, the function constructs a pandas DataFrame from that list of dictionaries and returns it.

By using this function in conjunction with the `get_data_for_stat()` function, we can efficiently retrieve, process, and format player statistics from the Yahoo Sports API into a structured DataFrame. 

We can test out this function using the `example_response` from the previous step. 


```python
example_dataframe = format_response_data(example_response)
example_dataframe.head()
```

|     | PLAYER_NAME    | PLAYER_ID      | TEAM_NAME    | GAMES | MINUTES_PLAYED | FIELD_GOALS_MADE | FIELD_GOAL_ATTEMPTS | FIELD_GOAL_PERCENTAGE | THREE_POINTS_MADE | THREE_POINT_ATTEMPTS | ... | FREE_THROW_PERCENTAGE | OFFENSIVE_REBOUNDS | DEFENSIVE_REBOUNDS | TOTAL_REBOUNDS | ASSISTS | TURNOVERS | STEALS | BLOCKS | FOULS | POINTS |
|-----|----------------|----------------|--------------|-------|----------------|------------------|---------------------|-----------------------|-------------------|----------------------|-----|----------------------|--------------------|--------------------|----------------|---------|-----------|--------|--------|-------|--------|
| 0   | Caitlin Clark  | ncaaw.p.64550  | Iowa         | 32    | 1088           | 332              | 719                 | 46.2                  | 168               | 437                  | ... | 85.8                 | 10                 | 224                | 234            | 282     | 151       | 55     | 17     | 61    | 1020   |
| 1   | JuJu Watkins   | ncaaw.p.112021 | USC          | 29    | 989            | 270              | 656                 | 41.2                  | 58                | 176                  | ... | 84.6                 | 52                 | 161                | 213            | 96      | 120       | 72     | 45     | 78    | 801    |
| 2   | Hannah Hidalgo | ncaaw.p.112250 | Notre Dame   | 31    | 1104           | 255              | 560                 | 45.5                  | 45                | 132                  | ... | 78.3                 | 25                 | 175                | 200            | 170     | 109       | 145    | 3      | 86    | 725    |
| 3   | Lucy Olsen     | ncaaw.p.67706  | Iowa         | 30    | 1087           | 268              | 612                 | 43.8                  | 47                | 158                  | ... | 80.9                 | 30                 | 114                | 144            | 115     | 73        | 57     | 18     | 72    | 697    |
| 4   | Ta'Niya Latson | ncaaw.p.70600  | Florida St. | 32    | 991            | 249              | 566                 | 44.0                  | 27                | 98                   | ... | 85.2                 | 17                 | 118                | 135            | 128     | 98        | 50     | 13     | 53    | 680    |


    5 rows × 23 columns


The middle of the dataframe might not be fully displayed, so let's print the full list of columns in this dataframe as well:


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

Here's a short description of each column: 
   - `PLAYER_NAME` The name of the basketball player.
   - `PLAYER_ID` The unique identifier for each basketball player.
   - `TEAM_NAME` The name of the player's basketball team.
   - `GAMES` The number of games the player has played in. This does not include attending a game but not playing (such as when a player is injured).
   - `MINUTES_PLAYED` The total number of minutes the player has been on the court. This is measured in the time on the game clock, not in real-time. There are typically 40 minutes of game-time, comprised of four 10-minute quarters. Overtime would be considered extra time. 
   - `FIELD_GOALS_MADE` The number of successful field goals made by the player. This includes both two-point and three-point field goals and does not include free throws. 
   - `FIELD_GOAL_ATTEMPTS` The total number of field goal attempts by the player.
   - `FIELD_GOAL_PERCENTAGE` The percentage of successful field goals made by the player.
   - `THREE_POINTS_MADE` The number of successful three-point baskets made by the player.
   - `THREE_POINT_ATTEMPTS` The total number of three-point basket attempts by the player.
   - `THREE_POINT_PERCENTAGE` The percentage of successful three-point baskets made by the player.
   - `FREE_THROWS_MADE` The number of successful free throws made by the player.
   - `FREE_THROW_ATTEMPTS` The total number of free throw attempts by the player.
   - `FREE_THROW_PERCENTAGE` The percentage of successful free throws made by the player.
   - `OFFENSIVE_REBOUNDS` The number of offensive rebounds grabbed by the player.
   - `DEFENSIVE_REBOUNDS` The number of defensive rebounds grabbed by the player.
   - `TOTAL_REBOUNDS` The total number of rebounds grabbed by the player.
   - `ASSISTS` The number of assists made by the player.
   - `TURNOVERS` The number of turnovers committed by the player.
   - `STEALS` The number of steals made by the player.
   - `BLOCKS` The number of baskets blocked by the player.
   - `FOULS` The number of fouls committed by the player.
   - `POINTS` The total number of points scored by the player.

Let's quickly verify that the dataframe has the proper number of rows using the `shape()` method. Since we set the `count` to `500` in the API request, there should be 500 rows in this dataframe. 

```python
example_dataframe.shape
```

    (500, 23)

This is correct, so we can move on to the next step.

<div class="email-subscription-container"></div>

## Write a Function to Format and Request Data
We could stop here and use those two functions to make our API requests, but let's define one more function that will request and format the data in one step.


```python
def get_and_format_data_for_stat(stat_name, season='2023', league='ncaaw'):
    """
    Retrieve basketball player statistics for a specified statistical category for a given season and league and format the data into a pandas DataFrame.

    Parameters:
    - stat_name (str): Specifies the name of the statistic to pull (points scored, field goals made, etc.).
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

This function combines the other two function calls and passes through the same parameters. In summary, it retrieves data for the top players for a specified statistic (`stat_name`), season (`season`), and league using the `get_data_for_stat()` function and then formats the data into a pandas DataFrame using the `format_response_data()` function. Since we'll be sending multiple requests to the API, this function streamlines the entire API request and formatting into one command and simplifies the process of generating the player statistics dataset. 

## Request and Format Statistics Data
Now we're ready to request data for each statistic. For this project, we'll pull the list of top players by five statistics: points, assists, rebounds, blocks, and steals. Here's what that looks like: 


```python
# Get and format data for each stat
points_top_players = get_and_format_data_for_stat('POINTS')
assists_top_players = get_and_format_data_for_stat('ASSISTS')
rebounds_top_players = get_and_format_data_for_stat('TOTAL_REBOUNDS')
blocks_top_players = get_and_format_data_for_stat('BLOCKS')
steals_top_players = get_and_format_data_for_stat('STEALS')
```

We now have five dataframes and each one contains the 500 players who scored the best in the given statistic. 

## Combine Statistics Data
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

We can see that this combined dataframe has the same number of columns as the individual dataframes, but will likely have fewer rows. 


```python
player_stats.shape
```

    (1392, 23)

The final player statistics dataset does have fewer rows, since we combined any duplicate rows. Now that we're done with this dataset, let's save it. As a reminder, you can use the `to_csv()` method instead of `.to_excel()` if you prefer. 

```python
player_stats.to_excel('player_stats.xlsx', index=False)
```

With that, we've finished acquiring both of the datasets needed for this project and can combine them into one final dataset that includes both the player information and player statistics.

# Combine Player Information and Statistics Datasets
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



Next, we'll use the pandas [`merge()`](https://pandas.pydata.org/docs/reference/api/pandas.merge.html) method to merge the two dataframes on the player's name. This final `player_data` dataframe includes columns for the player's information (height, position, class, etc.) as well as their statistics for the season (points, rebounds, blocks, assists, etc.)


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


So our final dataset has 900 rows and 27 columns. This is enough data for our project, so we're ready to move on to the cleaning and preprocessing steps.

# Export Data

If you're going to use a new Jupyter notebook / Python script for the next part of this series, then it's a good idea to export this dataset. As a reminder, you can use the [`to_csv()` method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_csv.html) instead of the [`.to_excel()` method](https://pandas.pydata.org/docs/reference/api/pandas.DataFrame.to_excel.html) if you prefer. 

```python
player_data.to_excel('player_data_raw.xlsx', index=False)
```

# Wrap up 
In this guide, we built a new dataset by acquiring and then combining the NCAA women's basketball player information dataset with the Yahoo Sports player statistics dataset. In the next part, we'll lay the groundwork for data analysis by cleaning and preprocessing the combined player data.

Also, all of the code snippets in today's guide are available in a Jupyter Notebook in the [ncaa-basketball-stats](https://github.com/pineconedata/ncaa-basketball-stats) repository on [GitHub](https://github.com/pineconedata/).

## Articles in this Series   
1. [Acquiring and Combining the Datasets](/2024-04-11-basketball-data-acquisition/) (Today's Guide)
2. [Cleaning and Preprocessing the Data](/2024-05-02-basketball-data-cleaning-preprocessing/)
3. [Engineering New Features](/2024-05-30-basketball-feature_engineering/)
4. [Exploratory Data Analysis](/2024-06-28-basketball-data-exploration/)
5. [Visualizations, Charts, and Graphs](/2024-07-29-basketball-visualizations/)
6. [Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)
7. [Training the Machine Learning Model](/2024-09-13-basketball-train-ols/)
8. [Evaluating the Machine Learning Model](/2024-11-27-basketball-evaluate-ols-model/)

<div class="email-subscription-container"></div>
<div id="sources"></div>

---
title: Basketball Data Science Project
subtitle: An end-to-end Python project using NCAA women's basketball data
permalink: /projects/basketball-data-science-project/
sitemap:
  priority: 1.0
---

This project is an end-to-end data science walkthrough using 2023–24 NCAA women's basketball player statistics. The goal is to explore whether Caitlin Clark's season was a statistical outlier and to build a machine learning workflow that predicts fantasy points from player performance metrics.

The project starts with data acquisition and preprocessing, then moves through feature engineering, exploratory data analysis, visualization, model selection, training, and evaluation. A bonus article extends the modeling section by comparing Ridge Regression with the ordinary least squares linear regression models trained earlier in the series. Each article focuses on one stage of the workflow and includes Python code, intermediate outputs, and explanations of the decisions made along the way.

![The steps for this data science project](/assets/img/posts/2024-04-11-basketball-data-acquisition/project_steps.png "The steps for this data science project")

## Project question

The central question for this project is:

> Was Caitlin Clark's 2023–24 season an outlier compared with other NCAA basketball players?

To explore that question, I built a Python analysis pipeline using player statistics from the 2023–24 season. The project uses common data science tools and techniques, including data cleaning, pandas transformations, exploratory analysis, data visualization, linear regression, and model evaluation.

## Project workflow

This project walks through the full data science process:

1. **Data acquisition** — collect data from the NCAA website and Yahoo Sports API.
2. **Data cleaning** — identify and correct missing, inconsistent, or invalid values.
3. **Data preprocessing** — convert datatypes, standardize values, and prepare the dataset for analysis.
4. **Feature engineering** — create derived metrics such as per-game statistics, conference labels, assist-to-turnover ratio, and fantasy points.
5. **Data exploration** — use summary statistics, correlations, and exploratory plots to understand relationships between variables.
6. **Data visualization** — create charts to compare players and highlight outliers.
7. **Machine learning** — select, train, and evaluate ordinary least squares linear regression models.

## Tools used

This project uses Python and common data science libraries:

- [Python](https://www.python.org/) — programming language used throughout the project
- [JupyterLab](https://jupyter.org/) — notebook environment used for analysis and visualization
- [pandas](https://pandas.pydata.org/docs/) — dataframe manipulation, cleaning, joins, and exports
- [NumPy](https://numpy.org/doc/) — numerical operations and array-based calculations
- [requests](https://requests.readthedocs.io/en/latest/) — API requests for basketball statistics data
- [json](https://docs.python.org/3/library/json.html) — working with JSON API responses
- [os](https://docs.python.org/3/library/os.html) — local file and path handling
- [openpyxl](https://openpyxl.readthedocs.io/en/stable/) — reading and writing Excel files
- [matplotlib](https://matplotlib.org/) — static charts and plot customization
- [seaborn](https://seaborn.pydata.org/) — statistical visualizations, including heatmaps and pairplots
- [plotly](https://plotly.com/) — interactive visualizations
- [scipy](https://scipy.org/) — scientific computing utilities
- [scikit-learn](https://scikit-learn.org/stable/index.html) — train/test splitting, linear regression, predictions, and model metrics
- [statsmodels](https://www.statsmodels.org/stable/index.html) — statistical modeling and regression analysis
- [joblib](https://joblib.readthedocs.io/en/stable/) — saving and loading trained models
- [Postman](https://www.postman.com/) — inspecting and testing API requests

## Data sources

The project uses two primary data sources:

- [NCAA basketball statistics](https://web1.ncaa.org/stats/StatsSrv/rankings?doWhat=archive&sportCode=WBB) — player information such as team, class, height, and position
- [Yahoo Sports NCAA basketball statistics](https://sports.yahoo.com/ncaaw/stats/individual/?selectedTable=0&leagueStructure=ncaaw.struct.div.1&sortStatId=FREE_THROWS_MADE) — individual player statistics such as points, rebounds, assists, blocks, steals, and field goal metrics

The raw datasets are cleaned, combined, transformed, and used throughout the rest of the project.

## Project articles

### 1. Project setup and data acquisition

[Read Part 1: Project Setup and Data Acquisition](/2024-04-11-basketball-data-acquisition/)

The project begins by collecting NCAA basketball player information and statistics. This article covers the data sources, project setup, and process for combining player information and player statistics into one dataset.

### 2. Data cleaning and preprocessing

[Read Part 2: Data Cleaning and Preprocessing](/2024-05-02-basketball-data-cleaning-preprocessing/)

This article cleans the raw basketball dataset so it is ready for analysis. It covers missing values, incorrect entries, datatype conversions, unit standardization, and other preprocessing steps needed before feature engineering.

### 3. Feature engineering

[Read Part 3: Feature Engineering](/2024-05-30-basketball-feature_engineering/)

This article creates new features from the cleaned dataset, including two-point metrics, conference labels, per-game statistics, assist-to-turnover ratio, and fantasy points.

### 4. Data exploration

[Read Part 4: Data Exploration](/2024-06-28-basketball-data-exploration/)

This article uses summary statistics, feature selection, correlation matrices, and exploratory plots to better understand the dataset before building visualizations and models.

### 5. Data visualizations

[Read Part 5: Data Visualizations](/2024-07-29-basketball-visualizations/)

This article creates visualizations to compare player performance, identify outliers, and better understand how Caitlin Clark's season compares with the rest of the dataset.

### 6. Model selection

[Read Part 6: Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/)

This article defines the prediction problem, chooses fantasy points as the target variable, selects input features, and explains why ordinary least squares linear regression is a reasonable starting model.

### 7. Training the model

[Read Part 7: Training a Linear Regression Model](/2024-09-13-basketball-train-ols/)

This article trains ordinary least squares linear regression models using the engineered basketball dataset. It covers train/test splitting, model fitting, reproducibility, coefficient interpretation, and alternate feature sets.

### 8. Evaluating the model

[Read Part 8: Evaluating a Linear Regression Model](/2024-11-27-basketball-evaluate-ols-model/)

This article evaluates the trained models using predictions, error metrics, residual analysis, and model diagnostics. It compares the full model with a reduced-feature model and discusses what the results suggest about model performance.

## Bonus articles

### Ridge vs. OLS linear regression models

[Read the bonus article: Ridge vs. OLS Linear Regression Models](/2025-04-04-ridge-regression-vs-ols-linear-regression-models/)

This article extends the machine learning portion of the project by comparing Ridge Regression with the ordinary least squares linear regression models trained earlier in the series. It revisits the same basketball dataset, trains Ridge models with scikit-learn, evaluates model performance, and discusses how regularization changes the model compared with OLS.

## Repository

The code for this project is available on GitHub:

[View the NCAA Basketball Stats repository](https://github.com/pineconedata/ncaa-basketball-stats)

## Why this project matters

This project is designed to show the full workflow behind a data science analysis, not just the final result. Each step builds on the previous one, which makes the series useful for readers who want to see how a messy sports dataset becomes a structured analysis and machine learning project.

The project also serves as a practical portfolio example of:

- acquiring data from online sources and APIs
- cleaning and transforming real-world data
- building reproducible analysis workflows
- creating interpretable visualizations
- applying regression modeling to sports data
- evaluating model performance
- communicating technical results clearly

## Start here

Start with [Part 1: Project Setup and Data Acquisition](/2024-04-11-basketball-data-acquisition/) to follow the project from the beginning.

For readers most interested in modeling, start with [Part 6: Selecting a Machine Learning Model](/2024-08-12-basketball-select-ml-ols/), then continue to the model training and evaluation articles.
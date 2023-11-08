---
layout: post
readtime: true
gh-repo: pineconedata/automated-feed-generator
gh-badge: [star, fork, follow]
title: Automate Your News Feed
subtitle: "How Web Scraping Powers a Python-Generated RSS Feed"
share-title: "Automate Your News Feed: How Web Scraping Powers a Python-Generated RSS Feed"
share-description: Want to subscribe to updates from your favourite website but they don’t publish an RSS feed? Learn how to automatically generate an RSS feed with Python, Selenium, and FeedGen. 
tags: [Python, Selenium, RSS, web scraping, automation]
thumbnail-img: /assets/img/rss-icon-96.png
---

RSS feeds are a fantstic tool for reading the latest content from your favourite websites without cluttering up your email inbox or manually visiting each website. However, not every website owner publishes an official RSS feed anymore, making it difficult to access up-to-date content in one place. That's why today we'll be digging into how to generate your own personalized RSS feed using Python and web scraping. 

## What is an RSS feed?
First off, an RSS (Really Simple Syndication) feed is a type of web feed that allows you to access updates to websites in a standardized, computer-readable format. Typically, a website owner will publish one RSS feed per website (or, for larger websites, one per category) that is updated regularly with new information. You can subscribe to multiple RSS feeds within an RSS feed reader (aka aggregator) - all you need to subscribe is the URL of the RSS feed. The RSS feed reader then displays an overview of the latest stories and information from all of your subscribed sites in one consolidated location. 

## Why use an RSS feed reader? 
If you haven't used an RSS feed reader before, you might not be familiar with the benefits they can offer over visiting a website directly or using a third-party news service. 
- Personalization - An RSS feed reader lets you personalize your news feed by subscribing only to the sources, topics, and categories that interest you. 
- Organization - You can easily organize your subscriptions in an RSS feed reader by creating folders, tagging articles, and prioritizing sources. 
- Improved Privacy - By using an RSS feed reader instead of visiting websites directly, you can protect your browsing data from being tracked by third-parties. 
- Fewer Distractions - Similar to the previous benefit, you can often bypass advertisements and intrusive popups that you might otherwise see on the original website. 
- Offline Accessibility - Many RSS feed readers offer the ability to save content for offline reading, allowing you to catch up on news or updates during periods of limited connectivity. 

## How do we get started? 
Now that we've covered the basics of RSS feeds and feed readers, let's dive in to how to generate an RSS feed for a website. In today's project, we'll use a website that does already publish an official RSS feed, but that will be useful for santiy-checking the end result. 

To get started, you'll need to have the following software installed on your system: 

- [Python](https://www.python.org/downloads/) (version 3.6 or higher)

- [Selenium](https://pypi.org/project/selenium/) (Python library for web scraping)

- [Firefox WebDriver](https://github.com/mozilla/geckodriver) (the browser we'll use for web scraping)

- [feedgen](https://pypi.org/project/feedgen/) (Python library for RSS feed generation)

Additional details about dependencies and version numbers can be found in the [`requirements.txt`](https://github.com/pineconedata/automated-feed-generator/blob/main/requirements.txt) file. 

*Note*: If you want to skip straight to implementation, you can follow the instructions in the [GitHub repo](https://github.com/pineconedata/automated-feed-generator) for this project. Otherwise, keep reading for step-by-step instructions and a breakdown of the code. 

## How will this process work?
At a high level, this process will work by regularly running a Python script that will visit the desired website, scrapte the latest content, and export that content to an RSS feed file. For convenience, the configuration options have been separated from the Python script itself (so that it's easy to execute the same script for multiple websites), but you could easily combine these two files if you want. You can either run the Python script on your own computer (self-host) or use a third-party service (like AWS, GCP, Azure, etc.). This post will briefly cover configuring this script to run locally, so you won't need any accounts with any third-parties to run this process.

# Configuration File
Since the configuration options are stored in a separate file (in JSON format) from the Python script, let's take a look at the required configuration parameters first. In order to scrape a website to generate an RSS feed, we'll need to set basic parameters like the website URL and title, as well as more detailed parameters like how to identify the details for each individual item in the generated RSS feed. For example, if we look at a blog, these details would include the post title, post URL, post image, etc.

In summary, the minimum parameters we need to specify are: 

- `website_url`: *Required*, URL of the website to scrape.
- `website_title`: *Required*, Title for the RSS feed.
- `website_description`: *Required*, Description for the RSS feed.
- `posts_list_selector`: *Required*, CSS selector for the list of posts to include in the RSS feed. 
- `title_selector`: *Required*, CSS selector for the title of each element.
- `link_selector`: *Required*, CSS selector for the link of each element.

Here's an example of what those required configuration parameters would look like for NASA's Space Station blog: 
{
  "website_url": "https://blogs.nasa.gov/spacestation/",
  "website_title": "NASA Space Station Blog",
  "website_description": "The official blog of NASA's space station news.",
  "posts_list_selector": "main#main > article",
  "title_selector": "h2.entry-title",
  "link_selector": "h2.entry-title > a"
}
If we run the Python process with only these parameters, then we will get a bare-bones RSS feed that only populates the title and URL of each blog post from the website. However, you might want to generate a more robust RSS feed that includes the date, thumbnail image, and description of each blog post as well. To that end, there are several optional parameters that can be set in the configuration file: 

- `image_selector`: *Optional*, CSS selector for the image of each element.
- `description_selector`: *Optional*, CSS selector for the description of each element.
- `description_type`: *Optional*, Determines if the description should pull from the `text` or `innerHTML` attribute of the element located via the `description_selector`. Certain RSS readers can have issues with HTML description content, so the default is `text`. 
- `date_selector`: *Optional - if specified, `date_format` is required*, CSS selector for the date of each element.
- `date_format`: *Optional - if specified, `date_selector` is required*, Date format of the date on the web page. This is used in conjunction with the `date_selector` to convert the date string to a datetime object using the [strptime](https://docs.python.org/3/library/datetime.html#strftime-strptime-behavior) method.
- `file_name`: *Optional*, Name of the output feed file. If not specified, the alphanumeric characters of `website_title` will be used instead.

Here's an example of what the entire configuration JSON file would look like for NASA's Space Station blog: 
{
  "website_url": "https://blogs.nasa.gov/spacestation/",
  "website_title": "NASA Space Station Blog",
  "website_description": "The official blog of NASA's space station news.",
  "posts_list_selector": "main#main > article",
  "title_selector": "h2.entry-title",
  "link_selector": "h2.entry-title > a",
  "image_selector": "div.entry-content > figure img",
  "description_selector": "div.entry-content",
  "description_type": "html",
  "date_selector": "footer.entry-footer > span.posted-on > a > time",
  "date_format": "%B %d, %Y"
}
## CSS Selectors

Once the script has opened the proper website URL, it will identify the list of posts to include in the RSS feed by using `posts_list_selector` (where `document.querySelectorAll(posts_list_selector)` should return the same number of HTML elements as the number of posts that should be included in your output RSS feed's content). Using the `document.querySelectorAll()` method is one of the quickest ways to identify your desired `posts_list_selector` value. MDN has additional details on the [`document.querySelectorAll()` method](https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelectorAll) if you are not familiar with it.

Then, the script will scrape the details of each post using the other selectors (`title_selector`, `link_selector`, `image_selector`, `description_selector`, and `date_selector`. The selectors for the post details (`title_selector`, `link_selector`, etc.) are sub-selectors of the `posts_list_selector`. For example, this sub-selector logic for the `title_selector` would be implemented in JavaScript as `document.querySelectorAll(posts_list_selector)...querySelector(title_selector)` (this script uses Python and Selenium, but the JS logic can be helpful for identifying the proper value of `title_selector`, etc. more quickly). MDN has additional details on the [`document.querySelector()` method](https://developer.mozilla.org/en-US/docs/Web/API/Document/querySelector) as well if you are not familiar with it.

Writing precise, reliable CSS selectors can be challenging, but you can always start by right-clicking "Inspect Element" and then right-clicking "Copy > CSS Selector" on the desired HTML element. 

# Python Process
Now that we've finished going through the configuration file, we can start going through the Python script.

## Import Dependencies
First, import all of the libraries that the script will depend on to function (aka dependencies). If you get any errors during this step, then you'll likely need to run `pip install` for any missing libraries. 


```python
import os
import pytz
import time
import json
import argparse
from datetime import datetime
from selenium import webdriver
from selenium.webdriver import FirefoxOptions
from selenium.webdriver.common.by import By
from feedgen.feed import FeedGenerator
```

## Import Configuration and Export Feed
Now that we've covered all of the parameters in the configuration file, the first step of the Python script will be to import those paramters. 


```python
if __name__ == "__main__":
    # Create an argument parser for command-line options
    parser = argparse.ArgumentParser(description="Scrape a website and generate an RSS feed.")

    # Add and parse a  command-line argument for specifying the path to the configuration file 
    parser.add_argument('--config_file', required=True, help="Path to the configuration file")
    args = parser.parse_args()

    # Open and read the specified configuration file that contains the website and scraping parameters
    with open(os.path.join("config", args.config_file), 'r') as config_file:
        config = json.load(config_file)


```

Once the configuration file is imported, we'll call the function that will scrape the website and generate the RSS feed. 


```python
    # Scrape website and generate the RSS feed
    rss_feed = scrape_and_generate_rss(config)
```

We'll go over the details of that function next, but the last item in the main function is to export the RSS feed that is generated by that function. 


```python
    # Save the generated RSS feed to a file
    if rss_feed:
        file_name = config["file_name"] if config.get("file_name") else "".join(x for x in config["website_title"] if x.isalnum())
        file_path = os.path.join("feeds", f'{file_name}.xml')
        with open(file_path, 'wb') as rss_file:
            rss_file.write(rss_feed)
        print(f'RSS feed generated and saved as "{file_path}".')
```

## Scrape the Website
Now we'll look at the details of the function that scrapes the website and generates the RSS feed. First, we'll want to parse the configuration options and configure the browser (which is the Firefox webdriver in this script). 


```python
def scrape_and_generate_rss(config):
    # Parse all of the settings from the configuration dictionary
    website_url = config['website_url']
    website_title = config['website_title']
    website_description = config['website_description']
    posts_list_selector = config['posts_list_selector']
    title_selector = config.get('title_selector', None)
    link_selector = config.get('link_selector', None)
    image_selector = config.get('image_selector', None)
    description_selector = config.get('description_selector', None)
    description_type = config.get('description_type', None)
    date_selector = config.get('date_selector', None)
    date_format = config.get('date_format', None)

    # Initialize a headless Firefox WebDriver for Selenium
    opts = FirefoxOptions()
    opts.add_argument("--headless")
    driver = webdriver.Firefox(options=opts)
```

Once the webdriver is setup, we can navigate to the website's URL and create a (mostly empty) RSS feed object. 


```python
    # Navigate to the specified website URL and wait for any dynamic content to load
    driver.get(website_url)
    time.sleep(2)

    # Create an RSS feed using FeedGenerator
    fg = FeedGenerator()
    fg.title(website_title)
    fg.link(href=website_url, rel='alternate')
    fg.description(website_description)
```

Once the feed object is created, we can begin populating it with the details of each post. Some of these details are optional, such as the thumbnail image and post date, and others are required. 


```python
    # Find and iterate through the list of posts on the web page
    posts_list = driver.find_elements(By.CSS_SELECTOR, posts_list_selector)
    for post in posts_list:
        # Create a new entry in the RSS feed for each post
        fe = fg.add_entry()

        # Extract information about each post (title, link, description, date, etc.) and add it to the feed entry
        # Extract and set the post title
        post_title = post.find_element(By.CSS_SELECTOR, title_selector).text
        fe.title(post_title)

        # Extract and set the post link
        post_link = post.find_element(By.CSS_SELECTOR, link_selector).get_attribute('href')
        fe.link(href=post_link)
        fe.guid(post_link)

        # Extract and set the post description
        if description_selector:
            if description_type and description_type == 'html':
                post_description = post.find_element(By.CSS_SELECTOR, description_selector).get_attribute('innerHTML')
            else:
                post_description = f'<p>{post.find_element(By.CSS_SELECTOR, description_selector).text}</p>'

        # Extract and set the post image
        if image_selector:
            image_link = post.find_element(By.CSS_SELECTOR, image_selector).get_attribute('src')
            post_description += f'<img src="{image_link}" alt="{post_title}">'
        fe.description(post_description)

        # Extract and set the published date
        if date_selector and date_format:
            post_date = post.find_element(By.CSS_SELECTOR, date_selector).text
            post_date = datetime.strptime(post_date, date_format).replace(tzinfo=pytz.utc)
            fe.pubDate(post_date)

```

After iterating through all of the posts, we can pretty-print and return the final RSS feed, as well as close the webdriver.

```python
    # Generate the RSS feed and return it as a string
    rss_feed = fg.rss_str(pretty=True)

    # Close the WebDriver
    driver.quit()

    return rss_feed
```

That's the entire Python script! It's a fairly simple process, and in the next section we'll go over how to run this script from the command line. 

# Running the Process
In order to run the process, there should be a particular folder structure for the files (illustrated below). 
-[parent folder]
--automated_feed_generator.py (the Python script)
--[config] (a sub-folder that holds the configuration files)
---NASASpaceStationBlog.json (the configuration file)
--[feeds] (a sub-folder that holds the RSS feed files, once generated)
---NASASpaceStationBlog.xml
Once you have this folder structure setup (although the `feeds` directory will be empty right now), you can run the script from the same folder as the `automated_feed_generator.py` file with the configuration file as a command-line argument. Here's an example: 

```shell
python3 automated_feed_generator.py --config_file 'NASASpaceStationBlog.json'
```
Once you run this command, the script will parse the configuration file, scrape the website, generate an RSS feed, and save it as an XML file in the `feeds` directory. If the output filename is not specified in the configuration file, then the filename is derived from the title of the website (with any non-alphanumeric characters removed).

# Scheduling

The entire process up to this point will only generate the RSS feed once. To keep your feed(s) up-to-date, you can schedule this Python script to run regularly. There are a variety of ways to do this, but the simplest example is setting it up as a cron job. You can use any cron job manager you like, but the example provided below works with [crontab](https://man7.org/linux/man-pages/man5/crontab.5.html).

1. Open your crontab configuration by running `crontab -e` as usual. 

2. Add a cron job entry to schedule the script at your [desired frequency](https://crontab.guru). For example, to run the script every day at 2:00 AM, you can add the following line:

```bash
0 2 * * * python3 ~/path/to/dir/automated-feed-generator/automated_feed_generator.py --config_file 'NASASpaceStationBlog.json'
```

- Make sure to replace `~/path/to/dir/automated-feed-generator/` with the actual directory where your Python script (`automated_feed_generator.py`) is located. 

- Add a separate line to your crontab file for each job that you want to schedule (typically one per configuration file).

3. Alternatively, you might want to run this Python script for all of the configuration files in the `config` directory at once and add only one line to your crontab configuration. In that case, you can move the Python script into a Shell script, like this: 

```bash
#!/bin/bash
cd ~/path/to/dir/automated-feed-generator

# Directory containing configuration files
configs_dir="config"

# Iterate over each file in the configs directory
for config_file in "$configs_dir"/*.json; do
    if [ -e "$config_file" ]; then
        # Extract the file name without the directory path
        config_file_name=$(basename "$config_file")

        echo "Processing $config_file_name..."
        python3 automated_feed_generator.py --config_file "$config_file_name"
    fi
done
```

- Just like above, make sure to replace `~/path/to/dir/automated-feed-generator` with the actual directory where your Python script (`automated_feed_generator.py`) is located. 

Then, you can add this script as a single cron job that will update all of the feeds at once. Here's an example of scheduling this script to run daily: 

```bash
@daily ~/path/to/dir/automated_feed_generator.sh
```

Now each `config_file.json` in the `config` directory will be passed to the `automated_feed_gneerator.py` script and will output a resulting file in the `feeds` directory. All that's left is to host your `feeds` directory somewhere that a RSS feed reader can pull from.

# Limitations

Before we wrap up, there are a few limitations to this Python process. There are workarounds for these limitations, but they are not covered in today's project. 

- **iFrames**: This script does not out-of-the-box support selectors that are within [iframes](https://developer.mozilla.org/en-US/docs/Web/HTML/Element/iframe).
    - However, there could be ways around this limitation, depending on the site structure and iframe details. A first step might be to fork this repo and modify the `posts_list = driver.find_elements(By.CSS_SELECTOR, posts_list_selector)` logic to something like `posts_list = driver.find_element(By.CSS_SELECTOR, iframe_selector).find_elements(By.CSS_SELECTOR, posts_list_selector)`. Note that this example logic is untested and might not work in all scenarios. 
- **Shadow DOMs**: This script does not out-of-the-box support selectors that are within [shadow DOMs](https://developer.mozilla.org/en-US/docs/Web/API/Web_components/Using_shadow_DOM).
    - Similar to iframes, there could be ways around this limitation. One possible solution might involve selecting the shadow DOM element and then selecting the posts_list (in JavaScript, this would look something like `document.querySelector(shadow_dom_selector).shadowRoot.querySelectorAll(posts_list_selector)`). 
- **Blocking**: This script is meant to be run at a low-volume (once per day) from a personal machine that has access to the website that you are scraping. This is not intended to be used for any malicious purposes, and, as such, no steps have been taken to ensure that the website owner does not block or tarpit your traffic.
    - Your traffic typically will not get blocked from running this script once per day. However, website owners have different policies and some might be more aggressive about blocking traffic (such as blocking all Linux+Firefox traffic). If you are concerned about getting blocked, then there is plenty of additional logic that could be added to this script to mitigate those risks.

# Wrap up
We've finished thoroughly going over how to write, run, and schedule a Python script that will scrape a website and generate an RSS feed. If you found this information helpful, please give it a like, share, or fork the [GitHub repo](https://github.com/pineconedata/automated-feed-generator). If you have any questions or suggestions, feel free to [contact me](/workwithme) or open a pull request! 


```python

```

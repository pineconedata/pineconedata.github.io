---
layout: post
title: "Create Colorful Custom QR Codes"
subtitle: "Designing, Customizing, and Formatting"
tags:  [Python, pillow, image manipulation, image processing]
share-title: "Creating Custom QR Codes: Designing, Customizing, and Formatting" 
share-description: Ever wondered how a QR code is created? Today, let's take a look at how to transform your logo (or any other image) into a customized QR code using Python in conjunction with the Pillow and Segno libraries!
thumbnail-img: /assets/img/posts/2025-02-21-create-colorful-custom-qr-code/thumbnail.png
share-img: /assets/img/posts/2025-02-21-create-colorful-custom-qr-code/social.png
gh-repo: pineconedata/pineconedata.github.io
gh-badge: [star, fork, follow]
---

Today we'll look into how a QR code is created using Python and the powerful Pillow library. Then, we'll learn how to transform a logo (or any other image) into a customized QR code using the Segno library. 

<div id="toc"></div>

# Getting Started
First, let's cover what you'll need if you want to follow along with this guide. This project depends on having Python, a package manager (such as `pip`), and the relevant packages (listed below) installed before moving forward. If you already have a Python environment up and running and are familiar with how to install packages, then feel free to skip to the [next section](#import-packages).

Before starting, you should have: 
- A computer with the appropriate access level to install and remove programs.
  - This guide uses a Linux distribution (specifically Ubuntu), but this code can work on any major OS with a few minor changes.
- A reliable internet connection to download the necessary software. 
- A text editor or [IDE](https://en.wikipedia.org/wiki/Integrated_development_environment) to create and edit program files.
- Basic programming knowledge is a plus. If you've never used Python before, then going through the [beginner's guide](https://wiki.python.org/moin/BeginnersGuide) first might be helpful.

## Python
This project depends on [Python](https://www.python.org), which is probably already installed on your computer if you're using a common OS. You can verify if Python is installed and which version is currently being used by running:
```bash
$ python3 --version
Python 3.10.12
```
If you get a response that says `command not found` instead of a version number, then Python is not installed. In that case, you can follow [Python's official installation instructions](https://www.python.org/downloads/). You can also try running `python --version` to see if you have Python 2 installed instead of Python 3, but ultimately you should install Python 3 for this project. 

## Package Manager (pip)
You'll also need a package manager to install dependencies. I've used [`pip`](https://pypi.org/project/pip/), but you can use any package manager (`conda` is another popular option). You can check if `pip` is installed by running: 
```bash
$ pip --version
pip 24.0 from /home/scoops/.local/lib/python3.10/site-packages/pip (python 3.10)
```
Similar to the command for Python, if you get a response that says `command not found` instead of a version number, then `pip` might not be installed on your device. You should follow the [official instructions](https://pip.pypa.io/en/stable/installation/) to install `pip` (or your package manager of choice) before proceeding. 

## Specific Packages
You should use your package manager to install the following required packages: 
  - [segno](https://pypi.org/project/segno/)
  - [qrcode-artistic](https://pypi.org/project/qrcode-artistic/)
  - [Pillow](https://pypi.org/project/pillow/) (aka Python Imaging Library or PIL)
  
There are also a few packages that are entirely optional: 
  - [numpy](https://numpy.org/)

## Import Packages
Once the required packages have been installed, we can start by importing them:


```python
import segno
import qrcode_artistic
from PIL import Image

# optional
import numpy as np
```

# QR Code Basics
Before we create our first QR code, let's quickly cover a few basics of QR codes. A QR code (Quick Response code) is a type of two-dimensional barcode (similar to the barcode on groceries) that can be scanned using a smartphone's camera (or similarly equipped tool). Your smartphone's OS likely already has QR code scanning functionality built in, so it's highly recommended to use your smartphone to scan the QR code examples in this article. QR codes are [commonly used](https://en.wikipedia.org/wiki/QR_code#Uses) to share links to websites, advertise contact information for a business, confirm an event ticket, or join a Wi-Fi network. 

QR codes make it easier and faster to share data quickly while reducing manual errors (like typing in a URL wrong), making them quite useful in advertising and marketing. They're also quite useful for their error correction - the ability for the data to still be readable even when part or the QR code is missing or damaged. Here's an example of a QR code that is substantially torn but still readable (it contains a link to `https://en.m.wikipedia.org/`): 

!["Example of a damaged but readable QR code"](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/example_damaged_qr.jpg)

However, QR codes also come with potential security risks (if the code is tampered with or replaced) as well as requiring a user to have a smartphone to access the data (unlike printed contact information). As such, QR codes are well suited to low-risk, high-visibility information sharing. 

# Create a QR Code
QR codes can contain a few different kinds of data, but let's start with the classic "Hello world!" example. We'll use [segno's .make_qr() function](https://segno.readthedocs.io/en/latest/api.html#segno.make_qr) with the rest of the parameters at default values.


```python
qrcode = segno.make_qr('Hello world!')
```

Great, we've made our first QR code! But what does it look like? Let's explore a few methods to look at the output. 

## Save the File
The simplest option is to save the QR code to a file with [segno's .save() function](https://segno.readthedocs.io/en/latest/api.html#segno.QRCode.save): 


```python
qrcode.save('qr_hello_world.png')
```

The downside of this approach is that we then have to open the file browser and manually open the file to see how the QR code turned out. Depending on your system configuration, this might also automatically open the image using your OS's default image viewer. 

Since these examples were created in a Jupyter Notebook, it'd be helpful to render the output directly. We could load the saved image using the Pillow library: 


```python
Image.open('qr_hello_world.png')
```




    
![Hello World QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/output_9_0.png)
    



This output is quite small, so let's use the `scale` parameter to display it a bit bigger: 


```python
qrcode.save('qr_hello_world.png', scale=10)
Image.open('qr_hello_world.png')
```




    
![Hello World QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_hello_world.png)
    



We can also use Pillow's [resize() function](https://pillow.readthedocs.io/en/stable/reference/Image.html#PIL.Image.Image.resize) to standardize the displayed size of the QR codes.  


```python
display_size = [250, 250]
img = Image.open('qr_hello_world.png')
img.resize(display_size)
```




    
![Hello World QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_hello_world.png)
    



You might not want to create a file for each QR code, so let's look at a few other methods that will render directly to the Jupyter Notebook. 

## Use the Memory Buffer
From segno's documentation, we can save the QR code into a memory buffer as a PNG and then let Pillow load the image: 


```python
import io


out = io.BytesIO()
qrcode.save(out, scale=10, kind='png')
out.seek(0)
Image.open(out)
```




    
![Hello World QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_hello_world.png)
    



## Use Data URI and HTML
Since Jupyter Notebooks support rendering HTML, we can also use [segno's .png_data_uri() function](https://segno.readthedocs.io/en/latest/api.html#segno.QRCode.png_data_uri) to generate and display the QR code as HTML: 


```python
from IPython import display


data_uri = qrcode.png_data_uri(scale=10)
display.display(display.HTML(f'<img src="{data_uri}"/>'))
```


<img src="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAASIAAAEiAQAAAAB1xeIbAAABBUlEQVR42u2aQQ6DIBREf9oDeCSuzpE8AAnFj19pV3bBmJjHAgVm9cBhIFq9ULKhQoUKFapHqGwv7+hfoye56mVXCqr/VDv71Ik39nmJt3MA9lPZbws9NeJmwd4/A9gr2afehP0d7I8K9mK/91nA75Xsj5wT656co2J/ln2v/Umi8Jrt957v81Is8v0b9hL2Y7QfLQj20/1+8UPuGHFaMxwJ9rM9p6/7rfKpwHNUe+1XvDyx4znSnBMuPyQe2Evyffcc33rrit+Lz7W1tN4taGb8/p77HB8onGul7I94id/r2bvf1zJcqMFe4/f9gEW+vyPn9GeKXRf2wnzPvwWoUKFC9UzVB5JITYUwZSyJAAAAAElFTkSuQmCC"/>


## Use segno's to_pil() Function
Conveniently, QR codes can also have a `to_pil()` function that returns a [Pillow Image](https://pillow.readthedocs.io/en/stable/reference/Image.html) that will display directly within the notebook. (*Note:* This function is only available if you have installed the `qrcode-artistic` plugin.)


```python
qrcode.to_pil(scale=10)
```




    
![Hello World QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_hello_world.png)
    



We'll use this `to_pil()` function when possible while designing and testing QR codes and then we can use the `save()` function for QR codes that we want to keep. 

# Data Contained in a QR Code
In the example above, the data contained in the QR code is the alphanumeric text "Hello world!". Other kinds of data can be stored in a QR code, including URLs, WiFi connection details, contact cards, emails, or even the [entire Windows executable of the game Snake](https://www.youtube.com/watch?v=ExwqNreocpg)! Segno also has [helper functions](https://segno.readthedocs.io/en/stable/api.html#module-segno.helpers) to make creating these QR codes even easier. [How much data can be stored](https://en.wikipedia.org/wiki/QR_code#Information_capacity) in a given QR code depends on three factors: 

## The type of data stored
Certain types of data take up more space (more bits per character or value) than other types of data. For example, alphanumeric characters take up more space than numeric-only characters, a numeric-only QR code can store more data than an alphanumeric QR code.

The `mode` parameter in the [segno.make_qr()](https://segno.readthedocs.io/en/latest/api.html#segno.make_qr) function can be used to set the type of data stored. Otherwise, the appropriate mode will be automatically determined.

The available modes are (listed in descending order of storage capacity):
  - numeric only
  - alphanumeric
  - binary
  - kanji/hanzi    

## The version of the QR code
Older versions of QR codes have less storage capacity than newer versions. The maximum size/dimensions of the QR code can be calculated by: 
\\((4*\text{version_number})+17\\). 

The `version` parameter in the segno.make_qr() function can be used to set the version number. Otherwise, the smallest version that fits the input data will be used.

Here are example QR code dimensions for a few versions:
    
|version_number|calculation|dimensions|
|-|-|-|
|1|(4*1)+17|21x21|
|2|(4*2)+17|25x25|
|10|(4*10)+17|57x57|
|40|(4*40)+17|177x177|

## The error correction level 
As mentioned earlier, QR codes have error correction capabilities (where the data can still be read even if the QR code is damaged or missing sections). How much error can be corrected depends on the error correction level.

The `error` parameter of the segno.make_qr() function can be used to set the desired error correction level. Otherwise, 'L' for Low is used by default. *Note* The segno.make_qr() function also includes a `boost_error` parameter that can boost the error correction level by appropriating any unused data bytes for additional error correction. 

The [standard error correction levels](https://en.wikipedia.org/wiki/QR_code#Error_correction) are: 
    
|Error Correction Level|Error Correction Name|Approximate Correction Capability|
|-|-|-|
|L|Low|Up to ~7% of bytes|
|M|Medium|Up to ~15% of bytes|
|Q|Quartile|Up to ~25% of bytes|
|H|High|Up to ~30% of bytes|

## Error Correction Example

To illustrate the error correction capabilities of QR codes, we can use numpy to slice a big chunk out of the middle of the QR code.


```python
qrcode_boosted = segno.make_qr('Hello world!', error='H', boost_error=True, version=25)
img = qrcode_boosted.to_pil(scale=10).convert("RGB")

arr = np.array(img)

stripe_size = round(img.size[1] * 0.25)
start = (arr.shape[0] - stripe_size) // 2
arr[:, start:start+stripe_size] = [255, 255, 255]

Image.fromarray(arr).resize(display_size)
```


    
![Example of QR Code with error correction](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/output_24_0.png)
    


<div class="email-subscription-container"></div>

# QR Code Customization
Next up, let's look at customizing QR codes. There are several ways to do this, but today we'll focus on changing the colors and adding a background image. These customizations make the QR code more visually appealing and distinct compared to traditional QR codes. 

## Creating an Artistic QR Code
We'll be using the [qrcode-artistic plugin](https://pypi.org/project/qrcode-artistic/) to add the background image to the QR code today. First, let's start with our basic QR code plus `H` for high error correction:


```python
qrcode = segno.make_qr('Hello world!', error='H')
```

Next, let's define the background image (*Note:* You can make a URL request to get the background image instead of using a local file if you prefer.): 


```python
background_image = 'logo.png'
Image.open(background_image).resize(display_size)
```



Now let's add this image as the background to the QR code using the `to_artistic()` function: 


```python
qrcode.to_artistic(background=background_image, target='qr_artistic.png', kind='png', scale=10)
Image.open('qr_artistic.png').resize(display_size)
```




    
![Artistic QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic.png)
    



Excellent! We can see the background image did get added to the QR code and it is still readable. We could stop here, but we can make a few modifications to make this more visually appealing.

## How NOT to Create an Artistic QR Code
Before jumping into improving the QR code, let's quickly cover a few things to avoid. 

### Not Setting Error Correction
First, when embedding an image into a QR code like this, it's important to set the error correction. 


```python
qrcode = segno.make_qr('Hello world!')  # this is incorrect as it is missing the 'error' parameter
qrcode.to_artistic(background=background_image, target='qr_artistic_without_error_correction.png', kind='png', scale=10)
Image.open('qr_artistic_without_error_correction.png').resize(display_size)
```




    
![Artistic QR Code without error correction](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic_without_error_correction.png)
    



### Setting the Target to the Background
Next, let's highlight the importance of setting the `target` parameter to a different value than the `background` parameter. Segno's documentation for [artistic QR codes](https://segno.readthedocs.io/en/latest/artistic-qrcodes.html) has the following example: 

![segno example of static background QR code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/example_segno_static_background.png)

When I first looked at this example, I saw that the file names were the same for the `target` and `background` parameters and didn't notice that the background parameter had `src/`. This led to running the following code: 


```python
# this is incorrect as the target should be different than the background
# qrcode.to_artistic(background=background_image, target=background_image, kind='png', scale=10)
```

Running this code once is fine. However, it replaces the background image with the QR code that was just generated, so continuing to run this code quickly ends up with an absolutely unreadable (but neat) QR code: 


```python
Image.open('qr_artistic_endless.png').resize(display_size)
```




    
![Endless Artistic QR Code](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic_endless.png)
    



Incredibly, this QR code is still readable, but it's not recommended to create QR codes this way for obvious reasons!

## Additional Customizations

Now that we've covered what **not** to do, let's move on to customizing the QR code. 

### Change the Content
First off, storing "Hello world!" is a good starting point, but let's update the content to a URL: 


```python
content = 'https://www.pineconedata.com/hello/'
qrcode = segno.make_qr(content, error='H')
qrcode.to_artistic(background=background_image, target='qr_artistic_content.png', kind='png', scale=10)
Image.open('qr_artistic_content.png').resize(display_size)
```




    
![Artistic QR Content](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic_content.png)
    



### Add a Border to the Background Image
The next improvement is to make more of the logo visible and avoid part of it being covered by the finders. To do this, we can add a white border to the background image (you can do this either with Pillow or numpy or an image editing tool). 

*Note*: Do not use the `border` parameter to add space to the logo. The `border` parameter will instead change the border around the entire QR code, not just around the background image (which in this case is the logo).


```python
background_img = 'logo_with_border.png'
qrcode = segno.make_qr(content, error='H')
qrcode.to_artistic(background=background_img, target='qr_artistic_with_border.png', kind='png', scale=10)
Image.open('qr_artistic_with_border.png').resize(display_size)
```




    
![Artistic QR Code with border](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic_with_border.png)
    



### Customize Colors
Another customization that makes your QR code look really polished is modifying the colors of the QR code. There are several options for this, starting with [Segno's documentation](https://segno.readthedocs.io/en/latest/serializers.html#color-of-dark-and-light-modules) on the `light` and `dark` parameters that set the colors of all of the light and dark modules of the QR code. We can use the `dark` parameter to replace all of the black modules with the logo's color: 


```python
logo_color = '#615EEA'
qrcode = segno.make_qr(content, error='H')
qrcode.to_artistic(background=background_img, target='qr_artistic_colors.png', kind='png', scale=10, dark=logo_color)
Image.open('qr_artistic_colors.png').resize(display_size)
```




    
![Artistic QR Code with custom colors](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic_colors.png)
    



This looks great! However, it doesn't have quite as much contrast, so we can swap the `dark` parameter with additional parameters for more specific control over the colors. There are parameters for the finders, alignment, timing, version, data, etc., so we'll use a few of these parameters to alter the colors slightly: 


```python
qrcode = segno.make_qr(content, error='H')
qrcode.to_artistic(background=background_img, target='qr_artistic_colors_updated.png', kind='png', scale=10,
                   finder_dark='#615EEA', alignment_dark='615EEA', timing_dark='615EEA')
Image.open('qr_artistic_colors_updated.png').resize(display_size)
```




    
![Artistic QR Code with updated custom colors](/assets/img/posts/2025-02-21-create-colorful-custom-qr-code/qr_artistic_colors_updated.png)
    



Segno's [colorful QR codes documentation page](https://segno.readthedocs.io/en/latest/colorful-qrcodes.html) is full of excellent examples of each of these different parameters, so it's highly recommended to check those out when customizing your own QR code. 

# Wrap Up
In today's guide, we covered QR code characteristics, including content and error correction levels, as well as options to customize QR codes like adding an image and refining the colors. 

<div class="email-subscription-container"></div>
<div id="sources"></div>

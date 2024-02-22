---
layout: post
readtime: true
gh-repo: pineconedata
gh-badge: [star, fork, follow]
title: Jump into Data Science with JupyterLab
subtitle: "A Comprehensive Guide on Debian Linux"
share-title: "Jump into Data Science with JupyterLab: A Comprehensive Guide for Debian Linux"
share-img: /assets/img/posts/2023-11-14-installing-jupyterlab-social.png
share-description: Interested in self-hosting JupyterLab on Debian Linux? Discover the power of JupyterLab and Jupyter Notebooks in this comprehensive guide that is perfect for data scientists and Python enthusiasts.
tags: [Python, Jupyter, Linux, data science, server, self-host]
thumbnail-img: /assets/img/jupyterlab_icon.jpeg
---

Today we'll be covering how to self-host JupyterLab on a Linux machine for a single user. If you're not familiar with [Project Jupyter](http://jupyter.org/), I highly recommend that you check it out, especially if you work in data science or with Python. Project Jupyter is migrating the classic [Jupyter Notebooks](https://jupyterlab.readthedocs.io/en/stable/user/notebook.html#notebook) into the new [JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html), so we'll be installing and hosting JupyterLab today. JupyterLab is a fantastic tool that includes Jupyter Notebooks, terminals, and text editors, as well as a library of extensions to customize your experience. 

**Please note: this method of hosting only works for a single user. Here is additional information that JupyterLab's documentation provided at the time of writing, but please check the most up-to-date documentation and ensure this works for your use case before proceeding.**

![warning about hosting for a single user only](/assets/img/posts/2023-11-14_jupyterlab_single_user_warning.png "warning about hosting for a single user only")

## Getting Started

Official documentation for running a JupyterLab server has not been published yet (as of November 2023), so I primarily followed the documentation on creating a [Jupyter Notebook server](https://jupyter-notebook.readthedocs.io/en/stable/public_server.html). Modifications between Jupyter Notebook and JupyterLab are mentioned throughout this article, but I recommend that you check JupyterLab's documentation for recent updates. 

### Requirements
Before we get started, you should have: 
1. A Linux server, preferably running a Debian-based distribution. At its simplest, this Linux server could be a computer that is left on all the time and has Ubuntu Desktop installed. 
   - The instructions here should work for any Debian-based distribution, including [Ubuntu](https://ubuntu.com/), but I am using [Pop!\_OS](https://pop.system76.com/) specifically.
2. A reliable internet connection. 
3. A basic familiarity with the terminal, such as [Ubuntu's introductory guide to the terminal](https://ubuntu.com/tutorials/command-line-for-beginners#1-overview).
4. The ability to edit files using a text editor. I completed this entire installation and configuration via SSH, so I edited files directly in the terminal using `nano`. Linux Hint provides an [introductory guide to nano](https://linuxhint.com/nano-editor-beginner-guide/) and full documentation can be found on [nano's website](https://www.nano-editor.org/dist/v2.2/nano.html).

### Conventions
If this is your first time using the terminal, it might be helpful to review the conventions before proceeding. 
- This article includes example screenshots of my terminal for certain commands. Terminal input lines start with `scraps@pop-os:~$` since the computer used for this article is named `scraps` and is running on `pop-os`. Your terminal lines will start with a different string, but the inputs should otherwise be the same. Everything after the `$` is the terminal input; the examples will show commands like `scraps@pop-os:~$ python3 --version`, so you would input just `python3 --version` into your terminal. 
- This article uses the `+` sign between two keys (such as `CTRL+C` to indicate when you should hold-press those keys together.
- This article uses the `>` sign between two commands (such as `File` and `Log Out`) to indicate when you should click one menu item and then another.
- As a quick reminder, if you want to copy/paste from the terminal, you should typically right-click and select copy/paste instead of using the `CTRL+C` or `CTRL+V` shortcuts. Those shortcuts are often mapped to different behavior in the terminal (for example, once the server is up and running, then `CTRL+C` is a shortcut to shutdown the server), so it's recommended to avoid using the copy/paste shortcuts entirely when using the terminal. 

### Dependencies 
JupyterLab depends on [Python](https://www.python.org), which is probably already installed on your computer if you're using Ubuntu or Pop!OS. You can verify if Python is installed and which version is currently being used by running:
```bash
scraps@pop-os:~$ python3 --version
Python 3.10.6
```
If you get a response that says `command not found` instead of a version number, then Python is not installed. In that case, you can follow [Python's official installation instructions](https://www.python.org/downloads/). You can also try running `python --version` to see if you have Python 2 installed instead of Python 3, but ultimately you should install Python 3 for this project. 

You'll also need a package manager to install JupyterLab. I've used `pip`, but this works with any package manager. You can check if `pip` is installed by running: 
```bash
scraps@pop-os:~$ pip --version
pip 22.0.2 from /usr/lib/python3/dist-packages/pip (python 3.10)
```
Similar to the command for Python, if you get a response that says `command not found` instead of a version number, then `pip` might not be installed on your device. You should follow the [official instructions](https://pip.pypa.io/en/stable/installation/) to install `pip` before proceeding.

## Installing JupyterLab
Now we're ready to start installing JupyterLab. Project Jupyter publishes [instructions on how to install JupyterLab](https://jupyterlab.readthedocs.io/en/stable/getting_started/installation.html) through a variety of package managers and methods. Since I'm using `pip`, the command is:
```bash
scraps@pop-os:~$ pip install jupyterlab
```
The terminal output will be quite detailed and is usually about a hundred lines long. The output might pause with a message about the amount of disk space that will be used for the installation, which you can confirm by typing `y` and pressing `enter`. 

**Please note: this installation modifies your `PATH` variable, so you should log out and back in before before moving on to the next step.**

To verify the installation worked, you can test out the `jupyter lab` command by listing all of the currently running servers with `jupyter lab list`. If you get a `command not found` response like shown below, then you should restart the machine and double-check your `PATH` variable. 
```bash
scraps@pop-os:~$ jupyter lab list
jupyter-lab: command not found
```
If the output is similar to what is shown below, then you know the installation worked! 
```bash
scraps@pop-os:~/Code/jupyterlab$ jupyter lab list
Currently running servers:
scraps@pop-os:~/Code/jupyterlab$ 
```
There aren't any currently running servers listed because JupyterLab hasn't been started up yet. 

### Starting the Server
We'll be modifying JupyterLab's configuration in the next section, but we can start the JupyterLab server with all of the default settings for now: 
```bash
scraps@pop-os:~$ jupyter lab
[I 2022-12-04 20:30:42.533 ServerApp] jupyterlab | extension was successfully linked.
[I 2022-12-04 20:30:42.541 ServerApp] nbclassic | extension was successfully linked.
[I 2022-12-04 20:30:42.543 ServerApp] Writing Jupyter server cookie secret to /home/scraps/.local/share/jupyter/runtime/jupyter_cookie_secret
[I 2022-12-04 20:30:42.701 ServerApp] notebook_shim | extension was successfully linked.
[I 2022-12-04 20:30:42.717 ServerApp] notebook_shim | extension was successfully loaded.
[I 2022-12-04 20:30:42.718 LabApp] JupyterLab extension loaded from /home/scraps/.local/lib/python3.10/site-packages/jupyterlab
[I 2022-12-04 20:30:42.718 LabApp] JupyterLab application directory is /home/scraps/.local/share/jupyter/lab
[I 2022-12-04 20:30:42.722 ServerApp] jupyterlab | extension was successfully loaded.
[I 2022-12-04 20:30:42.725 ServerApp] nbclassic | extension was successfully loaded.
[I 2022-12-04 20:30:42.725 ServerApp] Serving notebooks from local directory: /home/scraps
[I 2022-12-04 20:30:42.725 ServerApp] Jupyter Server 1.23.3 is running at:
[I 2022-12-04 20:30:42.725 ServerApp] http://localhost:8888/lab?token=355d4dac266e24d7d93645945dbbb8ecda9287539a2ff9df
[I 2022-12-04 20:30:42.725 ServerApp]  or http://127.0.0.1:8888/lab?token=355d4dac266e24d7d93645945dbbb8ecda9287539a2ff9df
[I 2022-12-04 20:30:42.725 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 2022-12-04 20:30:42.728 ServerApp] No web browser found: could not locate runnable browser.
[C 2022-12-04 20:30:42.729 ServerApp]

    To access the server, open this file in a browser:
        file:///home/scraps/.local/share/jupyter/runtime/jpserver-27212-open.html
    Or copy and paste one of these URLs:
        http://localhost:8888/lab?token=355d4dac266e24d7d93645945dbbb8ecda9287539a2ff9df
     or http://127.0.0.1:8888/lab?token=355d4dac266e24d7d93645945dbbb8ecda9287539a2ff9df
^C[I 2022-12-04 20:31:51.227 ServerApp] interrupted
Serving notebooks from local directory: /home/scraps
0 active kernels
```
Now that JupyterLab is running, you might notice that this terminal window is linked to that process and you can no longer run new terminal commands in this window. If you want to see the output of `jupyter lab list` now, you can open a new terminal and run that command again to see the location of the current instance: 
```bash
scraps@pop-os:~/Code/jupyterlab$ jupyter lab list
Currently running servers:
http://localhost:8888/ :: /home/scraps/Code/jupyterlab
scraps@pop-os:~/Code/jupyterlab$ 
```
You can open a web browser and check out the new install now by visiting the URL listed in the terminal (which is `http://localhost:8888/` in the example above). The basic installation of JupyterLab is now complete! 

### Stopping the Server
Your JupyterLab server will now continue running until it's told to stop or is forcibly shut down (if the desktop suddenly loses power, for example). There are a few common ways to stop a JupyterLab server that is already running. 
1. Enter `CTRL+C` in the terminal window where you initially ran the `jupyter lab` command. You should see a message like this: `Shutdown this Jupyter server (y/[n])?` If you type `y` and press `enter` to confirm, then the server will shut down. Note that the server will wait 5 seconds by default for the confirmation before it times out and requires a new `CTRL+C` input to shut down. 
2. Press `CTRL+C` twice in rapid succession in the terminal window where you initially ran the `jupyter lab` command to bypass the confirmation prompt and shut down the server immediately. 
3. In any terminal window, run the command `jupyter lab stop <port number>` where the `<port number>` is replaced by the port number listed in the output of the `jupyter lab list` command (in the example above, the port is `8888`, so the full command would be `jupyter lab stop 8888`). This command is useful if you no longer have the original terminal window up or if you started the server in the background. Here is a full example of that:
```bash
scraps@pop-os:~$ jupyter-lab list
Currently running servers:
https://localhost:9999/ :: /home/scraps/Code/jupyterlab
scraps@pop-os:~$ jupyter-lab stop 9999
Shutting down server on 9999...
```
4. You can use the JupyterLab GUI in the web browser (at `http://localhost:8888/ http://localhost:8888/ ` for example) by clicking File > Shut Down to shut down the server. This is also useful if you no longer have the original terminal window open or if you started the server as a background process.

**Note: if you want to end your session but leave the server running (perhaps to take a break or switch computers), then you can also use the GUI to save your open notebooks, close all tabs, and then log out. If you leave too many tabs open for too long, then the server might shut itself down.**

Regardless of which method of shutting down the server you choose, you should confirm the server shut down properly. For the terminal commands, you'll know the server shut down successfully when the terminal outputs messages like this: 
```bash
[C 2022-12-04 20:31:55.830 ServerApp] Shutdown confirmed
[I 2022-12-04 20:31:55.830 ServerApp] Shutting down 3 extensions
[I 2022-12-04 20:31:55.831 ServerApp] Shutting down 0 terminals
```
For the web browser interface, you'll know the server shut down successfully when the web page (at `http://localhost:8888/` for example) no longer shows JupyterLab and instead says "Unable to Connect". If you already had that web page open, you might have to refresh it to confirm that the server has shut down. 

It's not recommended to run the server without configuring some basic settings (such as a password and HTTPS), so we'll start customizing our JupyterLab installation in the next section. If you haven't done so already, be sure to shut down your JupyterLab server before proceeding. 

## Configuring JupyterLab
JupyterLab's configuration is primarily stored in the `jupyter_lab_config.py` file under the `.jupyter` folder in your home directory, so we'll start by navigating to that folder. If you're using the terminal, then navigate back to your home directory and then into the `.jupyter` folder with `cd .jupyter`. Note that folders beginning with `.` are hidden by default, so you might want to run `ls -a` instead of just `ls` to make sure you can see the folder. If you are navigating folders via the UI, then you might want to check "Show hidden files" in the file browser. The rest of the commands in this section should be executed from inside the `.jupyter` folder unless otherwise noted. 
**If you don't see the `.jupyter folder`, then run the `jupyter lab --generate-config` command first.**
```bash
scraps@pop-os:~/Code/jupyterlab$ cd ~
scraps@pop-os:~$ cd .jupyter
-bash: cd: .jupyter: No such file or directory
scraps@pop-os:~$ jupyter lab --generate-config
scraps@pop-os:~$ cd .jupyter
scraps@pop-os:~/.jupyter$ ls -a
.  ..  jupyter_lab_config.py
```
Now we're ready to start customizing our JupyterLab install! Before making any edits to the configuration file, we can start by creating a password and configuring HTTPS (instead of the default HTTP). [This article](https://web.dev/articles/when-to-use-local-https) lists the benefits of enabling HTTPS for projects that are only intended to run locally.

### Setting a Password
You can configure a password for JupyterLab using the following command:  
```bash
scraps@pop-os:~/.jupyter$ jupyter lab password
Enter password:
Verify password:
[JupyterPasswordApp] Wrote hashed password to /home/scraps/.jupyter/jupyter_server_config.json
```
You'll be prompted to enter the password twice, and then it will save a hash of that password to a newly created JSON file. We'll need to get a copy of the hashed password from the JSON file for later. To do that, you can use any text editor to open the JSON file (the example below is using `nano` since I SSH'd into this machine): 
```bash
scraps@pop-os:~/.jupyter$ ls -a
.  ..  jupyter_lab_config.py  jupyter_server_config.json
scraps@pop-os:~/.jupyter$ nano jupyter_server_config.json
```
When you open that JSON file, it should look something like this: 
```json
{
  "ServerApp": {
    "password": ""
  }
}
```
Copy and paste the value of the `password` field somewhere else so you can reference it later. The password value should begin with something like `argon2` or `sha1`. The JupyterLab instructions use `sha1`, but my installation uses `argon2` by default, so either should work. 

**Please note: if your password is stored in plain text (if you can read out exactly what you entered in the previous step), then you should follow the instructions in [this StackOverflow thread](https://stackoverflow.com/questions/64299457/jupyter-password-not-hashed) to ensure your password is hashed.** You can also use those instructions to change the hashing algorithm (from `argon2` to `sha1`, for example).

### Creating an SSL Certificate
To enable HTTPS on the JupyterLab server, we first need to create an SSL certificate. You can use either [`lets-encrypt`](https://letsencrypt.org/) or [`openssl`](https://www.openssl.org/docs/man1.0.2/man1/openssl-req.html) to create the certificate, but we'll be using `openssl` for this project. For simplicity, we'll be creating a [self-signed certificate](https://en.wikipedia.org/wiki/Self-signed_certificate) today, but it's generally recommended to use a certificate issued by a [certificate authority](https://en.wikipedia.org/wiki/Certificate_authority) instead. 

**Please note: you should run the certificate creation commands from the directory that you want the certificate files to be stored in. For simplicity, today's example will show creating the certificate files in the `.jupyter` folder, but you can change directories to the desired location before creating the certificate.**

To create the certificate with openssl, we'll start with the command listed in JupyterLab's documentation: 
```bash
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout mykey.key -out mycert.pem
```
You can choose the filenames for the `.key` and `.pem` files with this command as well, so I modified the `mykey` and `mycert` filenames to `jupyterkey` and `jupytercert` respectively. Here's an example of that command with modified filenames: 
```bash
scraps@pop-os:~/.jupyter$ openssl req -x509 -nodes -days 365 rsa:2048 -keyout jupyterkey.key -out jupytercert.pem
req: Use -help for summary.
```

However, that command did not work for me. After a bit of research, I found that adding the `-newkey` parameter worked, like this:
```bash
scraps@pop-os:~/.jupyter$ openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout jupyterkey.key -out jupytercert.pem
-----
You are about to be asked to enter information that will be incorporated
into your certificate request.
What you are about to enter is what is called a Distinguished Name or a DN.
There are quite a few fields but you can leave some blank
For some fields there will be a default value,
If you enter '.', the field will be left blank.
-----
Country Name (2 letter code) [AU]:
State or Province Name (full name) [Some-State]:
Locality Name (eg, city) []:
Organization Name (eg, company) [Internet Widgits Pty Ltd]:
Organizational Unit Name (eg, section) []:
Common Name (e.g. server FQDN or YOUR name) []:
Email Address []:
scraps@pop-os:~/.jupyter$
```
After running this command, you will be prompted to populate some fields or leave them at their default values. Once you finish entering the details, you can run `ls -a` again to see that there are two new files, with the filenames specified in the previous command. 
```bash
scraps@pop-os:~/.jupyter$ ls -a
.  ..  jupytercert.pem  jupyterkey.key  jupyter_lab_config.py  jupyter_server_config.json
```
### Updating the Configuration File
Now that we have the password and SSL certificate, we need to update JupyterLab's configuration file to use these parameters. The configuration file is located within the `.jupyter` folder and is named `jupyter_lab_config.py`. You can use any text editor to modify this file, but the examples below are shown using `nano`:  
scraps@pop-os:~/.jupyter$ nano jupyter_lab_config.py
Once you open the file, you should see some commented-out lines with configuration parameters: 

![example of editing jupyter_lab_config.py file using nano](/assets/img/posts/2023-11-14_nano_jupyterlab_config.png "example of editing jupyter_lab_config.py file using nano")

Jupyter Notebook's documentation recommends setting the parameters for `keyfile`, `certfile`, `ip`, `port`, `password`, and `open_browser`, so we're going to set the same parameters within JupyterLab. For reference, here is the example configuration for Jupyter Notebook: 

```python
# Set options for certfile, ip, password, and toggle off
# browser auto-opening
c.NotebookApp.certfile = u'/absolute/path/to/your/certificate/mycert.pem'
c.NotebookApp.keyfile = u'/absolute/path/to/your/certificate/mykey.key'
# Set ip to '*' to bind on all interfaces (ips) for the public server
c.NotebookApp.ip = '*'
c.NotebookApp.password = u'sha1:bcd259ccf...<your hashed password here>'
c.NotebookApp.open_browser = False

# It is a good idea to set a known, fixed port for server access
c.NotebookApp.port = 9999 
```

Since we're installing JupyterLab instead of Jupyter Notebook, the configuration parameters will be a bit different. Most notably, the parameters will begin with `c.ServerApp` rather than `c.NotebookApp`. I found it quickest to search for the parameter names (using `CTRL+W` in `nano`, but syntax may differ depending on your text editor) and then uncomment those lines and set the values. 

Here are snippets from my `jupyter_server_config.py` file after the edits were made: 

```python
## The full path to an SSL/TLS certificate file.
#  Default: ''
c.ServerApp.certfile = u'/home/scraps/.jupyter/jupytercert.pem'

## The full path to a private key file for usage with SSL/TLS.
#  Default: ''
c.ServerApp.keyfile = u'/home/scraps/.jupyter/jupyterkey.key'

## The IP address the Jupyter server will listen on.
#  Default: 'localhost'
c.ServerApp.ip = '*'

## Whether to open in a browser after starting.
#                          The specific browser used is platform dependent and
#                          determined by the python standard library `webbrowser`
#                          module, unless it is overridden using the --browser
#                          (ServerApp.browser) configuration option.
#  Default: False
c.ServerApp.open_browser = False

## The port the server will listen on (env: JUPYTER_PORT).
#  Default: 0
c.ServerApp.port = 9999

## Hashed password to use for web authentication.
#
#                        To generate, type in a python/IPython shell:
#
#                          from jupyter_server.auth import passwd; passwd()
#
#                        The string should be of the form type:salt:hashed-
#  password.
#  Default: ''
c.ServerApp.password = 'argon2:                                          '
```

**Please note: you should input your hashed password and paths to the certificate files that we generated in the previous two steps instead of the example values shown here.**

Once you make these edits, you'll want to save the file and close the text editor. In `nano`, that means pressing `CTRL+O` to save, `enter` to confirm the filename, and then `CTRL+X` to close the text editor.

### Verifying the Configuration
Now that we have finished the basic configuration items, we can start up the server to verify the changes. We can immediately notice two changes: 
1. The port number in the URL is now `9999` (or whichever port number you entered in the configuration file) instead of `8888`
2. The URL should start with `https` instead of `http`
```bash
scraps@pop-os:~/.jupyter$ jupyter lab
[I 2022-12-04 22:00:45.667 ServerApp] jupyterlab | extension was successfully linked.
[I 2022-12-04 22:00:45.676 ServerApp] nbclassic | extension was successfully linked.
[I 2022-12-04 22:00:45.827 ServerApp] notebook_shim | extension was successfully linked.
[I 2022-12-04 22:00:45.843 ServerApp] notebook_shim | extension was successfully loaded.
[I 2022-12-04 22:00:45.844 LabApp] JupyterLab extension loaded from /home/scraps/.local/lib/python3.10/site-packages/jupyterlab
[I 2022-12-04 22:00:45.844 LabApp] JupyterLab application directory is /home/scraps/.local/share/jupyter/lab
[I 2022-12-04 22:00:45.847 ServerApp] jupyterlab | extension was successfully loaded.
[I 2022-12-04 22:00:45.851 ServerApp] nbclassic | extension was successfully loaded.
[I 2022-12-04 22:00:45.851 ServerApp] Serving notebooks from local directory: /home/scraps/.jupyter
[I 2022-12-04 22:00:45.851 ServerApp] Jupyter Server 1.23.3 is running at:
[I 2022-12-04 22:00:45.851 ServerApp] https://localhost:9999/lab
[I 2022-12-04 22:00:45.851 ServerApp]  or https://127.0.0.1:9999/lab
[I 2022-12-04 22:00:45.851 ServerApp] Use Control-C to stop this server and shut down all kernels (twice to skip confirmation).
[W 2022-12-04 22:04:11.774 ServerApp] SSL Error on 9 ('::1', 42318, 0, 0): [SSL: SSLV3_ALERT_CERTIFICATE_UNKNOWN] sslv3 alert certificate unknown (_ssl.c:997)
[W 2022-12-04 22:04:14.883 ServerApp] SSL Error on 9 ('::1', 42334, 0, 0): [SSL: SSLV3_ALERT_CERTIFICATE_UNKNOWN] sslv3 alert certificate unknown (_ssl.c:997)
[I 2022-12-04 22:04:14.896 ServerApp] 302 GET / (::1) 0.58ms
[I 2022-12-04 22:04:14.951 LabApp] 302 GET /lab? (::1) 0.87ms
```
This means that the modifications we made to the port and configuring HTTPS both worked! If you open the URL in the console output (which is `https://localhost:9999/lab` or `https://127.0.0.1:9999/lab` in the example above), then you should see a page like this:

![jupyter lab's login page](/assets/img/posts/2023-11-14_jupyterlab_login_page.png "jupyter lab's login page")

**Note for remotely connected users: opening the URL listed in the console output will only work on the computer that is running the JupyterLab server.** If you are SSH'd into the host computer like I am (or if you want to access your JupyterLab instance from another computer on the same network), then you instead need to enter the IP address of the host computer between the protocol and the port number. You can try running `hostname -I` in the terminal to get the computer's private IP address or look up instructions online for your particular Linux distribution. 

**Note: Depending on your browser and whether you used `openssl` or `lets-encrypt`, you might receive a prompt to accept the self-signed HTTP certificate first. Since we trust this web page, it is okay to click "Advanced" and then "Proceed".**

Since this web page is prompting for a password, it means the password configuration also worked!

Once you log in, you should see the full JupyterLab interface: 

![jupyter lab's landing page](/assets/img/posts/2023-11-14_jupyterlab_landing_page.png "jupyter lab's landing page")

Congratulations, your JupyterLab server is up and running! Check out the next section for recommended customization and next steps. 

## Customizing JupyterLab
That's all you need to do to start using JupyterLab! In general, it's a good idea to go back into the configuration file and review it for any other parameters that might be relevant to your particular installation and use case, but this section goes over a few bonus parameters and configuration options that are recommended.

### Changing the Theme
This first item is pretty minor but super easy. If you prefer to work in dark mode (like I do), then you can easily swap the theme in the JupyterLab GUI by clicking on Settings > Theme > JupyterLab Dark. You can also modify the font size of the code, content, or overall UI from this theme sub-menu. 

### Changing the Startup Directory 
By default, your JupyterLab workspace and all of the associated files will be stored in the directory where you ran the `jupyter lab` startup command (which could be the home directory, the `.jupyter` directory, or any other directory on your machine). [JupyterLab's documentation](https://jupyterlab.readthedocs.io/en/stable/getting_started/starting.html) mentions that you can specify a different directory by using the `--preferred-dir` parameter in the startup command.

If you always want JupyterLab to use the same directory, it might be better to instead change the default startup directory of JupyterLab. This can be set in the same configuration file (`jupyter_server_config.pyjupyter_server_config.py`) that we were editing earlier. The documentation online recommends editing the `notebook_dir` parameter:

```python
## DEPRECATED, use root_dir.
#  Default: ''
# c.ServerApp.notebook_dir = ''
```
There is a note about how `notebook_dir` has been deprecated in JupyterLab and that `root_dir` should be used instead, so that's what we'll use instead. This configuration file is a Python file, so we can even use libraries to build the folder path if desired: 

```python
import os
c.ServerApp.root_dir = os.path.expanduser('~/Code/jupyterlab/')
```
Once you save and exit the configuration file as usual, the workspace files will now be saved in whatever directory you specified above.

### Running in the Background 
It might be useful to start up and run the JupyterLab server as a background process in the terminal by adding `&` to the end of the command.
```bash
scraps@pop-os:~$ jupyter-lab &
```
By running this as a background process, the output of the server will no longer be printed in the terminal. In that case, it might be helpful to redirect the server output (`stdout`) to a logfile by running this startup command instead:
```bash
scraps@pop-os:~$ jupyter-lab &>> ~/Code/jupyterlab/_logs/2022-12-06_jupterlab.log
```
This will redirect all of the server messages to the specified log file (which is a file named `2022-12-06_jupyterlab.log` in the `~/Code/jupyterlab/_logs/` directory in the example above). In summary, the `&` directs the terminal to run that process in the background while the `>>` redirects the output to another location (which is a log file in this case). 

As a quick reminder, the best way to shut down the server might be slightly different when starting the JupyterLab server in the background. You can review the [Stopping the Server](#stopping-the-server) section above to find your preferred way of shutting down the server. One difference to note is that if you press `CTRL+C` once while the terminal output is being redirected to a logfile, the terminal won't show the prompt to confirm shutting down the server. However, your input will still get passed to JupyterLab, so you can type `y` or `yes` and press `enter` to confirm the shut down. It's not generally recommended to send inputs when you cannot see the outputs, but it is technically possible.

## Wrap Up
Congratulations! Your new JupyterLab server is ready to go and you are now self-hosting your Jupyter Notebooks! That's not all though - subscribe to be notified when part two of this project is released. Here is a preview of items we'll cover in part two: 
- Adding helpful JupyterLab extensions, including ones that:
  - Incorporate an AI model right into your JupyterLab interface (OpenAI's ChatGPT is supported)
  - Run a light-weight server to enable code completion, documentation hints, and style guide hints
  - Add a GUI for Git versioning, commits, pushes, pulls, etc. 
  - Show the system's and server's current resource usage (CPU, RAM, etc.) 
- Cover the basics of using JupyterLab's included debugging tool
- How to streamline your workflow with Jupyter's magic commands
- And more! 
If you found this guide helpful, please give it a like, share, and subscribe to be notified of new articles. If you have any questions or suggestions, feel free to [contact me](/workwithme) directly!

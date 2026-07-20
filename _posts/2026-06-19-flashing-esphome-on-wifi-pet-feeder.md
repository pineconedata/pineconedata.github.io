---
layout: post
readtime: true
title: Flashing ESPHome on a WiFi Pet Feeder
description: Learn how to convert a Petlibro Air WiFi Feeder to ESPHome by safely wiring the ESP32 board, entering bootloader mode, backing up the factory firmware, flashing ESPHome, and adding the feeder to Home Assistant.
share-title: "Converting the Petlibro PLAF108 to ESPHome"
share-description: A field-tested guide to flashing ESPHome onto a Petlibro Air WiFi feeder, preserving the factory firmware, and avoiding the bootloader and flashing issues that can trip up first-time conversions.
tags: [ESPHome, Home Assistant, Petlibro, smart home, electronics, automation]
thumbnail-img: /assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/thumbnail.jpg
share-img: /assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/social.png
gh-repo: pineconedata/esphome-petlibro-plaf108
gh-badge: [star, fork, follow]
last-updated: 2026-06-19
sitemap:
  priority: 0.8
---

Today we'll walk through how to convert a Petlibro Air WiFi Feeder to ESPHome, including how to access the ESP32 board, wire it for flashing, enter bootloader mode, back up the factory firmware, flash ESPHome correctly, and add the feeder to Home Assistant.

# What is ESPHome? 
[ESPHome](https://esphome.io) is an open-source project for creating firmware using simple YAML configurations instead of writing embedded code from scratch. It most commonly works with ESP32 and ESP8266 microcontrollers. I have never written firmware and hadn't even heard of ESPHome before starting this project, but it was extremely beginner-friendly.

# Why ESPHome?

The Petlibro PLAF108 is already a WiFi feeder, but the stock firmware is designed around Petlibro's ecosystem. This means that using the pet feeder requires installing Petlibro's app, having a Petlibro account, and maintaining internet access to reach Petlibro's servers. Flashing ESPHome moves the device into a local Home Assistant setup and gives you direct control over the hardware that is already inside the feeder.

For me, the main benefits are:

- Local Home Assistant integration instead of relying on the stock app for day-to-day control
- Direct visibility into feeder sensors, such as wall power, battery connection, food at the chute, and motor rotation
- More flexible automations for feeding, notifications, lights, and diagnostics
- Local over-the-air (OTA) firmware updates after the initial flash
- The ability to rename and model entities based on what the hardware actually does
- A better understanding of the feeder's behavior and failure modes

This approach makes sense if:

- You are comfortable opening the feeder and working with low-voltage electronics
- You want local control through Home Assistant
- You are willing to lose stock cloud/app behavior unless you restore the factory firmware
- You can make and safely store a factory firmware backup before flashing anything
- You are willing to try these steps out on your own hardware, even if it causes problems

This is not the right project for you if:
- you need the feeder to keep working with the Petlibro app
- you are not comfortable with serial flashing (or interested in trying it)
- you do not want to risk causing issues with your hardware

# Safety Notes

Before getting started, there are a few important warnings:

1. Disconnect power before opening the feeder, moving cables, or changing wiring. This means disconnecting the wall power **and** the battery (once you have the feeder open). 
2. Use a regulated 3.3V supply if you power the ESP board directly. Do not feed 5V into the ESP board's 3.3V rail.
3. Do not connect multiple power sources to the board unless you know they are safe to connect together.
4. Keep the USB-to-TTL adapter ground connected to the feeder board ground.
5. Do not connect the USB-to-TTL adapter's 5V pin to the feeder board.
6. Back up the factory firmware before flashing ESPHome.
7. Do not publish your factory firmware dump. It may contain device-specific secrets, cloud credentials, certificates, or other private data.
8. Be careful with motor GPIOs. A bad configuration or exposed web server could damage the feeder motor.
9. These instructions are for the Petlibro PLAF108 hardware I have specifically. Your board revision may differ.

# Prerequisites

You will likely need:

- Petlibro Air WiFi Feeder (model PLAF108)
- USB-to-TTL serial adapter
  - I used [this CH340 adapter](https://www.amazon.com/dp/B00LZV1G6K). 
- Jumper wires or test leads
  - I used three female-to-male and one male-to-male [jumper wires](https://www.amazon.com/dp/B07GD2PGY4).
- A machine capable of running Python tools (I used Ubuntu 24 LTS)
    - Python 3
    - `esptool`
    - `esphome`
        - I used ESPHome version `2025.2.2` , but a newer version with potentially different behavior may be available in the future. If anything in this guide conflicts with ESPHome's official documentation, follow the instructions in ESPHome.
- [Home Assistant](https://www.home-assistant.io/) (or similar hub) with the [ESPHome](https://esphome.io/) integration
- External regulated 3.3V power supply
  - This may not be necessary depending on whether your feeder board receives power properly

These are not strictly required, but are very helpful:

- Multimeter with DC voltage and continuity modes
- A way to take clear photos of the board before disconnecting anything
- Labels or tape for marking cables
- Fine tweezers for disconnecting cables


![IMG_20260715_160145_299.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260715_160145_299.jpg)

# Disassemble the Feeder

Before we can flash new firmware onto the device, we need to disassemble the feeder enough to access the control board. I'd recommend removing the food bowl and all food from the feeder at this point. 

![IMG_20260617_134752_223.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260617_134752_223.jpg)

Unclip and remove the cylindrical top shell from the feeder. Inside, there's a clear plate that the food sits on:

![IMG_20260617_005248_513.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260617_005248_513.jpg)

Rotate the clear plate in the unlock direction (marked on the plate) and carefully lift it out:

![IMG_20260617_005227_017.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260617_005227_017.jpg)

Next, there are four screws on the bottom of the feeder that need to be removed. Flip the feeder over to access these screws. You do not need to remove any rubber feet or covering to access the screws.

Your feeder dimensions may bary, but I needed a screwdriver with a thin, Philips-head shaft that was just over 10 cm long.  Be aware that the screws in my pet feeder were quite soft, so be careful not to strip them. I'd recommend putting these screws in a labelled bag or bowl until it's time for reassembly.

![IMG_20260715_155833_160.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260715_155833_160.jpg)

Next, carefully lift off the bottom cover. You should see something that looks like this: 

![IMG_20260605_161733_799.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_161733_799.jpg)

![IMG_20260605_161741_694.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_161741_694.jpg)

There is a small board visible now: 

![IMG_20260605_155953_588.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_155953_588.jpg)

This board does include the connection for the battery. I'd recommend unplugging the 2-pin battery connector now (the device can still be powered if plugged into the wall): 

![IMG_20260605_163317_893.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_163317_893.jpg)

Unfortunately, this is not the board we are looking for. Follow the wires from the 8-pin connector to the top of the feeder:

![IMG_20260617_004654_985.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260617_004654_985.jpg)

The wires go behind the food chute, so I removed two screws holding it in place:

![IMG_20260617_004525_986.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260617_004525_986.jpg)

Then carefully pulled off the plastic food chute: 

![IMG_20260605_161803_414.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_161803_414.jpg)

Next, there are six plastic clips holding the panel in place. Carefully unclip these clips without damaging them. I found it easiest to use tweezers to unclip both clips on the left side (further from the wires) first. 

![IMG_20260605_163343_336.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_163343_336.jpg)

Pull this panel away from the top shell of the feeder: 

![IMG_20260605_163329_827.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_163329_827.jpg)

![IMG_20260605_163400_051.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_163400_051.jpg)

I'd recommend confirming that your pet feeder has an ESP32 chip before proceeding any further. The chip in my feeder specifically is an Espressif ESP32-C3-WROOM-02. 

![IMG_20260605_163636_790.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260605_163636_790.jpg)

Directly underneath that chip are the `VCC`, `RX`, `TX`, and `GND` pads that we'll be using in the next step. There are also two unlabelled pads (directly to the left of `GND` in the previous image) that will be used for entering programming mode during boot-up.

# Wire the Serial Adapter

Now, we're ready to connect the pet feeder board to the computer using the USB-to-TTL adapter. If your adapter has a bridging pin to select between 5V and 3.3V, ensure it is set to 3.3V. Exact instructions depend on your adapter, but I connected: 

| Wire                 | USB-to-TTL adapter pin | Feeder board pad |
|----------------------|------------------------|------------------|
| Black female to male | GND                    | GND              |
| Blue female-to-male  | TXD                    | RX               |
| White female-to-male | RXD                    | TX               |
| Brown male-to-male   | None                   | unlabelled pads  |

Note: The serial TX/RX lines are crossed: the adapter's `TXD` pin goes to `RX` on the board, and the adapter `RXD` pin connects to `TX` on the board.

![IMG_20260616_232338_094.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260616_232338_094.jpg)

![IMG_20260617_003840_627.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260617_003840_627.jpg)

The wiring at this point is:

```
USB-TTL GND       -> feeder GND
USB-TTL TXD       -> feeder RX
USB-TTL RXD       -> feeder TX
unlabelled pad    -> other unlabeled pad
8-pin cable       -> connected
wall adapter      -> disconnected (until power-on)
battery           -> disconnected, if accessible
```

If your device is hooked up properly, you should be able to check for the serial device by running: 

```bash
ls /dev/ttyUSB*
```

You should see a response containing `dev/ttyUSB0`. If you have a result like `dev/ttyUSB1`, then use that identifier for the commands in the rest of this guide. 

If you do not receive a response: 
1. Check the wiring and contacts between the serial adapter, the feeder board, and your computer 
2. Check the adapter works properly with a loopback test 
3. If you're using a CH340 on Linux, check if the device is being captured by [`brltty`](https://github.com/brltty/brltty). I discovered this by running `dmesg -w` while plugging in the serial adapter to my laptop. 

The wiring is now hooked up properly, and we're ready to back up and flash the firmware. 

# Install ESPHome and esptool

Now that the hardware is ready, we can set up the required software. Create a working directory wherever you would like:

```bash
mkdir -p ~/Downloads/plaf108
cd ~/Downloads/plaf108
```

Install the tools using your preferred Python environment. One simple option is:

```bash
python3 -m pip install --user esphome esptool
```

Verify that both tools are available:

```bash
python3 -m esptool version
python3 -m esphome version
```

If your system has multiple Python environments, make sure you use the same one consistently.

# Power On in Bootloader Mode

Next, we can enter bootloader mode by powering on the feeder while the two unmarked boot pads are bridged (with the brown male-to-male wire connected earlier). Once the device is powered on, you can either remove the wire bridging the two pads or leave it in place. I didn't have any issues with leaving that wire in place during the entire flashing process. The important part is that the unmarked pads are bridged during startup to ensure bootloader mode is entered correctly.

Before powering on, you can run this to check that the adapter is connected to the feeder board properly: 

```bash
python3 -m serial.tools.miniterm /dev/ttyUSB0 115200
```

Once the board is powered on, if it entered the bootloader mode correctly, then the output should look something like this: 

```bash
--- Miniterm on /dev/ttyUSB0  115200,8,N,1 ---
--- Quit: Ctrl+] | Menu: Ctrl+T | Help: Ctrl+T followed by Ctrl+H ---
ESP-ROM:esp32c3-api1-20210207
Build:Feb  7 2021
rst:0x1 (POWERON),boot:0x5 (DOWNLOAD(USB/UART0/1))
waiting for download
```

If you see normal factory firmware logs instead, the boot pads were not bridged correctly or were not held at the right time during power-on. Here are what the normal factory firmware logs looked like for me: 

```bash
--- Miniterm on /dev/ttyUSB0  115200,8,N,1 ---
--- Quit: Ctrl+] | Menu: Ctrl+T | Help: Ctrl+T followed by Ctrl+H ---
ESP-ROM:esp32c3-api1-20210207
Build:Feb  7 2021
rst:0x1 (POWERON),boot:0xd (SPI_FAST_FLASH_BOOT)
SPIWP:0xee
mode:DIO, clock div:2
load:0x3fcd6100,len:0x38c
load:0x403ce000,len:0x6ac
load:0x403d0000,len:0x24e4
entry 0x403ce000
```

Once you see the `waiting for download` response, you're ready to move on to the next step. Ensure you exit the tool with `CTRL+]` first.

<div class="email-subscription-container"></div>

# Confirm esptool Communication

With the board in bootloader mode, run:

```bash
python3 -m esptool --port /dev/ttyUSB0 chip_id
```

A successful result should identify an ESP32-C3 and print the chip information and MAC address: 

```bash
esptool.py v4.7.0
Serial port /dev/ttyUSB0
Connecting....
Detecting chip type... ESP32-C3
Chip is ESP32-C3 (QFN32) (revision v0.4)
Features: WiFi, BLE
Crystal is 40MHz
MAC:
```

If the board prints `SPI_DOWNLOAD_BOOT` but `esptool` fails with this message:

```
Failed to connect to ESP32-C3: No serial data received
```

then the computer is sending sync packets but not receiving valid responses from the bootloader. You should check all wiring is making proper contact with the pads and ensure you have the proper type of USB-to-Serial adapter. 

## A Note about Power Source

I continually got the above `No serial data received` message and eventually tracked it down to an issue with the power source. Ideally, you can power on your feeder in bootloader mode by simply plugging in the wall adapter.

Unfortunately, that did not work for me. For some reason, while powering the board from the wall adapter, the chip was at 3.8V instead of at 3.3V. I tried a few fixes, but ultimately the only thing that worked was powering the feeder board directly from external 3.3V power supply while the feeder's internal 8-pin cable, wall adapter, and battery were disconnected.

Note: When I flash the other pet feeder, I will check if the adapter can supply enough power if the feeder's `VCC` pad is wired to the adapter's 3.3V pin. If that works, it would eliminate the need for an external power supply. 

The wiring with an external power supply looked like this: 

```
External 3.3V +   -> feeder VCC
External GND      -> feeder GND
USB-TTL GND       -> feeder GND
USB-TTL TXD       -> feeder RX
USB-TTL RXD       -> feeder TX
unlabelled pad    -> other unlabelled pad
8-pin cable       -> disconnected
wall adapter      -> disconnected 
battery           -> disconnected
```


![IMG_20260616_232300_011.jpg](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/IMG_20260616_232300_011.jpg)


Ensure you do not connect the USB-to-TTL adapter's 5V pin to anything. Your adapter should be set to 3.3V, not 5V. If you use an external 3.3V power supply, do not connect the USB-to-TTL adapter to the feeder's VCC pad. 

Once you've successfully confirmed esptool can communicate with the board, you're ready to back up the factory firmware. 

# Back Up the Factory Firmware

Before flashing ESPHome, it's recommended to back up the factory firmware. This ensures that if anything goes wrong with the ESPHome firmware (or you decide to reset it to the stock firmware in the future), then you have a backup to restore from.

You can name the backup file anything you want, but I went with `petlibro_factory_dump.bin`: 

```bash
python3 -m esptool --chip esp32c3 --port /dev/ttyUSB0 read_flash 0x0 0x400000 petlibro_factory_dump.bin
```

Creating the backup on my device took about 7 minutes and I left the unlabeled pads bridged the entire time. If it works successfully, you should see an output that looks like this: 

```bash
esptool.py v4.7.0
Serial port /dev/ttyUSB0
Connecting....
Chip is ESP32-C3 (QFN32) (revision v0.4)
Features: WiFi, BLE
Crystal is 40MHz
MAC: [REDACTED]
Uploading stub...
Running stub...
Stub running...
4194304 (100 %)
4194304 (100 %)
Read 4194304 bytes at 0x00000000 in 380.0 seconds (88.3 kbit/s)...
Staying in bootloader.
```

Verify the backup file actually exists on your disk and is around 4 MB. 

Note: Keep this file private. Do not upload it to GitHub, issue trackers, forums, or public chats.

# Create the ESPHome Configuration

There are a lot of configuration options for ESPHome, but I'd recommend starting with the configuration files in [pineconedata/esphome-petlibro-plaf108](https://github.com/pineconedata/esphome-petlibro-plaf108) or [taylorfinnell/petlibro-esphome](https://github.com/taylorfinnell/petlibro-esphome).

You need to replace the placeholder WiFi ssid and password information with your own. You can do this by editing `plaf108.yaml` directly: 

```yaml
wifi:
  ssid: "YOUR_WIFI_NETWORK_NAME"
  password: "YOUR_WIFI_NETWORK_PASSWORD"
```

Or you can store your credentials in a separate `secrets.yaml` file. In that case, edit plaf108.yaml to include: 

```yaml
wifi:
  ssid: !secret wifi_ssid
  password: !secret wifi_password
```

And then create a separate `secrets.yaml` file (in the same directory) that contains: 
```yaml
wifi_ssid: "YOUR_WIFI_NETWORK_NAME"
wifi_password: "YOUR_WIFI_NETWORK_PASSWORD"
```

You can also make any other configuration changes that you'd like at this time, but the WiFi credentials are the only necessary edit. You will be able to update the firmware over-the-air (without disassembling it) after ESPHome is flashed to the device. 

# Compile the Firmware

From the directory containing `plaf108.yaml` (and `secrets.yaml`, if you're using it), run:

```bash
python3 -m esphome compile plaf108.yaml
```

A successful compilation should take just over a minute and have an output like: 

```bash
INFO ESPHome 2025.2.2
INFO Reading configuration plaf108.yaml...
WARNING GPIO2 is a strapping PIN and should only be used for I/O with care.
Attaching external pullup/down resistors to strapping pins can cause unexpected failures.
See https://esphome.io/guides/faq.html#why-am-i-getting-a-warning-about-strapping-pins
WARNING GPIO8 is a strapping PIN and should only be used for I/O with care.
Attaching external pullup/down resistors to strapping pins can cause unexpected failures.
See https://esphome.io/guides/faq.html#why-am-i-getting-a-warning-about-strapping-pins
WARNING GPIO9 is a strapping PIN and should only be used for I/O with care.
Attaching external pullup/down resistors to strapping pins can cause unexpected failures.
See https://esphome.io/guides/faq.html#why-am-i-getting-a-warning-about-strapping-pins
INFO Generating C++ source...
INFO Core config, version or integrations changed, cleaning build files...
INFO Compiling app...
********************************************************************************
If you like PlatformIO, please:
- star it on GitHub > https://github.com/platformio/platformio-core
- follow us on LinkedIn to stay up-to-date on the latest project news > https://www.linkedin.com/company/platformio/
- try PlatformIO IDE for embedded development > https://platformio.org/platformio-ide
********************************************************************************

Processing plaf108-a (board: esp32-c3-devkitm-1; framework: arduino; platform: platformio/espressif32@5.4.0)
--------------------------------------------------------------------------------
Platform Manager: Installing platformio/espressif32 @ 5.4.0
INFO Installing platformio/espressif32 @ 5.4.0
Downloading  [####################################]  100%
Unpacking  [####################################]  100%
Platform Manager: espressif32@5.4.0 has been installed!

...
Additional downloading, unpacking, and compiling lines...
...

Building .pioenvs/plaf108-a/firmware.bin
Creating esp32c3 image...
Successfully created esp32c3 image.
esp32_create_combined_bin([".pioenvs/plaf108-a/firmware.bin"], [".pioenvs/plaf108-a/firmware.elf"])
Wrote 0xe1530 bytes to file /home/user/Downloads/plaf108/.esphome/build/plaf108-a/.pioenvs/plaf108-a/firmware.factory.bin, ready to flash to offset 0x0
esp32_copy_ota_bin([".pioenvs/plaf108-a/firmware.bin"], [".pioenvs/plaf108-a/firmware.elf"])
======================================================== [SUCCESS] Took 86.34 seconds ========================================================
```

If there are any YAML or compile errors, fix those before flashing. Once your firmware is successfully compiled, you can flash it to the device. 

# Flash ESPHome

Note: This step actually replaces the firmware on the device. Once you run the commands in this section, your device will no longer work with Petlibro's app and will instead be using ESPHome. Double-check that the factory firmware is safely backed up and that you are ready to proceed. 

To flash ESPHome to the device, run (from the folder containing your `plaf108.yaml` file, or provide the full filepath):

```bash
python3 -m esphome upload plaf108.yaml --device /dev/ttyUSB0
```

During a successful upload, the logs should show several segments, including:

```
bootloader at 0x00000000
partition table at 0x00008000
OTA data at 0x0000e000
application at 0x00010000
```

Once you receive those messages, your pet feeder's firmware has been replaced!

# Reconnect the Feeder and Boot Normally

After ESPHome is flashed successfully, power off the device by unplugging the wall adapter or powering off the external power supply. You can unplug the USB-to-TTL adapter and disconnect all wires from the feeder board. 

I would recommend testing that the device boots properly before fully reassembling it. You can power on the device by plugging in the wall adapter (and reconnecting the internal 8-pin cable if you previously unplugged it). 

If you entered WiFi credentials in the `plaf108.yaml` or `secrets.yaml` files, then the device should automatically join your WiFi network. You can confirm that it is reachable by pinging the device name (or using the IP address assigned by your router): 

```bash
ping plaf108-a.local
```

If the device is online, you should see results like this: 

```bash
64 bytes from plaf108-a (IP_ADDRESS): icmp_seq=1 ttl=254 time=93.1 ms
64 bytes from plaf108-a (IP_ADDRESS): icmp_seq=2 ttl=254 time=231 ms
64 bytes from plaf108-a (IP_ADDRESS): icmp_seq=3 ttl=254 time=36.0 ms
```

If that works, you can reassemble the device and start configuring it in Home Assistant! To reassemble it, reverse the disassembly process: 

1. Remove all power sources from the device (wall adapter, battery, external power supply)
2. Ensure all jumper wires and the adapter are unplugged
3. Reconnect the 8-pin cable, if previously disconnected
4. Reconnect the 2-pin battery cable
5. Clip the feeder board back into the top shell
6. Place the food chute (carefully not crushing or catching any wires) and screw it in place
7. Slide the shell with the food chute back over the base of the feeder 
8. Reattach the clear food plate (aligning the central gear with the motor shaft)
9. Slide on and clip down the upper cylindrical food-containment shell
10. Attach the food hopper and dish
11. Refill with food and replace the button-lid
12. Connect to wall power


# Add the Feeder to Home Assistant

In Home Assistant, ensure you have the [official ESPHome integration](https://www.home-assistant.io/integrations/esphome/). You can add it with:

```
Settings -> Devices & services -> Add integration -> ESPHome
```

Once you have ESPHome, you can add the device specifically with: 

```
Settings -> Devices & services -> Devices -> Add Device -> ESPHome
```

You should see a screen like this (the `Host` field will likely be empty): 

![Screenshot from 2026-06-17 00-23-46.png](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/Screenshot%20from%202026-06-17%2000-23-46.png)

Enter the feeder's hostname or IP address into the `Host` field and click Submit. 

![Screenshot from 2026-06-17 00-38-05.png](/assets/img/posts/2026-06-19-flashing-esphome-on-wifi-pet-feeder/Screenshot%20from%202026-06-17%2000-38-05.png)

If Home Assistant has any issues connecting to the device, check that your `plaf108.yaml` file contains the required `api` entry.

Once connected, Home Assistant should show the device's status, sensors, and controls.

# Wrapping Up

At this point, the firmware on the pet feeder has been replaced. It started as a perfectly usable WiFi feeder, but required Petlibro's app, account system, and cloud services to operate. Flashing ESPHome lets you use the device entirely locally and use the native Home Assistant integration.

Now, all that's left is configuration: deciding how to use the feeder's sensors and controls, naming entities clearly, and building safe Home Assistant automations around the hardware. This guide was focused on the flashing process itself, but I'm planning to publish a future article around the Home Assistant configuration. 

<div class="email-subscription-container"></div>
<div id="sources"></div>
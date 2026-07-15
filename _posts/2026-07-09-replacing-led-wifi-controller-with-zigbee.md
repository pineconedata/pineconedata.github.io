---
layout: post
readtime: true
title: Replacing an LED Strip WiFi Controller with Zigbee
description: Learn how to replace a WiFi LED strip controller with a compatible Zigbee controller by identifying the LED strip type, mapping the wires, connecting the new controller, and pairing it with Home Assistant or another Zigbee hub.
share-title: "Replacing an LED Strip WiFi Controller with Zigbee"
share-description: Learn how to replace a WiFi LED strip controller with a compatible Zigbee controller without removing an already-mounted LED strip.
tags: [Zigbee, Home Assistant, electronics, smart home, automation]
thumbnail-img: /assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/thumbnail.png
share-img: /assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/social.png
last-updated: 2026-07-09
sitemap:
  priority: 0.8
---

Today we'll walk through how to replace an existing LED strip controller with a compatible Zigbee controller, including how to identify the strip wiring, connect the new controller, and pair it with Home Assistant or another Zigbee hub.

# Zigbee instead of WiFi

In many LED strips, the strip itself works perfectly fine, but the controller is not ideal. For example, the original WiFi controller on my LED strip required a third-party account for setup, and I'd rather keep all lighting devices on a single account where possible. Plus, Zigbee is often a better fit for smart lighting because it keeps devices off your WiFi network, supports a local low-power mesh, and usually avoids relying on a manufacturer's cloud service for day-to-day control. 

Replacing just the controller can also be much easier than replacing the whole strip. In my case, the LED strip was already mounted inside a closet, so removing it would have meant pulling down an otherwise-working installation. By swapping the WiFi controller for a compatible Zigbee controller, I was able to keep the strip physically in place and move it into my preferred smart home setup.

This approach makes sense if:

- Your LED strip still works
- Your existing controller is inconvenient, unsupported, cloud-dependent, or unreliable
- You prefer Zigbee over WiFi for smart lighting
- The strip is already installed somewhere you don't want to disturb
- You can identify the strip type, voltage, and wiring well enough to connect a compatible replacement controller

# Safety Notes

Before getting started, let's cover a few safety precautions:

1. Disconnect the power supply of the LED strip from the wall before touching, cutting, stripping, or moving any wiring. 
    1. It's recommended to use a multimeter to confirm no power is going to the system.
2. Do not let bare LED wires touch each other while the system is powered. 
3. If you are not comfortable doing basic electronics work, proceed with extreme caution or consider replacing your LED strip entirely. 
4. These instructions are for replacing a low-voltage controller for a basic LED strip. They're not suitable for high-voltage systems, complex strips, or any system with large capacitors.
5. Ensure you are complying with all local electrical code requirements and safety instructions.


# Getting Started

## Identify the LED Strip Type

Before purchasing or installing a new controller, let's go through how to identify what kind of LED strip you have. The wiring and controller type depend on whether the strip is analog RGB, RGBW, RGB+CCT, tunable white, single-color, or addressable/digital.

First, look at the strip itself, especially near the cut points, connector, or solder pads. Most LED strips have small printed labels next to the copper pads.

Common labels include: 

1. The voltage: `5V`, `12V`, `24V`
2. The RGB: `R`, `G`, `B`
3. The white: `C`, `W` 

If your RGB strip is single-color, it may have only two markings (such as `+` and `-` or `V+` and `V-`). 

Here are some common LED strip types and their typical terminals: 

| LED Strip Type            | LED Strip Markings                |
|---------------------------|-----------------------------------|
| Single-color strip        | `-`  and `+`  or `V-` and `V+`    |
| Tunable white (CCT) strip | `+ C W` or `V+ CW WW`             |
| RGB strip                 | `+ R G B` or `V+ R G B`           |
| RGBW strip                | `+ R G B W` or `V+ R G B W`       |
| RGB+CCT strip             | `+ R G B C W` or `V+ R G B CW WW` |
| Addressable strip         | `+ DI BI GND` or data labels      |


<div class="alert alert-info" role="note">
    <strong>Note:</strong> This guide only covers the first five types of LED strips. If you have an addressable LED strip (each LED can be controlled independently), then you should use a controller specifically designed for the strip and may need alternate instructions.
</div>

If the strip has no visible labels (such as if it is coated, installed in a channel, or hard to reach), then you can also try: 

- Checking for markings on the old controller (including on output terminals or the PCB)
- Searching for the model number printed on the strip, controller, or power supply
- Looking at the product page where you originally purchased the LED strip, if available

Let's take a look at the LED strip that I have: 

![IMG_20260710_091117_800.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260710_091117_800.jpg)

The number of terminals matters because each strip type needs a controller with matching output channels. For example, an RGB strip needs `V+`, `R`, `G`, and `B`, while an RGB+CCT strip also needs separate white channels such as `C` and `W`. The markings on this LED strip say `+12V G R B` so this is an RGB strip. Most of this guide walks through replacing the controller for this strip specifically, but I'll note wiring differences for other common strip types where possible.

## Acquire a Compatible Controller

Once you know the details of your LED strip (type, channels, and voltage) identified in the previous step, choose a compatible controller. For the new controller, check compatibility:

- The power supply voltage matches the LED strip voltage, and the controller supports that voltage
    - Common LED strip voltages are 12V, 24V, or 5V.
- The output terminals fully support the terminals on your LED strip
    - This means there should be one output terminal on the controller for each LED strip marking identified in the previous step. For my LED strip, this is `+ R G B` .
- It is rated for at least as much current as the LED strip can draw (check the power supply as well)

If your strip has additional channels, such as `W`, `C`, `CW`, or `WW`, make sure the replacement controller has matching output terminals for those channels. In my case, I purchased [this controller](https://www.amazon.com/dp/B0F83MNQMS) that supports RGB+CCT strips and I only used the RGB portion (the `C` and `W` terminals on my new controller will be unused). 

# Prerequisites

Before getting started, you'll likely need: 

- Existing RGB LED strip and controller
- New Zigbee LED controller
- Compatible DC power supply (you may or may not need to replace this, depending on your new controller)
- Small/precision screwdriver

These items aren't strictly necessary, but are really nice to have: 

- Multimeter with DC voltage and continuity modes
- Wire strippers or sharp knife
- Electrical tape, heat shrink, wire connectors, or another safe way to insulate and terminate wires
- Labels or tape for labeling wires

# Disconnect the Current Controller

## Document Current State

Before actually disconnecting anything, I'd recommend taking a few photos of: 

- Power input wires
- LED output connector
- Markings on the LED strip
- Wire colors and positions (if visible)

Most controllers have two main wiring areas: the power input and the LED output. The power input typically includes a red and a black wire carrying input power. The LED output typically includes multiple colored wires carrying power and color channels to the LED strip. 

## Unplug the Power

Unplug your power supply or wall wart that powers the current controller and LED strip from the wall. 

## Disconnect the Controller

Disconnecting the controller from the LED strip depends on your exact setup. You might be able to unplug a connector from the end of the LED strip. Unfortunately, the end of my LED strip is wrapped, so there is no simple connector to unplug. 

![IMG_20260709_105834_429.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_105834_429.jpg)

If you cannot unplug the LED strip from the controller, then you'll need to disassemble the controller. Disassembling the controller depends on what exact controller you currently have. For my controller, there are screws on the back of the housing and then the PCB pops out of place. These screws were underneath foam mounting tape. 

![IMG_20260710_090908_767.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260710_090908_767.jpg)

Once the cover is removed, there are often additional screws or plastic clips holding the PCB in place. Unscrew or unclip the PCB and lift it out of its housing. 

![IMG_20260709_111026_502.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_111026_502.jpg)

![IMG_20260709_111030_933.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_111030_933.jpg)

![IMG_20260709_111118_873.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_111118_873.jpg)

![IMG_20260709_111126_964.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_111126_964.jpg)

Now that the controller PCB is removed from the housing, check it for markings to identify the purpose of each wire. On the backside of my controller, there are some markings for `GND`  and `G R B`. 

![IMG_20260709_111720_148.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_111720_148.jpg)

In this case, the wire terminals do not clearly line up with the labels on the back of the PCB. You need to be absolutely certain which wires carry the input and output voltage, as wiring them incorrectly can damage your components. 

## Identify the Wire Terminals

A digital multimeter is extremely useful for identifying each wire. My multimeter was set to continuity mode. Some multimeter devices combine continuity and diode-test modes, but the goal here is simply to identify which wire connects to each LED strip terminal.

Exact instructions depend on your multimeter and strip. For my multimeter, I connected the black probe to one wire terminal near the controller. Then I touched the red probe to each of the four terminals at the end of the LED strip to identify the corresponding wire.

![IMG_20260710_091200_238.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260710_091200_238.jpg)

For my controller, the black and red pair of wires carry the power from the power supply to the controller. The group of four wires going to the LED strip needed to be identified using the multimeter. Your set-up may differ, but my LED strip specifically mapped to: 

| Controller wire color | LED terminal marking | LED terminal meaning |
|-----------------------|----------------------|----------------------|
| green                 | `+12V`               | 12V power input      |
| brown                 | `G`                  | green color channel  |
| yellow                | `R`                  | red color channel    |
| red                   | `B`                  | blue color channel   |

For other analog strip types, the process is the same: identify the shared power wire first, then map each remaining channel wire to its matching terminal. An RGBW strip will have an additional `W` channel, and an RGB+CCT strip will usually have two white channels, often labeled `C`/`W` or `CW`/`WW`. If the strip has `DI`, `DATA`, or `CLK`, it is likely addressable/digital and needs a different controller.

<div class="alert alert-info" role="note">
    <strong>Note:</strong> You cannot rely on wire color alone. The markings on the strip, old controller, or connector are more reliable than the insulation color. For example, you might assume the red wire was the power input or the red color channel, but in this case it controls the blue color channel. 
</div>

## Disconnect the Controller

Now that we've identified the wires, we can disconnect the old controller from the LED strip. If there's a convenient adapter to unplug, I'd recommend you do that. 

![IMG_20260709_133239_344.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_133239_344.jpg)

This adapter unplugged the controller PCB from the LED strip. However, I didn't have an adapter that exactly matched the old controller, so I cut the connection entirely:

![IMG_20260709_141636_145.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_141636_145.jpg)

The outer wire shielding was cut back and removed to expose the four wires needed to connect the new controller. 

# Connect the New Controller

## Strip the Wires

If you have matching adapters, skip this step and simply plug your new adapter into the old adapter. If not, then you should use wire strippers or a sharp knife to carefully strip the wire insulation without damaging the wires. It's recommended to strip the wires to the length specified by your controller (for example, my controller's manual specified 9-10mm of exposed wire). 

![IMG_20260709_142506_348.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_142506_348.jpg)

## Wire the New Controller

With each of these wires stripped, you can now connect them to your new controller. This step depends on your exact controller, but common connection methods include: 

- terminal holes where the wires are inserted and secured with screws or pressure plates
- wires looped around terminal posts and screwed in
- pigtail connectors where terminal wires are wrapped with the LED strip wires

### Connect LED Strip Wires

Connect each of your LED strip wires to the new controller using the terminal labels identified in the previous step. For my controller, I connected the green wire to the `V+` terminal, the brown wire to the `G` terminal, the yellow wire to the `R` terminal, and the red wire to the `B` terminal. The `C` and `W` terminals on my controller are unused since my LED strip is RGB and not RGB+CCT. 

![image.png](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/image.png)

![IMG_20260709_144206_476.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_144206_476.jpg)

For other analog strip types, the same general idea applies: connect the shared positive wire to `V+`, then connect each channel wire to the matching controller terminal. For example, an RGBW strip would also use `W`, and an RGB+CCT strip would use both white channels, usually labeled `C`/`W` or `CW`/`WW`. If your strip is addressable or digital and has a data terminal such as `DI`, `DATA`, or `CLK`, do not wire it to this style of analog controller.

### Connect Power Supply Wires

My new controller accepts the same barrel plug and is compatible with the same power supply as my old controller, so I didn't need to do any additional wiring for power input. However, if you do need to wire the power input, then follow the same process as the LED strip wires. Ensure you connect the controller's positive input terminal, usually labeled `V+` or `+`, to the power supply positive wire, and the negative input terminal, usually labeled `V-`, `-`, or `GND`, to the power supply negative wire.

## Verify Connection

Take a moment to verify all wiring before moving on. Check that: 

- Each wire is secure in its terminal (often with a gentle pull/tug between the wire and the controller)
- There are no loose wire strands bridging terminals
- `V+` is connected to the proper terminal
- The power supply voltage matches the voltage of the controller and the LED strip
- There are no exposed wires (everything should be securely in a terminal or covered with electrical tape/heat shrink)

## Power On the System

After verifying the wiring and connections are correct, plug in the power supply. Your LED strip may flash on, start blinking, or stay off. If there is a physical button on the controller, you should test it now to ensure it properly turns the strip on and off. 

## Pair with Zigbee Hub

Follow the instructions for your exact controller to pair it with your Zigbee hub. My controller automatically entered pairing mode when it was first powered on, so I opened [Home Assistant](https://www.home-assistant.io/) and added the new device. 

If you use Home Assistant, exact instructions are slightly different depending on if you use the [built-in Zigbee integration](https://www.home-assistant.io/integrations/zha/) or if you use [Zibee2MQTT](https://www.zigbee2mqtt.io/guide/usage/pairing_devices.html). 

## Test the Functionality

Use your new controller's remote or Zigbee hub app to test your LED strip. Ensure the strip turns on and off properly, and test each color. If your strip has white channels, test those separately as well.

![IMG_20260709_145322_982.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_145322_982.jpg)

![IMG_20260709_145319_156.jpg](/assets/img/posts/2026-07-09-replacing-led-wifi-controller-with-zigbee/IMG_20260709_145319_156.jpg)

# Troubleshooting

Hopefully everything works properly the first time. If not, here are a few common issues and troubleshooting steps. 

## Strip does not turn on

If the strip does not turn on: 

- Unplug power
- Verify the power input polarity
- Verify `V+`  wiring on the LED strip
- Verify that the controller supports the strip voltage
- Verify that the controller is configured for RGB mode (if it has mode settings)

## Only one color works

If only one color works, check for: 

- loose wires in the controller terminals
- broken conductors
- damage to the LED strip
- wires that are accidentally shorted (such as the yellow and brown wires accidentally touching where the insulation was stripped/cut)

If you are using an RGBW or RGB+CCT strip, also check that the white-channel wires are connected to the correct terminals and that the controller is configured for the correct strip type.

## Colors look wrong

If the colors look wrong, then you might have swapped the color channels of the LED strip. For example: 

| Requested Color | Actual Strip Color | Recommended Fix  |
|-----------------|--------------------|------------------|
| Red             | Green              | Swap `R` and `G` |
| Red             | Blue               | Swap `R` and `B` |
| Green           | Blue               | Swap `G` and `B` |

If this happens, then you should power off the strip, disconnect the power supply from the wall, and rewire the color channels. **Do not** rewire the `V+` wire. 

If your strip has white channels and the white balance does not look correct, check if warm white and cool white are reversed. Warm white is usually labeled `W` or `WW` and cool white is usually labeled `C` or `CW`. If warm white and cool white are reversed, power off the system and swap only those two white-channel wires.

## Device does not pair with hub

If you have issues pairing your device, check the instructions for your Zigbee hub and controller. Common fixes include moving the controller closer to the Zigbee coordinator, factory resetting the controller, reopening pairing mode or permit join, and confirming that the controller is Zigbee rather than WiFi, Bluetooth, or RF-only.

# Wrapping Up

Replacing the controller ended up being a much smaller project than replacing the entire LED strip. The entire process took me about an hour and a half from unplugging the LED strip to controlling it in the Home Assistant app. The important parts were identifying the strip type, confirming the voltage, and mapping each wire before connecting anything to the new controller.

As with most small electronics projects, the actual wiring is straightforward once the labels are clear. The main thing is to avoid relying on wire color alone: identify `V+`, map the color channels, double-check the voltage, and only then power everything back on.

In my case, the existing strip was already mounted inside a closet and worked perfectly well. Swapping the WiFi controller for a Zigbee controller let me keep the installation in place while moving the light into my preferred smart home setup. After pairing it with Home Assistant, the strip behaves like any other Zigbee light. It no longer depends on the original WiFi controller or requires a third-party account.
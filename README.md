<div align="center">

# wordsearch-bot

**A hands-off auto-solver for *Word Search Explorer*, on Android and iPhone.**

It looks at the phone's screen, reads the letter board, finds the words, and swipes them in by itself.
It closes popups, recovers when the game gets stuck, and can play thousands of levels without you touching anything.

[![Release](https://img.shields.io/github/v/release/NikoMyBelovedd/wordsearch-bot?color=orange)](https://github.com/NikoMyBelovedd/wordsearch-bot/releases/latest)
[![CI](https://github.com/NikoMyBelovedd/wordsearch-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/NikoMyBelovedd/wordsearch-bot/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Platforms](https://img.shields.io/badge/runs%20on-Windows%20%7C%20Linux%20%7C%20macOS-555)
![Devices](https://img.shields.io/badge/plays%20on-Android%20%7C%20iOS-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## Contents

1. [What this is (in plain words)](#what-this-is-in-plain-words)
2. [What you need](#what-you-need)
3. [Step 1: open a terminal](#step-1-open-a-terminal)
4. [Step 2: install the tools](#step-2-install-the-tools)
5. [Step 3: download the bot](#step-3-download-the-bot)
6. [Step 4: prepare your phone](#step-4-prepare-your-phone) ([Android](#android-phone-or-emulator) · [iPhone](#iphone))
7. [Step 5: start the bot](#step-5-start-the-bot)
8. [Using the bot's screens](#using-the-bots-screens)
9. [Stopping, pausing and resuming](#stopping-pausing-and-resuming) · [Updating](#updating-to-a-new-version)
10. [Troubleshooting](#troubleshooting)
11. [FAQ](#faq)
12. [Glossary](#glossary)
13. [Advanced: command-line options](#advanced-command-line-options)
14. [Advanced: teaching the bot a new popup](#advanced-teaching-the-bot-a-new-popup)
15. [For developers: how it works](#for-developers-how-it-works)

---

## What this is (in plain words)

*Word Search Explorer* is a phone game: find hidden words in a grid of letters by swiping across them.
This bot plays it for you. It runs on your **computer**, and your **phone is plugged into the computer by USB cable**.
Over and over, the bot:

1. takes a picture of the phone's screen,
2. finds the letter grid and reads every letter,
3. looks up every word hidden in the grid (it knows about 200,000 words),
4. swipes each word on the phone, like a finger would,
5. taps "Next level" and does it again.

It also closes popups (rewards, tutorials, invites) by itself, never spends your coins or hints, never taps ads, and restarts the game if it ever freezes.

You don't need to know how to code. You'll copy and paste a few commands, and this guide says exactly which ones and where.

## What you need

- **A computer** running Windows 10/11, macOS, or Linux.
- **A USB cable** that carries data, not just charging. The one that came with your phone is usually fine.
- **One of these phones**, with *Word Search Explorer* installed from the Play Store or App Store:
  - **An Android phone**, any recent model. An Android emulator on your computer works too.
  - **An iPhone on iOS 27 or newer.** Tested on the **iPhone SE (3rd generation)** and the **iPhone 17**. Other Face ID iPhones use the iPhone 17 layout and should work too (see the [FAQ](#faq)).
- About **20 minutes** for the first setup. After that, starting the bot takes a few seconds.

> **Before you start:** open *Word Search Explorer* on the phone and play the first few tutorial levels by hand, so the game's intro is out of the way.

---

## Step 1: open a terminal

A **terminal** is a window where you type commands. Every command in this guide is shown in a grey box. Copy it, paste it into the terminal, and press **Enter**.

| Your computer | How to open a terminal |
|---|---|
| **Windows** | Press the **Windows key**, type `PowerShell`, and press **Enter**. |
| **macOS** | Press **Cmd + Space**, type `Terminal`, and press **Enter**. |
| **Linux** | Press **Ctrl + Alt + T**, or find *Terminal* in your app menu. |

> **Pasting:** in PowerShell, right-click pastes. On macOS use **Cmd + V**. In most Linux terminals use **Ctrl + Shift + V**.

## Step 2: install the tools

The bot needs three free programs:

- **uv** runs the bot and sets up Python for you. You don't need to install Python yourself.
- **Tesseract** is a letter-recognition program. It helps the bot learn the game's font.
- **adb** talks to Android phones. Skip it if you only use an iPhone.

<details open>
<summary><b>Windows</b></summary>

Paste these into PowerShell one at a time. Each one can take a minute. If Windows asks *"Do you want to allow this app to make changes?"*, click **Yes**.

```powershell
winget install --id astral-sh.uv -e
winget install --id UB-Mannheim.TesseractOCR -e
winget install --id Google.PlatformTools -e
```

The last line (adb) is for Android only.

**For an iPhone,** you also need Apple's USB driver: install the free **[Apple Devices](https://apps.microsoft.com/detail/9np83lwlpz9k)** app from the Microsoft Store (or iTunes).

**Then close PowerShell and open a new one**, so it finds the programs you just installed.

<sub>If `winget` isn't found, update **App Installer** from the Microsoft Store. You can also use the installers from the [uv](https://docs.astral.sh/uv/getting-started/installation/), [Tesseract](https://github.com/UB-Mannheim/tesseract/wiki) and [platform-tools](https://developer.android.com/tools/releases/platform-tools) websites.</sub>

</details>

<details>
<summary><b>macOS</b></summary>

First install **Homebrew**, a free installer for Mac programs. Paste this into Terminal and follow what it prints. It may ask for your Mac password; the password doesn't show as you type, which is normal.

```bash
/bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"
```

Then:

```bash
brew install uv tesseract
brew install android-platform-tools    # Android only
```

The iPhone USB driver is already built into macOS.

</details>

<details>
<summary><b>Linux</b></summary>

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then Tesseract and adb from your package manager:

| Distro | Command |
|---|---|
| Ubuntu / Debian / Mint | `sudo apt install tesseract-ocr adb usbmuxd` |
| Fedora | `sudo dnf install tesseract android-tools usbmuxd` |
| Arch | `sudo pacman -S tesseract tesseract-data-eng android-tools usbmuxd` |

`usbmuxd` is the iPhone USB service. Close the terminal and open a new one afterwards.

</details>

**Check that it worked.** In a new terminal, run `uv --version`. You should see a version number, like `uv 0.9.x`. If you see "not found" or "not recognized", close the terminal, open a new one, and try again.

## Step 3: download the bot

**Easiest way (no git needed):**

1. Download **[wordsearch-bot.zip](https://github.com/NikoMyBelovedd/wordsearch-bot/releases/latest/download/wordsearch-bot.zip)** (the latest release; older versions and what changed are on the [Releases page](https://github.com/NikoMyBelovedd/wordsearch-bot/releases)).
2. Unzip it (right-click the file → *Extract All* on Windows; double-click on a Mac).
3. Move the extracted **`wordsearch-bot`** folder somewhere easy to find, like your **Documents** folder.

<details>
<summary>Or with git, if you have it</summary>

```bash
git clone https://github.com/NikoMyBelovedd/wordsearch-bot.git
```

</details>

**Now point the terminal at that folder.** Type `cd ` (the letters c, d and a space), then **drag the folder from your file manager into the terminal window**, and press **Enter**. The folder's path appears on its own. For example:

```bash
cd ~/Documents/wordsearch-bot
```

**Download everything the bot uses** (about 300 MB, one time only):

```bash
uv sync
```

When it finishes without a red error, the bot is installed.

## Step 4: prepare your phone

### Android (phone or emulator)

1. **Turn on Developer options.** Open **Settings → About phone** and tap **Build number** 7 times, until it says *"You are now a developer"*. On some phones, *Build number* is inside **Software information**.
2. **Turn on USB debugging.** Open **Settings → System → Developer options** (or search Settings for "USB debugging") and switch **USB debugging** on.
3. **Plug the phone into the computer.** A popup on the phone asks *"Allow USB debugging?"*. Tick **Always allow from this computer** and tap **Allow**.
4. **Check it.** In the terminal, run:
   ```bash
   adb devices
   ```
   Your phone should appear with the word `device` next to it. If it says `unauthorized`, look at the phone for the popup from step 3.
5. **In the phone's settings, keep the screen awake while charging.** Search Settings for **"Stay awake"** (it's in Developer options) and turn it on.

> The bot's layout is tuned on a 1080×2400 screen. Other screens are scaled by their width (the game fits the width), so taller or shorter phones keep the same picture size and the top bar where it is.

### iPhone

This path talks to the iPhone over USB with [pymobiledevice3](https://github.com/doronz88/pymobiledevice3). You don't need a Mac, a jailbreak, or any app on the phone. It needs **iOS 27 or newer**.

**On the iPhone (one time):**

1. Plug the iPhone into the computer and tap **Trust**, then enter the phone's passcode.
2. Turn on **Settings → Privacy & Security → Developer Mode**. The phone restarts; confirm **Turn On** afterwards.
   - If *Developer Mode* isn't in the list, run this on the computer with the phone plugged in, then look again:
     ```bash
     uv run pymobiledevice3 amfi reveal-developer-mode
     ```
3. Turn on **Settings → Developer → UI Automation**. It lets the computer send touches.
4. Set **Settings → Display & Brightness → Auto-Lock → Never**, so the screen doesn't lock mid-game.

**On the computer, every time you use the bot (Windows and Linux only):** start the **tunnel**, a helper that must keep running in its own terminal window while the bot plays. It needs admin rights. **On a Mac, skip this:** the bot opens its own tunnel, and a separate tunnel window only gets in the way (the Mac firewall can block the phone's video through it).

<details open>
<summary><b>Windows</b></summary>

1. Press the Windows key, type `PowerShell`, **right-click** it and choose **Run as administrator**.
2. `cd` into the bot's folder (the same as in Step 3), then run:
   ```powershell
   .venv\Scripts\pymobiledevice3 remote tunneld
   ```
3. Leave this window open. Use a second, normal PowerShell window for the bot.

</details>

<details>
<summary><b>Linux</b></summary>

In a terminal, `cd` into the bot's folder and run the command below. Enter your computer password when asked. Nothing shows as you type, which is normal.

```bash
sudo .venv/bin/pymobiledevice3 remote tunneld
```

Leave this terminal open, and open a second terminal for the bot.

</details>

**One time, and again after every iPhone restart or iOS update,** mount the *Developer Disk Image* (a small file the phone needs for screen sharing). In the second terminal:

```bash
uv run pymobiledevice3 mounter auto-mount --tunnel ''
```

On a Mac (no tunnel window), leave out the tunnel part: `uv run pymobiledevice3 mounter auto-mount`. It's fine if it says the image is already mounted.

## Step 5: start the bot

With the phone plugged in and unlocked, and the game open:

| Windows | macOS / Linux |
|---|---|
| Double-click **`start.bat`** in the bot's folder. You can also run `.\start.ps1` in PowerShell. | In the terminal, inside the bot's folder: `./start.sh` |

The bot's screen opens in the terminal. The next section explains it.

> **iPhone bonus:** while the bot runs, open the viewer address it logs at start (`viewer http://127.0.0.1:<port>/`) in your web browser to watch the phone's screen live. Each bot picks a free port, so several phones can run on one computer; set `WSBOT_STREAM_PORT` to pin one.

---

## Using the bot's screens

Use your keyboard: **↑ / ↓** to move, **Enter** to choose, **Esc** to go back, **Q** to quit.

**Screen 1: SELECT A DEVICE.** This lists every Android (from adb) and iPhone (through the tunnel) the computer can see. Pick yours and press **Enter**. If the list is empty, press **R** to refresh, then check [Troubleshooting](#troubleshooting).

**Screen 2: CHOOSE A GOAL.**

| Mode | What it does |
|---|---|
| **PLAY** | The long run: 14,000 levels over 10 days, 1,400 a day. Each day it starts around 9:00 and spreads the levels over 10–16 hours, with 25–35 minute breaks every 1–2 hours, so it plays like a person rather than a machine. |
| **SINGLE LEVEL** | Plays one level, then stops. Good for a first test. |
| **CUSTOM** | Your own pace. Press **Enter** on it to set levels per day (up to 5,000), how many hours to spread them over, breaks, and the start time. Use **← / →** to change a value; hold **Shift** for steps 10 times bigger. Your settings are remembered. |

Below the modes, **STOP AFTER** sets how many levels this run plays before it stops: **← / →** changes it by 1, **Shift** by 10, and 0 means no limit. It's remembered too.

Then move to **▶ PLAY** and press **Enter**.

**Screen 3: the run.** This screen shows:

- **BOARD**: the letter grid the bot read, with the cells it swiped and the words the game accepted.
- **TODAY / PLAN / SESSION**: levels done today, overall plan progress, and this session's count.
- **SPEED**: average seconds per level.
- **PACE**: whether it's playing, resting between levels, or on a break (and when it's back).
- **INPUT / SAFETY / WATCHER**: health checks. *SAFETY* counts taps the bot refused because they'd have hit a coin, hint, or ad button.
- **The log**: a running list of what the bot is doing. Orange `LEVEL` lines are levels starting and finishing; yellow or red lines are warnings and recoveries.

**Don't touch the phone while the bot is playing.** Your finger and the bot's swipes would mix. Press **P** to pause first.

## Stopping, pausing and resuming

- **Pause / resume:** press **P** on the run screen. The bot finishes its current swipe and waits.
- **Stop:** press **Q** (or **Esc**) on the run screen, or **Ctrl + C** in the terminal. It's safe to stop at any time.
- **Resume:** start the bot again. Progress is saved after every level, so the daily count and plan carry on where they left off. If you stop mid-level, it remembers which words it already swiped there.
- Your progress lives in the **`local`** folder inside the bot's folder. Android and iPhone keep separate files, because they are separate game accounts. Deleting that folder resets the bot's counters; your progress in the game itself is kept.


## Updating to a new version

1. Stop the bot.
2. Download the new **[wordsearch-bot.zip](https://github.com/NikoMyBelovedd/wordsearch-bot/releases/latest/download/wordsearch-bot.zip)** and unzip it.
3. **Copy the `local` folder** from your old bot folder into the new one. It holds your progress and settings.
4. Delete the old folder, and use the new one from now on (run `uv sync` in it once, as in Step 3).

The [Releases page](https://github.com/NikoMyBelovedd/wordsearch-bot/releases) lists what changed in each version. Click **Watch → Custom → Releases** at the top of the GitHub page to get an email when a new one comes out.

---

## Troubleshooting

When something goes wrong, the bot saves a screenshot in the **`diagnostics`** folder and explains the problem in the log. Here are the common problems:

| What you see | What to do |
|---|---|
| `uv` / `adb` is "not recognized" or "not found" | Close the terminal and open a new one. If it still fails, repeat [Step 2](#step-2-install-the-tools). |
| Android phone not in the device list | Run `adb devices`. If it's empty, try another cable or USB port (some cables only charge). If it says `unauthorized`, unlock the phone and tap **Allow** on the popup. |
| `waiting for the iPhone` / `no USB tunnel for an iPhone` | The bot waits until the phone shows up. On Windows/Linux the tunnel isn't running: start it again ([Step 4 → iPhone](#iphone)) and keep that window open. Make sure the phone is plugged in, unlocked, and you tapped **Trust**. |
| `screen stream did not start` | Mount the Developer Disk Image again (the `mounter auto-mount` command in Step 4). Check that **UI Automation** is still on. |
| iPhone was working, then "disappeared" while still plugged in | Unplug it and plug it back in, then restart the tunnel. On Linux you can instead run `sudo systemctl restart usbmuxd` and restart the tunnel. |
| iPhone missing from the list on Windows | Install the **Apple Devices** app (or iTunes). It has the USB driver. |
| `Tesseract not found` warning | Install Tesseract (Step 2). The bot still plays without it, but can't learn new letter shapes. |
| The bot taps at nothing, or reads no board | The screen layout doesn't match. Run `uv run wsbot --calibrate` and look at the picture it saves in `diagnostics/`. The boxes should sit on the board and buttons. |
| A popup the bot doesn't know blocks the game | It saves `diagnostics/unknown_popup_*.png`. See [teaching the bot a new popup](#advanced-teaching-the-bot-a-new-popup), or open an issue on GitHub with that picture. |
| The game restarts over and over on one level | Update the bot (download it again, Step 3) and check the log for `WARNING` lines. Open an issue with the `level_stuck_*.png` picture if it keeps happening. |
| The iPhone's screen locked and the bot stopped | Set **Auto-Lock → Never** (Step 4). |
| `Another copy of the bot is already playing on this phone` | Only one bot can play on a phone at a time. Stop the other one (another window, or a scheduled task) first. |
| `The bot stopped: the iPhone screen stream ...` | The phone's screen sharing broke and reopening it didn't help. AutomationHQ restarts the bot; running by hand, start it again (unplug/replug the phone if it keeps happening). |

## FAQ

**Will it spend my coins or hints?**
No. Every tap is checked against "no-go zones" around the coin shop, hints, boosters, ads and video-reward buttons, and those taps are refused. It only swipes letters and taps safe buttons like *Next Level*, *Collect* and *Got it*.

**Does it watch ads?**
No. It never taps ad or "watch a video" buttons.

**How fast is it?**
Usually 5–30 seconds per level. In *PLAY* mode it deliberately waits between levels to stay on a human-looking daily pace. Levels with unusual theme words can take up to a minute, because the bot then tries every straight line on the board.

**Can I use my computer while it runs?**
Yes. Keep the terminal window open, and keep the computer from going to sleep.

**Can I use the phone while it runs?**
Not for the game. Press **P** to pause first if you need to.

**Does it work on my iPhone model?**
It needs iOS 27 or newer: it sees and touches the screen through iOS 27's USB screen sharing, which older iOS versions don't have. On an older iPhone it stops right away and says which iOS the phone has. It's tested on the iPhone SE (3rd generation) and the iPhone 17. Home-button iPhones (SE 2/3) use the SE layout; every Face ID iPhone uses the iPhone 17 layout, which finds the game's top bar on the live screen, so other models should work too. If yours doesn't, run it with `--debug`, then `uv run wsbot --report`, and send the `wsbot-report.zip` it makes (open an issue on GitHub), or add the support yourself and send a pull request.

**Will it get my account banned?**
It might; automating a game can break its terms of service. See the [disclaimer](#disclaimer).

**Can it run on a server with no screen?**
Yes, see `--headless` below.

## Glossary

| Word | Meaning |
|---|---|
| **Terminal** | A text window where you type commands (PowerShell on Windows, Terminal on Mac/Linux). |
| **Folder path** | Where a folder lives, like `C:\Users\you\Documents\wordsearch-bot`. |
| **`cd`** | "Change directory": makes the terminal work inside a folder. |
| **adb** | Android Debug Bridge: the tool computers use to control Android phones over USB. |
| **Tunnel / tunneld** | A helper that opens a secure channel to the iPhone over USB. It must keep running. |
| **Developer Disk Image** | A small file from Apple that turns on screen sharing and touch control on the iPhone. |
| **Headless** | Running without the menu screens, just the log. |
| **Template** | A small cut-out picture of a button that the bot looks for on screen. |

---

## Advanced: command-line options

The menu screens are the easy way. Everything can also be run as a single command:

```bash
uv run wsbot --headless --mode play                       # paced daily target
uv run wsbot --headless --mode custom --per-day 1000      # your own daily target
uv run wsbot --headless --mode single                     # one level
uv run wsbot --headless --serial emulator-5554 --fast --levels 20
```

| Flag | Meaning |
|---|---|
| `--serial` | Which device. An adb serial (from `adb devices`), or `ios` / `ios:<UDID>` for an iPhone. Default: the `WSBOT_SERIAL` environment variable, then `ios`. |
| `--headless` | No menu screens; just log to the terminal. |
| `--mode play\|custom\|single` | The goal for `--headless` runs. |
| `--per-day N` | Daily target for `--mode custom`. |
| `--fast` | No idling or breaks (for testing). |
| `--levels N` | Stop after N levels. |
| `--dry-run` | Makes every decision but touches nothing. |
| `--diagnose` | Scores every popup template against the current screen. |
| `--calibrate` | Saves an annotated picture (board, no-go zones, popups) to `diagnostics/`. |
| `--capture NAME [--crop X,Y,W,H] [--level-done]` | Saves a new popup template from the screen and registers it. |
| `--debug` | Writes detailed logs (`local/debug.log`) and screen snapshots (`diagnostics/debug/`) for bug reports. Same as setting `WSBOT_DEBUG=1`. |
| `--report` | Zips the logs, system info and newest screenshots into `wsbot-report.zip` (under 14 MB, small enough for Discord). |

Logs are also written to `local/wsbot.log`.

**Catching up a missed day:** put a number of extra levels in a file `local/catchup-once` (or set `WSBOT_CATCHUP`) before starting. That run plays that many more levels today, with no breaks; the file is deleted as it's read, and at midnight the normal daily target and breaks come back.

## Advanced: teaching the bot a new popup

When the bot meets a screen it doesn't know, it saves `diagnostics/unknown_popup_*.png` (once per different screen), and after a few seconds it taps the screen to try to clear it, but only while it's sure it is still in the game. If the screen stays unknown it relaunches the game (45 s), then restarts the game and the phone connection (another minute), and finally stops with an error so AutomationHQ restarts it. To teach it the popup for good:

```bash
uv run wsbot --capture my_popup                            # 1. save the full screen
uv run wsbot --capture my_popup --crop 400,1800,280,90     # 2. cut out the button
```

Crop from **inside** the button (x, y, width, height, in the full-screen picture's pixels) so no background ends up in the template. The template is added to `templates/popups.json` (Android) or `templates/ios/popups.json` (iPhone). There you can tune `threshold`, `confirm`, `tap_point`, `holdoff` and `blocking`, plus:

| Key | Meaning |
|---|---|
| `"tap": false` | A known screen to wait on (level complete, loading): no tap, no "unknown" dump. With `tap_after: N` it taps (its `tap_point` or the match) once it has stayed N seconds. |
| `scales` | Extra sizes to match (e.g. `[1.035, 1.07]`) for buttons that pulse. |
| `group` | Entries of one group share their cooldown (one button, several templates): never a double tap. |
| `avoid` + `avoid_box` | An ad / spend button: never tapped, and while it's visible no tap lands in `[left, top, right, bottom]` px around it. |
| `system` / `relaunch` | The phone's own screens (an iOS alert, the home screen): not proof of being in the game; `relaunch` relaunches the game instead of tapping. |

---

## For developers: how it works

```
 frame ─▶ board.py ─▶ letters.py ─▶ solver.py ─▶ bot.py ─▶ device (swipe)
            panel        glyph →        words        pass ladder,
            + grid       letter         + paths      verification
                           ▲
 watcher.py (thread): popups · foreground app · reconnects · safety zones
```

- **Reading the board.** It finds the white letter panel, splits it into cells, and reads each glyph by template matching against the game's font (Lato Black). Tesseract acts as a second opinion. A new glyph shape is learned only when **both** agree, so one bad read can't poison the cache.
- **Solving.** An 8-direction prefix search over a frequency-ordered wordlist solves a board in about 0.5 ms. Common words get swiped first.
- **Pass ladder.**
  1. *Fast:* every dictionary word once, on its likeliest path: not a copy lying inside a longer word (*TAB* inside *BATTER* read backwards). The other paths of 3-4 letter words get a short pass of their own if the level isn't done.
  2. *Exhaustive:* every straight line with an unlit cell, most untouched cells first. This catches theme words the dictionary lacks, like *ORANGUTAN* or *SCAVENGER*.
  3. *Repeat:* dictionary words again, slower.
  4. *Restart* the app. The level's "already swiped" memory is then partly forgotten, so swipes the game ignored get fired again.

  It never swipes the same word twice in a pass, because re-swiping an already-collected word pops a toast that swallows the next swipes.
- **Popup watcher thread.** It owns the screen frames, matches popups against a template registry (first match wins), taps them, and dumps unknown screens. It also checks that the game is in the foreground and reconnects the device.
- **Safety layer.** Taps inside no-go zones are refused: static zones (back, shop, bonus jar), a dynamic zone below the board during a level (boosters, ads), and temporary zones while a video-reward button is visible.
- **Liveness probe.** When nothing new gets found, the bot holds a two-letter drag (never a word) and checks that the game draws the selection. If it doesn't, twice in a row, the game is ignoring input and gets restarted.

| Module | Role |
|---|---|
| `device.py` | Android backend: uiautomator2 screenshots with an `adb screencap` fallback, and one persistent `adb shell` for fast input. Also the safety zones. |
| `iphone.py` / `ios_device.py` | iOS backend: the phone's screen stream (HEVC, decoded with PyAV) and HID touch reports via pymobiledevice3's CoreDevice services. Frames are scaled so the Android calibration and templates apply. Two layouts: `se` (home-button iPhones, fixed zones) and `tall` (Face ID iPhones: zones placed from the top bar found on the live screen). On macOS it opens its own USB tunnel (`WSBOT_TUNNEL=auto\|userspace\|tunneld`). |
| `board.py` | Panel and grid detection, partial-board rejection, highlighted-cell detection. |
| `letters.py` | Glyph reading (64×64 templates plus Tesseract). |
| `solver.py` | Dictionary prefix search. |
| `watcher.py` | Popup watcher thread and frame owner. |
| `bot.py` | Main loop, pass ladder, liveness probe, recovery. |
| `goal.py`, `schedule.py` | Progress tracking (atomic JSON) and daily pacing. |
| `tui.py`, `cli.py` | Terminal UI and command-line entry point. |

```bash
uv sync                           # includes dev tools
uv run pytest                     # device-free smoke tests
uv run ruff check .               # lint
uv run ruff format .              # format
uv run python tools/tui_demo.py   # the TUI with a fake bot, no device needed
```

CI runs lint, the tests, and a CLI start-up check on **Windows, Linux and macOS**.

<details>
<summary><b>Repository layout</b></summary>

```
src/wsbot/          the bot
templates/          Android popup templates + registry, letter templates
templates/ios/      iPhone SE popup templates + registry
templates/ios_tall/ Face ID iPhone popup templates, registry + top-bar anchors
data/words.txt      200k-word frequency-ordered dictionary
tools/              reference-glyph renderer, TUI demo
tests/              smoke tests
start.bat/.ps1/.sh  one-click launchers
local/              (created at runtime) progress, settings, logs
diagnostics/        (created at runtime) screenshots of unknown or stuck states
```

</details>

## Disclaimer

This is an independent hobby project. It is not affiliated with or endorsed by PlaySimple Games or Apple. Automating a game may break its terms of service, so use it at your own risk. The bot is built never to spend in-game currency or interact with ads.

## License

[MIT](LICENSE)

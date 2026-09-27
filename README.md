<div align="center">

# wordsearch-bot

**A hands-off auto-solver for *Word Search Explorer*, running on Android (ADB) and iPhone (USB).**

It reads the board from the screen and solves it with a 200k-word dictionary, then swipes the answers in.
It clears popups on its own and can run for thousands of levels without supervision.

[![CI](https://github.com/NikoMyBelovedd/wordsearch-bot/actions/workflows/ci.yml/badge.svg)](https://github.com/NikoMyBelovedd/wordsearch-bot/actions/workflows/ci.yml)
![Python 3.12](https://img.shields.io/badge/python-3.12-3776AB?logo=python&logoColor=white)
![Platforms](https://img.shields.io/badge/runs%20on-Windows%20%7C%20Linux%20%7C%20macOS-555)
![Devices](https://img.shields.io/badge/plays%20on-Android%20%7C%20iOS-orange)
[![License: MIT](https://img.shields.io/badge/license-MIT-green)](LICENSE)

</div>

---

## Features

- **Reads the board from pixels.** It finds the letter panel and splits it into cells, then reads each glyph by template matching. Tesseract OCR acts as a second opinion. A new glyph is learned only when **both** readers agree, so the template cache can't be poisoned by one bad read.
- **Solves a board in about 0.5 ms.** An 8-direction prefix search runs over a frequency-ordered wordlist, so common words get swiped first.
- **Solves in passes:** first a fast pass over dictionary words, then an exhaustive pass over every unlit line (this catches theme words that aren't in the dictionary, like *ORANGUTAN*), then a slow repeat pass, and a last-resort app restart. It never swipes the same word twice in a level.
- **Handles popups on a separate thread.** A watcher thread matches known popups (level complete, rewards, tutorials, invites) against a template registry and taps them away. It saves a screenshot of anything it doesn't recognise.
- **Safety layer.** Taps are refused inside no-go zones: the coin shop, hints, boosters, ads, and "watch a video" buttons. Some zones exist only while an ad button is on screen. The bot **never spends coins or hints**.
- **Self-recovery.** A liveness probe detects when the game has stopped responding to input. The bot also detects stuck levels and restarts or reconnects the app and device.
- **Human-like pacing.** *Play* mode spreads a daily level target over a randomised window with breaks. You can set the target yourself in *Custom* mode, or play a *Single* level.
- **Terminal UI** built on [Textual](https://textual.textualize.io/): pick a device and a mode, then watch live progress and logs. A headless mode is included for servers and scripts.

## How it works

```
 frame ─▶ board.py ─▶ letters.py ─▶ solver.py ─▶ bot.py ─▶ device (swipe)
            panel        glyph →        words        pass ladder,
            + grid       letter         + paths      verification
                           ▲
 watcher.py (thread): popups · foreground app · reconnects · safety zones
```

| Module | Role |
|---|---|
| `device.py` | Android backend: uiautomator2 screenshots with an `adb screencap` fallback, and one persistent `adb shell` for fast input. Also holds the safety zones. |
| `iphone.py` / `ios_device.py` | iOS backend: the phone's screen stream is decoded with PyAV, and touches are injected as HID reports through [pymobiledevice3](https://github.com/doronz88/pymobiledevice3). No jailbreak, WebDriverAgent, or Mac is needed. |
| `board.py` | Finds the panel and the letter grid, rejects partly covered boards, and detects which cells are already highlighted. |
| `letters.py` | Reads glyphs using 64×64 templates (Lato reference set plus learned glyphs) and Tesseract. |
| `solver.py` | Dictionary prefix search in 8 directions. |
| `watcher.py` | Popup watcher thread and frame owner. |
| `bot.py` | Main loop, pass ladder, liveness probe, recovery. |
| `goal.py`, `schedule.py` | Progress tracking (atomic JSON) and daily pacing. |
| `tui.py`, `cli.py` | Terminal UI and command-line entry point. |

## Requirements

| | Windows | Linux | macOS |
|---|---|---|---|
| [uv](https://docs.astral.sh/uv/) (installs Python 3.12 for you) | ✅ | ✅ | ✅ |
| [Tesseract OCR](https://github.com/tesseract-ocr/tesseract) | [UB Mannheim installer](https://github.com/UB-Mannheim/tesseract/wiki) | `apt/pacman/dnf install tesseract` | `brew install tesseract` |
| **Android:** [platform-tools](https://developer.android.com/tools/releases/platform-tools) (`adb`) | ✅ | ✅ | ✅ |
| **iPhone:** Apple USB driver | [Apple Devices](https://apps.microsoft.com/detail/9np83lwlpz9k) app or iTunes | `usbmuxd` | built in |

> Tesseract is found automatically on `PATH` or in `C:\Program Files\Tesseract-OCR`. You can also point `TESSERACT_CMD` at `tesseract.exe`.
> Without Tesseract the bot still runs, but it can't learn new glyphs.

## Installation

```bash
git clone https://github.com/NikoMyBelovedd/wordsearch-bot.git
cd wordsearch-bot
uv sync
```

<details>
<summary><b>Don't have uv yet?</b></summary>

```powershell
# Windows (PowerShell)
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

```bash
# Linux / macOS
curl -LsSf https://astral.sh/uv/install.sh | sh
```

</details>

## Quick start

Start the TUI:

| Windows | Linux / macOS |
|---|---|
| double-click **`start.bat`**, or run `.\start.ps1` | `./start.sh` |

You can also run `uv run wsbot` directly. Pick a device and a mode, then press **PLAY**.

### Headless

```bash
uv run wsbot --headless --mode play                       # paced daily target
uv run wsbot --headless --mode custom --per-day 1000      # your own daily target
uv run wsbot --headless --mode single                     # one level
uv run wsbot --headless --serial emulator-5554 --fast --levels 20
```

| Flag | Meaning |
|---|---|
| `--serial` | ADB serial, or `ios` / `ios:<UDID>` for an iPhone. Default: `$WSBOT_SERIAL`, then `ios`. |
| `--mode play\|custom\|single` | Goal for `--headless` runs. |
| `--per-day N` | Daily target for `--mode custom`. |
| `--fast` | No idling or breaks (for testing). |
| `--levels N` | Stop after N levels. |
| `--dry-run` | Makes every decision but touches nothing. |
| `--diagnose` | Scores every template against the current frame. |
| `--calibrate` | Saves an annotated overlay (board, zones, popups) to `diagnostics/`. |
| `--capture NAME [--crop X,Y,W,H] [--level-done]` | Saves a new popup template from the screen and registers it. |

Press `Ctrl+C` to stop cleanly. Progress is saved after every level.

## Device setup

### Android

1. Turn on **Developer options → USB debugging** (an emulator works too).
2. Check that `adb devices` lists the device.
3. Install *Word Search Explorer* (`in.playsimple.wordsearch`) and play through the first tutorial by hand.

The calibration space is 1080×2400. Other resolutions are scaled to it.

### iPhone (iOS 27+)

The iPhone backend uses pymobiledevice3 over USB. It needs a USB tunnel service running with admin rights.

1. Plug the phone in and tap **Trust**.
2. Turn on **Settings → Privacy & Security → Developer Mode**. If the option is hidden, run `uv run pymobiledevice3 amfi reveal-developer-mode`.
3. Turn on **Settings → Developer → UI Automation**, and set **Auto-Lock → Never**.
4. Start the tunnel service and leave it running:
   ```bash
   # Linux / macOS
   sudo .venv/bin/pymobiledevice3 remote tunneld
   ```
   ```powershell
   # Windows: in an *Administrator* terminal
   .venv\Scripts\pymobiledevice3 remote tunneld
   ```
5. Mount the Developer Disk Image. You need to do this once, and again after each reboot or iOS update:
   ```bash
   uv run pymobiledevice3 mounter auto-mount --tunnel ''
   ```
6. Run `uv run wsbot` and pick the iPhone. A live viewer of the phone's screen runs at <http://127.0.0.1:8090/>.

The iOS layout is calibrated on an **iPhone SE (750×1334)** only. Other models (for example Face ID iPhones with taller screens) need a new scale factor, safety zones and possibly templates in `ios_device.py`; run `--calibrate` and check the overlay before letting the bot play.

## Adding a new popup

When the bot hits a screen it doesn't know, it saves `diagnostics/unknown_popup_*.png`. To teach it:

```bash
uv run wsbot --capture my_popup                            # 1. save the full frame
uv run wsbot --capture my_popup --crop 400,1800,280,90     # 2. crop INSIDE the button
```

Crop from **inside** the button so no background ends up in the template. The template is added to `templates/popups.json` (Android) or `templates/ios/popups.json` (iPhone). You can then tune `threshold`, `confirm`, `tap_point`, `holdoff` and `blocking` there.

## Development

```bash
uv sync                 # includes dev tools
uv run pytest           # device-free smoke tests
uv run ruff check .     # lint
uv run ruff format .    # format
uv run python tools/tui_demo.py   # the TUI with a fake bot, no device needed
```

CI runs lint, the tests, and a CLI start-up check on **Windows, Linux and macOS**.

<details>
<summary><b>Repository layout</b></summary>

```
src/wsbot/          the bot (see "How it works")
templates/          Android popup templates + registry, letter templates
templates/ios/      iPhone popup templates + registry
data/words.txt      200k-word frequency-ordered dictionary
tools/              reference-glyph renderer, TUI demo
tests/              smoke tests
local/              (created at runtime) progress, settings, logs
diagnostics/        (created at runtime) screenshots of unknown or stuck states
```

</details>

## Troubleshooting

| Symptom | Fix |
|---|---|
| `adb not found` | Install platform-tools and add them to `PATH`. |
| `no USB tunnel for an iPhone` | Start `pymobiledevice3 remote tunneld` as root or Administrator, and make sure the phone is unlocked and trusted. |
| `screen stream did not start` | Mount the Developer Disk Image (setup step 5 above). |
| iPhone missing from the list on Windows | Install the Apple Devices app or iTunes, which provides the USB driver. |
| `Tesseract not found` warning | Install Tesseract or set `TESSERACT_CMD`. |
| Bot keeps tapping at nothing | Run `--calibrate` and check the overlay in `diagnostics/`. |
| An unknown popup blocks play | Capture it (see [Adding a new popup](#adding-a-new-popup)). |

## Disclaimer

This is an independent hobby project. It is not affiliated with or endorsed by PlaySimple Games or Apple. Automating a game may break its terms of service, so use this at your own risk. The bot is built never to spend in-game currency or interact with ads.

## License

[MIT](LICENSE)

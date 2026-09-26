# HANDOFF — wordsearch-bot

Snapshot as of 2026-09-26 ~19:20. Read this first after `/clear`.

## What this is

An ADB + OCR bot that plays **Word Search Explorer** (`in.playsimple.wordsearch`) forever. It runs on `emulator-5554`, which is 1080x2400, and that is also the calibration space.

- **Repo:** `~/Documents/wordsearch-bot`, private GitHub `NikoMyBelovedd/wordsearch-bot`.
- **Plan:** `~/.claude/plans/yo-just-got-max-inherited-raccoon.md`.
- **User's ADB playbook:** `~/.claude/adb-automation/`. `failure-modes.md` there has entries 11–19 from this project. **Standing rule:** log every break→fix there right away.
- **Old project:** `~/Documents/wordsearchexplorer` is abandoned. Never restore its code; only `data/words.txt` came from it.

## User's requirements (don't drift)

- **Card:** ignore the hint card, which uses emojis on later levels. Brute-force a frequency-ordered wordlist (common words first, Scrabble oddities last), words of 3+ letters.
- **Threads:** a popup-watcher™ thread plus a main thread.
  - Templates are **crops from inside buttons only, never background**.
  - While the board is hidden with no known popup, the bot does a clear tap. The user asked for "a click in the middle of the board".
- **Spending:** **never spend coins or hints.** The SAFETY layer enforces this (see below).
- **Priorities:** stability first, then speed.
- **Testing:** by live runs, try→fail→fix. No big test suites.
- **Target:** has to hold up for **14k levels**.
- **TUI** (Textual, orange/white, big WORDSEARCH-SOLVER banner):
  1. Pick an adb device.
  2. Pick a mode, top to bottom: **Play** = 14,000 levels in 10 days (1,400/day, then idle till midnight); **Custom target** = levels/day via ←/→, capped at 5,000; **Single level**.
  3. Arrow down to **PLAY**.

## How to run

```bash
./start.sh                                   # TUI (uv run wsbot)
uv run wsbot --headless --mode custom --per-day 1000   # headless (what I test with)
uv run wsbot --diagnose                      # every popup template score + board read on current screen
uv run wsbot --calibrate                     # same + overlay PNG in diagnostics/
uv run wsbot --capture NAME [--crop x,y,w,h] [--level-done]   # add a popup template
uv run python tools/tui_demo.py              # TUI with a fake bot (no device)
```

My test loop was this:

```bash
rm -f diagnostics/*.png
timeout -s INT 480 .venv/bin/wsbot --headless --mode custom --per-day 1000 > diagnostics/runN.log 2>&1
grep -E '\[(LEVEL|PASS|WARN|ERROR|RECOVERY|SAFETY|GOAL)\]|WATCHER\] [a-z_]+ score' diagnostics/runN.log
```

After that, look at `diagnostics/unknown_popup_*.png` / `level_stuck_*.png` / `no_board_*.png`.

- **Progress:** stored in `local/progress.json` (per-day and per-plan counts). This is the *real* count, and it persists across restarts.
- **Log:** `local/wsbot.log`, rotating.
- `local/` and `diagnostics/` are both gitignored.

## Architecture (`src/wsbot/`)

| file | role |
|---|---|
| `device.py` | Screenshots: u2 (~115 ms) run under a 2.5 s guard thread, falling back to raw `adb exec-out screencap` (u2 can hang 30 s+ while it restarts its server). Input: **one persistent `adb shell`** running `input swipe/tap` (~50 ms/swipe). App start/stop/foreground via plain adb with timeouts. **SAFETY zones:** static (back, star=bonus jar, coins/shop), plus a dynamic **below-board** zone during levels that covers the booster row (GET ad, burst hint, lightbulb, shuffle). `tap(..., allow="zone")` exempts one named zone. |
| `board.py` | Board = largest near-white panel (after a 9x9 MORPH_OPEN). Letters = dark connected components clustered into rows/cols (any NxM). `highlighted()` samples just above each glyph: white means unfound, a colored pill means found. |
| `letters.py` | Glyph → normalized 64x64. Templates come from `templates/letters/ref/` (A–Z rendered from Lato Black by `tools/render_reference.py`; the game's font is Lato Black) plus `game/` (learned). A glyph is learned **only when Tesseract and the template agree**. |
| `solver.py` | Prefix-set search in 8 directions over `data/words.txt` (200k words, frequency-ordered). Emits every path of each word. |
| `watcher.py` | Thread that owns the frames (`latest()`), computes `board_visible` (during a level: panel bbox matches ±12 px; between levels: a full grid read), and matches popups at half resolution. **The first match in registry order wins the frame.** Per-entry `threshold`, `cooldown`, `confirm` (consecutive frames), `tap_point`, `allow`, `level_done`. `busy()` returns True for 1 s after it taps. Also: foreground check and relaunch every 5 s, reconnect after 5 errors, idle mode. |
| `bot.py` | Loop: `wait_for_board` (2 identical reads, clear taps alternating board center / (540,620), a `[WAIT]` reason every 5 s) → `solve_level` → record in goal. The pass ladder is described below. Recovery: no board for 150 s → restart the app; stale previous board after 15 s → solve it again; level end needs a template, OR hidden ≥1.5 s, OR a different grid read twice. |
| `goal.py` | `Goal.play/custom/single`, `Progress` (atomic JSON). |
| `tui.py`, `banner.py` | Textual app: DeviceScreen → ModeScreen → RunScreen. Built by a subagent and pilot-tested with a fake bot. **Not yet run against the real device.** |
| `diagnostics.py`, `cli.py` | `--diagnose` / `--calibrate` / `--capture`, headless flags. |

**The pass ladder** in `solve_level`:
1. **fast:** every dictionary word once, 60 ms swipes, no gap. The game does NOT block input during word-found animations.
2. **unfound:** every path of each word not yet highlighted, 120 ms swipes.
3. **exhaustive:** every straight line of 3+ letters with an unlit cell. This finds words missing from the dictionary (e.g. LUDO).
4. **app restart.**

**Popup registry** (`templates/popups.json`, priority order): next_level (level_done), collect, set_continue, back_to_level (Daily Challenge done), tutorial_claim_bonus (tap_point = star (216,205), allow star_bonus_jar), bonus_claim, awesome (daily streak), got_it, close_x_grey (confirm 8, so Claim animates in first).

## Current state

- **Committed and pushed up to `bfc5196`.** The commit that goes with this handoff adds:
  - the guarded u2/screencap device,
  - watcher priority/confirm/busy,
  - stricter level end and the stale-previous valve,
  - set_continue and back_to_level templates,
  - the TUI.
- **Performance:** a typical level takes **5–25 s** (Next Level to the next board is ~1 s). Big boards that need the exhaustive pass take 40–60 s. Run 12 did about 1 level / 25 s including popups.
- **Scale:** it reached game level ~35 across several runs, handling the Daily Challenge, the country/set completes, the bonus-jar tutorial and daily streak unattended.

## Open items / next steps

1. **Tournament tutorial** ("Tap here to check your position and rewards!", arrow at the trophy at top-left, about (112,560); it appeared on the level-complete "Magnificent!" screen). The user fixed it by hand. **If it comes back:** template the text inside the box, and use `tap_point` on the trophy (it likely opens a leaderboard with an X or Close). Neither clear-tap spot closes it: the box sits over the board center, and (540,620) doesn't work either. Frames of it are in `diagnostics/unknown_popup_20260926_1914*.png`.
2. **Not yet verified live since the last edits:**
   - the stricter `_wait_level_end` and stale-previous valve in `bot.py`,
   - close_x confirm=8,
   - the set_continue and back_to_level templates.
   Run headless ~10 min and read the log.
3. **TUI on the real device:** launch `./start.sh`, pick emulator-5554, try each mode, and check that stats, board and log update and that `p`/`q` work.
4. **"Picture Puzzle" level type** is coming: the Next Level button shows a "Picture Puzzle" tag. It may need its own handling.
5. **Speed tuning left:**
   - the exhaustive pass is slow on 9x8 boards (1,274 swipes);
   - swipe_ms could try 40;
   - watch that 60 ms swipes never get missed (the `unfound` pass count is the signal).
6. **Learning mid-animation glyphs:** a skewed "S" was learned once (deleted). Consider learning only from stable reads.
7. **Soak test:** an hours-long soak before starting the 10-day Play run. Check `restarts`, `unknown_popup` frequency, and `local/progress.json`.

## Gotchas learned here (details in failure-modes.md #11–19)

- The white panel can merge with the hint card through anti-aliased bridges, so run MORPH_OPEN before connected components.
- Tesseract alone misreads V→A. Never learn a glyph from one reader.
- Popup bodies are white too, so between levels "board visible" must mean a real letter grid.
- Forced tutorials can block **all** board input without dimming the board. If an exhaustive pass can't finish a level, suspect an overlay first.
- Static SAFETY zones can cover legitimate buttons on other screens (Awesome! at (540,1978)). That's why the booster row is covered by the dynamic zone.
- u2 blocks silently while self-healing. Nothing in a watchdog path may call it unguarded.
- One odd frame must never end a level.

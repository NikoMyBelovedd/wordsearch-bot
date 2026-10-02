"""Render iOS system alerts and notification banners onto real game frames, for the
sysalert tests (tests/data/alerts/). The iOS 26/27 "Liquid Glass" look was measured on a
real frame (tests/data/screens/alert_watch_notifications.jpg, iPhone SE): a 318 pt wide
box, 46 pt pill buttons, dimmed screen. Real labels, a sans font close to Apple's.

    uv run python tools/make_alert_frames.py

The tall frames are SE game frames fitted to an iPhone 17's width (1206x2622), the way
the game lays out there, with the top and bottom extended."""

from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

ROOT = Path(__file__).resolve().parents[1]
SCREENS = ROOT / "tests" / "data" / "screens"
OUT = ROOT / "tests" / "data" / "alerts"
FONTS = [
    ("/usr/share/fonts/noto/NotoSans-Regular.ttf", "/usr/share/fonts/noto/NotoSans-Bold.ttf"),
    ("/usr/share/fonts/TTF/DejaVuSans.ttf", "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf"),
    ("C:/Windows/Fonts/segoeui.ttf", "C:/Windows/Fonts/segoeuib.ttf"),
]

DARK = {
    "box": (33, 33, 34),
    "button": (65, 64, 66),
    "title": (255, 255, 255),
    "text": (150, 150, 152),
}
LIGHT = {
    "box": (244, 244, 246),
    "button": (218, 218, 222),
    "title": (0, 0, 0),
    "text": (70, 70, 72),
}


def font(size: float, bold: bool = False) -> ImageFont.FreeTypeFont:
    for regular, heavy in FONTS:
        path = heavy if bold else regular
        if Path(path).is_file():
            return ImageFont.truetype(path, round(size))
    raise SystemExit("no TrueType sans font found")


def base(name: str, tall: bool) -> np.ndarray:
    img = cv2.imread(str(SCREENS / f"{name}.jpg"))
    if not tall:
        return img
    w, h = 1206, 2622
    fit = cv2.resize(
        img, (w, round(img.shape[0] * w / img.shape[1])), interpolation=cv2.INTER_CUBIC
    )
    top = (h - fit.shape[0]) // 2
    return cv2.copyMakeBorder(fit, top, h - fit.shape[0] - top, 0, 0, cv2.BORDER_REPLICATE)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, width: float) -> list[str]:
    lines, line = [], ""
    for word in text.split():
        trial = f"{line} {word}".strip()
        if draw.textlength(trial, font=f) <= width or not line:
            line = trial
        else:
            lines.append(line)
            line = word
    return [*lines, line] if line else lines


def alert(
    img: np.ndarray, title: str, message: str, buttons: list[str], *, dark: bool, stacked: bool
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """The frame with the alert over it, and each button's center (phone px)."""
    h, w = img.shape[:2]
    pt = w / (375 if w < 1000 else 402)
    c = DARK if dark else LIGHT
    under = cv2.GaussianBlur((img * 0.45).astype(np.uint8), (0, 0), 20 * pt)
    out = (img * 0.45).astype(np.uint8)  # iOS dims everything behind an alert
    pil = Image.fromarray(cv2.cvtColor(out, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(pil)
    bw, pad = 318 * pt, 24 * pt
    ft, fm, fb = font(17 * pt, True), font(15 * pt), font(17 * pt, bold=not dark)
    tl = wrap(draw, title, ft, bw - 2 * pad)
    ml = wrap(draw, message, fm, bw - 2 * pad) if message else []
    bh, gap = 46 * pt, 10 * pt
    rows = len(buttons) if stacked else 1
    text_h = len(tl) * 22 * pt + (8 * pt + len(ml) * 20 * pt if ml else 0)
    box_h = pad + text_h + 20 * pt + rows * bh + (rows - 1) * gap + 16 * pt
    x0, y0 = (w - bw) / 2, (h - box_h) / 2
    box = (round(x0), round(y0), round(x0 + bw), round(y0 + box_h))
    # translucent glass: the blurred, dimmed screen under a mostly opaque gray
    glass = Image.fromarray(cv2.cvtColor(under, cv2.COLOR_BGR2RGB))
    tint = Image.new("RGB", pil.size, c["box"])
    glass = Image.blend(glass, tint, 0.92 if dark else 0.96)
    mask = Image.new("L", pil.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=round(34 * pt), fill=255)
    pil.paste(glass, (0, 0), mask)
    y = y0 + pad
    for line in tl:
        draw.text((x0 + pad, y), line, font=ft, fill=c["title"])
        y += 22 * pt
    if ml:
        y += 8 * pt
        for line in ml:
            draw.text((x0 + pad, y), line, font=fm, fill=c["text"])
            y += 20 * pt
    y += 20 * pt
    inner = bw - 2 * 16 * pt
    centers = []
    for i, label in enumerate(buttons):
        if stacked:
            bx0, by0, bwid = x0 + 16 * pt, y + i * (bh + gap), inner
        else:
            bwid = (inner - (len(buttons) - 1) * 9 * pt) / len(buttons)
            bx0, by0 = x0 + 16 * pt + i * (bwid + 9 * pt), y
        b = (round(bx0), round(by0), round(bx0 + bwid), round(by0 + bh))
        draw.rounded_rectangle(b, radius=round(bh / 2), fill=c["button"])
        tw = draw.textlength(label, font=fb)
        draw.text(
            ((b[0] + b[2] - tw) / 2, b[1] + bh / 2), label, font=fb, fill=c["title"], anchor="lm"
        )
        centers.append(((b[0] + b[2]) // 2, (b[1] + b[3]) // 2))
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), centers


def banner(
    img: np.ndarray, app: str, text: str, *, dark: bool, lines: int = 1
) -> tuple[np.ndarray, tuple]:
    """A notification banner over the top of the frame, and its box."""
    h, w = img.shape[:2]
    pt = w / (375 if w < 1000 else 402)
    top = 8 * pt if w < 1000 else 56 * pt
    box = (round(8 * pt), round(top), round(w - 8 * pt), round(top + (54 + 20 * lines) * pt))
    under = cv2.GaussianBlur(img, (0, 0), 24 * pt)
    glass = Image.blend(
        Image.fromarray(cv2.cvtColor(under, cv2.COLOR_BGR2RGB)),
        Image.new("RGB", (w, h), (40, 40, 42) if dark else (246, 246, 248)),
        0.78,
    )
    pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    mask = Image.new("L", pil.size, 0)
    ImageDraw.Draw(mask).rounded_rectangle(box, radius=round(24 * pt), fill=255)
    pil.paste(glass, (0, 0), mask)
    draw = ImageDraw.Draw(pil)
    icon = (box[0] + round(14 * pt), box[1] + round(17 * pt))
    s = round(40 * pt)
    draw.rounded_rectangle(
        (*icon, icon[0] + s, icon[1] + s), radius=round(9 * pt), fill=(52, 199, 89)
    )
    ink = (255, 255, 255) if dark else (0, 0, 0)
    tx = icon[0] + s + round(12 * pt)
    draw.text((tx, box[1] + 16 * pt), app, font=font(15 * pt, True), fill=ink)
    for i, line in enumerate(wrap(draw, text, font(15 * pt), box[2] - tx - 14 * pt)[:lines]):
        draw.text((tx, box[1] + (38 + 20 * i) * pt), line, font=font(15 * pt), fill=ink)
    return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR), box


NOTIF = (
    "Notifications may include alerts, sounds, and icon badges. These can be configured in "
    "Settings."
)
ALERTS = [
    # name, game frame, tall, dark, stacked, title, message, buttons, button to tap (or None)
    ("se_notifications_dark", "board_readable", False, True, False,
     "\u201cWord Search\u201d Would Like to Send You Notifications", NOTIF,
     ["Don\u2019t Allow", "Allow"], "Don't Allow"),
    ("se_low_battery_light", "next_level_plain", False, False, True,
     "Low Battery", "20% of battery remaining", ["Close", "Low Power Mode"], "Close"),
    ("se_update_stacked_dark", "level_complete_no_button", False, True, True,
     "Software Update", "iOS 27.0.1 is available and will be installed tonight.",
     ["Install Tonight", "Remind Me Later", "Details"], "Remind Me Later"),
    ("se_update_row_light", "board_letters_flying", False, False, False,
     "Update Available", "An update to iOS 27.0.1 is available. Would you like to update now?",
     ["Later", "Install Now"], "Later"),
    ("se_trust_light", "board_letters_flying", False, False, False,
     "Trust This Computer?",
     "Your settings and data will be accessible from this computer when connected.",
     ["Trust", "Don\u2019t Trust"], None),
    ("se_settings_ok_dark", "keep_playing", False, True, False,
     "Turn Off Airplane Mode or Use Wi-Fi to Access Data", "", ["Settings", "OK"], None),
    ("se_single_ok_light", "country_gift", False, False, False,
     "Cannot Connect to App Store", "", ["OK"], "OK"),
    ("se_tracking_stacked_dark", "next_level_plain", False, True, True,
     "Allow \u201cWord Search\u201d to track your activity across other companies\u2019 apps "
     "and websites?", "", ["Ask App Not to Track", "Allow"], "Ask App Not to Track"),
    ("se_update_required_dark", "board_letters_flying", False, True, False,
     "Update Required", "Update this app to keep playing.", ["Update"], None),
    ("tall_notifications_light", "board_letters_flying", True, False, False,
     "\u201cWord Search\u201d Would Like to Send You Notifications", NOTIF,
     ["Don\u2019t Allow", "Allow"], "Don't Allow"),
    ("tall_low_battery_stacked_dark", "next_level_plain", True, True, True,
     "Low Battery", "10% of battery remaining", ["Low Power Mode", "Close"], "Close"),
    ("tall_trust_dark", "level_complete_no_button", True, True, False,
     "Trust This Computer?",
     "Your settings and data will be accessible from this computer when connected.",
     ["Trust", "Don\u2019t Trust"], None),
    ("tall_not_now_light", "keep_playing", True, False, False,
     "Sign In to Apple Account", "Enter the password for your Apple Account in Settings.",
     ["Not Now", "Settings"], "Not Now"),
]  # fmt: skip
BANNERS = [
    ("se_banner_light", "board_readable", False, False, 1, "Messages", "Mom: Call me when you can"),
    ("se_banner_dark", "board_readable", False, True, 1, "Watch", "Your workout summary is ready"),
    ("se_banner_two_lines_light", "board_readable", False, False, 2, "Mail",
     "Your order has shipped and will arrive on Friday. Track it in the app."),
    ("tall_banner_light", "board_readable", True, False, 1, "Calendar", "Dentist in 15 minutes"),
]  # fmt: skip


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    truth: dict[str, dict] = {}
    for name, frame, tall, dark, stacked, title, message, buttons, tap in ALERTS:
        img, centers = alert(base(frame, tall), title, message, buttons, dark=dark, stacked=stacked)
        cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        labels = [b.replace("\u2019", "'") for b in buttons]
        truth[name] = {
            "buttons": labels,
            "centers": centers,
            "tap": tap,
            "tap_center": centers[labels.index(tap)] if tap else None,
        }
    for name, frame, tall, dark, lines, app, text in BANNERS:
        img, box = banner(base(frame, tall), app, text, dark=dark, lines=lines)
        cv2.imwrite(str(OUT / f"{name}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 85])
        truth[name] = {"banner": box, "frame": frame, "tall": tall}
    (OUT / "truth.json").write_text(json.dumps(truth, indent=1) + "\n", encoding="utf-8")
    print(f"wrote {len(truth)} frames to {OUT}")


if __name__ == "__main__":
    main()

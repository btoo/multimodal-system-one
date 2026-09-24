"""Controlled panels and an independent symbolic oracle; never used by inference.

Real keyword recordings are paired with original generated panels. These panels
are an algorithm experiment, not screenshots of real applications.
"""
from __future__ import annotations

from PIL import Image, ImageDraw
import numpy as np

WORDS = ["down", "go", "left", "no", "right", "stop", "up", "yes"]
COLORS = {"red": (220, 66, 74), "green": (42, 170, 105), "blue": (56, 117, 225), "yellow": (239, 188, 48)}
POSITIONS = ["top left", "top right", "bottom left", "bottom right"]
OPPOSITE = {"left":"right", "right":"left", "up":"down", "down":"up", "yes":"no", "no":"yes", "go":"stop", "stop":"go"}
JOINT_TASKS = ["color", "opposite_color", "position", "opposite_position", "present", "absent"]
TASKS = [*JOINT_TASKS, "heard_word", "tile_word"]
TEMPLATES = {
    "color": ["what color marks the spoken command", "which color surrounds the command in the audio", "on the screen which color marks the command in the audio"],
    "opposite_color": ["what color marks the opposite of the spoken command", "which color surrounds the opposite command in the audio", "on the screen which color marks the opposite command in the audio"],
    "position": ["where is the spoken command on the screen", "what position contains the command in the audio", "on the screen where is the command in the audio"],
    "opposite_position": ["where is the opposite of the spoken command on the screen", "what position contains the opposite command in the audio", "on the screen where is the opposite command in the audio"],
    "present": ["is the spoken command on the screen", "is the command in the audio visible", "on the screen is the command in the audio visible"],
    "absent": ["is the spoken command missing from the screen", "is the command in the audio not visible", "on the screen is the command in the audio not visible"],
    "heard_word": ["what word is spoken", "which command is in the audio", "in the audio what word is spoken"],
    "tile_word": ["which command is in the {position}", "what word does the {position} symbol represent", "in the {position} which command is visible"],
}


def held_pair(word, color):
    return (WORDS.index(word) + list(COLORS).index(color)) % 4 == 0


def make_panel(seed, compositional=False):
    rng = np.random.default_rng(seed)
    words = rng.choice(WORDS, 4, replace=False).tolist()
    tiles = []
    for word in words:
        eligible = [c for c in COLORS if held_pair(word, c) == compositional]
        tiles.append({"word": word, "color": str(rng.choice(eligible)),
                      "jitter": rng.integers(-2, 3, 2).tolist(), "stroke": int(rng.integers(4, 7))})
    return {"tiles": tiles, "background": int(rng.integers(239, 250))}


def render_panel(panel):
    image = Image.new("RGB", (128, 128), (panel["background"],)*3)
    d = ImageDraw.Draw(image)
    for index, tile in enumerate(panel["tiles"]):
        x, y = (index % 2)*64, (index // 2)*64
        d.rounded_rectangle((x+6,y+6,x+58,y+58), radius=7, fill=COLORS[tile["color"]], outline=(35,45,57), width=2)
        cx, cy = x+32+tile["jitter"][0], y+32+tile["jitter"][1]
        word = tile["word"]; ink=(24,32,44); width=tile["stroke"]
        if word in {"left","right","up","down"}:
            base=[(-15,0),(-3,-12),(-3,-5),(14,-5),(14,5),(-3,5),(-3,12)]
            if word=="right":base=[(-a,b) for a,b in base]
            elif word=="up":base=[(-b,a) for a,b in base]
            elif word=="down":base=[(b,-a) for a,b in base]
            d.polygon([(cx+a,cy+b) for a,b in base],fill=ink)
        elif word=="yes":d.line([(cx-13,cy),(cx-4,cy+10),(cx+14,cy-12)],fill=ink,width=width,joint="curve")
        elif word=="no":
            d.line([(cx-11,cy-11),(cx+11,cy+11)],fill=ink,width=width)
            d.line([(cx-11,cy+11),(cx+11,cy-11)],fill=ink,width=width)
        elif word=="go":d.polygon([(cx-10,cy-14),(cx+14,cy),(cx-10,cy+14)],fill=ink)
        else:d.rectangle((cx-11,cy-11,cx+11,cy+11),fill=ink)
    return image


def answer(panel, heard_word, task, position=0):
    """Symbolic labels only; model forward/predict never calls this function."""
    if task=="heard_word":return heard_word
    if task=="tile_word":return panel["tiles"][position]["word"]
    target=OPPOSITE[heard_word] if task.startswith("opposite_") else heard_word
    found=next((i for i,t in enumerate(panel["tiles"]) if t["word"]==target),None)
    if task=="present":return "yes" if found is not None else "no"
    if task=="absent":return "no" if found is not None else "yes"
    if found is None:return "not present"
    return panel["tiles"][found]["color"] if task.endswith("color") else POSITIONS[found]


def answer_space(task):
    if task.endswith("color"):return [*COLORS,"not present"]
    if task.endswith("position"):return [*POSITIONS,"not present"]
    if task in {"present","absent"}:return ["yes","no"]
    return WORDS.copy()

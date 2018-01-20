import argparse
import random
from enum import Enum, auto
import os

import itertools
import pyautogui
import pyscreenshot
import time

import sys
from fann2 import libfann
from PIL import Image


class Marble(Enum):
    none = -1
    Salt = 0
    Air = auto()
    Fire = auto()
    Water = auto()
    Earth = auto()
    Vitae = auto()
    Mors = auto()
    Quintessence = auto()
    Quicksilver = auto()
    Lead = auto()
    Tin = auto()
    Iron = auto()
    Copper = auto()
    Silver = auto()
    Gold = auto()

    def symbol(self):
        if self.value is self.none.value:
            return "-"
        if self.value in range(self.Quicksilver.value, self.Gold.value + 1):
            return self.name[0].upper()
        else:
            return self.name[0].lower()


FIELD_X = 1052
FIELD_DX = 66
FIELD_Y = 221
FIELD_DY = 57
FIELD_SIZE = 6

SCAN_RADIUS = 17
MARBLE_BY_SYMBOL = dict(zip([Marble.symbol(e) for e in Marble], [e.name for e in Marble]))


class State:
    # state = dict.fromkeys([e.name for e in Marble], ())
    state = dict()

    def __str__(self):
        py = -1
        for (x, y) in FIELD_POSITIONS:
            if y != py:
                if y > 0: sys.stdout.write('\n')
                sys.stdout.write(" " * abs(y - FIELD_SIZE + 1))
            sys.stdout.write(self.state.get((x, y), "-"))
            sys.stdout.write(' ')
            py = y
        return ''


def field_positions():
    d = FIELD_SIZE - 1
    result = []
    for y in range(-d, d + 1):
        for x in range(-d, d + 1):
            if not abs(y - x) > d:
                result.append((x + d, y + d))
    return result


def pixels_to_scan():
    pxs = []
    for dy in range(-SCAN_RADIUS + 1, SCAN_RADIUS):
        for dx in range(-SCAN_RADIUS + 1, SCAN_RADIUS):
            if (abs(dx) + abs(dy)) * 2 > SCAN_RADIUS * 3:
                continue
            if dy * 2 < -SCAN_RADIUS and dx * 5 < SCAN_RADIUS:
                continue
            else:
                pxs.append((dx, dy))
    return pxs


FIELD_POSITIONS = field_positions()
PIXELS_TO_SCAN = pixels_to_scan()
ANN = libfann.neural_net()


def img_pos(x, y):
    return FIELD_X + FIELD_DX * (x * 2 - y) / 2, FIELD_Y + FIELD_DY * y


def lightness_at(img, x, y):
    _min, _max = img.getpixel((x, y))
    return _min / _max


def edges_at(img, x, y):
    def sorting(d):
        dx, dy = d

        def neigh(dd):
            ddx, ddy = dd
            return lightness_at(img, x + dx + ddx, y + dy + ddy)

        _neigh = list(map(neigh, [(-1, 0), (0, -1), (1, 0), (0, 1)]))
        _max, _min = max(_neigh), min(_neigh)
        return -(_max - _min)

    result = sorted(PIXELS_TO_SCAN, key=sorting)
    return result[:int(len(result) / 4)]


TRAIN_CASES = dict.fromkeys([e.name for e in Marble], [])


def sample():
    for i in range(1, 7):
        img = Image.open(os.path.join("sample", str(i) + ".png")).convert('LA')
        samples = list(itertools.chain.from_iterable(
            [lines.split() for lines in open(os.path.join("sample", str(i) + ".txt"), "r").readlines()]))
        for j, (pos, symbol) in enumerate(zip(FIELD_POSITIONS, samples)):
            marble = MARBLE_BY_SYMBOL[symbol]
            edge_pixels = set(edges_at(img, *img_pos(*pos)))
            TRAIN_CASES[marble] = TRAIN_CASES[marble] + [edge_pixels]


def train():
    for i in range(len(FIELD_POSITIONS * 15)):
        marble = random.choice([e.name for e in Marble])
        edge_pixels = random.choice(TRAIN_CASES[marble])
        a = list(map(lambda x: 1.0 if x in edge_pixels else 0.0, PIXELS_TO_SCAN))
        b = list(map(lambda x: 1.0 if marble is x else 0.0, [e.name for e in Marble]))
        ANN.train(a, b)


def initMap(img):
    status = State()
    for pos in FIELD_POSITIONS:
        try_edges = edges_at(img, *img_pos(*pos))
        result = ANN.run(list(map(lambda x: 1.0 if x in try_edges else 0.0, PIXELS_TO_SCAN)))
        marble = sorted(list(zip(result, [e.name for e in Marble])), reverse=True)[0]
        status.state[pos] = Marble.symbol(Marble[marble[1]])
    print(status)


def init():
    input_y = [len(PIXELS_TO_SCAN)]
    hidden_y = [int(len(PIXELS_TO_SCAN) / 2), int(len(PIXELS_TO_SCAN) / 4)]
    output_y = [len(Marble)]
    layer = input_y + hidden_y + output_y
    if os.path.exists("network.fann"):
        print("Load Network from network.fann")
        ANN.create_from_file("network.fann")
    else:
        print("Train Network")
        sample()
        ANN.create_standard_array(layer)
        train()
        # ANN.save("network.fann")


def main():
    # print(Marble.symbol(Marble.Fire))
    init()
    initMap(Image.open(os.path.join("sample", "1.png")).convert('LA'))
    # print(pixels_to_scan())
    # print(field_positions())
    # print(img_pos(1, 1))
    pass


if __name__ == '__main__':
    main()

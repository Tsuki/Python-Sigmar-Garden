import argparse
import random
from enum import Enum
import os

import itertools
import pyautogui
import pyscreenshot
import time
from fann2 import libfann
from PIL import Image


class Marble(Enum):
    none = '-'
    Salt = 's'
    Air = 'a'
    Fire = 'f'
    Water = 'w'
    Earth = 'e'
    Vitae = 'v'
    Mors = 'm'
    Quintessence = 'q'
    Quicksilver = 'Q'
    Lead = 'L'
    Tin = 'T'
    Iron = 'I'
    Copper = 'C'
    Silver = 'S'
    Gold = 'G'


FIELD_X = 1052
FIELD_DX = 66
FIELD_Y = 221
FIELD_DY = 57
FIELD_SIZE = 6

SCAN_RADIUS = 17


def field_positions():
    d = FIELD_SIZE - 1
    result = []
    for y in range(-d, d + 1):
        for x in range(-d, d + 1):
            if not abs(y - x) > d:
                result.append((x + d, y + d))
    print(len(result))
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


def img_pos(x, y):
    return FIELD_X + FIELD_DX * (x * 2 - y) / 2, FIELD_Y + FIELD_DY * y


def lightness_at(img, x, y):
    r, g, b, a = img.getpixel((x, y))
    _max, _min = max([r, g, b]), min([r, g, b])
    return (_max + _min) / 2


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
        img = Image.open(os.path.join("sample", str(1) + ".png"))
        samples = list(itertools.chain.from_iterable(
            [lines.split() for lines in open(os.path.join("sample", str(1) + ".txt"), "r").readlines()]))
        for j, (pos, symbol) in enumerate(zip(FIELD_POSITIONS, samples)):
            marble = Marble(symbol)
            edge_pixels = edges_at(img, *img_pos(*pos))
            TRAIN_CASES[marble.name].append(set(edge_pixels))


ANN = libfann.neural_net()


def train():
    for i in range(len(FIELD_POSITIONS * 15)):
        marble = random.choice([e.name for e in Marble])
        edge_pixels = random.choice(TRAIN_CASES[marble])
        a = list(map(lambda x: 1.0 if x in edge_pixels else 0.0, PIXELS_TO_SCAN))
        b = list(map(lambda x: 1.0 if marble is x else 0.0, [e.name for e in Marble]))
        ANN.train(a, b)


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
        ANN.create_standard_array(layer)
        print(ANN)
        train()
        ANN.save("network.fann")


def main():
    # print(pixels_to_scan())
    # print(field_positions())
    # print(img_pos(1, 1))
    sample()
    init()
    pass


if __name__ == '__main__':
    main()

import argparse
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


def edges_at(img, x, y):
    pass


TRAIN_CASES = dict.fromkeys([e.name for e in Marble])


def sample():
    for i in range(1, 7):
        img = Image.open(os.path.join("sample", str(1) + ".png"))
        samples = list(itertools.chain.from_iterable(
            [lines.split() for lines in open(os.path.join("sample", str(1) + ".txt"), "r").readlines()]))
        for j, (pos, symbol) in enumerate(zip(FIELD_POSITIONS, samples)):
            marble = Marble(symbol)
            edge_pixels = edges_at(img, *img_pos(*pos)).to_set
            TRAIN_CASES[marble.name].append(edge_pixels)

        #   img = StumpyPNG.read("samples/#{i}.png")


#   samples = File.read("samples/#{i}.txt").split
#   FIELD_POSITIONS.zip(samples) do |pos, symbol|
#     marble = MARBLE_BY_SYMBOL[symbol]
#     edge_pixels = edges_at(img, *img_pos(*pos)).to_set
#     TRAIN_CASES[marble] << edge_pixels
#   end
# end

def ann():
    if os.path.isfile("network.fann"):
        train_data = libfann.training_data()
        train_data.read_train_from_file(os.path.join("network.fann"))
        return libfann.neural_net()


def main():
    # print(pixels_to_scan())
    # print(field_positions())
    # print(img_pos(1, 1))
    sample()
    # ann()
    pass


if __name__ == '__main__':
    main()

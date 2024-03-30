#  Copyright (c) 2024. Kromek Group Ltd.

from enum import Enum

class Relationships(Enum):
    CONSUMES = 0
    PREDATORPREY = 1

class Size(Enum):
    SMALL = 0
    MEDIUM = 1
    LARGE = 2

class Species(Enum):
    LION = 0
    TIGER = 1
    ELEPHANT = 2
    GIRAFFE = 3
    ZEBRA = 4
    BEE = 5
    ANT = 6
    BUTTERFLY = 7
    DRAGONFLY = 8
    BEETLE = 9
    OAK = 10
    PINE = 11
    ROSE = 12
    SUNFLOWER = 13
    TULIP = 14

class Classification(Enum):
    ANIMAL = 0
    PLANT = 1
    INSECT = 2
    RESOURCE = 3

class Diet(Enum):
    CARNIVORE = 0
    HERBIVORE = 1
    OMNIVORE = 2

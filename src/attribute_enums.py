from enum import Enum

class Relationships(Enum):
    CONSUMES = 0
    PREDATORPREY = 1

class Size(Enum):
    SMALL = 1
    MEDIUM = 2
    LARGE = 3

class Species(Enum):
    LION = 1
    TIGER = 2
    ELEPHANT = 3
    GIRAFFE = 4
    ZEBRA = 5
    BEE = 6
    ANT = 7
    BUTTERFLY = 8
    DRAGONFLY = 9
    BEETLE = 10
    OAK = 11
    PINE = 12
    ROSE = 13
    SUNFLOWER = 14
    TULIP = 15

class Classification(Enum):
    ANIMAL = 1
    PLANT = 2
    INSECT = 3
    RESOURCE = 4

class Diet(Enum):
    CARNIVORE = 1
    HERBIVORE = 2
    OMNIVORE = 3

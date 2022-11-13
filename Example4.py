#!/usr/bin/env python

# ---- MODULE DOCSTRING

__doc__ = """

(C) Hive, Romain Wuilbercq, 2017
     _
    /_/_      .'''.
 =O(_)))) ...'     `.
    \_\              `.    .'''X
                       `..'
.---.  .---..-./`) ,---.  ,---.   .-''-.
|   |  |_ _|\ .-.')|   /  |   | .'_ _   \
|   |  ( ' )/ `-' \|  |   |  .'/ ( ` )   '
|   '-(_{;}_)`-'`"`|  | _ |  |. (_ o _)  |
|      (_,_) .---. |  _( )_  ||  (_,_)___|
| _ _--.   | |   | \ (_ o._) /'  \   .---.
|( ' ) |   | |   |  \ (_,_) /  \  `-'    /
(_{;}_)|   | |   |   \     /    \       /
'(_,_) '---' '---'    `---`      `'-..-'

The Artificial Bee Colony (ABC) algorithm is based on the
intelligent foraging behaviour of honey bee swarm, and was first proposed
by Karaboga in 2005.

Description:
-----------

This example shows how to evolve a famous painting using polygons.

The location of a number of polygons and RGB colors are evolved by an Artificial
Bee Colony algorithm to replicate a famous painting from Henri Matisse.

This example is inspired by a blog post written by Roger Alsing.

Reference:
---------

http://rogeralsing.com/2008/12/07/genetic-programming-evolution-of-mona-lisa/

Dependencies:
------------

- PIL
- sklearn-image
- numpy
- matplotlib

"""

# ---- IMPORT MODULES

# import internal modules

from Hive import Hive
from Hive import Utilities

# import external modules

import numpy as np
import random

from sklearn.metrics import mean_squared_error as mse

try:
    from PIL import ImageChops, Image
except:
    raise ImportError("PIL module not found.")

try:
    import matplotlib.path as mpath
    import matplotlib.pyplot as plt
    import matplotlib as mpl
except:
    raise ImportError("matplotlib module not found.")

try:
    from skimage import color
except:
    raise ImportError("sklearn-image module not found.")

# ---- DEFINE BLANK CANVAS

# define image polygons parameters
nb_solutions = 3
# nb_pts_per_polygon, nb_rgb = 8, 4, 3

def solution_vector():
    """ Creates a polygon. """

    return [int(random.randrange(1,10)), int(random.randrange(1,10)), int(random.randrange(1,10))]

def create_solution_vectors(vector):
    """ Creates an image from a set of polygons. """

    vectors = []
    for _ in range(nb_solutions):
        vectors.append(solution_vector())

    return vectors


# ---- CREATE EVALUATOR

def compare_func(vector):
    # compare input vector to [3,2,1]
    x = -10 if (vector[0] < -10) else vector[0]
    x = 10 if (x > 10) else x
    y = -10 if (vector[1] < -10) else vector[1]
    y = 10 if (y > 10) else y
    z = -10 if (vector[2] < -10) else vector[2]
    z = 10 if (z > 10) else z

    f = (x-1)**2 + (y-2)**2 + (z-3)**2

    return f

def evaluator(vector):
    # print(vector)
    return compare_func(vector)


# ---- SOLVE TEST CASE

def run():

    # creates model
    ndim = int(nb_solutions)
    model = Hive.BeeHive(lower     = [-10]*ndim   ,
                         upper     = [10]*ndim   ,
                         fun       = evaluator  ,
                         numb_bees = 10         ,
                         max_itrs  = 10        ,
                         verbose   = True       ,)

    # runs model
    result = model.run()
    Utilities.ConvergencePlot(result)

    # # saves an image of the end result
    # solution = create_solution_vectors(model.solution)
    # print(solution)


if __name__ == "__main__":
    run()


# ---- END

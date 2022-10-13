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
import cobra
from cobra.io import load_model
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
ox_lb = 0
ox_ub = 10
fitness = 0
nb_solutions = 1
model = load_model("iJO1366")
# nb_pts_per_polygon, nb_rgb = 8, 4, 3

def solution_vector():
    """ Creates a polygon. """

    return [int(random.randrange(ox_lb,ox_ub))]

def create_solution_vectors(vector):
    """ Creates an image from a set of polygons. """
    #Set the objective to the genome scale biomass reactions
    model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").objective_coefficient = 0
    model.reactions.get_by_id("BIOMASS_Ec_iJO1366_WT_53p95M").objective_coefficient = 1.0
    #Set constrants for anaerobic growth in glucose minimal media
    model.reactions.get_by_id("EX_glc__D_e").lower_bound= 18.5
    model.reactions.get_by_id("EX_o2_e").lower_bound = vector[0]
    solution=model.optimize()
    fluxes = []
    for flux in solution.fluxes:
        fluxes.append(flux)
    fluxes.append(solution.objective_value)
    return fluxes


# ---- CREATE EVALUATOR


def compare_func(vector):
    #Set the objective to the genome scale biomass reactions
    model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").objective_coefficient = 0
    model.reactions.get_by_id("BIOMASS_Ec_iJO1366_WT_53p95M").objective_coefficient = 1.0
    #Set constrants for anaerobic growth in glucose minimal media
    model.reactions.get_by_id("EX_glc__D_e").lower_bound= 18.5
    model.reactions.get_by_id("EX_o2_e").lower_bound = vector[0]
    solution=model.optimize()
    # fluxes = []
    # for flux in solution.fluxes:
    #     fluxes.append(flux)
    
    return solution.objective_value

def evaluator(vector):
    print(vector)
    return compare_func(vector)


# ---- SOLVE TEST CASE

def run():

    # creates model
    ndim = int(nb_solutions)
    model = Hive.BeeHive(lower     = [ox_lb]*ndim   ,
                         upper     = [ox_ub]*ndim   ,
                         fun       = evaluator  ,
                         numb_bees = 50         ,
                         max_itrs  = 50        ,
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

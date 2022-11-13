#!/usr/bin/env python

from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

microbes = [
    {
        "Name": 'E.coli',
        "microbeName": 'iJO1366'
    },
    {
        "Name": 'M.tuberculosis',
        "microbeName": 'iEK1008'
    },
    {
        "Name": 'P.putida',
        "microbeName": 'iJN746'
    },
] 

@app.route('/')
def x():
    return 'Home Page'

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

import warnings
warnings.filterwarnings("ignore")

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

try:
    from escher import Builder
except:
    raise ImportError("escher module not found.")

from bs4 import BeautifulSoup

# ---- DEFINE BLANK CANVAS

# define image polygons parameters
ox_lb = -1000
ox_ub = 0
fitness = 0
nb_solutions = 1
model = load_model("iJO1366")
# nb_pts_per_polygon, nb_rgb = 8, 4, 3


def solution_vector():
    """ Creates a polygon. """

    return [int(random.randrange(ox_lb,ox_ub))]

# def create_solution_vectors(vector):
#     """ Creates an image from a set of polygons. """
#     #Set the objective to the genome scale biomass reactions
#     model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").objective_coefficient = 0
#     model.reactions.get_by_id("BIOMASS_Ec_iJO1366_WT_53p95M").objective_coefficient = 1.0
#     #Set constrants for anaerobic growth in glucose minimal media
#     model.reactions.get_by_id("EX_glc__D_e").lower_bound= 18.5
#     model.reactions.get_by_id("EX_o2_e").lower_bound = vector[0]
#     solution=model.optimize()
#     fluxes = []
#     for flux in solution.fluxes:
#         fluxes.append(flux)
#     fluxes.append(solution.objective_value)
#     return fluxes


# ---- CREATE EVALUATOR


def compare_func(vector, met, obj):
    #Set the objective to the genome scale biomass reactions
    # model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").objective_coefficient = 0
    # model.reactions.get_by_id("BIOMASS_Ec_iJO1366_WT_53p95M").objective_coefficient = 1.0
    #Set constrants for anaerobic growth in glucose minimal media
    # model.reactions.get_by_id("EX_glc__D_e").lower_bound= 18.5
    model.reactions.get_by_id(obj).objective_coefficient = 1.0
    model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").objective_coefficient = 0
    model.reactions.get_by_id(met).lower_bound = vector[0]
    solution=model.optimize()
    # fluxes = []
    # for flux in solution.fluxes:
    #     fluxes.append(flux)
    
    return solution.objective_value

def evaluator(vector, met, obj):
    # print(vector)
    return compare_func(vector, met, obj)

def visulaize(vector, met, obj):
    #Set the objective to the genome scale biomass reactions
    # model.reactions.get_by_id("BIOMASS_Ec_iJO1366_core_53p95M").objective_coefficient = 0
    # model.reactions.get_by_id("BIOMASS_Ec_iJO1366_WT_53p95M").objective_coefficient = 1.0
    # #Set constrants for anaerobic growth in glucose minimal media
    # model.reactions.get_by_id("EX_glc__D_e").lower_bound= 18.5
    # model.reactions.get_by_id("EX_o2_e").lower_bound = vector[0]
    model.reactions.get_by_id(obj).objective_coefficient = 1.0
    model.reactions.get_by_id(met).lower_bound = vector[0]
    solution=model.optimize()
    b = Builder(map_name='e_coli_core.Core metabolism', reaction_data=solution.fluxes)
    b.save_html("/home/albee/Documents/GitHub/Capstone/Community-FBA-Hive/Output/output.html")
    return solution.objective_value, solution.fluxes

@app.route('/legoflux', methods=['GET'])
def show():
    model = None
    if 'name' in request.args :
        name = request.args['name']
    else :
        return 'unknown request'

    for x in microbes :
        if x['Name'] == name:
            model = load_model(x['microbeName'])

    if 'metabolite' not in request.args :
            return jsonify(list(model.summary().uptake_flux.reaction))

    if 'metabolite' in request.args:
        met = request.args['metabolite']
    else :
        return 'unknown request'
    if 'lbound' in request.args:
        lb = int(request.args['lbound'])
    else :
        return 'unknown request'
    if 'ubound' in request.args:
        ub = int(request.args['ubound'])
    else :
        return 'unknown request'
    # if 'objective' in request.args:
    #     obj = request.args['obj']
    # else :
    obj = "BIOMASS_Ec_iJO1366_core_53p95M"
    
    # fba(met,lb,ub,obj)
    objVal, fluxes, solution, mapOutput = run(met,lb,ub,obj)
    return jsonify({"growthRate": objVal, "Fluxes": list(fluxes), "optimalValue": solution, "mapOutput": mapOutput})

# ---- SOLVE TEST CASE

def run(met,lb,ub,obj):

    # creates model
    ndim = int(nb_solutions)
    model = Hive.BeeHive(lower     = [lb]*ndim   ,
                         upper     = [ub]*ndim   ,
                         met = met,
                         obj = obj,
                         fun       = evaluator   ,
                         numb_bees = 10         ,
                         max_itrs  = 10        ,
                         verbose   = True       ,)

    # runs model
    model.run()
    # result = model.run()
    # Utilities.ConvergencePlot(result)

    # # saves an image of the end result
    # solution = create_solution_vectors(model.solution)
    # print(solution)
    print("Solution:", model.solution)
    objVal, fluxes = visulaize(model.solution, met, obj)
    # Opening the html file
    HTMLFile = open("./Output/output.html", "r")
    
    # Reading the file
    output = HTMLFile.read()
    return objVal, fluxes, model.solution, output


if __name__ == "__main__":
    app.debug=True
    app.run(host='localhost', port = 5000)
    # run()

# ---- END

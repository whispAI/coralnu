#   [ALPHA] whisp/core — lightweight flask API for coref resolution
#   -------------------------------------------------------------------
#   Copyright (C) 2020-2022 Whisp Limited
#   Endpoint Base URL: https://api.whisp.dev
#   Documentation: https://whispai.atlassian.net/wiki/spaces/WHISPDEV/
#
#   ‼️ INTERNAL USE ONLY — DO NOT DISTRIBUTE ‼️
#   =========================================
#   -------------------------------------------------------------------
#
#   Last modified: 2022-OCT-18
#
#   Contributor(s): 
#       @lucafrost ~ Luca J Frost
#       @goodaytar ~ Lee Dudek
#       ** a significant portion of this project was enabled by
#          the work of @mmaslankowska-neurosys !!
#
#   -------------------------------------------------------------------
#
#   TODO: implement a proper logging system @lucafrost


# IMPORTS ######################################################################################### 

import os
import sys
from flask import Flask, request, jsonify

# models -------------------------------------------------------------------------------
import spacy
import neuralcoref
from allennlp.predictors.predictor import Predictor
from utils import load_models, print_clusters
from utils import IntersectionStrategy, StrictIntersectionStrategy, PartialIntersectionStrategy, FuzzyIntersectionStrategy


# logging & utils ---------------------------------------------------------------------------------
import logging
import json
import requests


# FLASK ###########################################################################################

## initialize the flask app
app = Flask(__name__)


# LOAD MODELS #####################################################################################

predictor, nlp = load_models()


# FLASK ROUTES ####################################################################################

@app.route("/")
def hello():
    return "<h1 style='color:blue'>whisp.dev apis</h1>"


# coref resolution ------------------------------------------------------------------------

@app.route('/coref', methods=['POST'])
def coref():
    """
    Resolves coreferences using AllenNLP and spaCy neuralcoref 
    with a fuzzy intersection strategy.
    """
    # get the request data
    data = request.get_json()
    text = data['text']
    
    fuzzy = FuzzyIntersectionStrategy(predictor, nlp)
    clusters = fuzzy.clusters(text)
    resolved = fuzzy.resolve_coreferences(text)
    
    output = {
        "method": "fuzzyIntersection",
        "clusters": clusters,
        "resolved": resolved
    }
    
    return jsonify(output)
    

# RUN #############################################################################################

if __name__ == "__main__":
    app.run(host='0.0.0.0')
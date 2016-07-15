# General Information

This repo is DEPRECATED and is not supported anymore.

This repository contains implementations for a variety of AKBC 
(Automatic Knowledge Base Completion) scoring models. In its 
current version only [scoring models](model/models.py) for triples 
(subject, predicate, object) are supported.

It also supports [compositional scoring models](model/comp_models.py)
using various [composition functions](model/comp_functions.py).

# Training and Supported Datasets

Currently only the [FB15k-237](http://research.microsoft.com/en-us/downloads/3a9bf02d-b791-4e95-b88d-389feef3e421/) 
dataset is supported, which was introduced [here](http://research.microsoft.com/apps/pubs/default.aspx?id=249127) 
and [here](http://research.microsoft.com/apps/pubs/default.aspx?id=254916).

For training see command line options of [train.py](train.py). 

# Installation

requires: tensorflow, pandas



For further questions, please don't hesitate to contact.

Contact: dirk.weissenborn@dfki.de



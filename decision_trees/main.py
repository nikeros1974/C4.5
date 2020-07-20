#!/usr/bin/env python
from decision_trees.c45 import C45
from decision_trees.visualizer import Visualizer
import logging

logging.basicConfig(level=logging.INFO)

c1 = C45("../data/iris/iris.data", "../data/iris/iris.names")
c1.fetch_data()

c1.pre_process_data()
c1.generate_tree()

printer = Visualizer()

printer.print(c1)

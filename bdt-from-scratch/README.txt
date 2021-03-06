SUMMARY
=======
There are 2 types of classifiers used:
	Belief Decision Tree (BDT)
	Selective Iterative BDT

There are three types of distributions these classifiers are run on:
	Five label distributions
	Two label weighted distributions
	Two label unweighted distributions

DATA
====
We have multiple datasets that were used at different points of the development process. We first tried balancing data by looking at the mean of the radiologist ratings (meanBalanced), but moved to looking at the mode of the ratings (modeBalanced).

FIGURES
=======
All related figures, graphs, tables, and diagrams are contained in this folder.

OUTPUT
======
Four output files are generated each time one of these files is run, which can be found in the output folder:
	BDT Output#.txt
	TestOutput#.csv
	TrainOutput.csv
	tree.txt (or trees#.txt)

SIMPLEBDT
=========
This is our original implementation of a Belief Decision Tree without any extra features.

SRC
===
For the two classifiers and three distribution types listed in SUMMARY, we have six different files to test them with, each of which was run on our ModeBalanced_170_LIDC_809_Random dataset.

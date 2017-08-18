# REU
Word done at Medix NSF REU in Summer 2017

The file structure is layed out as following way:

Project Directory:

An Evaluation of Consensus Techniques  -- The finalizied abstract submitted to SPIE
Midterm Presentation
Final Presentation
Tutorial.ipynb -- Jupyter notebook tutorial of project

2016_REU_Documents -- Predecessing work done in the previous year REU

REU Papers -- Most of the scholarly papers collected for research on
              this project

bdt-from-scratch:

    data -- the data used for training the BDT, most used is the
            modeBalanced Directory

    figures -- feature boxplots generated to attempt to find one
               discriminating feature between typical and atypical  
               nodules

    output -- directory where output files from BDT training and testing
              are stored

    src:
          cross_conf --

          five_label: source code mostly used in 2017 REU

                    BDT_FiveRating_AT.py  -- Belief Decision Tree recording Agreement and Typicality 
                    SIC_BDT_FiveRating.py -- "SIC" stands for selective iterative classification
                    Std_Decision_Tree.py -- Standard decision tree from python sklearn
                    BDT_FiveRating_CP.py -- Belief Decision Tree that produces Confidence and 
                                            Credibility with Conformal Prediction

                    BDT_FiveRating_One.py -- "One" referring to one train/test split of 70/30

                    BDT_FiveRating_One_Fair_Shot.py -- "Fair Shot" meaning accuracy is based on 
                                                        selecting the most probable malignancy class
                    Std_Decision_Tree_One.py

                    tree_files -- All testing sets and training sets for all kfolds along with 
                                  the trees produced for each k-fold


          simpleBDT -- directory where basic BDT structure can be generated

          two_label -- finalized source from novel method from 2016 REU

    tools: tools used through the course of the program

          LDA -- Linear Discriminant Analysis on most discerning feature of atypical nodules
          MDS -- Mulidimensional Scaling for ^^
          distance_boxplot -- typcical vs. atypical case boxplots based on sum of pairwise distances

lectures -- notes taken on lectures presenented during this REU program

literature reviews -- a few structured literature reviews taken from
                      papers from the REU Papers directory

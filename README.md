Hello!

Project Authors: Andrew Cohn, Bryan Bielawa, Ben Seckeler, Noah Sher
Class: CSC 396
Date: Fall 2024

This is the readme for our LLM Detection Project.

All of our models are stored in src/models 
To run one of these models please simply compile and run every cell of the Jupyter Notebook file.

We generally use these libraries and thus you will need to have them installed/available on your system:
 * Gensim
 * numpy
 * pandas
 * sklearn
 * torch
 * random
 * matplotlib
 * sys
 * codecs
 * nltk
 * os

Additionally we have a src/utils directory which serves as housing for our result recording code, and can be used for additional model testing 



For our raw data csv we used this schema
LABEL, SOURCE, TEXT

The LABEL indicates if the text is human written or not with 0 = Human written and 1 = ai written text

The SOURCE indicates where the data was gathered, this was a precautionary value and should be used if and when the certain data needs to be tracked. Luckily we did not find any need to utilize this value for our project, we would have used it if we found any issue with a certain data source.

The TEXT column is used for the training/testing text that the models would train on. TLDR this is where the essays go.



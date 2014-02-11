# How to do
preparation: download all files with extension .txt, .cpp, .h and makefile, then put them into a same folder.

build command: make

run command: ./proc

# Inputs: No input required.

# Outputs: The heatmap, and highest/lowest accuracies.

# What is it: This program runs 10-fold cross validation experiment on the CU dataset, and prints the heatmap of different model parameter combinations.

# How it works: 

It first divides the entire dataset into 10 disjointed partitions with each partition having approximately the same percentage of samples for each class. Then, the program picks up one partition as testing data and the rest as training data based on which our model is trained. The program is trained by first selecting top several attributes with highest SVM coefficients of linear SVM, followed by graph construction based on Pearson correlation and graph training based on sovling linear equation systems. At the prediction stage, we classify the testing samples based on the minimum reconstruction error calculated from graphs of different classes. The training and testing procedures are run 10 times with each time using different partitions as testing, and the total accuracies are reported as the mean accuracy of different folds.

# Known problems:

1) when threshold for correlation is too large, the linear equation system becomes under determined due to the number of unknown variables is more than the number of equations. (Currently I introduced a penalty on the solution so as to get a stable solution).

2) when the threshold becomes too small howevere, many vertices in the graph becomes totally isolated (no neighbor vertex). This will generate artifacts since for these vertices, at the testing stage no prediction is available for the corresponding attributes. (Currently I constrained the minimum degree of a graph to solve this problem)


# Edit by Dihong

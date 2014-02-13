/* 
 * File:   common.h
 * Author: dihong
 *
 * Created on February 6, 2014, 9:36 PM
 */

#ifndef COMMON_H
#define	COMMON_H

#define Pearson

#include <cstdlib>
#include "svm.h"
#include <vector>
#include <cstdio>
#include<string>
#include<memory.h>
#include<cmath>
#include<algorithm>
#include<time.h>
#include<iostream>
#include <sys/time.h>
#include <unistd.h>

typedef struct DATASET { //This is used to describe the entire dataset.
    float* feat; //features organized as NxP array (row-major)
    int* label; //1xN label for 1,...,C where C is the total number of classes.
    int N, P; //N is the total number of samples, and P is the dimension of features.
    int num_class; //number of classes.

    DATASET() {
        feat = 0;
        label = 0;
    }

    ~DATASET() {
        if (label) delete label;
        if (feat) delete feat;
    }
} DATASET;

typedef struct MODEL_PARAM { //This is used to describe the training/testing dataset, where dimension has been reduced.
    float* feat1; //N1xT matrix (containing reduced-dimension training features) where N1 is the number of training samples in total, T is the number of selected attributes.
    float* feat2; //N2xT matrix where N2 is the number of testing samples, containing testing samples.
    int N1, N2; //#training and #testing, respectively.
    int T; //#selected attributes.
    int* L1; //1xN1: the class label for the training samples. the label must be 1,2,...,C.
    int* L2; //1xN2: similarly to L1, for testing samples.
    float eps; //correlation threshold.
    int degree; //number of neighbors.
    int* conmat; //the confusion matrix for testing samples.
    int num_class; //number of classes.
    bool valid; //true if the GRAPH is determined [coefficients are well determined], false otherwise.

    MODEL_PARAM() {
        valid = true;
        feat1 = 0;
        feat2 = 0;
        L1 = 0;
        L2 = 0;
    }

    ~MODEL_PARAM() {
        if (feat1) delete feat1;
        if (feat2) delete feat2;
        if (L1) delete L1;
        if (L2) delete L2;
        if (conmat) delete conmat;
    }
} MODEL_PARAM;

typedef struct VERTEX { //the vertex of the graph.
    int nn; //#neighbors, i.e. vertices connecting to this vertex. 0,1,2,...
    float* coef; //the coefficients of the vertex.
    int * neighbors; //index for neighbors of the vertex.

    VERTEX() {
        coef = 0;
        neighbors = 0;
        nn = 0;
    }

    ~VERTEX() {
        if (coef) delete coef;
        if (neighbors) delete neighbors;
    }
} VERTEX;

typedef struct GRAPH { //the graph, one graph for each class.

    GRAPH() {
        vertices = 0;
        samples = 0;
        min_deg = -1;
        max_deg = -1;
        num_edge = -1;
        ns = -1;
        T = -1;
    }
    VERTEX* vertices; //1xT marix containing the the vertices of the graph, where D is the #selected attributes (equal to #vertices).
    int min_deg; //minimum degree of the vertex.
    int max_deg;
    int num_edge; //number of edges in the graph.
    int ns; //number of training samples associated with the graph
    int T; //the dimension of the features.
    int num_isolated; //#vertices isolated.
    float* samples; //ns x T matrix for selected training samples for the graph [Note that it contains training samples only for one class.]

    ~GRAPH() {
        delete samples, vertices;
    }
} GRAPH;

int* select_features_one_vs_rest_svm(const DATASET& data, int* flag);

std::vector<GRAPH*> graph_construction(MODEL_PARAM* param);

void learn_coefficients(std::vector<GRAPH*>& Gs);

int* predict(const std::vector<GRAPH*>& Gs, float* test, int nt);

void show(double* cc, int D1, int D2);

void show(float* cc, int D1, int D2);

void show(int* cc, int D1, int D2);

void show(GRAPH* G);


void quick_sort(float* arr, int N, int* order, float* sorted = 0, bool descend = true);
void quick_sort(int* arr, int N, int* order, int* sorted = 0, bool descend = true);
void quick_sort(double* arr, int N, int* order, double* sorted = 0, bool descend = true);

bool solve_least_squares(double*A, float*x, float*b, int n);

#endif	/* COMMON_H */

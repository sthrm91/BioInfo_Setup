/* 
 * File:   main.cpp
 * Author: dihong
 *
 * Created on February 6, 2014, 1:50 PM
 */


#include"common.h"

using namespace std;

void print_null(const char *s) {
}

void show(GRAPH* G) {
    puts("\n\n=========Graph information========");
    if (G->min_deg != -1) printf("min_deg: %d\n", G->min_deg);
    else printf("min_deg: N/A\n");
    if (G->min_deg != -1) printf("max_deg: %d\n", G->max_deg);
    else printf("max_deg: N/A\n");
    if (G->ns != -1) printf("ns: %d\n", G->ns);
    else printf("ns: N/A\n");
    if (G->num_edge != -1) printf("num_edge: %d\n", G->num_edge);
    else printf("num_edge: N/A\n");
    if (G->T != -1) printf("T: %d\n", G->T);
    else printf("T: N/A\n");
    float* ptr = 0, pav;
    int T = G->T;
    if (G->vertices) {
        char reconstruct_err[10] = "N/A";
        puts("-------------------------------");
        puts("Vertices information:");
        for (int i = 0; i < G->T; i++) {
            //use the weights to reconstruct the training samples.
            if (G->vertices[i].coef) {
                float se = 0; //squared reconstruction error.
                float L2 = 0;
                for (int n = 0; n < G->ns; n++) { //for each training sample
                    ptr = G->samples + n*T;
                    pav = 0;
                    for (int k = 0; k < G->vertices[i].nn; k++) //for each neighbor.
                        pav += ptr[G->vertices[i].neighbors[k]] * G->vertices[i].coef[k]; //linear combination of the connecting components to reconstruct the attribute.
                    pav += G->vertices[i].coef[G->vertices[i].nn]; //constant.
                    se += (ptr[i] - pav)*(ptr[i] - pav);
                    L2 += ptr[i] * ptr[i];
                }
                se = sqrt(se); //sqared root.
                L2 = sqrt(L2);
                sprintf(reconstruct_err, "%.3f %.3f", se, L2);
            }
            //-------------------
            printf("ID: %03d [%s]\n", i, reconstruct_err);
            printf("Neighbors:");
            if (G->vertices[i].neighbors)
                for (int j = 0; j < G->vertices[i].nn; j++) {
                    printf(" %03d  ", G->vertices[i].neighbors[j]);
                } else
                printf("N/A");
            puts("");
            printf("Weights: ");
            if (G->vertices[i].coef)
                for (int j = 0; j <= G->vertices[i].nn; j++) {
                    if (G->vertices[i].coef[j] < 0)
                        printf("%.2f ", G->vertices[i].coef[j]);
                    else
                        printf(" %.2f ", G->vertices[i].coef[j]);
                } else
                printf("N/A");
            puts("");

        }
        if (G->ns < 40 && G->T < 20) {
            puts("-------------------------------");
            puts("Sample information:");
            for (int j = 0; j < G->T; j++)
                printf("  %03d ", j);
            puts("");
            for (int i = 0; i < G->ns; i++) {
                for (int j = 0; j < G->T; j++) {
                    if (G->samples[i * G->T + j] < 0)
                        printf("%.2f ", G->samples[i * G->T + j]);
                    else
                        printf("% .2f ", G->samples[i * G->T + j]);
                }
                puts("");
            }
        }
    } else {
        puts("Vertices information unavailable.");
    }
    fflush(stdout);
}

void show(double* cc, int D1, int D2) {
    puts("------------");
    for (int d1 = 0; d1 < D1; d1++) {
        for (int d2 = 0; d2 < D2; d2++) {
            printf("%.4f ", cc[d1 * D2 + d2]);
        }
        puts("\n");
    }
    fflush(stdout);
}

void show(float* cc, int D1, int D2) {
    puts("------------");
    for (int d1 = 0; d1 < D1; d1++) {
        for (int d2 = 0; d2 < D2; d2++) {
            printf("%.4f ", cc[d1 * D2 + d2]);
        }
        puts("\n");
    }
    fflush(stdout);
}

void show(int* cc, int D1, int D2) {
    for (int d1 = 0; d1 < D1; d1++) {
        for (int d2 = 0; d2 < D2; d2++) {
            printf("%d ", cc[d1 * D2 + d2]);
        }
        puts("\n");
    }
    fflush(stdout);
}

bool mycompfunc_double(const pair<double, int>& l, const pair<double, int>& r) {
    return l.first > r.first;
}

bool mycompfunc_int(const pair<int, int>& l, const pair<int, int>& r) {
    return l.first > r.first;
}

bool mycompfunc_float(const pair<float, int>& l, const pair<float, int>& r) {
    return l.first > r.first;
}

void quick_sort(float* arr, int N, int* order, float* sorted, bool descend) {
    vector< pair<float, int> > WI;
    pair<float, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_float);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        } else
        for (int i = N - 1; i >= 0; i--) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
}

void quick_sort(int* arr, int N, int* order, int* sorted, bool descend) {
    vector< pair<int, int> > WI;
    pair<int, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_int);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        } else
        for (int i = N - 1; i >= 0; i--) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
}

void quick_sort(double* arr, int N, int* order, double* sorted, bool descend) {
    vector< pair<double, int> > WI;
    pair<double, int> val_ind;
    for (int i = 0; i < N; i++) {
        val_ind.first = arr[i]; //value.
        val_ind.second = i; //index.
        WI.push_back(val_ind);
    }
    sort(WI.begin(), WI.end(), mycompfunc_double);
    if (descend)
        for (int i = 0; i < N; i++) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        } else
        for (int i = N - 1; i >= 0; i--) {
            if (sorted) sorted[i] = WI[i].first;
            order[i] = WI[i].second;
        }
}

svm_model* svm_train_2c(float* feat, int* label, int* flag, int N, int P, svm_node* x_space) {
    /* two-class svm training.
     * feat [in]: features organized as NxP array (row major) with each row contains one subject, totally N subjects.
     * label [in]: 1xN array where label[n] represents the class id of the n-th subject. i.e. label[i] = 1 means the i-subject belongs to the class 1. label must be either +1 or -1.
     * flag [in]: 1xN array where flag[n] = 0 if the n-th subject is used for training, otherwise flag[n] = 1. Note that samples with flag = 0 will not be used for svm training.
     * N [in]: the number of subjects in total.
     * P [in]: the dimension of each feature.
     * [return]: return the trained svm model if success, return NULL otherwise.
     */
    //setting svm model parameters.
    struct svm_parameter param;
    /*	"options:\n"
            "-s svm_type : set type of SVM (default 0)\n"
            "	0 -- C-SVC		(multi-class classification)\n"
            "	1 -- nu-SVC		(multi-class classification)\n"
            "	2 -- one-class SVM\n"
            "	3 -- epsilon-SVR	(regression)\n"
            "	4 -- nu-SVR		(regression)\n"
            "-t kernel_type : set type of kernel function (default 2)\n"
            "	0 -- linear: u'*v\n"
            "	1 -- polynomial: (gamma*u'*v + coef0)^degree\n"
            "	2 -- radial basis function: exp(-gamma*|u-v|^2)\n"
            "	3 -- sigmoid: tanh(gamma*u'*v + coef0)\n"
            "	4 -- precomputed kernel (kernel values in training_set_file)\n"
            "-d degree : set degree in kernel function (default 3)\n"
            "-g gamma : set gamma in kernel function (default 1/num_features)\n"
            "-r coef0 : set coef0 in kernel function (default 0)\n"
            "-c cost : set the parameter C of C-SVC, epsilon-SVR, and nu-SVR (default 1)\n"
            "-n nu : set the parameter nu of nu-SVC, one-class SVM, and nu-SVR (default 0.5)\n"
            "-p epsilon : set the epsilon in loss function of epsilon-SVR (default 0.1)\n"
            "-m cachesize : set cache memory size in MB (default 100)\n"
            "-e epsilon : set tolerance of termination criterion (default 0.001)\n"
            "-h shrinking : whether to use the shrinking heuristics, 0 or 1 (default 1)\n"
            "-b probability_estimates : whether to train a SVC or SVR model for probability estimates, 0 or 1 (default 0)\n"
            "-wi weight : set the parameter C of class i to weight*C, for C-SVC (default 1)\n"
            "-v n: n-fold cross validation mode\n"
            "-q : quiet mode (no outputs)\n"*/
    // default values
    param.svm_type = 0;
    param.kernel_type = 0;
    param.degree = 3;
    param.gamma = 1.0 / N;
    param.coef0 = 0;
    param.nu = 0.5;
    param.cache_size = 100;
    param.C = 1;
    param.eps = 1e-3;
    param.p = 0.1;
    param.shrinking = 1;
    param.probability = 0;
    param.nr_weight = 0;
    param.weight_label = NULL;
    param.weight = NULL;

    svm_set_print_string_function(print_null);

    //setting the svm problem to be solved.
    struct svm_problem prob;
    int cnt_train_samples = 0; //number of samples used as training.
    prob.y = new double [N]; //label of the training feat.
    prob.x = new struct svm_node * [N];
    for (int n = 0; n < N; n++) { //scan the input features.
        if (flag[n] == 0) { //it is a sample used for training.
            for (int i = 0; i < P; i++) {
                x_space[cnt_train_samples * (P + 1) + i].index = i + 1; // the index of the feature, i.e. 1,2,...,P
                x_space[cnt_train_samples * (P + 1) + i].value = feat[n * P + i];
            }
            x_space[cnt_train_samples * (P + 1) + P].index = -1; //ending tag.
            prob.y[cnt_train_samples] = label[n]; //label for this training sample.
            prob.x[cnt_train_samples] = x_space + cnt_train_samples * (P + 1); //assign the starting address of feature.
            cnt_train_samples++;
        }
    }
    prob.l = cnt_train_samples; //set the number of samples used for training.

    //check the validity of the svm parameters.
    const char* err_msg = svm_check_parameter(&prob, &param);
    if (err_msg) {
        delete prob.x;
        delete prob.y;
        delete x_space;
        printf("ERROR: sv\n", err_msg);
        return NULL;
    }
    //training svm.
    svm_model* model = svm_train(&prob, &param);
    //for(int i = 0;i<cnt_train_samples;i++)
    //printf("[%d %d]\n",(int)svm_predict(model,prob.x[i]),(int)prob.y[i]);

    //clear up and return.
    delete prob.x;
    delete prob.y;
    return model;
}

int* select_features_one_vs_rest_svm(const DATASET& data, int* flag) {
    /*this function compute the significance of different features (i.e. genes) based on 1-vs-rest dividing scheme.
     *[return]: the index of features in decreasing significance order.
     *flag[in]: indicates which part of data will be used for svm training. 0 for training 1 for testing.
     *data[in]: the DATASET describing the entire data.
     */
    int C = data.num_class;
    int* L = new int [data.N]; // +1/-1 labels.
    double* W = new double [data.P]; //squared weights of each attribute.
    memset(W, 0, sizeof (double)*data.P);
    double* w = new double [data.P];
    svm_node* x_space = new struct svm_node [data.N * (data.P + 1)]; //containing the feature matrix. [with -1 ending tag]
    for (int i = 0; i < C; i++) { //run the svm for C-1 times where C is the number of classes.
        for (int n = 0; n < data.N; n++) {
            if (data.label[n] == i + 1)
                L[n] = 1; //belong to +1
            else
                L[n] = -1; //the rest: belong to -1
        }

        svm_model* model = svm_train_2c(data.feat, L, flag, data.N, data.P, x_space);
        //compute the significance.
        int nsv = model->l; //#sv
        double** sv_coef = model->sv_coef; //sv[0][l]
        struct svm_node** SVs = model->SV;
        memset(w, 0, sizeof (double)*data.P);
        for (int p = 0; p < data.P; p++)
            for (int i = 0; i < nsv; i++)
                w[p] += SVs[i][p].value * sv_coef[0][i];

        //accumulate the w: W[i] = sqrt(sum_c(w[i,c]*w[i,c]))
        for (int p = 0; p < data.P; p++)
            W[p] += w[p] * w[p];
    }

    for (int p = 0; p < data.P; p++) W[p] = sqrt(W[p]);
    //sort the attributes according to the descending orders of W.
    int* indice_dec_ordered = new int [data.P];

    quick_sort(W, data.P, indice_dec_ordered);

    delete L, W, w, x_space;
    return indice_dec_ordered;
}

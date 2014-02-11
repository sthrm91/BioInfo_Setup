/* 
 * File:   main.cpp
 * Author: dihong
 *
 * Created on February 6, 2014, 1:23 PM
 */

#include"common.h"
#include"parallel.h"  

struct timeval begin_ts, end_ts;  //for time elapsed measurement.

#define tic gettimeofday(&begin_ts, NULL);

#define toc gettimeofday(&end_ts, NULL); \
    	printf("Elapsed time: %ld seconds\n\n", end_ts.tv_sec-begin_ts.tv_sec);

using namespace std;

DATASET* load_features(string file_name) {
    //this function loads features into memory, and return DATASET describing the file.
    //the data file should be organized as follow:
    /*
     * P N
     * label1 feature1
     * label2 feature2
     * label3 feature3
     * ...
     * The P is the dimension of features, and N is the total number of subjects in the dataset. label is 1,...,C indicates the class of the subject, and feature is a row vector of 1xN.
     */
    FILE* fp = fopen(file_name.c_str(), "r");
    if (!fp) {
        printf("Failed to open the file: %s\n", file_name.c_str());
        exit(-1);
    }
    DATASET* ds = new DATASET;
    char buf[1000000]; //buffer containing each line of the file.
    if (!fgets(buf, sizeof (buf), fp)) {
        printf("Failed to read data.\n");
        exit(-1);
    }
    sscanf(buf, "%d %d\n", &ds->P, &ds->N);
    ds->feat = new float [ds->N * ds->P];
    ds->label = new int [ds->N];
    int nc = 0; //number of classes.
    char* pch = 0;
    for (int n = 0; n < ds->N; n++) {
        if (!fgets(buf, sizeof (buf), fp)) {
            printf("Failed to read data.\n");
            exit(-1);
        }
        pch = strtok(buf, " "); //split the data
        sscanf(pch, "%d", ds->label + n); //read the label.
        if (ds->label[n] > nc)
            nc = ds->label[n];
        for (int i = 0; i < ds->P; i++) {
            pch = strtok(NULL, " ");
            if (!pch) {
                printf("Failed to read data.\n");
                puts(buf);
                exit(-1);
            }
            sscanf(pch, "%f", ds->feat + n * ds->P + i);
        }
    }
    ds->num_class = nc;
    printf("\nSuccessfully loaded dataset '%s' which has %d samples of dimension %d. Total number of classes is %d.\n", file_name.c_str(), ds->N, ds->P, ds->num_class);
    fflush(stdout);
    return ds;
}

int* generate_cross_validation_flags(int K, DATASET* ds) { //generate K fold flags, label is the label of the entire dataset. N is the number of samples in the dataset.
    int N = ds->N;
    int* label = ds->label;
    if (K > N || K < 1) {
        printf("\n[Error] - You have only %d samples, can't do %d fold cross validation.\n", N, K);
        exit(-1); //not enough samples for K fold cross validation.
    }

    int* F = new int [K * N]; //flags. totally K folds, each fold has N samples.
    memset(F, 0, sizeof (int)* K * N);

    int num_hold_out = N / K; //total number of samples hold out for testing.
    if (K == 1)
        num_hold_out = N / 2; //hold out half.
    int* num_smpl_per_class = new int [N]; //number of samples of each class.
    memset(num_smpl_per_class, 0, sizeof (int)*N);
    for (int n = 0; n < N; n++)
        num_smpl_per_class[label[n] - 1]++;
    vector<int> num_hold_out_per_class;
    for (int i = 0; num_smpl_per_class[i] != 0; i++)
        num_hold_out_per_class.push_back((float) num_hold_out * num_smpl_per_class[i] / N);
    ds->num_class = num_hold_out_per_class.size(); ///#classes in the data.
    for (int k = 0; k < K; k++) {
        for (int c = 0; c < num_hold_out_per_class.size(); c++) { //for each class.
            //find the address of first k*num_hold_out_per_class[c] samples, and reserve the subsequent num_hold_out_per_class[i] samples for testing.
            int begin_index_of_first_smpl_in_c = 0;
            while (label[begin_index_of_first_smpl_in_c] != c + 1) begin_index_of_first_smpl_in_c++;
            int cnt = 0;
            while (cnt < num_hold_out_per_class[c] * k) {
                if (label[begin_index_of_first_smpl_in_c] == c + 1)
                    cnt++;
                begin_index_of_first_smpl_in_c++;
            }
            //begin from the begin_index_of_first_smpl_in_c, mark num_hold_out_per_class[c] samples to be testing samples.
            cnt = 0;
            while (cnt < num_hold_out_per_class[c]) {
                if (label[begin_index_of_first_smpl_in_c] == c + 1) {
                    F[k * N + begin_index_of_first_smpl_in_c] = 1;
                    cnt++;
                }
                begin_index_of_first_smpl_in_c++;
            }
        }
    }
    delete num_smpl_per_class;
    return F;
}

typedef struct K_FOLD_PARAM {
    DATASET* dataset;
    int* flag; //1xN
    int* order; //0,1,2...
} K_FOLD_PARAM;

void run_all_folds(void* p) {
    MODEL_PARAM* param = (MODEL_PARAM*) p;
    vector<GRAPH*> Gs = graph_construction(param); //construct graphs.
    if (!param->valid) //invalid graph.
        return;
    learn_coefficients(Gs); //learn the coefficients of the graphs.
    int* predicted = predict(Gs, param->feat2, param->N2); //predict labels of testing samples based on the graph.
    Gs.clear();
    //compute the confusion matrix.
    param->conmat = new int [param->num_class * param->num_class];
    memset(param->conmat, 0, sizeof (int)*param->num_class * param->num_class);
    for (int i = 0; i < param->N2; i++)
        param->conmat[(param->L2[i] - 1) * param->num_class + (predicted[i] - 1)]++;
    delete predicted;
    return;
}

void k_fold_feat_selection(void* param) {
    K_FOLD_PARAM* p = (K_FOLD_PARAM*) param;
    p->order = select_features_one_vs_rest_svm(*(p->dataset), p->flag); //the indices of all the attributes in decreasing order.
}

int main(int argc, char** argv) {
    
    tic

    DATASET* data = load_features("CU.txt");
    
    int K = 10; //k-fold cross validation.

    int num_class = data->num_class;

    Parallel PL;

    int* flag = generate_cross_validation_flags(K, data); //set num_class and return flags.
    const int MAX_RUN = 10000; //the maximum number of numbers for combination of [sel,eps].
    float* avg_ac = new float [MAX_RUN];
    memset(avg_ac, 0, sizeof (float)*MAX_RUN);

    K_FOLD_PARAM* feat_sel_space = new K_FOLD_PARAM [K];
    MODEL_PARAM* model_par_space = new MODEL_PARAM [MAX_RUN];
    vector<void*> par_k_fold;
    vector<void*> mdl_param;

    //feature selction. 
    for (int k = 0; k < K; k++) {
        feat_sel_space[k].dataset = data;
        feat_sel_space[k].flag = flag + k * data->N;
        par_k_fold.push_back(&feat_sel_space[k]);
    }
    printf("\nComputing SVM coefficients...");
    fflush(stdout);
    PL.Run(k_fold_feat_selection, par_k_fold); //this will generate the 'order' in the DATASET.
    puts("Done.");
    fflush(stdout);

    int min_sel = 50;
    int max_sel = 300;
    int step_sel = 25;

    float min_eps = 0.4;
    float max_eps = 0.7;
    float step_eps = 0.025;

    int min_deg = 3;
    int max_deg = 30;
    int step_deg = 3;

    int num_sel = (max_sel - min_sel) / step_sel + 1;
    int num_eps = (max_eps - min_eps) / step_eps + 1;
    int num_deg = (max_deg - min_deg) / step_deg + 1;


    int cnt = 0;
    for (int k = 0; k < K; k++) { //for each fold.
        //compute #training samples N1 and #testing samples N2.
        int N1 = 0, N2 = 0;
        for (int i = 0; i < data->N; i++)
            N2 += flag[k * data->N + i];
        N1 = data->N - N2;
        for (int sel = min_sel; sel <= max_sel; sel += step_sel) { //how many features you are selecting
            if (sel > ((K_FOLD_PARAM*) par_k_fold[0])->dataset->P) {
                printf("\n[Error] - You have only %d features, but want to select %d features.\n", ((K_FOLD_PARAM*) par_k_fold[0])->dataset->P, sel);
                exit(-1);
            }
            //split the data into training and testing according to flags, and select some attributes according to sel.
            int* order = ((K_FOLD_PARAM*) par_k_fold[k])->order; //0,1,2,...
            float* feat1 = new float[N1 * sel];
            float* feat2 = new float[N2 * sel];
            int* L1 = new int [N1]; //#label for training samples.
            int* L2 = new int [N2]; //#label for testing samples.
            int cnt1 = 0, cnt2 = 0;
            for (int n = 0; n < data->N; n++) {
                if (flag[k * data->N + n] == 0) { //training
                    for (int d = 0; d < sel; d++)
                        feat1[sel * cnt1 + d] = data->feat[data->P * n + order[d]];
                    L1[cnt1] = data->label[n];
                    cnt1++;
                } else {
                    for (int d = 0; d < sel; d++)
                        feat2[sel * cnt2 + d] = data->feat[data->P * n + order[d]];
                    L2[cnt2] = data->label[n];
                    cnt2++;
                }
            }
#ifdef Pearson
            for (float eps = min_eps; eps <= max_eps; eps += step_eps) { //correlation threshold.
#else
            for (int n = min_deg; n <= max_deg; n += step_deg) {
#endif
                if (cnt > MAX_RUN) {
                    puts("\n[Error] - MAX_RUN exceeded.");
                    return -1;
                }
                model_par_space[cnt].feat1 = feat1;
                model_par_space[cnt].feat2 = feat2;
                model_par_space[cnt].L1 = L1;
                model_par_space[cnt].L2 = L2;
#ifdef Pearson
                model_par_space[cnt].eps = eps;
#else
                model_par_space[cnt].degree = n;
#endif
                model_par_space[cnt].T = sel;
                model_par_space[cnt].num_class = num_class;
                model_par_space[cnt].N1 = N1;
                model_par_space[cnt].N2 = N2;

                mdl_param.push_back(&model_par_space[cnt]);
                cnt++;
            }
        }
    }

    //clean up.
    delete feat_sel_space, flag, data;
    par_k_fold.clear();
#ifdef Pearson  
    printf("\nRunning %d-folds cross-validation with %d parameter combinations [Pearson correlation]...", K, (int) mdl_param.size() / K);
    fflush(stdout);
#else
    printf("\nRunning %d-folds cross-validation with %d parameter combinations [Rank correlation]...", K, (int) mdl_param.size() / K);
    fflush(stdout);
#endif
    //run_all_folds(mdl_param[0]);
    PL.Run(run_all_folds, mdl_param);
    puts("Done.");


    //compute the accuracy of the first fold for different parameters.
    int NP = mdl_param.size() / K; //#parameter combinations.
    for (int k = 0; k < K; k++)
        for (int i = 0; i < NP; i++) {
            if ((*((MODEL_PARAM*) mdl_param[k * NP + i])).valid) {
                float ac = 0;
                int num_correct_predicted = 0;
                int NT = (*((MODEL_PARAM*) mdl_param[k * NP + i])).N2;
                for (int j = 0; j < num_class; j++)
                    num_correct_predicted += (*((MODEL_PARAM*) mdl_param[k * NP + i])).conmat[j * num_class + j]; //total correctly predicted number.
                ac = 100 * (float) num_correct_predicted / NT; //accuracy.
                avg_ac[i] += ac;
            }
        }
    puts("\n*The 10-fold averaged accuracy:*");
    for (int i = 0; i < NP; i++)
        avg_ac[i] /= K;
    int* index = new int [NP];
    quick_sort(avg_ac, NP, index);
    string ptag;
#ifdef Pearson
    for (int i = 0; i < NP; i++) {
        if (i == index[0])
            ptag = " - Highest";
        else if (i == index[NP - 1])
            ptag = " - Lowest";
        else
            ptag = "";
        if ((*((MODEL_PARAM*) mdl_param[i])).valid)
            printf("[Dim = %03d, Eps = %.3f]: %2.2f%% %s\n", (*((MODEL_PARAM*) mdl_param[i])).T, (*((MODEL_PARAM*) mdl_param[i])).eps, avg_ac[i], ptag.c_str());
        else
            printf("[Dim = %03d, Eps = %.3f]:  N/A\n", (*((MODEL_PARAM*) mdl_param[i])).T, (*((MODEL_PARAM*) mdl_param[i])).eps);
    }

    puts("--------------------");

    printf("\nHeat Map: x(eps)-[%.3f:%.3f:%.3f] y(sel)-[%d:%d:%d]:\n\n", min_eps, step_eps, max_eps, min_sel, step_sel, max_sel);
    for (int i = 0; i < num_sel; i++) {
        for (int j = 0; j < num_deg; j++) {
            printf("%.4f ", avg_ac[i * num_deg + j] / 100);
        }
        puts("");
    }
#else
    for (int i = 0; i < NP; i++) {
        if (i == index[0])
            ptag = "- [Highest]";
        else if (i == index[NP - 1])
            ptag = "- [Lowest]";
        else
            ptag = "";
        if ((*((MODEL_PARAM*) mdl_param[i])).valid) {
            printf("[Dim = %03d, degree = %02d]: %2.2f%% %s\n", (*((MODEL_PARAM*) mdl_param[i])).T, (*((MODEL_PARAM*) mdl_param[i])).degree, avg_ac[i], ptag.c_str());
        } else
            printf("[Dim = %03d, degree = %02d]:  N/A\n", (*((MODEL_PARAM*) mdl_param[i])).T, (*((MODEL_PARAM*) mdl_param[i])).degree);
    }

    puts("--------------------");

    printf("\nHeat Map: x(deg)-[%d:%d:%d] y(sel)-[%d:%d:%d]:\n\n", min_deg, step_deg, max_deg, min_sel, step_sel, max_sel);
    for (int i = 0; i < num_sel; i++) {
        for (int j = 0; j < num_deg; j++) {
            printf("%.4f ", avg_ac[i * num_deg + j] / 100);
        }
        puts("");
    }


#endif


    //clean up.
    delete avg_ac, model_par_space;
    mdl_param.clear();
    
    puts("\n");

    toc
    
    puts("\n");

    return 0;
}

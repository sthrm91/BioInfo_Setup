#include"common.h"

using namespace std;

int cholesky(double *A, int n) { //cholesky decomposition for symmetric PD matrix A.
    int i, j, k, in;
    for (i = 0; i < n; i++) {
        in = i*n;
        for (k = 0; k < i * n; k += n)
            A[in + i] -= A[k + i] * A[k + i];
        if (A[in + i] <= 0)
            return 0; //ERROR: non-positive definite matrix!
        A[in + i] = sqrt(A[in + i]);
        for (j = i + 1; j < n; j++) {
            A[in + j] = A[in + j];
            for (k = 0; k < i * n; k += n)
                A[in + j] -= A[k + i] * A[k + j];
            A[in + j] /= A[in + i];
        }
    }
    return 1;
}

void luEvaluate(double* U, float* b, float* x, int n) {
    // Ax = b -> LUx = b. Then y is defined to be Ux  [L = U']
    int i = 0;
    int j = 0;
    // Forward solve Lx = b
    for (i = 0; i < n; i++) {
        x[i] = b[i];
        for (j = 0; j < i; j++) {
            x[i] -= U[j * n + i] * x[j];
        }
        x[i] /= U[i * n + i];
    }
    // Backward solve Ux = x
    for (i = n - 1; i >= 0; i--) {
        for (j = i + 1; j < n; j++) {
            x[i] -= U[i * n + j] * x[j];
        }
        x[i] /= U[i * n + i];
    }
}

bool solve_least_squares(double*A, float*x, float*b, int n) {
    //solve Ax = b problem, return x. n is the size of b.
    if (!cholesky(A, n)) return false; //cholesky decomposition.
    luEvaluate(A, b, x, n); //forward-backward substitution.
}

void invert(double* data, int size) { //data is size x size matrix. in place inversion.
    if (size <= 0) return; // sanity check
    if (size == 1) return; // must be of dimension >= 2
    for (int i = 1; i < size; i++) data[i] /= data[0]; // normalize row 0
    for (int i = 1; i < size; i++) {
        for (int j = i; j < size; j++) { // do a column of L
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += data[j * size + k] * data[k * size + i];
            data[j * size + i] -= sum;
        }
        if (i == size - 1) continue;
        for (int j = i + 1; j < size; j++) { // do a row of U
            double sum = 0.0;
            for (int k = 0; k < i; k++)
                sum += data[i * size + k] * data[k * size + j];
            data[i * size + j] =
                    (data[i * size + j] - sum) / data[i * size + i];
        }
    }
    for (int i = 0; i < size; i++) // invert L
        for (int j = i; j < size; j++) {
            double x = 1.0;
            if (i != j) {
                x = 0.0;
                for (int k = i; k < j; k++)
                    x -= data[j * size + k] * data[k * size + i];
            }
            data[j * size + i] = x / data[j * size + j];
        }
    for (int i = 0; i < size; i++) // invert U
        for (int j = i; j < size; j++) {
            if (i == j) continue;
            double sum = 0.0;
            for (int k = i; k < j; k++)
                sum += data[k * size + j]*((i == k) ? 1.0 : data[i * size + k]);
            data[i * size + j] = -sum;
        }
    for (int i = 0; i < size; i++) // final inversion
        for (int j = 0; j < size; j++) {
            double sum = 0.0;
            for (int k = ((i > j) ? i : j); k < size; k++)
                sum += ((j == k) ? 1.0 : data[j * size + k]) * data[k * size + i];
            data[j * size + i] = sum;
        }
};

void Abs_Pearson_Coefficient(float*cc, float*A, int ns, int T) {
    //A is the nsxT matrix where ns is the #samples, and T is the #dimension. cc is TxT computed correlation matrix.
    //mean centering the data.
    float m; //mean 
    for (int d1 = 0; d1 < T; d1++) {
        m = 0;
        for (int n = 0; n < ns * T; n += T)
            m += A[n + d1];
        m /= ns; //mean of attribute d1.
        for (int n = 0; n < ns * T; n += T)
            A[n + d1] -= m;
    }
    //L2 normalization.
    float L2;
    for (int d1 = 0; d1 < T; d1++) {
        L2 = 0;
        for (int n = 0; n < ns * T; n += T)
            L2 += pow(A[n + d1], 2);
        L2 = sqrt(L2); //L2 of attribute d1.
        if (L2 == 0) {
            printf("[Error] - Bad data. L2 norm = 0 for attribute %d\n", d1);
            exit(-1);
        }
        for (int n = 0; n < ns * T; n += T)
            A[n + d1] /= L2;
    }
    //compute the correlation coefficients.
    memset(cc, 0, sizeof (float)*T * T);
    for (int d1 = 0; d1 < T; d1++) {
        for (int d2 = d1 + 1; d2 < T; d2++) {
            for (int n = 0; n < ns * T; n += T)
                cc[d1 * T + d2] += A[n + d1] * A[n + d2];
            cc[d1 * T + d2] = fabs(cc[d1 * T + d2]);
            cc[d2 * T + d1] = cc[d1 * T + d2]; //by symmetric: the diagonal elements are all zero as we don't need to select the self vertex.
        }
    }
}

bool Rank_Coefficient(float*cc, float*A, int ns, int T, int rank) {
    if (rank + 1 > T) {
        return false; //"[Error] - Degree can not exceed #selected attributes."
    }
    if (rank + 1 > ns) {
        return false; //"[Error] - Degree = %d, #training samples = %d.\n";
    }
    Abs_Pearson_Coefficient(cc, A, ns, T); //compute Pearson coefficients.
    int* index = new int [T];
    for (int t = 0; t < T; t++) {
        quick_sort(cc + t*T, T, index);
        memset(cc + t*T, 0, sizeof (float)*T);
        for (int i = 0; i < rank; i++)
            cc[t * T + index[i]] = 1;
    }
    delete index;
    return true;
}

vector<GRAPH*> graph_construction(MODEL_PARAM* param) {
    //This function construct one graph for each class.
    vector<GRAPH*> Gs; //graph for each class.
    int T = param->T; //number of selected attributes.
    float* train_feat = new float [param->N1 * T]; //selected features for training the graph
    float* cc = new float [T * T]; //correlation coefficient. 
    int num_class = param->num_class;
    int cnt_cur_smpl_id = 0;
    float* A = new float [param->N1 * T]; //temporary memory for holding data matrix.
    for (int c = 1; c <= num_class; c++) { //scan through all the classes, class label must be 1,2,...,C.
        GRAPH* G = new GRAPH; //the graph for the current class.
        //graph initialization.
        G->T = T; //# selected attributes.
        G->vertices = new VERTEX [T]; //allocate memory for the current graph.
        G->samples = train_feat + cnt_cur_smpl_id*T; //point to the beginning address of the training data for current graph.
        for (int i = 0; i < T; i++) {
            G->vertices[i].nn = 0;
            G->vertices[i].neighbors = new int [T - 1];
        }
        //group samples used for the training of the current graph.
        int ns = 0; //number of samples for the current class.
        for (int i = 0; i < param->N1; i++) { //scan the entire training dataset and select samples with label = c.
            if (param->L1[i] == c) {
                memcpy(train_feat + cnt_cur_smpl_id*T, param->feat1 + i*T, T * sizeof (float)); //copy the i-th sample of entire training set.
                ns++; //increase the number of training samples for the current graph.
                cnt_cur_smpl_id++; //we have processed one more samples, totally param->N1 samples.
            }
        }
        G->ns = ns; //set #training samples for the graph.
        memcpy(A, G->samples, ns * T * sizeof (float));
#ifdef Pearson
        Abs_Pearson_Coefficient(cc, A, ns, T);
#else
        if (!Rank_Coefficient(cc, A, ns, T, param->degree)) {
            param->valid = false;
            return Gs;
        }
#endif
        //show(cc,T,T); //[debug]
        //applying threshold to remove week edges.
        int ne = 0, mxd = 0, mnd = T; //#edges, minimum degree and maximum degree, respectively.
        G->num_isolated = 0;
        for (int d1 = 0; d1 < T; d1++) { //identify the neighbors of each vertex by comparing correlation coefficients.
            for (int d2 = 0; d2 < T; d2++) {
#ifdef Pearson
                if (cc[d1 * T + d2] >= param->eps) {
#else
                if (cc[d1 * T + d2] > 0.5) {
#endif
                    G->vertices[d1].neighbors[G->vertices[d1].nn] = d2; //add d2 as the neighbor of d1.
                    G->vertices[d1].nn++;
                    ne++;
                }
            }
            if (G->vertices[d1].nn > mxd) mxd = G->vertices[d1].nn; //max degree
            if (G->vertices[d1].nn < mnd) mnd = G->vertices[d1].nn; //min degree
            if (G->vertices[d1].nn == 0) G->num_isolated++;
        }
        G->max_deg = mxd;
        G->min_deg = mnd;
        G->num_edge = ne / 2;
        Gs.push_back(G);
    }
    delete cc, A;
    return Gs;
}

void learn_coefficients(vector<GRAPH*>& Gs) {
    //This function learn coefficients of each vertex for each graph.
    int T = Gs[0]->T; //#selected attributes.
    double* AA = new double [T * T]; //A'*A where A is the ns x T data matrix for training of a graph. ns may be different for different graphs.
    float* ptr_smpl = 0; //a free point for better memory access efficient.
    double* ptr_AA = 0;
    for (int i = 0; i < Gs.size(); i++) { //for each graph
        ptr_smpl = Gs[i]->samples;
        float*b = new float [Gs[i]->ns];
        float* A = new float [Gs[i]->ns * T]; //temporary memory for holding data matrix. //w = (A'A)^-1*A'*b

        for (int j = 0; j < Gs[i]->T; j++) { //for each vertex

            int nn = Gs[i]->vertices[j].nn; //#vertices connecting to it.
            int nc = nn + 1; //number of coefficients.

            if (nc > T) {
                puts("#Coefficients cannot exceed #Attributes.");
                exit(-1);
            }

            //compute A
            for (int p = 0; p < nc; p++) { //for each vertex connecting to it. [+1 because we need extra constant input 1]
                if (p == nn) { //the 1.0 pseudo input.
                    for (int n = 0; n < Gs[i]->ns * nc; n += nc)
                        A[n + p] = 1; //pseudo input.
                } else { //take samples from neighbors.
                    int id = Gs[i]->vertices[j].neighbors[p]; //the id of the neighbor node.
                    for (int n = 0; n < Gs[i]->ns; n++)
                        A[n * nc + p] = ptr_smpl[n * T + id];
                }
            }

            //compute AA = A'*A
            memset(AA, 0, sizeof (double)*nc * nc);
            for (int d1 = 0; d1 < nc; d1++) {
                for (int d2 = d1; d2 < nc; d2++) {
                    ptr_AA = AA + d1 * nc + d2;
                    for (int n = 0; n < Gs[i]->ns * nc; n += nc)
                        *ptr_AA += A[n + d1] * A[n + d2];
                    AA[d2 * nc + d1] = *ptr_AA; //by symmetric.
                }
            }

            for (int d = 0; d < nc; d++) AA[d * nc + d] += 0.1; //regularization.

            for (int n = 0; n < Gs[i]->ns; n++) b[n] = ptr_smpl[n * T + j];

            Gs[i]->vertices[j].coef = new float [nc];

            solve_least_squares(AA, Gs[i]->vertices[j].coef, b, nc); //A*x = b.
        }
        delete A, b;
    }
    delete AA;
}

int* predict(const vector<GRAPH*>& Gs, float* test, int nt) {
    //this function predict the class of testing samples based on graph.
    //test is NxT matrix and nt is the N = #testing samples.
    int T = Gs[0]->T; //dimension of features.
    float pav = 0; //predicted attribute value.
    float* ptr = 0;
    int* ret = new int [nt]; //predicted class.
    for (int n = 0; n < nt; n++) { //for each testing sample
        ptr = test + n*T;
        float mse = 0; //minimum squared error.
        int pc = 0; //predicted class.
        for (int c = 0; c < Gs.size(); c++) { //for each class.
            float se = 0; //squared reconstruction error.
            for (int d = 0; d < T; d++) { //for each attribute.
                pav = 0;
                for (int k = 0; k < Gs[c]->vertices[d].nn; k++) //for each neighbor.
                    pav += ptr[Gs[c]->vertices[d].neighbors[k]] * Gs[c]->vertices[d].coef[k]; //linear combination of the connecting components to reconstruct the attribute.
                pav += Gs[c]->vertices[d].coef[Gs[c]->vertices[d].nn]; //constant factor.
                se += (ptr[d] - pav)*(ptr[d] - pav);
            }
            se = sqrt(se);
            if (se < mse || c == 0) {
                mse = se;
                pc = c;
            }
        }
        ret[n] = pc + 1; //class: 1,2,3,...C.
    }
    return ret;
}




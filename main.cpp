
#include <iostream>
#include <vector>
#include <complex>
#include "cblas.h"

using namespace std;

typedef complex<double> cdouble;
const double pi = 3.1415926;

void write_points(string filename, vector<double> X, vector<double> psi, string stats, int N)
{
    FILE * file;
    file = fopen(filename.c_str(), "w");
    int i;
    fprintf(file, "%s", stats.c_str());
    for (i = 0; i < N ; i++)
        fprintf(file, "\n%lf,%.10lf", X[i], psi[i]);
    fclose(file);
}

vector<double> grid(double xmin, double dx, int N) {
    int i;
    vector<double> X(N);
    for (i = 0 ; i < N ; i++)
        X[i] = xmin + i*dx;
    return X;
}

vector<cdouble> initial( vector<double> X, int N, double x0, double alpha, double dx) {
    int i;
    double C = sqrt( dx*sqrt((alpha/pi)) );
    vector<cdouble> psi( N );
    for (i = 0 ; i < N ; i++)
        psi[i] = C*exp(-0.5*alpha* (X[i]-x0)*(X[i]-x0) );
    return psi;
}

double SHO(double x) {
    return x*x/2;
}

vector<double> probability(vector<cdouble> &psi, int N ) {
    int i;
    double total = 0.0;
    vector<double> P(N);
    for (i = 0 ; i < N ; i++) {
        P[i] = ( psi[i]*conj(psi[i]) ).real();
        total += P[i];
    }
    total = 1.0/total;
    double total2 = sqrt(total);
    for (i = 0 ; i < N ; i++) {
        P[i] *= total;
        psi[i] *= total2;
    }
    return P;
}

vector<cdouble> kernel(vector<double> X, int N, cdouble A_inv, function<double(double)> V, double eps) {
    int i, j;
    double action, v;
    vector<cdouble> K( N*N );
    for (i = 0 ; i < N ; i++)
        for (j = 0 ; j < N ; j++) {
            v = ( X[j] - X[i] )/eps;
            action = v*v/2 - V( (X[j]+X[i])/2 );
            K[ i*N + j ] = A_inv*exp( cdouble(0, eps*action) );
        }
    return K;
}

string calculate_averages(vector<double> X, vector<double> P, vector<cdouble> psi, double t, int N,  function<double(double)> V, double dx2) {
    int i, j;
    cdouble k(0.0, 0.0);
    //cdouble p(0.0, 0.0);
    double p, x;
    p = 0; x = 0;
    for (i = 0 ; i < N ; i++) {
        x += P[i]*X[i];
        p += P[i]*V(X[i]);
        if ( (i != 0) && (i != N-1) )
            k += -0.5*conj(psi[i]) * ( psi[i+1] + psi[i-1] - 2.0*psi[i])/dx2;
    }
    string result = "t=" + to_string(t) + ",<X>=" + to_string(x) + ",<K>=" + to_string( k.real() ) + ",<P>=" + to_string( p ) + ",<E>=" + to_string( p + k.real() );
    return result;
}

int main(int argc, const char * argv[]) {
    
    string directory = argv[1];
    string output = argv[2];
    int Nd = atoi(argv[3]);
    double xmin = atof(argv[4]);
    double xmax = atof(argv[5]);
    double xstart = atof(argv[6]);
    double alpha = atof(argv[7]);
    int N1 = atoi(argv[8]);
    int N2 = atoi(argv[9]);
    double eps = atof(argv[10]);
    
    
    int N = Nd + 1;
    double dx = (xmax-xmin)/Nd;
    double dt = eps*pow(2,N1);
    
    cdouble A = sqrt( eps * pi * cdouble(0,2) ) ;
    cdouble scaling(dx, 0.0);
    cdouble C0(0.0, 0.0);
    
    vector<double> X = grid(xmin, dx, N);
    vector<cdouble> psi = initial(X, N, xstart, alpha, dx);
    vector<double> P = probability(psi, N);;
    
    string stats = calculate_averages(X, P, psi, 0.0, N, SHO, dx*dx);
    write_points(directory + output + "0.txt", X, P, stats, N);
    
    vector<cdouble> Ke = kernel(X, N, 1./A, SHO, eps);
    vector<cdouble> K( N*N );
    vector<cdouble> psi2(N);
    
    K = Ke;
    for (int i=0; i<N1; i++) {
        cblas_zgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, N, N, N, &scaling, &Ke[0], N, &Ke[0], N, &C0, &K[0], N);
        Ke = K;
    }
    
    for (int i=0; i<N2; i++) {
        cblas_zgemv(CblasRowMajor, CblasNoTrans, N, N, &scaling, &Ke[0], N, &psi[0], 1, &C0, &psi2[0], 1);
        P = probability(psi2, N);
        psi = psi2;
        stats = calculate_averages(X, P, psi, (i+1)*dt, N, SHO, dx*dx);
        write_points(directory + output + to_string(i+1) + ".txt", X, P, stats, N);
    }
    
    return 0;
}

# include <iostream>
using namespace std;
# include <complex>


// function to do general tracing
void transpose_2d(std::complex<double>  *array, int dim, std::complex<double> *out){

    for(int i=0;i<dim;i++){
        for(int j=0;j<dim;j++){
            // i+j*dim traverses 2d array rows-first
            // j+i*dim traverses 2d array cols-first
            out[j+i*dim] = array[i+j*dim]; // transposes the 2d array
            // std::cout << out[j+i*dim] << " , " << array[i+j*dim] << endl;
        }
    }
}
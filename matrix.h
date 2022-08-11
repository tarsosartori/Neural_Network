#include<iostream>
#include<math.h>
#include<vector>
#include<time.h>
#include<cstdlib>
#include <bits/stdc++.h>


template<typename T>
class Matrix{
	private:
		std::vector<std::vector<T>> matrix;
		unsigned n_rows, n_cols;
	public:
		// constructors
		Matrix(unsigned n_rows, unsigned n_cols); // Makes a matrix of zeros shape (n_rows, n_cols)
		Matrix(T *M, unsigned n_rows, unsigned n_cols); // creates a matrix from a vector
		Matrix();
		~Matrix(); // destructor
		
		// useful functions
		void printM(); // print matrix
		unsigned getRows() const;
		unsigned getCols() const;
		void shape(); // print shape
		Matrix transpose();
		void randn(); // creates a matrix with random elements following the normal distribution
		Matrix mult_ewise(Matrix B); // elementwise multiplication
		Matrix div_ewise(Matrix B); // elementwise division
		Matrix logM(); // return a matrix with the logs of the components
		T norm(); // returns the norm of a matrix of shape n x 1
		void reshape(unsigned n_rows, unsigned n_cols); // reshapes the matrix and set values to 0
		Matrix slice_cols(int col_a = -1, int col_b = -1); // slice from column col_a until col_b - 1
		Matrix filter_category();
		
		// matrix operators
		T& operator()(const unsigned int row, const unsigned int col);
		Matrix operator+(Matrix B);
		Matrix operator-(Matrix B);
		Matrix operator*(Matrix B);
		void operator=(Matrix M);
		Matrix operator*(const T c);
		Matrix operator+(const T c);
		Matrix operator-(const T c);
		
};	

template<class T>
inline Matrix<T> Matrix<T>::filter_category(){ // get the columns from [col_a, col_b - 1]
	Matrix<T> bin(n_rows,n_cols);
	T max_val;
	
	for(int j=0;j<n_cols;j++){
		max_val = matrix[0][j];
		for(int i=1;i<n_rows;i++){
			if(matrix[i][j] > max_val){
				max_val = matrix[i][j];
			}
		}
		for(int i=0;i<n_rows;i++){
			if(matrix[i][j] == max_val){
				bin(i,j) = 1;
			}else{
				bin(i,j) = 0;
			}
		}
	}
	return bin;
}

template<class T>
inline Matrix<T> Matrix<T>::slice_cols(int col_a, int col_b){ // get the columns from [col_a, col_b - 1]
	if(col_b < col_a && col_b != -1){
		std::cout<<"col_a needs to be < col_b"<<std::endl;
		exit(0);
	}
	
	Matrix<T> result;
	
	if(col_a >= 0 && col_b == -1 && col_a < n_cols){
		result.reshape(n_rows,1);
		for(int i = 0; i < n_rows;i++){
			result(i,0) = this->matrix[i][col_a];
		}
		
	}else if(col_a >=0 && col_b >=0 && col_b <= n_cols && col_a < n_cols){
		result.reshape(n_rows, col_b - col_a);
		for(int i = 0; i < n_rows;i++){
			for(int j = col_a; j < col_b; j++){
				result(i,j-col_a) = this->matrix[i][j];
			}
		}
	}else if(col_a < 0 && col_b == -1 && col_a*-1 <= n_cols){
		result.reshape(n_rows,1);
		for(int i = 0; i < n_rows;i++){
			result(i,0) = this->matrix[i][n_cols+col_a];
		}
	}else{
		std::cout<<"Invalid slice"<<std::endl;
		exit(0);
	}
	
	return result;
}

template<class T>
inline void Matrix<T>::reshape(unsigned n_rows, unsigned n_cols){
	this->n_rows = n_rows;
	this->n_cols = n_cols;
	this->matrix.clear();
	this->matrix.resize(n_rows);
	for(int i = 0; i < n_rows; i++){
		this->matrix[i].resize(n_cols, 0);
	}
}


template<class T>
inline T Matrix<T>::norm(){
	if(this->n_cols != 1){
		std::cout<<"Shape must be n x 1 for using norm"<<std::endl;
		exit(0);
	}
	
	T n = 0;
	for(int i = 0;i<this->n_rows;i++){
		n += matrix[i][0]*matrix[i][0];	
	}
	
	return sqrt(n);
	
}

template<class T>
inline Matrix<T> Matrix<T>::mult_ewise(Matrix<T> B){
	Matrix<T> result(this->n_rows,this->n_cols);
		
	if(B.getRows()==this->n_rows && B.getCols() ==  this->n_cols){
		for(int i=0;i<this->n_rows;i++){
			for(int j=0;j<this->n_cols;j++){
				result(i,j) = B(i,j) * this->matrix[i][j];
			}
		}
	}else if(B.getRows() == this->n_rows && B.getCols() == 1){
		for(int i=0;i<this->n_rows;i++){
			for(int j=0;j<this->n_cols;j++){
				result(i,j) = B(i,0) * this->matrix[i][j];
			}
		}
	}else{
		std::cout<<"Dimensions mismatch multiplication elementwise"<<std::endl;
		exit(0);
	}
	
	return result;
}

template<class T>
inline Matrix<T> Matrix<T>::div_ewise(Matrix<T> B){
	Matrix<T> result(this->n_rows,this->n_cols);
	if(B.getRows()==this->n_rows && B.getCols() ==  this->n_cols){
		for(int i=0;i<this->n_rows;i++){
			for(int j=0;j<this->n_cols;j++){
				if(B(i,j) == 0){
					B(i,j) = 0.000000001;
					std::cout<<"singularity"<<std::endl;
				}
				result(i,j) = this->matrix[i][j]/B(i,j);
			}
		}
	}else if(B.getRows() == this->n_rows && B.getCols() == 1){
		for(int i=0;i<this->n_rows;i++){
			for(int j=0;j<this->n_cols;j++){
				if(B(i,j) == 0){
					B(i,0) = 0.000000001;
					std::cout<<"singularity"<<std::endl;
				}
				result(i,j) = this->matrix[i][j]/B(i,0);
			}
		}
	}else{
		std::cout<<"Dimensions mismatch division elementwise"<<std::endl;
		exit(0);
	}
	
	return result;
}


template<class T>
inline Matrix<T>::Matrix(){
this->n_rows = 0;
this->n_cols = 0;
this->matrix.resize(0);
}

template<class T>
inline Matrix<T>::Matrix(unsigned n_rows, unsigned n_cols){
	
	this -> n_rows = n_rows;
	this -> n_cols = n_cols;
	
	this->matrix.resize(n_rows);
	for(unsigned i=0; i < n_rows; i++){
		this->matrix[i].resize(n_cols, 0.0);
	}  
}

template<class T>
inline Matrix<T>::Matrix(T *M, unsigned n_rows, unsigned n_cols){
	this->n_rows = n_rows;
	this->n_cols = n_cols;
	this->matrix.resize(n_rows);
	for(int i=0; i < n_rows; i++){
		this->matrix[i].resize(n_cols);
		for(int j=0;j<n_cols;j++){
			this -> matrix[i][j] = M[i*n_cols+j];
		}
	}	
}

template<class T>
inline Matrix<T>::~Matrix(){}

template<class T>
inline void Matrix<T>::printM(){
	for(int i = 0; i<n_rows; i++){
		for (int j = 0; j< n_cols; j++){
			std::cout<<this->matrix[i][j]<<" ";
		}
		std::cout<<std::endl;
	}
		std::cout<<"---------------------"<<std::endl;
}

template<class T>
inline unsigned Matrix<T>::getRows() const{
	return this->n_rows;
}

template<class T>
inline unsigned Matrix<T>::getCols() const{
	return this->n_cols;
}

template<class T>
inline void Matrix<T>::shape(){
	std::cout<<"("<<getRows()<<", "<<getCols()<<")"<<std::endl;
}

template<class T>
inline T& Matrix<T>::operator()(const unsigned int row, const unsigned int col){
	return this->matrix[row][col];
}

template<class T>
inline Matrix<T> Matrix<T>::operator+(Matrix B){
	if (B.getRows() != this->n_rows){
		exit(0);
	}
	if (B.getCols() != this->n_cols){
		if(B.getCols() == 1){			// sum matrix vector
			Matrix<T> result(getRows(),getCols());
			for(unsigned int i=0; i<getRows();i++){
				for(unsigned int j=0; j<getCols();j++){
					result(i,j) = this->matrix[i][j] + B(i,0);
				}
			}
			return result;
		}
		else{
			exit(0);
		}
	}
	
	Matrix<T> result(getRows(),getCols());
	for(unsigned int i=0; i<getRows();i++){
		for(unsigned int j=0; j<getCols();j++){
			result(i,j) = this->matrix[i][j] + B(i,j);
		}
	}
	return result;
}

template<class T>
inline Matrix<T> Matrix<T>::operator-(Matrix B){
	if (B.getRows() != this->n_rows){
		exit(0);
	}
	if (B.getCols() != this->n_cols){
		if(B.getCols() == 1){			// sum matrix vector
			Matrix<T> result(getRows(),getCols());
			for(unsigned int i=0; i<getRows();i++){
				for(unsigned int j=0; j<getCols();j++){
					result(i,j) = this->matrix[i][j] - B(i,0);
				}
			}
			return result;
		}
		else{
			exit(0);
		}
	}
	
	Matrix<T> result(getRows(),getCols());
	for(unsigned int i=0; i<getRows();i++){
		for(unsigned int j=0; j<getCols();j++){
			result(i,j) = this->matrix[i][j] - B(i,j);
		}
	}
	return result;
}


template<class T>
inline Matrix<T> Matrix<T>::operator*(Matrix B){
	if (getCols() != B.getRows()){
		std::cout<<"Matrix multiplitcation mismatch"<<std::endl;
		exit(0);
	}
	
	T sum = 0;
	
	Matrix<T> result(getRows(),B.getCols());
	
	for(int i=0;i<getRows();i++){
		for(int j=0;j<B.getCols();j++){
			for(int k=0; k<getCols(); k++){
				sum += this->matrix[i][k] * B(k,j);
			}
			result(i,j) = sum;
			sum = 0;
		}
	}
	return result;
}

template<class T>
inline void Matrix<T>::operator=(Matrix<T> M){
	this->n_cols = M.getCols();
	this->n_rows = M.getRows();
	this->matrix.resize(this->n_rows);
	for(int i=0; i < this->n_rows; i++){
		this->matrix[i].resize(this->n_cols);
		for(int j=0;j<this->n_cols;j++){
			this -> matrix[i][j] = M(i,j);
		}
	}
	//return *this;
	
}

template<class T>
inline Matrix<T> Matrix<T>::operator*(const T c){
	Matrix result(getRows(),getCols());
	for(unsigned int i=0; i<getRows();i++){
		for(unsigned int j=0; j<getCols();j++){
			result(i,j) = this->matrix[i][j] * c;
		}
	}
	return result;
}


template<class T>
inline Matrix<T> Matrix<T>::operator+(const T c){
	Matrix result(getRows(),getCols());
	for(unsigned int i=0; i<getRows();i++){
		for(unsigned int j=0; j<getCols();j++){
			result(i,j) = this->matrix[i][j] + c;
		}
	}
	return result;
}


template<class T>
inline Matrix<T> Matrix<T>::operator-(const T c){
	Matrix result(getRows(),getCols());
	for(unsigned int i=0; i<getRows();i++){
		for(unsigned int j=0; j<getCols();j++){
			result(i,j) = this->matrix[i][j] - c;
		}
	}
	return result;
}


template<class T>
inline Matrix<T> Matrix<T>::transpose(){
	Matrix result(getCols(),getRows());
	for(int i=0;i<getRows();i++){
		for(int j=0;j<getCols();j++){
			result(j,i) = this->matrix[i][j];
		}
	}
	return result;
}


template<typename T>
inline void Matrix<T>::randn(){
	unsigned i, j, k = 0;
	srand(time(0));
	
	unsigned n = this->n_rows * this->n_cols;
	std::vector<T> v(n);
	for ( i = 0; i < n; i ++ )
        {
            T x,y,rsq,f;
            do {
                x = 2.0 * rand() / (T)RAND_MAX - 1.0;  
				y = 2.0 * rand() / (T)RAND_MAX - 1.0;
                rsq = x * x + y * y;       
            }while( rsq >= 1. || rsq == 0. );
            
			f = sqrt(-2.0 * log(rsq) / rsq);
		
            v[i]= x*f;
            //v[i+1]= y*f;
	    }
	
	for ( i=0; i<this->n_rows; i++){
		for( j=0; j<this->n_cols;j++){
			this->matrix[i][j] = v[i*this->n_cols+j];
		}
	}

}

template<typename T>
inline Matrix<T> Matrix<T>::logM(){
	
	Matrix<T> result(n_rows,n_cols);
	for ( int i=0; i<this->n_rows; i++){
		for(int  j=0; j<this->n_cols;j++){
			result(i,j) = log(this->matrix[i][j]);
		}
	}
	return result;
	
}

#include "Matrix.h"
#include<string>

template<class T>
Matrix<T> sigmoid(Matrix<T> vet){
		Matrix<T> result(vet.getRows(),vet.getCols());
		for(int i=0;i<vet.getRows();i++){
			for(int j=0;j<vet.getCols();j++){
					result(i,j) = 1/(1+exp(-vet(i,j)));
			}
		}
		return result;
	}
	
template<class T>
Matrix<T> d_sigmoid(Matrix<T> vet){
	Matrix<T> s = sigmoid(vet);
	Matrix<T> result = s.mult_ewise(s*-1 + 1);

	return result;
}	
	

template<class T>
Matrix<T> relu(Matrix<T> vet){
		Matrix<T> result(vet.getRows(),vet.getCols());
		for(int i=0;i<vet.getRows();i++){
			for(int j=0;j<vet.getCols();j++){
					result(i,j) = vet(i,j)*(vet(i,j)>0);
			}
		}
		return result;
	}

template<class T>
Matrix<T> d_relu(Matrix<T> vet){
	Matrix<T> result(vet.getRows(),vet.getCols());
	for(int i = 0; i<vet.getRows(); i++){
		for(int j = 0; j< vet.getCols(); j++){
			if(vet(i,j)<=0){
				result(i,j) = 0;
			}else{
				result(i,j) = 1;
			}
		}
	}
	return result;
}

template<class T>
Matrix<T> softmax(Matrix<T> vet){
	
		Matrix<T> sum_exp_vet(vet.getCols(),1);
		for(int j = 0; j<vet.getCols();j++){
			for(int i = 0; i<vet.getRows();i++){
				sum_exp_vet(j,0) += exp(vet(i,j));
			}
		}
		Matrix<T> result(vet.getRows(),vet.getCols());
		for(int i=0;i<vet.getRows();i++){
			for(int j=0;j<vet.getCols();j++){
				result(i,j) = exp(vet(i,j))/sum_exp_vet(j,0);
			}	
		}
		return result;
	}


template<class T>
Matrix<T> d_softmax(Matrix<T> dA, Matrix<T> dZ){
	
		Matrix<T> result(dZ.getRows(),dZ.getCols());
		
		Matrix<T> S = softmax(dZ);
		Matrix<T> s(dZ.getRows(),1), vet(dZ.getRows(),1), da(dA.getRows(),1);
		
		Matrix<T> J(s.getRows(),s.getRows());
		
		
		for(int k = 0; k<dZ.getCols(); k++){
			for(int n = 0; n< s.getRows(); n++){
				s(n,0) = S(n,k);
				da(n,0) = dA(n,k);
			}
			
			
			for(int i=0; i<J.getRows();i++){
				for(int j=0;j<J.getCols();j++){	
					if (i == j){
						J(i,j)	= s(i,0)*(1-s(i,0));
					}else{
						J(i,j) = -s(i,0)*s(j,0);
					}	 
				}	
			}
			
			vet = J * da;
			
			for(int n = 0; n< s.getRows(); n++){
				result(n,k) = vet(n,0);
			}
		}
		return result;
	}

template<class T>
Matrix<T> tanh(Matrix<T> vet){
		
		Matrix<T> result(vet.getRows(),vet.getCols());
		for(int i=0;i<vet.getRows();i++){
			for(int j=0;j<vet.getCols();j++){
				result(i,j) = (exp(vet(i,j)) - exp(-vet(i,j))) / (exp(vet(i,j)) + exp(-vet(i,j)));	
			}
		}
		return result;
	}


template<class T>
Matrix<T> d_tanh(Matrix<T> vet){
		
		Matrix<T> result = tanh(vet).mult_ewise(tanh(vet))*-1 + 1;
		
		return result;
	}


template<class T>
class Dense_Layer{
	
	private:
		std::string activation;
		Matrix<T> W, b;
	
	public:
		Dense_Layer(Matrix<T>& W1, Matrix<T>& b1, std::string activation);
		Matrix<T>& get_W();
		Matrix<T>& get_b();
		std::string get_activation();
		Matrix<T> output(Matrix<T> input);	
};


template<class T>
inline Dense_Layer<T>::Dense_Layer(Matrix<T>& W1, Matrix<T>& b1, std::string activation){
			if(W1.getRows() != b1.getRows()){
				std::cout<<"W rows and b rows mismatch in dense layer creation"<<std::endl;
				exit(0);
			}
			this->activation = activation;
			this->W = W1;
			this->b = b1;
			
		}
		
template<class T>		
inline Matrix<T>& Dense_Layer<T>::get_W(){
	return W;
}

template<class T>	
inline Matrix<T>& Dense_Layer<T>::get_b(){
	return b;
}

template<class T>
std::string Dense_Layer<T>::get_activation(){
	return this -> activation;
}

template<class T>	
inline Matrix<T> Dense_Layer<T>::output(Matrix<T> input){
		
		Matrix<T> result = this->W*input + this->b;
		
		if (this->activation == "None"){
			return result;
		}else if (this->activation == "sigmoid"){
			result = sigmoid(result);
		}else if (this->activation == "relu"){
			result = relu(result);
		}else if (this->activation == "softmax"){
			result = softmax(result);
		}else if (this->activation == "tanh"){
			result = tanh(result);
		}else{
			std::cout<<"Invalid activation"<<std::endl;
			exit(0);
		}
		
		return result;
}



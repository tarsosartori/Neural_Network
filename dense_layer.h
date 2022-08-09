#include "matrix.h"
#include<string>
#include "activation_functions.h"

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



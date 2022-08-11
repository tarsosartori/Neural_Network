#include "dense_layer.h"
#include<tuple>
#include<stdio.h>
#include <bits/stdc++.h>
#include<algorithm>


template<class T>
class network{
	
	// W (neurons [l], neurons[l-1])
	// b (neurons [l], 1)
	// X (features, m_samples)
	// y (n_outputs, m_samples)
	
	private:
		std::vector<Dense_Layer<T>*> layers; // stores the NN layers	
		std::vector<Matrix<T>> caches; // stores the gradients for each layer, linear and non-linear activations. dW, db, dA, A, Z
		unsigned input_size;
		
	public:
		network(unsigned input_size);
		void add_Dense(unsigned neurons, std::string activation = "None", T initialize_weights = 0.01);
		void set_Dense(T *W_, T *b_, unsigned n_rows, unsigned n_cols, std::string activation = "None");
		void summary();
		Matrix<T> predict(Matrix<T>& input);
		Matrix<T>& get_W(unsigned layer_number);
		Matrix<T>& get_b(unsigned layer_number);
		void update_weights(T learning_rate);
		void train(Matrix<T>& X, Matrix<T>& y, T learning_rate = 0.001, unsigned epochs = 1000, std::string cost_function = "cross_entropy", int batch = 0, bool shuffle = true, bool print_cost = true);
		void backward_prop(Matrix<T>& X, Matrix<T>& y, std::string cost_function, int col_a = -1, int col_b = -1);
		void forward_prop(Matrix<T>& X, int col_a = -1, int col_b = -1);
		Matrix<T> linear_activation_backward(Matrix<T>& dAl, int layer, Matrix<T>& X, int col_a= -1, int col_b = -1);
		T compute_cost(Matrix<T>& AL, Matrix<T>& y, std::string cost_function);
		void gradient_clipping(Matrix<T>& dA);
		void accuracy(Matrix<T>& y_real, Matrix<T>& y_pred);
};

template<class T>
inline void network<T>::accuracy(Matrix<T>& y_real, Matrix<T>& y_pred){
	
	T acc = 0;
	T max_val;
	Matrix<T> bin = y_pred.filter_category();

	bool cond;
	for(int j=0;j<y_real.getCols();j++){
		cond = true;
		for(int i=0;i<y_real.getRows();i++){
			if(y_real(i,j) != bin(i,j)){
				cond = false;
			}
		}
		if(cond){
			acc += 1;
		}
	}
	
	std::cout<<"Accuracy = "<<100*acc/y_real.getCols()<<"%"<<std::endl;

}

template<class T>
inline void network<T>::train(Matrix<T>& X_, Matrix<T>& y_, T learning_rate, unsigned epochs, std::string cost_function, int batch, bool shuffle, bool print_cost){
	
	// X (n_features x m_samples), y (n_outputs x m_samples) 
	
	if(X_.getRows() != this->input_size){
		std::cout<<"X dimension are incorrect. It should be: (n_features x m_samples)-> n_features = "<<this->input_size <<std::endl;
		exit(0);
	}
	if(y_.getRows() != this->layers[this->layers.size()-1]->get_b().getRows()){
		std::cout<<"y dimensions are incorrect. It should be: (n_outputs x m_samples)-> n_ouputs = "<<this->layers[this->layers.size()-1]->get_b().getRows()<<std::endl;
		exit(0);
	}
	if(X_.getCols() != y_.getCols()){
		std::cout<<"number of samples in X and y mismatch. They should be X (n_features x m_samples), y (n_outputs x m_samples) "<<std::endl;
		exit(0);
	}
	if(batch > X_.getCols()){
		std::cout<<"Batch size must be <= number of samples m"<<std::endl;
		exit(0);
	}
	
	Matrix<T> X, y, AL, Xb, yb ;
	X = X_;
	y = y_;
	T cost;
	int n_batches, start, end;

	std::vector<int> v;
	if(shuffle){
		for(int i = 0; i < X.getCols(); i++){
			v.push_back(i);
		}
	}
		
	for(int i = 0; i < this->layers.size(); i++){
		
		caches.push_back(Matrix<T>(layers[i]->get_W().getRows(), layers[i]->get_W().getCols())); // dW = dJ/dW
		caches.push_back(Matrix<T>(layers[i]->get_b().getRows(), layers[i]->get_b().getCols()));// db = dJ/db
		caches.push_back(Matrix<T>(layers[i]->get_b().getRows(), X.getCols())); // dA = dJ/dA
		caches.push_back(Matrix<T>(layers[i]->get_b().getRows(), X.getCols())); // Z
		caches.push_back(Matrix<T>(layers[i]->get_b().getRows(), X.getCols())); // A	
		
	}
	
	if(batch == 0 || batch == X.getCols()){
		n_batches = 1;
	}else{
		n_batches = (int) X.getCols()/batch;
		Xb.reshape(X.getRows(), batch);
		yb.reshape(y.getRows(), batch);
	}
	
	for(int i = 0; i < epochs; i++){
		
		if(shuffle){
			std::random_shuffle(v.begin(),v.end());
			for(int j = 0; j < X.getCols(); j++){
				for(int i = 0; i < X.getRows(); i++){
					X(i,j) = X_(i,v[j]);
				}
				for(int i = 0; i < y.getRows(); i++){
					y(i,j) = y_(i,v[j]);
				}
			}
		}
		
		if(n_batches == 1){
		
			this->forward_prop(X);
				
			this->backward_prop(X, y, cost_function);
		
			this->update_weights(learning_rate);
		
		}else{
			
			for(int k = 0; k < n_batches; k++){
				start = k * batch;
				end = start + batch;
				Xb = X.slice_cols(start,end);
				yb = y.slice_cols(start,end);
				
				this->forward_prop(Xb, start, end);
			
				this->backward_prop(Xb, yb, cost_function, start, end);
		
				this->update_weights(learning_rate);		
			}
			
			if(end < X.getCols()){
				Xb = X.slice_cols(end, X.getCols()); 
				yb = y.slice_cols(end, y.getCols());
				
				this->forward_prop(Xb, end, X.getCols());
			
				this->backward_prop(Xb, yb, cost_function, end, y.getCols());
		
				this->update_weights(learning_rate);
			}		
			
		}
		
		if(print_cost){
			AL = this->caches[5*(layers.size()-1)+4];
			cost = this->compute_cost(AL, y, cost_function);
			std::cout<<"Iteration ["<<i+1<<"/"<<epochs<<"] Cost "<<cost_function<< " = "<<cost<<std::endl;
		} 		
		
	}

}

template<class T>
inline void network<T>::forward_prop(Matrix<T>& input, int col_a, int col_b){
			
	if(input.getRows() != this->input_size){
		std::cout<<"Input rows need to be = "<< this->input_size << " (x(features,m))"<<std::endl;
		exit(0);
	}
	
	Matrix<T> activation_out, linear_out;
	activation_out = input;
	linear_out = input;
	
	if(col_a == -1 && col_b == -1){
		for(int i = 0; i < this->layers.size(); i++){
			linear_out = this->layers[i]->get_W()*linear_out + layers[i]->get_b(); 
			this->caches[5*i+3] = linear_out; // Z
			activation_out = this->layers[i]->output(activation_out);
			this->caches[5*i+4] = activation_out; //A
		}	
	}else{
		for(int i = 0; i < this->layers.size(); i++){
			linear_out = this->layers[i]->get_W()*linear_out + layers[i]->get_b(); 
			activation_out = this->layers[i]->output(activation_out);
			
			for(int line = 0; line <linear_out.getRows(); line++){
				for(int col = col_a; col < col_b; col++){
					this->caches[5*i+3](line,col) = linear_out(line,col-col_a); // Z
					this->caches[5*i+4](line,col) = activation_out(line,col-col_a); //A
				}
			}
			//std::cout<<"tchau"<<std::endl;
			
		}
	}
	
}

template<class T>
inline Matrix<T> network<T>::linear_activation_backward(Matrix<T>& dAl, int layer, Matrix<T>& X, int col_a, int col_b){
	
	//backpropagation algorithm:
	//A_prev = A_(l-1)
	//dW_l = dJ/dW = 1/m * dZ_l * A_prev^T 
	//db_l = dJ/db = 1/m * sum_i=1_m(dZ_l_i)
	//dA_prev = dJ/dA_prev = W_l^T * dZ_l 
	//dZ_l = dA_l .* g'(Z_l)
	
	Matrix<T> dZl, dWl, dA_prev, Al_prev, Wl, Zl;
	Matrix<T> dbl(dAl.getRows(),1);
	std::string activation = this->layers[layer]->get_activation();
	T m;
	
	if(col_a == -1 && col_b == -1){ // No batch
		Zl = this->caches[5*layer+3];
		Wl = this->layers[layer]->get_W();
		m = (T) Zl.getCols();
		
		if(layer > 0){
			Al_prev = this->caches[5*(layer-1) + 4];
		}else{
			Al_prev = X;
		}
		
	}else{ // With batch
		Wl = this->layers[layer]->get_W();
		Zl = this->caches[5*layer+3].slice_cols(col_a, col_b);
		
		m = (T) Zl.getCols();
		
		if(layer > 0){
			Al_prev = this->caches[5*(layer-1) + 4].slice_cols(col_a, col_b);
		}else{
			Al_prev = X;
		}
	
	}
	
	if(activation == "relu"){
		dZl = dAl.mult_ewise(d_relu(Zl));
	}else if(activation == "tanh"){
		dZl = dAl.mult_ewise(d_tanh(Zl));
	}else if(activation == "sigmoid"){
		dZl = dAl.mult_ewise(d_sigmoid(Zl));
	}else if(activation == "None"){
		dZl = dAl;
	}else if(activation == "softmax"){
		dZl = d_softmax(dAl, Zl);
	}
	
	dWl = dZl * Al_prev.transpose() * (1/m);
	
	for(int i=0;i<Zl.getRows();i++){
		for(int j=0;j<Zl.getCols();j++){
			dbl(i,0) = dbl(i,0) + dZl(i,j);
		}
	}
	dbl = dbl * (1/m);
	
	//gradient_clipping(dWl);
	//gradient_clipping(dbl);
	
	this->caches[5*layer] = dWl;
	this->caches[5*layer+1] = dbl;
	
	dA_prev = Wl.transpose()*dZl;
	
	//gradient_clipping(dA_prev);
	
	return dA_prev;
}

template<class T>
inline void network<T>::backward_prop(Matrix<T>& X, Matrix<T>& y, std::string cost_function, int col_a, int col_b){
	
	T m = (T) y.getCols();
	int L = this->layers.size();
	Matrix<T> dAL, AL;
	if(col_a == -1 && col_b == -1){
		
		AL = caches[5*(L-1)+4];

	}else{

		AL = caches[5*(L-1)+4].slice_cols(col_a, col_b);
	}
	

	if (cost_function == "cross_entropy"){
			//dAL = ((y.div_ewise(AL)) - ((y * -1 + 1).div_ewise(AL*-1 + 1))) * -1;
			dAL = y.div_ewise(AL) * -1;
		
		}else if(cost_function == "mse"){
		
			dAL = (AL-y) * 2;
		
		}else{
		
			std::cout<<"Cost function should be 'cross_entropy' or 'mse'"<<std::endl;
			exit(0);
			
		}
	
	Matrix<T> dA_prev = this->linear_activation_backward(dAL, L - 1, X, col_a, col_b);
	
	for(int i = L - 2; i >= 0; i--){
	
		dA_prev = this->linear_activation_backward(dA_prev, i, X, col_a, col_b);	
	
	}

	
}

template<class T>
inline void network<T>::gradient_clipping(Matrix<T>& dA){
	
	Matrix<T> da(dA.getRows(),1);
	T c = 1;
	
	for(int i=0;i<dA.getCols();i++){
		for(int k=0;k<dA.getRows();k++){
			da(k,0) = dA(k,i);
		}
		
		if(abs(da.norm()) >= c){
			da = da*(c/(da.norm()));
			for(int k=0;k<dA.getRows();k++){
				dA(k,i) = da(k,0);
			}	
		}
	}
}


template<class T>
inline T network<T>::compute_cost(Matrix<T>& AL, Matrix<T>& y, std::string cost_function){

	T cost = 0;
	T m = (T) y.getCols();
	Matrix<T> C;
	
	if(cost_function == "cross_entropy"){
		
		C = y.mult_ewise(AL.logM())*-1;
			
		for(int j=0;j<y.getCols();j++){
			for(int i =0 ;i<y.getRows();i++){
				cost += C(i,j);	
			}
		}
	}
			 
	else if(cost_function == "mse"){
		
		C = (y-AL).mult_ewise(y-AL);
		for(int j=0;j<y.getCols();j++){
			for(int i =0 ;i<y.getRows();i++){
				cost += C(i,j);	
			}
		}
	}
		
return 1/m * cost;

}


template<class T>
inline void network<T>::update_weights(T learning_rate){
	
	// gradient descent
	for(int i = 0;i < this->layers.size();i++){
		this->get_W(i+1) = this->get_W(i+1) - this->caches[5*i] * learning_rate; // W = W - alpha * dJ/dW
		this->get_b(i+1) = this->get_b(i+1) - this->caches[5*i + 1] * learning_rate; // b = b - alpha * dJ/db
	}

}

template<class T>
inline Matrix<T>& network<T>::get_W(unsigned layer_number){
	if(layer_number <= 0 || layer_number > this->layers.size()){
		std::cout<<"Invalid layer." << std::endl;
		exit(0);
	}

	return this->layers[layer_number - 1]->get_W();
}

template<class T>
inline Matrix<T>& network<T>::get_b(unsigned layer_number){
	
	if(layer_number <= 0 || layer_number > this->layers.size()){
		std::cout<<"Invalid layer." << std::endl;
		exit(0);
	}
	
	return this->layers[layer_number - 1]->get_b();
}


template<class T>
inline network<T>::network(unsigned input_size){
		this->input_size = input_size;
}
		
template<class T>
inline void network<T>::add_Dense(unsigned neurons, std::string activation, T initialize_weights){
	unsigned columns;
	if(layers.empty()){
		columns = this->input_size;
	}else{
		columns = layers.back()->get_b().getRows();		
	}
	
	Matrix<T> W(neurons, columns), b(neurons, 1);
	W.randn();
	W = W * initialize_weights;
	
	Dense_Layer<T> *DL = new Dense_Layer<T>(W, b, activation);
	layers.push_back(DL);		
	}
		
template<class T>
inline void network<T>::set_Dense(T *W_, T *b_, unsigned n_rows, unsigned n_cols, std::string activation){
	
	unsigned columns;
	if(layers.empty()){
		columns = this->input_size;
	}else{
		columns = layers.back()->get_b().getRows();		
	}
	
	if(n_cols != columns){
		std::cout<<"Matrices shape of layer "<<this->layers.size() + 1 << "are not compatible with previous layer"<<std::endl;
		exit(0);
	}
	
	Matrix<T> W(W_, n_rows, n_cols), b(b_, n_rows, 1);
	Dense_Layer<T> *DL = new Dense_Layer<T>(W, b, activation);
	layers.push_back(DL);		

}
	
template<class T>
inline void network<T>::summary(){
	
	std::cout<< "Input Layer " << "-> Features: " << this->input_size<<std::endl;
	
	for(int i = 0; i < layers.size(); i++){
		std::cout<< "Layer "<< i + 1 <<"-> Neurons: " << layers[i]->get_b().getRows()<<"; Activation: "<<layers[i]->get_activation()<<std::endl;
	}
	
}
		
template<class T>
inline Matrix<T> network<T>::predict(Matrix<T>& input){
	
	if(input.getRows() != this->input_size){
		std::cout<<"Input rows need to be = "<< this->input_size << " (x(features,m))"<<std::endl;
		exit(0);
	}
	
	Matrix<T> out = input;
	
	for(int i = 0; i < this->layers.size(); i++){
		out = this->layers[i]->output(out);
	}
	
	return out;
}

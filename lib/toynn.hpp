
#ifndef TOYNN_HPP
#define TOYNN_HPP

#ifndef LOCATION
#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ " : " S2(__LINE__)
#endif

#ifndef MATRIX_HPP
#include"matrix.hpp"
#endif

#ifndef CMATH_H
#include<cmath>
#endif

#ifndef  STDLIB_H
#include <stdlib.h>     /* srand, rand */
#endif

#ifndef TIME_H
#include <time.h> 
#endif
#ifndef FSTREAM_H
#include <fstream> 
#endif
// Activationfunction for storing function and its derivative
class ActivationFunction{
    public:
    long double (*f)(long double);
    long double (*derf)(long double);
    std::string fname;
    ActivationFunction(){

    }
    ActivationFunction operator=(ActivationFunction e){
        this->f = e.f;
        this->derf = e.derf; 
        this->fname = e.fname;
    }
    ActivationFunction(std::string fname,long double (*f)(long double),long double (*derf)(long double)){
        this->f = f;
        this->derf = derf;
        this->fname = fname;
    }
};

//sigmoid activation
long double sigmoid_f(long double x){
    return 1/(1+expl(-x));
}
long double sigmoid_derf(long double x){
    return x*(1-x);
}

//tanh activation
long double tanh_f(long double x){
    return tanhl(x);
}
long double tanh_derf(long double x){
    return 1-x*x;
}

//helper for getting random initialization first
long double get_random(long double _){
    return ((long double)rand()/RAND_MAX)*2 - 1; 
    // random between -1 and 1
}
// Exception helper 
class ToyNNInputException{
    std::string error_message,loc;
    public:
    ToyNNInputException(std::string msg,std::string loc){
        this->error_message = msg;
        this->loc = loc;
    }
    std::string getError(){
        return this->loc+" : ToyNNInputException : "+this->error_message;
    }   
};

class NeuralNetwork{
    unsigned int input_nodes,hidden_nodes,output_nodes;
    Matrix weights_ih,weights_ho,bias_h,bias_o;
    ActivationFunction activation_function;
    long double lr;
    public:
    NeuralNetwork(){}//an empty constructor
    NeuralNetwork(unsigned int ni,unsigned int nh,unsigned no,bool reload){
        // when loading of trained weights from a file
        this->input_nodes = ni;
        this->hidden_nodes = nh;
        this->output_nodes = no;
        srand((unsigned)time(NULL));
    }
    NeuralNetwork(unsigned int ni,unsigned int nh,unsigned no){
        //normal network intialization
        this->input_nodes = ni;
        this->hidden_nodes = nh;
        this->output_nodes = no;

        this->weights_ih = Matrix(this->hidden_nodes,this->input_nodes);
        this->weights_ho = Matrix(this->output_nodes,this->hidden_nodes);
        srand((unsigned)time(NULL));

        // randomizing weights
        this->weights_ih.map(get_random);
        this->weights_ho.map(get_random);

        this->bias_h = Matrix(this->hidden_nodes,1);
        this->bias_o = Matrix(this->output_nodes,1);

        //randomizing bias
        this->bias_h.map(get_random);
        this->bias_o.map(get_random);

        //default ones
        this->setLearningRate(0.1);
        this->setActivationFunction(ActivationFunction("sigmoid",sigmoid_f,sigmoid_derf));
    }
    Matrix predict(Matrix inputs){
        // FeedForward Algorithm
        if(inputs.getNRows() != this->input_nodes)
            throw ToyNNInputException("in function predict() no correct input is given ",LOCATION);
        Matrix hidden = this->weights_ih * inputs;
        hidden = hidden + this->bias_h;
        hidden.map(this->activation_function.f);
        
        Matrix outputs = this->weights_ho * hidden;
        outputs = outputs + this->bias_o;
        outputs.map(this->activation_function.f);

        return outputs;
    }
    static Matrix softmax(Matrix outputs){
        if(outputs.getNCols() != 1){
            throw ToyNNInputException("in softmax() the matrix should be nx1 size ",LOCATION);
        }
        long double sum=0.0;
        outputs.map(expl);
        for(int i=0;i<outputs.getNRows();i++){
            sum+=outputs.at(i,0);
        }
        for(int i=0;i<outputs.getNRows();i++){
            outputs.assign(i,0,outputs.at(i,0)/sum);
        }
        return outputs;
    }

    void train(Matrix inputs,Matrix targets){
        // Back Propagation Algorithm
        if(inputs.getNRows() != this->input_nodes)
            throw ToyNNInputException("in function predict() no correct input is given",LOCATION);
        if(targets.getNRows() != this->output_nodes)
            throw ToyNNInputException("in function predict() no correct output size is given",LOCATION);
        // first FeedForward
        Matrix hidden = this->weights_ih * inputs;
        hidden = hidden + this->bias_h;
        hidden.map(this->activation_function.f);

        Matrix outputs = this->weights_ho * hidden;
        outputs = outputs + this->bias_o;
        outputs.map(this->activation_function.f);

        //ERRORS = TRAGETS - OUTPUTS

        //1).OUTPUTS TO HIDDEN part

        //calculte output layer errors
        Matrix output_errors = targets - outputs;
        //gradients i.e. gradients = df(x)*errors*learning_rate
        Matrix gradients = Matrix::map(outputs,this->activation_function.derf);
        gradients = gradients * output_errors;
        gradients = gradients * this->lr;

        //calculate amount of changes by gradients i.e. deltas
        Matrix hidden_t = Matrix::transpose(hidden);
        Matrix weights_ho_deltas = gradients * hidden_t;

        //adjust weights by it's deltas
        this->weights_ho = this->weights_ho + weights_ho_deltas;
        //adjust bias by its deltas i.e. gradients
        this->bias_o = this->bias_o + gradients;
        
        // 2).HIDEEN TO INPUTS  part
        //calculate hidden layer errors
        Matrix weigh_ho_t = Matrix::transpose(this->weights_ho);
        Matrix hidden_errors = weigh_ho_t * output_errors;

        //hidden layer gradients 
        Matrix hidden_gradient = Matrix::map(hidden,this->activation_function.derf);
        hidden_gradient = hidden_gradient * hidden_errors;
        hidden_gradient = hidden_gradient * this->lr;

        // changes i.e. deltas
        Matrix inputs_t = Matrix::transpose(inputs);
        Matrix weight_ih_deltas = hidden_gradient * inputs_t;
        
        // adjust weights by deltas
        this->weights_ih = this->weights_ih + weight_ih_deltas;
        this->bias_h = this->bias_h + hidden_gradient;
    }

    void setLearningRate(long double lr=0.1){
        this->lr = lr;
    }
    void setActivationFunction(ActivationFunction f){
        this->activation_function = f;
    }
    static void save(NeuralNetwork nn,std::string filename){
        std::ofstream outp;
        outp.open(filename.c_str());
        if(!outp){
            throw ToyNNInputException("Something wrong while saving file "+filename,LOCATION);
        }
        // write NeuralNetwork size
        outp<<nn.input_nodes<<" "<<nn.hidden_nodes<<" "<<nn.output_nodes<<"\n";
        // write weights
        Matrix::write(nn.weights_ih,outp);
        Matrix::write(nn.weights_ho,outp);
        Matrix::write(nn.bias_h,outp);
        Matrix::write(nn.bias_o,outp);
        //write actiation function name
        outp<<nn.activation_function.fname;
        outp.close();
    }
    static NeuralNetwork load(std::string filename){
        std::ifstream infile;
        infile.open(filename.c_str());
        unsigned int input_nodes,hidden_nodes,output_nodes;
        if(!infile){
            throw ToyNNInputException("in load() File: "+filename+" is not found",LOCATION);
        }
        infile>>input_nodes>>hidden_nodes>>output_nodes;// read Neural Network Size
        NeuralNetwork nn(input_nodes,hidden_nodes,output_nodes,true);
        // read weights
        nn.weights_ih = Matrix::read(infile);
        nn.weights_ho = Matrix::read(infile);
        nn.bias_h = Matrix::read(infile);
        nn.bias_o = Matrix::read(infile);
        // read activation function name
        std::string fname;
        infile>>fname;
        if(fname == "sigmoid"){
            nn.setActivationFunction(ActivationFunction("sigmoid",sigmoid_f,sigmoid_derf));
        }else if(fname == "tanh"){
            nn.setActivationFunction(ActivationFunction("tanh",tanh_f,tanh_derf));
        }
        infile.close();
        // nn.weights_ih.print();
        // nn.weights_ho.print();
        // nn.bias_h.print();
        // nn.bias_o.print();
        return nn;
    }
};
#endif
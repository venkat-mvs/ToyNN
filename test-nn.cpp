#include<iostream>
#include<sstream>
#include<iomanip>
#include"lib/toynn.hpp"
using namespace std;

/*
Problem : A color predictor,on background of rgb colors
which would be the better as text color either white or black. 
Used Algo: ToyNeuralNetwork (implemented in c++)
Algo inputs: 3x1 Matrix of [r, g, b]
ALgo outputs: 2x1 Matrix of [probability to black, probability of white]
*/


int main(){
    srand((unsigned)time(NULL));
    int data_size = 1000;
    int nepochs = 5;
    NeuralNetwork nn(3,3,2);// NeuralNetwork with (3 inputs,3 hidden,2 output) 
    try{
        long double r,g,b;
        Matrix preds;
        int tcorrect;
        // Data prepration
        Matrix *x;int *y;
        x = new Matrix[data_size];
        y = new int[data_size];
        for(int i=0;i<data_size;i++){
            x[i] = Matrix(3,1);
            r = rand()%256; 
            g = rand()%256; 
            b = rand()%256;
            x[i].assign(0,0,r);
            x[i].assign(1,0,g);
            x[i].assign(2,0,b);
            x[i] = x[i]/255;//normalize
            if(r+g+b >300)
                y[i] = 0;//black
            else
                y[i] = 1; //white
        }
        // training part
        for(int epoch=1;epoch<=nepochs;epoch++){
            // each epoch
            // evaluate
            tcorrect = 0.0;
            for(int i=0;i<data_size;i++){
                preds = NeuralNetwork::softmax(nn.predict(x[i]));
                int index = -1;
                //max 
                if(preds.at(0,0)>preds.at(1,0)){
                    index = 0;
                }else{
                    index = 1;
                }
                if(y[i] == index)
                    tcorrect++;    
            }
            cout<<"`\taccuracy : "<<(double)((double)tcorrect/data_size)*100<<"%"<<endl;
            // train
            for(int i=0;i<data_size;i++){
                Matrix targets(2,1);
                targets.fill(0);
                targets.assign(y[i],0,1);
                nn.train(x[i],targets);
            }
            cout<<"\tEpoch "<<epoch<<" completed.\n";
        }
        cout<<"\ttraining done.\n";

        // evaluating part
        tcorrect = 0;
        for(int i=0;i<data_size;i++){
            preds = nn.predict(x[i]);
            r = x[i].at(0,0);
            g = x[i].at(1,0);
            b = x[i].at(2,0); 
            int index = -1;
            if(preds.at(0,0)>preds.at(1,0)){
                index = 0;
            }else{
                index = 1;
            }
            if(y[i] == index)
                tcorrect++; 
        }
        cout<<"Obtained Accuracy : "<<(double)((double)tcorrect/data_size)*100<<"%"<<endl;

        NeuralNetwork::save(nn,"weights.txt");
        cout<<"weights are saved";
    }catch(ToyNNInputException e){
        cerr<<e.getError();
    }
    catch(MatrixException e){
        cerr<<e.getError();
    }
}
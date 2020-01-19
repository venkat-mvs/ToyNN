
#ifndef MATRIX_HPP
#define MATRIX_HPP

#ifndef STRING_H
#include<string>
#endif

#ifndef LOCATION
#define S1(x) #x
#define S2(x) S1(x)
#define LOCATION __FILE__ " : " S2(__LINE__)
#endif

#ifndef FSTREAM_H
#include<fstream>
#endif
class MatrixException{
	std::string error_message,loc;
	public:
	MatrixException(std::string s,std::string loc){
		this->error_message = s;
		this->loc = loc;
	}
    std::string getError(){
        return this->loc+" : MatrixException : "+this->error_message;
    }

};
class Matrix{
	unsigned int rows,cols;
	long double **mat;
	public:
	Matrix(){
		this->rows = 0;
		this->cols = 0;
		mat = NULL;
	}
	Matrix(unsigned int n,unsigned int m){
		this->rows = n;
		this->cols = m;
		mat = new long double*[n];
		for(int i=0;i<n;i++)
			mat[i] = new long double[m];
	}
	unsigned int getNRows(){// ok
		return this->rows;
	}
	unsigned int getNCols(){// ok
		return this->cols;
	}
	void map(long double (*f)(long double)){// ok
		for(int i=0;i<this->rows;i++){
			for(int j=0;j<this->cols;j++){
				mat[i][j] = f(mat[i][j]);
			}
		}
	}
	static void write(Matrix omat,std::ofstream &ofp){//ok
		ofp<<omat.rows<<" "<<omat.cols;
		ofp<<"\n";
		for(int i=0;i<omat.rows;i++){
			for(int j=0;j<omat.cols;j++){
				ofp<<omat.mat[i][j]<<" ";
			}
			ofp<<"\n";
		}
	}
	static Matrix read(std::ifstream &ifp){//ok
		// read size
		unsigned int nrows,ncols;
		ifp>>nrows;
		ifp>>ncols;

		// allocate memory
		Matrix newmat(nrows,ncols);
		// read matrix
		for(int i=0;i<nrows;i++){
			for(int j=0;j<ncols;j++){
				ifp>>newmat.mat[i][j];
			}
		}
		return newmat;
	}
	static Matrix map(Matrix,long double (*f)(long double));// ok
	Matrix multiply(Matrix); // ok
    Matrix multiply(long double);// ok
	Matrix hadmard_product(Matrix);//ok
	Matrix add(Matrix);// ok
	Matrix add(long double);// ok
	Matrix subtract(Matrix);// ok
    Matrix subtract(long double);// ok
	static Matrix transpose(Matrix);// ok
	Matrix operator*(Matrix);// ok
	Matrix operator*(long double);// ok
	Matrix operator+(Matrix);// ok
	Matrix operator+(long double);// ok
	Matrix operator-(Matrix);// ok
	Matrix operator-(long double);// ok
	Matrix operator/(const long double);// ok
    static Matrix fill(unsigned int,unsigned int,long double);// ok

	void print(unsigned int digits=10){// ok
		std::cout.precision(digits);
		digits = 6;
		std::cout<<std::endl;
		for(int i=0;i<this->cols;i++)
			std::cout<<std::setw(digits+8)<<i;
		std::cout<<std::endl;
		for(int i=0;i<this->rows;i++){
			std::cout<<i<<"|";
			for(int j=0;j<this->cols;j++){
			
				std::cout<<std::setw(digits+8)<<mat[i][j];
			}
			std::cout<<"|"<<std::endl;
		}
		std::cout<<std::endl;
	} 
	void fill(long double value){// ok
		for(int i=0;i<this->rows;i++){
			for(int j=0;j<this->cols;j++){
				mat[i][j] = value;
			}
		}
	}
	Matrix operator=(Matrix omat){// ok
		this->rows = omat.rows;
		this->cols = omat.cols;
		mat = new long double*[this->rows];
		for(int i=0;i<this->rows;i++)
			mat[i] = new long double[this->cols];
		for(int i=0;i<this->rows;i++){
			for(int j=0;j<this->cols;j++){
				mat[i][j] = omat.mat[i][j];
			}
		}
		return *this;
	}
	Matrix formArray(long double **amat,unsigned int n,unsigned int m){
		Matrix omat(n,m);
	}

	long double assign(unsigned int i,unsigned int j,long double value){// ok
		mat[i][j] = value;
		return value;
	}

	long double at(unsigned int i,unsigned int j){// ok
		return this->mat[i][j];
	}
};

Matrix Matrix::map(Matrix other,long double (*f)(long double)){
	Matrix newmat(other.rows,other.cols);
	newmat.fill(0);
	for(int i=0;i<newmat.rows;i++){
			for(int j=0;j<newmat.cols;j++){
				newmat.mat[i][j] = f(other.mat[i][j]);
		}
	}
	return newmat;
}

Matrix Matrix::fill(unsigned int n,unsigned int m,long double value){
	Matrix newmat(n,m);
	newmat.fill(value);
	return newmat;
}
Matrix Matrix::hadmard_product(Matrix omat){
	Matrix mulmat(this->rows,this->cols);
	for(int i=0;i<mulmat.rows;i++){
		for(int j=0;j<mulmat.cols;j++){
			mulmat.mat[i][j] = this->mat[i][j] * omat.mat[i][j];
		}
	}
	return mulmat;
}
Matrix Matrix::multiply(Matrix omat){ 
	// std::cout<<"Matrix("<<this->rows<<","<<this->cols<<")*";
	// std::cout<<"Matrix("<<omat.rows<<","<<omat.cols<<")"<<std::endl
	if(this->rows == omat.rows && this->cols == omat.cols){
		return this->hadmard_product(omat);
	}
	if(this->cols != omat.rows)
		throw MatrixException("in function multiply() columns and rows are not matching",LOCATION);
	Matrix mulmat(this->rows,omat.cols);
	mulmat.fill(0);
	for(int i=0;i<mulmat.rows;i++){
		for(int j=0;j<mulmat.cols;j++){
			for(int k=0;k<this->cols;k++){
			mulmat.mat[i][j] += this->mat[i][k] * omat.mat[k][j] ;
			//std::cout<<this->mat[i][j]<<","<<omat.mat[k][j]<<mulmat.mat[i][j]<<"\n";
			} 
		}
	}
	return mulmat;
}
Matrix Matrix::operator*(Matrix omat){
	return this->multiply(omat);
}
Matrix Matrix::multiply(long double value){
	Matrix mulmat(this->rows,this->cols);
	mulmat.fill(0);
	for(int i=0;i<mulmat.rows;i++){
		for(int j=0;j<mulmat.cols;j++){
			mulmat.mat[i][j] += this->mat[i][j]*value ; 
		}
	}
	return mulmat;
}
Matrix Matrix::operator*(long double value){
	return this->multiply(value);
}

Matrix Matrix::add(Matrix omat){
	// std::cout<<"Matrix("<<this->rows<<","<<this->cols<<")+";
	// std::cout<<"Matrix("<<omat.rows<<","<<omat.cols<<")"<<std::endl;
	if(omat.rows != this->rows && omat.cols != this->cols)
		throw MatrixException("in function add() rows and columns doesnot match",LOCATION);

	Matrix addmat(omat.rows,omat.cols);
	for(int i=0;i<addmat.rows;i++){
		for(int j=0;j<addmat.cols;j++){
			addmat.mat[i][j] = this->mat[i][j] + omat.mat[i][j];
		}
	}
	return addmat;
}
Matrix Matrix::operator+(Matrix omat){
	return this->add(omat);
}
Matrix Matrix::add(long double value){
	Matrix addmat(this->rows,this->cols);
	for(int i=0;i<addmat.rows;i++){
		for(int j=0;j<addmat.cols;j++){
			addmat.mat[i][j] = this->mat[i][j] + value;
		}
	}
	return addmat;
}
Matrix Matrix::operator+(long double value){
	return this->add(value);
}
Matrix Matrix::subtract(Matrix omat){
	if(omat.rows != this->rows || omat.cols != this->cols)
		throw MatrixException("in function subtract() rows and columns doesnot match",LOCATION);

	Matrix submat(omat.rows,omat.cols);
	for(int i=0;i<submat.rows;i++){
		for(int j=0;j<submat.cols;j++){
			submat.mat[i][j] = this->mat[i][j] - omat.mat[i][j];
		}
	}
	return submat;
}
Matrix Matrix::operator-(Matrix omat){
	return this->subtract(omat);
}
Matrix Matrix::subtract(long double value){
	Matrix submat(this->rows,this->cols);
	for(int i=0;i<submat.rows;i++){
		for(int j=0;j<submat.cols;j++){
			submat.mat[i][j] = this->mat[i][j] - value;
		}
	}
	return submat;
}
Matrix Matrix::operator-(long double value){
	return this->subtract(value);
}
Matrix Matrix::operator/(long double value){
	Matrix submat(this->rows,this->cols);
	for(int i=0;i<submat.rows;i++){
		for(int j=0;j<submat.cols;j++){
			submat.mat[i][j] = this->mat[i][j] / value;
		}
	}
	return submat;
}
Matrix Matrix::transpose(Matrix omat){
    //if(omat.cols != omat.rows){
    //    throw MatrixException("in function tarnspose() rows and cols doesn't match"+LOCATION);
    //}
    Matrix tmat(omat.cols,omat.rows);
    for(int i=0;i<tmat.rows;i++){
        for(int j=0;j<tmat.cols;j++){
            tmat.mat[i][j] = omat.mat[j][i];
        }
    }
    return tmat;
}
#endif
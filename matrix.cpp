#include <iostream>
#include <fstream>
#include <vector>
#include <random>

using namespace std;


const unsigned int DEFAULT_SEED = 7833466; // STEF(an) ION(ut)
const unsigned int MAX_VAL = 10;


class Matrix {
    class Row {
        // in order to make the double-index operator [][] work
    public:
        vector<double> _values;

        Row(long length, double init_value=0) {
            _values.assign(length, init_value);
        }

        double& operator[](long index) {
            return _values[index];
        }
        double operator[](long index) const {
            return _values[index];
        }

        long length() const {
            return _values.size();
        }
    };

    vector<Row> _rows;

public:
    Matrix() { /* nothing? */ }

    Matrix(long n_rows, long n_cols) {
        Row blank_row(n_cols, 0);
        _rows.assign(n_rows, blank_row);
    }

    Matrix(string filename) {
        // int ReverseInt (int i)
        // {
        //     unsigned char ch1, ch2, ch3, ch4;
        //     ch1=i&255;
        //     ch2=(i>>8)&255;
        //     ch3=(i>>16)&255;
        //     ch4=(i>>24)&255;
        //     return((int)ch1<<24)+((int)ch2<<16)+((int)ch3<<8)+ch4;
        // }
        // void ReadMNIST(int NumberOfImages, int DataOfAnImage,vector<vector<double>> &arr)
        // {
        //     arr.resize(NumberOfImages,vector<double>(DataOfAnImage));
        //     ifstream file ("C:\\t10k-images.idx3-ubyte",ios::binary);
        //     if (file.is_open())
        //     {
        //         int magic_number=0;
        //         int number_of_images=0;
        //         int n_rows=0;
        //         int n_cols=0;
        //         file.read((char*)&magic_number,sizeof(magic_number));
        //         magic_number= ReverseInt(magic_number);
        //         file.read((char*)&number_of_images,sizeof(number_of_images));
        //         number_of_images= ReverseInt(number_of_images);
        //         file.read((char*)&n_rows,sizeof(n_rows));
        //         n_rows= ReverseInt(n_rows);
        //         file.read((char*)&n_cols,sizeof(n_cols));
        //         n_cols= ReverseInt(n_cols);
        //         for(int i=0;i<number_of_images;++i)
        //         {
        //             for(int r=0;r<n_rows;++r)
        //             {
        //                 for(int c=0;c<n_cols;++c)
        //                 {
        //                     unsigned char temp=0;
        //                     file.read((char*)&temp,sizeof(temp));
        //                     arr[i][(n_rows*r)+c]= (double)temp;
        //                 }
        //             }
        //         }
        //     }
        // }
    }

    Row& operator[](long index) {
        return _rows[index];
    }
    Row operator[](long index) const {
        return _rows[index];
    }

    double* array_from_row(long r) {
        return _rows[r]._values.data();
    }

    long n_rows() const { return _rows.size(); }
    long n_cols() const {
        if (_rows.empty())
            return 0;
        return _rows[0].length();
    }

    void randomize_values() {
        for (long i = 0; i < n_rows(); ++i)
            for (long j = 0; j < n_cols(); ++j)
                _rows[i][j] = rand() % MAX_VAL;
    }

    friend ostream &operator<<(ostream &os, const Matrix::Row &row);
    friend ostream &operator<<(ostream &os, const Matrix &matrix);
};

ostream& operator<<(ostream& os, const Matrix::Row& row) {
    for (const auto& elem : row._values)
        os << elem << ' ';
    return os;
}

ostream& operator<<(ostream& os, const Matrix& matrix) {
    for (const auto& row : matrix._rows) {
        os << row << endl;
    }
    return os;
}

Matrix operator* (const Matrix& a, const Matrix& b) {
    if (a.n_cols() != b.n_rows())
        throw "Dimensions do not agree for matrix multiplication!";

    long n = a.n_rows();
    long p = a.n_cols();
    long m = b.n_cols();
    Matrix c(n, m);

    // http://stackoverflow.com/questions/10163948/multiplying-two-matrices-with-different-dimensions
    for (long i = 0; i < n; ++i)
        for (long j = 0; j < m; ++j)
            for (long k = 0; k < p; ++k)
                c[i][j] += a[i][k] * b[k][j];

    return c;
}

Matrix operator+ (const Matrix& a, const Matrix& b) {
    long n_rows = a.n_rows();
    long n_cols = a.n_cols();

    if (b.n_rows() != n_rows || b.n_cols() != n_cols)
        throw "Dimensions are not the same for for matrix addition!";

    Matrix result(n_rows, n_cols);
    for(long r = 0; r < n_rows; ++r)
        for(long c = 0; c < n_cols; ++c)
            result[r][c] = a[r][c] + b[r][c];

    return result;
}

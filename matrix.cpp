#include <iostream>
#include <fstream>
#include <vector>
#include <random>
#include "mpi.h"

#include <unistd.h>

using namespace std;


const unsigned int DEFAULT_SEED = 7833466; // STEF(an) ION(ut)
const unsigned int MAX_VAL = 10;


class Matrix {
public:
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

    Matrix() { /* nothing? */ }

    Matrix(long n_rows, long n_cols) {
        Row blank_row(n_cols, 0);
        _rows.assign(n_rows, blank_row);
    }

    Matrix(string filename) {
        ifstream file(filename);

        long n_rows, n_cols;
        file >> n_rows >> n_cols;
        Matrix m(n_rows, n_cols);

        double x;
        for (long i = 0; i < n_rows; ++i)
            for (long j = 0; j < n_cols; ++j) {
                file >> x;
                m[i][j] = x;
            }

        _rows = m._rows;
        file.close();
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

Matrix multiply(const Matrix& lhs, const Matrix& rhs, long row_from, long row_to) {
    if (lhs.n_cols() != rhs.n_rows())
        throw "Dimensions do not agree for matrix multiplication!";

    long n_rows = lhs.n_rows();
    long p      = lhs.n_cols();
    long n_cols = lhs.n_cols();

    Matrix result(n_rows, n_cols);
    for (long r = row_from; r < row_to; ++r)
        for (long c = 0; c < n_cols; ++c)
            for (long k = 0; k < p; ++k)
                result[r][c] += lhs[r][k] * rhs[k][c];

    return result;
}

void dummy_multiplication(long matrix_dim)
{
    srand(DEFAULT_SEED);

    // Generate matrices for input
    Matrix a(matrix_dim, matrix_dim);
    Matrix b(matrix_dim, matrix_dim);
    a.randomize_values();
    b.randomize_values();

    // Initialize MPI
    int rank, n_processes;
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);
    MPI_Comm_size(MPI_COMM_WORLD, &n_processes);

    // Number of rows each worker process gets
    const int tag = 0;
    long chunk_rows = matrix_dim / (n_processes - 1);

    // Master
    if (rank == 0) {
        Matrix result(matrix_dim, matrix_dim); // initialize result Matrix with zeros

        // Wait for each worker to finish
        for (int worker = 1; worker < n_processes; ++worker) {
            long start_row = chunk_rows * (worker - 1); // workers start from rank 1
            for (long row = start_row; row < start_row + chunk_rows; ++row)
                // The result has to be sent row-by-row, instead of all at once
                // because std vectors are guaranteed to be contiguous
                // but not 2-dimensional vectors
                MPI_Recv(result.array_from_row(row), matrix_dim, MPI_DOUBLE, worker,
                         tag, MPI_COMM_WORLD, MPI_STATUS_IGNORE);
        }

    }

    // Workers
    else {
        long start_row = chunk_rows * (rank - 1); // workers start from rank 1
        Matrix partial_result = multiply(a, b, start_row, start_row + chunk_rows);

        // See comment on master's task on reason for sending row-by-row
        for (long row = start_row; row < start_row + chunk_rows; ++row)
            MPI_Send(partial_result.array_from_row(row), matrix_dim, MPI_DOUBLE, 0,
                     tag, MPI_COMM_WORLD);
    }
}

// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "psMatrix.h"
#include "PsuadeUtil.h"

//#define PS_DEBUG 1
// ************************************************************************
// ************************************************************************
// external functions
// ------------------------------------------------------------------------
extern "C" {
  void dgetri_(int *, double *, int *, int *, double *, int *, int *);
  void dgels_(char *, int *, int *, int *, double *, int *,
              double *, int *, double *, int *, int *);
  void dsyev_(char *, char *, int *, double *, int *, double *,
              double *, int *, int *);
  void dpotrf_(char *, int *, double *, int *, int *);
  void dpotrs_(char *, int *, int *, double *, int *, double *,int *,int *);
  void dgetrf_(int *, int *, double *, int *, int *, int *);
  void dgetrs_(char *,int *,int *,double *,int *,int *,double *,int *,int *);
  void dgesvd_(char *, char *, int *, int *, double *, int *, double *,
               double *, int *, double *, int *, double *, int *, int *);
}

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psMatrix::psMatrix()
{
#ifdef PS_DEBUG
  printf("psMatrix constructor\n");
#endif
  nRows_ = 0;
  nCols_ = 0;
  Mat2D_ = NULL;
  Mat1D_ = NULL;
  status_ = 0;
  determinant_ = 0.0;
  pivots_ = NULL;
  format_ = 1; /* default storage format is 1D, for 2D: Mat2D[row][col] */
#ifdef PS_DEBUG
  printf("psMatrix constructor ends\n");
#endif
}

// ************************************************************************
// Copy Constructor by Bill Oliver
// ------------------------------------------------------------------------
psMatrix::psMatrix(const psMatrix & ma)
{
  int ii, jj;

  nRows_ = ma.nRows_;
  nCols_ = ma.nCols_;
  status_ = ma.status_;
  format_ = ma.format_;
  determinant_ = ma.determinant_;
  pivots_ = NULL;
  Mat2D_ = NULL;
  Mat1D_ = NULL;
  if (nRows_ > 0 && nCols_ > 0)
  {
    if (format_ == 1)
    {
      Mat1D_ = new double[nRows_*nCols_];
      assert(Mat1D_ != NULL);
      for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = ma.Mat1D_[ii];
    }
    else
    {
      Mat2D_ = new double*[nRows_];
      assert(Mat2D_ != NULL);
      for (ii = 0; ii < nRows_; ii++)
      {
        Mat2D_[ii] = new double[nCols_];
        assert(Mat2D_[ii] != NULL);
        for(jj = 0; jj < nCols_; jj++)
          Mat2D_[ii][jj] = ma.Mat2D_[ii][jj];
      }
    }
    if (ma.pivots_ != NULL && nRows_ > 0)
    {
      pivots_ = new int[nRows_];
      assert(pivots_ != NULL);
      for (ii = 0; ii < nRows_; ii++) pivots_[ii] = ma.pivots_[ii];
    } 
  }
}

// ************************************************************************
// operator=  by Bill Oliver
// ------------------------------------------------------------------------
psMatrix & psMatrix::operator=(const psMatrix & ma)
{
  int ii, jj;

  if (this == &ma) return *this;
  clean();
  nRows_ = ma.nRows_;
  nCols_ = ma.nCols_;
  status_ = ma.status_;
  format_ = ma.format_;
  determinant_ = ma.determinant_;
  if (nRows_ > 0 && nCols_ > 0)
  {
    if (format_ == 1)
    {
      Mat1D_ = new double[nRows_*nCols_];
      assert(Mat1D_ != NULL);
      for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = ma.Mat1D_[ii];
    }
    else
    {
      Mat2D_ = new double*[nRows_];
      assert(Mat2D_ != NULL);
      for(ii = 0; ii < nRows_; ii++)
      {
        Mat2D_[ii] = new double[nCols_];
        assert(Mat2D_[ii] != NULL);
        for(jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] = ma.Mat2D_[ii][jj];
      }
    }
    if (ma.pivots_ != NULL && nRows_ > 0)
    {
      pivots_ = new int[nRows_];
      assert(pivots_ != NULL);
      for (ii = 0; ii < nRows_; ii++) pivots_[ii] = ma.pivots_[ii];
    } 
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psMatrix::~psMatrix()
{
  clean();
}

// ************************************************************************
// get number of rows 
// ------------------------------------------------------------------------
int psMatrix::nrows()
{
  return nRows_;
}

// ************************************************************************
// get number of columns 
// ------------------------------------------------------------------------
int psMatrix::ncols()
{
  return nCols_;
}

// ************************************************************************
// load matrix from another matrix
// ------------------------------------------------------------------------
int psMatrix::load(psMatrix &inMat)
{
  int ii, jj;

#ifdef PS_DEBUG
  printf("psMatrix load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(this != &inMat);
  nRows_  = inMat.nrows();
  nCols_  = inMat.ncols();
  format_ = inMat.format_;
  if (nRows_ > 0 && nCols_ > 0)
  {
    if (format_ == 1)
    {
      Mat1D_ = new double[nRows_*nCols_];
      assert(Mat1D_ != NULL);
      for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = inMat.Mat1D_[ii];
    }
    else
    {
      Mat2D_ = new double*[nRows_];
      assert(Mat2D_ != NULL);
      for (ii = 0; ii < nRows_; ii++)
      {
        Mat2D_[ii] = new double[nCols_];
        assert(Mat2D_[ii] != NULL);
        for (jj = 0; jj < nCols_; jj++) 
          Mat2D_[ii][jj] = inMat.getEntry(ii,jj);
      }
    }
    if (inMat.pivots_ != NULL && nRows_ > 0)
    {
      pivots_ = new int[nRows_];
      assert(pivots_ != NULL);
      for (ii = 0; ii < nRows_; ii++) pivots_[ii] = inMat.pivots_[ii];
    } 
  }
  status_ = inMat.status_;
  determinant_ = inMat.determinant_;
#ifdef PS_DEBUG
  printf("psMatrix load ends\n");
#endif
  return 0;
}

// ************************************************************************
// load matrix from another matrix
// ------------------------------------------------------------------------
int psMatrix::load(psVector &inVec, int nrows, int ncols, char *how)
{
  int ii, jj;

#ifdef PS_DEBUG
  printf("psMatrix load\n");
#endif
  //**/ clean it first, if needed
  assert(nrows*ncols == inVec.length());
  clean();

  //**/ load from the incoming matrix
  nRows_  = nrows;
  nCols_  = ncols;
  format_ = 1;
  Mat1D_ = new double[nRows_*nCols_];
  assert(Mat1D_ != NULL);
  if (!strcmp(how, "trans"))
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++) 
        Mat1D_[ii+jj*nRows_] = inVec[ii+jj*nRows_];
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++) 
        Mat1D_[ii+jj*nRows_] = inVec[ii*nCols_+jj];
  }
  return 0;
}

// ************************************************************************
// load matrix from doubles
// ------------------------------------------------------------------------
int psMatrix::load(int nrows, int ncols, double **mat)
{
  int ii, jj;
#ifdef PS_DEBUG
  printf("psMatrix load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(nrows);
  assert(ncols);
  assert(mat);
  nRows_ = nrows;
  nCols_ = ncols;
  if (format_ == 1)
  {
    Mat1D_ = new double[nRows_*nCols_];
    assert(Mat1D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++) 
        Mat1D_[ii+jj*nRows_] = mat[ii][jj];
  }
  else
  {
    Mat2D_ = new double*[nRows_];
    assert(Mat2D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
    {
      Mat2D_[ii] = new double[nCols_];
      assert(Mat2D_[ii] != NULL);
      for (jj = 0; jj < nCols_; jj++) 
        Mat2D_[ii][jj] = mat[ii][jj];
    }
  }
  status_ = 0;
  determinant_ = 0.0;
#ifdef PS_DEBUG
  printf("psMatrix load ends\n");
#endif
  return 0;
}

// ************************************************************************
// load matrix from doubles
// ------------------------------------------------------------------------
int psMatrix::load(int nrows, int ncols, double *mat)
{
#ifdef PS_DEBUG
  printf("psMatrix load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(nrows);
  assert(ncols);
  assert(mat);
  nRows_ = nrows;
  nCols_ = ncols;
  if (format_ == 1)
  {
    Mat1D_ = new double[nrows*ncols];
    assert(Mat1D_ != NULL);
    for (int ii = 0; ii < nrows*ncols; ii++) Mat1D_[ii] = mat[ii];
  }
  else
  {
    Mat2D_ = new double*[nrows];
    for (int ii = 0; ii < nrows; ii++)
    {
      Mat2D_[ii] = new double[ncols];
      assert(Mat2D_[ii] != NULL);
      for (int jj = 0; jj < ncols; jj++)
        Mat2D_[ii][jj] = mat[ii+jj*nrows];
    }
  }
  status_ = 0;
  determinant_ = 0.0;
#ifdef PS_DEBUG
  printf("psMatrix load ends\n");
#endif
  return 0;
}

// ************************************************************************
// load a row 
// ------------------------------------------------------------------------
int psMatrix::loadRow(int rownum, int ncols, double *vec)
{
#ifdef PS_DEBUG
  printf("psMatrix loadRow\n");
#endif

  //**/ load from the incoming matrix
  assert(rownum >= 0 && rownum < nRows_);
  assert(ncols == nCols_);
  assert(vec);
  if (format_ == 1)
  {
    for (int ii = 0; ii < ncols; ii++) 
      Mat1D_[rownum+nRows_*ii] = vec[ii];
  }
  else
  {
    for (int jj = 0; jj < ncols; jj++)
      Mat2D_[rownum][jj] = vec[jj];
  }
#ifdef PS_DEBUG
  printf("psMatrix loadRow ends\n");
#endif
  return 0;
}

// ************************************************************************
// load a column 
// ------------------------------------------------------------------------
int psMatrix::loadCol(int colnum, int nrows, double *vec)
{
#ifdef PS_DEBUG
  printf("psMatrix loadCol\n");
#endif

  //**/ load from the incoming matrix
  assert(colnum >= 0 && colnum < nCols_);
  assert(nrows == nRows_);
  assert(vec);
  if (format_ == 1)
  {
    for (int ii = 0; ii < nrows; ii++) 
      Mat1D_[ii+nRows_*colnum] = vec[ii];
  }
  else
  {
    for (int jj = 0; jj < nrows; jj++)
      Mat2D_[jj][colnum] = vec[jj];
  }
#ifdef PS_DEBUG
  printf("psMatrix loadCol ends\n");
#endif
  return 0;
}

// ************************************************************************
// set matrix dimension
// ------------------------------------------------------------------------
int psMatrix::setDim(int nrows, int ncols)
{
  int ii, jj;

  //**/ clean it first, if needed
  clean();

  //**/ allocate matrix and set to 0
  nRows_ = nrows;
  nCols_ = ncols;
  if (nRows_ <= 0 || nCols_ <= 0) return -1;
  if (format_ == 1)
  {
    Mat1D_ = new double[nRows_*nCols_];
    assert(Mat1D_ != NULL);
    for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = 0;
  }
  else
  {
    Mat2D_ = new double*[nRows_];
    assert(Mat2D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
    {
      Mat2D_[ii] = new double[nCols_];
      assert(Mat2D_[ii] != NULL);
      for (jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] = 0.0;
    }
  }
  status_ = 0;
  return 0;
}

// ************************************************************************
// set entry
// ------------------------------------------------------------------------
void psMatrix::setEntry(const int row, const int col, const double ddata)
{
  //**/ error checking
  if (status_ != 0)
  {
    printf("psMatrix setEntry ERROR: matrix decomposed previously.\n");
    printf("                         Should not modify entries.\n");
    exit(1);
  }
  if (row < 0 || row >= nRows_ || col < 0 || col >= nCols_)
  {
    printf("psMatrix setEntry ERROR: index (%d,%d) out of range (%d,%d)\n",
           row, col, nRows_, nCols_);
    exit(1);
  }
  if (format_ == 1) Mat1D_[row+col*nRows_] = ddata;
  else              Mat2D_[row][col] = ddata;
  status_ = 0;
  determinant_ = 0;
  if (pivots_ != NULL)
  {
    delete [] pivots_;
    pivots_ = NULL;
  }
}

// ************************************************************************
// set format
// ------------------------------------------------------------------------
int psMatrix::setFormat(int format)
{
  if (format != 1 && format != 2)
  {
    printf("psMatrix setFormat ERROR: invalid format (should be 1 or 2)\n");
    exit(1);
  }
  format_ = format;
  clean();
  return 0;
}

// ************************************************************************
// set format
// ------------------------------------------------------------------------
int psMatrix::getFormat()
{
  return format_;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
double psMatrix::getEntry(const int row, const int col)
{
  if (row < 0 || row >= nRows_ || col < 0 || col >= nCols_)
  {
    printf("psMatrix getEntry ERROR: entry (%d,%d) (0-based) not in range\n",
           row, col);
    printf("  Matrix has %d rows and %d columns.\n", nRows_, nCols_);
    exit(1);
  }
  if (format_ == 1) return Mat1D_[row+col*nRows_];
  else              return Mat2D_[row][col];
}

// ************************************************************************
// get determinant 
// ------------------------------------------------------------------------
double psMatrix::getDeterminant()
{
  return determinant_;
}

// ************************************************************************
// get matrix in 1 dimensional array
// ------------------------------------------------------------------------
void psMatrix::getMatrix1D(psVector &mat)
{
  int ii, jj;
#ifdef PS_DEBUG
  printf("psMatrix getMatrix1D\n");
#endif
  assert(nRows_ >= 0);
  assert(nCols_ >= 0);
  mat.setLength(nRows_ * nCols_);
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_*nCols_; ii++) mat[ii] = Mat1D_[ii];
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++)
        mat[ii+jj*nRows_] = Mat2D_[ii][jj];
  }
#ifdef PS_DEBUG
  printf("psMatrix getMatrix1D ends\n");
#endif
  return;
}

// ************************************************************************
// get matrix 
// ------------------------------------------------------------------------
double *psMatrix::getMatrix1D()
{
  if (format_ == 1) return Mat1D_;
  else
  {
    printf("psMatrix getMatrix1D ERROR: internal format is 2D.\n");
    exit(1);
  }
}

// ************************************************************************
// take matrix 
// ------------------------------------------------------------------------
double *psMatrix::takeMatrix1D()
{
  double *matOut;
  if (format_ == 1) 
  {
    matOut = Mat1D_;
    Mat1D_ = NULL;
    clean();
    return matOut;
  }
  else
  {
    printf("psMatrix getMatrix1D ERROR: internal format is 2D.\n");
    exit(1);
  }
}

// ************************************************************************
// get a column 
// ------------------------------------------------------------------------
void psMatrix::getCol(const int col, psVector &vec)
{
  assert(col >= 0 && col < nCols_);
  vec.setLength(nRows_);
  if (format_ == 1) 
    for (int ii = 0; ii < nRows_; ii++) vec[ii] = Mat1D_[col*nRows_+ii];
  else
    for (int jj = 0; jj < nRows_; jj++) vec[jj] = Mat2D_[jj][col];
}

// ************************************************************************
// get matrix 
// ------------------------------------------------------------------------
double **psMatrix::getMatrix2D()
{
  if (format_ == 1)
  {
    printf("psMatrix getMatrix2D ERROR: internal format is 1D.\n");
    exit(1);
  }
  return Mat2D_;
}

// ************************************************************************
// take matrix 
// ------------------------------------------------------------------------
double **psMatrix::takeMatrix2D()
{
  double **matOut;
  if (format_ == 1)
  {
    printf("psMatrix getMatrix2D ERROR: internal format is 1D.\n");
    exit(1);
  }
  else
  {
    matOut = Mat2D_;
    Mat2D_ = NULL;
    clean();
    return matOut;
  }
}

// ************************************************************************
// extract submatrix
// ------------------------------------------------------------------------
int psMatrix::submatrix(psMatrix &inMat, const int nrows, const int ncols)
{
  if (nrows > inMat.nrows())
  {
    printf("psMatrix::submatrix ERROR : incoming matrix has less rows.\n");
    exit(1);
  }
  if (ncols > inMat.ncols())
  {
    printf("psMatrix::submatrix ERROR : incoming matrix has less cols.\n");
    exit(1);
  }
  format_ = inMat.getFormat();
  setDim(nrows, ncols);
  for (int ii = 0; ii < nrows; ii++)
  {
    for (int jj = 0; jj < ncols; jj++)
    {
      if (format_ == 1) Mat1D_[ii+jj*nRows_] = inMat.getEntry(ii,jj);
      else              Mat2D_[ii][jj]       = inMat.getEntry(ii,jj);
    }
  }
  return 0;
}

// ************************************************************************
// extract submatrix
// ------------------------------------------------------------------------
int psMatrix::submatrix(psMatrix &inMat, const int num, const int *indices)
{
  int nrows, ncols, ii, jj, row, col;

#ifdef PS_DEBUG
  printf("psMatrix submatrix\n");
#endif
  nrows = inMat.nrows();
  ncols = inMat.ncols();
  if (nrows != ncols)
  {
    printf("psMatrix::submatrix ERROR : incoming matrix is rectangular.\n");
    exit(1);
  }
  if (num <= 0 || indices == NULL)
  {
    printf("psMatrix::submatrix ERROR : no incoming indices.\n");
    exit(1);
  }
  for (ii = 0; ii < num; ii++)
  {
    row = indices[ii];
    if (row < 0 || row >= nrows)
    {
      printf("psMatrix::submatrix ERROR : index out of bound (%d)\n",row); 
      exit(1);
    }
  }
  setDim(num, num);
  format_ = inMat.getFormat();
  for (ii = 0; ii < num; ii++)
  {
    row = indices[ii];
    for (jj = 0; jj < num; jj++)
    {
      col = indices[jj];
      if (format_ == 1) Mat1D_[ii+jj*num] = inMat.getEntry(row,col);
      else              Mat2D_[ii][jj] = inMat.getEntry(row, col);
    }
  }
#ifdef PS_DEBUG
  printf("psMatrix::submatrix: incoming matrix\n");
  for (ii = 0; ii < nrows; ii++)
  {
    for (jj = 0; jj < nrows; jj++) printf("%e ",inMat.getEntry(ii,jj));;
    printf("\n");
  }
  printf("psMatrix::submatrix: outgoing matrix\n");
  for (ii = 0; ii < num; ii++)
  {
    for (jj = 0; jj < num; jj++) 
    {
      if (format_ == 1) printf("%e ",Mat1D_[ii+jj*num]);
      else              printf("%e ",Mat2D_[ii][jj]);
    }
    printf("\n");
  }
#endif
#ifdef PS_DEBUG
  printf("psMatrix submatrix ends\n");
#endif
  return 0;
}

// ************************************************************************
// Cholesky decomposition (A = L L^T)
// ------------------------------------------------------------------------
int psMatrix::CholDecompose()
{
  int    ii, jj, status=0;
  double ddata, *mat;
  char   uplo='L';

#ifdef PS_DEBUG
  printf("psMatrix CholDecompose\n");
#endif
  assert(nRows_ == nCols_);
  //**/ status == 2 means LUDecompose done before
  if (status_ == 2)
  {
    printf("psMatrix CholDecompose ERROR : LUDecompose done before.\n");
    exit(1);
  }
  else if (status_ == 1)
  {
    printf("psMatrix CholDecompose ERROR : CholDecompose done before.\n");
    exit(1);
  }

  //**/ allocate temporary array if internal storage is 2D
  if (format_ == 1) mat = Mat1D_;
  else
  {
    mat = new double[nRows_*nRows_];
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
        mat[ii+jj*nRows_] = Mat2D_[ii][jj];
    } 
  }

  //**/ perform Cholesky factorization
  dpotrf_(&uplo, &nRows_, mat, &nRows_, &status);

  //**/ copy the factored matrix if format is 2D
  if (format_ != 1)
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
        Mat2D_[ii][jj] = mat[ii+jj*nRows_]; 
    } 
    delete [] mat;
  } 
  status_ = 1;

  //**/ error checking 
  if (status != 0)
    printf("psMatrix ERROR (1): failed in Cholesky factorization.\n");
#ifdef PS_DEBUG
  printf("psMatrix CholDecompose ends\n");
#endif
  return status;
}

// ************************************************************************
// matrix vector multiply (by the L factor)  
// ------------------------------------------------------------------------
int psMatrix::CholLMatvec(psVector &ivec, psVector &ovec)
{
  int    ii, jj, status=0;
  double ddata;

#ifdef PS_DEBUG
  printf("psMatrix CholMatvec\n");
#endif
  assert(ivec.length() == nCols_);
  //**/ status == 2 means LUDecompose done before
  if (status_ == 2)
  {
    printf("psMatrix CholLMatvec ERROR : LUDecompose done before.\n");
    exit(1);
  }
  else if (status_ == 0) status = CholDecompose();

  if (status != 0)
  {
    printf("psMatrix ERROR (2): failed in Cholesky factorization.\n");
    return status;
  }
  ovec.setLength(nRows_);
  if (format_ == 1)
  {
    for (ii = nRows_-1; ii >= 0; ii--)
    {
      ddata = 0.0;
      for (jj = 0; jj <= ii; jj++) ddata += Mat1D_[ii+jj*nRows_] * ivec[jj];
      ovec[ii] = ddata;
    }
  }
  else
  {
    for (ii = nRows_-1; ii >= 0; ii--)
    {
      ddata = 0.0;
      for (jj = 0; jj <= ii; jj++) ddata += Mat2D_[ii][jj] * ivec[jj];
      ovec[ii] = ddata;
    }
  }
#ifdef PS_DEBUG
  printf("psMatrix CholMatvec ends\n");
#endif
  return 0;
}

// ************************************************************************
// Cholesky solve 
// ------------------------------------------------------------------------
int psMatrix::CholSolve(psVector &ivec, psVector &ovec)
{
  int    ii, jj, iOne=1, status=0;
  double *vec, *mat;
  char   uplo='L';

#ifdef PS_DEBUG
  printf("psMatrix CholSolve\n");
#endif
  assert(ivec.length() == nCols_);
  //**/ status == 2 means LUDecompose done before
  if (status_ == 2)
  {
    printf("psMatrix CholSolve ERROR : LUDecompose done before.\n");
    exit(1);
  }
  else if (status_ == 0) status = CholDecompose();

  if (status != 0)
  {
    printf("psMatrix ERROR (3): failed in Cholesky factorization.\n");
    return status;
  }
  ovec.setLength(nRows_);
  for (ii = 0; ii < nRows_; ii++) ovec[ii] = ivec[ii];
  vec = ovec.getDVector();
  if (format_ == 1) mat = Mat1D_;
  else
  {
    mat = new double[nRows_*nRows_];
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
        mat[ii+jj*nRows_] = Mat2D_[ii][jj];
    } 
  }
  dpotrs_(&uplo,&nRows_,&iOne, mat, &nRows_,vec, &nRows_, &status);
  if (status != 0)
  {
    printf("psMatrix ERROR (1): failed in Cholesky solve.\n");
    return status;
  }
  if (format_ != 1) delete [] mat;
#ifdef PS_DEBUG
   printf("psMatrix CholSolve ends\n");
#endif
  return 0;
}

// ************************************************************************
// Cholesky L-solve 
// ------------------------------------------------------------------------
int psMatrix::CholLSolve(psVector &ivec, psVector &ovec)
{
  int    ii, jj, status=0;
  double ddata;

#ifdef PS_DEBUG
  printf("psMatrix CholLSolve\n");
#endif
  assert(ivec.length() == nCols_);
  //**/ status == 2 means LUDecompose done before
  if (status_ == 2)
  {
    printf("psMatrix CholLSolve ERROR : LUDecompose done before.\n");
    exit(1);
  }
  else if (status_ == 0) status = CholDecompose();

  if (status != 0)
  {
    printf("psMatrix ERROR (4): failed in Cholesky factorization.\n");
    return status;
  }
  ovec.setLength(nRows_);
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      ddata = ivec[ii];
      for (jj = 0; jj < ii; jj++) ddata -= Mat1D_[ii+jj*nRows_] * ovec[jj];
      ovec[ii] = ddata / Mat1D_[ii+ii*nRows_];
    }
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      ddata = ivec[ii];
      for (jj = 0; jj < ii; jj++) ddata -= Mat2D_[ii][jj] * ovec[jj];
      ovec[ii] = ddata / Mat2D_[ii][ii];
    }
  }
#ifdef PS_DEBUG
  printf("psMatrix CholLSolve ends\n");
#endif
  return 0;
}

// ************************************************************************
// Cholesky LT-solve 
// ------------------------------------------------------------------------
int psMatrix::CholLTSolve(psVector &ivec, psVector &ovec)
{
  int    ii, jj, status=0;
  double ddata;

#ifdef PS_DEBUG
  printf("psMatrix CholTSolve (transpose)\n");
#endif
  assert(ivec.length() == nCols_);
  //**/ status == 2 means LUDecompose done before
  if (status_ == 2)
  {
    printf("psMatrix CholLSolve ERROR : LUDecompose done before.\n");
    exit(1);
  }
  else if (status_ == 0) status = CholDecompose();

  if (status != 0)
  {
    printf("psMatrix ERROR (1): failed in Cholesky factorization.\n");
    return status;
  }
  ovec.setLength(nRows_);
  if (format_ == 1)
  {
    for (ii = nRows_-1; ii > 0; ii--)
    {
      ddata = ivec[ii];
      for (jj = ii+1; jj < nRows_; jj++) 
        ddata -= Mat1D_[jj+ii*nRows_] * ovec[jj];
      ovec[ii] = ddata / Mat1D_[ii+ii*nRows_];
    }
  }
  else
  {
    for (ii = nRows_-1; ii > 0; ii--)
    {
      ddata = ivec[ii];
      for (jj = ii+1; jj < nRows_; jj++) ddata -= Mat2D_[jj][ii] * ovec[jj];
      ovec[ii] = ddata / Mat2D_[ii][ii];
    }
  }
#ifdef PS_DEBUG
  printf("psMatrix CholTSolve ends\n");
#endif
  return 0;
}

// ************************************************************************
// Compute LU factorization 
// ------------------------------------------------------------------------
int psMatrix::LUDecompose()
{
  int    ii, jj, lwork, status=0;
  double *localMatrix, *work;

  //**/ error checking and initial setup
  assert(nRows_ == nCols_);
  //**/ status == 1 means CholDecompose done before
  if (status_ == 1)
  {
    printf("psMatrix LUDecompose ERROR : CholDecompose done before.\n");
    exit(1);
  }
  else if (status_ == 2)
  {
    printf("psMatrix LUDecompose ERROR : LUDecompose done before.\n");
    exit(1);
  }
  if (pivots_ != NULL) delete [] pivots_;
  pivots_ = new int[nRows_];
  lwork = nRows_ * nRows_;
  work = new double[lwork];

  //**/ prepare matrix
  if (format_ == 1) localMatrix = Mat1D_;
  else
  {
    localMatrix = new double[nRows_ * nRows_];
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nRows_; jj++)
        localMatrix[ii*nRows_+jj] = Mat2D_[jj][ii];
  }

  //**/ perform decomposition
  dgetrf_(&nRows_, &nRows_, localMatrix, &nRows_, pivots_, &status);

  //**/ store results
  if (format_ != 1)
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nRows_; jj++)
        Mat2D_[ii][jj] = localMatrix[jj*nRows_+ii];
    delete [] localMatrix;
  }
  delete [] work;
  if (status != 0)
  {
    printf("psMatrix computeLUDecompose ERROR: failed in LUfact\n");
    delete [] pivots_;
    pivots_ = NULL;
  }
  else status_ = 2;
  return status;
}

// ************************************************************************
// LU solve 
// ------------------------------------------------------------------------
int psMatrix::LUSolve(psVector &ivec, psVector &ovec)
{
  int    ii, jj, iOne=1, status=0;
  double *vec, *mat;
  char   trans='N';

#ifdef PS_DEBUG
  printf("psMatrix LUSolve\n");
#endif

  //**/ initial checking and setup
  assert(ivec.length() == nCols_);
  if (status_ != 2)
  {
    printf("psMatrix ERROR (3): LUDecompose has not been called.\n");
    status = -1;
    return status;
  }

  //**/ prepare solution vector
  ovec.setLength(nRows_);
  for (ii = 0; ii < nRows_; ii++) ovec[ii] = ivec[ii];
  vec = ovec.getDVector();

  //**/ prepare factorized matrix
  if (format_ == 1) mat = Mat1D_;
  else
  {
    mat = new double[nRows_*nRows_];
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
        mat[ii+jj*nRows_] = Mat2D_[ii][jj];
    } 
  }

  //**/ perform solve
  dgetrs_(&trans,&nRows_,&iOne,mat,&nRows_,pivots_,vec,&nRows_,&status);
  if (format_ != 1) delete [] mat;

  if (status != 0)
  {
    printf("psMatrix ERROR (1): failed in LU solve.\n");
    return status;
  }
#ifdef PS_DEBUG
   printf("psMatrix LUSolve ends\n");
#endif
  return 0;
}

// ************************************************************************
// print matrix
// ------------------------------------------------------------------------
void psMatrix::print()
{
  printf("psMatrix print (%d,%d): \n",nRows_,nCols_);
  for (int ii = 0; ii < nRows_; ii++)
  {
    for (int jj = 0; jj < nCols_; jj++) printf("%11.4e ", getEntry(ii,jj));
    printf("\n");
  }
}

// ************************************************************************
// Compute determinant (by Bourke)
// ------------------------------------------------------------------------
double psMatrix::computeDeterminant()
{
  int    ii, jj, kk, ind;
  double result = 0.0;
  double **localMat = NULL;

  assert(nRows_ == nCols_);
  if (status_ == 1)
  {
    result = 1.0;
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_; ii++) result *= Mat1D_[ii+ii*nRows_];
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++) result *= Mat2D_[ii][ii];
    }
    return result;
  }
  if (nRows_ == 1)
  {
    if (format_ == 1) result = Mat1D_[0];
    else              result = Mat2D_[0][0];
  }
  else if (nRows_ == 2)
  {
    if (format_ == 1)
    {
      result = Mat1D_[0] * Mat1D_[1+nRows_] - Mat1D_[1] * Mat1D_[nRows_];
    }
    else
    {
      result = Mat2D_[0][0] * Mat2D_[1][1] - Mat2D_[1][0] * Mat2D_[0][1];
    }
  }
  else
  {
    result = 0.0;
    for (ii = 0; ii < nRows_; ii++)
    {
      localMat = new double*[nRows_-1];
      for (kk = 0; kk < nRows_-1; kk++)
        localMat[kk] = new double[nRows_-1];
      for (kk = 1; kk < nRows_; kk++)
      {
        ind = 0;
        for (jj = 0; jj < nRows_; jj++)
        {
          if (jj == ii) continue;
          if (format_ == 1) localMat[kk-1][ind] = Mat1D_[kk+jj*nRows_];
          else              localMat[kk-1][ind] = Mat2D_[kk][jj];
          ind++;
        }
      }
      if (format_ == 1)
      {
        result += pow(-1.0,1.0+ii+1.0) * Mat1D_[ii*nRows_] * 
                  computeDeterminant(nRows_-1, localMat);
      }
      else
      {
        result += pow(-1.0,1.0+ii+1.0) * Mat2D_[0][ii] * 
                  computeDeterminant(nRows_-1, localMat);
      }
      for (kk = 0; kk < nRows_-1; kk++) delete [] localMat[kk];
      delete [] localMat;
    }
  }
  return result;
}

// ************************************************************************
// Compute determinant (by Bourke)
// ------------------------------------------------------------------------
double psMatrix::computeDeterminant(int ndim, double **mat)
{
  int    ii, jj, kk, ind;
  double result = 0.0;
  double **localMat = NULL;

  if (ndim == 1)
  {
    result = mat[0][0];
  }
  else if (ndim == 2)
  {
    result = mat[0][0] * mat[1][1] - mat[1][0] * mat[0][1];
  }
  else
  {
    result = 0.0;
    for (ii = 0; ii < ndim; ii++)
    {
      localMat = new double*[ndim-1];
      for (kk = 0; kk < ndim-1; kk++)
        localMat[kk] = new double[ndim-1];
      for (kk = 1; kk < ndim; kk++)
      {
        ind = 0;
        for (jj = 0; jj < ndim; jj++)
        {
          if (jj == ii) continue;
          localMat[kk-1][ind] = mat[kk][jj];
          ind++;
        }
      }
      result += pow(-1.0,1.0+ii+1.0) * mat[0][ii] * 
                computeDeterminant(ndim-1, localMat);
      for (kk = 0; kk < ndim-1; kk++) delete [] localMat[kk];
      delete [] localMat;
    }
  }
  return(result);
}

// ************************************************************************
// Compute inverse 
// ------------------------------------------------------------------------
int psMatrix::computeInverse(psMatrix &inverse)
{
  int    ii, jj, *ipiv, lwork, status;
  double *localMatrix, *work, ddata;

  assert(nRows_ == nCols_);
  if (nRows_ == 1)
  {
    inverse.setDim(nRows_, nRows_);
    if (format_ == 1) ddata = Mat1D_[0];
    else              ddata = Mat2D_[0][0];
    if (ddata == 0)
    {
      printf("psMatrix computeInverse ERROR: entry = 0\n");
      exit(1);
    }
    ddata = 1.0 / ddata;
    inverse.setEntry(0, 0, ddata);
  }
  ipiv = new int[nRows_];
  lwork = nRows_ * nRows_;
  work = new double[lwork];
  localMatrix = new double[nRows_ * nRows_];
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nRows_; jj++)
        localMatrix[ii*nRows_+jj] = Mat1D_[jj+ii*nRows_];
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nRows_; jj++)
        localMatrix[ii*nRows_+jj] = Mat2D_[jj][ii];
  }
  dgetrf_(&nRows_, &nRows_, localMatrix, &nRows_, ipiv, &status);
  if (status != 0)
  {
    printf("psMatrix computeInverse ERROR: failed in LU factorization\n");
    delete [] localMatrix;
    return status;
  }
  dgetri_(&nRows_, localMatrix, &nRows_, ipiv, work, &lwork, &status);
  if (status != 0)
  {
    printf("psMatrix computeInverse ERROR: failed in matrix inverse\n");
    delete [] localMatrix;
    delete [] ipiv;
    delete [] work;
    return status;
  }
  inverse.setDim(nRows_, nRows_);
  for (ii = 0; ii < nRows_; ii++)
    for (jj = 0; jj < nRows_; jj++)
      inverse.setEntry(ii, jj, localMatrix[jj*nRows_+ii]);
  delete [] work;
  delete [] ipiv;
  delete [] localMatrix;
  return status;
}

// ************************************************************************
// matrix vector multiply
// ------------------------------------------------------------------------
int psMatrix::matvec(psVector &inVec, psVector &outVec, int transp)
{
  int    ii, jj;
  double ddata, *vdata;

  if (transp == 0)
  {
    assert(inVec.length() == nCols_);
    outVec.setLength(nRows_);
    vdata = inVec.getDVector();
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nCols_; jj++) 
          ddata += Mat1D_[ii+jj*nRows_] * vdata[jj];
        outVec[ii] = ddata;
      }
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nCols_; jj++) 
          ddata += Mat2D_[ii][jj] * vdata[jj];
        outVec[ii] = ddata;
      }
    }
  }
  else
  {
    assert(inVec.length() == nRows_);
    outVec.setLength(nCols_);
    vdata = inVec.getDVector();
    if (format_ == 1)
    {
      for (ii = 0; ii < nCols_; ii++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nRows_; jj++)
          ddata += Mat1D_[jj+ii*nRows_] * vdata[jj];
        outVec[ii] = ddata;
      }
    }
    else
    {
      for (ii = 0; ii < nCols_; ii++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nRows_; jj++)
          ddata += Mat2D_[jj][ii] * vdata[jj];
        outVec[ii] = ddata;
      }
    }
  }
  return 0;
}

// ************************************************************************
// matrix matrix multiply
// ------------------------------------------------------------------------
void psMatrix::matmult(psMatrix &inMat, psMatrix &outMat)
{
  int      ii, jj, kk, ncols, format;
  double   ddata, *vdata, *matB, *matC, **matB2, **matC2;
  psVector colVec;

  assert(inMat.nrows() == nCols_);
  format = inMat.getFormat();
  if (format != format_)
  {
    printf("psMatrix matmult ERROR: C=AB : B has different format from A\n");
    exit(1);
  }
  if (format_ == 1)
  {
    outMat.setFormat(1);
    outMat.setDim(nRows_, inMat.ncols());
    matB = inMat.getMatrix1D();
    matC = outMat.getMatrix1D();
    ncols = inMat.ncols();
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < ncols; jj++)
      {
        ddata = 0.0;
        for (kk = 0; kk < nCols_; kk++)
          ddata += Mat1D_[ii+kk*nRows_] * matB[kk+jj*nCols_];
        matC[ii+jj*nRows_] = ddata;
      }
    }
  }
  else
  {
    outMat.setFormat(2);
    outMat.setDim(nRows_, inMat.ncols());
    matB2 = inMat.getMatrix2D();
    matC2 = outMat.getMatrix2D();
    ncols = inMat.ncols();
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < ncols; jj++)
      {
        ddata = 0.0;
        for (kk = 0; kk < nCols_; kk++)
          ddata += Mat2D_[ii][kk] * matB2[kk][jj];
        matC2[ii][jj] = ddata;
      }
    }
  }
}

// ************************************************************************
// take row sum 
// ------------------------------------------------------------------------
void psMatrix::rowsum(psVector &vecOut)
{
  int    ii, jj;
  double ddata;
  vecOut.setLength(nRows_);
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      ddata = 0.0;
      for (jj = 0; jj < nCols_; jj++)
        ddata += Mat1D_[ii+jj*nRows_];
      vecOut[ii] = ddata;
    }
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      ddata = 0.0;
      for (jj = 0; jj < nCols_; jj++)
        ddata += Mat2D_[ii][jj];
      vecOut[ii] = ddata;
    }
  }
}

// ************************************************************************
// generate random number
// ------------------------------------------------------------------------
void psMatrix::genRandom(int nrows, int ncols, double lb, double ub)
{
  int    ii, jj;
  double width;

  if (ub <= lb)
  {
    printf("psMatrix genRandom ERROR: invalid range [%e,%e].\n",lb,ub);
    exit(1);
  }
  setDim(nrows, ncols);
  width = ub - lb;
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_*nCols_; ii++)
      Mat1D_[ii] = PSUADE_drand() * width + lb;
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++)
        Mat2D_[ii][jj] = PSUADE_drand() * width + lb;
  }
}

// ************************************************************************
// add a vector to all columns of the matrix
// ------------------------------------------------------------------------
void psMatrix::addVec2AllCols(psVector inVec)
{
  int ii, jj;
  assert(inVec.length() == nRows_);
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
        Mat1D_[ii+jj*nRows_] += inVec[ii];
    }
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
        Mat2D_[ii][jj] += inVec[ii];
    }
  }
}

// ************************************************************************
// transform a matrix element wise
// ------------------------------------------------------------------------
void psMatrix::elementOp(char *op)
{
  int      ii, jj;
  double   ddata;

  if (!strcmp(op, "sigmoid"))
  {
    if (format_ == 1)
      for (ii = 0; ii < nRows_*nCols_; ii++)
        Mat1D_[ii] = 1.0 / (1.0 + exp(-Mat1D_[ii]));
    else
      for (ii = 0; ii < nRows_; ii++)
        for (jj = 0; jj < nCols_; jj++)
           Mat2D_[ii][jj] = 1.0 / (1.0 + exp(-Mat2D_[ii][jj]));
  }
  else if (!strcmp(op, "sigmoid_prime"))
  {
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_*nCols_; ii++)
      {
        ddata = 1.0 / (1.0 + exp(-Mat1D_[ii]));
        Mat1D_[ii] = ddata * (1.0 - ddata);
      }
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        for (jj = 0; jj < nCols_; jj++)
        {
           ddata = 1.0 / (1.0 + exp(-Mat2D_[ii][jj]));
           Mat2D_[ii][jj] = ddata * (1.0 - ddata);
        }
      }
    }
  }
  else if (!strcmp(op, "tanh"))
  {
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_*nCols_; ii++)
      {
        ddata = exp(2.0*Mat1D_[ii]);
        Mat1D_[ii] = (ddata - 1.0) / (ddata + 1.0);
      }
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        for (jj = 0; jj < nCols_; jj++)
        {
          ddata = exp(2.0*Mat2D_[ii][jj]);
          Mat2D_[ii][jj] = (ddata - 1.0) / (ddata + 1.0);
        }
      }
    }
  }
  else if (!strcmp(op, "tanh_prime"))
  {
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_*nCols_; ii++)
      {
        ddata = exp(2.0*Mat1D_[ii]);
        ddata = (ddata - 1.0) / (ddata + 1.0);
        Mat1D_[ii] = 1.0 * ddata * ddata;
      }
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        for (jj = 0; jj < nCols_; jj++)
        {
          ddata = exp(2.0*Mat2D_[ii][jj]);
          ddata = (ddata - 1.0) / (ddata + 1.0);
          Mat2D_[ii][jj] = ddata * (1.0 - ddata);
        }
      }
    }
  }
  else if (!strcmp(op, "relu"))
  {
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_*nCols_; ii++)
      {
        ddata = Mat1D_[ii];
        if (ddata <= 0) Mat1D_[ii] = 0;
        else            Mat1D_[ii] = ddata;
      }
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        for (jj = 0; jj < nCols_; jj++)
        {
          ddata = Mat2D_[ii][jj];
          if (ddata <= 0) Mat2D_[ii][jj] = 0;
          else            Mat2D_[ii][jj] = ddata;
        }
      }
    }
  }
  else if (!strcmp(op, "relu_prime"))
  {
    if (format_ == 1)
    {
      for (ii = 0; ii < nRows_*nCols_; ii++)
      {
        ddata = Mat1D_[ii];
        if (ddata <= 0) Mat1D_[ii] = 0;
        else            Mat1D_[ii] = 1;
      }
    }
    else
    {
      for (ii = 0; ii < nRows_; ii++)
      {
        for (jj = 0; jj < nCols_; jj++)
        {
          ddata = Mat2D_[ii][jj];
          if (ddata <= 0) Mat2D_[ii][jj] = 0;
          else            Mat2D_[ii][jj] = 1;
        }
      }
    }
  }
  else
  {
    printf("psMatrix elementOp ERROR: invalid operation (%s).\n",op);  
    exit(1);
  }
}

// ************************************************************************
// transform a matrix element wise
// ------------------------------------------------------------------------
void psMatrix::elementOp(char *op, psMatrix inMat)
{
  int ii, jj;
  if (!strcmp(op, "multiply"))
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++)
        setEntry(ii,jj,getEntry(ii,jj)*inMat.getEntry(ii,jj));
  }
  else if (!strcmp(op, "add"))
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++)
        setEntry(ii,jj,getEntry(ii,jj)+inMat.getEntry(ii,jj));
  }
  else
  {
    printf("psMatrix elementOp ERROR: invalid operation (%s).\n",op);  
    exit(1);
  }
}

// ************************************************************************
// multiply a matrix by a scalar 
// ------------------------------------------------------------------------
void psMatrix::scale(double scale)
{
  int ii, jj;
  if (format_ == 1)
    for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] *= scale;
  else
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] *= scale;
}

// ************************************************************************
// matrix transpose 
// ------------------------------------------------------------------------
void psMatrix::transpose()
{
  int    ii, jj;
  double **tmpMat2, *tmpMat;

  assert(nCols_ > 0 && nRows_ > 0);
  if (format_ == 1)
  {
    tmpMat = new double[nRows_*nCols_];
    assert(tmpMat != NULL);
    for (ii = 0; ii < nCols_; ii++)
      for (jj = 0; jj < nRows_; jj++) 
        tmpMat[ii+jj*nCols_] = Mat1D_[jj+ii*nRows_];
    delete [] Mat1D_;
    Mat1D_ = tmpMat;
  }
  else
  {
    tmpMat2 = new double*[nCols_];
    for (ii = 0; ii < nCols_; ii++)
    {
      tmpMat2[ii] = new double[nRows_];
      assert(tmpMat2[ii] != NULL);
      for (jj = 0; jj < nRows_; jj++) tmpMat2[ii][jj] = Mat2D_[jj][ii];
    }
    for (ii = 0; ii < nRows_; ii++) delete [] Mat2D_[ii];
    delete [] Mat2D_;
    Mat2D_ = tmpMat2;
  }
  ii = nRows_;
  nRows_ = nCols_;
  nCols_ = ii;
}

// ************************************************************************
// eigen solve (flag = 1 ==> compute eigenvalues only) 
// ------------------------------------------------------------------------
void psMatrix::eigenSolve(psMatrix &eigMat, psVector &eigVals, int flag)
{
  int    ii, jj, lwork, N, info;
  double *work, *eigs, *mat;
  char   jobz='V', uplo='U';

  //**/ preparation
  if (flag == 1) jobz = 'N';
  N     = nCols_;
  lwork = 3 * N;
  work  = new double[lwork];
  eigs  = new double[N];

  //**/ copy matrix to mat for format compatibility to dsyev
  mat   = new double[N*N];
  if (format_ == 1) 
  {
    for (ii = 0; ii < N*N; ii++) mat[ii] = Mat1D_[ii];
  }
  else
  {
    for (ii = 0; ii < N; ii++)
      for (jj = 0; jj < N; jj++) mat[ii+jj*N] = Mat2D_[ii][jj];
  }

  //**/ call eigen solver
  dsyev_(&jobz,&uplo,&N,mat,&N,eigs,work,&lwork,&info);
  if (info != 0)
  {
    printf("ERROR: dsyev returns a nonzero (%d).\n", info);
    delete [] mat;
    delete [] eigs;
    delete [] work;
    exit(1);
  }

  //**/ copy results
  eigMat.setDim(N,N);
  for (ii = 0; ii < N; ii++)
    for (jj = 0; jj < N; jj++)
      eigMat.setEntry(ii,jj,mat[ii+jj*N]);
  eigVals.setLength(N);
  for (ii = 0; ii < N; ii++) eigVals[ii] = eigs[ii];

  //**/ clean up
  delete [] mat;
  delete [] eigs;
  delete [] work;
}

// ************************************************************************
// matrix solve using QR 
// ------------------------------------------------------------------------
void psMatrix::matSolve(psVector &invec, psVector &outvec)
{
  int    iOne=1, info, lwork, nn, ii, jj;
  double *work, *b, *x, *dmat;
  char   trans[1];

  nn = invec.length();
  (*trans) = 'N';
  b = invec.getDVector();
  outvec.setLength(nn);
  x = outvec.getDVector();
  for (ii = 0; ii < nn; ii++) x[ii] = b[ii];
  lwork = 2 * nn * nn;
  work = new double[lwork];
  dmat = new double[nn*nn];
  if (format_ == 1)
  {
    for (ii = 0; ii < nn*nn; ii++) dmat[ii] = Mat1D_[ii];
  }
  else
  {
    for (ii = 0; ii < nn; ii++)
      for (jj = 0; jj < nn; jj++) dmat[ii+jj*nn] = Mat2D_[ii][jj];
  }
  dgels_(trans, &nn, &nn, &iOne, dmat, &nn, x, &nn, work, &lwork, &info);
  if (info != 0)
  {
    printf("psMatrix matSolve ERROR: dgels returns error %d.\n",info);
    exit(1);
  }
  delete [] dmat;
  delete [] work;
  return;
}

// ************************************************************************
// matrix SVD (effcient version of SVD)
// ------------------------------------------------------------------------
int psMatrix::computeSVD(psMatrix &MatU, psVector &S, psMatrix &MatV)
{
  int    M, N, mm, nn, wlen, info;
  double *AA, *SS, *UU, *VV, *WW, **array2D;
  char   jobu  = 'S', jobvt = 'A';
  psVector VecW, VecA, VecU, VecV;

  M  = nRows_;
  N  = nCols_;

  //**/ load AA
  VecA.setLength(M*N);
  AA = VecA.getDVector();
  if (format_ == 1)
  {
    for (mm = 0; mm < M*N; mm++) AA[mm] = Mat1D_[mm];
  }
  else
  {
    for (mm = 0; mm < M; mm++)
      for (nn = 0; nn < N; nn++) AA[mm+nn*M] = Mat2D_[mm][nn];
  }

  //**/ allocate temporary storage: SS, UU, VV, WW
  S.setLength(N);
  SS = S.getDVector();
  MatU.setFormat(format_);
  MatU.setDim(M, N);
  MatV.setFormat(format_);
  MatV.setDim(N, N);
  if (format_ == 1)
  {
    UU = MatU.getMatrix1D();
    VV = MatV.getMatrix1D();
  }
  else
  {
    VecU.setLength(M * N);
    UU = VecU.getDVector();
    VecV.setLength(N * N);
    VV = VecV.getDVector();
  }
  wlen = 5 * M;
  VecW.setLength(wlen);
  WW = VecW.getDVector();

  //**/ perform SVD
  dgesvd_(&jobu, &jobvt, &M, &N, AA, &M, SS, UU, &M, VV, &N, WW,
          &wlen, &info);
  if (info != 0)
  {
    for (nn = 0; nn < N; nn++)
      printf("SVD: singular value %3d = %e\n", nn+1, SS[nn]);
  }

  //**/ copy the U, V matrix over
  if (format_ != 1)
  {
    array2D = MatU.getMatrix2D();
    for (mm = 0; mm < M; mm++)
      for (nn = 0; nn < N; nn++)
        array2D[mm][nn] = UU[mm+nn*M];

    array2D = MatV.getMatrix2D();
    for (mm = 0; mm < N; mm++)
      for (nn = 0; nn < N; nn++)
        array2D[mm][nn] = VV[mm+nn*N];
  }
  return info;
}


// ************************************************************************
// form union between two matrices (C = A intersects B)
// ------------------------------------------------------------------------
int psMatrix::intersect(psMatrix matB, psMatrix &matC)
{
  if (format_ != matB.getFormat())
  {
    printf("psMatrix::intersect ERROR: different format.\n");
    exit(1);
  }
  if (nCols_ != matB.ncols())
  {
    printf("psMatrix::intersect ERROR: different number of columns.\n");
    exit(1);
  }
  int nrowsC = nRows_;
  int nrowsB = matB.nrows();
  if (nrowsB < nrowsC) nrowsC = nrowsB; 
  matC.setFormat(format_);
  matC.setDim(nrowsC, nCols_);
  psIVector vecBFlags;
  vecBFlags.setLength(nrowsB);
  int ii, jj, kk;
  double *matB1D, **matB2D, *matC1D, **matC2D;
  if (format_ == PS_MAT1D)
  {
    matB1D = matB.getMatrix1D();
    matC1D = matC.getMatrix1D();
    nrowsC = 0;
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nrowsB; jj++)
      {
        if (vecBFlags[jj] == 0)
        {
          for (kk = 0; kk < nCols_; kk++)
            if (Mat1D_[ii*nCols_+kk] != matB1D[jj*nCols_+kk]) 
              break;
          if (kk == nCols_)
          {
            for (kk = 0; kk < nCols_; kk++)
              matC1D[nrowsC*nCols_+kk] = Mat1D_[ii*nCols_+kk];
            vecBFlags[jj] = 1;
            nrowsC++;
            break;
          }
        }
      }
    }
  }
  else
  {
    matB2D = matB.getMatrix2D();
    matC2D = matC.getMatrix2D();
    nrowsC = 0;
    for (ii = 0; nRows_; ii++)
    {
      for (jj = 0; jj < nrowsB; jj++)
      {
        if (vecBFlags[jj] == 0)
        {
          for (kk = 0; kk < nCols_; kk++)
            if (Mat2D_[ii][kk] != matB2D[jj][kk]) 
              break;
          if (kk == nCols_)
          {
            for (kk = 0; kk < nCols_; kk++)
              matC2D[nrowsC][kk] = Mat2D_[ii][kk];
            vecBFlags[jj] = 1;
            nrowsC++;
            break;
          }
        }
      }
    }
  }
  if (nrowsC < matC.nrows()) matC.removeLastNRows(matC.nrows()-nrowsC);
  return 0;
}
   
// ************************************************************************
// remove a row 
// ------------------------------------------------------------------------
int psMatrix::removeOneRow(int irow)
{
  int ii;
  if (irow < 0 || irow >= nRows_)
  {
    printf("psMatrix removeOneRow ERROR: row not valid\n");
    return -1;
  }
  if (Mat2D_ != NULL)
  {
    if (Mat2D_[irow] != NULL) delete [] Mat2D_[irow];
    for (ii = irow+1; ii < nRows_; ii++)
      Mat2D_[ii-1] = Mat2D_[ii];
    nRows_--;
  }
  if (Mat1D_ != NULL)
  {
    for (ii = (irow+1)*nCols_; ii < nRows_*nCols_; ii++)
      Mat1D_[ii-nCols_] = Mat1D_[ii];
  }
  if (pivots_ != NULL) delete [] pivots_;
  pivots_ = NULL;
  status_ = 0;
  determinant_ = 0;
  return 0;
}

// ************************************************************************
// remove last N rows 
// ------------------------------------------------------------------------
int psMatrix::removeLastNRows(int N)
{
  int ii, M = nRows_- N;
  if (N <= 0)
  {
    printf("psMatrix::removeLastNRows INFO: N <= 0\n");
    return -1;
  }
  if (N > nRows_)
  {
    printf("psMatrix::removeLastNRows ERROR: not enough rows to remove\n");
    exit(1);
  }
  if (M == 0) 
  {
    clean();
    return 0;
  }
  if (format_ == PS_MAT1D)
  {
    double *mat1D = Mat1D_;
    Mat1D_ = new double[M * nCols_];
    for (ii = 0; ii < M*nCols_; ii++) Mat1D_[ii] = mat1D[ii];
    delete [] mat1D;
  }
  else
  {
    double **mat2D = Mat2D_;
    Mat2D_ = new double*[M];
    for (ii = 0; ii < M; ii++) Mat2D_[ii] = mat2D[ii];
    for (ii = M; ii < nRows_; ii++) delete [] mat2D[ii];
    delete [] mat2D;
  }
  nRows_ = M;
  return 0;
}
    
// ************************************************************************
// convert to vector 
// ------------------------------------------------------------------------
int psMatrix::convert2Vector(psVector &vecReturn)
{
  if (Mat1D_ != NULL)
  {
    vecReturn.load(nRows_*nCols_, Mat1D_);
  }
  if (Mat2D_ != NULL)
  {
    vecReturn.setLength(nRows_ * nCols_);
    for (int ii = 0; ii < nRows_; ii++)
      for (int jj = 0; jj < nCols_; jj++)
        vecReturn[ii*nCols_+jj] = Mat2D_[ii][jj];
  }
  return 0;
}

// ************************************************************************
// convert to vector 
// ------------------------------------------------------------------------
void psMatrix::sortMatrix()
{
  if (Mat1D_ != NULL)
  {
    printf("psMatrix ERROR: sortMatrix not available for Mat1D format.\n");
    exit(1);
  }
  psIVector vecIT;
  vecIT.setLength(nRows_);
  sortnDbleList(nRows_, nCols_, Mat2D_, vecIT.getIVector());
}

// ************************************************************************
// clean up
// ------------------------------------------------------------------------
void psMatrix::clean()
{
  if (Mat2D_ != NULL)
  {
    for (int ii = 0; ii < nRows_; ii++)
      if (Mat2D_[ii] != NULL) delete [] Mat2D_[ii];
    delete [] Mat2D_;
    Mat2D_ = NULL;
  }
  if (Mat1D_ != NULL) delete [] Mat1D_;
  if (pivots_ != NULL) delete [] pivots_;
  Mat1D_ = NULL;
  Mat2D_ = NULL;
  pivots_ = NULL;
  nRows_ = nCols_ = 0;
  status_ = 0;
  determinant_ = 0;
}

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psIMatrix::psIMatrix()
{
#ifdef PS_DEBUG
  printf("psIMatrix constructor\n");
#endif
  nRows_ = 0;
  nCols_ = 0;
  Mat2D_ = NULL;
  Mat1D_ = NULL;
  format_ = 1; /* default storage format is 1D, for 2D: Mat2D[row][col] */
#ifdef PS_DEBUG
  printf("psIMatrix constructor ends\n");
#endif
}

// ************************************************************************
// Copy Constructor 
// ------------------------------------------------------------------------
psIMatrix::psIMatrix(const psIMatrix & ma)
{
  int ii, jj;

  nRows_ = ma.nRows_;
  nCols_ = ma.nCols_;
  format_ = ma.format_;
  Mat2D_ = NULL;
  Mat1D_ = NULL;
  if (nRows_ > 0 && nCols_ > 0)
  {
    if (format_ == 1)
    {
      Mat1D_ = new int[nRows_*nCols_];
      assert(Mat1D_ != NULL);
      for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = ma.Mat1D_[ii];
    }
    else
    {
      Mat2D_ = new int*[nRows_];
      assert(Mat2D_ != NULL);
      for (ii = 0; ii < nRows_; ii++)
      {
        Mat2D_[ii] = new int[nCols_];
        assert(Mat2D_[ii] != NULL);
        for(jj = 0; jj < nCols_; jj++)
          Mat2D_[ii][jj] = ma.Mat2D_[ii][jj];
      }
    }
  }
}

// ************************************************************************
// operator=  
// ------------------------------------------------------------------------
psIMatrix & psIMatrix::operator=(const psIMatrix & ma)
{
  int ii, jj;

  if (this == &ma) return *this;
  clean();
  nRows_ = ma.nRows_;
  nCols_ = ma.nCols_;
  format_ = ma.format_;
  if (nRows_ > 0 && nCols_ > 0)
  {
    if (format_ == 1)
    {
      Mat1D_ = new int[nRows_*nCols_];
      assert(Mat1D_ != NULL);
      for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = ma.Mat1D_[ii];
    }
    else
    {
      Mat2D_ = new int*[nRows_];
      assert(Mat2D_ != NULL);
      for(ii = 0; ii < nRows_; ii++)
      {
        Mat2D_[ii] = new int[nCols_];
        assert(Mat2D_[ii] != NULL);
        for(jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] = ma.Mat2D_[ii][jj];
      }
    }
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psIMatrix::~psIMatrix()
{
  clean();
}

// ************************************************************************
// get number of rows 
// ------------------------------------------------------------------------
int psIMatrix::nrows()
{
  return nRows_;
}

// ************************************************************************
// get number of columns 
// ------------------------------------------------------------------------
int psIMatrix::ncols()
{
  return nCols_;
}

// ************************************************************************
// load matrix from another matrix
// ------------------------------------------------------------------------
int psIMatrix::load(psIMatrix &inMat)
{
  int ii, jj;

#ifdef PS_DEBUG
  printf("psIMatrix load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(this != &inMat);
  nRows_  = inMat.nrows();
  nCols_  = inMat.ncols();
  format_ = inMat.format_;
  if (nRows_ > 0 && nCols_ > 0)
  {
    if (format_ == 1)
    {
      Mat1D_ = new int[nRows_*nCols_];
      assert(Mat1D_ != NULL);
      for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = inMat.Mat1D_[ii];
    }
    else
    {
      Mat2D_ = new int*[nRows_];
      assert(Mat2D_ != NULL);
      for (ii = 0; ii < nRows_; ii++)
      {
        Mat2D_[ii] = new int[nCols_];
        assert(Mat2D_[ii] != NULL);
        for (jj = 0; jj < nCols_; jj++) 
          Mat2D_[ii][jj] = inMat.getEntry(ii,jj);
      }
    }
  }
#ifdef PS_DEBUG
  printf("psIMatrix load ends\n");
#endif
  return 0;
}

// ************************************************************************
// load matrix from integers
// ------------------------------------------------------------------------
int psIMatrix::load(int nrows, int ncols, int **mat)
{
  int ii, jj;
#ifdef PS_DEBUG
  printf("psIMatrix load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(nrows);
  assert(ncols);
  assert(mat);
  nRows_ = nrows;
  nCols_ = ncols;
  if (format_ == 1)
  {
    Mat1D_ = new int[nRows_*nCols_];
    assert(Mat1D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++) 
        Mat1D_[ii+jj*nRows_] = mat[ii][jj];
  }
  else
  {
    Mat2D_ = new int*[nRows_];
    assert(Mat2D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
    {
      Mat2D_[ii] = new int[nCols_];
      assert(Mat2D_[ii] != NULL);
      for (jj = 0; jj < nCols_; jj++) 
        Mat2D_[ii][jj] = mat[ii][jj];
    }
  }
#ifdef PS_DEBUG
  printf("psIMatrix load ends\n");
#endif
  return 0;
}

// ************************************************************************
// load matrix from integers
// ------------------------------------------------------------------------
int psIMatrix::load(int nrows, int ncols, int *mat)
{
#ifdef PS_DEBUG
  printf("psIMatrix load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(nrows);
  assert(ncols);
  assert(mat);
  nRows_ = nrows;
  nCols_ = ncols;
  if (format_ == 1)
  {
    Mat1D_ = new int[nrows*ncols];
    assert(Mat1D_ != NULL);
    for (int ii = 0; ii < nrows*ncols; ii++) Mat1D_[ii] = mat[ii];
  }
  else
  {
    Mat2D_ = new int*[nrows];
    for (int ii = 0; ii < nrows; ii++)
    {
      Mat2D_[ii] = new int[ncols];
      assert(Mat2D_[ii] != NULL);
      for (int jj = 0; jj < ncols; jj++)
        Mat2D_[ii][jj] = mat[ii+jj*nrows];
    }
  }
#ifdef PS_DEBUG
  printf("psIMatrix load ends\n");
#endif
  return 0;
}

// ************************************************************************
// set matrix dimension
// ------------------------------------------------------------------------
int psIMatrix::setDim(int nrows, int ncols)
{
  int ii, jj;

  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  nRows_ = nrows;
  nCols_ = ncols;
  if (nRows_ <= 0 || nCols_ <= 0) return -1;
  if (format_ == 1)
  {
    Mat1D_ = new int[nRows_*nCols_];
    assert(Mat1D_ != NULL);
    for (ii = 0; ii < nRows_*nCols_; ii++) Mat1D_[ii] = 0;
  }
  else
  {
    Mat2D_ = new int*[nRows_];
    assert(Mat2D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
    {
      Mat2D_[ii] = new int[nCols_];
      assert(Mat2D_[ii] != NULL);
      for (jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] = 0.0;
    }
  }
  return 0;
}

// ************************************************************************
// set entry
// ------------------------------------------------------------------------
void psIMatrix::setEntry(const int row, const int col, const int idata)
{
  if (row < 0 || row >= nRows_ || col < 0 || col >= nCols_)
  {
    printf("psIMatrix setEntry ERROR: index (%d,%d) out of range (%d,%d)\n",
           row, col, nRows_, nCols_);
    exit(1);
  }
  if (format_ == 1) Mat1D_[row+col*nRows_] = idata;
  else              Mat2D_[row][col] = idata;
}

// ************************************************************************
// set format
// ------------------------------------------------------------------------
int psIMatrix::setFormat(int format)
{
  if (format != 1 && format != 2)
  {
    printf("psIMatrix setFormat ERROR: invalid format (should be 1 or 2)\n");
    exit(1);
  }
  format_ = format;
  clean();
  return 0;
}

// ************************************************************************
// set format
// ------------------------------------------------------------------------
int psIMatrix::getFormat()
{
  return format_;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
int psIMatrix::getEntry(const int row, const int col)
{
  if (row < 0 || row >= nRows_ || col < 0 || col >= nCols_)
  {
    printf("psIMatrix getEntry ERROR: index (%d,%d) out of range (%d,%d)\n",
           row, col, nRows_, nCols_);
    exit(1);
  }
  if (format_ == 1) return Mat1D_[row+col*nRows_];
  else              return Mat2D_[row][col];
}

// ************************************************************************
// get matrix in 1 dimensional array
// ------------------------------------------------------------------------
void psIMatrix::getIMatrix1D(psIVector &mat)
{
  int ii, jj;
#ifdef PS_DEBUG
  printf("psIMatrix getIMatrix1D\n");
#endif
  assert(nRows_ >= 0);
  assert(nCols_ >= 0);
  mat.setLength(nRows_ * nCols_);
  if (format_ == 1)
  {
    for (ii = 0; ii < nRows_*nCols_; ii++) mat[ii] = Mat1D_[ii];
  }
  else
  {
    for (ii = 0; ii < nRows_; ii++)
      for (jj = 0; jj < nCols_; jj++)
        mat[ii+jj*nRows_] = Mat2D_[ii][jj];
  }
#ifdef PS_DEBUG
  printf("psIMatrix getIMatrix1D ends\n");
#endif
  return;
}

// ************************************************************************
// get matrix 
// ------------------------------------------------------------------------
int *psIMatrix::getIMatrix1D()
{
  if (format_ == 1) return Mat1D_;
  else
  {
    printf("psIMatrix getIMatrix1D ERROR: internal format is 2D.\n");
    exit(1);
  }
}

// ************************************************************************
// get matrix 
// ------------------------------------------------------------------------
int **psIMatrix::getIMatrix2D()
{
  if (format_ == 1)
  {
    printf("psIMatrix getIMatrix2D ERROR: internal format is 1D.\n");
    exit(1);
  }
  return Mat2D_;
}

// ************************************************************************
// take matrix 
// ------------------------------------------------------------------------
int **psIMatrix::takeIMatrix2D()
{
  if (format_ == 1)
  {
    printf("psIMatrix takeIMatrix2D ERROR: internal format is 1D.\n");
    exit(1);
  }
  int **tmpmat = Mat2D_;
  Mat2D_ = NULL;
  nRows_ = 0;
  nCols_ = 0;
  return tmpmat;
}

// ************************************************************************
// print matrix
// ------------------------------------------------------------------------
void psIMatrix::print()
{
  printf("psIMatrix print (%d,%d): \n",nRows_,nCols_);
  for (int ii = 0; ii < nRows_; ii++)
  {
    for (int jj = 0; jj < nCols_; jj++) printf("%d ", getEntry(ii,jj));
    printf("\n");
  }
}

// ************************************************************************
// clean up
// ------------------------------------------------------------------------
void psIMatrix::clean()
{
  if (Mat2D_ != NULL)
  {
    for (int ii = 0; ii < nRows_; ii++)
      if (Mat2D_[ii] != NULL) delete [] Mat2D_[ii];
  }
  if (Mat1D_ != NULL) delete [] Mat1D_;
  nRows_ = nCols_ = 0;
  Mat1D_ = NULL;
  Mat2D_ = NULL;
}


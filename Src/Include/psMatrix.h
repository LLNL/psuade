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
// psMatrix definition
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#ifndef __MATRIXH__
#define __MATRIXH__
#define PS_MAT1D 1
#define PS_MAT2D 2

/**
 * @name psMatrix class
 *
 **/
/*@{*/

#include "psVector.h"

// ************************************************************************
// class definition
// ************************************************************************

class psMatrix
{
  int    format_;
  int    nRows_, nCols_, status_;
  double **Mat2D_, determinant_;
  double *Mat1D_;

public:
  int *pivots_;

public:

  psMatrix();
  // Copy constructor by Bill Oliver
  psMatrix(const psMatrix & ma);
  // overload = operator
  psMatrix & operator=(const psMatrix & ma);
  ~psMatrix();
  int    nrows();
  int    ncols();
  int    load(psMatrix &);
  int    load(psVector &, int, int, char *);
  int    load(int, int, double **);
  int    load(int, int, double *);
  int    loadRow(int, int, double *);
  int    loadCol(int, int, double *);
  int    setFormat(int);
  int    setDim(int, int);
  void   setEntry(const int, const int, const double);
  double getEntry(const int, const int);
  int    getFormat();
  double getDeterminant();
  void   getMatrix1D(psVector &);
  double *getMatrix1D();
  double **getMatrix2D();
  double *takeMatrix1D();
  double **takeMatrix2D();
  void   getCol(const int col, psVector &vec);
  int    submatrix(psMatrix &, const int, const int);
  int    submatrix(psMatrix &, const int, const int *);
  int    CholDecompose();
  int    CholLMatvec(psVector &, psVector &);
  int    CholSolve(psVector &, psVector &);
  int    CholLSolve(psVector &, psVector &);
  int    CholLTSolve(psVector &, psVector &);
  int    LUDecompose();
  int    LUSolve(psVector &, psVector &);
  void   print();
  int    computeInverse(psMatrix &);
  double computeDeterminant();
  int    matvec(psVector &, psVector &, int);
  void   matmult(psMatrix &, psMatrix &);
  void   rowsum(psVector &);
  void   genRandom(int, int, double lb, double ub);
  void   addVec2AllCols(psVector);
  void   elementOp(char *);
  void   elementOp(char *, psMatrix);
  void   scale(double);
  void   transpose();
  void   eigenSolve(psMatrix &, psVector &, int);
  void   matSolve(psVector &, psVector &);
  int    computeSVD(psMatrix &, psVector &, psMatrix &);
  int    intersect(psMatrix, psMatrix &);
  int    removeOneRow(int);
  int    removeLastNRows(int);
  int    convert2Vector(psVector &);
  void   sortMatrix();
  void   clean();

private:
  double computeDeterminant(int, double **);
};

// ------------------------------------------------------------------------
// integer matrix
// ------------------------------------------------------------------------

class psIMatrix
{
  int format_;
  int nRows_, nCols_;
  int **Mat2D_;
  int *Mat1D_;

public:

  psIMatrix();
  // Copy constructor by Bill Oliver
  psIMatrix(const psIMatrix & ma);
  // overload = operator
  psIMatrix & operator=(const psIMatrix & ma);
  ~psIMatrix();
  int  nrows();
  int  ncols();
  int  load(psIMatrix &);
  int  load(int, int, int **);
  int  load(int, int, int *);
  int  setFormat(int);
  int  setDim(int, int);
  void setEntry(const int, const int, const int);
  int  getEntry(const int, const int);
  int  getFormat();
  void getIMatrix1D(psIVector &);
  int  *getIMatrix1D();
  int  **getIMatrix2D();
  int  **takeIMatrix2D();
  void print();
  void   clean();
};

/*@}*/

#endif /* __MATRIXH__ */


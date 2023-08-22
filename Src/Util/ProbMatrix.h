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
// ProbMatrix functions
// AUTHOR : CHARLES TONG
// DATE   : 2019
// ************************************************************************
#ifndef __PROBMATRIXH__
#define __PROBMATRIXH__

/**
 * @name ProbMatrix class
 *
 **/
/*@{*/

#include "psVector.h"

// ************************************************************************
// class definition
// ************************************************************************

class ProbMatrix
{
  int    nRows_, nCols_, *counts_;
  double **Mat2D_;

public:

  ProbMatrix();
  // Copy constructor 
  ProbMatrix(const ProbMatrix & ma);
  // overload = operator
  ProbMatrix & operator=(const ProbMatrix & ma);
  ~ProbMatrix();
  int    nrows();
  int    ncols();
  int    load(ProbMatrix &);
  int    load(int, int, double **);
  int    load(int, int, double *);
  int    load(int, int, double **, int *);
  int    setDim(int, int);
  double **getMatrix2D();
  double getEntry(const int, const int);
  void   setEntry(const int, const int, const double);
  void   print();
  int    *getCounts();
  int    getCount(int);
  int    setCount(int, int);
  int    compress();
  int    convert2Hist(int, psVector, psVector);
  int    binAdaptive(int, psVector, psVector);
  int    multiply(ProbMatrix &, ProbMatrix &);
  int    multiply3(ProbMatrix &, ProbMatrix &, ProbMatrix &);
  void   clean();
};

/*@}*/

#endif /* __PROBMATRIXH__ */


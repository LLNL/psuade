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
// ProbMatrix Class: special matrix type for probability distribution 
//                   handling (binning)
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "ProbMatrix.h"
#include "Psuade.h"

//#define PS_DEBUG 1
// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
ProbMatrix::ProbMatrix()
{
  nRows_ = 0;
  nCols_ = 0;
  Mat2D_ = NULL;
  counts_ = NULL;
}

// ************************************************************************
// Copy Constructor 
// ------------------------------------------------------------------------
ProbMatrix::ProbMatrix(const ProbMatrix & ma)
{
  int ii, jj;

  nRows_ = ma.nRows_;
  nCols_ = ma.nCols_;
  Mat2D_ = NULL;
  counts_ = NULL;
  if (nRows_ > 0 && nCols_ > 0)
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
    if (ma.counts_ != NULL)
    {
      counts_ = new int[nRows_];
      assert(counts_ != NULL);
      for (ii = 0; ii < nRows_; ii++) counts_[ii] = ma.counts_[ii];
    } 
  }
}

// ************************************************************************
// operator=  
// ------------------------------------------------------------------------
ProbMatrix & ProbMatrix::operator=(const ProbMatrix & ma)
{
  int ii, jj;

  if (this == &ma) return *this;
  clean();
  nRows_ = ma.nRows_;
  nCols_ = ma.nCols_;
  if (nRows_ > 0 && nCols_ > 0)
  {
    Mat2D_ = new double*[nRows_];
    assert(Mat2D_ != NULL);
    for(ii = 0; ii < nRows_; ii++)
    {
      Mat2D_[ii] = new double[nCols_];
      assert(Mat2D_[ii] != NULL);
      for(jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] = ma.Mat2D_[ii][jj];
    }
    if (ma.counts_ != NULL)
    {
      counts_ = new int[nRows_];
      assert(counts_ != NULL);
      for (ii = 0; ii < nRows_; ii++) counts_[ii] = ma.counts_[ii];
    } 
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
ProbMatrix::~ProbMatrix()
{
  clean();
}

// ************************************************************************
// get number of rows 
// ------------------------------------------------------------------------
int ProbMatrix::nrows()
{
  return nRows_;
}

// ************************************************************************
// get number of columns 
// ------------------------------------------------------------------------
int ProbMatrix::ncols()
{
  return nCols_;
}

// ************************************************************************
// load matrix from another matrix
// ------------------------------------------------------------------------
int ProbMatrix::load(ProbMatrix &inMat)
{
  int    ii, jj, *inCnts;
  double **matIn;

  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(this != &inMat);
  nRows_  = inMat.nrows();
  nCols_  = inMat.ncols();
  if (nRows_ > 0 && nCols_ > 0)
  {
    inCnts = inMat.getCounts();
    matIn  = inMat.getMatrix2D();
    Mat2D_ = new double*[nRows_];
    assert(Mat2D_ != NULL);
    for (ii = 0; ii < nRows_; ii++)
    {
      Mat2D_[ii] = new double[nCols_];
      assert(Mat2D_[ii] != NULL);
      for (jj = 0; jj < nCols_; jj++) 
        Mat2D_[ii][jj] = matIn[ii][jj];
    }
    if (inMat.counts_ != NULL)
    {
      counts_ = new int[nRows_];
      assert(counts_ != NULL);
      for (ii = 0; ii < nRows_; ii++) 
        counts_[ii] = inCnts[ii];
    } 
  }
  sortnDbleList(nRows_, nCols_, Mat2D_, counts_);
  compress();
  return 0;
}

// ************************************************************************
// load matrix 
// ------------------------------------------------------------------------
int ProbMatrix::load(int nrows, int ncols, double **mat)
{
  int ii, jj;

  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(nrows);
  assert(ncols);
  assert(mat);
  nRows_ = nrows;
  nCols_ = ncols;
  Mat2D_ = new double*[nRows_];
  assert(Mat2D_ != NULL);
  for (ii = 0; ii < nRows_; ii++)
  {
    Mat2D_[ii] = new double[nCols_];
    assert(Mat2D_[ii] != NULL);
    for (jj = 0; jj < nCols_; jj++) 
      Mat2D_[ii][jj] = mat[ii][jj];
  }
  counts_ = new int[nRows_];
  assert(counts_ != NULL);
  for (ii = 0; ii < nRows_; ii++) counts_[ii] = 1;
  sortnDbleList(nRows_, nCols_, Mat2D_, counts_);
  compress();
  return 0;
}

// ************************************************************************
// load matrix 
// ------------------------------------------------------------------------
int ProbMatrix::load(int nrows, int ncols, double *mat)
{
  int ii, jj;

  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(nrows);
  assert(ncols);
  assert(mat);
  nRows_ = nrows;
  nCols_ = ncols;
  Mat2D_ = new double*[nRows_];
  assert(Mat2D_ != NULL);
  for (ii = 0; ii < nRows_; ii++)
  {
    Mat2D_[ii] = new double[nCols_];
    assert(Mat2D_[ii] != NULL);
    for (jj = 0; jj < nCols_; jj++) 
      Mat2D_[ii][jj] = mat[ii+nRows_*jj];
  }
  counts_ = new int[nRows_];
  assert(counts_ != NULL);
  for (ii = 0; ii < nRows_; ii++) counts_[ii] = 1;
  sortnDbleList(nRows_, nCols_, Mat2D_, counts_);
  compress();
  return 0;
}

// ************************************************************************
// load matrix 
// ------------------------------------------------------------------------
int ProbMatrix::load(int nrows, int ncols, double **mat, int *counts)
{
  int ii, jj;

  //**/ load the matrix (and not counts)
  load(nrows, ncols, mat);

  //**/ load counts 
  assert(counts);
  counts_ = new int[nRows_];
  assert(counts_ != NULL);
  for (ii = 0; ii < nRows_; ii++) counts_[ii] = counts[ii];
  sortnDbleList(nRows_, nCols_, Mat2D_, counts_);
  compress();
  return 0;
}

// ************************************************************************
// set matrix dimension
// ------------------------------------------------------------------------
int ProbMatrix::setDim(int nrows, int ncols)
{
  int ii, jj;

  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  nRows_ = nrows;
  nCols_ = ncols;
  if (nRows_ <= 0 || nCols_ <= 0) return -1;
  Mat2D_ = new double*[nRows_];
  assert(Mat2D_ != NULL);
  for (ii = 0; ii < nRows_; ii++)
  {
    Mat2D_[ii] = new double[nCols_];
    assert(Mat2D_[ii] != NULL);
    for (jj = 0; jj < nCols_; jj++) Mat2D_[ii][jj] = 0.0;
  }
  counts_ = new int[nRows_];
  for (ii = 0; ii < nRows_; ii++) counts_[ii] = 0.0;
  return 0;
}

// ************************************************************************
// get entry 
// ------------------------------------------------------------------------
double ProbMatrix::getEntry(const int row, const int col)
{
  if (row < 0 || row >= nRows_ || col < 0 || col >= nCols_)
  {
    printf("ProbMatrix getEntry ERROR: nrows = %d (%d), ncols = %d (%d)\n",
           row+1, nRows_, col+1, nCols_);
    exit(1);
  }
  return Mat2D_[row][col];
}

// ************************************************************************
// set entry 
// ------------------------------------------------------------------------
void ProbMatrix::setEntry(const int row, const int col, const double ddata)
{
  if (row < 0 || row >= nRows_ || col < 0 || col >= nCols_)
  {
    printf("ProbMatrix setEntry ERROR: nrows = %d (%d), ncols = %d (%d)\n",
           row+1, nRows_, col+1, nCols_);
    exit(1);
  }
  Mat2D_[row][col] = ddata;
}

// ************************************************************************
// get matrix 
// ------------------------------------------------------------------------
double **ProbMatrix::getMatrix2D()
{
  return Mat2D_;
}

// ************************************************************************
// get counts for all bins (after binUniform or binAdaptive)
// ------------------------------------------------------------------------
int *ProbMatrix::getCounts()
{
  return counts_;
}

// ************************************************************************
// get counts for individual bin (after binUniform or binAdaptive)
// ------------------------------------------------------------------------
int ProbMatrix::getCount(int ii)
{
  if (ii < 0 || ii >= nRows_)
  {
    printf("ProbMatrix getCount ERROR: Index %d not in range (%d,%d)\n",
           ii, 0, nRows_-1);
    exit(1);
  }
  return counts_[ii];
}

// ************************************************************************
// set count 
// ------------------------------------------------------------------------
int ProbMatrix::setCount(int ii, int ival)
{
  if (ii < 0 || ii >= nRows_)
  {
    printf("ProbMatrix setCount ERROR: index %d not in range (%d,%d)\n",
           ii, 0, nRows_-1);
    exit(1);
  }
  counts_[ii] = ival;
  return 0;
}

// ************************************************************************
// print matrix
// ------------------------------------------------------------------------
void ProbMatrix::print()
{
  printf("ProbMatrix::print (%d,%d): \n",nRows_,nCols_);
  for (int ii = 0; ii < nRows_; ii++)
  {
    printf("%7d: ", ii+1);
    for (int jj = 0; jj < nCols_; jj++) printf("%e ", Mat2D_[ii][jj]);
    printf(" - count = %d\n", counts_[ii]);
  }
}

// ************************************************************************
// compress matrix (remove entries with zero counts ==> less rows)
// ------------------------------------------------------------------------
int ProbMatrix::compress()
{
  int ii, jj, kk;
  //**/ this part assumes that counts[ii] is either 0 or 1
  for (ii = 0; ii < nRows_; ii++)
  {
    if (counts_[ii] > 0)
    {
      for (jj = ii+1; jj < nRows_; jj++)
      {
        if (counts_[jj] > 0)
        {
          for (kk = 0; kk < nCols_; kk++)
            if (Mat2D_[ii][kk] != Mat2D_[jj][kk]) break;
          if (kk == nCols_) 
          {
            counts_[ii]++;
            counts_[jj] = 0;
          }
          else break;
        }
      }
    }
  }
  kk = 0;
  for (ii = 0; ii < nRows_; ii++) if (counts_[ii] > 0) kk++;
  int    *tmpCounts  = new int[kk];
  double **tmpMatrix = new double*[kk];
  kk = 0;
  for (ii = 0; ii < nRows_; ii++) 
  {
    if (counts_[ii] > 0)
    {
      tmpCounts[kk] = counts_[ii];
      tmpMatrix[kk] = Mat2D_[ii];
      kk++;
    }
    else delete [] Mat2D_[ii];
  }
  nRows_ = kk;
  delete [] counts_;
  delete [] Mat2D_;
  Mat2D_ = tmpMatrix;
  counts_ = tmpCounts;
  return 0;
} 

// ************************************************************************
// Construct bins using vector values (uniform bin width version)
// Divide the vector into nLevels bins each with the same width.
// As a result, the number of values for each bin is different from
// the others
// ------------------------------------------------------------------------
//**/ E.g. If the current matrix is Nxn, this function creates a
//**/      ProbMatrix of nLevels^n rows (where n is the number of
//**/      columns of A) with each row corresponding to the coordinate
//**/      the center of the nLevels^n boxes
// ------------------------------------------------------------------------
int ProbMatrix::binUniform(int nLevels, psVector veclbs, psVector vecubs)
{
  int    ii, jj, kk, totCnt;
  double dub, dlb, dstep;
  psIVector vecCnts;
  ProbMatrix matPT;

  //**/ -----------------------------------------------------------
  //**/ error checking
  //**/ -----------------------------------------------------------
  if (veclbs.length() < nCols_ || vecubs.length() < nCols_)
  {
    printf("ProbMatrix binUniform ERROR: Invalid incoming bounds\n");
    printf("    INFO: Incoming lower bound array length = %d\n",
           veclbs.length());
    printf("    INFO: Incoming upper bound array length = %d\n",
           vecubs.length());
    printf("ProbMatrix INFO: binUniform returns a status of -1.\n");
    return -1;
  }
  for (jj = 0; jj < nCols_; jj++) 
  {
    if (veclbs[jj] >= vecubs[jj])
    {
      printf("ProbMatrix binUniform ERROR: LBound[%d] >= UBound[%d]\n",
             jj+1,jj+1);
      printf("ProbMatrix INFO: binUniform returns a status of -1.\n");
      return -1;
    }
  }
    
  //**/ -----------------------------------------------------------
  //**/ allocate temporary storage and counters: matPT, vecCnts
  //**/ e.g. if nLevels=10, and the number of columns=3, then
  //**/      matPT will be of dimension [1000,3].
  //**/ -----------------------------------------------------------
  totCnt = 1;
  for (jj = 0; jj < nCols_; jj++) totCnt *= nLevels;
  matPT.setDim(totCnt, nCols_);
  int    *ptCnts = matPT.getCounts();
  double **ptMat = matPT.getMatrix2D();
  vecCnts.setLength(nCols_);

  //**/ -----------------------------------------------------------
  //**/ binning (if nLevels=10 and ncol=3), do 1000 times:
  //**/ For each row i and column j (M(i,j)), 
  //**/    Find which bin M(i,j) belongs to in the range of col j
  //**/ -----------------------------------------------------------
  totCnt = 0;
  while (vecCnts[nCols_-1] < nLevels)
  {
    //**/ scan the whole matrix and accumulate bins (also set 
    //**/ the matrix element to be bin center)
    for (ii = 0; ii < nRows_; ii++)
    {
      for (jj = 0; jj < nCols_; jj++)
      {
        //**/ bin width for column j
        dstep = (vecubs[jj] - veclbs[jj]) / nLevels;
        //**/ lower and upper bounds for current bin for column j
        dlb = dstep * vecCnts[jj] + veclbs[jj];
        dub = dstep * (vecCnts[jj] + 1) + veclbs[jj];
        //**/ set ptMat to be the center of the bin
        if (ii == 0) ptMat[totCnt][jj] = 0.5 * (dlb + dub);
        if (Mat2D_[ii][jj] < veclbs[jj] || Mat2D_[ii][jj] > vecubs[jj]) 
        {
          if (psConfig_.DiagnosticsIsOn())
          {
            printf("ProbMatrix INFO: binUniform did not succeed.\n");
            printf("                 Matrix entry %d %d = %e not in range\n",
                   ii+1,jj+1,Mat2D_[ii][jj]);
            printf("                 Bounds = [%e %e]\n",veclbs[jj],
                   vecubs[jj]);
            printf("ProbMatrix INFO: binUniform returns a status of -1.\n");
          }
          return -1;
        }
        //**/ if current point (ii,jj) does not belong to the 
        //**/ current bin for column j, skip
        if (Mat2D_[ii][jj] < dlb || Mat2D_[ii][jj] > dub) break;
      }
      if (jj == nCols_) ptCnts[totCnt]++;
    }
    totCnt++;

    //**/ check termination 
    vecCnts[0]++;
    ii = 0;
    while (vecCnts[ii] >= nLevels && ii < nCols_-1)
    {
      vecCnts[ii] = 0;
      ii++;
      vecCnts[ii]++;
    }
  }
  //printf("ProbMatrix binUniform total number of bins = %d\n",totCnt);
  //printf("ProbMatrix binUniform total count before   = %d\n",nRows_);
  //jj = 0;
  //for (ii = 0; ii < totCnt; ii++) jj += ptCnts[ii];
  //printf("ProbMatrix binUniform total count after    = %d\n",jj);

  //**/ -----------------------------------------------------------
  //**/ check histogram
  //**/ -----------------------------------------------------------
  int actualCnt = 0;
  for (ii = 0; ii < totCnt; ii++) if (ptCnts[ii] > 0) actualCnt++;
  //printf("ProbMatrix binUniform number of occupied bins = %d\n",
  //       actualCnt);
  if (actualCnt == 0)
  {
    printf("ProbMatrix binUniform ERROR: binning problem\n");
    printf("    Something is wrong. Consult PSUADE developers.\n");
    printf("ProbMatrix INFO: binUniform returns a status of -1.\n");
    return -1;
  }

  //**/ -----------------------------------------------------------
  //**/ allocate storage for histogram
  //**/ -----------------------------------------------------------
  clean();
  nRows_  = nLevels;
  nCols_  = matPT.ncols();
  Mat2D_  = new double*[nRows_];
  counts_ = new int[nRows_];
  assert(Mat2D_ != NULL);
  for (ii = 0; ii < nRows_; ii++)
  {
    Mat2D_[ii] = new double[nCols_];
    assert(Mat2D_[ii] != NULL);
  }

  //**/ -----------------------------------------------------------
  //**/ compress ==> Mat2D_
  //**/ 2023 : no compression (keep the zeros)
  //**/ counts_[kk] - number of points in box kk (n-dimensional)
  //**/ -----------------------------------------------------------
  kk = 0;
  for (ii = 0; ii < totCnt; ii++)
  {
    //if (ptCnts[ii] > 0)
    //{
      for(jj = 0; jj < nCols_; jj++) Mat2D_[kk][jj] = ptMat[ii][jj];
      counts_[kk] = ptCnts[ii];
      kk++;
    //}
  }
  return 0;
} 

// ************************************************************************
// Construct bins using vector values (adaptive version)
// Divide the vector into nLevels bins each with approximately the same
// number of elements. As a result, the width of each bin is different from
// the others. This function works for ncol=1 only.
// ------------------------------------------------------------------------
int ProbMatrix::binAdaptive(int nLevels, psVector veclbs, psVector vecubs)
{
  //**/ -----------------------------------------------------------
  //**/ error checking
  //**/ -----------------------------------------------------------
  if (veclbs.length() != 1 || vecubs.length() != 1)
  {
    printf("ProbMatrix binAdaptive ERROR: Invalid bound lengths.\n");
    printf("    Incoming lower bound array length = %d\n",
           veclbs.length());
    printf("    Incoming upper bound array length = %d\n",
           vecubs.length());
    printf("ProbMatrix INFO: binAdaptive returns a status of -1.\n");
    return -1;
  }
  if (veclbs[0] >= vecubs[0])
  {
    printf("ProbMatrix binAdaptive ERROR: LBound (%e) >= UBound (%e)\n",
           veclbs[0],vecubs[0]);
    printf("ProbMatrix INFO: binAdaptive returns a status of -1.\n");
    return -1;
  }
    
  //**/ -----------------------------------------------------------
  //**/ first need to sort the incoming array
  //**/ -----------------------------------------------------------
  int    ii;
  double *vec = new double[nRows_];
  for (ii = 0; ii < nRows_; ii++) vec[ii] = Mat2D_[ii][0];
  if (psConfig_.DiagnosticsIsOn())
  {
    printf("ProbMatrix binAdaptive: Sorting\n");
    printf("NOTE: It may fail here if sample size is too large.\n");
  }
  //**/ quicksort cannot handle many equal values (too many recursion)
  //sortDbleList(nRows_, vec);
  //**/ bubble sort is slow
  //sortDbleListBubble(nRows_, vec);
  sortDbleListMerge(nRows_, vec);
  if (psConfig_.DiagnosticsIsOn())
    printf("ProbMatrix binAdaptive: Sorting completed\n");

  //**/ -----------------------------------------------------------
  //**/ prepare for binning
  //**/ -----------------------------------------------------------
  psVector vecLs, vecUs;
  vecLs.setLength(nLevels);
  vecUs.setLength(nLevels);
  int aveLeng = nRows_ / nLevels;
  if ((nRows_ - aveLeng * nLevels) > 0.5 * nLevels) aveLeng++;
  if (counts_ != NULL) delete [] counts_;
  counts_ = new int[nLevels];

  //**/ -----------------------------------------------------------
  //**/ scan the whole vector and count the number of elements
  //**/ and also find the lower and upper bounds of each bin
  //**/ which is the mid point between 2 extreme points in the
  //**/ adjacent partition
  //**/ -----------------------------------------------------------
  int hInd=0, isum=0, kk;
  vecLs[0] = veclbs[0];
  vecUs[nLevels-1] = vecubs[0];
  for (ii = aveLeng; ii < nRows_; ii+=aveLeng)
  {
    if (hInd < nLevels-1)
    {
      vecUs[hInd] = 0.5 * (vec[ii-1] + vec[ii]);
      if (vecUs[hInd] == vecLs[hInd] && (ii+1) >= nRows_)
      {
        if (psConfig_.DiagnosticsIsOn())
        {
          printf("ProbMatrix INFO: binAdaptive did not succeed (LB = UB).\n");
          printf("      Offending bin = %d (of %d)\n",hInd,nLevels);
          for (kk = 0; kk <= hInd; kk++)
            printf("      Bin %d bounds = [%10.3e, %10.3e]\n",kk+1,
                   vecLs[kk],vecUs[kk]);
          printf("ProbMatrix INFO: binAdaptive returns a status of -1.\n");
        }
        return -1;
      }
      //**/ look in the next 10 elements
      kk = ii + 1;
      while (vecUs[hInd] == vecLs[hInd] && kk < ii+5 && kk < nRows_)
      {
        vecUs[hInd] = 0.5 * (vec[ii-1] + vec[kk]);
        kk++;
      }
      if (vecUs[hInd] == vecLs[hInd])
      {
        if (psConfig_.DiagnosticsIsOn())
        {
          printf("ProbMatrix binAdaptive ERROR: LB = UB\n");
          printf("ProbMatrix INFO: binAdaptive returns a status of -1.\n");
        }
        return -1;
      }
    }  
    if (hInd < nLevels-1) vecLs[hInd+1] = vecUs[hInd];
    counts_[hInd] = aveLeng;
    isum += aveLeng;
    hInd++;
  }
  counts_[nLevels-1] = nRows_ - isum;

  //**/ -----------------------------------------------------------
  //**/ allocate storage for histogram
  //**/ -----------------------------------------------------------
  delete [] vec;
  if (Mat2D_ != NULL)
  {
    for (int ii = 0; ii < nRows_; ii++)
      if (Mat2D_[ii] != NULL) delete [] Mat2D_[ii];
    delete [] Mat2D_;
  }
  nRows_  = nLevels;
  Mat2D_  = new double*[nRows_];
  assert(Mat2D_ != NULL);
  for (ii = 0; ii < nRows_; ii++)
  {
    Mat2D_[ii] = new double[nCols_];
    assert(Mat2D_[ii] != NULL);
  }

  //**/ -----------------------------------------------------------
  //**/ store the bin widths in the matrix
  //**/ -----------------------------------------------------------
  for (ii = 0; ii < nLevels; ii++)
  {
    Mat2D_[ii][0] = vecUs[ii] - vecLs[ii];
    if (Mat2D_[ii][0] <= 0)
    {
      if (psConfig_.DiagnosticsIsOn())
      {
        printf("ProbMatrix INFO: binAdaptive did not succeed.\n");
        printf("           Bin %d with <= 0\n",ii+1);
        for (int jj = 0; jj < nLevels; jj++)
          printf("         Bin %d range = [%e,%e]\n",jj+1,
                 vecLs[jj],vecUs[jj]);
        printf("ProbMatrix INFO: Maybe due to no variation.\n");
        printf("ProbMatrix INFO: binAdaptive returns a status of -1.\n");
      }
      return -1;
    }
  }
  return 0;
} 

// ************************************************************************
// multiply 2 probability matrix (C = A * B)
// ------------------------------------------------------------------------
int ProbMatrix::multiply(ProbMatrix &matB, ProbMatrix &matC)
{
  //**/ -----------------------------------------------------------
  //**/ error checking
  //**/ -----------------------------------------------------------
  if (nCols_ != matB.ncols())
  {
    printf("ProbMatrix multiply ERROR: Different number of columns.\n");
    printf("      INFO: %d (local) versus %d (incoming)\n",nCols_, 
           matB.ncols());
    exit(1);
  }

  //**/ -----------------------------------------------------------
  //**/ extract pointers and allocate space 
  //**/ -----------------------------------------------------------
  int ii, jj, kk, index;
  int nrowsB = matB.nrows();
  int nrowsC = nRows_ + nrowsB;
  double **Bmat = matB.getMatrix2D();
  int    *cntsB = matB.getCounts();
  double **Cmat = new double*[nrowsC];
  int    *cntsC = new int[nrowsC];

#if 1
  //**/ perform multiplication
  index = 0;
  int irowA = 0, irowB = 0;
  while (irowA < nRows_ && irowB < nrowsB)
  {
    for (ii = 0; ii < nCols_; ii++)
    {
      if (Mat2D_[irowA][ii] < Bmat[irowB][ii]) 
      {
        irowA++;
        break;
      }
      else if (Mat2D_[irowA][ii] > Bmat[irowB][ii]) 
      {
        irowB++;
        break;
      }
    }
    //**/ match
    if (ii == nCols_)
    {
      Cmat[index] = new double[nCols_];
      for (ii = 0; ii < nCols_; ii++)
        Cmat[index][ii] = Mat2D_[irowA][ii];
      cntsC[index] = counts_[irowA] * cntsB[irowB];
      index++;
      irowA++;
      irowB++;
      //**/ if more space is needed
      if (index > nrowsC)
      {
        printf("ProbMatrix multiply ERROR: Something wrong.\n");
        exit(1);
      }
    }
  }
#else
  //**/ perform multiplication
  index = 0;
  for (ii = 0; ii < nRows_; ii++)
  {
    for (jj = 0; jj < nrowsB; jj++)
    {
      //**/ compare row ii of A with row jj of B
      for (kk = 0; kk < nCols_; kk++)
        if (Mat2D_[ii][kk] != Bmat[jj][kk]) 
          break;
      //**/ if they are equal, copy to C and register counts
      if (kk == nCols_)
      {
        Cmat[index] = new double[nCols_];
        for (kk = 0; kk < nCols_; kk++)
          Cmat[index][kk] = Mat2D_[ii][kk];
        cntsC[index] = counts_[ii] * cntsB[jj];
        index++;
        //**/ if more space is needed
        if (index > nrowsC)
        {
          printf("ProbMatrix multiply ERROR: something wrong.\n");
          exit(1);
        }
        break;
      }
    }
  }
#endif
  if (index == 0)
  {
    if (psConfig_.DiagnosticsIsOn())
    {
      printf("ProbMatrix multiply WARNING: matrix product = 0.\n");
      printf("    ==> no overlap in 2 probability distributions.\n");
      printf("ProbMatrix INFO: multiply returns a status of -1.\n");
    }
    return -1;
  }
  matC.load(index, nCols_, Cmat, cntsC);
  for (ii = 0; ii < index; ii++) delete [] Cmat[ii];
  delete [] Cmat;
  delete [] cntsC;
  return 0;
}

// ************************************************************************
// multiply 3 probability matrices (D = A * B *C)
// ------------------------------------------------------------------------
int ProbMatrix::multiply3(ProbMatrix &matB,ProbMatrix &matC,ProbMatrix &matD)
{
  //**/ -----------------------------------------------------------
  //**/ error checking
  //**/ -----------------------------------------------------------
  if (nCols_ != matB.ncols())
  {
    printf("ProbMatrix multiply3 ERROR: different number of columns.\n");
    printf("      INFO: %d (local) versus %d (incoming)\n",nCols_, 
           matB.ncols());
    exit(1);
  }
  if (nCols_ != matC.ncols())
  {
    printf("ProbMatrix multiply3 ERROR: different number of columns.\n");
    printf("      INFO: %d (local) versus %d (incoming)\n",nCols_, 
           matC.ncols());
    exit(1);
  }

  //**/ -----------------------------------------------------------
  //**/ extract pointers and allocate space 
  //**/ -----------------------------------------------------------
  int ii, jj, kk, ll, mm, index;
  int nrowsB = matB.nrows();
  int nrowsC = matC.nrows();
  int nrowsD = nRows_ + nrowsB + nrowsC;
  double **Bmat = matB.getMatrix2D();
  double **Cmat = matC.getMatrix2D();
  int    *cntsB = matB.getCounts();
  int    *cntsC = matC.getCounts();
  double **Dmat = new double*[nrowsD];
  int    *cntsD = new int[nrowsD];

  //**/ -----------------------------------------------------------
  //**/ perform multiplication
  //**/ -----------------------------------------------------------
  index = 0;
  for (ii = 0; ii < nRows_; ii++)
  {
    for (jj = 0; jj < nrowsB; jj++)
    {
      //**/ compare row ii of A with row jj of B
      for (ll = 0; ll < nCols_; ll++)
        if (Mat2D_[ii][ll] != Bmat[jj][ll]) 
          break;
      //**/ if Amat(ii) == Bmat(jj), examine C
      if (ll == nCols_)
      {
        for (kk = 0; kk < nrowsC; kk++)
        {
          //**/ compare Amat(ii) with Cmat(kk)
          for (mm = 0; mm < nCols_; mm++)
            if (Mat2D_[ii][mm] != Cmat[kk][mm]) 
              break;
          //**/ if Amat(ii) == Cmat(jj), process
          if (mm == nCols_)
          {
            Dmat[index] = new double[nCols_];
            for (mm = 0; mm < nCols_; mm++)
              Dmat[index][mm] = Mat2D_[ii][mm];
            cntsD[index] = counts_[ii]*cntsB[jj]*cntsC[kk];
            index++;
            //**/ if more space is needed
            if (index > nrowsC)
            {
              printf("ProbMatrix multiply3 ERROR: Something wrong.\n");
              exit(1);
            }
            //**/ break from loop kk
            break;
          }
        }
        //**/ break from loop jj
        break;
      }
    }
  }
  if (index == 0)
  {
    printf("ProbMatrix multiply3 WARNING: product = 0.\n");
    return -1;
  }
  matD.load(index, nCols_, Dmat, cntsD);
  for (ii = 0; ii < index; ii++) delete [] Dmat[ii];
  delete [] Dmat;
  delete [] cntsD;
  return 0;
}

// ************************************************************************
// clean up
// ------------------------------------------------------------------------
void ProbMatrix::clean()
{
  if (Mat2D_ != NULL)
  {
    for (int ii = 0; ii < nRows_; ii++)
      if (Mat2D_[ii] != NULL) delete [] Mat2D_[ii];
    delete [] Mat2D_;
    Mat2D_ = NULL;
  }
  if (counts_ != NULL) delete [] counts_;
  Mat2D_ = NULL;
  counts_ = NULL;
  nRows_ = nCols_ = 0;
}


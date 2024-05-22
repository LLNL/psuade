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
// DATE   : 2023
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "psMatrix3D.h"
#include "PsuadeUtil.h"

//#define PS_DEBUG 1
// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psMatrix3D::psMatrix3D()
{
#ifdef PS_DEBUG
  printf("psMatrix3D constructor\n");
#endif
  nDim1_ = 0;
  nDim2_ = 0;
  nDim3_ = 0;
  Mat3D_ = NULL;
#ifdef PS_DEBUG
  printf("psMatrix3D constructor ends\n");
#endif
}

// ************************************************************************
// Copy Constructor 
// ------------------------------------------------------------------------
psMatrix3D::psMatrix3D(const psMatrix3D & ma)
{
  int ii, jj, kk;

  nDim1_ = ma.nDim1_;
  nDim2_ = ma.nDim2_;
  nDim3_ = ma.nDim3_;
  Mat3D_ = NULL;
  if (nDim1_ > 0 && nDim2_ > 0 && nDim3_ > 0)
  {
    Mat3D_ = new double**[nDim1_];
    assert(Mat3D_ != NULL);
    for (ii = 0; ii < nDim1_; ii++)
    {
      Mat3D_[ii] = new double*[nDim2_];
      assert(Mat3D_[ii] != NULL);
      for (jj = 0; jj < nDim2_; jj++)
      {
        Mat3D_[ii][jj] = new double[nDim3_];
        assert(Mat3D_[ii][jj] != NULL);
        for (kk = 0; kk < nDim3_; kk++)
          Mat3D_[ii][jj][kk] = ma.Mat3D_[ii][jj][kk];
      }
    }
  }
}

// ************************************************************************
// operator=  
// ------------------------------------------------------------------------
psMatrix3D & psMatrix3D::operator=(const psMatrix3D & ma)
{
  int ii, jj, kk;

  if (this == &ma) return *this;
  clean();
  nDim1_ = ma.nDim1_;
  nDim2_ = ma.nDim2_;
  nDim3_ = ma.nDim3_;
  Mat3D_ = NULL;
  if (nDim1_ > 0 && nDim2_ > 0 && nDim3_ > 0)
  {
    Mat3D_ = new double**[nDim1_];
    assert(Mat3D_ != NULL);
    for (ii = 0; ii < nDim1_; ii++)
    {
      Mat3D_[ii] = new double*[nDim2_];
      assert(Mat3D_[ii] != NULL);
      for (jj = 0; jj < nDim2_; jj++)
      {
        Mat3D_[ii][jj] = new double[nDim3_];
        assert(Mat3D_[ii][jj] != NULL);
        for (kk = 0; kk < nDim3_; kk++)
          Mat3D_[ii][jj][kk] = ma.Mat3D_[ii][jj][kk];
      }
    }
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psMatrix3D::~psMatrix3D()
{
  clean();
}

// ************************************************************************
// get number of elements for each dimension 
// ------------------------------------------------------------------------
int psMatrix3D::ndim1()
{
  return nDim1_;
}

int psMatrix3D::ndim2()
{
  return nDim2_;
}

int psMatrix3D::ndim3()
{
  return nDim3_;
}

// ************************************************************************
// load matrix from another matrix
// ------------------------------------------------------------------------
int psMatrix3D::load(psMatrix3D &inMat)
{
  int ii, jj, kk;

#ifdef PS_DEBUG
  printf("psMatrix3D load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(this != &inMat);
  nDim1_  = inMat.ndim1();
  nDim2_  = inMat.ndim2();
  nDim3_  = inMat.ndim3();
  if (nDim1_ > 0 && nDim2_ > 0 && nDim3_ > 0)
  {
    Mat3D_ = new double**[nDim1_];
    assert(Mat3D_ != NULL);
    for (ii = 0; ii < nDim1_; ii++)
    {
      Mat3D_[ii] = new double*[nDim2_];
      assert(Mat3D_[ii] != NULL);
      for (jj = 0; jj < nDim2_; jj++)
      {
        Mat3D_[ii][jj] = new double[nDim3_];
        assert(Mat3D_[ii][jj] != NULL);
        for (kk = 0; kk < nDim3_; kk++)
          Mat3D_[ii][jj][kk] = inMat.getEntry(ii,jj,kk);
      }
    }
  }
#ifdef PS_DEBUG
  printf("psMatrix3D load ends\n");
#endif
  return 0;
}

// ************************************************************************
// load matrix from doubles
// ------------------------------------------------------------------------
int psMatrix3D::load(int ndim1, int ndim2, int ndim3, double ***mat)
{
  int ii, jj, kk;
#ifdef PS_DEBUG
  printf("psMatrix3D load\n");
#endif
  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(ndim1);
  assert(ndim2);
  assert(ndim3);
  assert(mat);
  nDim1_  = ndim1;
  nDim2_  = ndim2;
  nDim3_  = ndim3;
  Mat3D_ = new double**[nDim1_];
  assert(Mat3D_ != NULL);
  for (ii = 0; ii < nDim1_; ii++)
  {
    Mat3D_[ii] = new double*[nDim2_];
    assert(Mat3D_[ii] != NULL);
    for (jj = 0; jj < nDim2_; jj++)
    {
      Mat3D_[ii][jj] = new double[nDim3_];
      assert(Mat3D_[ii][jj] != NULL);
      for (kk = 0; kk < nDim3_; kk++)
        Mat3D_[ii][jj][kk] = mat[ii][jj][kk];
    }
  }
#ifdef PS_DEBUG
  printf("psMatrix3D load ends\n");
#endif
  return 0;
}

// ************************************************************************
// set matrix dimension
// ------------------------------------------------------------------------
int psMatrix3D::setDim(int ndim1, int ndim2, int ndim3)
{
  int ii, jj, kk;

  //**/ clean it first, if needed
  clean();

  //**/ load from the incoming matrix
  assert(ndim1);
  assert(ndim2);
  assert(ndim3);
  nDim1_  = ndim1;
  nDim2_  = ndim2;
  nDim3_  = ndim3;
  Mat3D_ = new double**[nDim1_];
  assert(Mat3D_ != NULL);
  for (ii = 0; ii < nDim1_; ii++)
  {
    Mat3D_[ii] = new double*[nDim2_];
    assert(Mat3D_[ii] != NULL);
    for (jj = 0; jj < nDim2_; jj++)
    {
      Mat3D_[ii][jj] = new double[nDim3_];
      assert(Mat3D_[ii][jj] != NULL);
      for (kk = 0; kk < nDim3_; kk++)
        Mat3D_[ii][jj][kk] = 0;
    }
  }
  return 0;
}

// ************************************************************************
// set entry
// ------------------------------------------------------------------------
void psMatrix3D::setEntry(const int dim1, const int dim2, const int dim3,
                          const double ddata)
{
  //**/ error checking
  if (dim1 < 0 || dim1 >= nDim1_ || dim2 < 0 || dim2 >= nDim2_ ||
      dim3 < 0 || dim3 >= nDim3_)
  {
    printf("Matrix3D setEntry ERROR: index (%d,%d,%d) not in range (%d,%d,%d)\n",
           dim1,dim2,dim3,nDim1_,nDim2_,nDim3_);
    exit(1);
  }
  Mat3D_[dim1][dim2][dim3] = ddata;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
double psMatrix3D::getEntry(const int dim1, const int dim2, const int dim3)
{
  //**/ error checking
  if (dim1 < 0 || dim1 >= nDim1_ || dim2 < 0 || dim2 >= nDim2_ ||
      dim3 < 0 || dim3 >= nDim3_)
  {
    printf("Matrix3D getEntry ERROR: index (%d,%d,%d) not in range (%d,%d,%d)\n",
           dim1,dim2,dim3,nDim1_,nDim2_,nDim3_);
    exit(1);
  }
  return Mat3D_[dim1][dim2][dim3];
}

// ************************************************************************
// get matrix 
// ------------------------------------------------------------------------
double ***psMatrix3D::getMatrix3D()
{
  return Mat3D_;
}

// ************************************************************************
// take matrix 
// ------------------------------------------------------------------------
double ***psMatrix3D::takeMatrix3D()
{
  double ***matOut;
  matOut = Mat3D_;
  Mat3D_ = NULL;
  clean();
  return matOut;
}

// ************************************************************************
// collapse matrix in 2D along a given dimension
// ------------------------------------------------------------------------
void psMatrix3D::collapse2D(int ind, psMatrix &matOut)
{
  int    ii, jj, kk;
  double ddata;
  if (ind < 0 || ind > 2)
  {
    printf("Matrix3D collapse2D ERROR: dimension should be 0, 1, or 2.\n");
    exit(1);
  }

  if (ind == 0) 
  {
    matOut.setDim(nDim2_,nDim3_);
    for (ii = 0; ii < nDim2_; ii++)
    {
      for (jj = 0; jj < nDim3_; jj++)
      {
        ddata = matOut.getEntry(ii,jj);
        for (kk = 0; kk < nDim1_; kk++)
          ddata += Mat3D_[kk][ii][jj];
        matOut.setEntry(ii,jj,ddata);
      }
    }
  }
  else if (ind == 1) 
  {
    matOut.setDim(nDim1_,nDim3_);
    for (ii = 0; ii < nDim1_; ii++)
    {
      for (jj = 0; jj < nDim3_; jj++)
      {
        ddata = matOut.getEntry(ii,jj);
        for (kk = 0; kk < nDim2_; kk++)
          ddata += Mat3D_[ii][kk][jj];
        matOut.setEntry(ii,jj,ddata);
      }
    }
  }
  else
  {
    matOut.setDim(nDim1_,nDim2_);
    for (ii = 0; ii < nDim1_; ii++)
    {
      for (jj = 0; jj < nDim2_; jj++)
      {
        ddata = matOut.getEntry(ii,jj);
        for (kk = 0; kk < nDim3_; kk++)
          ddata += Mat3D_[ii][jj][kk];
        matOut.setEntry(ii,jj,ddata);
      }
    }
  }
}

// ************************************************************************
// clean up
// ------------------------------------------------------------------------
void psMatrix3D::clean()
{
  if (Mat3D_ != NULL)
  {
    for (int ii = 0; ii < nDim1_; ii++)
    {
      for (int jj = 0; jj < nDim2_; jj++)
        if (Mat3D_[ii][jj] != NULL) delete [] Mat3D_[ii][jj];
      delete [] Mat3D_[ii];
    }
    delete [] Mat3D_;
  }
  Mat3D_ = NULL;
  nDim1_ = nDim2_ = nDim3_ = 0;
}


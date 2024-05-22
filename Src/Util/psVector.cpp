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
// psVector functions
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <fstream>
#include "sysdef.h"
#include "psVector.h"
#include "PsuadeUtil.h"
using namespace std;

//#define PS_DEBUG

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psVector::psVector()
{
#ifdef PS_DEBUG
  printf("psVector constructor\n");
#endif
  length_ = 0;
  Vec_ = NULL;
#ifdef PS_DEBUG
  printf("psVector constructor ends\n");
#endif
}

// ************************************************************************
// Copy Constructor by Bill Oliver
// ------------------------------------------------------------------------
psVector::psVector(const psVector & v)
{
  length_ = v.length_;
  Vec_ = NULL;
  if (length_ > 0)
  {
    try {
      Vec_ = new double[length_];
    }
    catch (const exception& e)
    {
      printf("psVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
}

// ************************************************************************
// overload operator= by Bill Oliver
// ------------------------------------------------------------------------
psVector & psVector::operator=(const psVector & v)
{
  if (this == &v) return *this;
  delete [] Vec_;
  Vec_ = NULL;
  length_ = v.length_;
  if (length_ > 0)
  {
    try {
      Vec_ = new double[length_];
    }
    catch (const exception& e)
    {
      printf("psVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psVector::~psVector()
{
#ifdef PS_DEBUG
  printf("psVector destructor\n");
#endif
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
#ifdef PS_DEBUG
  printf("psVector destructor ends\n");
#endif
}

// ************************************************************************
// get length 
// ------------------------------------------------------------------------
int psVector::length() 
{
  return length_;
}

// ************************************************************************
// load vector
// ------------------------------------------------------------------------
int psVector::load(psVector &inVec)
{
#ifdef PS_DEBUG
  printf("psVector load\n");
#endif
  assert(this != &inVec);
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = inVec.length();
  if (length_ <= 0) return -1;
  try {
    Vec_ = new double[length_];
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = inVec[ii];
  }
  catch (const exception& e)
  {
    printf("psVector load ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
#ifdef PS_DEBUG
  printf("psVector load ends\n");
#endif
  return 0;
}

// ************************************************************************
// give vector
// ------------------------------------------------------------------------
int psVector::giveDVector(int leng, double *inVec)
{
  if (Vec_ != NULL) delete [] Vec_;
  length_ = leng;
  Vec_ = inVec;
  return 0;
}

// ************************************************************************
// set dimension
// ------------------------------------------------------------------------
int psVector::setLength(int leng)
{
#ifdef PS_DEBUG
  printf("psVector setLength\n");
#endif
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  if (leng == 0) return -1;
  length_ = leng;
  try {
    Vec_ = new double[length_];
  }
  catch (const exception& e)
  {
    printf("psVector setLength ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (int ii = 0; ii < leng; ii++) Vec_[ii] = 0.0;
#ifdef PS_DEBUG
  printf("psVector setLength ends\n");
#endif
  return 0;
}

// ************************************************************************
// load vector
// ------------------------------------------------------------------------
int psVector::load(int leng, double *data)
{
#ifdef PS_DEBUG
  printf("psVector load, length = %d\n", leng);
#endif
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  assert(leng > 0);
  assert(data != NULL);
  length_ = leng;
  try {
    Vec_ = new double[length_];
    for (int ii = 0; ii < leng; ii++) Vec_[ii] = data[ii];
  }
  catch (const exception& e)
  {
    printf("psVector load ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
#ifdef PS_DEBUG
  printf("psVector load ends\n");
#endif
  return 0;
}

// ************************************************************************
// set entry
// ------------------------------------------------------------------------
int psVector::setEntry(int ind, double ddata)
{
  if (ind < 0 || ind >= length_)
  {
    printf("psVector setEntry ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  Vec_[ind] = ddata;
  return 0;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
double psVector::getEntry(int ind)
{
  if (ind < 0 || ind >= length_)
  {
    printf("psVector getEntry ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
double& psVector::operator[](int ind) 
{
  if (ind < 0 || ind >= length_)
  {
    printf("psVector operator[] ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// find max 
// ------------------------------------------------------------------------
double psVector::max()
{
  int    ii;
  double dmax = -PSUADE_UNDEFINED;
  for (ii = 0; ii < length_; ii++) 
    if (Vec_[ii] > dmax) dmax = Vec_[ii];
  return dmax;
}
   
// ************************************************************************
// find min 
// ------------------------------------------------------------------------
double psVector::min()
{
  int    ii;
  double dmin = PSUADE_UNDEFINED;
  for (ii = 0; ii < length_; ii++) 
    if (Vec_[ii] < dmin) dmin = Vec_[ii];
  return dmin;
}
   
// ************************************************************************
// compute vector sum 
// ------------------------------------------------------------------------
double psVector::sum()
{
  int    ii;
  double dsum=0.0;
  for (ii = 0; ii < length_; ii++) dsum += Vec_[ii];
  return dsum;
}

// ************************************************************************
// compute vector standard deviation 
// ------------------------------------------------------------------------
double psVector::stdev()
{
  int    ii;
  double dmean=0.0, dstdv=0.0;
  for (ii = 0; ii < length_; ii++) dmean += Vec_[ii];
  dmean /= (double) length_;
  for (ii = 0; ii < length_; ii++) 
    dstdv += pow(Vec_[ii] - dmean, 2.0);
  dstdv = sqrt(dstdv / (double) length_);
  return dstdv;
}

// ************************************************************************
// check if there is any UNDEFINED
// ------------------------------------------------------------------------
int psVector::countUndefined()
{
  int count = 0;
  for (int ii = 0; ii < length_; ii++) 
    if (Vec_[ii] > 0.9 * PSUADE_UNDEFINED) count++;
  return count;
}

// ************************************************************************
// scale vector 
// ------------------------------------------------------------------------
void psVector::scale(double alpha)
{
  for (int ii = 0; ii < length_; ii++) Vec_[ii] *= alpha;
}

// ************************************************************************
// vector norm
// ------------------------------------------------------------------------
double psVector::norm()
{
  int    ii;
  double dnorm=0.0;
  for (ii = 0; ii < length_; ii++) dnorm += Vec_[ii] * Vec_[ii];
  dnorm = sqrt(dnorm);
  return dnorm;
}

// ************************************************************************
// set the entire vector to a constant
// ------------------------------------------------------------------------
void psVector::setConstant(double value)
{
  for (int ii = 0; ii < length_; ii++) Vec_[ii] = value;
}

// ************************************************************************
// vector y = y + a x 
// ------------------------------------------------------------------------
void psVector::axpy(const double scale, psVector v1)
{
  assert(length_ == v1.length());
  for (int ii = 0; ii < length_; ii++) Vec_[ii] += scale * v1[ii];
}

// ************************************************************************
// sort 
// ------------------------------------------------------------------------
void psVector::sort()
{
  sortDbleList(length_, Vec_);
}
   
// ************************************************************************
// add to vector
// ------------------------------------------------------------------------
int psVector::addElements(int leng, double *data)
{
#ifdef PS_DEBUG
  printf("psVector addElements, length = %d\n", leng);
#endif
  int    ii;
  double *tmpVec = Vec_;
  try {
    Vec_ = new double[leng+length_];
    for (ii = 0; ii < length_; ii++) Vec_[ii] = tmpVec[ii];
    if (data == NULL)
         for (ii = 0; ii < leng; ii++) Vec_[length_+ii] = 0.0;
    else for (ii = 0; ii < leng; ii++) Vec_[length_+ii] = data[ii];
  }
  catch (const exception& e)
  {
    printf("psVector addElements ERROR: Failed to allocate (length=%d)\n",
           length_);
    exit(1);
  }
  delete [] tmpVec;
  length_ += leng;
#ifdef PS_DEBUG
   printf("psVector addElements ends\n");
#endif
   return 0;
}

// ************************************************************************
// generate a subvector 
// ------------------------------------------------------------------------
void psVector::subvector(int ibeg, int iend)
{
  int    ii, leng;
  double *tmpVec;

  leng = iend - ibeg + 1;
  if (leng < 0 || ibeg < 0 || iend >= length_)
  {
    printf("psVector subvector range ERROR: beg/end = %d %d\n",ibeg,iend);
    exit(1);
  }
  tmpVec = Vec_;
  length_ = leng;
  try {
    Vec_ = new double[length_];
  }
  catch (const exception& e)
  {
    printf("psVector subvector ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (ii = ibeg; ii < iend+1; ii++) Vec_[ii-ibeg] = tmpVec[ii];
  if (tmpVec != NULL) delete [] tmpVec;
  return;
}

// ************************************************************************
// clean
// ------------------------------------------------------------------------
void psVector::clean()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}

// ************************************************************************
// print
// ------------------------------------------------------------------------
void psVector::print(char *name)
{
  for (int ii = 0; ii < length_; ii++) 
    printf("%s %5d = %16.8e\n", name, ii+1, Vec_[ii]);
}

// ************************************************************************
// generate vector of random numbers
// ------------------------------------------------------------------------
void psVector::genRandom(int length, double lb, double ub)
{
  if (ub <= lb)
  {
    printf("psVector genRandom ERROR: invalid range [%e,%e].\n",lb,ub);
    exit(1);
  } 
  setLength(length);
  double width = ub - lb;
  for (int ii = 0; ii < length_; ii++) 
    Vec_[ii] = PSUADE_drand() * width + lb;
}

// ************************************************************************
// get vector
// ------------------------------------------------------------------------
double *psVector::getDVector() 
{
  return Vec_;
}

// ************************************************************************
// take vector
// ------------------------------------------------------------------------
double *psVector::takeDVector() 
{
  double *tmpVec = Vec_;
  Vec_ = NULL;
  clean();
  return tmpVec;
}

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psIVector::psIVector()
{
  length_ = 0;
  Vec_ = NULL;
}

// ************************************************************************
// Copy Constructor 
// ------------------------------------------------------------------------
psIVector::psIVector(const psIVector & v)
{
  length_ = v.length_;
  Vec_ = NULL;
  if (length_ > 0)
  {
    try {
      Vec_ = new int[length_];
    }
    catch (const exception& e)
    {
      printf("psIVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
}

// ************************************************************************
// set entry
// ------------------------------------------------------------------------
int psIVector::setEntry(int ind, int idata)
{
  if (ind < 0 || ind >= length_)
  {
    printf("psIVector setEntry ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  Vec_[ind] = idata;
  return 0;
}

// ************************************************************************
// compute vector sum 
// ------------------------------------------------------------------------
int psIVector::sum()
{
  int ii, isum=0;
  for (ii = 0; ii < length_; ii++) isum += Vec_[ii];
  return isum;
}

// ************************************************************************
// compute vector max 
// ------------------------------------------------------------------------
int psIVector::max()
{
  int ii, imax=-1000000000;
  for (ii = 0; ii < length_; ii++) 
    if (Vec_[ii] > imax) imax = Vec_[ii];
  return imax;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
int psIVector::getEntry(int ind)
{
  if (ind < 0 || ind >= length_)
  {
    printf("psIVector getEntry ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// overload operator= 
// ------------------------------------------------------------------------
psIVector & psIVector::operator=(const psIVector & v)
{
  if (this == &v) return *this;
  delete [] Vec_;
  Vec_ = NULL;
  length_ = v.length_;
  if (length_ > 0)
  {
    try {
      Vec_ = new int[length_];
    }
    catch (const exception& e)
    {
      printf("psIVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psIVector::~psIVector()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}

// ************************************************************************
// get length 
// ------------------------------------------------------------------------
int psIVector::length() 
{
  return length_;
}

// ************************************************************************
// load vector
// ------------------------------------------------------------------------
int psIVector::load(psIVector &inVec)
{
  assert(this != &inVec);
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = inVec.length();
  if (length_ <= 0) return -1;
  try {
    Vec_ = new int[length_];
  }
  catch (const exception& e)
  {
    printf("psIVector load ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (int ii = 0; ii < length_; ii++) Vec_[ii] = inVec[ii];
  return 0;
}

// ************************************************************************
// set dimension
// ------------------------------------------------------------------------
int psIVector::setLength(int leng)
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  assert(leng >= 0);
  if (leng == 0) return -1;
  length_ = leng;
  try {
    Vec_ = new int[length_];
  }
  catch (const exception& e)
  {
    printf("psIVector setLength ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (int ii = 0; ii < leng; ii++) Vec_[ii] = 0;
  return 0;
}

// ************************************************************************
// load vector
// ------------------------------------------------------------------------
int psIVector::load(int leng, int *data)
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  assert(leng > 0);
  assert(data != NULL);
  length_ = leng;
  try {
    Vec_ = new int[length_];
    for (int ii = 0; ii < leng; ii++) Vec_[ii] = data[ii];
  }
  catch (const exception& e)
  {
    printf("psIVector load ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  return 0;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
int& psIVector::operator[](int ind) 
{
  if (ind < 0 || ind >= length_)
  {
    printf("psIVector operator[] ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// generate a subvector 
// ------------------------------------------------------------------------
void psIVector::subvector(int ibeg, int iend)
{
  int ii, leng, *tmpVec;

  leng = iend - ibeg + 1;
  if (leng < 0 || ibeg < 0 || iend >= length_)
  {
    printf("psIVector subvector range ERROR: beg/end = %d %d\n",ibeg,iend);
    exit(1);
  }
  tmpVec = Vec_;
  length_ = leng;
  try {
    Vec_ = new int[length_];
  }
  catch (const exception& e)
  {
    printf("psIVector subvector: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (ii = ibeg; ii < iend+1; ii++) Vec_[ii-ibeg] = tmpVec[ii];
  if (tmpVec != NULL) delete [] tmpVec;
  return;
}

// ************************************************************************
// set the entire vector to a constant
// ------------------------------------------------------------------------
void psIVector::setConstant(int ival)
{
  for (int ii = 0; ii < length_; ii++) Vec_[ii] = ival;
}

// ************************************************************************
// clean
// ------------------------------------------------------------------------
void psIVector::clean()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}

// ************************************************************************
// print
// ------------------------------------------------------------------------
void psIVector::print(char *name)
{
  for (int ii = 0; ii < length_; ii++) 
    printf("%s %5d = %d\n", name, ii+1, Vec_[ii]);
}

// ************************************************************************
// get vector
// ------------------------------------------------------------------------
int *psIVector::getIVector() 
{
  return Vec_;
}

// ************************************************************************
// take vector
// ------------------------------------------------------------------------
int *psIVector::takeIVector() 
{
  int *tmpVec = Vec_;
  Vec_ = NULL;
  clean();
  return tmpVec;
}

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psFVector::psFVector()
{
  length_ = 0;
  Vec_ = NULL;
}

// ************************************************************************
// Copy Constructor 
// ------------------------------------------------------------------------
psFVector::psFVector(const psFVector & v)
{
  length_ = v.length_;
  Vec_ = NULL;
  if (length_ > 0)
  {
    try {
      Vec_ = new float[length_];
    }
    catch (const exception& e)
    {
      printf("psFVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
}

// ************************************************************************
// overload operator= 
// ------------------------------------------------------------------------
psFVector & psFVector::operator=(const psFVector & v)
{
  if (this == &v) return *this;
  delete [] Vec_;
  Vec_ = NULL;
  length_ = v.length_;
  if (length_ > 0)
  {
    try {
      Vec_ = new float[length_];
    }
    catch (const exception& e)
    {
      printf("psFVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psFVector::~psFVector()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}

// ************************************************************************
// get length 
// ------------------------------------------------------------------------
int psFVector::length() 
{
  return length_;
}

// ************************************************************************
// load vector
// ------------------------------------------------------------------------
int psFVector::load(psFVector &inVec)
{
  assert(this != &inVec);
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = inVec.length();
  if (length_ <= 0) return -1;
  try {
    Vec_ = new float[length_];
  }
  catch (const exception& e)
  {
    printf("psFVector load ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (int ii = 0; ii < length_; ii++) Vec_[ii] = inVec[ii];
  return 0;
}

// ************************************************************************
// set dimension
// ------------------------------------------------------------------------
int psFVector::setLength(int leng)
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  assert(leng > 0);
  length_ = leng;
  try {
    Vec_ = new float[length_];
  }
  catch (const exception& e)
  {
    printf("psFVector setLength ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (int ii = 0; ii < leng; ii++) Vec_[ii] = 0;
  return 0;
}

// ************************************************************************
// load vector
// ------------------------------------------------------------------------
int psFVector::load(int leng, float *data)
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  assert(leng > 0);
  assert(data != NULL);
  length_ = leng;
  try {
    Vec_ = new float[length_];
    for (int ii = 0; ii < leng; ii++) Vec_[ii] = data[ii];
  }
  catch (const exception& e)
  {
    printf("psFVector load ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  return 0;
}

// ************************************************************************
// set entry
// ------------------------------------------------------------------------
int psFVector::setEntry(int ind, float fdata)
{
  if (ind < 0 || ind >= length_)
  {
    printf("psFVector setEntry ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  Vec_[ind] = fdata;
  return 0;
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
float psFVector::getEntry(int ind)
{
  if (ind < 0 || ind >= length_)
  {
    printf("psFVector getEntry ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
float& psFVector::operator[](int ind) 
{
  if (ind < 0 || ind >= length_)
  {
    printf("psFVector operator[] ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// set the entire vector to a constant
// ------------------------------------------------------------------------
void psFVector::setConstant(float fval)
{
  for (int ii = 0; ii < length_; ii++) Vec_[ii] = fval;
}

// ************************************************************************
// clean
// ------------------------------------------------------------------------
void psFVector::clean()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}

// ************************************************************************
// get vector
// ------------------------------------------------------------------------
float *psFVector::getFVector() 
{
  return Vec_;
}

// ************************************************************************
// take vector
// ------------------------------------------------------------------------
float *psFVector::takeFVector() 
{
  float *tmpVec = Vec_;
  Vec_ = NULL;
  clean();
  return tmpVec;
}

// ************************************************************************
// Constructor
// ------------------------------------------------------------------------
psLDVector::psLDVector()
{
  length_ = 0;
  Vec_ = NULL;
}

// ************************************************************************
// Copy Constructor 
// ------------------------------------------------------------------------
psLDVector::psLDVector(const psLDVector & v)
{
  length_ = v.length_;
  Vec_ = NULL;
  if (length_ > 0)
  {
    try {
      Vec_ = new long double[length_];
    }
    catch (const exception& e)
    {
      printf("psLDVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
}

// ************************************************************************
// overload operator= 
// ------------------------------------------------------------------------
psLDVector & psLDVector::operator=(const psLDVector & v)
{
  if (this == &v) return *this;
  delete [] Vec_;
  Vec_ = NULL;
  length_ = v.length_;
  if (length_ > 0)
  {
    try {
      Vec_ = new long double[length_];
    }
    catch (const exception& e)
    {
      printf("psLDVector ERROR: Failed to allocate (length = %d)\n",
             length_);
      exit(1);
    }
    for (int ii = 0; ii < length_; ii++) Vec_[ii] = v.Vec_[ii];
  }
  return *this;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
psLDVector::~psLDVector()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}

// ************************************************************************
// get length 
// ------------------------------------------------------------------------
int psLDVector::length() 
{
  return length_;
}

// ************************************************************************
// set dimension
// ------------------------------------------------------------------------
int psLDVector::setLength(int leng)
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  assert(leng >= 0);
  if (leng == 0) return -1;
  length_ = leng;
  try {
    Vec_ = new long double[length_];
  }
  catch (const exception& e)
  {
    printf("psLDVector setLength ERROR: Failed to allocate (length = %d)\n",
           length_);
    exit(1);
  }
  for (int ii = 0; ii < leng; ii++) Vec_[ii] = 0.0;
  return 0;
}

// ************************************************************************
// print
// ------------------------------------------------------------------------
void psLDVector::print(char *name)
{
  for (int ii = 0; ii < length_; ii++) 
    printf("%s %5d = %24.16llf\n", name, ii+1, Vec_[ii]);
}

// ************************************************************************
// get entry
// ------------------------------------------------------------------------
long double& psLDVector::operator[](int ind) 
{
  if (ind < 0 || ind >= length_)
  {
    printf("psLDVector operator[] ERROR: index = %d (%d)\n",ind+1,length_);
    exit(1);
  }
  return Vec_[ind];
}

// ************************************************************************
// clean
// ------------------------------------------------------------------------
void psLDVector::clean()
{
  if (Vec_ != NULL) delete [] Vec_;
  Vec_ = NULL;
  length_ = 0;
}


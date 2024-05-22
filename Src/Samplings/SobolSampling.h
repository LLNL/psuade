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
// Definition for the Sobol's one-at-a-time sampling class
// AUTHOR : CHARLES TONG
// DATE   : 2004
// ************************************************************************
#ifndef __SOBOLSAMPLINGH__
#define __SOBOLSAMPLINGH__

#include <stdlib.h>
#include <string.h>
#include <math.h>
#include "Sampling.h"
#include "psMatrix.h"
#include "psVector.h"
#include "pdfData.h"

/**
 * @name Sobol samping method
 *
 **/

// ************************************************************************
// class definition
// ************************************************************************
class SobolSampling: public Sampling 
{
  int order_;
  psVector VecM1_;
  psVector VecM2_;

public:

  /** constructor */
  SobolSampling();

  /** destructor */
  ~SobolSampling();

  /** initialization 
      @param flag: flag to signal how far to initialize
   */
  int initialize(int flag);

  /** initialization for second order 
      @param flag: flag to signal how far to initialize
   */
  int initialize2(int flag);

  /** initialization for second order 
      @param flag: flag to signal how far to initialize
      @param matGrp: matrix to provide group information
   */
  int initialize3(int flag, psIMatrix matGrp);

  /** This function refines an incoming sample
      @param ratio: refinement ratio
      @param randomize: generate randomized sample
      @param thresh: threshold
      @param nSamples: sample size
      @param sampleErrs: errors for each sample point
   */
  int refine(int ratio,int randomize,double thresh,int nSamples,
             double *sampleErrs);

  /** This function sets M1 and M2
      @param vecM - 2 samples
   */
  int setM1M2(psVector vecM);

  /** This function creates M1 and M2 from some distributions
      @param sobj 
   */
  int createM1M2(pdfData sobj);

  /** This function sets first or second order 
      @param order - 1, 2, or 3
   */
  int setOrder(int order);

  /** This function sets internal parameter
      @param sparam 
   */
  int setParam(char *sparam);

  /** This function overloads the assignment operator
      @param obj : Sampling object
   */
  SobolSampling& operator=(const SobolSampling &);

private:

  int generate(double **, int);
};

#endif // __SOBOLSAMPLINGH__


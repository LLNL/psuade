// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team. 
// All rights reserved.
//
// Please see the COPYRIGHT and LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// PSUADE is free software; you can redistribute it and/or modify it under the
// terms of the GNU Lesser General Public License (as published by the Free 
// Software Foundation) version 2.1 dated February 1999.
//
// PSUADE is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU Lesser
// General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Definition for the class aData (for passing analysis data)
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#ifndef __ADATAH__
#define __ADATAH__

#include "PsuadeData.h"

// ************************************************************************
// class definition
// ************************************************************************
class aData 
{
public:

  int    printLevel_;

  //**/ sample information
  int    nSamples_;
  int    nInputs_;
  int    nOutputs_;
  int    outputID_;
  int    *sampleStates_;
  double *sampleInputs_;
  double *sampleOutputs_;
  double *iLowerB_;
  double *iUpperB_;
  //**/ nSubSamples * nReps = nSamples
  int    nSubSamples_;
  //**/ which input needs transformation
  int    *inputXsforms_;

  //**/ input probability distribution functions
  int    *inputPDFs_;
  double *inputMeans_;
  double *inputStdevs_;

  //**/ sampling method and sampling pointer
  int    samplingMethod_;
  void   *sampler_;

  //**/ refinement partition information
  int    currRefineLevel_;
  int    *refineSeparators_;   

  //**/ threshold for convergence in analysis
  double analysisThreshold_;
  //**/ interpolation error of individual sample
  double *sampleErrors_;

  //**/ file passed to analyzer
  char grpFileName_[200];

  //**/ cross validation flag
  int    cvFlag_;
  //**/ regression weights (which output)
  int    regWgtID_;
  //**/ response surface type
  int    faType_;

  //**/ pointer to all user-specific information
  PsuadeData *ioPtr_;

  //**/ values returned from analyzer
  double *retValues_;

  //**/ constructor
  aData();

  //**/ destructor
  ~aData();

  //**/ assign operator
  //**/ @param obj - contains all data
  aData& operator=(const aData &obj);
};

#endif // __ADATAH__


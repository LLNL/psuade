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
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include <stdio.h>
#include <assert.h>
#include "dtype.h"
#include "sysdef.h"
#include "aData.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------ 
aData::aData()
{
  printLevel_    = 0;
  nSamples_      = 0;
  nSubSamples_   = 0;
  nInputs_       = 0;
  nOutputs_      = 0;
  outputID_      = 0;
  cvFlag_        = 0;
  samplingMethod_    = -1;
  regWgtID_          = -1;
  currRefineLevel_   = 0;
  refineSeparators_  = 0;
  analysisThreshold_ = 0.0;
  sampleStates_  = NULL;
  sampleInputs_  = NULL;
  sampleOutputs_ = NULL;
  iLowerB_       = NULL;
  iUpperB_       = NULL;
  inputPDFs_     = NULL;
  inputMeans_    = NULL;
  inputStdevs_   = NULL;
  sampleErrors_  = NULL;
  inputXsforms_  = NULL;
  retValues_     = NULL;
  ioPtr_         = NULL;
  sampler_       = NULL;
  faType_        = -1;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------ 
aData::~aData()
{ 
  //**/ other variables are passed by reference, so no need to deallocate
  if (sampleErrors_ != NULL) delete [] sampleErrors_;
  if (inputXsforms_ != NULL) delete [] inputXsforms_;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
aData& aData::operator=(const aData &)
{
  printOutTS(PL_ERROR, "aData operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}


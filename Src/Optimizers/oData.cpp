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
// Definition for the class oData
// AUTHOR : CHARLES TONG
// DATE   : 2005
// ************************************************************************
#include "oData.h"

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
oData::oData()
{
  outputLevel_ = -1;
  nInputs_ = 0;
  nSamples_ = 0;
  nOutputs_ = 0;
  numFuncEvals_ = 0;
  maxParallelJobs_ = 0;
  setOptDriver_ = -1;
  initialX_ = NULL;
  lowerBounds_ = NULL;
  upperBounds_ = NULL;
  optimalX_ = NULL;
  targetY_ = PSUADE_UNDEFINED;
  funcIO_ = NULL;
  psIO_ = NULL;
  intData_ = -1;
  optFunction_ = NULL;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
oData::~oData()
{
  if (lowerBounds_ != NULL) delete [] lowerBounds_;
  if (upperBounds_ != NULL) delete [] upperBounds_;
  if (initialX_ != NULL) delete [] initialX_;
  if (optimalX_ != NULL) delete [] optimalX_;
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
oData& oData::operator=(const oData &)
{
  printf("oData operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}


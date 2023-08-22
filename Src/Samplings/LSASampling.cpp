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
// Functions for the Local Sensitivity Analysis sampling class
// AUTHOR : CHARLES TONG
// DATE   : 2012
// ************************************************************************
//**/using namespace std;
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "LSASampling.h"
#include "Psuade.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
LSASampling::LSASampling() : Sampling()
{
  samplingID_ = PSUADE_SAMP_LSA;
  nPtsPerInput_ = 1;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
LSASampling::~LSASampling()
{
}

// ************************************************************************
// initialization 
// ------------------------------------------------------------------------
int LSASampling::initialize(int initLevel)
{
  int    ii, ss;
  double ddata;
  char   pString[1000];

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples_ == 0)
  {
    printf("LSASampling::initialize ERROR - nSamples = 0.\n");
    exit(1);
  }
  if (nInputs_ == 0)
  {
    printf("LSASampling::initialize ERROR - input not set up.\n");
    exit(1);
  }
  if (psConfig_.SamExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    printf("LSASampling: useful for local sensitivity analysis.\n");
    printf("The default pattern consists of one sample ");
    printf("point at the midpoint\n");
    printf("of the parameter space and one sample point on ");
    printf("one of the 2 faces\n");
    printf("for each input. So the default nSamples = nInputs + 1.\n");
    printf("However, you can define more points per input dimension.\n");
    snprintf(pString,100,"How many points per input dimension (default=1)? ");
    nPtsPerInput_ = getInt(1,10000,pString);
  }
  nSamples_ = nPtsPerInput_ * nInputs_ + 1;

  //**/ ----------------------------------------------------------------
  //**/ diagnostics
  //**/ ----------------------------------------------------------------
  if (printLevel_ > 4)
  {
    printf("LSASampling::initialize: nSamples = %d\n", nSamples_);
    printf("LSASampling::initialize: nInputs  = %d\n", nInputs_);
    printf("LSASampling::initialize: nOutputs = %d\n", nOutputs_);
    for (ii = 0; ii < nInputs_; ii++)
      printf("    LSASampling input %3d = [%e %e]\n", ii+1,
             vecLBs_[ii], vecUBs_[ii]);
  }

  //**/ ----------------------------------------------------------------
  //**/ sanitize and set parameters
  //**/ ----------------------------------------------------------------
  if (initLevel != 0) return 0;

  //**/ ----------------------------------------------------------------
  //**/ generate samples
  //**/ ----------------------------------------------------------------
  allocSampleData();
  for (ii = 0; ii < nInputs_; ii++) 
    vecSamInps_[ii] = 0.5*(vecUBs_[ii]+vecLBs_[ii]);
  if (nPtsPerInput_ == 1)
  {
    for (ss = 1; ss < nSamples_; ss++)
    {
      for (ii = 0; ii < nInputs_; ii++) 
        vecSamInps_[ss*nInputs_+ii] = 0.5*(vecUBs_[ii]+vecLBs_[ii]);
      ddata = PSUADE_drand();
      if (ddata >= 0.5) ddata = 1.0;
      else              ddata = 0.0;
      vecSamInps_[ss*nInputs_+ss-1] = vecLBs_[ss-1] + ddata *
                              (vecUBs_[ss-1] - vecLBs_[ss-1]);
    }
  }
  else
  {
    int n1 = nPtsPerInput_ / 2;
    int n2 = nPtsPerInput_ - n1;
    int cnt = 1;
    for (ss = 0; ss < nSamples_; ss++)
    {
      for (ii = 0; ii < nInputs_; ii++) 
        vecSamInps_[ss*nInputs_+ii] = 0.5*(vecUBs_[ii]+vecLBs_[ii]);
    }
    for (ii = 0; ii < nInputs_; ii++) 
    {
      for (ss = 0; ss < n1; ss++)
      {
        ddata = ss*(vecUBs_[ii]-vecLBs_[ii])/(2.0*n1) + vecLBs_[ii];  
        vecSamInps_[cnt*nInputs_+ii] = ddata;
        cnt++;
      }
      for (ss = 1; ss <= n2; ss++)
      {
        ddata = ss*(vecUBs_[ii]-vecLBs_[ii])/(2.0*n2)+
                0.5*(vecLBs_[ii] + vecUBs_[ii]);  
        vecSamInps_[cnt*nInputs_+ii] = ddata;
        cnt++;
      }
    }
  }
  return 0;
}

// ************************************************************************
// refine 
// ------------------------------------------------------------------------
int LSASampling::refine(int, int, double, int, double *)
{
  printf("LSASampling ERROR - refine not available.\n");
  return -1;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
LSASampling& LSASampling::operator=(const LSASampling &)
{
  printf("LSASampling operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}


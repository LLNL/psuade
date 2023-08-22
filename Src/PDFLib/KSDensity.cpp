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
// Functions for the KSDensity
// AUTHOR : CHARLES TONG
// DATE   : 2017
// ************************************************************************
#include <assert.h>
#include <stdio.h>
#include <math.h>
#include "KSDensity.h"
#define PABS(x) ((x >= 0) ? x : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
KSDensity::KSDensity()
{
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
KSDensity::~KSDensity()
{
}

// ************************************************************************
// generate histogram (Xp, Pp) from a small sample vecData for 1D only
// ------------------------------------------------------------------------
void KSDensity::genDensity1D(psVector &vecData, psVector &vecXp, 
                             psVector &vecPp)
{
  int    outLeng=101, ii, jj, kk;
  double lower, upper, width, median, accum, alpha, pi2Inv=0.5/3.1415928;
  double ddata, ddata2;
  psVector vecDS;

  //**/ sort data set and take median
  vecDS = vecData;
  vecDS.sort();
  kk = vecDS.length() / 2;
  median = vecDS[kk]; 

  //**/ take absolute value of data after subtracting median
  for (ii = 0; ii < vecDS.length(); ii++) 
    vecDS[ii] = PABS(vecDS[ii] - median);

  //**/ sort again and take median
  vecDS.sort();
  median = vecDS[kk]; 

  //**/ compute alpha for prescribing lower and upper bounds
  alpha = median / 0.6745;
  if (alpha == 0) alpha = vecData.max() - vecData.min();
  alpha *= pow(4.0 / (3.0 * vecData.length()), 0.2);
  if (alpha == 0) alpha = 1.0;
  lower = vecData.min() - 2.0 * alpha;
  upper = vecData.max() + 2.0 * alpha;
  width = (upper - lower) / outLeng;

  //**/ compute at each X increment the corresponding probability
  //**/ by summing the exponential of its distance from other points
  vecXp.setLength(outLeng);
  vecPp.setLength(outLeng);
  for (ii = 0; ii < outLeng; ii++)
  {
    vecXp[ii] = ddata = lower + width * (0.5 + ii);
    accum = 0.0;
    for (jj = 0; jj < vecData.length(); jj++)
    {
      ddata2 = (ddata - vecData[jj]) / alpha;
      accum += exp(-0.5 * ddata2 * ddata2) * pi2Inv;
    }
    vecPp[ii] = accum / (double) vecData.length();
  }

  //**/ finally normalize
  ddata = vecPp.sum();
  if (ddata != 0.0)
  {
    ddata = 1.0 / ddata;
    vecPp.scale(ddata);
  }
}


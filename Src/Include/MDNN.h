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
// Definition for the class MDNN
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ************************************************************************
#ifndef __MDNNH__
#define __MDNNH__

#include <stdio.h>
#include "FuncApprox.h"
#include "psMatrix.h"
#include "psVector.h"
#include "DNN.h"

// ************************************************************************
// // class definition
// // ************************************************************************
class MDNN_Box
{
public:
  int nSamples_;
  psVector vecLBs_;
  psVector vecUBs_;
  psVector vecX_;
  psVector vecY_;
  FuncApprox *rsPtr_;
};

// ************************************************************************
// class definition
// ************************************************************************
class MDNN : public FuncApprox 
{
  psVector XData_;
  psVector YData_;
  psVector vecXd_;
  MDNN_Box **boxes_;
  int      nPartitions_;
  int      partSize_;
  int      nLevels_;
  int      numNodes_;
  int      optScheme_;
  int      maxIter_;

public:

  MDNN(int, int);
  ~MDNN();

  int    initialize(double*,double*);
  int    genNDGridData(double*,double*,int*,double**,double**);
  int    gen1DGridData(double*,double *,int,double*, 
                       int *, double **, double **);
  int    gen2DGridData(double*,double *,int,int,double*, 
                       int *, double **, double **);
  int    gen3DGridData(double*,double *,int,int,int,double*, 
                       int *, double **, double **);
  int    gen4DGridData(double*,double *,int,int,int,int,double*, 
                       int *, double **, double **);

  double evaluatePoint(double *);
  double evaluatePoint(int, double *, double *);
  double evaluatePointFuzzy(double *, double &);
  double evaluatePointFuzzy(int, double *, double *, double *);

private:
  int    interpolate(int, double *, double *, double *);
};

#endif // __MDNNH__


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
// Functions for the class SumOfTrees
// AUTHOR : CHARLES TONG
// DATE   : 2010
// ************************************************************************
#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>

#include "SumOfTrees.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "Psuade.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// Constructor for object class SumOfTrees
// ------------------------------------------------------------------------
SumOfTrees::SumOfTrees(int nInputs,int nSamples) : 
                 FuncApprox(nInputs,nSamples)
{
  char pString[101];

  //**/ set identifier
  faID_ = PSUADE_RS_SOTS;

  //**/ set mode (mode = 0: bagging, mode = 1 : boosting
  mode_ = 0;

  //**/ default number of trees and other parameters
  numTrees_ = 100;
  treePtrs_ = NULL;
  //**/ the unimportant ones can be bumped up if minPtsPerNode too small
  //**/ a recommendation is 10.
  minPtsPerNode_ = 10;
  shrinkFactor_ = 0.005;
  tolerance_ = 1.0e-6;

  //**/ user adjustable parameters
  if (psConfig_.RSExpertModeIsOn())
  {
    snprintf(pString,100,"Enter the desired number of trees (>10): ");
    numTrees_ = getInt(10, 10000, pString);
    snprintf(pString,100,"Enter minimum points per node (>1, default=10): ");
    minPtsPerNode_ = getInt(2, nSamples/5, pString);
  }

  //**/ allocate trees
  treePtrs_ = new TreeNode*[numTrees_];
  for (int ii = 0; ii < numTrees_; ii++) treePtrs_[ii] = new TreeNode();
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
SumOfTrees::~SumOfTrees()
{
  if (treePtrs_ != NULL)
  {
    for (int ii = 0; ii < numTrees_; ii++)
      if (treePtrs_[ii] != NULL) delete treePtrs_[ii];
  }
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------
int SumOfTrees::initialize(double *X, double *Y)
{
  //**/ ---------------------------------------------------------------
  //**/ clean up and initialize trees
  //**/ ---------------------------------------------------------------
  initTrees();

  //**/ ---------------------------------------------------------------
  //**/ build trees 
  //**/ ---------------------------------------------------------------
  buildTrees(X, Y);
  return 0;
}
  
// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int SumOfTrees::genNDGridData(double *X, double *Y, int *N2, double **XOut, 
                              double **YOut)
{
  int totPts;

  //**/ ---------------------------------------------------------------
  //**/ clean up and initialize trees
  //**/ ---------------------------------------------------------------
  initTrees();

  //**/ ---------------------------------------------------------------
  //**/ build trees 
  //**/ ---------------------------------------------------------------
  buildTrees(X, Y);

  //**/ ---------------------------------------------------------------
  //**/ if requested not to create mesh, just return
  //**/ ---------------------------------------------------------------
  if ((*N2) == -999) return 0;
  
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(N2, XOut);
  if ((*N2) == 0) return 0;
  totPts = (*N2);

  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
  psVector vecYOut;
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  evaluatePoint(totPts, *XOut, *YOut);

  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int SumOfTrees::gen1DGridData(double *X, double *Y, int ind1, 
                              double *settings, int *N, double **XOut, 
                              double **YOut)
{
  int    ii, ss, totPts;
  double HX;
  psVector vecXT, vecXOut, vecYOut;

  //**/ clean up and initialize trees
  initTrees();

  //**/ build trees 
  buildTrees(X, Y);

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*N) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ss = 0; ss < totPts; ss++) 
  {
    vecXT[ss*nInputs_+ind1]  = HX * ss + VecLBs_[ind1];
    (*XOut)[ss] = HX * ss + VecLBs_[ind1];
    (*YOut)[ss] = 0.0;
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int SumOfTrees::gen2DGridData(double *X, double *Y, int ind1, int ind2, 
                              double *settings, int *N, double **XOut, 
                              double **YOut)
{
  int ii, ss, jj, index, totPts;
  psVector vecXT, vecXOut, vecYOut, vecHX;
 
  //**/ clean up and initialize trees
  initTrees();

  //**/ build trees 
  buildTrees(X, Y);

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts*2);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*N) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      index = ii * nPtsPerDim_ + jj;
      vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int SumOfTrees::gen3DGridData(double *X, double *Y, int ind1, int ind2, 
                              int ind3, double *settings, int *N, 
                              double **XOut, double **YOut)
{
  int ii, ss, jj, ll, index, totPts;
  psVector vecXT, vecXOut, vecYOut, vecHX;

  //**/ clean up and initialize trees
  initTrees();

  //**/ build trees 
  buildTrees(X, Y);

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts*3);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*N) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++)
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii];

  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int SumOfTrees::gen4DGridData(double *X, double *Y, int ind1, int ind2, 
                              int ind3, int ind4, double *settings, 
                              int *N, double **XOut, double **YOut)
{
  int ii, ss, jj, ll, mm, index, totPts;
  psVector vecXT, vecXOut, vecYOut, vecHX;

  //**/ clean up and initialize trees
  initTrees();

  //**/ build trees 
  buildTrees(X, Y);

  //**/ set up for generating regular grid data
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1])/(nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2])/(nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3])/(nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4])/(nPtsPerDim_ - 1); 

  //**/ allocate storage for the data points
  vecXOut.setLength(totPts*4);
  (*XOut) = vecXOut.takeDVector();
  vecYOut.setLength(totPts);
  (*YOut) = vecYOut.takeDVector();
  (*N) = totPts;

  //**/ allocate local storage for the data points
  vecXT.setLength(totPts*nInputs_);
  for (ss = 0; ss < totPts; ss++) 
    for (ii = 0; ii < nInputs_; ii++) 
      vecXT[ss*nInputs_+ii] = settings[ii]; 
    
  //**/ generate the data points 
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++)
    {
      for (ll = 0; ll < nPtsPerDim_; ll++)
      {
        for (mm = 0; mm < nPtsPerDim_; mm++)
        {
          index = ii*nPtsPerDim_*nPtsPerDim_ * nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
          vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
          vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
          vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
          vecXT[index*nInputs_+ind4] = vecHX[3] * mm + VecLBs_[ind4];
          (*XOut)[index*4]   = vecHX[0] * ii + VecLBs_[ind1];
          (*XOut)[index*4+1] = vecHX[1] * jj + VecLBs_[ind2];
          (*XOut)[index*4+2] = vecHX[2] * ll + VecLBs_[ind3];
          (*XOut)[index*4+3] = vecHX[3] * mm + VecLBs_[ind4];
        }
      }
    }
  }

  //**/ evaluate 
  evaluatePoint(totPts, vecXT.getDVector(), *YOut);
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double SumOfTrees::evaluatePoint(double *X)
{
  int    ii;
  double Y=0.0;

  for (ii = 0; ii < numTrees_; ii++) Y += evaluateTree(treePtrs_[ii], X);
  //**/ boostrap mode: do average
  Y /= (double) numTrees_;
  return Y;
}

// ************************************************************************
// Evaluate a number of points
// ------------------------------------------------------------------------
double SumOfTrees::evaluatePoint(int npts, double *X, double *Y)
{
  int ii, ss;

  for (ss = 0; ss < npts; ss++)
  {
    Y[ss] = 0.0;
    for (ii = 0; ii < numTrees_; ii++)
      Y[ss] += evaluateTree(treePtrs_[ii], &X[ss*nInputs_]);
    //**/ boostrap mode: do average
    if (mode_ == 0) Y[ss] /= (double) numTrees_;
  }
  return 0.0;
}

// ************************************************************************
// Evaluate a given point with standard deviation
// ------------------------------------------------------------------------
double SumOfTrees::evaluatePointFuzzy(double *X, double &Ystd)
{
  int    ii;
  double Ymean;
  Ymean = 0.0;
  for (ii = 0; ii < numTrees_; ii++) 
    Ymean += evaluateTree(treePtrs_[ii],X);
  Ymean /= (double) numTrees_;
  Ystd = 0.0;
  for (ii = 0; ii < numTrees_; ii++) 
    Ystd += pow(evaluateTree(treePtrs_[ii], X) - Ymean, 2.0);
  Ystd = sqrt(Ystd/(numTrees_-1.0));
  return Ymean;
}

// ************************************************************************
// Evaluate a number of points with standard deviations
// ------------------------------------------------------------------------
double SumOfTrees::evaluatePointFuzzy(int npts, double *X, double *Y,
                                      double *Ystd)
{
  for (int ss = 0; ss < npts; ss++) 
    Y[ss] = evaluatePointFuzzy(&X[ss*nInputs_], Ystd[ss]);
  return 0.0;
}

// ************************************************************************
// initialize the trees
// ------------------------------------------------------------------------
int SumOfTrees::initTrees()
{
  int    ii;

  //**/ clean up, if needed
  if (psConfig_.InteractiveIsOn()) 
    printf("SumOfTrees: initializing trees.\n");
  if (treePtrs_ != NULL)
  {
    for (ii = 0; ii < numTrees_; ii++) 
    {
      if (treePtrs_[ii] != NULL) delete treePtrs_[ii];
      treePtrs_[ii] = new TreeNode();
    }
  }
  else
  {
    treePtrs_ = new TreeNode*[numTrees_];
    for (ii = 0; ii < numTrees_; ii++)
      treePtrs_[ii] = new TreeNode();
  }
  return 0;
}   

// ************************************************************************
// build a number of trees 
// ------------------------------------------------------------------------
int SumOfTrees::buildTrees(double *X, double *Y)
{
  int    jj, kk, mm, ss;
  double mult, checksum;
  FILE   *fp=NULL;

  //**/ build trees 
  if (psConfig_.RSCodeGenIsOn()) fp  = fopen("psuade_rs.info", "w");
  if (fp != NULL)
  {
    fprintf(fp,"This file contains information to re-construct sum-of-trees\n");
    fprintf(fp,"response surface offline. Follow the steps below:\n");
    fprintf(fp,"1. Search for the keywords 'SPLIT HERE' in this file.\n");
    fprintf(fp,"2. Store the lines below keywords into main.cpp\n");
    fprintf(fp,"3. Compile main.cpp (g++ -o main main.cpp -lm) and run\n");
    fprintf(fp,"\n");
    fprintf(fp,"PSUADE_BEGIN\n");
    fprintf(fp,"NT = %d\n", numTrees_);
    fclose(fp);
  }
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.py", "w");
  if (fp != NULL)
  {
    fwriteRSPythonHeader(fp);
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"# Sum-of-trees interpolation\n");
    fprintf(fp,"#==================================================\n");
    fwriteRSPythonCommon(fp);
    fclose(fp);
  }
  //**/ build trees 
  mult = shrinkFactor_;
  if (psConfig_.InteractiveIsOn()) 
    printf("SumOfTrees: building %d trees.\n", numTrees_);

  psVector vecXT, vecYT, vecYY;
#pragma omp parallel shared(X) \
  private(mm,jj,ss,kk,vecXT,vecYT,vecYY,checksum,mult)
{
#pragma omp for
  for (mm = 0; mm < numTrees_; mm++)
  {
    if (psConfig_.InteractiveIsOn() && outputLevel_ > 3) 
      printf("   building tree #%d\n",mm+1);

    vecXT.setLength(nSamples_*nInputs_);
    vecYT.setLength(nSamples_);
    vecYY.setLength(nSamples_);
    for (jj = 0; jj < nSamples_; jj++) vecYY[jj] = Y[jj];

#ifndef PSUADE_OMP
    if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    {
      fp = fopen("psuade_rs.info", "a");
      if (fp != NULL)
      {
        fprintf(fp, "TREE %d\n", mm);
        fclose(fp);
      }
      fp = fopen("psuade_rs.py", "a");
      if (fp != NULL)
      {
        fprintf(fp, "T%d = [\n", mm);
        fclose(fp);
      }
    }
#endif
    //**/ bagging mode: create a boostrap aggregate
    if (mode_ == 0)
    {
      //**/ generate an aggregate
      for (ss = 0; ss < nSamples_; ss++)
      {
        //**/ using PSUADE_rand will break OpenMP
        //kk = PSUADE_rand() % nSamples_;
        kk = lrand48() % nSamples_;
        for (jj = 0; jj < nInputs_; jj++)
          vecXT[ss*nInputs_+jj] = X[kk*nInputs_+jj];
        vecYT[ss] = Y[kk];
      }
    }
    //**/ boosting mode: use residual
    else
    {
      //**/ copy the predictor values
      for (jj = 0; jj < nSamples_*nInputs_; jj++) vecXT[jj] = X[jj];
      //**/ copy the result values
      for (jj = 0; jj < nSamples_; jj++) vecYT[jj] = vecYY[jj];
      //**/ generate an aggregate based on the current vecYT
    }
    //**/ build tree
    buildOneTree(treePtrs_[mm], nSamples_, vecXT.getDVector(), 
                 vecYT.getDVector(), 0);
    //**/ boosting mode: use residual
    if (mode_ == 1)
    {
      //**/ evaluate 
      for (jj = 0; jj < nSamples_; jj++) 
        vecYY[jj] -= (mult * evaluateTree(treePtrs_[mm], &X[jj*nInputs_]));
      checksum = 0.0;
      for (jj = 0; jj < nSamples_; jj++) checksum += vecYY[jj] * vecYY[jj];
      if (psConfig_.InteractiveIsOn())
        printf("SumOfTree boosting at iteration %d = %e\n", mm+1, 
               checksum/nSamples_);
      mult = shrinkFactor_ * mult / (shrinkFactor_ - mult * mult);
      if (mm == numTrees_-1) mult = 1.0;
    }
#ifndef PSUADE_OMP
    if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    {
      fp = fopen("psuade_rs.info", "a");
      if (fp != NULL)
      {
        fprintf(fp, "TREE %d\n", mm);
        fclose(fp);
      }
      fp = fopen("psuade_rs.py", "a");
      if (fp != NULL)
      {
        fprintf(fp, "]\n");
        fclose(fp);
      }
    }
#endif
  }
} /* omp */

#ifdef PSUADE_OMP
  printf("SumOfTrees INFO: No RS code created in OMP mode.\n"); 
#else
  fp = NULL;
  if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn()) 
    fp = fopen("psuade_rs.info", "a");
  if (fp != NULL)
  {
    fprintf(fp,"====================== SPLIT HERE =====================\n");
    fprintf(fp,"/* *******************************************/\n");
    fprintf(fp,"/* User regression interpolator form PSUADE. */\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"#include <math.h>\n");
    fprintf(fp,"#include <stdlib.h>\n");
    fprintf(fp,"#include <stdio.h>\n");
    fprintf(fp,"#include <string.h>\n");
    fprintf(fp,"// ============================================\n");
    fprintf(fp,"// class definition\n");
    fprintf(fp,"// ============================================\n");
    fprintf(fp,"class TreeNode {\n");
    fprintf(fp,"public:\n");
    fprintf(fp,"   TreeNode *leftNode_;\n");
    fprintf(fp,"   TreeNode *rightNode_;\n");
    fprintf(fp,"   int      whichInput_;\n");
    fprintf(fp,"   double   cutPoint_;\n");
    fprintf(fp,"   double   nodeValue_;\n");
    fprintf(fp,"   double   nodeStdev_;\n");
    fprintf(fp,"   TreeNode() {};\n");
    fprintf(fp,"   ~TreeNode() {};\n");
    fprintf(fp,"};\n");
    fprintf(fp,"int    initialize(int *, TreeNode ***);\n");
    fprintf(fp,"int    buildOneTree(FILE *, TreeNode *);\n");
    fprintf(fp,"double interpolate(int,TreeNode **,double *,double &);\n");
    fprintf(fp,"// ============================================\n");
    fprintf(fp,"// main program\n");
    fprintf(fp,"// ============================================\n");
    fprintf(fp,"main(int argc, char **argv) {\n");
    fprintf(fp,"  int    ii,nInps,nTrees;\n");
    fprintf(fp,"  double X[%d], Y, Std;\n",nInputs_);
    fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
    fprintf(fp,"  TreeNode **treePtrs;\n");
    fprintf(fp,"  if (argc < 3) {\n");
    fprintf(fp,"     printf(\"ERROR: not enough argument.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fIn = fopen(argv[1], \"r\");\n");
    fprintf(fp,"  if (fIn == NULL) {\n");
    fprintf(fp,"     printf(\"ERROR: cannot open input file.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fscanf(fIn, \"%%d\", &nInps);\n");
    fprintf(fp,"  if (nInps != %d) {\n", nInputs_);
    fprintf(fp,"    printf(\"ERROR - wrong nInputs.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  for (ii=0;ii<%d;ii++) fscanf(fIn,\"%%lg\",&X[ii]);\n",
            nInputs_);
    fprintf(fp,"  fclose(fIn);\n");
    fprintf(fp,"  initialize(&nTrees,&treePtrs);\n");
    fprintf(fp,"  Y = interpolate(nTrees,treePtrs, X, Std);\n");
    fprintf(fp,"  printf(\"Y = %%e (stdev = %%e)\\n\", Y, Std);\n");
    fprintf(fp,"  fOut = fopen(argv[2], \"w\");\n");
    fprintf(fp,"  if (fOut == NULL) {\n");
    fprintf(fp,"    printf(\"ERROR: cannot open output file.\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fprintf(fOut,\" %%e\\n\", Y);\n");
    fprintf(fp,"  fclose(fOut);\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* *******************************************/\n");
    fprintf(fp,"/* initialize */\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"int initialize(int *nTrees, TreeNode ***treePtrs) {\n");
    fprintf(fp,"  int  ntrees, ii, tind, status;\n");
    fprintf(fp,"  char line[1001], word[500], equal[5];\n");
    fprintf(fp,"  FILE *fp=NULL;\n");
    fprintf(fp,"  TreeNode **treeptrs;\n");
    fprintf(fp,"  fp = fopen(\"psuade_rs.info\",\"r\");\n");
    fprintf(fp,"  if (fp == NULL){\n");
    fprintf(fp,"     printf(\"Data file (psuade_rs.info) not found.\\n\");\n");
    fprintf(fp,"     exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  while (1) {\n");
    fprintf(fp,"    fgets(line, 500, fp);\n");
    fprintf(fp,"    sscanf(line, \"%%s\",word);\n");
    fprintf(fp,"    if (!strcmp(word, \"PSUADE_BEGIN\")) break;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fscanf(fp,\"%%s %%s %%d\", word, equal, nTrees);\n");
    fprintf(fp,"  if (strcmp(word, \"NT\") && equal[0] != '=') {\n");
    fprintf(fp,"    printf(\"ERROR: reading tree info (0)\\n\");\n");
    fprintf(fp,"    exit(1); }\n");
    fprintf(fp,"  treeptrs = new TreeNode*[ntrees];\n");
    fprintf(fp,"  for (ii = 0; ii < *nTrees; ii++) {\n");
    fprintf(fp,"    fscanf(fp, \"%%s %%d\", word, &tind);\n");
    fprintf(fp,"    if (tind != ii) {\n");
    fprintf(fp,"      printf(\"ERROR: reading tree info (1)\\n\");\n");
    fprintf(fp,"      exit(1); }\n");
    fprintf(fp,"    treeptrs[ii] = new TreeNode();\n");
    fprintf(fp,"    status = buildOneTree(fp, treeptrs[ii]);\n");
    fprintf(fp,"    if (status != 0) {\n");
    fprintf(fp,"      printf(\"ERROR: reading tree info (2)\\n\");\n");
    fprintf(fp,"      exit(1); }\n");
    fprintf(fp,"    fscanf(fp, \"%%s %%d\", word, &tind);\n");
    fprintf(fp,"    if (tind != ii) {\n");
    fprintf(fp,"      printf(\"ERROR: reading tree info (3)\\n\");\n");
    fprintf(fp,"      exit(1); }\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  fclose(fp);\n");
    fprintf(fp,"  (*treePtrs) = treeptrs;\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* *******************************************/\n");
    fprintf(fp,"/* build trees */\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"int buildOneTree(FILE *fp, TreeNode *treePtr) {\n");
    fprintf(fp,"  int    inp, level; double ddata;\n");
    fprintf(fp,"  char   word1[1001], colon[5], word2[1001], equal[5];\n");
    fprintf(fp,"  fscanf(fp,\"%%s %%d %%s %%s\",word1,&level,equal,word2);\n");
    fprintf(fp,"  if (strcmp(word1, \"Level\")) {\n");
    fprintf(fp,"    printf(\"ERROR: reading tree info\\n\");\n");
    fprintf(fp,"    exit(1);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  if (word2[0] == 'N') {\n");
    fprintf(fp,"    treePtr->leftNode_ = NULL;\n");
    fprintf(fp,"    treePtr->rightNode_ = NULL;\n");
    fprintf(fp,"    treePtr->whichInput_ = -1;\n");
    fprintf(fp,"    fscanf(fp,\"%%s %%d %%s %%s %%s %%lg\",word1,&level,\n");
    fprintf(fp,"           colon,word2,equal,&ddata);\n");
    fprintf(fp,"    treePtr->nodeValue_ = ddata;\n");
    fprintf(fp,"    fscanf(fp,\"%%s %%d %%s %%s %%s %%lg\",word1,&level,\n");
    fprintf(fp,"           colon,word2,equal,&ddata);\n");
    fprintf(fp,"    treePtr->nodeStdev_ = ddata;\n");
    fprintf(fp,"    return 0;\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  if (word2[0] == 'C') {\n");
    fprintf(fp,"    fscanf(fp,\"%%s %%lg\",word1,&ddata);\n");
    fprintf(fp,"    treePtr->cutPoint_ = ddata;\n");
    fprintf(fp,"    fscanf(fp,\"%%s %%d %%s %%s %%s %%d\",word1,&level,\n");
    fprintf(fp,"           colon,word2,equal,&inp);\n");
    fprintf(fp,"    treePtr->whichInput_ = inp;\n");
    fprintf(fp,"    treePtr->leftNode_ = new TreeNode();\n");
    fprintf(fp,"    buildOneTree(fp, treePtr->leftNode_);\n");
    fprintf(fp,"    treePtr->rightNode_ = new TreeNode();\n");
    fprintf(fp,"    buildOneTree(fp, treePtr->rightNode_);\n");
    fprintf(fp,"  }\n");
    fprintf(fp,"  return 0;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"/* *******************************************/\n");
    fprintf(fp,"/* evaluate */\n");
    fprintf(fp,"/* ==========================================*/\n");
    fprintf(fp,"double evaluateTree(TreeNode *tptr, double *X) {\n");
    fprintf(fp,"   int    ind; double Y=0.0;\n");
    fprintf(fp,"   if (tptr->whichInput_ < 0) Y = tptr->nodeValue_;\n");
    fprintf(fp,"   else {\n");
    fprintf(fp,"      ind = tptr->whichInput_;\n");
    fprintf(fp,"      if (X[ind] <= tptr->cutPoint_)\n");
    fprintf(fp,"           Y = evaluateTree(tptr->leftNode_, X);\n");
    fprintf(fp,"      else Y = evaluateTree(tptr->rightNode_, X);\n");
    fprintf(fp,"   }\n");
    fprintf(fp,"   return Y;\n");
    fprintf(fp,"}\n");
    fprintf(fp,"double interpolate(int nTrees, TreeNode **treePtrs,\n");
    fprintf(fp,"                   double *X, double &Ystd) {\n");
    fprintf(fp,"   int    ii; double Ymean=0.0;\n");
    fprintf(fp,"   for (ii = 0; ii < nTrees; ii++)\n");
    fprintf(fp,"      Ymean += evaluateTree(treePtrs[ii],X);\n");
    fprintf(fp,"   Ymean /= (double) nTrees;\n");
    fprintf(fp,"   Ystd = 0.0;\n");
    fprintf(fp,"   for (ii = 0; ii < nTrees; ii++)\n");
    fprintf(fp,"      Ystd += pow(evaluateTree(treePtrs[ii],X)-Ymean,2.0);\n");
    fprintf(fp,"   Ystd = sqrt(Ystd/(nTrees-1.0));\n");
    fprintf(fp,"   return Ymean;\n");
    fprintf(fp,"}\n");
    fclose(fp);
    printf("FILE psuade_rs.info contains sum-of-trees information.\n");
  }
  fp = NULL;
  if (psConfig_.RSCodeGenIsOn()) fp = fopen("psuade_rs.py", "a");
  if (fp != NULL)
  {
    fprintf(fp,"nInputs = %d\n", nInputs_);
    fprintf(fp,"nTrees = %d\n", numTrees_);
    fprintf(fp,"Trees = []\n");
    fprintf(fp,"T = []\n");
    for (mm = 0; mm < numTrees_; mm++) fprintf(fp,"T.append(T%d)\n",mm);
    fprintf(fp,"###################################################\n");
    fprintf(fp,"class treeNode :\n");
    fprintf(fp,"  def __init__(self,level=0,tlist=None,listInd=0,listLen=0):\n");
    fprintf(fp,"    self.level = level\n");
    fprintf(fp,"    self.index = 0\n");
    fprintf(fp,"    self.whichInput = -1\n");
    fprintf(fp,"    self.cutPoint = 0\n");
    fprintf(fp,"    self.left = None\n");
    fprintf(fp,"    self.right = None\n");
    fprintf(fp,"    self.nodeValue  = 0.0\n");
    fprintf(fp,"    self.nodeStdev  = 0.0\n");
    fprintf(fp,"    dataLen = listLen\n");
    fprintf(fp,"    if listLen == 0 : \n");
    fprintf(fp,"      dataLen = len(tlist) - listInd\n");
    fprintf(fp,"    if listInd < dataLen : \n");
    fprintf(fp,"      item = tlist[listInd]\n");
    fprintf(fp,"      lvl  = item[0]\n");
    fprintf(fp,"      typ  = item[1]\n");
    fprintf(fp,"      val  = item[2]\n");
    fprintf(fp,"      if lvl == self.level : \n");
    fprintf(fp,"        if typ == 'N' : \n");
    fprintf(fp,"          self.whichInput = -1\n");
    fprintf(fp,"          item1 = tlist[listInd+1]\n");
    fprintf(fp,"          self.nodeValue  = item1[2]\n");
    fprintf(fp,"          item2 = tlist[listInd+2]\n");
    fprintf(fp,"          self.nodeStdev  = item2[2]\n");
    fprintf(fp,"          self.left  = None\n");
    fprintf(fp,"          self.right = None\n");
    fprintf(fp,"          self.cutPoint = 0.0\n");
    fprintf(fp,"          self.index = 3\n");
    fprintf(fp,"        elif typ == 'C' : \n");
    fprintf(fp,"          self.cutPoint = item[2]\n");
    fprintf(fp,"          item1 = tlist[listInd+1]\n");
    fprintf(fp,"          self.whichInput = item1[2]\n");
    fprintf(fp,"          self.nodeValue  = 0.0\n");
    fprintf(fp,"          self.nodeStdev  = 0.0\n");
    fprintf(fp,"          self.left  = treeNode(level+1,tlist,listInd+2,dataLen)\n");
    fprintf(fp,"          depth1 = self.left.getIndex()\n");
    fprintf(fp,"          self.right = treeNode(level+1,tlist,listInd+depth1+2,dataLen)\n");
    fprintf(fp,"          depth2 = self.right.getIndex()\n");
    fprintf(fp,"          self.index = depth1 + depth2 + 2\n");
    fprintf(fp,"  def getIndex(self): \n");
    fprintf(fp,"    return self.index\n");
    fprintf(fp,"  def evaluate(self, X): \n");
    fprintf(fp,"    inp = self.whichInput\n");
    fprintf(fp,"    if inp < 0 : \n");
    fprintf(fp,"      return(self.nodeValue)\n");
    fprintf(fp,"    else : \n");
    fprintf(fp,"      if X[inp] <= self.cutPoint :\n");
    fprintf(fp,"        return self.left.evaluate(X)\n");
    fprintf(fp,"      else : \n");
    fprintf(fp,"        return self.right.evaluate(X)\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# Interpolation function  \n");
    fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
    fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
    fprintf(fp,"# ... \n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"def interpolate(XX): \n");
    fprintf(fp,"  nSamp = int(len(XX) / nInputs + 1.0e-8)\n");
    fprintf(fp,"  Xt = nInputs * [0.0]\n");
    fprintf(fp,"  Yt = nTrees * [0.0]\n");
    fprintf(fp,"  Ys = 2 * nSamp * [0.0]\n");
    fprintf(fp,"  for ss in range(nSamp) : \n");
    fprintf(fp,"    for ii in range(nInputs) : \n");
    fprintf(fp,"      Xt[ii] = XX[ss*nInputs+ii]\n");
    fprintf(fp,"    for tt in range(nTrees) : \n");
    fprintf(fp,"      Yt[tt] = Trees[tt].evaluate(Xt)\n");
    fprintf(fp,"    Ymean = 0.0\n");
    fprintf(fp,"    for tt in range(nTrees) : \n");
    fprintf(fp,"      Ymean = Ymean + Yt[tt]\n");
    fprintf(fp,"    Ymean = Ymean / nTrees\n");
    fprintf(fp,"    Ystd = 0.0\n");
    fprintf(fp,"    for tt in range(nTrees) : \n");
    fprintf(fp,"      Ystd = Ystd + (Yt[tt] - Ymean) * (Yt[tt] - Ymean)\n");
    fprintf(fp,"    Ystd = math.sqrt(Ystd / (nTrees - 1.0))\n");
    fprintf(fp,"    Ys[ss*2] = Ymean\n");
    fprintf(fp,"    Ys[ss*2+1] = Ystd\n");
    fprintf(fp,"  return Ys\n");
    fprintf(fp,"###################################################\n");
    fprintf(fp,"# main program\n");
    fprintf(fp,"#==================================================\n");
    fprintf(fp,"infileName  = sys.argv[1]\n");
    fprintf(fp,"outfileName = sys.argv[2]\n");
    fprintf(fp,"inputs = getInputData(infileName)\n");
    fprintf(fp,"for ii in range(nTrees):\n");
    fprintf(fp,"  tree = treeNode(0,T[ii])\n");
    fprintf(fp,"  Trees.append(tree)\n");
    fprintf(fp,"outputs = interpolate(inputs)\n");
    fprintf(fp,"genOutputFile(outfileName, outputs)\n");
    fprintf(fp,"###################################################\n");
    fclose(fp);
    printf("FILE psuade_rs.py contains the sum-of-trees interpolator.\n");
  }
#endif
  return 0;
}

// ************************************************************************
// build a tree 
// ------------------------------------------------------------------------
int SumOfTrees::buildOneTree(TreeNode *tnode, int leng, double *XT, 
                             double *YT, int level)
{
  int    ii, jj, kk, minInd, repeatCnt, inpImpurity, locImpurity;
  double mean1, mean2, sosSum, sosMin, Ymax;
  double lmean1, lmean2, sos1, sos2, sos, mean, maxImpurity, ddata;
  FILE   *fp;

  //**/ if node less than minimum size, stop branching
  if (leng < 2*minPtsPerNode_)
  {
    tnode->leftNode_ = NULL;
    tnode->rightNode_ = NULL;
    tnode->whichInput_ = -1;
    mean = 0.0;
    for (jj = 0; jj < leng; jj++) mean += YT[jj];
    if (leng > 0) mean /= (double) leng;
    else          mean = 0.0;
    tnode->nodeValue_ = mean;
    ddata = 0.0;
    for (jj = 0; jj < leng; jj++) ddata += pow(YT[jj]-mean,2.0);
    tnode->nodeStdev_ = sqrt(ddata/leng);
    if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    {
      fp = fopen("psuade_rs.info", "a");
      if (fp != NULL)
      {
        fprintf(fp, "Level %d : N\n", level);
        fprintf(fp, "Level %d : M = %e\n", level, mean);
        fprintf(fp, "Level %d : S = %e\n", level, tnode->nodeStdev_);
        fclose(fp);
      }
      fp = fopen("psuade_rs.py", "a");
      if (fp != NULL)
      {
        fprintf(fp, "[ %d , 'N', 0.0 ],\n", level);
        fprintf(fp, "[ %d , 'M', %e ],\n", level, mean);
        fprintf(fp, "[ %d , 'S', %e ],\n", level, tnode->nodeStdev_);
        fclose(fp);
      }
    }
    return -1;
  }

  //**/ scale to prevent overflow
  Ymax = PABS(YT[0]);
  for (jj = 1; jj < leng; jj++) 
    if (PABS(YT[jj]) > Ymax) Ymax = PABS(YT[jj]);
  if (Ymax == 0.0) Ymax = 1.0;
      
  //**/ compute aggregate sum of variances
  psVector vecXX, vecYY;
  vecXX.setLength(leng*nInputs_);
  vecYY.setLength(leng);
  for (jj = 0; jj < leng; jj++) vecYY[jj] = YT[jj] / Ymax;
  mean = sos = 0.0;
  for (jj = 0; jj < leng; jj++) mean += vecYY[jj];
  mean /= leng;
  for (jj = 0; jj < leng; jj++) 
    sos += (vecYY[jj]-mean)*(vecYY[jj]-mean);
  sos /= leng;

  //**/ search which input gives maximum impurity decrease
  maxImpurity = - PSUADE_UNDEFINED;
  inpImpurity = -1;
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < leng; jj++)
    {
      vecXX[jj] = XT[jj*nInputs_+ii];
      vecYY[jj] = YT[jj] / Ymax;
    }
    //**/ sort based on the given coordinate
    sortDbleList2(leng, vecXX.getDVector(), vecYY.getDVector());
    //**/ compute the initial sums of squares
    mean1 = sos1 = 0.0;
    for (jj = 0; jj < 2; jj++) mean1 += vecYY[jj];
    mean1 /= 2.0;
    for (jj = 0; jj < 2; jj++) 
      sos1 += (vecYY[jj] - mean1) * (vecYY[jj] - mean1);
    mean2 = sos2 = 0.0;
    for (jj = 2; jj < leng; jj++) mean2 += vecYY[jj];
    mean2 /= (leng - 2.0);
    for (jj = 2; jj < leng; jj++)
      sos2 += (vecYY[jj] - mean2) * (vecYY[jj] - mean2);
    sosSum = (sos1 + sos2) / leng;
    sosMin = sosSum;
    minInd = 1;
    repeatCnt = 0;
    //**/ search for minimum sum
    for (jj = 2; jj < leng - 2; jj++)
    {
      lmean1 = mean1;
      lmean2 = mean2;
      mean1 = (mean1 * jj + vecYY[jj]) / (jj + 1.0);
      mean2 = (mean2 * (leng - jj) - vecYY[jj]) / (leng-jj-1.0);
      sos1 = sos1 + lmean1 * lmean1 * jj + vecYY[jj] * vecYY[jj] - 
             (jj + 1.0) * mean1 * mean1;
      sos2 = sos2 + lmean2 * lmean2 * (leng - jj) - vecYY[jj] * 
             vecYY[jj] - (leng - jj - 1.0) * mean2 * mean2;
#if 0
//**/ direct calculation instead of iteration
//**/ mean1 = sos1 = 0.0;
//**/ for (kk = 0; kk <= jj; kk++) mean1 += vecYY[kk];
//**/ mean1 /= (jj + 1.0);
//**/ for (kk = 0; kk <= jj; kk++) 
//**/   sos1 += (vecYY[kk]-mean1) * (vecYY[kk]-mean1);
//**/ mean2 = sos2 = 0.0;
//**/ for (kk = jj+1; kk < leng; kk++) mean2 += vecYY[kk];
//**/ mean2 /= (leng - jj - 1.0);
//**/ for (kk = jj+1; kk < leng; kk++)
//**/    sos2 += (vecYY[kk]-mean2)*(vecYY[kk]-mean2);
#endif
      sosSum = (sos1 + sos2) / leng;
      if (sosSum == sosMin) repeatCnt++;
      else                  repeatCnt = 0; 
      if (sosSum < sosMin) 
      {
        sosMin = sosSum;
        minInd = jj;
      }
    }
    //printf("SumOfTree: input = %3d, max decrease in impurity at %d (%d)\n",
    //       ii+1, minInd+repeatCnt/2,leng);
    //printf("SumOfTree: input = %3d, actual decrease in impurity = %e\n",
    //       ii+1, sos - sosMin);
    ddata = sos - sosMin;
    if (ddata > maxImpurity)
    {
      maxImpurity = ddata;
      inpImpurity = ii;
      locImpurity = minInd + repeatCnt / 2;
    }
  }
  if (psConfig_.InteractiveIsOn() && outputLevel_ >= 2)
  {
    printf("SumOfTree: level = %d, leng = %d, input selected = %d ",
           level, leng, inpImpurity+1);
    printf("impurity = %e\n", maxImpurity);
  }

  //**/ check for convergence
  if (maxImpurity / sos < tolerance_) inpImpurity = -1;

  //**/ reorder the array before going down the tree
  if (inpImpurity >= 0)
  {
    for (jj = 0; jj < leng; jj++)
    {
      vecXX[jj] = XT[jj*nInputs_+inpImpurity];
      vecYY[jj] = 1.0 * jj;
    }
    sortDbleList2(leng, vecXX.getDVector(), vecYY.getDVector());
    for (jj = 0; jj < leng; jj++)
    {
      kk = (int) (vecYY[jj] + 0.0001);
      for (ii = 0; ii < nInputs_; ii++)
        vecXX[jj*nInputs_+ii] = XT[kk*nInputs_+ii];
      vecYY[jj] = YT[kk];
    }
    for (jj = 0; jj < leng*nInputs_; jj++) XT[jj] = vecXX[jj];
    for (jj = 0; jj < leng; jj++) YT[jj] = vecYY[jj];

    tnode->cutPoint_ = XT[locImpurity*nInputs_+inpImpurity]; 
    tnode->whichInput_ = inpImpurity; 
    if (psConfig_.InteractiveIsOn() && psConfig_.RSCodeGenIsOn())
    {
      fp = fopen("psuade_rs.info", "a");
      if (fp != NULL)
      {
        fprintf(fp, "Level %d : C = %e\n", level, tnode->cutPoint_);
        fprintf(fp, "Level %d : I = %d\n",level, inpImpurity);
        fclose(fp);
      }
      fp = fopen("psuade_rs.py", "a");
      if (fp != NULL)
      {
        fprintf(fp, "[ %d , 'C',  %e ],\n", level, tnode->cutPoint_);
        fprintf(fp, "[ %d , 'I',  %d ],\n",level, inpImpurity);
        fclose(fp);
      }
    }
    if (psConfig_.InteractiveIsOn() && outputLevel_ >= 3)
      printf("SumTree Level = %d, Cutting input %d at %e\n", level, 
             inpImpurity+1, tnode->cutPoint_);
    //**/ go to next level
    tnode->leftNode_ = new TreeNode();
    buildOneTree(tnode->leftNode_, locImpurity+1, XT, YT, level+1);
    tnode->rightNode_ = new TreeNode();
    buildOneTree(tnode->rightNode_, leng-locImpurity-1, 
                 &XT[(locImpurity+1)*nInputs_],
                 &YT[locImpurity+1], level+1);
  }
  else
  {
    tnode->leftNode_ = NULL;
    tnode->rightNode_ = NULL;
    tnode->whichInput_ = -1;
    mean = 0.0;
    for (jj = 0; jj < leng; jj++) mean += YT[jj];
    if (leng > 0) mean /= (double) leng;
    else          mean = 0.0;
    tnode->nodeValue_ = mean;
    ddata = 0.0;
    for (jj = 0; jj < leng; jj++) ddata += pow(YT[jj]-mean,2.0);
    tnode->nodeStdev_ = sqrt(ddata/leng);
    if (psConfig_.RSCodeGenIsOn())
    {
      fp = fopen("psuade_rs.info", "a");
      if (fp != NULL)
      {
        fprintf(fp, "Level %d : N\n", level);
        fprintf(fp, "Level %d : M = %e\n", level, mean);
        fprintf(fp, "Level %d : S = %e\n", level, tnode->nodeStdev_);
        fclose(fp);
      }
      fp = fopen("psuade_rs.py", "a");
      if (fp != NULL)
      {
        fprintf(fp, "[ %d , 'N', 0.0 ],\n", level);
        fprintf(fp, "[ %d , 'M', %e ],\n", level, mean);
        fprintf(fp, "[ %d , 'S', %e ],\n", level, tnode->nodeStdev_);
        fclose(fp);
      }
    }
  }
  return 0;
}

// ************************************************************************
// get the information about splitting 
// ------------------------------------------------------------------------
double SumOfTrees::tabulateSplits(TreeNode *tptr,double *scores,int level)
{
  double accum=0;
  if (tptr->whichInput_ < 0) return tptr->nodeStdev_;
  scores[tptr->whichInput_] += 1.0 / pow(2.0, 1.0*level);
  if (tptr->leftNode_  != NULL)
    accum += tabulateSplits(tptr->leftNode_,scores,level+1);
  if (tptr->rightNode_ != NULL) 
    accum += tabulateSplits(tptr->rightNode_,scores,level+1);
  return accum;
}

// ************************************************************************
// Evaluate a point for a tree
// ------------------------------------------------------------------------
double SumOfTrees::evaluateTree(TreeNode *tptr, double *X)
{
  int    ind;
  double Y=0.0;
  if (tptr->whichInput_ < 0) 
  {
     Y = tptr->nodeValue_;
     //for (int ii = 0; ii < nInputs_; ii++)
     //   printf("X %2d = %e\n", ii+1, X[ii]);
     //printf("returns Y = %e\n", Y);
  }
  else                        
  {
    ind = tptr->whichInput_;
    if (X[ind] <= tptr->cutPoint_) 
    {
      //printf("Go to X %d (%e) < %e\n", ind, X[ind], tptr->cutPoint_);
      Y = evaluateTree(tptr->leftNode_, X); 
      //printf("returns L Y = %e\n", Y);
    }
    else
    {
      //printf("Go to X %d (%e) > %e\n", ind, X[ind], tptr->cutPoint_);
      Y = evaluateTree(tptr->rightNode_, X); 
      //printf("returns R Y = %e\n", Y);
    }
  }
  return Y;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double SumOfTrees::setParams(int targc, char **targv)
{
  int    mm, ii, *iArray;
  double *inputScores, mmax;
  FILE   *fp;

  if (targc > 0 && !strcmp(targv[0], "mode0"))
  {
    mode_ = 0;
  }
  else if (targc > 0 && !strcmp(targv[0], "mode1"))
  {
    mode_ = 1;
  }
  else if (targc > 0 && !strcmp(targv[0], "rank"))
  {
    double val=0.0;
    psVector vecIScores;
    vecIScores.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++) vecIScores[ii] = 0.0;
    for (mm = 0; mm < numTrees_; mm++)
    {
      val += tabulateSplits(treePtrs_[mm],vecIScores.getDVector(),0);
    }
    val /= (double) numTrees_;
    mmax = 0.0;
    for (ii = 0; ii < nInputs_; ii++)
      if (vecIScores[ii] > mmax) mmax = vecIScores[ii];
    if (mmax > 0)
    {
      for (ii = 0; ii < nInputs_; ii++)
        vecIScores[ii] = vecIScores[ii] / mmax * 100;
    }

    //**/ output to a matlab file
    if (plotScilab()) fp = fopen("scilabsot.sci", "w");
    else              fp = fopen("matlabsot.m", "w");
    fwritePlotCLF(fp);
    fprintf(fp, "A = [\n");
    for (ii = 0; ii < nInputs_; ii++)
       fprintf(fp, "%e\n", 0.01 * vecIScores[ii]);
    fprintf(fp, "];\n");
    fprintf(fp, "bar(A, 0.8);\n");
    fwritePlotAxes(fp);
    fwritePlotTitle(fp, "Sum-of-trees Rankings");
    fwritePlotXLabel(fp, "Input parameters");
    fwritePlotYLabel(fp, "Sum-of-trees Metric (normalized)");
    fclose(fp);
    if (plotScilab())
         printf("Sum-of-trees ranking is now in scilabsot.sci.\n");
    else printf("Sum-of-trees ranking is now in matlabsot.m.\n");

    //**/ re-order for ordered outputs
    psIVector vecIT;
    vecIT.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++) vecIT[ii] = ii;
    sortDbleList2a(nInputs_,vecIScores.getDVector(),
                   vecIT.getIVector());
    if (targc == 1)
    {
      printAsterisks(PL_INFO, 0);
      printf("* SumOfTrees screening rankings ");
      if (mode_ == 0) printf("(with bootstrapping)\n");
      else            printf("(with boosting)\n");
      printf("* Minimum points per node = %d\n", minPtsPerNode_);
      printAsterisks(PL_INFO, 0);
      for (ii = nInputs_-1; ii >= 0; ii--)
        printf("*  Rank %3d : Input = %3d (score = %4.1f)\n",
               nInputs_-ii, vecIT[ii]+1, vecIScores[ii]);
      printAsterisks(PL_INFO, 0);
    }
    return val;
  }
  return 0.0;
}


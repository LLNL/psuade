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
// Ref: S. Lophaven H. B. Nielsen, and J. Sondergaard, "DACE: A Matlab
//      Kriging toolbox," Technical Report IMM-TR-2002-12, Informatics
//      and Mathematical Modelling, Technical University of Denmark.
// ************************************************************************
// Functions for the class Kriging
// AUTHOR : CHARLES TONG
// DATE   : 2012
// ************************************************************************
#ifdef WINDOWS
#include <windows.h>
#endif

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <math.h>
#include <unistd.h>

#include "Kriging.h"
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PsuadeConfig.h"
#include "Sampling.h"
#include "PrintingTS.h"

#define PABS(x) ((x) > 0 ? (x) : (-(x)))

// ************************************************************************
// ************************************************************************
// external functions
// ------------------------------------------------------------------------
extern "C" {
  void dtrsv_(char *, char *, char *, int *, double *, int *, double *, int *);
  void dgeqrf_(int *, int *, double *, int *, double *, double *, int *, int *);
  void dpotrf_(char *, int *, double *, int *, int *);
  void dpotrs_(char *, int *, int *, double *, int *, double *,
               int *, int *);
  void kbobyqa_(int *,int *, double *, double *, double *, double *,
               double *, int *, int *, double*);
  void newuoa_(int *,int *,double *,double *,double *,int *,
               int *,double*);
  void dormqr_(char *, char *, int *, int *, int *, double *, int *, 
               double *, double *, int *, double *, int *, int *);
}

// ************************************************************************
// ************************************************************************
// internal global variables
// ------------------------------------------------------------------------
int KRI_outputLevel=0;
int KRI_iter=-1;
int KRI_nInputs=-1;
int KRI_nSamples=-1;
int KRI_pOrder=-1;
int KRI_terminate=0;
int KRI_noProgressCnt=0;
double KRI_currY=0.0;
double KRI_OptY=0.0;
double KRI_Exponent=2.0;
double KRI_YStd=1.0;
double KRI_nugget=0.0;
psVector KRI_VecXDists;
psMatrix KRI_MatS;
psMatrix KRI_MatM;
psMatrix KRI_MatF;
psMatrix KRI_MatFTmp;
psVector KRI_VecX;
psVector KRI_VecY;
psVector KRI_VecYT;
psVector KRI_VecOptThetas;
psVector KRI_VecDataSD;

// ************************************************************************
// ************************************************************************
// resident function to perform evaluation 
// ------------------------------------------------------------------------
#ifdef __cplusplus
extern "C"
{
#endif
  void *kribobyqaevalfunc_(int *nInps, double *XValues, double *YValue)
  {
    int    nBasis, count, Cleng, ii, jj, kk, status;
    double dist, ddata, ddata2;
    char   uplo='L';
    FILE   *fp=NULL;

    //**/ =======================================================
    //**/ termination signal has been detected
    //**/ =======================================================
    for (ii = 0; ii < KRI_nInputs; ii++)
    {
      if (XValues[ii] < 0) 
      {
        (*YValue) = PSUADE_UNDEFINED;
        return NULL;
      }
    }
    if (KRI_noProgressCnt >= 200)
    {
      if (KRI_outputLevel >= 1)
      {
        //KRI_outputLevel = 0;
        printf("\t*** no progress for more than 200 iterations.\n");
      }
      (*YValue) = PSUADE_UNDEFINED;
      return NULL;
    }
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      fclose(fp);
      printf("Kriging: psuade_stop file found - terminating ...\n");
      (*YValue) = KRI_OptY;
      return NULL;
    }
    if (psConfig_.RSExpertModeIsOn() && KRI_outputLevel > 3)
      printf("\t Kriging : iteration %d\n", KRI_iter);

    //**/ =======================================================
    //**/ fill matrix
    //**/ =======================================================
    //**/ figure out the size of the matrix
    KRI_iter++;
    nBasis = 1;
    if (KRI_pOrder == 1) nBasis = KRI_nInputs + 1;
    Cleng = KRI_nSamples + nBasis;

    //**/ fill in the S matrix
    double *SMat = KRI_MatS.getMatrix1D();
    double *MMat = KRI_MatM.getMatrix1D();
    count = 0;
    for (jj = 0; jj < KRI_nSamples; jj++)
    {
      SMat[jj*KRI_nSamples+jj] = 1.0 + 1e-15 * KRI_nSamples;
      if (KRI_nugget != 0.0) 
        SMat[jj*KRI_nSamples+jj] += KRI_nugget;
      else if (KRI_nInputs == 2) 
        SMat[jj*KRI_nSamples+jj] += 0.01;

      if (KRI_VecDataSD.length() > 0) 
      {
        ddata = pow(KRI_VecDataSD[jj]/KRI_YStd,KRI_Exponent);
        SMat[jj*KRI_nSamples+jj] += ddata;
      }
      for (kk = jj+1; kk < KRI_nSamples; kk++)
      {
        dist = 0.0;
        for (ii = 0; ii < KRI_nInputs; ii++)
          dist += pow(KRI_VecXDists[count*KRI_nInputs+ii]/XValues[ii],
                      KRI_Exponent);
        dist = exp(-dist);
        if (dist < 1.0e-16) dist = 0;
        SMat[jj*KRI_nSamples+kk] = dist;
        SMat[kk*KRI_nSamples+jj] = dist;
        count++;
      }
    }

    //**/ =======================================================
    //**/ Solve the linear system (note: F has been filled in main
    //**/   | S   F | |alpha | = Y
    //**/   | F^T 0 | |beta  | = 0
    //**/ =======================================================
    //**/ -------------------------------------------------------
    //**/ Create F
    //**/ -------------------------------------------------------
    double *FMat = KRI_MatF.getMatrix1D();
    double *FMatTmp = KRI_MatFTmp.getMatrix1D();
    for (jj = 0; jj < KRI_nSamples; jj++)
    {
      FMat[jj] = 1.0;
      FMatTmp[jj] = 1.0;
      if (KRI_pOrder == 1)
      {
        for (kk = 1; kk < KRI_nInputs+1; kk++)
        {
          ddata = KRI_VecX[jj*KRI_nInputs+kk-1];
          FMatTmp[kk*KRI_nSamples+jj] = ddata;
          FMat[kk*KRI_nSamples+jj] = ddata;
        }
      }
    }
    //**/ -------------------------------------------------------
    //**/ S^{-1} F ==> FMatTmp
    //**/ -------------------------------------------------------
    dpotrf_(&uplo, &KRI_nSamples, SMat, &KRI_nSamples, &status);
    kk = nBasis;
    dpotrs_(&uplo, &KRI_nSamples, &kk, SMat, &KRI_nSamples,
            FMatTmp, &KRI_nSamples, &status);
    //**/ -------------------------------------------------------
    //**/ M = F^T S^{-1} F = F^T FMatTmp, create inv(M)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < nBasis; jj++)
    {
      for (kk = 0; kk < nBasis; kk++)
      {
        ddata = 0.0;
        for (ii = 0; ii < KRI_nSamples; ii++)
          ddata += FMat[ii+jj*KRI_nSamples]*
                   FMatTmp[ii+kk*KRI_nSamples];
        MMat[jj+kk*nBasis] = ddata;
      }
    }
    if (nBasis > 1) dpotrf_(&uplo,&nBasis,MMat,&nBasis,&status);
    //**/ -------------------------------------------------------
    //**/ VecYT = S^{-1} Y (Y is the output vector)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < KRI_nSamples; jj++) KRI_VecYT[jj] = KRI_VecY[jj];
    kk = 1;
    dpotrs_(&uplo,&KRI_nSamples,&kk,SMat,&KRI_nSamples,
            KRI_VecYT.getDVector(), &KRI_nSamples, &status);
    //**/ -------------------------------------------------------
    //**/ VecYT = F^T S^{-1} Y - X = F^T VecYT (X = 0)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < nBasis; jj++)
    {
      ddata = 0.0;
      for (ii = 0; ii < KRI_nSamples; ii++)
        ddata += FMat[ii+jj*KRI_nSamples]*KRI_VecYT[ii];
      KRI_VecYT[KRI_nSamples+jj] = ddata;
    }
    //**/ -------------------------------------------------------
    //**/ VecYT = (F^T S^{-1} F)^{-1} (F^T S^{-1}Y-X) = M^{-1} VecYT
    //**/ beta is now in VecYT
    //**/ -------------------------------------------------------
    if (nBasis == 1)
    {
       if (MMat[0] == 0)
       {
         printf("Kriging ERROR: divide by 0.\n");
         exit(1);
       }
       KRI_VecYT[KRI_nSamples] /= MMat[0];
    }
    else
    {
      kk = 1;
      double *YT = KRI_VecYT.getDVector();
      dpotrs_(&uplo,&kk,&kk,MMat,&nBasis,&YT[KRI_nSamples], 
              &kk, &status);
    }
    //**/ -------------------------------------------------------
    //**/ VecYT = (Y - F beta)
    //**/ -------------------------------------------------------
    for (jj = 0; jj < KRI_nSamples; jj++)
    {
      ddata = 0.0;
      for (ii = 0; ii < nBasis; ii++)
        ddata += FMat[jj+ii*KRI_nSamples]*
                 KRI_VecYT[ii+KRI_nSamples];
      KRI_VecYT[jj] = KRI_VecY[jj] - ddata;
    }
    //**/ -------------------------------------------------------
    //**/ alpha = S^{-1} (Y - F beta) = S^{-1} VecYT
    //**/ now everything is in VecYT
    //**/ -------------------------------------------------------
    kk = 1;
    dpotrs_(&uplo,&KRI_nSamples,&kk,SMat,&KRI_nSamples,
            KRI_VecYT.getDVector(),&KRI_nSamples, &status);

    //**/ =======================================================
    //**/ compute metric
    //**/ =======================================================
    ddata = 0.0;
    for (jj = 0; jj < KRI_nSamples; jj++)
      ddata += KRI_VecYT[jj] * KRI_VecY[jj];
    ddata /= (double) KRI_nSamples;

    for (jj = 0; jj < KRI_nSamples; jj++) 
    {
      ddata2 = SMat[KRI_nSamples*jj+jj];
      if (ddata2 < 0.0) ddata2 = - ddata2;
      ddata *= pow(ddata2, 2.0/(double) KRI_nSamples);
    }

    (*YValue) = KRI_currY = ddata;

    if (ddata < KRI_OptY || KRI_iter == 1)
    {
      KRI_noProgressCnt = 0;
      KRI_OptY = ddata;
      for (ii = 0; ii < KRI_nInputs; ii++)
        KRI_VecOptThetas[ii] = XValues[ii];
      if (KRI_outputLevel > 1)
      {
        printf("\t Kriging : iteration %d\n",KRI_iter);
        for (ii = 0; ii < KRI_nInputs; ii++)
          printf("\t    Current best theta %d = %e\n",ii+1,XValues[ii]);
        printf("\t    Current best objective value = %e\n",ddata);
      }
    }
    else KRI_noProgressCnt++;

    if (psConfig_.RSExpertModeIsOn() && KRI_outputLevel > 3)
    {
      printf("\t Kriging : iteration %d\n", KRI_iter);
      for (ii = 0; ii < KRI_nInputs; ii++)
        printf("\t    Current theta %d = %e\n", ii+1, XValues[ii]);
      printf("\t    Current Kriging objective value = %e\n", ddata);
      printf("\t* For early termination, just create a file\n");
      printf("\t* called 'psuade_stop' in your local directory.\n");
      fp = fopen("psuade_stop", "r");
      if (fp != NULL)
      {
        printf("\t*** psuade_stop file found.\n");
        (*YValue) = 0.0;
        fclose(fp);
      }
    }
    return NULL;
  }
#ifdef __cplusplus
}
#endif

// ************************************************************************
// Constructor for object class Kriging
// ------------------------------------------------------------------------
Kriging::Kriging(int nInputs,int nSamples) : FuncApprox(nInputs,nSamples)
{
  int    ii, jj;
  double ddata;
  char   pString[101], winput[500], winput2[500], fname[500], *strPtr;
  FILE   *fp;

  //**/ =======================================================
  // initialize variables and parameters
  //**/ =======================================================
  faID_ = PSUADE_RS_KR;
  pOrder_  = 0;
  initFlag_ = 0;
  workLength_ = 0;
  optTolerance_ = 1.0e-4;
  fastMode_ = 3;
  VecThetas_.setLength(nInputs_+1);
  for (ii = 0; ii < nInputs_+1; ii++) VecThetas_[ii] = 0.01;
  noReuse_ = 0;  /* 1: no genRSCode even if interactive */

  //**/ =======================================================
  // display banner and additonal information
  //**/ =======================================================
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*                Kriging Analysis\n");
    printf("* Set printlevel to 1-4 to see Kriging details.\n");
    printf("* Turn on rs_expert mode to set slow or fast mode.\n");
    printf("*  + Fast mode: no optimization of hyperparameters.\n");
    printf("*      - turn on rs_expert to set hyperparameters.\n");
    printf("*      - default values = 1.0\n");
    printf("*  + Slow mode : hyperparameters are optimized.\n");
    printf("*      - to change optimization parameters, turn on ");
    printf("rs_expert mode.\n");
    printf("*  + Snail mode (DEFAULT): use multi-start optimization.\n");
    printf("*      - to change optimization parameters, turn on ");
    printf("rs_expert mode.\n");
    printf("* Create 'psuade_stop' file to gracefully terminate.\n");
    printf("* Create 'psuade_print' file to set print level on the fly.\n");
    printEquals(PL_INFO, 0);
  }

  //**/ in RS expert mode, allow the re-use of optimal Kriging 
  //**/ parameters from a file to skip the expensive optimization
  //**/ step. Typically the file is called psuade_kriging_optdata,
  //**/ but it may have been renamed
  if (psConfig_.RSExpertModeIsOn() && psConfig_.InteractiveIsOn())
  {
    fp = fopen("psuade_kriging_optdata","r");
    if (fp != NULL)
    {
      printf("Kriging: psuade_kriging_optdata file found.\n");
      snprintf(pString,100,
               "Use Kriging parameters from the file? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        for (ii = 0; ii < nInputs_; ii++) 
        {
          fscanf(fp,"%lg",&ddata);
          VecThetas_[ii] = ddata; 
        }
        fastMode_ = 1;
      }
      fclose(fp);
    }
    else
    {
      printf("You may have pre-computed optimal Kriging parameter");
      printf(" values previously\n");
      printf("(in the psuade_kriging_optdata file that you might ");
      printf("have renamed to\n");
      printf(" something else)\n");
      snprintf(pString,100, "Do you have such a file ? (y or n) ");
      getString(pString, winput);
      if (winput[0] == 'y')
      {
        snprintf(pString,100, "Name of the file ? ");
        getString(pString, winput);
        winput[strlen(winput)-1] = '\0';
        fp = fopen(winput, "r");
        if (fp != NULL)
        {
          printf("INFO: The optimal Kriging parameter values are ");
          printf("(check correctness): \n");
          for (ii = 0; ii < nInputs_; ii++)
          {
            fscanf(fp,"%lg",&ddata);
            VecThetas_[ii] = ddata;
            printf("Parameter %d = %e\n",ii+1,ddata);
          }
          fastMode_ = 1;
        }
        else
        {
          printf("ERROR: file %s not found.\n", winput);
          printf("       Will continue with optimization.\n");
        }
      }
    }

    //**/ fastMode = 1 at this point means optimal Kriging parameters 
    //**/      have loaded to VecThetas_ from a file. If this did not
    //**/      happen, you have 3 choices:
    //**/      1. enter your preferred values by hand and avoid having
    //**/         to optimize
    //**/      2. use optimization to find optimal parameters
    //**/      3. use multi-start optimization to find optimal parameters
    if (fastMode_ != 1)
    {
      printf("There are three modes available: \n");
      printf("(1) fast mode with pre-specified thetas\n");
      printf("(2) slow mode with optimization on the thetas\n");
      printf("(3) very slow mode with multi-start optimization\n");
      printf("(4) another fast mode using another optimization\n");
      snprintf(pString,100,"Please select mode (1 - 3) : ");
      fastMode_ = getInt(1,4,pString);
      snprintf(pString,100,"RS_KRI_mode = %d", fastMode_);
      psConfig_.putParameter(pString);
      if (fastMode_ == 1)
      {
        for (ii = 0; ii < nInputs_; ii++)
        {
          snprintf(pString,100,
                   "Enter theta for Kriging parameter %d (>0): ", ii+1);
          VecThetas_[ii] = getDouble(pString);
          if (VecThetas_[ii] <= 0.0)
          {
            printf("ERROR: theta <= 0 not valid.\n");
            exit(1);
          }
          snprintf(pString,100,"RS_KRI_LENG_SCALE %d = %e",ii+1,
                   VecThetas_[ii]);
          psConfig_.putParameter(pString);
          if (ii == 0)
          {
            snprintf(pString,100,"Use %e for all other thetas? (y or n) ",
                    VecThetas_[0]);
            getString(pString, winput);
            if (winput[0] == 'y')
            {
              for (jj = 1; jj < nInputs_; jj++) 
              {
                VecThetas_[jj] = VecThetas_[0];
                snprintf(pString,100,"RS_KRI_LENG_SCALE %d = %e",jj+1,
                         VecThetas_[0]);
                psConfig_.putParameter(pString);
              }
              break;
            }
          }
        }
      }
      else
      {
        snprintf(pString,100,"Enter optimization tolerance (default = 1e-4): ");
        optTolerance_ = getDouble(pString);
        if (optTolerance_ <= 0 || optTolerance_ > 0.5)
        {
          printf("Kriging INFO: optimization tol should be in (0,0.5]).\n");
          printf("              Tolerance set to default = 1.0e-4.\n");
          optTolerance_ = 1.0e-4;
        }
        snprintf(pString,100,"RS_KRI_tol = %e", optTolerance_);
        psConfig_.putParameter(pString);

        if (fastMode_ == 2)
        {
          printf("Kriging: Current initial parameter values (thetas) are:\n");
          for (ii = 0; ii < nInputs_; ii++)
            printf("     Input %d: %e\n", ii+1, VecThetas_[ii]);
          printf("If good thetas are available as initial guess for ");
          printf("optimization, you\n");
          printf("can enter them below.\n");
          snprintf(pString,100,"Change initial thetas? (y or n) ");
          getString(pString, winput);
          if (winput[0] == 'y')
          {
            for (ii = 0; ii < nInputs_; ii++)
            {
              snprintf(pString,100,"Enter theta for input %d : ", ii+1);
              VecThetas_[ii] = getDouble(pString);
              if (VecThetas_[ii] <= 0.0)
                printf("Warning: theta < 0 not recommended.\n");
              if (ii == 0)
              {
                snprintf(pString,100,"Use %e for all other thetas? (y or n) ",
                        VecThetas_[0]);
                getString(pString, winput);
                if (winput[0] == 'y')
                {
                  for (jj = 1; jj < nInputs_; jj++) 
                    VecThetas_[jj] = VecThetas_[0];
                  break;
                }
              }
            }
          }
        }
      }

      //**/ nugget is a positive small value added to the diagonal of the
      //**/ covariance matrix
      if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
      {
        snprintf(pString,100,"Add nugget? (y or n) ");
        getString(pString, winput);
        if (winput[0] == 'y') 
        {
          KRI_nugget = 1.0;
          while (KRI_nugget >= 1.0 || KRI_nugget < 0.0)
          {
            snprintf(pString,100,"Enter nugget ([0,1)) : ");
            KRI_nugget = getDouble(pString);
          }
          snprintf(pString,100,"RS_KRI_nugget = %e",KRI_nugget);
          psConfig_.putParameter(pString);
        }
      }
    }
  }
  else
  {
    //**/ =======================================================
    // read from configure file, if any 
    //**/ =======================================================

    //**/ get mode
    strPtr = psConfig_.getParameter("RS_KRI_mode");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %d", winput, winput2, &ii);
      if (ii < 2 || ii > 3)
      {
        printf("Kriging INFO: mode from config not valid.\n");
        printf("              mode kept at %d.\n", fastMode_);
      }
      else
      {
        fastMode_ = ii;
        printf("Kriging INFO: mode from config = %d.\n",fastMode_);
      }
    }

    //**/ get tolerance
    strPtr = psConfig_.getParameter("RS_KRI_tol");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, winput2, &optTolerance_);
      if (optTolerance_ < 0.0 || optTolerance_ >= 1.0)
      {
        optTolerance_ = 1.0e-4;
        printf("Kriging INFO: tolerance from config not valid.\n");
        printf("              tolerance kept at %e.\n", optTolerance_);
      }
      else
      {
        printf("Kriging INFO: tolerance from config = %e.\n",
               optTolerance_);
      }
    }

    //**/ get length scales
    strPtr = psConfig_.getParameter("RS_KRI_LENG_SCALE");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %d %s %lg", winput, &ii, winput2, &ddata);
      if (ii < 1 || ii > nInputs_)
      {
        printf("Kriging INFO: Invalid input number for parameter.\n");
        printf("              Input number read = %d.\n", ii);
      }
      else
      {
        VecThetas_[ii-1] = ddata;
        printf("Kriging INFO: Parameter value %d set to %e.\n",
               ii, ddata);
      }
    }

    //**/ get nugget 
    strPtr = psConfig_.getParameter("RS_KRI_nugget");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %lg", winput, winput2, &KRI_nugget);
      if (KRI_nugget >= 1.0 || KRI_nugget < 0.0) KRI_nugget = 0.9;
      printf("Kriging INFO: nugget from config = %e.\n",KRI_nugget);
    }

    //**/ get standard deviations
    strPtr = psConfig_.getParameter("RS_KRI_DATA_STDEV_FILE");
    if (strPtr != NULL)
    {
      sscanf(strPtr, "%s %s %s", winput, winput2, fname);
      fp = fopen(fname, "r");
      if (fp == NULL)
      {
        printf("Kriging INFO: data variance file not found.\n");
      }
      else
      {
        fscanf(fp, "%d", &ii); 
        if (ii != nSamples_)
        {
          printf("Kriging ERROR: stdev file should have %d entries.\n",
                 nSamples_);
          fclose(fp);
        }
        else
        {
          VecDataSD_.setLength(nSamples_);
          for (ii = 0; ii < nSamples_; ii++)
          {
            fscanf(fp, "%d %lg", &jj, &ddata);
            VecDataSD_[ii] = ddata; 
            if (jj != ii+1)
            {
              printf("Kriging ERROR: line %d in the std dev file.\n",jj+1);
              VecDataSD_.setLength(0);
              break;
            }
          } 
          fclose(fp);
          if (VecDataSD_.length() > 0)
          {
            printf("Kriging INFO: std. dev. file has been read.\n");
            KRI_VecDataSD = VecDataSD_; 
          }
        }
      }
    }
  }
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
Kriging::~Kriging()
{
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int Kriging::initialize(double *X, double *Y)
{
  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Kriging training begins....\n");
  train(X,Y);
  if (outputLevel_ >= 1) printf("Kriging training completed.\n");
  return 0;
}
  
// ************************************************************************
// Generate results for display
// ------------------------------------------------------------------------
int Kriging::genNDGridData(double *XIn,double *YIn,int *NOut,double **XOut,
                           double **YOut)
{
  int totPts;

  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Kriging training begins....\n");
  train(XIn,YIn);
  if (outputLevel_ >= 1) printf("Kriging training completed.\n");
  if ((*NOut) == -999) return 0;
 
  //**/ ---------------------------------------------------------------
  //**/ generating regular grid data
  //**/ ---------------------------------------------------------------
  genNDGrid(NOut, XOut);
  if ((*NOut) == 0) return 0;
  totPts = (*NOut);

  //**/ ---------------------------------------------------------------
  //**/ generate the data points 
  //**/ ---------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();

  if (outputLevel_ >= 1) printf("Kriging interpolation begins....\n");
  predict(totPts, *XOut, *YOut, NULL);
  if (outputLevel_ >= 1) printf("Kriging interpolation completed.\n");
  return 0;
}

// ************************************************************************
// Generate 1D results for display
// ------------------------------------------------------------------------
int Kriging::gen1DGridData(double *XIn,double *YIn,int ind1,
                           double *settings,int *NOut, double **XOut, 
                           double **YOut)
{
  int    ii, kk, totPts;
  double HX;
  psVector vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Kriging training begins....\n");
  train(XIn,YIn);
  if (outputLevel_ >= 1) printf("Kriging training completed.\n");
 
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_;
  HX = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut;
  VecXOut.setLength(totPts);
  (*XOut) = VecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (kk = 0; kk < nInputs_; kk++) 
      vecXT[ii*nInputs_+kk] = settings[kk]; 
    vecXT[ii*nInputs_+ind1] = HX * ii + VecLBs_[ind1];
    (*XOut)[ii] = HX * ii + VecLBs_[ind1];
  }
    
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("Kriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("Kriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 2D results for display
// ------------------------------------------------------------------------
int Kriging::gen2DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                           double *settings, int *NOut, double **XOut, 
                           double **YOut)
{
  int ii, jj, kk, totPts, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Kriging training begins....\n");
  train(XIn, YIn);
  if (outputLevel_ >= 1) printf("Kriging training completed.\n");
  
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(2);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut;
  VecXOut.setLength(totPts * 2);
  (*XOut) = VecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      index = ii * nPtsPerDim_ + jj;
      for (kk = 0; kk < nInputs_; kk++) 
        vecXT[index*nInputs_+kk] = settings[kk]; 
      vecXT[index*nInputs_+ind1]  = vecHX[0] * ii + VecLBs_[ind1];
      vecXT[index*nInputs_+ind2]  = vecHX[1] * jj + VecLBs_[ind2];
      (*XOut)[index*2]   = vecHX[0] * ii + VecLBs_[ind1];
      (*XOut)[index*2+1] = vecHX[1] * jj + VecLBs_[ind2];
    }
  }
    
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("Kriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("Kriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 3D results for display
// ------------------------------------------------------------------------
int Kriging::gen3DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                           int ind3, double *settings, int *NOut, 
                           double **XOut, double **YOut)
{
  int ii, jj, ll, kk, totPts, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Kriging training begins....\n");
  train(XIn, YIn);
  if (outputLevel_ >= 1) printf("Kriging training completed.\n");
 
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(3);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut;
  VecXOut.setLength(totPts * 3);
  (*XOut) = VecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        index = ii * nPtsPerDim_ * nPtsPerDim_ + jj * nPtsPerDim_ + ll;
        for (kk = 0; kk < nInputs_; kk++) 
          vecXT[index*nInputs_+kk] = settings[kk]; 
        vecXT[index*nInputs_+ind1] = vecHX[0] * ii + VecLBs_[ind1];
        vecXT[index*nInputs_+ind2] = vecHX[1] * jj + VecLBs_[ind2];
        vecXT[index*nInputs_+ind3] = vecHX[2] * ll + VecLBs_[ind3];
        (*XOut)[index*3]   = vecHX[0] * ii + VecLBs_[ind1];
        (*XOut)[index*3+1] = vecHX[1] * jj + VecLBs_[ind2];
        (*XOut)[index*3+2] = vecHX[2] * ll + VecLBs_[ind3];
      }
    }
  }
   
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("Kriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("Kriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Generate 4D results for display
// ------------------------------------------------------------------------
int Kriging::gen4DGridData(double *XIn, double *YIn, int ind1, int ind2, 
                           int ind3, int ind4, double *settings, 
                           int *NOut, double **XOut, double **YOut)
{
  int ii, jj, ll, mm, kk, totPts, index;
  psVector vecHX, vecXT;

  //**/ ---------------------------------------------------------------
  //**/ generate the Gaussian hyperparameters
  //**/ ---------------------------------------------------------------
  if (outputLevel_ >= 1) printf("Kriging training begins....\n");
  train(XIn, YIn);
  if (outputLevel_ >= 1) printf("Kriging training completed.\n");
  
  //**/ ---------------------------------------------------------------
  //**/ set up for generating regular grid data
  //**/ ---------------------------------------------------------------
  totPts = nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_ * nPtsPerDim_;
  vecHX.setLength(4);
  vecHX[0] = (VecUBs_[ind1] - VecLBs_[ind1]) / (nPtsPerDim_ - 1); 
  vecHX[1] = (VecUBs_[ind2] - VecLBs_[ind2]) / (nPtsPerDim_ - 1); 
  vecHX[2] = (VecUBs_[ind3] - VecLBs_[ind3]) / (nPtsPerDim_ - 1); 
  vecHX[3] = (VecUBs_[ind4] - VecLBs_[ind4]) / (nPtsPerDim_ - 1); 

  //**/ ---------------------------------------------------------------
  //**/ allocate storage for the data points
  //**/ ---------------------------------------------------------------
  psVector VecXOut;
  VecXOut.setLength(totPts * 4);
  (*XOut) = VecXOut.takeDVector();
  vecXT.setLength(totPts*nInputs_);
  for (ii = 0; ii < nPtsPerDim_; ii++) 
  {
    for (jj = 0; jj < nPtsPerDim_; jj++) 
    {
      for (ll = 0; ll < nPtsPerDim_; ll++) 
      {
        for (mm = 0; mm < nPtsPerDim_; mm++) 
        {
          index = ii*nPtsPerDim_*nPtsPerDim_*nPtsPerDim_ +
                  jj*nPtsPerDim_*nPtsPerDim_ + ll*nPtsPerDim_ + mm;
          for (kk = 0; kk < nInputs_; kk++) 
            vecXT[index*nInputs_+kk] = settings[kk]; 
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
    
  //**/ ---------------------------------------------------------------
  //**/ interpolate
  //**/ ---------------------------------------------------------------
  psVector VecYOut;
  VecYOut.setLength(totPts);
  (*YOut) = VecYOut.takeDVector();
  if (outputLevel_ >= 1) printf("Kriging interpolation begins....\n");
  predict(totPts, vecXT.getDVector(), *YOut, NULL);
  if (outputLevel_ >= 1) printf("Kriging interpolation completed.\n");
  (*NOut) = totPts;
  return 0;
}

// ************************************************************************
// Evaluate a given point 
// ------------------------------------------------------------------------
double Kriging::evaluatePoint(double *X)
{
  int    iOne=1;
  double Y=0.0;
  predict(iOne, X, &Y, NULL);
  return Y;
}

// ************************************************************************
// Evaluate a number of points 
// ------------------------------------------------------------------------
double Kriging::evaluatePoint(int npts, double *X, double *Y)
{
  predict(npts, X, Y, NULL);
  return 0.0;
}

// ************************************************************************
// Evaluate a given point, return also the standard deviation 
// ------------------------------------------------------------------------
double Kriging::evaluatePointFuzzy(double *X, double &Ystd)
{
  int    iOne=1;
  double Y=0.0;
  predict(iOne, X, &Y, &Ystd);
  return Y;
}

// ************************************************************************
// Evaluate a number of points, return also the standard deviation 
// ------------------------------------------------------------------------
double Kriging::evaluatePointFuzzy(int npts, double *X, double *Y, 
                                   double *Ystds)
{
  predict(npts, X, Y, Ystds);
  return 0.0;
}

// ************************************************************************
// training 
// ------------------------------------------------------------------------
double Kriging::train(double *X, double *Y)
{
  int    ii, jj, kk, nBasis, count, status, Cleng;
  double dist, ddata; 
  char   pString[101];
  psVector VecXDists;

  //**/ ============================================================
  // normalize the input and outputs 
  //**/ ============================================================
  VecNormalX_.setLength(nSamples_*nInputs_);
  initInputScaling(X, VecNormalX_.getDVector(), 1);
  VecNormalY_.setLength(nSamples_);
  initOutputScaling(Y, VecNormalY_.getDVector());
  KRI_YStd = YStd_;

  //**/ ============================================================
  // compute distances between all pairs of inputs (nDists, XDists)
  //**/ needed for creating the correlation matrix (fastMode=2,3)
  //**/ if there are repeated points, the covariance matrix will be
  //**/ singular. 
  //**/ ============================================================
  if (fastMode_ >= 1 && fastMode_ <= 3) 
  {
    status = computeDistances(VecXDists);
    if (status != 0)
    {
      printf("Kriging ERROR: Since there are repeated sample ");
      printf("points, Kriging will\n");
      printf("        terminate because the covariance matrix ");
      printf("will be singular. BYE\n");
      exit(0);
    }
  }
  if (KRI_nugget != 0.0 && outputLevel_ > 0) 
    printf("Kriging INFO: nugget = %e\n",KRI_nugget);

  //**/ ============================================================
  // fast mode = 4: use some other optimization method (which does 
  // not work well so it is not used) 
  //**/ ============================================================
  if (fastMode_ == 4)
  {
    if (outputLevel_ >= 1) 
      printf("Kriging training (4) begins....\n");
    optimize();
    nBasis = 1;
    if (outputLevel_ >= 1) 
    {
      printf("Kriging training (4) ends.\n");
      printf("Kriging optimal parameter values are:\n");
      for (ii = 0; ii < nInputs_; ii++)
        printf("     Input %d : %e\n", ii+1, VecThetas_[ii]);
    }
    return 0.0;
  }
  //**/ ============================================================
  // no optimization mode = 1
  //**/ ============================================================
  else if (fastMode_ == 1)
  {
    if (outputLevel_ > 0)
    {
      printEquals(PL_INFO,0);
      printf("INFO: No optimization - Kriging parameter values are:\n");
      for (ii = 0; ii < nInputs_; ii++)
        printf("     Input %d : %e\n", ii+1, VecThetas_[ii]);
    }
    //**/ ----------------------------------------------------
    //**/ find order
    //**/ ----------------------------------------------------
    if (nSamples_ <= nInputs_ + 1) pOrder_ = 0;
    if (pOrder_ == 1) nBasis = nInputs_ + 1;
    else              nBasis = 1;
    if (outputLevel_ > 0) printEquals(PL_INFO,0);
  }
  else
  //**/ ============================================================
  // slow mode: optimize
  //**/ ============================================================
  {
    int    maxfun, pLevel, nPts, iOne=1, iZero=0, nSamOpt, mode;
    double rhobeg=1.0, rhoend=1.0e-4, optY=PSUADE_UNDEFINED;
    FILE   *fp=NULL;
    Sampling *sampler;
    psVector VecUppers, VecLowers, VecTVals, VecSamIns, VecSamOut;
    psVector VecW, VecOptThetas, VecTValSave;
    psIVector VecSamSts;

    //**/ ----------------------------------------------------
    //**/ if slow mode, need to make sure the psuade_stop
    //**/ file has been cleaned up
    //**/ ----------------------------------------------------
    fp = fopen("psuade_stop", "r");
    if (fp != NULL)
    {
      printf("Kriging ERROR: Remove the 'psuade_stop' file\n");
      printf("               first and re-do.\n");
      fclose(fp);
      exit(1);
    }

    //**/ ----------------------------------------------------
    //**/ get number of basis 
    //**/ ----------------------------------------------------
    if (nSamples_ <= nInputs_ + 1) pOrder_ = 0;
    if (pOrder_ == 1) nBasis = nInputs_ + 1;
    else              nBasis = 1;
    Cleng = nSamples_ + nBasis;

    //**/ ----------------------------------------------------
    //**/ prepare for optimization 
    //**/ ----------------------------------------------------
    VecUppers.setLength(nInputs_+1);
    VecLowers.setLength(nInputs_+1);
    double *TUppers = VecUppers.getDVector();
    double *TLowers = VecLowers.getDVector();
    for (ii = 0; ii < nInputs_; ii++)
    {
      if (VecXMeans_[ii] == 0 && VecXStds_[ii] == 1.0)
      {
        TUppers[ii] = 30.0 * (VecUBs_[ii]-VecLBs_[ii]);
        TLowers[ii] =  0.1 * (VecUBs_[ii]-VecLBs_[ii]);
      }
      else
      {
        TUppers[ii] = 30.0;
        TLowers[ii] = 0.1;
      }
      if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
      {
        printf("Kriging: for input %d :\n", ii+1);
        printf("Kriging: current optimization lower bound = %e\n",
               TLowers[ii]);
        snprintf(pString,100,
           "Kriging: Enter new optimization lower bound : ");
        TLowers[ii] = getDouble(pString);
        if (TLowers[ii] <= 0.0)
        {
          printf("Kriging ERROR: lower bound <= 0\n");
          exit(1);
        }
        printf("Kriging: current optimization upper bound = %e\n",
               TUppers[ii]);
        snprintf(pString,100,
           "Kriging: Enter optimization upper bound : ");
        TUppers[ii] = getDouble(pString);
        if (TLowers[ii] > TUppers[ii])
        {
          printf("Kriging ERROR: lower bound >= upper bound\n");
          exit(1);
        }
      }
    }
    rhobeg = TUppers[0] - TLowers[0];
    for (ii = 1; ii < nInputs_; ii++)
    {
      ddata = TUppers[ii] - TLowers[ii];
      if (ddata < rhobeg) rhobeg = ddata;
    }
    rhobeg *= 0.5;
    rhoend = rhobeg * optTolerance_;
    VecTVals.setLength(nInputs_+1);
    double *TValues = VecTVals.getDVector();
    nPts = (nInputs_ + 1) * (nInputs_ + 2) / 2;
    jj = (nPts+13) * (nPts+nInputs_) + 3*nInputs_*(nInputs_+3)/2;
    kk = (nPts+5)*(nPts+nInputs_)+3*nInputs_*(nInputs_+5)/2+1;
    if (jj > kk) VecW.setLength(jj);
    else         VecW.setLength(kk);
    if (outputLevel_ > 0)
    {
      printEquals(PL_INFO, 0);
      printf("* Kriging optimization tolerance = %e\n", rhoend);
    }

    //**/ ----------------------------------------------------
    //**/ generate a sample
    //**/ ----------------------------------------------------
    if (fastMode_ == 2)
    {
      if (outputLevel_ >= 1) 
        printf("Kriging training (2) begins.... (order = %d)\n",pOrder_);
      nSamOpt = 1;
      VecSamIns.setLength(nSamOpt * nInputs_);
      for (ii = 0; ii < nInputs_; ii++) VecSamIns[ii] = VecThetas_[ii];
    }
    else
    {
      if (outputLevel_ >= 1) 
        printf("Kriging training (3) begins.... (order = %d)\n",pOrder_);
      mode = 1;
      nSamOpt = 5;
      if (psConfig_.GMModeIsOn() && psConfig_.InteractiveIsOn())
      {
        printf("Kriging: slow mode with multi-start (5) optimization.\n");
        printf("Choose sampling method to generate multi-start.\n");
        snprintf(pString,100,"Sampling method (1-QMC, 2-LHS, 3-FF) : ");
        mode = getInt(1,3,pString);
        if (mode != 3)
        {
          snprintf(pString,100,"Enter sample size (1 - 100) : ");
          nSamOpt = getInt(1,100,pString);
        } 
      }
      if (mode == 1)
         sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LPTAU);
      else if (mode == 2)
         sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_LHS);
      else
         sampler = (Sampling *) SamplingCreateFromID(PSUADE_SAMP_FF4);
 
      sampler->setPrintLevel(0);
      sampler->setInputBounds(nInputs_, TLowers, TUppers);
      sampler->setOutputParams(iOne);
      sampler->setSamplingParams(nSamOpt+2, iOne, iZero);
      sampler->initialize(0);
      nSamOpt = sampler->getNumSamples();
      VecSamIns.setLength(nSamOpt*nInputs_);
      VecSamOut.setLength(nSamOpt);
      VecSamSts.setLength(nSamOpt);
      sampler->getSamples(nSamOpt,nInputs_,iOne,VecSamIns.getDVector(),
                     VecSamOut.getDVector(), VecSamSts.getIVector());
      //**/ the first 2 points usually not good
      nSamOpt -= 2;
      for (ii = 0; ii < nSamOpt*nInputs_; ii++)
        VecSamIns[ii] = VecSamIns[ii+2*nInputs_];
      delete sampler;
    }

    //**/ ----------------------------------------------------
    //**/ pass data to global variables for external calls
    //**/ ----------------------------------------------------
    if (VecXDists.length() > 0) KRI_VecXDists = VecXDists;
    KRI_nSamples = nSamples_;
    KRI_nInputs = nInputs_;
    KRI_pOrder = pOrder_;
    KRI_VecX = VecNormalX_;
    KRI_VecY = VecNormalY_;
    KRI_VecYT.setLength(Cleng);
    KRI_VecOptThetas = VecThetas_;
    maxfun = 5000;
    KRI_OptY = PSUADE_UNDEFINED;
    KRI_outputLevel = outputLevel_;
    VecOptThetas.setLength(nInputs_);
    KRI_MatS.setDim(nSamples_, nSamples_);
    double *SMatrix = KRI_MatS.getMatrix1D();
    KRI_MatF.setDim(nSamples_, nBasis);
    double *FMatrix = KRI_MatF.getMatrix1D();
    KRI_MatFTmp.setDim(nSamples_, nBasis);
    double *FMatTmp = KRI_MatFTmp.getMatrix1D();
    KRI_MatM.setDim(nBasis, nBasis);
    double *MMatrix = KRI_MatM.getMatrix1D();
    VecTValSave.setLength(nSamOpt*nInputs_);
    double dmean, dstd;
    int    stopFlag = 0;

    //**/ ----------------------------------------------------
    //**/ call optimizer 
    //**/ ----------------------------------------------------
    //**/ use pLevel=8888 to tell bobyqa_ to use this function
    //**/ for now set the initial guess to mid point
    for (kk = 0; kk < nSamOpt; kk++)
    {
      fp = fopen("psuade_print", "r");
      if (fp != NULL)
      {
        fscanf(fp, "%d", &outputLevel_);
        fclose(fp);
        fp = NULL;
        if (outputLevel_ > 0 && outputLevel_ <= 5)
          printf("Kriging: output level set to %d.\n", outputLevel_);
        else
        {
          outputLevel_ = 2;
          printf("Kriging: output level set to %d.\n", outputLevel_);
        }
        KRI_outputLevel = outputLevel_;
      }
      if (outputLevel_ > 1) 
        printf("Kriging multi-start optimization: start = %d (%d)\n",
               kk+1, nSamOpt);
      KRI_iter = 0;
      for (ii = 0; ii < nInputs_; ii++) 
        TValues[ii] = VecSamIns[kk*nInputs_+ii];
      if (outputLevel_ > 1) 
      {
        for (ii = 0; ii < nInputs_; ii++) 
          printf("Kriging: Input %4d initial length scale = %e\n",
                 ii+1,TValues[ii]);
      }
      KRI_noProgressCnt = 0;

      pLevel = -1;
#ifdef HAVE_BOBYQA
      pLevel = 8888;
#endif
#ifdef HAVE_NEWUOA
      pLevel = 6666;
#endif
#ifdef HAVE_BOBYQA
      if (pLevel == 8888)
        kbobyqa_(&nInputs_,&nPts,TValues,TLowers,TUppers,&rhobeg,
                 &rhoend,&pLevel, &maxfun, VecW.getDVector());
#endif
#ifdef HAVE_NEWUOA
      if (pLevel == 6666)
        newuoa_(&nInputs_, &nPts, TValues, &rhobeg, &rhoend, 
                &pLevel, &maxfun, VecW.getDVector());
#endif
      if (pLevel == -1)
      {
        printf("ERROR: No optimizer unavailable.\n");
        exit(1);
      }
      if (outputLevel_ > 1) 
      {
        printf("Kriging multi-start optimization: iteration = %d (%d)\n",
               kk+1, nSamOpt);
        for (ii = 0; ii < nInputs_; ii++) 
          printf("Kriging: Input %4d final length scale = %e\n",
                 ii+1,TValues[ii]);
        printf("Kriging final objective value = %e (ref = %e)\n", 
               KRI_currY, rhoend);
      }
      if (KRI_OptY < optY)
      {
        optY = KRI_OptY;
        for (ii = 0; ii < nInputs_; ii++)
          VecOptThetas[ii] = KRI_VecOptThetas[ii];
        if (optY < rhoend)
        {
          if (outputLevel_ > 1) 
            printf("Kriging INFO: termination (sufficiently accurate)\n");
          break;
        }
      }
      fp = fopen("psuade_stop", "r");
      if (fp != NULL)
      {
        fclose(fp);
        fp = NULL;
        printf("Kriging: psuade_stop file found - terminating ....\n");
        break;
      }
      fp = fopen("ps_rs_expert", "r");
      if (fp != NULL)
      {
        fclose(fp);
        fp = NULL;
        printf("Kriging: turn on rs_expert mode.\n");
        psConfig_.RSExpertModeOn();
      }
      for (ii = 0; ii < nInputs_; ii++) 
        VecTValSave[kk*nInputs_+ii] = TValues[ii];
      if (kk >= 4)
      {
        for (ii = 0; ii < nInputs_; ii++)
        {
          dmean = 0.0;
          for (jj = 1; jj < kk+1; jj++) dmean += VecTValSave[jj*nInputs_+ii];
          dmean /= (double) kk; 
          dstd = 0.0;
          for (jj = 1; jj < kk+1; jj++)
            dstd += pow(VecTValSave[jj*nInputs_+ii]-dmean,2.0);
          dstd = sqrt(dstd/(double) (kk-1));
          if (dstd/dmean < 0.01) stopFlag++;
          if (outputLevel_ > 1) 
            printf("Opt convergence check: input %d: mean=%e, std=%e\n",
                   ii+1,dmean,dstd);
        }
        if (stopFlag == nInputs_) 
        {
          if (outputLevel_ > 1) 
          {
            printf("Kriging INFO: Same optimum after %d iterations.\n",
                   kk+1);
            printf("              Stop further processing.\n");
          }
          break;
        }
      }  
    }
    //**/ ----------------------------------------------------
    //**/ display results and clean up
    //**/ ----------------------------------------------------
    for (ii = 0; ii < nInputs_; ii++) VecThetas_[ii] = VecOptThetas[ii];
    if (outputLevel_ > 1) 
    {
      for (ii = 0; ii < nInputs_; ii++) 
        printf("Kriging: Input %4d optimal length scale = %e\n",
               ii+1,VecThetas_[ii]);
      printf("Kriging: optimal objective value = %e\n",KRI_OptY);
    }
    if (outputLevel_ > 0)
    {
      if (fastMode_ == 2)
           printf("Kriging training (2) ends.\n");
      else printf("Kriging training (3) ends.\n");
    }
  }

  //**/ ============================================================
  //**/ compute output variance 
  //**/    ||inv(R)Y - RF inv([RF]^T RF) [RF]^T inv(R)Y||/nBasis
  //**/ where RF = inv(R) F
  //**/ At the end, this section will produce
  //**/ Rmat has inverse of R (Cholesky decomposition)
  //**/ Mmat has inverse of F^T R^-1 F (Cholesky decomposition)
  //**/ ============================================================
  //**/ -------------------------------------------------------
  //**/ 1. Cholesky decomposition for the covarance matrix 
  //**/ -------------------------------------------------------
  char uplo='L';
  char trans='N';
  char diag='N';
  count = 0;
  MatrixR_.setDim(nSamples_, nSamples_);
  double *Rmat = MatrixR_.getMatrix1D();
  for (jj = 0; jj < nSamples_; jj++)
  {
    Rmat[jj*nSamples_+jj] = 1.0 + 1e-15 * nSamples_;
    if (KRI_nugget != 0.0)  Rmat[jj*nSamples_+jj] += KRI_nugget;
    else if (nInputs_ == 2) Rmat[jj*nSamples_+jj] += 0.01;

    if (VecDataSD_.length() > 0) 
      Rmat[jj*nSamples_+jj] += pow(VecDataSD_[jj]/YStd_,KRI_Exponent);
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
        dist += pow(VecXDists[count*nInputs_+ii]/VecThetas_[ii],
                    KRI_Exponent);
      dist = exp(-dist);
      if (dist < 1.0e-16) dist = 0.0;
      Rmat[jj*nSamples_+kk] = dist;
      Rmat[kk*nSamples_+jj] = dist;
      count++;
    }
  }
  if (outputLevel_ > 4)
  {
    printf("Kriging: covariance matrix for Cholesky decompositon.\n");
    for (jj = 0; jj < nSamples_; jj++)
    {
      for (kk = 0; kk < nSamples_; kk++)
        printf("%e ", Rmat[jj*nSamples_+kk]);
      printf("\n");
    }
  }
  dpotrf_(&uplo, &nSamples_, Rmat, &nSamples_, &status);
  if (status != 0) 
  {
    printf("Kriging ERROR: Cholesky decomposition not successful.\n");
    exit(1);
  }
  //**/ -------------------------------------------------------
  //**/ 2. compute CY = inv(C) Y where R = C C^T
  //**/ -------------------------------------------------------
  int    inc=1;
  psVector VecCY;
  VecCY.setLength(nSamples_);
  double *CY = VecCY.getDVector();
  for (jj = 0; jj < nSamples_; jj++) CY[jj] = VecNormalY_[jj];
  dtrsv_(&uplo,&trans,&diag,&nSamples_,Rmat,&nSamples_,CY, &inc);
  //**/ -------------------------------------------------------
  //**/ 3. Create F (nSamples x nBasis)
  //**/ -------------------------------------------------------
  psMatrix MatCF, MatRF;
  MatCF.setDim(nSamples_, nBasis);
  MatRF.setDim(nSamples_, nBasis);
  double *CFmatrix = MatCF.getMatrix1D();
  double *RFmatrix = MatRF.getMatrix1D();
  for (jj = 0; jj < nSamples_; jj++)
  {
    CFmatrix[jj] = RFmatrix[jj] = 1.0;
    if (pOrder_ == 1)
    {
      for (kk = 1; kk < nInputs_+1; kk++)
      {
        ddata = VecNormalX_[jj*nInputs_+kk-1];
        CFmatrix[kk*nSamples_+jj] = RFmatrix[kk*nSamples_+jj] = ddata;
      }
    }
  }
  //**/ -------------------------------------------------------
  //**/ 4. F = inv(C) * F
  //**/ -------------------------------------------------------
  for (ii = 0; ii < nBasis; ii++)
  {
    dtrsv_(&uplo,&trans,&diag,&nSamples_,Rmat,&nSamples_,
           &CFmatrix[ii*nSamples_], &inc);
    //**/ Why did I put this here? (7/2014)
    //**/ dpotrs_(&uplo, &nSamples_, &inc, Rmat, &nSamples_,
    //**/         &RFmatrix[ii*nSamples_], &nSamples_, &status);
  } 
  //**/ -------------------------------------------------------
  //**/ 5. compute V1 = [inv(C) F]^T inv(C)Y 
  //**/ -------------------------------------------------------
  VecV1_.setLength(nBasis);
  for (jj = 0; jj < nBasis; jj++)
  {
    ddata = 0.0;
    for (ii = 0; ii < nSamples_; ii++)
      ddata += CFmatrix[ii+jj*nSamples_]*CY[ii];
    VecV1_[jj] = ddata;
  }
  //**/ -------------------------------------------------------
  //**/ 6. compute M = [inv(C) F]^T [inv(C) F] = F^T inv(R) F 
  //**/ -------------------------------------------------------
  MatrixM_.setDim(nBasis, nBasis);
  double *MMat = MatrixM_.getMatrix1D();
  for (jj = 0; jj < nBasis; jj++)
  {
    for (kk = jj; kk < nBasis; kk++)
    {
      ddata = 0.0;
      for (ii = 0; ii < nSamples_; ii++)
        ddata += CFmatrix[ii+jj*nSamples_]*
                 CFmatrix[ii+kk*nSamples_];
      MMat[jj+kk*nBasis] = MMat[kk+jj*nBasis] = ddata;
    }
  }
  //**/ -------------------------------------------------------
  //**/ 7. compute V1 = inv(M) F^T inv(R) Y 
  //**/ -------------------------------------------------------
  dpotrf_(&uplo, &nBasis, MMat, &nBasis, &status);
  if (status != 0) 
  {
    printf("Kriging ERROR: Cholesky decomposition not successful.\n");
    exit(1);
  }
  dpotrs_(&uplo, &nBasis, &inc, MMat, &nBasis, VecV1_.getDVector(), 
          &nBasis, &status);
  if (status != 0) 
  {
    printf("Kriging ERROR: LU solve not successful.\n");
    exit(1);
  }
  //**/ -------------------------------------------------------
  //**/ 8. compute CY = inv(C)Y - inv(C)F inv(M) CF^T inv(C) Y 
  //**/ -------------------------------------------------------
  for (ii = 0; ii < nBasis; ii++)
  {
    for (jj = 0; jj < nSamples_; jj++) 
      CY[jj] -= CFmatrix[jj+ii*nSamples_] * VecV1_[ii];
  } 
  KrigingVariance_ = 0.0;
  for (jj = 0; jj < nSamples_; jj++) KrigingVariance_ += CY[jj] * CY[jj];
  KrigingVariance_ /= (double) nSamples_;
  KrigingVariance_ *= YStd_ * YStd_;
  if (outputLevel_ > 0) printf("Kriging variance = %e\n",KrigingVariance_);
  //**/ -------------------------------------------------------
  //**/ 9. compute V2 = inv(C) [inv(C) Y - inv(C) CF V1
  //**/               = inv(R) Y - inv(R) F V1
  //**/ -------------------------------------------------------
  VecV2_.setLength(nSamples_);
  for (ii = 0; ii < nSamples_; ii++) VecV2_[ii] = CY[ii];
  trans = 'T';
  dtrsv_(&uplo,&trans,&diag,&nSamples_,Rmat,&nSamples_,
         VecV2_.getDVector(), &inc);
  //**/ ============================================================
  //**  END
  //**/ ============================================================

  //**/ ============================================================
  //**  store results
  //**/ ============================================================
  if (noReuse_ == 1 || psConfig_.RSExpertModeIsOn())
  {
    FILE *fp = fopen("psuade_kriging_optdata", "w");
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp,"%e\n", VecThetas_[ii]);
    fclose(fp);
    printf("INFO: Kriging optimal parameters are in ");
    printf("psuade_kriging_optdata.\n");
  }
  if (noReuse_ == 1 || !psConfig_.RSCodeGenIsOn()) return 0;
  genRSCode();
  return 0.0;
}

// ************************************************************************
// predict 
// ------------------------------------------------------------------------
double Kriging::predict(int length, double *X, double *Y, double *YStds)
{
  int    ii, jj, kk, ll, status, nRows, nBasis, leng, offset, maxLeng=500;
  double ddata, dist, mean, stdev;
  char   uplo='L';

  //**/ -------------------------------------------------------
  //**/ fast mode, call predict0
  //**/ -------------------------------------------------------
  if (fastMode_ == 0)
  {
    predict0(length, X, Y, YStds);
    return 0.0;
  }

  //**/ -------------------------------------------------------
  //**/ normalize the test set inputs and put into VecWX
  //**/ -------------------------------------------------------
  psVector vecWA, vecWX;
  nRows = nSamples_ + 1;
  if (pOrder_ == 1) nRows = nRows + nInputs_;
  nBasis = nRows - nSamples_;
  vecWA.setLength(2*nSamples_*maxLeng+maxLeng);
  vecWX.setLength(maxLeng*nInputs_+2*maxLeng*nBasis);
  workLength_ = maxLeng;
  double *Mmat = MatrixM_.getMatrix1D();
  double *Rmat = MatrixR_.getMatrix1D();

  //**/ -------------------------------------------------------
  //**/ compute correlation with sample inputs ==> r
  //**/ -------------------------------------------------------
  for (ll = 0; ll < length; ll+=maxLeng)
  {
    leng = maxLeng;
    if (ll+leng > length) leng = length - ll;
    //**/ normalize the inputs
    for (ii = 0; ii < nInputs_; ii++)
    {
      mean  = VecXMeans_[ii];
      stdev = VecXStds_[ii];
      for (kk = 0; kk < leng; kk++)
        vecWX[kk*nInputs_+ii] = (X[(ll+kk)*nInputs_+ii]-mean)/stdev;
    }
    //**/ compute the right hand side 
    for (kk = 0; kk < leng; kk++)
    {
      for (jj = 0; jj < nSamples_; jj++)
      {
        dist = 0.0;
        for (ii = 0; ii < nInputs_; ii++)
        {
          ddata = VecNormalX_[jj*nInputs_+ii]-vecWX[kk*nInputs_+ii];
          dist += ddata * ddata / (VecThetas_[ii] * VecThetas_[ii]);
        }
        ddata = exp(-dist);
        if (ddata < 1.0e-16) ddata = 0.0;
        vecWA[kk*nSamples_+jj] = ddata;
        vecWA[nSamples_*leng+kk*nSamples_+jj] = ddata;
      }
    }
    //**/ -------------------------------------------------------
    //**/ Recall V1 = inv(F^T inv(R) F) F^T inv(R) Y 
    //**/ Recall V2 = inv(R) Y - inv(R)F inv(M) F^T inv(R) Y 
    //**/           = inv(R) Y - inv(C) F V1
    //**/ estimate y = [1 x]' * V1 + r' * V2
    //**/ -------------------------------------------------------
    for (kk = 0; kk < leng; kk++)
    {
      ddata = 1.0 * VecV1_[0];
      if (pOrder_ == 1)
        for (ii = 1; ii <= nInputs_; ii++) 
          ddata += vecWX[kk*nInputs_+ii-1]*VecV1_[ii];
      for (jj = 0; jj < nSamples_; jj++) 
        ddata += vecWA[nSamples_*leng+kk*nSamples_+jj]*VecV2_[jj];
      Y[ll+kk] = ddata * YStd_ + YMean_;
    }
    if (YStds != NULL)
    {
      //**/ ----------------------------------------------------
      //**/ compute s_k = inv(R) r_k for all k
      //**/ (vecWA[leng*nSamples])
      //**/ ----------------------------------------------------
      double *WorkA = vecWA.getDVector();
      dpotrs_(&uplo, &nSamples_, &leng, Rmat, &nSamples_,
              &WorkA[leng*nSamples_], &nSamples_, &status);
      //**/ ----------------------------------------------------
      //**/ compute r_k^T inv(R) r_k for all k
      //**/ (vecWA[2*leng*nSamples])
      //**/ ----------------------------------------------------
      for (kk = 0; kk < leng; kk++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nSamples_; jj++)
          ddata += vecWA[kk*nSamples_+jj] *
                   vecWA[leng*nSamples_+kk*nSamples_+jj];
        vecWA[2*leng*nSamples_+kk] = ddata; 
      }
      //**/ ----------------------------------------------------
      //**/ compute u = F^T s_k - f (=> vecWX[leng*nInputs])
      //**/ ----------------------------------------------------
      for (kk = 0; kk < leng; kk++)
      {
        ddata = 0.0;
        for (jj = 0; jj < nSamples_; jj++)
           ddata += vecWA[leng*nSamples_+kk*nSamples_+jj]; 
        vecWX[leng*nInputs_+kk*nBasis] = ddata - 1.0; 
        vecWX[leng*nInputs_+leng*nBasis+kk*nBasis] = ddata - 1.0; 
        for (ii = 1; ii < nBasis; ii++)
        {
          ddata = 0.0;
          for (jj = 0; jj < nSamples_; jj++)
            ddata += vecWX[kk*nInputs_+ii-1] *
                     vecWA[leng*nSamples_+kk*nSamples_+jj]; 
          ddata -= vecWX[kk*nInputs_+ii-1];
          vecWX[leng*nInputs_+kk*nBasis+ii] = ddata; 
          vecWX[leng*nInputs_+leng*nBasis+kk*nBasis+ii] = ddata; 
        }
      }
      //**/ ----------------------------------------------------
      //**/ compute v = inv(M) u
      //**/ ----------------------------------------------------
      double *WorkX = vecWX.getDVector();
      dpotrs_(&uplo, &nBasis, &leng, Mmat, &nSamples_,
              &WorkX[leng*nInputs_+leng*nBasis], &nBasis, &status);

      //**/ ----------------------------------------------------
      //**/ compute final standard dev = sig(1+ u' * v - r'inv(R)r)
      //**/ ----------------------------------------------------
      for (kk = 0; kk < leng; kk++)
      {
        ddata = 0.0;
        for (ii = 0; ii < nBasis; ii++)
          ddata += vecWX[leng*nInputs_+kk*nBasis+ii] *
                   vecWX[leng*nInputs_+(leng+kk)*nBasis+ii];
        ddata = KrigingVariance_*
                (1.0 + ddata-vecWA[2*leng*nSamples_+kk]);
        if (ddata < 0.0)
        {
          printf("Kriging WARNING: prediction variance < 0\n");
          ddata = 0.0;
        }
        YStds[ll+kk] = sqrt(ddata);
      }
    }
  }
  return 0.0;
}

// ************************************************************************
// compute distances between all pairs of inputs
// ------------------------------------------------------------------------
int Kriging::computeDistances(psVector &VecLDists)
{
  int    ii, jj, kk, count, error=0;
  double dist;

  VecLDists.setLength((nSamples_*(nSamples_-1)/2)*nInputs_);
  count = 0;
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
      {
        VecLDists[count*nInputs_+ii] = VecNormalX_[jj*nInputs_+ii] - 
                            VecNormalX_[kk*nInputs_+ii];
        if (VecLDists[count*nInputs_+ii] < 0) 
          VecLDists[count*nInputs_+ii] = - VecLDists[count*nInputs_+ii]; 
        dist += pow(VecLDists[count*nInputs_+ii], 2.0);
      }
      if (dist == 0.0)
      {
        printf("Kriging ERROR: repeated sample points.\n");
        printf("               Prune repeated points and re-run.\n");
        printf("Sample %d : (with sample point %d)\n", kk+1, jj+1);
        for (ii = 0; ii < nInputs_; ii++)
          printf("   Input %d : %e\n",ii+1,
             VecNormalX_[kk*nInputs_+ii]*VecXStds_[ii]+VecXMeans_[ii]);
        error = 1;
      }
      count++;
    }
  }
  return error;
}

// ************************************************************************
// set parameters
// ------------------------------------------------------------------------
double Kriging::setParams(int targc, char **targv)
{
  int    ii;
  double mmax, range;
  char   pString[101];
  FILE   *fp=NULL;

  if (targc > 0 && !strcmp(targv[0], "noReuse"))
     noReuse_ = 1;
  else if (targc > 0 && !strcmp(targv[0], "setMode0"))
     fastMode_ = 0;
  else if (targc > 0 && !strcmp(targv[0], "setMode1"))
     fastMode_ = 1;
  else if (targc > 0 && !strcmp(targv[0], "setMode2"))
     fastMode_ = 2;
  else if (targc > 0 && !strcmp(targv[0], "setMode3"))
     fastMode_ = 3;
  else if (targc > 0 && !strcmp(targv[0], "rank"))
  {
    psVector vecLengScales;
    vecLengScales.setLength(nInputs_);
    mmax = 0.0;
    for (ii = 0; ii < nInputs_; ii++)
    {
      vecLengScales[ii] = 1.0 / VecThetas_[ii];
      if (VecXMeans_[ii] == 0 && VecXStds_[ii] == 1)
      {
        range = VecUBs_[ii] - VecLBs_[ii];
        vecLengScales[ii] *= range;
      }
      if (vecLengScales[ii] > mmax) mmax = vecLengScales[ii];
    }
    for (ii = 0; ii < nInputs_; ii++)
      vecLengScales[ii] = vecLengScales[ii] / mmax * 100.0;

    if (plotScilab())
         fp = fopen("scilabkrisa.sci", "w");
    else fp = fopen("matlabkrisa.m", "w");
    if (fp == NULL)
    {
      printf("Kriging ERROR: something wrong with opening a write file.\n");
    }
    else
    {
      fprintf(fp, "n = %d;\n", nInputs_);
      fprintf(fp, "Y = [\n");
      for (ii = 0; ii < nInputs_; ii++)
        fprintf(fp, "%24.16e \n", PABS(vecLengScales[ii]));
      fprintf(fp, "]; \n");
      fprintf(fp, "ymax = max(Y);\n");
      fprintf(fp, "ymin = 0;\n");
      fprintf(fp, "if (ymax == ymin)\n");
      fprintf(fp, "   ymax = ymax * 0.1;\n");
      fprintf(fp, "end;\n");
      fwritePlotCLF(fp);
      fprintf(fp, "bar(Y,0.8);\n");
      fwritePlotAxes(fp);
      snprintf(pString,100,"Kriging Ranking");
      fwritePlotTitle(fp, pString);
      snprintf(pString,100,"Input Numbers");
      fwritePlotXLabel(fp, pString);
      snprintf(pString,100,"Kriging Measure");
      fwritePlotYLabel(fp, pString);
      if (plotScilab())
      {
        fprintf(fp,"a=gca();\n");
        fprintf(fp,"a.data_bounds=[0,ymin; n+1,ymax+0.01*(ymax-ymin)];\n");
      }
      else
      {
        fprintf(fp,"axis([0 n+1 ymin ymax+0.01*(ymax-ymin)])\n");
      }
      fclose(fp);
      if (plotScilab())
           printf("Kriging ranking in file scilabkrisa.sci\n");
      else printf("Kriging ranking in file matlabkrisa.m\n");
    }
    psIVector vecIT;
    vecIT.setLength(nInputs_);
    for (ii = 0; ii < nInputs_; ii++) vecIT[ii] = ii;
    sortDbleList2a(nInputs_, vecLengScales.getDVector(), 
                   vecIT.getIVector());
    printAsterisks(PL_INFO, 0);
    printf("* Kriging screening rankings\n");
    printAsterisks(PL_INFO, 0);
    for (ii = nInputs_-1; ii >= 0; ii--)
      printf("*  Rank %3d : Input = %4d (score = %5.1f) (ref = %e)\n",
             nInputs_-ii, vecIT[ii]+1, vecLengScales[ii], 
             0.01*vecLengScales[ii]*mmax);
    printAsterisks(PL_INFO, 0);
  }
  return 0.0;
}

// ************************************************************************
// perform optimization evaluation (different optimization method, does not
// work well
// ------------------------------------------------------------------------
void Kriging::optimize()
{
  int    ii, jj, mm, nBasis, deg; 
  double objfcn, objtmp, ddata;
  char   pString[101];
  psVector VecTUs, VecTLs, VecThetas, VecThetas2, VecT1, VecS1, VecScales;
  psVector VecXDists;

  //**/ ----------------------------------------------------
  //**/ prepare for optimization 
  //**/ ----------------------------------------------------
  pOrder_ = 1;
  nBasis  = 1;
  if (pOrder_ == 1) nBasis = nInputs_ + 1;
  VecTUs.setLength(nInputs_);
  VecTLs.setLength(nInputs_);
  double *TUppers = VecTUs.getDVector();
  double *TLowers = VecTLs.getDVector();
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (VecXMeans_[ii] == 0 && VecXStds_[ii] == 1.0)
    {
      TUppers[ii] = 20.0 * (VecUBs_[ii]-VecLBs_[ii]);;
      TLowers[ii] =  0.1 * (VecUBs_[ii]-VecLBs_[ii]);;
    }
    else
    {
      TUppers[ii] = 20.0;
      TLowers[ii] = 0.1;
    }
    if (psConfig_.MasterModeIsOn() && psConfig_.InteractiveIsOn())
    {
      printf("Kriging: for input %d :\n", ii+1);
      printf("Kriging: current optimization lower bound = %e\n",
             TLowers[ii]);
      snprintf(pString,100,
         "Kriging: Enter new optimization lower bound : ");
      TLowers[ii] = getDouble(pString);
      if (TLowers[ii] <= 0.0)
      {
        printf("Kriging ERROR: lower bound <= 0\n");
        exit(1);
      }
      printf("Kriging: current optimization upper bound = %e\n",
             TUppers[ii]);
      snprintf(pString,100,
         "Kriging: Enter new optimization upper bound : ");
      TUppers[ii] = getDouble(pString);
      if (TUppers[ii] <= 0.0)
      {
        printf("Kriging ERROR: upper bound <= 0\n");
        exit(1);
      }
      if (TLowers[ii] >= TUppers[ii])
      {
        printf("Kriging ERROR: upper bound <= lower bound.\n");
        exit(1);
      }
    }
  }
  computeDistances(VecXDists);
  KRI_MatS.setDim(nSamples_, nSamples_);
  KRI_MatF.setDim(nSamples_, nBasis);
  KRI_MatFTmp.setDim(nSamples_, nBasis);
  KRI_VecXDists = VecXDists;
  KRI_iter    = 0;
  VecBetas_.setLength(nBasis);
  VecBetasOpt_.setLength(nBasis);
  VecGammas_.setLength(nSamples_);
  VecGammasOpt_.setLength(nSamples_);
  VecThetas.setLength(nInputs_+1);
  VecThetasOpt_.setLength(nInputs_);

  //**/ check starting point
  for (ii = 0; ii < nInputs_; ii++) VecThetas[ii] = 10.0;
  objfcn = evaluateFunction(VecThetas.getDVector());

  //**/ get ready to iterate
  int maxIts = 4, flag;
  if (nInputs_ < maxIts) maxIts = nInputs_;
  if (nInputs_ == 1) maxIts = 2;
  VecScales.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++) 
    VecScales[ii] = pow(2.0, (1.0+ii)/(2.0+nInputs_));

  VecThetas2.setLength(nInputs_);
  VecT1.setLength(nInputs_);
  VecS1.setLength(nInputs_);
  //**/ iterate (given Thetas)
  for (mm = 0; mm < maxIts; mm++)
  {
    //**/ keep a copy of current theta
    for (ii = 0; ii < nInputs_; ii++) VecThetas2[ii] = VecThetas[ii];
     
    //**/ now explore
    for (ii = 0; ii < nInputs_; ii++)
    {
      //**/ keep a copy
      for (jj = 0; jj < nInputs_; jj++) VecT1[jj] = VecThetas[jj];
      if (VecThetas[ii] >= TUppers[ii])
      {
        flag = 1;
        VecT1[ii] = VecThetas[ii] / sqrt(VecScales[ii]);
      }
      else if (VecThetas[ii] == TLowers[ii])
      {
        flag = 1;
        VecT1[ii] = VecThetas[ii] * sqrt(VecScales[ii]);
      }
      else
      {
        flag = 0;
        VecT1[ii] = VecThetas[ii] * VecScales[ii];
        if (TUppers[ii] < VecT1[ii]) VecT1[ii] = TUppers[ii];
      }
      //**/ evaluate at this new point
      objtmp = evaluateFunction(VecT1.getDVector());
      //**/ if better, update Thetas
      if (objtmp < objfcn)
      {
        for (jj = 0; jj < nInputs_; jj++) VecThetas[jj] = VecT1[jj];
        objfcn = objtmp;
        for (jj = 0; jj < nBasis; jj++) 
          VecBetasOpt_[jj] = VecBetas_[jj];
        for (jj = 0; jj < nSamples_; jj++) 
          VecGammasOpt_[jj] = VecGammas_[jj];
        for (jj = 0; jj < nInputs_; jj++) 
          VecThetasOpt_[jj] = VecThetas[jj];
      }
      else
      {
        //**/ otherwise and if point not on boundaries
        if (flag == 0)
        {
          VecT1[ii] = VecThetas[ii] / VecScales[ii];
          if (VecT1[ii] < TLowers[ii]) VecT1[ii] = TLowers[ii];
          objtmp = evaluateFunction(VecT1.getDVector());
          if (objtmp < objfcn)
          {
            for (jj = 0; jj < nInputs_; jj++) VecThetas[jj] = VecT1[jj];
            objfcn = objtmp;
            for (jj = 0; jj < nBasis; jj++) 
              VecBetasOpt_[jj] = VecBetas_[jj];
            for (jj = 0; jj < nSamples_; jj++) 
              VecGammasOpt_[jj] = VecGammas_[jj];
            for (jj = 0; jj < nInputs_; jj++) 
              VecThetasOpt_[jj] = VecThetas[jj];
          }
        }
      }
    }

    //**/ move
    flag = 1;
    deg  = 1;
    for (ii = 0; ii < nInputs_; ii++) 
      VecS1[ii] = VecThetas[ii]/VecThetas2[ii];
    while (flag == 1)
    {
      for (ii = 0; ii < nInputs_; ii++)
      {
        ddata = pow(VecS1[ii],1.0*deg) * VecThetas[ii];
        if (ddata < TLowers[ii]) ddata = TLowers[ii]; 
        if (ddata > TUppers[ii]) ddata = TUppers[ii]; 
        VecT1[ii] = ddata;
      }
      objtmp = evaluateFunction(VecT1.getDVector());
      if (objtmp < objfcn)
      {
        for (ii = 0; ii < nInputs_; ii++) VecThetas[ii] = VecT1[ii];
        objfcn = objtmp;
        deg *= 2;
        for (jj = 0; jj < nBasis; jj++) 
          VecBetasOpt_[jj] = VecBetas_[jj];
        for (jj = 0; jj < nSamples_; jj++) 
          VecGammasOpt_[jj] = VecGammas_[jj];
        for (jj = 0; jj < nInputs_; jj++) 
          VecThetasOpt_[jj] = VecThetas[jj];
      }
      else flag = 0;
      for (ii = 0; ii < nInputs_; ii++) 
      {
        if (VecT1[ii] <= TLowers[ii]) flag = 0;
        if (VecT1[ii] >= TUppers[ii]) flag = 0;
      }
    }
    ddata = VecScales[0];
    for (ii = 0; ii < nInputs_-1; ii++) VecScales[ii] = VecScales[ii+1];
    VecScales[nInputs_-1] = ddata;
    for (ii = 0; ii < nInputs_; ii++) VecScales[ii] = pow(VecScales[ii],0.25);
  }
}

// ************************************************************************
// function evaluation 
// ------------------------------------------------------------------------
double Kriging::evaluateFunction(double *thetas)
{
  int    nBasis, count, ii, jj, kk, status, inc=1;
  double dist, ddata, retVal;
  FILE   *fp=NULL;
  char   uplo='L', trans='N', diag='N', side='L';

  //**/ =======================================================
  //**/ fill matrix
  //**/ =======================================================
  //**/ figure out the size of the matrix
  KRI_iter++;
  nBasis = 1;
  if (pOrder_ == 1) nBasis = nInputs_ + 1;

  //**/ fill in the S matrix
  count = 0;
  double *SMatrix = KRI_MatS.getMatrix1D();
  for (jj = 0; jj < nSamples_; jj++)
  {
    //**/ slight enhancement
    //SMatrix[jj*nSamples_+jj] = 1.0 + 1e-15 * nSamples_;
    SMatrix[jj*nSamples_+jj] = 1.0 + 2.22e-16 * (nSamples_ + 10);
    if (KRI_nugget != 0.0) SMatrix[jj*nSamples_+jj] += KRI_nugget;
    else if (nInputs_ == 2) 
      SMatrix[jj*nSamples_+jj] += 0.01;
    if (KRI_VecDataSD.length() > 0) 
    {
      ddata = pow(KRI_VecDataSD[jj]/KRI_YStd,KRI_Exponent);
      SMatrix[jj*nSamples_+jj] += ddata;
    }
    //**/ main loop
    for (kk = jj+1; kk < nSamples_; kk++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs_; ii++)
        dist += pow(KRI_VecXDists[count*nInputs_+ii],KRI_Exponent)*
                thetas[ii];
      dist = exp(-dist);
      //if (dist < 1.0e-16) dist = 0;
      SMatrix[jj*nSamples_+kk] = dist;
      SMatrix[kk*nSamples_+jj] = dist;
      count++;
    }
  }
  //**/ -------------------------------------------------------
  //**/ Cholesky decomposition of S
  //**/ -------------------------------------------------------
  dpotrf_(&uplo, &nSamples_, SMatrix, &nSamples_, &status);
  if (status != 0) 
  {
    printf("Kriging ERROR: Cholesky decomposition not successful.\n");
    exit(1);
  }

  //**/ -------------------------------------------------------
  //**/ Ft = C^{-1} F
  //**/ -------------------------------------------------------
  double *FMatrix = KRI_MatF.getMatrix1D();
  double *FMatTmp = KRI_MatFTmp.getMatrix1D();
  for (ii = 0; ii < nSamples_; ii++) 
  {
    FMatrix[ii] = 1.0;
    FMatTmp[ii] = 1.0;
    if (pOrder_ == 1)
    {
      for (kk = 1; kk < nBasis; kk++) 
      {
        ddata = VecNormalX_[ii*nInputs_+kk-1];
        FMatTmp[kk*nSamples_+ii] = ddata;
        FMatrix[kk*nSamples_+ii] = ddata;
      }
    }
  }
  for (ii = 0; ii < nBasis; ii++) 
  { 
    dtrsv_(&uplo,&trans,&diag,&nSamples_,SMatrix,&nSamples_,
           &FMatTmp[ii*nSamples_], &inc);
  }
  for (ii = 0; ii < nSamples_*nBasis; ii++) 
    FMatrix[ii] = FMatTmp[ii];

  //**/ -------------------------------------------------------
  //**/ [Q, R] = qr(Ft)  (Cmat will have Q, FMatTmp has R)
  //**/ -------------------------------------------------------
  psVector VecTau, VecW, VecQ;
  VecTau.setLength(nSamples_);
  VecW.setLength(nSamples_);
  VecQ.setLength(nSamples_*nBasis);
  dgeqrf_(&nSamples_, &nBasis, FMatTmp, &nSamples_, 
          VecTau.getDVector(),VecW.getDVector(),&nSamples_,&status);
  for (ii = 0; ii < nSamples_*nBasis; ii++) VecQ[ii] = 0.0;
  for (ii = 0; ii < nBasis; ii++) VecQ[ii*nSamples_+ii] = 1.0;
  dormqr_(&side, &trans, &nSamples_, &nBasis, &nBasis, FMatTmp, 
          &nSamples_, VecTau.getDVector(), VecQ.getDVector(), 
          &nSamples_, VecW.getDVector(), &nSamples_, &status);

  //**/ -------------------------------------------------------
  //**/ Ytmp = C^{-1} Y
  //**/ -------------------------------------------------------
  psVector VecYTmp;
  VecYTmp.setLength(nSamples_);
  for (ii = 0; ii < nSamples_; ii++) VecYTmp[ii] = VecNormalX_[ii];
  dtrsv_(&uplo,&trans,&diag,&nSamples_,SMatrix,&nSamples_,
         VecYTmp.getDVector(), &inc);
  //**/ -------------------------------------------------------
  //**/ compute beta
  //**/ -------------------------------------------------------
  VecW.setLength(nBasis);
  for (jj = 0; jj < nBasis; jj++)
  {
    ddata = 0.0;
    for (kk = 0; kk < nSamples_; kk++)
      ddata += VecQ[kk+jj*nSamples_] * VecYTmp[kk];
    VecW[jj] = ddata;
  }
  for (jj = nBasis-1; jj >= 0; jj--)
  {
    ddata = VecW[jj];
    for (kk = jj+1; kk < nBasis; kk++)
      ddata -= VecBetas_[kk] * FMatTmp[kk*nSamples_+jj];
    VecBetas_[jj] = ddata/ FMatTmp[jj*nSamples_+jj];
  }
  //**/ -------------------------------------------------------
  //**/ compute VecRho
  //**/ -------------------------------------------------------
  psVector VecRho;
  VecRho.setLength(nSamples_);
  for (ii = 0; ii < nSamples_; ii++) VecRho[ii] = VecYTmp[ii];
  for (ii = 0; ii < nSamples_; ii++) 
  {
    ddata = 0.0;
    for (kk = 0; kk < nBasis; kk++) 
      ddata += FMatrix[kk*nSamples_+ii] * VecBetas_[kk];
    VecRho[ii] -= ddata;
  }
  //**/ -------------------------------------------------------
  //**/ compute sigma
  //**/ -------------------------------------------------------
  double sigma = 0.0;
  //for (ii = 0; ii < nSamples_; ii++) 
  //  sigma += VecRho[ii] * VecRho[ii];
  for (ii = 0; ii < nSamples_; ii++) sigma += pow(VecRho[ii],2.0);
  sigma /= (double) nSamples_;
  //**/ -------------------------------------------------------
  //**/ compute detR
  //**/ -------------------------------------------------------
  double detR = 1.0;
  for (ii = 0; ii < nSamples_; ii++) 
  {
    ddata = SMatrix[ii*nSamples_+ii];
    detR *= pow(ddata, 2.0/nSamples_);
  }
  //**/ -------------------------------------------------------
  //**/ compute gamma
  //**/ -------------------------------------------------------
  char transT = 'T';
  dtrsv_(&uplo,&transT,&diag,&nSamples_,SMatrix,&nSamples_,
         VecRho.getDVector(), &inc);
  for (ii = 0; ii < nSamples_; ii++) VecGammas_[ii] = VecRho[ii];
  //**/ -------------------------------------------------------
  //**/ compute value
  //**/ -------------------------------------------------------
  retVal = sigma * detR;
  return retVal;
}

// ************************************************************************
// predict 
// ------------------------------------------------------------------------
double Kriging::predict0(int length, double *X, double *Y, double *YStds)
{
  int    ii, jj, kk, nBasis;
  double ddata, Yt;
  psVector VecFMat, VecRMat, VecXT;

  nBasis = 1;
  if (pOrder_ >= 1) nBasis += nInputs_;
  VecFMat.setLength(nBasis);
  VecRMat.setLength(nSamples_);
  VecXT.setLength(nInputs_);

  if (YStds != NULL)
    for (ii = 0; ii < length; ii++) YStds[ii] = 0;

  for (ii = 0; ii < length; ii++) 
  {
    //**/ scale X
    for (jj = 0; jj < nInputs_; jj++) 
      VecXT[jj] = (X[ii*nInputs_+jj] - VecXMeans_[jj])/VecXStds_[jj];

    //*/ construct regression matrix F
    VecFMat[0] = 1.0;
    if (pOrder_ == 1)
      for (kk = 1; kk < nBasis; kk++) VecFMat[kk] = VecXT[kk-1];
    //*/ construct correlation R
    for (jj = 0; jj < nSamples_; jj++)
    {
      ddata = 0.0;
      for (kk = 0; kk < nInputs_; kk++)
        ddata += pow(VecXT[kk]-VecNormalX_[jj*nInputs_+kk],
                 KRI_Exponent)*VecThetasOpt_[kk];
      VecRMat[jj] = exp(-ddata);
    }
    //**/ prediction
    Yt = 0.0;
    for (jj = 0; jj < nBasis; jj++) 
      Yt += VecFMat[jj] * VecBetasOpt_[jj];
    for (jj = 0; jj < nSamples_; jj++) 
      Yt += VecRMat[jj] * VecGammasOpt_[jj];
    Y[ii] = Yt * YStd_ + YMean_;
  }
  return 0;
}

// ************************************************************************
// generate codes 
// ------------------------------------------------------------------------
void Kriging::genRSCode()
{
  int  ii, jj, nBasis;
  FILE *fp = fopen("psuade_rs.info", "w");
  if (fp == NULL) return;

  double *Rmat = MatrixR_.getMatrix1D();
  double *MMat = MatrixM_.getMatrix1D();

  if (pOrder_ == 1) nBasis = nInputs_ + 1;
  else              nBasis = 1;
  fprintf(fp,"This file contains information to re-construct Kriging\n");
  fprintf(fp,"response surface offline. Follow the steps below:\n");
  fprintf(fp,"1. Search for the keywords 'SPLIT HERE' in this file.\n");
  fprintf(fp,"2. Store the lines below keywords into main.c\n");
  fprintf(fp,"3. Add the to-be-evaluated points inside main() in main.c\n");
  fprintf(fp,"   (or, embed all except main() in your own program).\n");
  fprintf(fp,"4. Compile main.c (cc -o main main.c -lm) and run\n");
  fprintf(fp,"   (Do not change the psuade_rs.info file.\n");
  fprintf(fp,"Note: if std dev is desired, uncomment the corresponding\n");
  fprintf(fp,"      section and compile with: cc main.c -llapack -lm\n");
  fprintf(fp,"\n");
  fprintf(fp,"PSUADE_BEGIN\n");
  fprintf(fp, "%d\n", nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
    fprintf(fp, "%24.16e %24.16e %24.16e\n", VecXMeans_[ii], VecXStds_[ii],
            VecThetas_[ii]);
  fprintf(fp, "%24.16e %24.16e\n", YMean_, YStd_);
  fprintf(fp, "%d %d\n", nSamples_, nInputs_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nInputs_; ii++)
      fprintf(fp, "%24.16e ", VecNormalX_[jj*nInputs_+ii]);
    fprintf(fp, "\n");
  }
  fprintf(fp,"%d\n", pOrder_);
  fprintf(fp, "%24.16e \n", VecV1_[0]);
  if (pOrder_ == 1)
  {
    for (ii = 1; ii <= nInputs_; ii++) 
      fprintf(fp, "%24.16e \n", VecV1_[ii]);
  }
  fprintf(fp, "%d\n", nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
    fprintf(fp, "%24.16e \n", VecV2_[jj]);
  fprintf(fp,"%d\n", nSamples_);
  for (jj = 0; jj < nSamples_; jj++)
  {
    for (ii = 0; ii < nSamples_; ii++)
      fprintf(fp, "%24.16e ", Rmat[jj+ii*nSamples_]);
    fprintf(fp, "\n");
  }
  fprintf(fp,"%d\n", nBasis);
  for (jj = 0; jj < nBasis; jj++)
  {
    for (ii = 0; ii < nBasis; ii++)
      fprintf(fp, "%24.16e ", MMat[jj+ii*nBasis]);
    fprintf(fp, "\n");
  }
  fprintf(fp, "%24.16e;\n", KrigingVariance_);
  fprintf(fp,"====================== SPLIT HERE =====================\n");
  fprintf(fp,"/* *******************************************/\n");
  fprintf(fp,"/* Kriging interpolator from PSUADE. */\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"#include <math.h>\n");
  fprintf(fp,"#include <stdlib.h>\n");
  fprintf(fp,"#include <stdio.h>\n");
  fprintf(fp,"int initialize();\n");
  fprintf(fp,"int finalize();\n");
  fprintf(fp,"int interpolate(int,double*,double*,double*);\n");
  fprintf(fp,"int CholSolve(int, double *, double *);\n");
  fprintf(fp,"main(int argc, char **argv) {\n");
  fprintf(fp,"  int    i, iOne=1, nInps;\n");
  fprintf(fp,"  double X[%d], Y, Std;\n",nInputs_);
  fprintf(fp,"  FILE   *fIn=NULL, *fOut=NULL;\n");
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
  fprintf(fp,"  for (i=0; i<%d; i++) fscanf(fIn, \"%%lg\", &X[i]);\n",
          nInputs_);
  fprintf(fp,"  fclose(fIn);\n");
  fprintf(fp,"  initialize();\n");
  fprintf(fp,"  interpolate(iOne, X, &Y, &Std);\n");
  fprintf(fp,"  printf(\"Y = %%e (stdev = %%e)\\n\", Y, Std);\n");
  fprintf(fp,"  finalize();\n");
  fprintf(fp,"  fOut = fopen(argv[2], \"w\");\n");
  fprintf(fp,"  if (fOut == NULL) {\n");
  fprintf(fp,"     printf(\"ERROR: cannot open output file.\\n\");\n");
  fprintf(fp,"     exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fprintf(fOut,\" %%e\\n\", Y);\n");
  fprintf(fp,"  fclose(fOut);\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"/* Regression interpolation function         */\n");
  fprintf(fp,"/* X[0], X[1],   .. X[m-1]   - first point\n");
  fprintf(fp," * X[m], X[m+1], .. X[2*m-1] - second point\n");
  fprintf(fp," * ... */\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"int    pOrder, nInps, nSamples, nBasis;\n");
  fprintf(fp,"double *XMeans=NULL,*XStds=NULL,YMean,YStd,*Thetas=NULL;\n");
  fprintf(fp,"double *V1=NULL,*V2=NULL,*Rmat=NULL,*Mmat=NULL,variance;\n");
  fprintf(fp,"double *XNorm=NULL;\n");
  fprintf(fp,"int interpolate(int npts,double *X,double *Y,\n");
  fprintf(fp,"                double *YStds) {\n");
  fprintf(fp,"  int    ss, ii, jj, iOne=1, status;\n");
  fprintf(fp,"  double dd, *x, *w1, *w2, *wx, sig;\n");
  fprintf(fp,"  char uplo='L';\n");
  fprintf(fp,"  w1 = (double *) malloc(nSamples*sizeof(double));\n");
  fprintf(fp,"  w2 = (double *) malloc(nSamples*sizeof(double));\n");
  fprintf(fp,"  wx = (double *) malloc(2*nBasis*sizeof(double));\n");
  fprintf(fp,"  x  = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  for (ss = 0; ss < npts; ss++) {\n");
  fprintf(fp,"    for (ii = 0; ii < nInps; ii++)\n");
  fprintf(fp,"      x[ii] = (X[ss*nInps+ii]-XMeans[ii])/XStds[ii];\n");
  fprintf(fp,"    for (jj = 0; jj < nSamples; jj++) {\n");
  fprintf(fp,"      sig = 0.0;\n");
  fprintf(fp,"      for (ii = 0; ii < nInps; ii++) {\n");
  fprintf(fp,"        dd = XNorm[jj*nInps+ii] - x[ii];\n");
  fprintf(fp,"        sig += dd * dd / Thetas[ii] / Thetas[ii];\n");
  fprintf(fp,"      }\n");
  fprintf(fp,"      dd = exp(-sig);\n");
  fprintf(fp,"      if (dd < 1e-16) dd = 0.0;\n");
  fprintf(fp,"      w1[jj] = w2[jj] = dd;\n");
  fprintf(fp,"    }\n");
  fprintf(fp,"    dd = V1[0];\n");
  fprintf(fp,"    if (pOrder == 1)\n");
  fprintf(fp,"      for (jj = 1; jj <= nInps; jj++) \n");
  fprintf(fp,"        dd += x[jj-1] * V1[jj];\n");
  fprintf(fp,"    for (jj = 0; jj < nSamples; jj++) \n");
  fprintf(fp,"      dd += w2[jj] * V2[jj];\n");
  fprintf(fp,"    Y[ss] = dd * YStd + YMean;\n");
  fprintf(fp,"    /* ==== if need to compute std dev. =====*/\n");
  fprintf(fp,"    CholSolve(nSamples,Rmat,w2);\n");
  fprintf(fp,"    /*dpotrs_(&uplo,&nSamples,&iOne,Rmat,&nSamples,w2,\n");
  fprintf(fp,"            &nSamples,&status);*/\n");
  fprintf(fp,"    sig = 0.0;\n");
  fprintf(fp,"    for (jj = 0; jj < nSamples; jj++) \n");
  fprintf(fp,"      sig += w1[jj] * w2[jj];\n");
  fprintf(fp,"    dd = 0.0;\n");
  fprintf(fp,"    for (jj = 0; jj < nSamples; jj++) dd += w2[jj];\n");
  fprintf(fp,"    wx[0] = wx[nBasis] = dd - 1;\n");
  if (nBasis > 1)
  {
    fprintf(fp,"    for (ii = 1; ii < nBasis; ii++) {\n");
    fprintf(fp,"      dd = 0.0;\n");
    fprintf(fp,"      for (jj = 0; jj < nSamples; jj++)\n");
    fprintf(fp,"        dd += x[ii-1] * w2[jj];\n");
    fprintf(fp,"      dd -= x[ii-1];\n");
    fprintf(fp,"      wx[ii] = wx[nBasis+ii] = dd;\n");
    fprintf(fp,"    }\n");
  }
  fprintf(fp,"    CholSolve(nBasis,Mmat,wx);\n");
  fprintf(fp,"    /* dpotrs_(&uplo,&nBasis,&iOne,Mmat,&nSamples,\n");
  fprintf(fp,"            wx, &nBasis, &status); */\n");
  fprintf(fp,"    dd = 0.0;\n");
  fprintf(fp,"    for (ii = 0; ii < nBasis; ii++) \n");
  fprintf(fp,"      dd += wx[ii] * wx[ii+nBasis];\n");
  fprintf(fp,"    dd = variance*(1+dd-sig);\n");
  fprintf(fp,"    if (dd < 0) dd = 0.0;\n");
  fprintf(fp,"    YStds[ss] = sqrt(dd) * YStd;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  free(wx);\n");
  fprintf(fp,"  free(w1);\n");
  fprintf(fp,"  free(w2);\n");
  fprintf(fp,"  free(x);\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"int CholSolve(int n, double *A, double *X) {\n");
  fprintf(fp,"  int    ii, jj, kk;\n");
  fprintf(fp,"  double *Y, ddata;\n");
  fprintf(fp,"  Y = (double *) malloc(n*sizeof(double));\n");
  fprintf(fp,"  Y[0] = X[0] / A[0];\n");
  fprintf(fp,"  for (ii = 1; ii < n; ii++) {\n");
  fprintf(fp,"     ddata = 0.0;\n");
  fprintf(fp,"     for (jj = 0; jj < ii; jj++) \n");
  fprintf(fp,"       ddata += A[jj*n+ii] * Y[jj];\n");
  fprintf(fp,"     Y[ii] = (X[ii] - ddata) / A[ii*n+ii];\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  X[n-1] = Y[n-1] / A[(n-1)*n+n-1];\n");
  fprintf(fp,"  for (ii = n-2; ii >= 0; ii--) {\n");
  fprintf(fp,"     ddata = 0.0;\n");
  fprintf(fp,"     for (jj = ii+1; jj < n; jj++) \n");
  fprintf(fp,"       ddata += A[ii*n+jj] * X[jj];\n");
  fprintf(fp,"     X[ii] = (Y[ii] - ddata) / A[ii*n+ii];\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  free(Y);\n");
  fprintf(fp,"  return 0;\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"int initialize() {\n");
  fprintf(fp,"  int    ii, jj;\n");
  fprintf(fp,"  double ddata;\n");
  fprintf(fp,"  char   line[1001], word[1001];\n");
  fprintf(fp,"  FILE *fp = fopen(\"psuade_rs.info\", \"r\");\n");
  fprintf(fp,"  if (fp == NULL){\n");
  fprintf(fp,"     printf(\"Data file (psuade_rs.info) not found.\\n\");\n");
  fprintf(fp,"     exit(1);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  while (1) {\n");
  fprintf(fp,"    fgets(line, 500, fp);\n");
  fprintf(fp,"    sscanf(line, \"%%s\",word);\n");
  fprintf(fp,"    if (!strcmp(word, \"PSUADE_BEGIN\")) break;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fp, \"%%d\", &nInps);\n");
  fprintf(fp,"  XMeans = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  XStds  = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  Thetas = (double *) malloc(nInps*sizeof(double));\n");
  fprintf(fp,"  for (ii = 0; ii < nInps; ii++) {\n");
  fprintf(fp,"     fscanf(fp, \"%%lg\", &ddata);\n");
  fprintf(fp,"     XMeans[ii] = ddata;\n");
  fprintf(fp,"     fscanf(fp, \"%%lg\", &ddata);\n");
  fprintf(fp,"     XStds[ii] = ddata;\n");
  fprintf(fp,"     fscanf(fp, \"%%lg\", &ddata);\n");
  fprintf(fp,"     Thetas[ii] = ddata;\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fp, \"%%lg %%lg\", &YMean, &YStd);\n");
  fprintf(fp,"  fscanf(fp, \"%%d %%d\", &nSamples, &nInps);\n");
  fprintf(fp,"  XNorm = (double *) malloc(nSamples*nInps*sizeof(double));\n");
  fprintf(fp,"  for (jj = 0; jj < nSamples; jj++) \n");
  fprintf(fp,"    for (ii = 0; ii < nInps; ii++) \n");
  fprintf(fp,"      fscanf(fp, \"%%lg\", &XNorm[jj*nInps+ii]);\n");
  fprintf(fp,"  fscanf(fp, \"%%d\", &pOrder);\n");
  fprintf(fp,"  V1 = (double *) malloc((nInps+1)*sizeof(double));\n");
  fprintf(fp,"  fscanf(fp, \"%%lg\", &V1[0]);\n");
  fprintf(fp,"  if (pOrder == 1) { \n");
  fprintf(fp,"    for (ii = 1; ii <= nInps; ii++)\n");
  fprintf(fp,"        fscanf(fp, \"%%24.16e\", &V1[ii]);\n");
  fprintf(fp,"  }\n");
  fprintf(fp,"  fscanf(fp, \"%%d\", &nSamples);\n");
  fprintf(fp,"  V2 = (double *) malloc(nSamples*sizeof(double));\n");
  fprintf(fp,"  for (ii = 0; ii < nSamples; ii++)\n");
  fprintf(fp,"    fscanf(fp, \"%%lg\", &V2[ii]);\n");
  fprintf(fp,"  fscanf(fp, \"%%d\", &nSamples);\n");
  fprintf(fp,"  Rmat=(double*) malloc(nSamples*nSamples*sizeof(double));\n");
  fprintf(fp,"  for (jj = 0; jj < nSamples; jj++) \n");
  fprintf(fp,"    for (ii = 0; ii < nSamples; ii++)\n");
  fprintf(fp,"      fscanf(fp, \"%%lg \", &Rmat[jj+ii*nSamples]);\n");
  fprintf(fp,"  fscanf(fp, \"%%d\", &nBasis);\n");
  fprintf(fp,"  Mmat=(double*) malloc(nBasis*nBasis*sizeof(double));\n");
  fprintf(fp,"  for (jj = 0; jj < nBasis; jj++) \n");
  fprintf(fp,"    for (ii = 0; ii < nBasis; ii++)\n");
  fprintf(fp,"      fscanf(fp, \"%%lg \", &Mmat[jj+ii*nBasis]);\n");
  fprintf(fp,"  fscanf(fp, \"%%lg\", &variance);\n");
  fprintf(fp,"  fclose(fp);\n");
  fprintf(fp,"  return 0;\n");
  fprintf(fp,"}\n");
  fprintf(fp,"/* ==========================================*/\n");
  fprintf(fp,"int finalize() {\n");
  fprintf(fp,"  if (V1 != NULL) free(V1);\n");
  fprintf(fp,"  if (V2 != NULL) free(V2);\n");
  fprintf(fp,"  if (Thetas != NULL) free(Thetas);\n");
  fprintf(fp,"  if (XMeans != NULL) free(XMeans);\n");
  fprintf(fp,"  if (XStds != NULL) free(XStds);\n");
  fprintf(fp,"  if (Mmat != NULL) free(Mmat);\n");
  fprintf(fp,"  if (Rmat != NULL) free(Rmat);\n");
  fprintf(fp,"  if (XNorm != NULL) free(XNorm);\n");
  fprintf(fp,"}\n");
  fclose(fp);
  printf("Kriging response surface data file is in psuade_rs.info\n");
  fp = fopen("psuade_rs.py", "w");
  if (fp == NULL)
  {
    printf("ERROR: Cannot open file psuade_rs.py.\n");
    return;
  }
  fwriteRSPythonHeader(fp);
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"# Kriging interpolation\n");
  fprintf(fp,"#==================================================\n");
  fwriteRSPythonCommon(fp);
  fprintf(fp, "nSamples = %d\n", nSamples_);
  fprintf(fp, "nInputs = %d\n", nInputs_);
  fprintf(fp, "nBasis = %d\n", nBasis);
  fprintf(fp, "KrigVariance = %24.16e\n", KrigingVariance_);
  fprintf(fp, "XParams = [\n");
  for (ii = 0; ii < nInputs_; ii++)
    fprintf(fp, "[%24.16e , %24.16e , %24.16e ],\n", VecXMeans_[ii], 
            VecXStds_[ii],
            VecThetas_[ii]);
  fprintf(fp, "]\n");
  fprintf(fp, "Ymean = %24.16e\n", YMean_);
  fprintf(fp, "Ystd  = %24.16e\n", YStd_);
  fprintf(fp, "SamInputs = [\n");
  for (jj = 0; jj < nSamples_; jj++)
  {
    fprintf(fp, "[ %24.16e ", VecNormalX_[jj*nInputs_]);
    for (ii = 1; ii < nInputs_; ii++)
      fprintf(fp, ", %24.16e ", VecNormalX_[jj*nInputs_+ii]);
    fprintf(fp, "], \n");
  }
  fprintf(fp, "]\n");
  fprintf(fp, "V1 = [\n");
  fprintf(fp, " %24.16e , \n", VecV1_[0]);
  if (pOrder_ == 1)
  {
    for (ii = 1; ii <= nInputs_; ii++)
      fprintf(fp, " %24.16e ,  \n", VecV1_[ii]);
  }
  fprintf(fp, "]\n");
  fprintf(fp, "V2 = [\n");
  for (jj = 0; jj < nSamples_; jj++)
    fprintf(fp, " %24.16e , \n", VecV2_[jj]);
  fprintf(fp, "]\n");
  fprintf(fp, "Rmat = [\n");
  for (jj = 0; jj < nSamples_; jj++)
  {
    fprintf(fp, " [ %24.16e ", Rmat[jj]);
    for (ii = 1; ii < nSamples_; ii++)
      fprintf(fp, " , %24.16e ", Rmat[jj+ii*nSamples_]);
    fprintf(fp, "], \n");
  }
  fprintf(fp, "]\n");
  fprintf(fp, "Mmat = [\n");
  for (jj = 0; jj < nBasis; jj++)
  {
    fprintf(fp, " [ %24.16e ", MMat[jj]);
    for (ii = 1; ii < nBasis; ii++)
      fprintf(fp, " , %24.16e ", MMat[jj+ii*nBasis]);
    fprintf(fp, "], \n");
  }
  fprintf(fp, "]\n");
  fprintf(fp,"###################################################\n");
  fprintf(fp,"# solve for A X = B\n");
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"def MatSolve(n, Amat, B): \n");
  fprintf(fp,"  Y = n * [0.0]\n");
  fprintf(fp,"  Y[0] = B[0] / Amat[0][0]\n");
  fprintf(fp,"  for ii in range(n-1): \n");
  fprintf(fp,"    jj = ii + 1\n");
  fprintf(fp,"    ddata = 0.0\n");
  fprintf(fp,"    for kk in range(jj): \n");
  fprintf(fp,"      ddata += Amat[jj][kk] * Y[kk]\n");
  fprintf(fp,"    Y[jj] = (B[jj] - ddata)/Amat[jj][jj]\n");
  fprintf(fp,"  X = n * [0.0]\n");
  fprintf(fp,"  X[n-1] = Y[n-1] / Amat[n-1][n-1]\n");
  fprintf(fp,"  for ii in range(n-1): \n");
  fprintf(fp,"    jj = n - ii - 2\n");
  fprintf(fp,"    ddata = 0.0\n");
  fprintf(fp,"    for kk in range(ii+1): \n");
  fprintf(fp,"      kk2 = n - kk - 1\n");
  fprintf(fp,"      ddata += Amat[kk2][jj] * X[kk2]\n");
  fprintf(fp,"    X[jj] = (Y[jj] - ddata)/Amat[jj][jj]\n");
  fprintf(fp,"  return X\n");
  fprintf(fp,"###################################################\n");
  fprintf(fp,"# Interpolation function  \n");
  fprintf(fp,"# X[0], X[1],   .. X[m-1]   - first point\n");
  fprintf(fp,"# X[m], X[m+1], .. X[2*m-1] - second point\n");
  fprintf(fp,"# ... \n");
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"def interpolate(XX): \n");
  fprintf(fp,"  npts = int(len(XX) / nInputs + 1e-8)\n");
  fprintf(fp,"  Xt = nInputs * [0.0]\n");
  fprintf(fp,"  Ys = 2 * npts * [0.0]\n");
  fprintf(fp,"  W1 = nSamples * [0.0]\n");
  fprintf(fp,"  W2 = nSamples * [0.0]\n");
  fprintf(fp,"  WX = (2 * nBasis) * [0.0]\n");
  fprintf(fp,"  for ss in range(npts) : \n");
  fprintf(fp,"    for ii in range(nInputs) : \n");
  fprintf(fp,
     "      Xt[ii] = (XX[ss*nInputs+ii]-XParams[ii][0])/XParams[ii][1]\n");
  fprintf(fp,"    for jj in range(nSamples) : \n");
  fprintf(fp,"      sig = 0.0\n");
  fprintf(fp,"      for ii in range(nInputs) : \n");
  fprintf(fp,"        dd = SamInputs[jj][ii] - Xt[ii];\n");
  fprintf(fp,"        sig = sig + dd*dd/(XParams[ii][2]*XParams[ii][2])\n");
  fprintf(fp,"      dd = math.exp(-sig)\n");
  fprintf(fp,"      if (dd < 1e-16) : \n");
  fprintf(fp,"        dd = 0.0\n");
  fprintf(fp,"      W1[jj] = dd\n");
  fprintf(fp,"      W2[jj] = dd\n");
  fprintf(fp,"    dd = V1[0]\n");
  if (pOrder_ == 1)
  {
    fprintf(fp,"    for ii in range(nInputs) : \n");
    fprintf(fp,"      dd = dd + Xt[ii] * V1[ii+1]\n");
  }
  fprintf(fp,"    for jj in range(nSamples) : \n");
  fprintf(fp,"      dd = dd + W2[jj] * V2[jj]\n");
  fprintf(fp,"    Ys[2*ss] = dd * Ystd + Ymean\n");
  fprintf(fp,"    W2a = MatSolve(nSamples,Rmat,W2)\n");
  fprintf(fp,"    sig = 0.0\n");
  fprintf(fp,"    for jj in range(nSamples) : \n");
  fprintf(fp,"      sig += W1[jj] * W2a[jj]\n");
  fprintf(fp,"    dd = 0.0;\n");
  fprintf(fp,"    for jj in range(nSamples) : \n");
  fprintf(fp,"      dd = dd + W2a[jj]\n");
  fprintf(fp,"    WX[0] = WX[nBasis] = dd - 1\n");
  if (nBasis > 1)
  {
    fprintf(fp,"    for ii in range(nBasis-1) : \n");
    fprintf(fp,"      dd = 0.0\n");
    fprintf(fp,"      for jj in range(nSamples) : \n");
    fprintf(fp,"        dd = dd + Xt[ii] * W2a[jj]\n");
    fprintf(fp,"      dd = dd - Xt[ii];\n");
    fprintf(fp,"      WX[ii+1] = dd\n");
    fprintf(fp,"      WX[nBasis+ii+1] = dd\n");
  }
  fprintf(fp,"    WXa = MatSolve(nBasis,Mmat,WX)\n");
  fprintf(fp,"    dd = 0.0\n");
  fprintf(fp,"    for ii in range(nBasis) : \n");
  fprintf(fp,"      dd = dd + WXa[ii] * WX[ii]\n");
  fprintf(fp,"    dd = KrigVariance*(1+dd-sig)\n");
  fprintf(fp,"    if (dd < 0) :\n"); 
  fprintf(fp,"      dd = 0.0\n");
  fprintf(fp,"    Ys[2*ss+1] = math.sqrt(dd) * Ystd\n");
  fprintf(fp,"  return Ys\n");
  fprintf(fp,"###################################################\n");
  fprintf(fp,"# main program\n");
  fprintf(fp,"#==================================================\n");
  fprintf(fp,"infileName  = sys.argv[1]\n");
  fprintf(fp,"outfileName = sys.argv[2]\n");
  fprintf(fp,"inputs = getInputData(infileName)\n");
  fprintf(fp,"outputs = interpolate(inputs)\n");
  fprintf(fp,"genOutputFile(outfileName, outputs)\n");
  fprintf(fp,"###################################################\n");
  fclose(fp);
  printf("FILE psuade_rs.py contains the Kriging interpolator.\n");
  return;
}


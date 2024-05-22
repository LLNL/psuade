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
// Functions for FunctionInterface 
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <PsuadeCmakeConfig.h>

#ifdef WINDOWS
#include <windows.h>
#endif

#include <assert.h>
#include <stdio.h>
#include <string.h>
#include <unistd.h>
#include "dtype.h"
#include "PsuadeUtil.h"
#include "FunctionInterface.h"
#include "FuncApprox.h"
#include "pData.h"
#include "PDFBase.h"
#include "PsuadeData.h"
#include "PsuadeConfig.h"
#include "Psuade.h"
#include "ProbMatrix.h"
#include "MCMCAnalyzer.h"
#include "GPBasedSampling.h"
#include "MCMCAnalyzer.h"
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
FunctionInterface::FunctionInterface()
{ 
  nInputs_ = 0;
  nOutputs_ = 0;
  printLevel_ = 0;

  //**/ ---------------------------------------------------------------
  //**/ interface files for running simulation
  //**/ ---------------------------------------------------------------
  strcpy(appDriver_, "true");
  strcpy(optDriver_, "true");
  strcpy(auxOptDriver_, "true");
  strcpy(ensembleDriver_, "true");
  strcpy(ensembleOptDriver_, "true");
  strcpy(appInputTemplate_, "psuadeApps.in");
  strcpy(appOutputTemplate_, "psuadeApps.out");

  //**/ ---------------------------------------------------------------
  //**/ synchronous or asynchronous modes, simulation or opt driver
  //**/ ---------------------------------------------------------------
  executionMode_ = 0;
  launchInterval_ = 0;
  appOptFlag_ = 0;

  //**/ ---------------------------------------------------------------
  //**/ simulation model can also be RS model. rsInps_ is needed 
  //**/    if nInputs in the response surface model used is greater 
  //**/    than nInputs in 'evaluate' (e.g. in loaded sample into
  //**/    resident memory)
  //**/ rsRanFlag_: stochastic RS mode: 1: -3 sigma, 2: +3 sigma
  //**/             (This feature is not currently used)
  //**/ rsDoNotDelete_ : for preventing copied function pointers from
  //**/                  being deleted (RS object pointer may be
  //**/                  copied when duplicated)
  //**/ ---------------------------------------------------------------
  useRSModel_ = 0;
  rsPtrs_ = NULL;
  rsnInps_ = 0;
  rsRanFlag_ = 0;
  rsDoNotDelete_ = 0;

  //**/ ---------------------------------------------------------------
  //**/ select which local function to use
  //**/ ---------------------------------------------------------------
  whichLocalFunction_ = 0;
}

// ************************************************************************
// Copy Constructor by Bill Oliver
// ------------------------------------------------------------------------
FunctionInterface::FunctionInterface(const FunctionInterface & fi)
{
  nInputs_ = fi.nInputs_;
  nOutputs_ = fi.nOutputs_;
  strcpy(appDriver_, fi.appDriver_);
  strcpy(optDriver_, fi.optDriver_);
  strcpy(auxOptDriver_, fi.auxOptDriver_);
  strcpy(ensembleDriver_, fi.ensembleDriver_);
  strcpy(ensembleOptDriver_, fi.ensembleOptDriver_);
  strcpy(appInputTemplate_, fi.appInputTemplate_);
  strcpy(appOutputTemplate_, fi.appOutputTemplate_);
  rsRanFlag_ = fi.rsRanFlag_;
  printLevel_ = fi.printLevel_;
  rsnInps_ = fi.rsnInps_;

  //**/ ---------------------------------------------------------------
  //**/ copy input and output names
  //**/ ---------------------------------------------------------------
  StrInpNames_ = fi.StrInpNames_;
  StrOutNames_ = fi.StrOutNames_;

  //**/ ---------------------------------------------------------------
  //**/ copy rsPtrs_
  //**/ FuncApprox is a class whose copy constructor has been created 
  //**/ and the operator= has been defined so we can just do a copy
  //**/ As such, the copied function pointers are not to be deleted
  //**/ (the first instance should delete it, not the copied instances)
  //**/ ---------------------------------------------------------------
  rsPtrs_ = new FuncApprox*[nOutputs_];
  for (int ii = 0; ii < nOutputs_; ii++) rsPtrs_[ii] = fi.rsPtrs_[ii];
  rsDoNotDelete_ = 1;

  //**/ ---------------------------------------------------------------
  //**/ copy VecRSInds_ and VecRSVals_, which are used if nInputs in
  //**/ the sample used to build response surface is greater than the
  //**/ nInputs when evaluate is called
  //**/ ---------------------------------------------------------------
  VecRSInds_ = fi.VecRSInds_;
  VecRSVals_ = fi.VecRSVals_;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
FunctionInterface::~FunctionInterface()
{ 
  if (rsPtrs_ != NULL)
  {
    if (rsDoNotDelete_ == 0)
    {
      for (int kk = 0; kk < nOutputs_; kk++)
        if (rsPtrs_[kk] != NULL) delete rsPtrs_[kk];
    }
    delete [] rsPtrs_;
  }
  StrInpNames_.clean();
  StrOutNames_.clean();
  VecRSInds_.clean();
  VecRSVals_.clean();
}

// ************************************************************************
// load input data (input names and number of inputs)
// ------------------------------------------------------------------------
int FunctionInterface::loadInputData(int nInputs, char **names)
{ 
  char pString[101];
  if (nInputs <= 0)
  {
    printf("FuncIO LoadInput ERROR: nInputs <= 0\n");
    return 1; 
  }
  nInputs_ = nInputs;
  StrInpNames_.setNumStrings(nInputs_);
  for (int ii = 0; ii < nInputs; ii++)
  {
    if (names != NULL && names[ii] != NULL)
      StrInpNames_.loadOneString(ii, names[ii]);
    else 
    {
      snprintf(pString,100, "X%d", ii);
      StrInpNames_.loadOneString(ii, pString);
    }
  }
  return 0;
}

// ************************************************************************
// load output data (output names and number of outputs) 
// ------------------------------------------------------------------------
int FunctionInterface::loadOutputData(int nOutputs, char **names)
{ 
  char pString[101];
  if (nOutputs <= 0)
  {
    printf("FuncIO loadOutput ERROR: nOutputs <= 0\n");
    return 1; 
  }
  nOutputs_ = nOutputs;
  StrOutNames_.setNumStrings(nOutputs_);
  for (int ii = 0; ii < nOutputs_; ii++)
  {
    if (names != NULL && names[ii] != NULL)
         strcpy(pString, names[ii]);
    else snprintf(pString,100,"Y%d", ii);
    StrOutNames_.loadOneString(ii, pString);
  }
  return 0;
}

// ************************************************************************
// load the Function information 
// ------------------------------------------------------------------------
int FunctionInterface::loadFunctionData(int length, char **names)
{
  //**/ ---------------------------------------------------------------
  //**/ error checking (need 5 or more parameters)
  //**/ ---------------------------------------------------------------
  if (length < 5)
  {
    printf("FuncIO loadFunction ERROR: length < 5\n");
    return 1; 
  }

  //**/ ---------------------------------------------------------------
  //**/ copy in the drivers 
  //**/ useRSModel = 0 :  use externally given driver
  //**/ useRSModel = 1 :  response surface used as function
  //**/ useRSModel = 2 :  driver uses local function
  //**/ useRSModel = 3 :  opt_driver uses local function
  //**/ useRSModel = 4 :  both drivers use local function
  //**/ useRSModel = 6 :  ensemble driver function
  //**/ useRSModel = 7 :  ensemble opt driver function
  //**/ ---------------------------------------------------------------
  strcpy(appDriver_, names[0]);
  if (strcmp(names[1], "NONE")) strcpy(appInputTemplate_, names[1]);
  if (strcmp(names[2], "NONE")) strcpy(appOutputTemplate_, names[2]);
  strcpy(optDriver_, names[3]);
  strcpy(auxOptDriver_, names[4]);
  if (length >= 6) strcpy(ensembleDriver_, names[5]);
  if (length >= 7) strcpy(ensembleOptDriver_, names[6]);

  //**/ ---------------------------------------------------------------
  //**/ set up response surface, if requested
  //**/ ---------------------------------------------------------------
  setDriver(appOptFlag_);
 
  return 0;
}

// ************************************************************************
// set execution mode to be asynchronous
// ------------------------------------------------------------------------
int FunctionInterface::setAsynchronousMode()
{
  executionMode_ = 1;
  return 0;
}

// ************************************************************************
// set execution mode to be synchronous
// ------------------------------------------------------------------------
int FunctionInterface::setSynchronousMode()
{
  executionMode_ = 0;
  return 0;
}

// ************************************************************************
// set stochastic response surface mode (1:minus 3 sigma, 2: plus 3 sigma)
// ------------------------------------------------------------------------
int FunctionInterface::setStochasticRSMode(int indata)
{
  rsRanFlag_ = indata;
  return 0;
}

// ************************************************************************
// set time interval between asynchronous jobs
// ------------------------------------------------------------------------
int FunctionInterface::setLaunchInterval(int interval)
{
  launchInterval_ = interval;
  if (launchInterval_ < 0) launchInterval_ = 0;
  if (launchInterval_ > 100000) launchInterval_ = 0;
  return 0;
}

// ************************************************************************
// set print level
// ------------------------------------------------------------------------
int FunctionInterface::setOutputLevel(int level)
{
  printLevel_ = level;
  if (printLevel_ > 10) printLevel_ = 4;
  return 0;
}

// ************************************************************************
// choose between different local functions 
// ------------------------------------------------------------------------
int FunctionInterface::setLocalFunction(int problem)
{
  whichLocalFunction_ = problem;
  if (printLevel_ > 0)
  { 
    if (problem == 0)
      printf("FuncIO setLocalFunc: default OUU example\n");
    else if (problem == 10)
      printf("FuncIO setLocalFunc: ODOE_GOPTIMAL\n");
    else if (problem == 11)
      printf("FuncIO setLocalFunc: ODOE_IOPTIMAL\n");
    else if (problem == 12)
      printf("FuncIO setLocalFunc: ODOE_DOPTIMAL\n");
    else if (problem == 13)
      printf("FuncIO setLocalFunc: ODOE_AOPTIMAL\n");
    else if (problem == 14)
      printf("FuncIO setLocalFunc: ODOE_EOPTIMAL\n");
    else if (problem == 20)
      printf("FuncIO setLocalFunc: ODOE_GOPTIMAL (Bayes)\n");
    else if (problem == 21)
      printf("FuncIO setLocalFunc: ODOE_IOPTIMAL (Bayes)\n");
    else if (problem == 22)
      printf("FuncIO setLocalFunc: ODOE_DOPTIMAL (Bayes)\n");
    else if (problem == 23)
      printf("FuncIO setLocalFunc: ODOE_AOPTIMAL (Bayes)\n");
    else if (problem == 24)
      printf("FuncIO setLocalFunc: ODOE_EOPTIMAL (Bayes)\n");
    else if (problem == 25)
      printf("FuncIO setLocalFunc: ODOE_GOPTIMAL (Fisher)\n");
    else if (problem == 26)
      printf("FuncIO setLocalFunc: ODOE_IOPTIMAL (Fisher)\n");
    else if (problem == 27)
      printf("FuncIO setLocalFunc: ODOE_DOPTIMAL (Fisher)\n");
    else if (problem == 28)
      printf("FuncIO setLocalFunc: ODOE_AOPTIMAL (Fisher)\n");
    else if (problem == 29)
      printf("FuncIO setLocalFunc: ODOE_EOPTIMAL (Fisher)\n");
    else if (problem == 30)
      printf("FuncIO setLocalFunc: ODOE_MMD\n");
    else if (problem == 31)
      printf("FuncIO setLocalFunc: ODOE_MMV\n");
    else if (problem == 32)
      printf("FuncIO setLocalFunc: ODOE_MAV\n");
    else if (problem == 45)
      printf("FuncIO setLocalFunc: ODOE_GOPTIMAL (Fisher-Jacobi)\n");
    else if (problem == 46)
      printf("FuncIO setLocalFunc: ODOE_IOPTIMAL (Fisher-Jacobi)\n");
    else if (problem == 47)
      printf("FuncIO setLocalFunc: ODOE_DOPTIMAL (Fisher-Jacobi)\n");
    else if (problem == 48)
      printf("FuncIO setLocalFunc: ODOE_AOPTIMAL (Fisher-Jacobi)\n");
    else if (problem == 49)
      printf("FuncIO setLocalFunc: ODOE_EOPTIMAL (Fisher-Jacobi)\n");
    else if (problem == 999)
      printf("FuncIO setLocalFunction: clean up\n");
  }
  if (problem == 999)
  {
    psLocalFunction(0, NULL, 0, NULL);
    whichLocalFunction_ = 0;
  }
  return 0;
}

// ************************************************************************
// set which driver to use
// ------------------------------------------------------------------------
int FunctionInterface::setDriver(int which)
{
  int    ii, kk, nInps, status;
  char   inString[2000], fname[2000], *cString, pString[1001];
  char   winput1[1001], winput2[1001], winput3[1001];
  double ddata;
  FILE   *fp;
  PsuadeData *psIO;
  pData pPtr;

  if (printLevel_ > 4) printf("FuncIO setDriver = %d\n",which);

  //**/ ---------------------------------------------------------------
  //**/ select application or optimization driver
  //**/ appOptFlag = 0 :  use application driver
  //**/ appOptFlag = 1 :  use optimization driver
  //**/ appOptFlag = 2 :  use auxiliary optimization driver
  //**/                   (used for 2-level optimization)
  //**/ ---------------------------------------------------------------
  if      (which >= 2) appOptFlag_ = 2;
  else if (which == 1) appOptFlag_ = 1;
  else                 appOptFlag_ = 0;

  //**/ ---------------------------------------------------------------
  //**/ clear previously-allocated response surfaces
  //**/ ---------------------------------------------------------------
  if (rsPtrs_ != NULL)
  {
    if (rsDoNotDelete_ == 0)
    {
      for (ii = 0; ii < nOutputs_; ii++)
        if (rsPtrs_[ii] != NULL) delete rsPtrs_[ii];
    }
    delete [] rsPtrs_;
    rsDoNotDelete_ = 0;
    rsPtrs_ = NULL;
  }
  VecRSInds_.clean();
  VecRSVals_.clean();

  //**/ ---------------------------------------------------------------
  //**/ set useRSModel and choose the driver
  //**/ ---------------------------------------------------------------
  useRSModel_ = 0;
  if (!strcmp(appDriver_,"PSUADE_LOCAL") &&
       strcmp(optDriver_,"PSUADE_LOCAL"))
  {
    useRSModel_ = 2;
    return 0; 
  }
  if (strcmp(appDriver_,"PSUADE_LOCAL") &&
      !strcmp(optDriver_,"PSUADE_LOCAL"))
  {
    useRSModel_ = 3;
    return 0; 
  }
  if (!strcmp(appDriver_,"PSUADE_LOCAL") &&
      !strcmp(optDriver_,"PSUADE_LOCAL"))
  {
    useRSModel_ = 4;
    return 0; 
  }

  //**/ ---------------------------------------------------------------
  //**/ probe to see if response surface model is requested
  //**/ (if the driver name is in PSUADE format (with PSUADE_IO), then
  //**/ response surface is requested)
  //**/ construct response surfaces if requested
  //**/ ---------------------------------------------------------------
  if      (appOptFlag_ == 0) strcpy(fname, appDriver_);
  else if (appOptFlag_ == 1) strcpy(fname, optDriver_);
  if      (appOptFlag_ == 2) strcpy(fname, auxOptDriver_);
  fp = fopen(fname, "r");
  if (fp != NULL)
  {
    fscanf(fp, "%10c", inString);
    if (!strncmp(inString, "PSUADE_IO",9)) useRSModel_ = 1;
    fclose(fp);

    //**/ probe the RS file (in PSUADE format) to check everything
    if (useRSModel_ == 1)
    {
      psIO = new PsuadeData();
      psIO->setOutputLevel(0);
      status = psIO->readPsuadeFile(fname);
      if (status != 0)
      {
        printf("ERROR: cannot read file %s in PSUADE format\n",fname);
        printf("Where: file %s line %d, exiting\n", __FILE__, __LINE__);
        exit(1);
      }
      psIO->getParameter("input_ninputs", pPtr);
      int rsModelnInps = pPtr.intData_;

      if (printLevel_ > 0)
      {
        printAsterisks(PL_INFO,0);
        printf("FuncIO: Setting up RS driver from %s.\n",fname);
        printEquals(PL_INFO,0);
      }

      //**/ if rsModelnInps > nInputs, expect an rs_index_file
      if (rsModelnInps > nInputs_)
      {
        printf("FuncIO WARNING: nInputs in RS driver does not match ");
        printf("nInputs in the\n");
        printf("                original psuade file.\n");
        printf("   nInputs in original psuade file = %d\n",nInputs_);
        printf("   nInputs in RS driver data file  = %d\n",rsModelnInps);
        printf("ACTION: See if an RS index file can be found.\n");
        printf("NOTE: the number of fixed parameters in the RS ");
        printf("index file should be %d\n",PABS(nInputs_-rsModelnInps));

        //**/ if nInputs mismatch has not been resolved, see if
        //**/ a RS index file can be found in the RS data file
        psIO->getParameter("ana_rsindexfile", pPtr);

        //**/ if there is an index file in the RS data file
        if (strcmp(pPtr.strArray_[0], "NONE"))
        {
          printf("FuncIO INFO: rs_index_file found in RS driver file.\n");
          fp = fopen(pPtr.strArray_[0], "r");
          if (fp == NULL)
          {
            printf("ERROR: Missing rs_index_file %s in current folder.\n",
                   pPtr.strArray_[0]);
            printf("Where: file %s line %d, exiting\n", __FILE__, __LINE__);
            exit(1);
          }
          else
          {
            fscanf(fp,"%d", &nInps);
            rsnInps_ = nInps;
            if (rsModelnInps != nInps)
            {
              printf("ERROR: Invalid nInputs in rs index file.\n");
              printf("  It has to match nInputs in the RS data file.\n");
              printf("  nInputs in RS index file  = %d\n",nInps);
              printf("  nInputs in RS sample file = %d\n",rsModelnInps);
              printf("  Data format in rs index file should be: \n");
              printf("  line 1: nInputs in RS driver data file\n");
              printf("  line 2: 1 <num> <num == 0 ==> fixed>\n");
              printf("  line 3: 2 <num> <0 if num != 0 (active)>\n");
              printf("  line 4: 3 <num> <num == 0 ==> fixed>\n");
              printf("  line 5: 4 <num> <0 if num != 0 (active)>\n");
              printf("  ...\n");
              printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
              exit(1);
            }
            VecRSInds_.setLength(nInps);
            VecRSVals_.setLength(nInps);
            for (ii = 0; ii < nInps; ii++) VecRSInds_[ii] = 0;
            for (ii = 0; ii < nInps; ii++)
            {
              fscanf(fp, "%d", &kk);
              if (kk != ii+1)
              {
                printf("ERROR: first index in rs index file = %d.\n",kk);
                printf("       Must be equal to %d.\n",ii+1);
                printf("  Data format in rs index file should be: \n");
                printf("  line 1: nInputs in RS driver data file\n");
                printf("  line 2: 1 <num> <fixed value if num == 0>\n");
                printf("  line 3: 2 <num> <0 if num != 0 (active)>\n");
                printf("  line 4: 3 <num> <num == 0 ==> fixed>\n");
                printf("  line 5: 4 <num> <0 if num != 0 (active)>\n");
                printf("  ...\n");
                printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
                exit(1);
              } 
              fscanf(fp, "%d", &kk);
              VecRSInds_[ii] = kk;
              if (VecRSInds_[ii] < 0 || VecRSInds_[ii] > nInputs_)
              {
                printf("ERROR: input %3d = %d not valid\n",ii+1,
                       VecRSInds_[ii]);
                printf("       Need to be between 1 and %d\n",nInputs_);
                printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
                exit(1);
              }
              VecRSInds_[ii]--; /* change to zero-based */
              fscanf(fp, "%lg", &ddata);
              VecRSVals_[ii] = ddata;
              if (VecRSInds_[ii] == -1)
                printf("   RS Input %d inactive, fixed at %16.8e\n",
                       ii+1, VecRSVals_[ii]);
              else
              {
                printf("   RS Input %d   active, mapped to",ii+1);
                printf(" PSUADE input %d\n", VecRSInds_[ii]+1);
              }
            }
            fclose(fp);
          }
        }
        else
        {
          printf("ERROR: nInputs mismatch and no RS index file.\n");
          printf("       Cannot proceed.\n");
          printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
          exit(0);
        }
      }

      //**/ if rsModelnInps < nInputs, expect fixed variables 
      if (rsModelnInps < nInputs_)
      {
        printf("FuncIO WARNING: nInputs in RS driver does not match ");
        printf(" nInputs in the\n");
        printf("                original psuade file.\n");
        printf("   nInputs in original psuade file = %d\n",nInputs_);
        printf("   nInputs in RS driver data file  = %d\n",rsModelnInps);
        printf("ACTION: See if FIXED variables have been declared.\n");
        printf("NOTE: the number of fixed parameters should be %d\n",
               PABS(nInputs_-rsModelnInps));
        psIO->getParameter("input_names", pPtr);
        if (pPtr.strArray_ != NULL)
        {
          cString = psConfig_.getParameter("num_fixed");
          if (cString != NULL)
          {
            sscanf(cString, "%s %s %d", winput1, winput2, &nInps);
            if (nInps != nInputs_-rsModelnInps)
            {
              printf("ERROR: the number of fixed parameters = %d\n",
                     nInps);
              printf("       but it should be equal to %d\n",
                     nInputs_-rsModelnInps);
              printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
              exit(1);
            }

            //**/ set VecRSInds_ assuming all are not fixed variables
            VecRSInds_.setLength(rsModelnInps);
            VecRSVals_.setLength(rsModelnInps);
            for (ii = 0; ii < rsModelnInps; ii++) VecRSInds_[ii] = ii;

            //**/ for each fixed variable, compare its name to the 
            //**/ input names of the RS model and match
            for (ii = 0; ii < nInps; ii++)
            {
              snprintf(pString,100,"fixed-%d", ii+1);
              cString = psConfig_.getParameter(pString);
              if (cString != NULL)
              {
                //**/ if fixed input ii is found to match input kk of
                //**/ RS model, read its value and put into VecRSVals_
                sscanf(cString,"%s %s %s %lg",winput1,winput2,
                       winput3, &ddata);
                for (kk = 0; kk < rsModelnInps; kk++)
                {
                  if (!strcmp(winput2,pPtr.strArray_[kk]))
                  {
                    VecRSInds_[kk] = -1;
                    VecRSVals_[kk] = ddata;
                    printf("Input %4d is fixed at %e\n",kk+1,ddata);
                    break;
                  }
                }
                //**/ if fixed input ii does not find a match in the
                //**/ RS model, something is wrong 
                if (kk == rsModelnInps) break;
              }
            }
            if (ii != nInps)
            {
              printf("ERROR: all not fixed variables have been matched.\n");
              printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
              exit(1);
            }
            rsnInps_ = rsModelnInps;
          }
        }
        else
        {
          printf("ERROR: No input names found in the RS data file - ");
          printf("cannot match fixed\n");
          printf("       parameter names.\n");
          printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
          exit(1);
        }
      }
      delete psIO;

      //**/ create response surface models
      rsPtrs_ = new FuncApprox*[nOutputs_];
      for (ii = 0; ii < nOutputs_; ii++)
      {
        printf("FuncIO INFO: Creating response surface for output %d (%s)\n",
               ii+1, fname);
        rsPtrs_[ii] = (FuncApprox *) genFAFromFile(fname,ii);
        if (rsPtrs_[ii] == NULL)
        {
          printf("ERROR: no RS model created for output %d.\n",
                 ii+1);
          printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
          exit(1);
        }
      }
    }
  }
  //**/ if no app driver or opt driver or auxOpt driver, look for 
  //**/ ensemble driver
  else
  {
    if ((!strcmp(fname, "NONE") || !strcmp(fname, "true")) &&
        (!strcmp(ensembleDriver_,"NONE") || 
         !strcmp(ensembleDriver_,"true")))
    {
      printf("%s\n", fname);
      printf("%s\n", ensembleDriver_);
      printf("FuncIO loadFunctionData WARNING: \n");
      printf("       Neither driver and ensemble_driver is set.\n");
      winput1[0] = '0';
      while (winput1[0] != 'n' && winput1[0] != 'y')
      {
        printf("Proceed? (y or n) ");
        scanf("%s", winput1);
      }
      if (winput1[0] != 'y') exit(1);
      fgets(winput2, 1000, stdin);
    }
  }
  return 0;
}

// ************************************************************************
// run application
// ------------------------------------------------------------------------
//**/ flag = 0, this is the first time this job (sampleID) is launched 
//**/ (if same sampleID is called, it means checking for job progress)
//**/ flag = 1, just create input file and return 
//**/           (if local function is called, this will call the function)
//**/ flag = 2, just check the status of this job
// ------------------------------------------------------------------------
int FunctionInterface::evaluate(int sampleID,int nInputs,double *inputs, 
                                int nOutputs, double *outputs, int flag)
{
  int    ii, outputCount, length, nfixed, status;
  double value, *myInputs, stdev;
  char   lineIn[1001], command[1001], winput1[1001], winput2[1001];
  char   outfile[501], infile[501], *cString, equal[100];
  FILE   *fp, *fIn, *fOut;

  //**/ ---------------------------------------------------------------
  //**/ error checking
  //**/ ---------------------------------------------------------------
  if ((useRSModel_ == 1) && 
      (nInputs_ != nInputs || nOutputs_ != nOutputs))
  {
    printf("FuncIO::evaluate ERROR: nInputs/nOutputs mismatch.\n");
    printf("   nInputs  = %d versus %d (local)\n", nInputs, nInputs_);
    printf("   nOutputs = %d versus %d (local)\n", nOutputs, nOutputs_);
    printf("Where: file %s line %d, exiting\n", __FILE__, __LINE__);
    exit(1);
  }

  //**/ ---------------------------------------------------------------
  //**/ if local functions are requested
  //**/ ---------------------------------------------------------------
  if (appOptFlag_ == 0 && (useRSModel_ == 2 || useRSModel_ == 4))
  {
    if (printLevel_ > 1)
      printf("Function evaluation #%d\n", sampleID+1);
    psLocalFunction(nInputs, inputs, nOutputs, outputs);
    return 0;
  }
  if (appOptFlag_ == 1 && (useRSModel_ == 3 || useRSModel_ == 4))
  {
    if (printLevel_ > 1)
      printf("Function evaluation #%d\n", sampleID+1);
    psLocalFunction(nInputs, inputs, nOutputs, outputs);
    return 0;
  }

  //**/ ---------------------------------------------------------------
  //**/ when response surface is NOT used
  //**/ ---------------------------------------------------------------
  if (useRSModel_ == 0 && rsPtrs_ == NULL)
  {
    //**/ ----------------------------------------------------------------
    //**/ extract input file name from input template by appending sample 
    //**/ number to it (or if optimization, the input template is fixed)
    //**/ ----------------------------------------------------------------
    if (appOptFlag_ == 0)
    {
      length = strlen(appInputTemplate_);
      for (ii = length-1; ii >= 0; ii--)
        if (appInputTemplate_[ii] == '/') break; 
      snprintf(infile,500,
               "%s.%d", &(appInputTemplate_[ii+1]), sampleID+1);
    }
    else snprintf(infile,500,"psuadeOpt.in.%d", sampleID+1);

    //**/ -------------------------------------------------------------
    //**/ fetch the name of the output file to read
    //**/ -------------------------------------------------------------
    if (appOptFlag_ == 0)
    {
      length = strlen(appOutputTemplate_);
      for (ii = length-1; ii >= 0; ii--)
        if (appOutputTemplate_[ii] == '/') break; 
      snprintf(outfile,500, 
               "%s.%d",&(appOutputTemplate_[ii+1]),sampleID+1);
    }
    else snprintf(outfile,500,"psuadeOpt.out.%d", sampleID+1);

    //**/ -------------------------------------------------------------
    //**/ check to see if the output file is already present (new 2008)
    //**/ If not, run simulation/emulation
    //**/ -------------------------------------------------------------
    if ((fIn=fopen(outfile, "r")) == NULL) 
    {
      //**/ -----------------------------------------------------------
      //**/ file not found and mode is check only, return
      //**/ -----------------------------------------------------------
      if (flag == 2) return 2;

      //**/ -----------------------------------------------------------
      //**/ write to input file 
      //**/ -----------------------------------------------------------
      fOut = fopen(infile, "w");
      if (fOut == NULL)
      {
        printf("FuncIO ERROR: Cannot open %s file\n",infile);
        exit(1);
      }
      fprintf(fOut, "%d\n", nInputs);
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fOut, "%24.16e\n", inputs[ii]);
      cString = psConfig_.getParameter("num_fixed");
      if (cString != NULL)
      {
        sscanf(cString, "%s %s %d", winput1, winput2, &nfixed);
        fprintf(fOut,"num_fixed = %d\n", nfixed);
        ii = 0;
        while (ii < nfixed)
        {
          ii++;
          snprintf(winput1,100,"fixed-%d",ii);
          cString = psConfig_.getParameter(winput1);
          if (cString != NULL)
          {
            sscanf(cString,"%s %s %s %lg",
                   winput1,winput2,equal,&value);
            fprintf(fOut,"fixed %d %s = %24.16e\n",ii,winput2,value);
          }
        }
      }
      fclose(fOut);
      //**/ if the task is just to create an input file, then just return
      if (flag == 1) return 2;

      //**/ -----------------------------------------------------------
      //**/ Now input file has already been created. Next check to see
      //**/ if application or optimization driver exists
      //**/ -----------------------------------------------------------
      if (appOptFlag_ == 0)
      {
        if ((!strcmp(appDriver_,"NONE") || !strcmp(appDriver_,"true"))) 
        {
          printf("FuncIO ERROR: app driver not found.\n");
          exit(1);
        }
      }
      if (appOptFlag_ == 1)
      {
        if ((!strcmp(optDriver_,"NONE") || !strcmp(optDriver_,"true")))
        {
          printf("FuncIO ERROR: opt driver not found.\n");
          exit(1);
        }
      }
      if (appOptFlag_ == 2)
      {
        if ((!strcmp(auxOptDriver_, "NONE") || 
             !strcmp(auxOptDriver_, "true")))
        {
          printf("FuncIO ERROR: aux opt driver not found.\n");
          exit(1);
        }
      }

      //**/ -----------------------------------------------------------
      //**/ execute driver
      //**/ -----------------------------------------------------------
      if (appOptFlag_ == 0)
      {
        //**/ executionMode_ == 1 ==> asynchronous mode
        if (executionMode_ == 1)
          snprintf(command,1000,"\"%s\" %s %s %d %d %d&",appDriver_,
                   infile, outfile, sampleID, flag, printLevel_);
        else
          snprintf(command,1000,"\"%s\" %s %s %d %d %d", appDriver_,
                   infile, outfile, sampleID, flag, printLevel_);
        if (strstr((const char*) appDriver_, "rm ") != NULL ||
            strstr((const char*) appDriver_, "mv ") != NULL ||
            strstr((const char*) appDriver_, " -f ") != NULL ||
            strstr((const char*) appDriver_, "/bin/") != NULL) 
        {
          printf("FuncIO::evaluate ERROR: \n");
          printf("\t\t for security reason do not use rm in driver.\n");
          exit(1);
        }
      }
      else if (appOptFlag_ == 1)
      {
        if (executionMode_ == 1)
          snprintf(command,1000,"%s %s %s %d %d %d&",optDriver_,infile,
                   outfile, sampleID+1, flag, printLevel_);
        else
          snprintf(command,1000,"%s %s %s %d %d %d",optDriver_,infile,
                   outfile, sampleID+1, flag, printLevel_);
        if (strstr((const char*) optDriver_, "rm ") != NULL ||
            strstr((const char*) optDriver_, "mv ") != NULL ||
            strstr((const char*) optDriver_, " -f ") != NULL ||
            strstr((const char*) optDriver_, "/bin/") != NULL) 
        {
          printf("FuncIO::evaluate ERROR: \n");
          printf("\t\t for security reason do not use rm in driver.\n");
          exit(1);
        }
      }
      else if (appOptFlag_ == 2)
      {
        snprintf(command,1000,"%s %s %s %d %d %d",auxOptDriver_,infile, 
                 outfile, sampleID+1, flag, printLevel_);
        if (strstr((const char*) auxOptDriver_, "rm ") != NULL ||
            strstr((const char*) auxOptDriver_, "mv ") != NULL ||
            strstr((const char*) auxOptDriver_, " -f ") != NULL ||
            strstr((const char*) auxOptDriver_, "/bin/") != NULL) 
        {
          printf("FuncIO::evaluate ERROR: \n");
          printf("\t\t for security reason do not use rm in driver.\n");
          exit(1);
        }
      }
      if (psConfig_.PDFDiagnosticsIsOn())
        printf("Launching simulation: %s\n",command);
      status = system(command);   
      if (psConfig_.PDFDiagnosticsIsOn())
        printf("Simulation completed.\n");
      if (status != 0)
      {
        printf("FuncIO evaluate ERROR: system call returns %d.\n",
               status);
        printf("INFO: Command = %s\n", command);
        printf("      System call return status should be 0.\n");
        printf("      Check your simulation driver for correctiness.\n");
        printf("DO: CHECK YOUR DRIVER PROGRAM BY RUNNING THIS COMMAND:\n");
        printf("      %s\n", command);
        exit(1);
      }
      //**/ if asynchronous mode, wait before launching another job
      if (executionMode_ == 1 && launchInterval_ > 0)
      {
#ifdef WINDOWS
        Sleep(1000 * launchInterval_);
#else
        sleep(launchInterval_);
#endif
      }
    }
    else if (flag == 0)
    {
      printf("WARNING: Output file %s exists before it is run.\n",
             outfile);
      printf("         PSUADE will use this output file.\n");
      printf("         If this is a mistake, stop PSUADE and clean up.\n");
      fclose(fIn);
    }
    else fclose(fIn);

    //**/ -----------------------------------------------------------
    //**/ check output file
    //**/ -----------------------------------------------------------
    //**/ synchronous mode: wait until output file shows up
    if (executionMode_ == 0)
    {
      //if (launchInterval_ == 0) launchInterval_ = 1;

      while ((fIn=fopen(outfile, "r")) == NULL)
      {
        if (printLevel_ > 2)
        {
          printf("Waiting for Job %d to complete.\n", sampleID+1);
          printf("If you run the simulator yourself, use the inputs\n");
          if (appOptFlag_ == 0)
          {
            printf("from psuadeApps.in.%d for your simulation and\n", 
                   sampleID+1);
            printf("write the outputs to psuadeApps.out.%d\n",
                   sampleID+1);
          }
          else if (appOptFlag_ == 1)
          {
            printf("from psuadeOpt.in.%d for your simulation and\n", 
                   sampleID+1);
            printf("write the outputs to psuadeOpt.out.%d\n",sampleID+1);
          }
        }
#ifdef WINDOWS
        Sleep(1000 * launchInterval_);
#else
        sleep(launchInterval_);
#endif
      }
      fclose(fIn);
    }

    //**/ -----------------------------------------------------------
    //**/ asynchronous mode: if output file not found, just return
    //**/ -----------------------------------------------------------
    if (executionMode_ == 1)
    {
      if ((fIn=fopen(outfile, "r")) == NULL) 
      {
        //**/printf("FuncIO::evaluate ERROR : \n");
        //**/printf("\t\t cannot find output file %s.\n",outfile);
        return 2;
      }
      fclose(fIn);
    }
    //**/ wait a little bit 
    length = 0;
    for (ii = 0; ii < 100000; ii++) length = (length + ii) %  32768;

    //**/ -----------------------------------------------------------
    //**/ outfile ready, read outputs (if no output template, do not 
    //**/ need to search for output names)
    //**/ -----------------------------------------------------------
    fIn = fopen(outfile, "r");
    if (fIn == NULL) return 2;
    outputCount = 0;
    for (ii = 0; ii < nOutputs_; ii++)
    {
      fgets(lineIn, 100, fIn);
      sscanf(lineIn, "%lg", &(outputs[outputCount]));
      outputCount++;
      if (feof(fIn) == 1) break; 
    }
    fclose(fIn);

    //**/ -----------------------------------------------------------
    //**/ output error checking
    //**/ -----------------------------------------------------------
    if (outputCount != nOutputs) 
    {
      printf("\n\t output file %s found but nOutputs do not match.\n",
             outfile);
      printf("\t    nOutputs from simulator output file = %d\n",
             outputCount);
      printf("\t    nOutputs expected                   = %d\n", 
             nOutputs);
      printf("\t\t (check your output format).\n");
      return 2;
    }

    //**/ -----------------------------------------------------------
    //**/ remove all temporary files
    //**/ -----------------------------------------------------------
    if (strcmp(infile, "*"))  unlink(infile);
    if (strcmp(outfile, "*")) unlink(outfile);
  }
  //**/ -------------------------------------------------------------
  //**/ use response surface 
  //**/ -------------------------------------------------------------
  else if (useRSModel_ == 1 && rsPtrs_ != NULL)
  {
    if (VecRSInds_.length() > 0)
    {
      myInputs = new double[rsnInps_];
      //**/ fill in fixed inputs
      for (ii = 0; ii < rsnInps_; ii++)
      {
        if (VecRSInds_[ii] < 0) myInputs[ii] = VecRSVals_[ii];
        else                    myInputs[ii] = inputs[VecRSInds_[ii]];
      }
      //**/ evaluate
      if (rsRanFlag_ == 0)
      {
        for (ii = 0; ii < nOutputs; ii++)
          outputs[ii] = rsPtrs_[ii]->evaluatePoint(myInputs);
      }
      else
      {
        for (ii = 0; ii < nOutputs; ii++)
        {
          outputs[ii] = rsPtrs_[ii]->evaluatePointFuzzy(myInputs,stdev);
          value = 3.0 * stdev;
          if (rsRanFlag_ == 1) outputs[ii] -= value;
          else                 outputs[ii] += value;
        } 
      }
      delete [] myInputs;
    }
    else
    {
      if (rsRanFlag_ == 0)
      {
        for (ii = 0; ii < nOutputs; ii++)
          outputs[ii] = rsPtrs_[ii]->evaluatePoint(inputs);
      }
      else
      {
        for (ii = 0; ii < nOutputs; ii++)
        {
          outputs[ii] = rsPtrs_[ii]->evaluatePointFuzzy(inputs,stdev);
          value = 3.0 * stdev;
          if (rsRanFlag_ == 1) outputs[ii] -= value;
          else                 outputs[ii] += value;
        }
      }
    }
  }
  else
  {
    if (useRSModel_ == 1 && rsPtrs_ == NULL) 
    {
      printf("FuncIO ERROR: Response surface pointers empty.\n");
      printf("       Did you forget to specify driver?\n");
      printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
      exit(1);
    }
    printf("FuncIO ERROR: Don't know what to do.\n");
    printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
  }
  return 0;
}

// ************************************************************************
// run ensemble simulation
// ------------------------------------------------------------------------
int FunctionInterface::ensembleEvaluate(int nSamp,int nInputs,
                        double *inputs,int nOutputs,double *outputs,int ID)
{
  int    ii, ss, outputCount, nfixed, status;
  double value;
  char   outfile[500], infile[500], lineIn[5000], command[1001];
  char   winput1[500], winput2[500], winput3[500], *cString;
  FILE   *fp, *fIn, *fOut;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (nInputs_ != nInputs || nOutputs_ != nOutputs)
  {
    printf("FuncIO ERROR: nInputs/nOutputs mismatch.\n");
    printf("   nInputs  = %d versus %d (local)\n",nInputs,nInputs_);
    printf("   nOutputs = %d versus %d (local)\n",nOutputs,nOutputs_);
    printf("NOTE: ensembleEvaluate expects driver is an ");
    printf("actual simulator and not\n");
    printf("      RS data file. Therefore, it does not ");
    printf("take a rs_index_file.\n");
    exit(1);
  }

  //**/ -------------------------------------------------------------
  //**/ prepare input file 
  //**/ -------------------------------------------------------------
  snprintf(infile,100,"psuadeEnsemble.in.%d", ID);
  snprintf(outfile,100,"psuadeEnsemble.out.%d", ID);

  //**/ -------------------------------------------------------------
  //**/ check to see if the output file is already present 
  //**/ -------------------------------------------------------------
  if ((fIn=fopen(outfile, "r")) == NULL) 
  {
    //**/ -----------------------------------------------------------
    //**/ if local functions are requested
    //**/ -----------------------------------------------------------
    if (appOptFlag_ == 0)
    {
      if (!strcmp(ensembleDriver_,"PSUADE_LOCAL"))
      {
        psEnsembleLocalFunction(nSamp,nInputs,inputs,nOutputs,outputs);
        return 0;
      } 
    }
    else
    {
      if (!strcmp(ensembleOptDriver_,"PSUADE_LOCAL"))
      {
        psEnsembleLocalFunction(nSamp,nInputs,inputs,nOutputs,outputs);
        return 0;
      } 
    }

    //**/ -----------------------------------------------------------
    //**/ write sample data to input file
    //**/ -----------------------------------------------------------
    fOut = fopen(infile, "w");
    if (fOut == NULL)
    {
      printf("FuncIO ERROR: Cannot open %s file\n",infile);
      exit(1);
    }
    fprintf(fOut, "%d %d\n", nSamp, nInputs);
    for (ss = 0; ss < nSamp; ss++)
    {
      for (ii = 0; ii < nInputs; ii++)
        fprintf(fOut, "%20.12e ", inputs[ss*nInputs+ii]);
      fprintf(fOut, "\n");
    }
    cString = psConfig_.getParameter("num_fixed");
    if (cString != NULL)
    {
      sscanf(cString, "%s %s %d", winput1, winput2, &nfixed);
      fprintf(fOut,"num_fixed = %d\n", nfixed);
      ss = 0;
      for (ii = 0; ii < nfixed; ii++)
      {
        snprintf(winput1,100,"fixed-%d",ii+1);
        cString = psConfig_.getParameter(winput1);
        if (cString != NULL)
        {
          sscanf(cString, 
                 "%s %s %s %lg", winput1,winput2,winput3,&value);
          fprintf(fOut,"fixed %d %s = %24.16e\n",ii,winput2,value);
          ss++;
        }
        else printf("FuncInterface ERROR: %s not found.\n",winput1);
      }
      if (ss != nfixed)
      {
        printf("FuncInterface ERROR: No fixed variables in config.\n");
        psConfig_.print();
        exit(1);
      }
    }
    fclose(fOut);

    //**/ -----------------------------------------------------------
    //**/ Now input file has already been created. Next call driver
    //**/ -----------------------------------------------------------
    if (appOptFlag_ == 0)
    {
      if (!strcmp(ensembleDriver_, "NONE") || 
          !strcmp(ensembleDriver_, "true"))
      {
        printf("FuncIO ERROR: ensemble driver not set.\n");
        exit(1);
      }
      fp = fopen(ensembleDriver_, "r");
      if (fp == NULL)
      {
        printf("FuncIO ERROR: ensemble driver %s not found.\n",
               ensembleDriver_);
        exit(1);
      }
      else fclose(fp);
      snprintf(command,1000,"%s %s %s %d %d", ensembleDriver_,infile,
               outfile,ID,printLevel_);
    }
    else
    {
      if (!strcmp(ensembleOptDriver_, "NONE") || 
          !strcmp(ensembleOptDriver_, "true"))
      {
        printf("FuncIO ERROR: ensemble opt driver not set.\n");
        exit(1);
      }
      fp = fopen(ensembleOptDriver_, "r");
      if (fp == NULL)
      {
        printf("FuncIO ERROR: ensemble opt driver %s not found\n",
               ensembleOptDriver_);
        exit(1);
      }
      else fclose(fp);
      snprintf(command,1000,"%s %s %s %d %d", ensembleOptDriver_,infile,
              outfile, ID, printLevel_);
    }

    //**/ -----------------------------------------------------------
    //**/ execute driver
    //**/ -----------------------------------------------------------
    if (psConfig_.PDFDiagnosticsIsOn())
      printf("Launching ensemble simulations: %s\n",command);
    status = system(command);   
    if (psConfig_.PDFDiagnosticsIsOn())
      printf("Ensemble simulations completed\n");
    if (status != 0)
    {
      printf("FuncIO INFO: system call returns status = %d.\n",
             status);
      printf("       System call return status should be 0.\n");
      printf("       Check your simulation driver for correctiness.\n");
      exit(1);
    }
  }
  else 
  {
    printf("WARNING: Output file %s exists before it is run.\n",outfile);
    printf("         PSUADE will use this output file.\n");
    printf("         If this is a mistake, stop PSUADE and clean up.\n");
    fclose(fIn);
  }

  //**/ -------------------------------------------------------------
  //**/ check output file
  //**/ -------------------------------------------------------------
  fIn = fopen(outfile, "r");
  while (fIn == NULL)
  {
    if (printLevel_ > 2)
      printf("FuncIO INFO: waiting for Job %d to complete.\n",ID);
#ifdef WINDOWS
    Sleep(1000 * launchInterval_);
#else
    sleep(launchInterval_);
#endif
    fIn = fopen(outfile, "r");
  }

  //**/ -------------------------------------------------------------
  //**/ outfile ready, read outputs 
  //**/ -------------------------------------------------------------
  if (fIn == NULL)
  {
    printf("FuncIO ERROR: output file %s not found.\n",outfile);
    exit(1);
  }
  outputCount = 0;
  for (ss = 0; ss < nSamp; ss++)
  {
    fgets(lineIn, 4000, fIn);
    for (ii = 0; ii < nOutputs_; ii++)
      sscanf(lineIn, "%lg", &(outputs[ss*nOutputs_+ii]));
    outputCount++;
    if (feof(fIn) == 1) break; 
  }
  fclose(fIn);

  //**/ -------------------------------------------------------------
  //**/ output error checking
  //**/ -------------------------------------------------------------
  if (outputCount != nSamp) 
  {
    printf("FuncIO ERROR: output file %s found but ",outfile);
    printf("with insufficient data.\n");
    printf("Advice: Check output format of your ensemble driver.\n");
    printf("        It should have %d lines each with %d outputs.\n",
           nSamp, nOutputs_);
    exit(1);
  }

  //**/ -------------------------------------------------------------
  //**/ remove all temporary files
  //**/ -------------------------------------------------------------
  if (strcmp(infile, "*")) unlink(infile);
  if (strcmp(outfile, "*")) unlink(outfile);
  return 0;
}

// ************************************************************************
// get number of input variables
// ------------------------------------------------------------------------
int FunctionInterface::getNumInputs()
{
  return nInputs_;
}

// ************************************************************************
// get number of output variables
// ------------------------------------------------------------------------
int FunctionInterface::getNumOutputs()
{
  return nOutputs_;
}

// ************************************************************************
// get current driver code
// ------------------------------------------------------------------------
int FunctionInterface::getDriver()
{
  return appOptFlag_;
}

// ************************************************************************
// get names of input variables
// ------------------------------------------------------------------------
char **FunctionInterface::getInputNames()
{
  return StrInpNames_.getStrings();
}

// ************************************************************************
// get names of output variables
// ------------------------------------------------------------------------
char **FunctionInterface::getOutputNames()
{
  return StrOutNames_.getStrings();
}

// ************************************************************************
// get application driver
// ------------------------------------------------------------------------
char *FunctionInterface::getApplicationDriver()
{
  return appDriver_;
}

// ************************************************************************
// get optimization driver
// ------------------------------------------------------------------------
char *FunctionInterface::getOptimizationDriver()
{
  return optDriver_;
}

// ************************************************************************
// get auxiliary optimization driver
// ------------------------------------------------------------------------
char *FunctionInterface::getAuxOptimizationDriver()
{
  return auxOptDriver_;
}

// ************************************************************************
// Local functions (use setLocalFunction to select)
// (some of these local functions are used for, for example, optimal
//  experimental design - coupled with some PSUADE's global optimizers). 
// NOTE: inputsIn has 1-based candidate indices
// ------------------------------------------------------------------------
int FunctionInterface::psLocalFunction(int nInputsIn, double *inputsIn,
                                       int nOutputs, double *outputs)
{
  int    ss, ii, jj, kk, ll, ind, lcnt, status, iOne=1, count;
  double dmean, ddata, dtmp, aggrVar, dvar, GMetric, IMetric, DMetric;
  double AMetric, EMetric;
  char   fname[2000], pString[2000], lineIn[2000];
  pData  pdata;
  psMatrix matCov;

  //**/ =============================================================
  //**/ make a local copy of inputs
  //**/ This is needed because user can call this function with
  //**/ nInputsIn=0 and inputsIn=NULL
  //**/ -------------------------------------------------------------
  int    nInputs = nInputsIn;
  double *inputs = NULL;
  psVector vecInps;
  if (nInputsIn > 0) 
  { 
    vecInps.setLength(nInputsIn);
    if (inputsIn != NULL)
      for (ii = 0; ii < nInputs; ii++) vecInps[ii] = inputsIn[ii];
    inputs = vecInps.getDVector();
  }
 
  //**/ =============================================================
  //**/ data structures to be used for optimal design
  //**/ =============================================================
  static int nHist, nInps=3, nOuts=1, maxHist=10000;
  static int ProblemInitialized=0;
  static psMatrix  matHistory;
  static psIVector vecUInputs, vecIT;
  static psMatrix matCandidates;
  static psMatrix matPriorSample;
  static psMatrix matEvalSet;
  static ProbMatrix **CandPostSamples=NULL;
  static ProbMatrix matProbPrior;
  static ProbMatrix matProbUniform;
  static FuncApprox **rsPtrs=NULL;
  static psMatrix matDistances;
  static int nPreSelected=0;
  static double optimalVal;
  static McmcData mobj14;
  static psVector VecYUniform;
  static psVector VecYPrior;
  static FunctionInterface *FisherFuncIO=NULL;
  static psMatrix matFisherSimStore;
  static psMatrix matFisherEvalStore;
  static GPBasedSampling *gpMVSampler=NULL;
  //**/ -------------------------------------------------------------
  //**/ the following is for the deterministic optimization for
  //**/ least-squares given experiment and training sample
  //**/ -------------------------------------------------------------
  static psIVector vecDParams;
  static psMatrix matOptInps, matOptMeans, matOptStds;
#if 0
  //**/ -------------------------------------------------------------
  //**/ the following is used to select OUU test problem 
  //**/ -------------------------------------------------------------
  static int toyOption = -1;
#endif
 
  //**/ =============================================================
  //**/ option 999: clean up (after optimization is completed) 
  //**/ -------------------------------------------------------------
  //**/ The call sequence to use these optimiztions is:
  //**/ - instantiate FunctionInterface ==> funcIO
  //**/ - call funcIO->setLocalFunction(option) to select method
  //**/ - feed this funcIO to some optimization method, which
  //**/   calls this repeatedly for design evaluations
  //**/ - when completed, call funcIIO->setLocalFunction(999) 
  //**/ =============================================================
  if (whichLocalFunction_ == 999)
  {
    if (printLevel_ > 0)
      printf("FuncIO Local INFO: cleaning up.\n");
    if (CandPostSamples != NULL)
    {
      for (ii = 0; ii < matCandidates.nrows(); ii++)
        if (CandPostSamples[ii] != NULL)
          delete CandPostSamples[ii];
      delete [] CandPostSamples;
    } 
    CandPostSamples = NULL;
    if (rsPtrs != NULL)
    {
      for (ii = 0; ii < nOuts; ii++) 
        if (rsPtrs[ii] != NULL) delete rsPtrs[ii];
      delete [] rsPtrs;
    }
    rsPtrs = NULL;
    ProblemInitialized = 0;
    optimalVal = PSUADE_UNDEFINED;
    if (mobj14.rsPtrs_ != NULL)
    {
      delete [] mobj14.rsPtrs_;
      mobj14.rsPtrs_ = NULL;
    }
    matProbPrior.clean();
    matProbUniform.clean();
    matDistances.clean();
    matPriorSample.clean();
    matEvalSet.clean();
    VecYUniform.clean();
    VecYPrior.clean();
    if (FisherFuncIO != NULL) delete FisherFuncIO;
    matFisherSimStore.clean();
    matFisherEvalStore.clean();
    if (gpMVSampler != NULL)
    {
      delete gpMVSampler;
      gpMVSampler = NULL;
    }
  }
  //**/ =============================================================
  //**/ option: ODOE G/I/A/D/E-optimal with SCE optimization
  //**/ (Use fast algorithm without using rsmsmc at every iteration)
  //**/ -------------------------------------------------------------
  //**/ 10: G; 11: I; 12: A; 13: D; 14: E
  //**/ =============================================================
  else if (whichLocalFunction_ >= 10 && whichLocalFunction_ <= 14)
  {
    //**/ -----------------------------------------------------------
    //**/ error checking (the objective function must have size 1
    //**/ even though the simulator may have multiple outputs if
    //**/ it is called by SCE).
    //**/ But if nInputsIn == 0, it means odoeu_eval calls this
    //**/ function and so it is acceptable to have nOutputs > 1
    //**/ to store the 2-5 metrics
    //**/ Also, even if nInputsIn != 0, if nOutputs = 5, it is okay
    //**/ because it may be called from odoeu_boptnbf.
    //**/ -----------------------------------------------------------
    if (nOutputs != 1 && nOutputs != 5 && nInputsIn > 0)
    {
      printf("FuncIO ERROR: nOutputs must be = 1 or 5 if nInputs > 0\n");
      exit(1);
    }

    //**/ -----------------------------------------------------------
    //**/ initialization (called once initially and reset up by 
    //**/ calling this function with code = 999) 
    //**/ -----------------------------------------------------------
    psMatrix matUniformSample;
    psVector vecXT, vecYT;
    if (ProblemInitialized == 0)
    {
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (FMCMC): Initialization begins ..\n");

      //**/ --- read training sample in PSUADE data format
      //**/ ==> psuadeIO has training sample
      //**/ --- nInps = number of inputs in training sample
      //**/ --- nOuts = number of outputs in training sample
      snprintf(pString,100,
        "Enter name of the training sample (for creating RS): ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      PsuadeData *psuadeIO = new PsuadeData();
      status = psuadeIO->readPsuadeFile(fname);
      if (status != 0)
      {
        printf("FuncIO ERROR when reading training sample\n");
        printf("       Maybe this file is in wrong format?\n");
        exit(1);
      }
      psuadeIO->getParameter("input_ninputs", pdata);
      nInps = pdata.intData_;
      psuadeIO->getParameter("output_noutputs", pdata);
      nOuts = pdata.intData_; // this is size of simulation output

      //**/ --- user selects uncertain inputs (==> vecUInputs)
      //**/     ==> vecUInputs has uncertain input indices
      //**/ --- set uncertain parameter input to 1 
      //**/ --- (e.g. vecIT[kk]>=1) if kk is an uncertain parameter
      vecUInputs.setLength(nInps);
      snprintf(pString,100,
         "Enter uncertain input number (1 - %d, 0 to end) : ",
         nInps);
      ii = 0;
      while (1)
      {
        kk = getInt(0, nInps, pString);
        if (kk == 0 || kk > nInps) break;
        vecUInputs[ii] = kk - 1;
        ii++;
      }
      vecUInputs.subvector(0, ii-1);
      int nUInps = vecUInputs.length();

      vecIT.setLength(nInps);
      kk = 1;
      for (ii = 0; ii < vecUInputs.length(); ii++)
      {
        vecIT[vecUInputs[ii]] = kk;
        kk++; 
      }

      //**/ --- read prior sample (in iread format)
      //**/ ==> matPriorSample
      printf("Uncertain parameters need a prior sample for inference.\n");
      printf("The prior sample format should be the following format:\n");
      printf("Line 1: <nSamples> <nInputs>\n");
      printf("Line 2: 1 <sample 1 values>\n");
      printf("Line 3: 2 <sample 2 values>\n");
      printf("Line 4: ...\n");
      snprintf(pString,100,"Enter the file name of your prior sample : ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      status = readIReadDataFile(fname, matPriorSample);
      if (status != 0)
      {
        printf("FuncIO ERROR when reading prior sample\n");
        FILE *fchk = fopen(fname, "r");
        if (fchk == NULL)
          printf("       Prior sample file does not exist.\n");
        else
        {
          fclose(fchk);
          printf("       Maybe this file is in wrong format?\n");
        }
        exit(1);
      }
      if (matPriorSample.ncols() != vecUInputs.length())
      {
        printf("FuncIO ERROR: prior sample nInputs is incorrect.\n");
        printf("       Should be equal to %d (as indicated before).\n",
               vecUInputs.length());
        exit(1);
      }

      //**/ --- read uniform sample (needed for fast algorithm)
      //**/ ==> matUniformSample
      //**/ --- NOTE: PRIOR SAMPLE MUST BE DRAWN FROM UNIFORM
      //**/           FOR THE FAST VERSION
      //**/ NOTE: this is not needed if nInputsIn=nOutputs=0
      //**/       which means odoeu_beval is calling this function
      if (nInputsIn != 0 || nOutputs != 0)
      {
        printf("A uniform sample (uniform probably ");
        printf("distribution) for the\n");
        printf("uncertain parameters is needed for this ");
        printf("faster algorithm.\n");
        printf("If your prior sample is uniform, the same ");
        printf("file can be used.\n");
        printf("IMPORTANT: THE PRIOR SAMPLE MUST BE A SUBSET OF ");
        printf("THE UNIFORM\n");
        printf("           SAMPLE, OR THIS ALGORITHM WILL FAIL.\n");
        snprintf(pString,100,
             "Enter the file name of your uniform sample : ");
        getString(pString, fname);
        kk = strlen(fname);
        fname[kk-1] = '\0';
        status = readIReadDataFile(fname, matUniformSample);
        if (status != 0)
        {
          printf("FuncIO ERROR when reading uniform sample.\n");
          printf("       Maybe this file is in wrong format?\n");
          exit(1);
        }
        if (matUniformSample.ncols() != vecUInputs.length())
        {
          printf("FuncIO ERROR: uniform sample nInputs is incorrect.\n");
          printf("       Should be equal to %d (as indicated before).\n",
                 vecUInputs.length());
          exit(1);
        }
      }

      //**/ --- read candidate set ==> matCandidates
      //**/ NOTE: each row is one candidate point
      //**/       first m1 columns corresponds to uncertain inputs in
      //**/       ascending order; last nOuts*2 columns are simulation
      //**/       output means and std devs
      printf("Next, provide a candidate set from which ");
      printf("the final design is to be\n");
      printf("selected. The format of this file should be:\n");
      printf("Line 1: <number of candidates> <number of columns>\n");
      printf("Line 2: 1 <design inputs> <expected outputs and std dev>\n");
      printf("Line 3: 2 <design inputs> <expected outputs and std dev>\n");
      printf("...\n");
      snprintf(pString,100,
         "Enter the file name of your candidate set : ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      status = readIReadDataFile(fname, matCandidates);
      if (status != 0)
      {
        printf("FuncIO ERROR when reading candidate set.\n");
        printf("       Maybe this file is in wrong format?\n");
        exit(1);
      }
      int nCandidates = matCandidates.nrows();
      printf("Size of Candidate set = %d\n", nCandidates);
      if (matCandidates.ncols() != 2*nOuts+nInps-vecUInputs.length())
      {
        printf("FuncIO ERROR: Candidate file must have %d columns.\n",
               2*nOuts+nInps-vecUInputs.length());
        printf("              Only %d columns are found.\n",
               matCandidates.ncols());
        printf("INFO: The reason for this error may be because ");
        printf("you need for each\n");
        printf("      candidate in your candidate set file also ");
        printf("estimates of output\n");
        printf("      mean and standard deviation. To do so, use ");
        printf("odoeu_rseval (after\n");
        printf("      loading training sample, provide your ");
        printf("candidate set and also a\n");
        printf("      prior sample for the uncertain inputs) to ");
        printf("append your candidate\n");
        printf("      set with output means and standard deviations.\n");
        exit(1);
      }
      for (ii = 0; ii < nOuts; ii++)
      {
        for (jj = 0; jj < matCandidates.nrows(); jj++)
        {
          ddata = matCandidates.getEntry(jj,nInps-nUInps+2*ii);
          if (ddata <= 0)
          {
            printf("funcIO WARNING: Candidate %d output %d <= 0\n",
                   jj+1, ii+1);
          }
          ddata = matCandidates.getEntry(jj,nInps-nUInps+2*ii+1);
          //if (ddata < 0.1 || ddata > 1)
          //{
          //  printf("funcIO WARNING: Candidate %d weight not in [0.1,1]\n",
          //         jj+1);
          //  printf("INFO: reset output %d weight to 1.\n",ii+1);
          //  matCandidates.setEntry(jj,nInps-nUInps+2*ii+1,1.0);
          //}
        }
      }

      //**/ --- read evaluation set (only needed for G and I-metric)
      //**/ ===> matEvalSet
      if (whichLocalFunction_ == 10 || whichLocalFunction_ == 11)
      {
        printf("An evaluation sample is needed to ");
        printf("compute the optimality metrics.\n");
        printf("This can be the same as the candidate set");
        printf(" (but not recommended).\n");
        snprintf(pString,100,
                 "Enter the file name of your evaluation set : ");
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        status = readIReadDataFile(fname, matEvalSet);
        if (status != 0)
        {
          printf("FuncIO ERROR when reading evaluation set\n");
          printf("       Maybe this file is in wrong format?\n");
          exit(1);
        }
        if (matEvalSet.ncols() != 2*nOuts+nInps-vecUInputs.length() &&
            matEvalSet.ncols() != nInps-vecUInputs.length())
        {
          printf("FuncIO ERROR: evaluation set must have %d or %d columns.\n",
                 2*nOuts+nInps-nUInps,nInps-nUInps);
          printf("       If it has %d columns, the last %d columns ",
                 2*nOuts+nInps-nUInps,2*nOuts);
          printf("are not used.\n");
          exit(1);
        }
      }

      //**/ --- use training sample to construct response surface 
      //**/ --- psuadeIO ==> rsPtrs
      int faFlag = 1, rsMethod;
      pData pInps, pOuts, pLBs, pUBs;
      psuadeIO->getParameter("method_nsamples", pdata);
      int nSamp = pdata.intData_;
      psuadeIO->getParameter("input_lbounds", pLBs);
      psuadeIO->getParameter("input_ubounds", pUBs);
      psuadeIO->getParameter("input_sample", pInps);
      psuadeIO->getParameter("output_sample", pOuts);
  
      vecYT.setLength(nSamp);
      rsPtrs = new FuncApprox*[nOuts];
      printf("FunctionIO: constructing response surfaces\n");
      for (ii = 0; ii < nOuts; ii++)
      {
        if (ii == 0) 
        {
          rsPtrs[ii] = genFAInteractive(psuadeIO, faFlag);
          rsMethod = rsPtrs[ii]->getID();
        }
        else 
        {
          //**/ All outputs use the same response surface method
          psuadeIO->updateAnalysisSection(-1,-1,rsMethod,-1,-1,-1);
          faFlag = 0;
          rsPtrs[ii] = genFAInteractive(psuadeIO, faFlag);
        }
        rsPtrs[ii]->setBounds(pLBs.dbleArray_,pUBs.dbleArray_);
        rsPtrs[ii]->setOutputLevel(0);
        for (jj = 0; jj < nSamp; jj++)
          vecYT[jj] = pOuts.dbleArray_[jj*nOuts+ii];
        status = rsPtrs[ii]->initialize(pInps.dbleArray_,
                                       vecYT.getDVector());
      }
      delete psuadeIO;

//**/ THIS SEGMENT IS NOT USED CURRENTLY
//#define USE_FAST
#ifdef USE_FAST
      //**/ evaluate the uniform sample on the response surface
      //**/ this is used to speed up evaluation later so that
      //**/ expensive response surface evaluation is not needed.
      //**/ However, this (VecYUniform) takes up a lot of memory
      //**/ NOTE: This is needed only for G and I metrics
      //**/ NOTE: if nInputs = nOutputs = 0, odoeu_beval is calling
      //**/       this function and no uniform sample is needed
      if ((whichLocalFunction_ == 10 || whichLocalFunction_ == 11) &&
           (nInputsIn != 0 || nOutputs != 0))
      {
        matProbUniform.load(matUniformSample.nrows(),
           matUniformSample.ncols(),matUniformSample.getMatrix1D());
        int USamSize = matProbUniform.nrows();
        int YSize = nOuts * USamSize * matEvalSet.nrows();
        vecXT.setLength(USamSize * nInps);
        VecYUniform.setLength(YSize);
        double *YPtr = VecYUniform.getDVector();
        for (ii = 0; ii < matEvalSet.nrows(); ii++)
        {
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          {
            if (vecIT[jj] == 0)
            {
              vecXT[jj] = matEvalSet.getEntry(ii,lcnt);
              for (kk = 1; kk < USamSize; kk++)
                vecXT[kk*nInps+jj] = vecXT[jj];
              lcnt++;
            }
          }
          for (ll = 0; ll < vecUInputs.length(); ll++)
          {
            ind = vecUInputs[ll];
            for (kk = 0; kk < USamSize; kk++)
              vecXT[kk*nInps+ind] = matProbUniform.getEntry(kk,ll);
          }
          for (jj = 0; jj < nOuts; jj++)
          {
            rsPtrs[jj]->evaluatePoint(USamSize,vecXT.getDVector(),
                         &(YPtr[ii*USamSize*nOuts+jj*USamSize]));
          }
        }
      }
#endif
    
      //**/ --- construct CandidatePostSample that will be used for 
      //**/     (i.e. use MCMC on individual candidate)
      //**/ --- the cascading algorithm for multiple selections
      //**/ ===> CandPostSamples
      //**/ NOTE: if nInputs = nOutputs = 0, odoeu_beval is calling
      //**/       this function. The following is still needed but
      //**/       the prior sample (not uniform sample) should be used
      int nDesInps = nInps - vecUInputs.length();
      psMatrix matExpInps, matExpMeans, matExpStds;
      matExpInps.setDim(iOne, nInps-vecUInputs.length());
      matExpMeans.setDim(iOne, nOuts);
      matExpStds.setDim(iOne, nOuts);
      CandPostSamples = new ProbMatrix*[nCandidates];
      McmcData mobj;
      mobj.printLevel_ = 0;
      mobj.nSamples_ = nSamp;
      mobj.nInputs_ = nInps;
      mobj.nOutputs_ = nOuts;
      mobj.VecLowerB_.load(nInps, pLBs.dbleArray_);
      mobj.VecUpperB_.load(nInps, pUBs.dbleArray_);
      mobj.VecSamInputs_.load(nInps*nSamp, pInps.dbleArray_);
      mobj.VecSamOutputs_.load(nOuts*nSamp, pOuts.dbleArray_);
      mobj.VecCUInputs_ = vecUInputs;
      mobj.faType_ = PSUADE_RS_GP3;
      if (nInputsIn == 0 && nOutputs == 0)
         mobj.MatPriorSample_ = matPriorSample;
      else
         mobj.MatPriorSample_ = matUniformSample;
      MCMCAnalyzer *mcmcAnalyzer = new MCMCAnalyzer();
      printf("Processing candidates...\n");
      for (ii = 0; ii < nCandidates; ii++)
      {
        if (printLevel_ > 0)
          printf("Inference using on candidate %d (%d)\n", ii+1,
                 nCandidates);
        vecXT.setLength(nInps);
        lcnt = 0;
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, compute vecXT = mean
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            dmean = 0;
            for (kk = 0; kk < matPriorSample.nrows(); kk++)
              dmean += matPriorSample.getEntry(kk,ind);
            vecXT[jj] = dmean / (double) matPriorSample.nrows();
          }
          //**/ else get the vecXT from candidate matrix
          else
          {
            vecXT[jj] = matCandidates.getEntry(ii,lcnt);
            matExpInps.setEntry(0, lcnt, vecXT[jj]);
            lcnt++;
          }
        }
        for (jj = 0; jj < nOuts; jj++)
        {
          //**/ get experimental mean/std from user 
          dmean = matCandidates.getEntry(ii,nDesInps+jj*2);
          matExpMeans.setEntry(0,jj,dmean);
          //**/ FEB 2021: set std dev = 0.1 * mean 
          //ddata = matCandidates.getEntry(ii,nDesInps+jj*2+1);
          ddata = 0.1 * dmean;
          if      (ddata < 0)  ddata = - ddata;
          else if (ddata == 0) ddata = 0.1;
          matExpStds.setEntry(0,jj,ddata);
        }
        mobj.MatExpInputs_ = matExpInps;
        mobj.MatExpMeans_ = matExpMeans;
        mobj.MatExpStds_ = matExpStds;
        mcmcAnalyzer->analyzeDirect(mobj);
        CandPostSamples[ii] = new ProbMatrix();
        CandPostSamples[ii]->load(mobj.MatPostSample_.nrows(),
                                  mobj.MatPostSample_.ncols(),
                                  mobj.MatPostSample_.getMatrix2D());
        if (printLevel_ > 0)
          printf("  * posterior sample size = %d (unique samples)\n",
                 CandPostSamples[ii]->nrows());
      }
      delete mcmcAnalyzer;
      matProbPrior.load(matPriorSample.nrows(), matPriorSample.ncols(),
                        matPriorSample.getMatrix1D());
      ProblemInitialized = 1;
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (FMCMC) Initialization complete.\n");
      matHistory.setDim(maxHist,nInputs+1);
      nHist = 0;
      optimalVal = PSUADE_UNDEFINED;
    }

    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL and outputs == NULL, evaluate each point
    //**/ in the set individually (support odoeu_beval)
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0 && nOutputs == 0)
    {
      int      cc, nCandidates = matCandidates.nrows();
      double   dmean2, dcov;
      psVector vecEigs;
      psMatrix matEig;
      ProbMatrix matProbProduct;

      //**/ compute metric for each candidate
      printf("Candidate G-metric\t   I-metric\t   D-metric\t  ");
      printf("A-metric\t  E-metric\n");
      for (cc = 0; cc < nCandidates; cc++)
      {
        GMetric = IMetric = AMetric = EMetric = 0;
        DMetric = 1;
        matProbProduct = *(CandPostSamples[cc]);
        vecXT.setLength(nInps);
        vecYT.setLength(matProbProduct.nrows());
        for (ii = 0; ii < matEvalSet.nrows(); ii++)
        {
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          {
            if (vecIT[jj] == 0)
            {
              vecXT[jj] = matEvalSet.getEntry(ii,lcnt);
              lcnt++;
            }
          }
          aggrVar = 0;
          for (jj = 0; jj < nOuts; jj++)
          {
            for (kk = 0; kk < matProbProduct.nrows(); kk++)
            {
              for (ll = 0; ll < vecUInputs.length(); ll++)
              {
                ind = vecUInputs[ll];
                vecXT[ind] = matProbProduct.getEntry(kk,ll);
              }
              ddata = rsPtrs[jj]->evaluatePoint(vecXT.getDVector());
              vecYT[kk] = ddata;
            }
            dmean = dvar = 0;
            count = 0;
            for (kk = 0; kk < matProbProduct.nrows(); kk++)
            {
              dmean += vecYT[kk] * matProbProduct.getCount(kk);
              count += matProbProduct.getCount(kk);
            }
            dmean /= (double) count;
            for (kk = 0; kk < matProbProduct.nrows(); kk++)
              dvar += (pow(vecYT[kk]-dmean,2)*matProbProduct.getCount(kk));
            dvar = dvar / (double) count;
            aggrVar += dvar;
          }
          aggrVar /= nOuts;
          if (aggrVar > GMetric) GMetric = aggrVar;
          IMetric += aggrVar;
        }
        IMetric /= (double) matEvalSet.nrows();
        matCov.setDim(vecUInputs.length(),vecUInputs.length());
        for (jj = 0; jj < vecUInputs.length(); jj++)
        {
          dmean = dvar = 0.0;
          count = 0;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
          {
            dmean += matProbProduct.getEntry(kk,jj)*
                     matProbProduct.getCount(kk);
            count += matProbProduct.getCount(kk);
          }
          dmean /= (double) count;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
            dvar += pow(matProbProduct.getEntry(kk,jj)-dmean,2.0) *
                    matProbProduct.getCount(kk);
          dvar = dvar / (double) count;
          matCov.setEntry(jj,jj,dvar);
          AMetric += dvar;
          for (ll = jj+1; ll < vecUInputs.length(); ll++)
          {
            dmean2 = 0;
            for (kk = 0; kk < matProbProduct.nrows(); kk++)
              dmean2 += matProbProduct.getEntry(kk,ll) *
                        matProbProduct.getCount(kk);
            dmean2 /= (double) count;
            dcov = 0;
            for (kk = 0; kk < matProbProduct.nrows(); kk++)
              dcov += (matProbProduct.getEntry(kk,jj)-dmean) *
                      (matProbProduct.getEntry(kk,ll)-dmean2) *
                      matProbProduct.getCount(kk);
            dcov /= (double) count;
            matCov.setEntry(jj,ll,dcov);
            matCov.setEntry(ll,jj,dcov);
          }
        }
        matCov.eigenSolve(matEig, vecEigs, 1);
        for (jj = 0; jj < vecUInputs.length(); jj++)
        {
          DMetric *= vecEigs[jj];
          if (vecEigs[jj] > EMetric) EMetric = vecEigs[jj];
        }
        printf("%5d \t%12.4e\t%12.4e\t%12.4e\t%12.4e\t%12.4e\n",cc+1,
               GMetric,IMetric,DMetric,AMetric,EMetric);      
      }
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL, create an inputs of all candidates
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0)
    {
      nInputs = matCandidates.nrows();
      vecInps.setLength(nInputs);
      inputs = vecInps.getDVector();
      for (ii = 0; ii < nInputs; ii++) inputs[ii] = ii + 1;
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (a candidate cannot be
    //**/ be selected twice in the selected set)
    //**/ If inputs have duplicate, just return with a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = inputs[ii];
        kk  = inputs[jj];
        if (ind == kk) count++;
      }
    }
    if (count > 0)
    {
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ error checking (inputs have to be in [1,n] where n is the
    //**/ number of candidates in the candidate set matCandidates) 
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0 && printLevel_ > 0) 
    {
      if      (whichLocalFunction_ == 10) printf("GOPTIMAL inputs: ");
      else if (whichLocalFunction_ == 11) printf("IOPTIMAL inputs: ");
      else if (whichLocalFunction_ == 12) printf("DOPTIMAL inputs: ");
      else if (whichLocalFunction_ == 13) printf("AOPTIMAL inputs: ");
      else if (whichLocalFunction_ == 14) printf("EOPTIMAL inputs: ");
    }
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) inputs[ii];
      if (ind < 1 || ind > matCandidates.nrows())
      {
        printf("ODOE Function Evaluator ERROR: wrong input values.\n");
        printf("     Check candidate set size consistency.\n");
        printf("The erroneous inputs are:\n");
        for (jj = 0; jj < nInputs; jj++)
          printf("Candidate %3d for evaluation = %d\n", jj+1, 
                 (int) inputs[jj]);
        printf("But they should all be in the range of [1,%d]\n",
               matCandidates.nrows());
        exit(1);
      }
      if (nInputsIn > 0 && printLevel_ >= 0) printf("%5d ", ind);
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if this input selection has been evaluated 
    //**/ before from the history list. If so, reuse results
    //**/ -----------------------------------------------------------
    for (jj = 0; jj < nHist; jj++)
    {
      count = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        for (kk = 0; kk < nInputs; kk++)
          if (matHistory.getEntry(jj,kk) == inputs[ii])
            break;
        if (kk != nInputs) count++;
      }
      if (count == nInputs)
      {
        outputs[0] = matHistory.getEntry(jj,nInputs);
        if (printLevel_ > 0) 
          printf(" ===> output = %11.4e (revisit)\n", outputs[0]);
        return 0; 
      }
    }

    //**/ -----------------------------------------------------------
    //**/ evaluation (MCMC equivalence) using the cascading algorithm
    //**/ -----------------------------------------------------------
    ProbMatrix matProbTemp, matProbProduct;

    ind = (int) inputs[0] - 1;
    matProbProduct = *(CandPostSamples[ind]);
    matProbTemp = matProbProduct;
    status = 0;
    for (ii = 1; ii < nInputs; ii+=2)
    {
      //**/ --- select 2-argument or 3-argument matrix multiply to
      //**/ --- speed things up
      if ((ii + 1) >= nInputs)
      {
        ind = int(inputs[ii]) - 1;
        if (ind < 0) ind = 0;
        status = matProbProduct.multiply(*(CandPostSamples[ind]),
                                         matProbTemp);
      }
      else
      {
        ind = int(inputs[ii]) - 1;
        if (ind < 0) ind = 0;
        jj  = int(inputs[ii+1]) - 1;
        if (jj < 0) jj = 0;
        status = matProbProduct.multiply3(*(CandPostSamples[ind]),
                   *(CandPostSamples[jj]),matProbTemp);
        if (status != 0) 
        {
          printf("  INFO: This selection yields empty posterior => skip.\n");
          break;
        }
      }
      matProbProduct = matProbTemp;
    }

    //**/ -----------------------------------------------------------
    //**/ if it is empty posterior, return large value since it is
    //**/ not acceptable
    //**/ -----------------------------------------------------------
    if (status != 0) 
    {   
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      printf(" ==> empty posterior.\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ finally creating the final product by combining with the 
    //**/ prior distribution
    //**/ -----------------------------------------------------------
    matProbTemp.multiply(matProbPrior, matProbProduct);
    if (matProbProduct.ncols() == 0) 
    {
      printf(" ==> empty posterior ==> skip.\n");
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ G- and I-optimal metrics need the following first part
    //**/ -----------------------------------------------------------
    GMetric = IMetric = AMetric = EMetric = 0;
    DMetric = 1;
    if (whichLocalFunction_ == 10 || whichLocalFunction_ == 11)
    {
#ifdef USE_FAST
      //**/ #########################################################
      //**/ this portion does not need to call rsPtr, but uses the 
      //**/ pre-computed Ys (this should be faster but not sure)
      //**/ Jan 2021
      //**/ #########################################################
      double **uniformPtr = matProbUniform.getMatrix2D();
      double **productPtr = matProbProduct.getMatrix2D();
      vecXT.setLength(nInps);
      vecYT.setLength(matProbProduct.nrows()*nOuts);
      int uniformRow, productRow, nColumns = matProbUniform.ncols();
      int uniformNrows = matProbUniform.nrows();
      int productNrows = matProbProduct.nrows();
      for (ii = 0; ii < matEvalSet.nrows(); ii++)
      {
        uniformRow = productRow = 0;
        aggrVar = 0;
        while (productRow < matProbProduct.nrows())
        {
          while (1)
          {
            //**/ compare rows
            for (jj = 0; jj < nColumns; jj++)
            {
              if (uniformPtr[uniformRow][jj] != 
                  productPtr[productRow][jj])
                break;
            }
	    //**/ if equal (jj == nColumns), found
            if (jj == nColumns) break;
            else uniformRow++;
          }
          //**/ fetch outputs
          for (kk = 0; kk < nOuts; kk++)
            vecYT[kk*productNrows+productRow] = 
             VecYUniform[ii*uniformNrows*nOuts+uniformNrows*kk+uniformRow];
          productRow++;
          uniformRow++;
        }
        for (kk = 0; kk < nOuts; kk++)
        {
          dmean = dvar = 0;
          count = 0;
          for (jj = 0; jj < productNrows; jj++)
          {
            dmean += vecYT[productNrows*kk+jj]*matProbProduct.getCount(jj);
            count += matProbProduct.getCount(jj);
          }
          dmean /= (double) count;
          for (jj = 0; jj < productNrows; jj++)
            dvar += pow(vecYT[productNrows*kk+jj]-dmean,2)*
                    matProbProduct.getCount(jj);
          dvar = dvar / (double) count;
          aggrVar += dvar;
        }
        aggrVar /= nOuts;
        if (aggrVar > GMetric) GMetric = aggrVar;
        IMetric += aggrVar;
      }
      IMetric /= (double) matEvalSet.nrows();
#else
      //**/ #########################################################
      //**/ this part is slower compared to the other option? Ans: NO
      //**/ this version call rsPtr one at a time (2019)
      //**/ #########################################################
      vecXT.setLength(nInps);
      vecYT.setLength(matProbProduct.nrows());
      for (ii = 0; ii < matEvalSet.nrows(); ii++)
      {
        lcnt = 0;
        for (jj = 0; jj < nInps; jj++)
        {
          if (vecIT[jj] == 0)
          {
            vecXT[jj] = matEvalSet.getEntry(ii,lcnt);
            lcnt++;
          }
        }
        aggrVar = 0;
        for (jj = 0; jj < nOuts; jj++)
        {
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
          {
            for (ll = 0; ll < vecUInputs.length(); ll++)
            {
              ind = vecUInputs[ll];
              vecXT[ind] = matProbProduct.getEntry(kk,ll);
            }
            ddata = rsPtrs[jj]->evaluatePoint(vecXT.getDVector());
            vecYT[kk] = ddata;
          }
          dmean = dvar = 0;
          count = 0;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
          {
            dmean += vecYT[kk] * matProbProduct.getCount(kk);
            count += matProbProduct.getCount(kk);
          }
          dmean /= (double) count;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
            dvar += (pow(vecYT[kk]-dmean,2)*matProbProduct.getCount(kk));
          dvar = dvar / (double) count;
          aggrVar += dvar;
        }
        aggrVar /= nOuts;
        if (aggrVar > GMetric) GMetric = aggrVar;
        IMetric += aggrVar;
      }
      IMetric /= (double) matEvalSet.nrows();
#endif
#if 0
      //**/ #########################################################
      //**/ (12/17/2020) This version call rsPtr a BATCH at a time.
      //**/ Is this faster than calling rsPtr one at a time as above?
      //**/ The answer is: NO - the key is RS evaluation speed
      //**/ This part should be deleted eventually
      //**/ #########################################################
      int pNrows = matProbProduct.nrows();
      vecXT.setLength(nInps*pNrows);
      vecYT.setLength(pNrows);
      for (ii = 0; ii < matEvalSet.nrows(); ii++)
      {
        lcnt = 0;
        for (jj = 0; jj < nInps; jj++)
        {
          if (vecIT[jj] == 0)
          {
            vecXT[jj] = matEvalSet.getEntry(ii,lcnt);
            for (kk = 1; kk < pNrows; kk++)
              vecXT[kk*nInps+jj] = vecXT[jj];
            lcnt++;
          }
        }
        for (ll = 0; ll < vecUInputs.length(); ll++)
        {
          ind = vecUInputs[ll];
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
            vecXT[kk*nInps+ind] = matProbProduct.getEntry(kk,ll);
        }
        aggrVar = 0;
        for (jj = 0; jj < nOuts; jj++)
        {
          rsPtrs[jj]->evaluatePoint(pNrows,vecXT.getDVector(),
                                    vecYT.getDVector());
          dmean = dvar = 0;
          count = 0;
          for (kk = 0; kk < pNrows; kk++)
          {
            dmean += vecYT[kk] * matProbProduct.getCount(kk);
            count += matProbProduct.getCount(kk);
          }
          dmean /= (double) count;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
            dvar += (pow(vecYT[kk]-dmean,2)*matProbProduct.getCount(kk));
          dvar = dvar / (double) count;
          aggrVar += dvar;
        }
        aggrVar /= nOuts;
        if (aggrVar > GMetric) GMetric = aggrVar;
        IMetric += aggrVar;
      }
      IMetric /= (double) matEvalSet.nrows();
#endif
    } // G- or I-metric
    else
    //**/ -----------------------------------------------------------
    //**/ D-, A, E-optimal metrics use only the final posterior
    //**/ -----------------------------------------------------------
    {
      psMatrix matEig;
      psVector vecEigs;
      matCov.setDim(vecUInputs.length(),vecUInputs.length());
      double dmean2, dcov;
      AMetric = 0.0;
      for (jj = 0; jj < vecUInputs.length(); jj++)
      {
        dmean = dvar = 0.0;
        count = 0;
        for (kk = 0; kk < matProbProduct.nrows(); kk++)
        {
          dmean += matProbProduct.getEntry(kk,jj)*
                   matProbProduct.getCount(kk);
          count += matProbProduct.getCount(kk);
        }
        dmean /= (double) count;
        for (kk = 0; kk < matProbProduct.nrows(); kk++)
          dvar += pow(matProbProduct.getEntry(kk,jj)-dmean,2.0) *
                  matProbProduct.getCount(kk);
        dvar = dvar / (double) count;
        matCov.setEntry(jj,jj,dvar);
        AMetric += dvar;
        for (ll = jj+1; ll < vecUInputs.length(); ll++)
        {
          dmean2 = 0;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
            dmean2 += matProbProduct.getEntry(kk,ll) *
                      matProbProduct.getCount(kk);
          dmean2 /= (double) count;
          dcov = 0;
          for (kk = 0; kk < matProbProduct.nrows(); kk++)
            dcov += (matProbProduct.getEntry(kk,jj)-dmean) *
                    (matProbProduct.getEntry(kk,ll)-dmean2) *
                    matProbProduct.getCount(kk);
          dcov /= (double) count;
          matCov.setEntry(jj,ll,dcov);
          matCov.setEntry(ll,jj,dcov);
        }
      }
      matCov.eigenSolve(matEig, vecEigs, 1);
      DMetric = 1.0;
      EMetric = 0;
      for (jj = 0; jj < vecUInputs.length(); jj++)
      {
        DMetric *= vecEigs[jj];
        if (vecEigs[jj] > EMetric) EMetric = vecEigs[jj];
      }
    }

    //**/ -----------------------------------------------------------
    //**/ return the relevant metric information
    //**/ -----------------------------------------------------------
    if (whichLocalFunction_ == 10)
      outputs[0] = GMetric;
    else if (whichLocalFunction_ == 11)
      outputs[0] = IMetric;
    else if (whichLocalFunction_ == 12)
      outputs[0] = DMetric;
    else if (whichLocalFunction_ == 13)
      outputs[0] = AMetric;
    else if (whichLocalFunction_ == 14)
      outputs[0] = EMetric;
    if (outputs[0] < optimalVal) optimalVal = outputs[0];
    if (printLevel_ >= 0) 
    {
      printf(" ===> output = %11.4e (BEST SO FAR = %11.4e) : ",
             outputs[0], optimalVal);
      for (kk = 0; kk < nInputs; kk++)
        printf("%d ", (int) inputs[kk]);
      printf("\n");
    }

    //**/ -----------------------------------------------------------
    //**/ update history to speed up re-visits
    //**/ -----------------------------------------------------------
    if (nHist >= maxHist)
    {
      for (jj = 0; jj < maxHist/2; jj++)
        for (ii = 0; ii < nInputs+1; ii++)
          matHistory.setEntry(jj,ii,
                    matHistory.getEntry(jj+maxHist/2,ii));
      nHist = maxHist/2;
    }
    for (ii = 0; ii < nInputs; ii++)
      matHistory.setEntry(nHist,ii,inputs[ii]);
    matHistory.setEntry(nHist,nInputs,outputs[0]);
    nHist++;
  }

  //**/ =============================================================
  //**/ option: ODOE slow G/I/A/D/E-optimal (option 20-24, as opposed 
  //**/ to LocalFunction 10-14, uses MCMC at every iteration and thus 
  //**/ may be slower.) This is used by:
  //**/ - SCE optimization (odoeu_boptn)
  //**/ - brute force search (odoeu_boptnbf)
  //**/ - for single design evaluation (odoeu_beval) 
  //**/ -------------------------------------------------------------
  //**/ Bayes  20: G; 21: I; 22: A; 23: D; 24: E 
  //**/ =============================================================
  else if (whichLocalFunction_ >= 20 && whichLocalFunction_ <= 24)
  {
    //**/ -----------------------------------------------------------
    //**/ error checking (the objective function must have size = 1
    //**/ even though the simulator has multiple outputs), except
    //**/ when called by odoeu_boptnbf, which uses nOutputs=5. But, 
    //**/ if if nInputsIn == 0, it is asking for multiple metric
    //**/ evaluation so it is okay to for nOutputs>1
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0)
    {
      //**/ nOutputs=5 is allowed if function code = 20, meaning
      //**/ that odoeu_boptnbf is calling this function
      if (nOutputs == 5 && whichLocalFunction_ != 20)
      {
        printf("FuncIO ERROR: nOutputs has to be = 1 (unless code=20)\n");
        exit(1);
      }
      //**/ otherwise, if nInputsIn > 0 and nOutputs != 1, there
      //**/ is a problem
      else if (nOutputs != 1 && whichLocalFunction_ != 20)
      {
        printf("FuncIO ERROR: nOutputs has to be = 1\n");
        exit(1);
      }
    }

    //**/ -----------------------------------------------------------
    //**/ initialization 
    //**/ -----------------------------------------------------------
    psVector vecXT, vecYT;
    if (ProblemInitialized == 0)
    {
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (MCMC): Initialization begins ..\n");

      //**/ --- read training sample ==> psuadeIO
      printf("This method needs a training sample to create ");
      printf("response surfaces for\n");
      printf("inference and for analyzing the evaluation set.\n");
      snprintf(pString,100,
          "Enter name of the training sample (in PSUADE format): ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      PsuadeData *psuadeIO = new PsuadeData();
      status = psuadeIO->readPsuadeFile(fname);
      if (status != 0)
      {
        printf("FuncIO ERROR when reading training sample\n");
        printf("       Maybe this file is in wrong format?\n");
        exit(1);
      }
      psuadeIO->getParameter("input_ninputs", pdata);
      nInps = pdata.intData_;
      psuadeIO->getParameter("output_noutputs", pdata);
      nOuts = pdata.intData_;
      if (nOuts != 1)
      {
        printf("FuncIO ERROR: This method only works with ");
        printf("nOutputs=1 in the training\n");
        printf("              sample.\n");
        printf("Suggestion: Delete all but one output and re-run.\n");
        exit(1);
      }

      //**/ --- user selects uncertain inputs ==> vecUInputs
      vecUInputs.setLength(nInps);
      snprintf(pString,100,
         "Enter uncertain input number (1 - %d, 0 to end) : ",nInps);
      ii = 0;
      while (1)
      {
        kk = getInt(0, nInps, pString);
        if (kk == 0 || kk > nInps) break;
        vecUInputs[ii] = kk - 1;
        ii++;
      }
      vecUInputs.subvector(0, ii-1);
      int nUInps = vecUInputs.length();

      //**/ --- set uncertain parameter input to 1 in vecIT
      //**/ --- e.g. vecIT[kk]=1 if input kk is uncertain
      vecIT.setLength(nInps);
      kk = 1;
      for (ii = 0; ii < vecUInputs.length(); ii++)
      {
        vecIT[vecUInputs[ii]] = kk;
        kk++; 
      }

      //**/ --- read prior sample ==> matPriorSample
      printf("Uncertain parameters need a prior sample for inference.\n");
      printf("This file should have the following format:\n");
      printf("Line 1: <nSamples> <nInputs>\n");
      printf("Line 2: 1 <sample 1 values>\n");
      printf("Line 3: 2 <sample 2 values>\n");
      printf("Line 4: ...\n");
      snprintf(pString,100,"Enter the file name of your prior sample : ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      matPriorSample.setFormat(PS_MAT2D); // This is needed for later sort
      status = readIReadDataFile(fname, matPriorSample);
      if (status != 0)
      {
        printf("FuncIO ERROR when reading prior sample\n");
        FILE *fchk = fopen(fname, "r");
        if (fchk == NULL)
          printf("       Prior sample file does not exist.\n");
        else
        {
          fclose(fchk);
          printf("       Maybe this file is in wrong format?\n");
        }
        exit(1);
      }
      if (matPriorSample.ncols() != vecUInputs.length())
      {
        printf("FuncIO ERROR: prior sample nInputs=%d is not correct.\n",
               matPriorSample.ncols());
        printf("       Should be equal to %d.\n",vecUInputs.length());
        exit(1);
      }

      //**/ --- read candidate set ==> matCandidates
      //**/ --- If no inputs is given, it means the calling function
      //**/ --- is requesting using selected designs for analysis
      if (nInputs == 0 && nOutputs != 0)
      {
        printf("Next please provide a SELECTED candidate ");
        printf("subset to be evaluated as a\n");
        printf("whole (that is, not individually). The file ");
        printf("should be in the following\n");
        printf("format: (output mean and std dev may be computed by odoeu_rseval)\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <design 1 values> <output mean> <std dev>\n");
        printf("Line 3: <design 2 values> <output mean> <std dev>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,
          "Enter the file name of your selected designs : ");
      }
      else
      {
        printf("Next please provide a candidate set. The file must ");
        printf("be in the following\n");
        printf("format: (output mean and std dev may be computed ");
        printf("by odoeu_rseval)\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <design 1 values> <output mean> <std dev>\n");
        printf("Line 3: <design 2 values> <output mean> <std dev>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,"Enter the file name of your candidate set : ");
      }
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      status = readIReadDataFile(fname, matCandidates);
      if (status != 0)
      {
        printf("FuncIO ERROR when reading candidate set\n");
        printf("       Maybe this file is in wrong format?\n");
        exit(1);
      }
      int nCandidates = matCandidates.nrows();
      if (nInputs == 0 && nOutputs != 0)
        printf("Size of the selected set = %d\n", nCandidates);
      else
        printf("Size of Candidate set = %d\n", nCandidates);

      //**/ check that the candidate set has right information
      if (matCandidates.ncols() != 2*nOuts+nInps-nUInps)
      {
        printf("FuncIO ERROR: Candidate file must have %d columns.\n",
               2*nOuts+nInps-nUInps);
        printf("              Only %d columns are found.\n",
               matCandidates.ncols());
        printf("INFO: The reason for this error may be because ");
        printf("you need for each\n");
        printf("      candidate in your candidate set file also ");
        printf("estimates of output\n");
        printf("      mean and standard deviation. To do so, use ");
        printf("odoeu_rseval (after\n");
        printf("      loading training sample, provide your ");
        printf("candidate set and also a\n");
        printf("      prior sample for the uncertain inputs) to ");
        printf("append your candidate\n");
        printf("      set with output means and standard deviations.\n");
        exit(1);
      }

      //**/ --- read evaluation set (for G and I) ==> matEvalSet
      if ((whichLocalFunction_ == 20 || whichLocalFunction_ == 21))
      {
        printf("An evaluation sample is needed to ");
        printf("compute the optimality metrics.\n");
        printf("This can be the same as the candidate set (but ");
        printf("not recommended). The\n");
        printf("evaluation set file must be in the following format :\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <design 1 values>\n");
        printf("Line 3: <design 2 values>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,
                "Enter the file name of your evaluation set : ");
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        status = readIReadDataFile(fname, matEvalSet);
        if (status != 0)
        {
          printf("FuncIO ERROR when reading evaluation set\n");
          printf("       Maybe this file is in wrong format?\n");
          exit(1);
        }
        if (matEvalSet.ncols() != 2*nOuts+nInps-nUInps &&
            matEvalSet.ncols() != nInps-nUInps)
        {
          printf("FuncIO ERROR: evaluation file must have %d or %d columns.\n",
                 2*nOuts+nInps-nUInps,nInps-nUInps);
          exit(1);
        }
      }

      //**/ --- construct response surface (psuadeIO ==> rsPtr)
      pData pInps, pOuts, pLBs, pUBs;
      psuadeIO->getParameter("method_nsamples", pdata);
      int nSamp = pdata.intData_;
      psuadeIO->getParameter("input_lbounds", pLBs);
      psuadeIO->getParameter("input_ubounds", pUBs);
      psuadeIO->getParameter("input_sample", pInps);
      psuadeIO->getParameter("output_sample", pOuts);
  
      int faFlag = 1, rsMethod=0;
      vecYT.setLength(nSamp);
      rsPtrs = new FuncApprox*[nOuts];
      for (ii = 0; ii < nOuts; ii++)
      {
        if (ii == 0) 
        {
          rsPtrs[ii] = genFAInteractive(psuadeIO, faFlag);
          rsMethod = rsPtrs[ii]->getID();
        }
        else 
        {
          //**/ all outputs use the same RS method
          psuadeIO->updateAnalysisSection(-1,-1,rsMethod,-1,-1,-1);
          faFlag = 0;
          rsPtrs[ii] = genFAInteractive(psuadeIO, faFlag);
        }
        rsPtrs[ii]->setBounds(pLBs.dbleArray_,pUBs.dbleArray_);
        rsPtrs[ii]->setOutputLevel(0);
        for (jj = 0; jj < nSamp; jj++)
          vecYT[jj] = pOuts.dbleArray_[jj*nOuts+ii];
        status = rsPtrs[ii]->initialize(pInps.dbleArray_,
                                        vecYT.getDVector());
      }

      //**/ --- the rest of the initialization
      delete psuadeIO;
      ProblemInitialized = 1;
      matHistory.setDim(maxHist,nInputs+1);
      nHist = 0;
      optimalVal = PSUADE_UNDEFINED;

      //**/ --- initialize MCMC object for subsequent processing
      mobj14.printLevel_ = 0;
      mobj14.nSamples_ = nSamp;
      mobj14.nInputs_ = nInps;
      mobj14.nOutputs_ = nOuts;
      mobj14.VecLowerB_.load(nInps, pLBs.dbleArray_);
      mobj14.VecUpperB_.load(nInps, pUBs.dbleArray_);
      mobj14.VecSamInputs_.load(nInps*nSamp, pInps.dbleArray_);
      mobj14.VecSamOutputs_.load(nOuts*nSamp, pOuts.dbleArray_);
      mobj14.VecCUInputs_ = vecUInputs;
      mobj14.faType_ = rsMethod;
      mobj14.MatPriorSample_ = matPriorSample;
      mobj14.rsPtrs_ = new FuncApprox*[nOuts];
      for (ii = 0; ii < nOuts; ii++) mobj14.rsPtrs_[ii] = rsPtrs[ii];
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (MCMC) Initialization complete.\n");
    }

    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL and outputs == NULL, evaluate each point
    //**/ in the set individually (support odoeu_beval)
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0 && nOutputs == 0)
    {
      int cc, nCandidates = matCandidates.nrows(), iOne=1;
      int nUInps = vecUInputs.length();
      double   dmean, dmean2, dcov;
      psVector vecEigs;
      psMatrix matEig, matCov;
      psMatrix matExpInps, matExpMeans, matExpStds;

      matExpInps.setDim(iOne, nInps-nUInps);
      matExpMeans.setDim(iOne, nOuts);
      matExpStds.setDim(iOne, nOuts);
      MCMCAnalyzer *mcmcAnalyzer = new MCMCAnalyzer();
      vecXT.setLength(nInps);
      double **postSample = NULL;

      //**/ compute metric for each candidate
      printf("Candidate G-metric\t   I-metric\t   D-metric\t");
      printf("  A-metric\t  E-metric\n");
      for (cc = 0; cc < nCandidates; cc++)
      {
        GMetric = IMetric = AMetric = EMetric = 0;
        DMetric = 1;

        //**/ fill vecXT with mean of uncertain inputs and values
        //**/ of candidate cc
        lcnt = 0;
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, compute vecXT = mean
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            dmean = 0;
            for (kk = 0; kk < matPriorSample.nrows(); kk++)
              dmean += matPriorSample.getEntry(kk,ind);
            vecXT[jj] = dmean / (double) matPriorSample.nrows();
          }
          //**/ else get the vecXT from candidate matrix
          else
          {
            vecXT[jj] = matCandidates.getEntry(cc,lcnt);
            matExpInps.setEntry(0, lcnt, vecXT[jj]);
            lcnt++;
          }
        }
        //**/ fill in the experimental data matrix
        for (jj = 0; jj < nOuts; jj++)
        {
          //**/ get experimental mean/std from user
          dmean = matCandidates.getEntry(cc,nInps-nUInps+jj*2);
          matExpMeans.setEntry(0,jj,dmean);
          ddata = matCandidates.getEntry(cc,nInps-nUInps+jj*2+1);
          matExpStds.setEntry(0,jj,ddata);
        }
        //**/ inference
        mobj14.MatExpInputs_ = matExpInps;
        mobj14.MatExpMeans_ = matExpMeans;
        mobj14.MatExpStds_ = matExpStds;
        mcmcAnalyzer->analyzeDirect(mobj14);

        //**/ get posterior sample and evaluate G- and I-metric
        postSample = mobj14.MatPostSample_.getMatrix2D();
        if (mobj14.MatPostSample_.nrows() > 0)
        {
          vecYT.setLength(mobj14.MatPostSample_.nrows());
          for (ii = 0; ii < matEvalSet.nrows(); ii++)
          {
            lcnt = 0;
            for (jj = 0; jj < nInps; jj++)
            {
              if (vecIT[jj] == 0)
              {
                vecXT[jj] = matEvalSet.getEntry(ii,lcnt);
                lcnt++;
              }
            }
            aggrVar = 0;
            for (jj = 0; jj < nOuts; jj++)
            {
              for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
              {
                for (ll = 0; ll < vecUInputs.length(); ll++)
                {
                  ind = vecUInputs[ll];
                  vecXT[ind] = postSample[kk][ll];
                }
                ddata = rsPtrs[jj]->evaluatePoint(vecXT.getDVector());
                vecYT[kk] = ddata;
              }
              dmean = dvar = 0;
              for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
                dmean += vecYT[kk];
              dmean /= (double) mobj14.MatPostSample_.nrows();
              for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
                dvar += pow(vecYT[kk]-dmean,2);
              dvar = dvar / (double) mobj14.MatPostSample_.nrows();
              aggrVar += dvar;
            }
            aggrVar /= nOuts;
            if (aggrVar > GMetric) GMetric = aggrVar;
            IMetric += aggrVar;
          }
          IMetric /= (double) matEvalSet.nrows();
        }
        else GMetric = IMetric = PSUADE_UNDEFINED;

        // D-, A-, and E-metrics
        if (mobj14.MatPostSample_.nrows() > 0)
        {
          matCov.setDim(vecUInputs.length(),vecUInputs.length());
          for (jj = 0; jj < vecUInputs.length(); jj++)
          {
            dmean = dvar = 0.0;
            for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
              dmean += postSample[kk][jj];
            dmean /= (double) mobj14.MatPostSample_.nrows();
            for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
              dvar += pow(postSample[kk][jj]-dmean,2.0);
            dvar /= (double) mobj14.MatPostSample_.nrows();
            dvar = dvar / (double) mobj14.MatPostSample_.nrows();
            matCov.setEntry(jj,jj,dvar);
            AMetric += dvar;
            for (ll = jj+1; ll < vecUInputs.length(); ll++)
            {
              dmean2 = 0;
              for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
                dmean2 += postSample[kk][ll];
              dmean2 /= (double) mobj14.MatPostSample_.nrows();
              dcov = 0;
              for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
                dcov += (postSample[kk][jj]-dmean)*
                        (postSample[kk][ll]-dmean2);
              dcov /= (double) mobj14.MatPostSample_.nrows();
              matCov.setEntry(jj,ll,dcov);
              matCov.setEntry(ll,jj,dcov);
            }
          }
          matCov.eigenSolve(matEig, vecEigs, 1);
          for (jj = 0; jj < vecUInputs.length(); jj++)
          {
            DMetric *= vecEigs[jj];
            if (vecEigs[jj] > EMetric) EMetric = vecEigs[jj];
          }
        }
        else DMetric = AMetric = EMetric = PSUADE_UNDEFINED;
        printf("%5d \t%12.4e\t%12.4e\t%12.4e\t%12.4e\t%12.4e\n",cc+1,
               GMetric,IMetric,DMetric,AMetric,EMetric);      
      }
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL, create an inputs of all candidates
    //**/ (because when nInputsIn == 0, it is assumed that callers
    //**/ wants to use the candidate set instead of from 'inputs')
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0)
    {
      nInputs = matCandidates.nrows();
      vecInps.setLength(nInputs);
      inputs = vecInps.getDVector();
      for (ii = 0; ii < nInputs; ii++) inputs[ii] = ii + 1;
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (duplication selection
    //**/ is not allowed). If so, just return a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = inputs[ii];
        kk  = inputs[jj];
        if (ind == kk) count++;
      }
    }
    if (count > 0)
    {
      if (nInputsIn == 0)
      {
        printf(" ==> ERROR: duplicate selection ");
        for (ii = 0; ii < nOutputs; ii++) outputs[ii] = 0;
        return 0;
      }
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      printf(" ==> duplicate selection ");
      for (ii = 0; ii < nInputs; ii++) printf("%d ", (int) inputs[ii]);
      printf(" ==> skip\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ error checking (whether inputs are valid) and display
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0 && printLevel_ >= 0) 
    {
      if (whichLocalFunction_ == 20) 
        printf("GOPTIMAL (MCMC) inputs: ");
      else if (whichLocalFunction_ == 21) 
        printf("IOPTIMAL (MCMC) inputs: ");
      else if (whichLocalFunction_ == 22)
        printf("DOPTIMAL (MCMC) inputs: ");
      else if (whichLocalFunction_ == 23) 
        printf("AOPTIMAL (MCMC) inputs: ");
      else if (whichLocalFunction_ == 24) 
        printf("EOPTIMAL (MCMC) inputs: ");
    }

    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) inputs[ii];
      if (ind < 1 || ind > matCandidates.nrows())
      {
        printf("ODOE Function Evaluator ERROR: wrong input values.\n");
        printf("     Check candidate set size consistency.\n");
        printf("The erroneous inputs are:\n");
        for (jj = 0; jj < nInputs; jj++)
          printf("Candidate %3d for evaluation = %d\n", jj+1, 
                 (int) inputs[jj]);
        printf("But they should all be in the range of [1,%d]\n",
               matCandidates.nrows());
        exit(1);
      }
      if (nInputsIn > 0 && printLevel_ >= 0) printf("%5d ", ind);
    }
    if (nInputsIn > 0 && printLevel_ >= 0) printf("\n");

    //**/ -----------------------------------------------------------
    //**/ check history to see whether this has been evaluated before
    //**/ if so, just fetch it from history record
    //**/ -----------------------------------------------------------
    if (nOutputs == 1)
    {
      for (jj = 0; jj < nHist; jj++)
      {
        count = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          for (kk = 0; kk < nInputs; kk++)
            if (matHistory.getEntry(jj,kk) == inputs[ii])
              break;
          if (kk != nInputs) count++;
        }
        if (count == nInputs)
        {
          outputs[0] = matHistory.getEntry(jj,nInputs);
          if (printLevel_ > 0) 
            printf(" ===> output = %11.4e (revisit)\n", outputs[0]);
          return 0; 
        }
      }
    }

    //**/ -----------------------------------------------------------
    //**/ For the Bayesian metrics (G, I, D, A, E)
    //**/ instantiate MCMC object 
    //**/ -----------------------------------------------------------
    int ind2, nDesInps = nInps - vecUInputs.length();
    psMatrix matExpInps, matExpMeans, matExpStds;
    matExpInps.setDim(nInputs, nInps-vecUInputs.length());
    matExpMeans.setDim(nInputs, nOuts);
    matExpStds.setDim(nInputs, nOuts);
    MCMCAnalyzer *mcmcAnalyzer = new MCMCAnalyzer();
    vecXT.setLength(nInps);

    //**/ -----------------------------------------------------------
    //**/ fill in the experimental data matrix for MCMC
    //**/ (matExpInps, matExpMeans, matExpStds)
    //**/ -----------------------------------------------------------
    for (ii = 0; ii < nInputs; ii++)
    {
      ind2 = (int) inputs[ii] - 1;
      lcnt = 0;
      for (jj = 0; jj < nInps; jj++)
      {
        //**/ if parameter is uncertain, compute vecXT = mean
        if (vecIT[jj] >= 1)
        {
          ind = vecIT[jj] - 1;
          dmean = 0;
          for (kk = 0; kk < matPriorSample.nrows(); kk++)
            dmean += matPriorSample.getEntry(kk,ind);
          vecXT[jj] = dmean / (double) matPriorSample.nrows();
        }
        //**/ else get the vecXT from candidate matrix
        else
        {
          vecXT[jj] = matCandidates.getEntry(ind2,lcnt);
          matExpInps.setEntry(ii, lcnt, vecXT[jj]);
          lcnt++;
        }
      }
      for (jj = 0; jj < nOuts; jj++)
      {
        //**/ get experimental mean/std from user 
        dmean = matCandidates.getEntry(ind2,nDesInps+jj*2);
        matExpMeans.setEntry(ii,jj,dmean);
        ddata = matCandidates.getEntry(ind2,nDesInps+jj*2+1);
        //**/ July 2021: not set std dev = 0.1 * mean 
        //ddata = 0.2 * dmean;
        //if      (ddata < 0)  ddata = - ddata;
        //else if (ddata == 0) ddata = 0.1;
        matExpStds.setEntry(ii,jj,ddata);
      }
    }

    //**/ -----------------------------------------------------------
    //**/ stuff the MCMC data object and call MCMC
    //**/ -----------------------------------------------------------
    mobj14.MatExpInputs_ = matExpInps;
    mobj14.MatExpMeans_ = matExpMeans;
    mobj14.MatExpStds_ = matExpStds;
    mcmcAnalyzer->analyzeDirect(mobj14);

    //**/ -----------------------------------------------------------
    //**/ extract posterior sample and evaluate
    //**/ -----------------------------------------------------------
    double **postSample = mobj14.MatPostSample_.getMatrix2D();
    GMetric = IMetric = AMetric = EMetric = 0;
    DMetric = 1;

    //**/ -----------------------------------------------------------
    //**/ only G- and I-optimal metrics need the following
    //**/ -----------------------------------------------------------
    if (whichLocalFunction_ == 20 || whichLocalFunction_ == 21)
    {
      //**/ #########################################################
      //**/ this portion calls rsPtr, and is done at each iteration.
      //**/ so if RS evaluation is expensive, this portion may be
      //**/ slow (2020)
      //**/ #########################################################
      vecXT.setLength(nInps);
      if (mobj14.MatPostSample_.nrows() > 0)
      {
        vecYT.setLength(mobj14.MatPostSample_.nrows());
        for (ii = 0; ii < matEvalSet.nrows(); ii++)
        {
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          {
            if (vecIT[jj] == 0)
            {
              vecXT[jj] = matEvalSet.getEntry(ii,lcnt);
              lcnt++;
            }
          }
          aggrVar = 0;
          for (jj = 0; jj < nOuts; jj++)
          {
            for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
            {
              for (ll = 0; ll < vecUInputs.length(); ll++)
              {
                ind = vecUInputs[ll];
                vecXT[ind] = postSample[kk][ll];
              }
              ddata = rsPtrs[jj]->evaluatePoint(vecXT.getDVector());
              vecYT[kk] = ddata;
            }
            dmean = dvar = 0;
            for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
              dmean += vecYT[kk];
            dmean /= (double) mobj14.MatPostSample_.nrows();
            for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
              dvar += pow(vecYT[kk]-dmean,2);
            dvar = dvar / (double) mobj14.MatPostSample_.nrows();
            aggrVar += dvar;
          }
          aggrVar /= nOuts;
          if (aggrVar > GMetric) GMetric = aggrVar;
          IMetric += aggrVar;
        }
        IMetric /= (double) matEvalSet.nrows();
      }
      else IMetric = GMetric = PSUADE_UNDEFINED;
    }
    //**/ -----------------------------------------------------------
    //**/ D-, A-, and E-optimal metrics need the following
    //**/ (These are computed even only G- or I-metric is requested
    //**/ to facilitate the odoeu_eval command) 
    //**/ -----------------------------------------------------------
    psMatrix matCov, matEig;
    psVector vecEigs;
    matCov.setDim(vecUInputs.length(),vecUInputs.length());
    double dmean2, dcov;
    if (mobj14.MatPostSample_.nrows() > 0)
    {
      AMetric = 0.0;
      for (jj = 0; jj < vecUInputs.length(); jj++)
      {
        dmean = 0.0;
        for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
          dmean += postSample[kk][jj];
        dmean /= (double) mobj14.MatPostSample_.nrows();
        for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
          dvar += pow(postSample[kk][jj]-dmean,2.0);
        dvar /= (double) mobj14.MatPostSample_.nrows();
        matCov.setEntry(jj,jj,dvar);
        AMetric += dvar;
        for (ll = jj+1; ll < vecUInputs.length(); ll++)
        {
          dmean2 = 0;
          for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
            dmean2 += postSample[kk][ll];
          dmean2 /= (double) mobj14.MatPostSample_.nrows();
          dcov = 0;
          for (kk = 0; kk < mobj14.MatPostSample_.nrows(); kk++)
            dcov += (postSample[kk][jj]-dmean)*
                    (postSample[kk][ll]-dmean2);
          dcov /= (double) mobj14.MatPostSample_.nrows();
          matCov.setEntry(jj,ll,dcov);
          matCov.setEntry(ll,jj,dcov);
        }
      }
      matCov.eigenSolve(matEig, vecEigs, 1);
      DMetric = 1.0;
      EMetric = 0;
      for (jj = 0; jj < vecUInputs.length(); jj++)
      {
        DMetric *= vecEigs[jj];
        if (vecEigs[jj] > EMetric) EMetric = vecEigs[jj];
      }
    }
    else AMetric = DMetric = EMetric = PSUADE_UNDEFINED;

    //**/ -----------------------------------------------------------
    //**/ return the metric information
    //**/ if nInputsIn = 0, it is assumed odoeu_eval calls this 
    //**/ function is called to evaluate all metrics; and to
    //**/ expedite things, one call with code 20 will return all 5
    //**/ metrics 
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0) 
    {
      if (nOutputs >= 1) outputs[0] = GMetric;
      if (nOutputs >= 2) outputs[1] = IMetric;
      if (nOutputs >= 3) outputs[2] = DMetric;
      if (nOutputs >= 4) outputs[3] = AMetric;
      if (nOutputs >= 5) outputs[4] = EMetric;
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ return the relevant metric information
    //**/ -----------------------------------------------------------
    if (whichLocalFunction_ == 20 && nOutputs == 5)
    {
      outputs[0] = GMetric;
      outputs[1] = IMetric;
      outputs[2] = DMetric;
      outputs[3] = AMetric;
      outputs[4] = EMetric;
    }
    else 
    {
      if      (whichLocalFunction_ == 20) outputs[0] = GMetric;
      else if (whichLocalFunction_ == 21) outputs[0] = IMetric;
      else if (whichLocalFunction_ == 22) outputs[0] = DMetric;
      else if (whichLocalFunction_ == 23) outputs[0] = AMetric;
      else if (whichLocalFunction_ == 24) outputs[0] = EMetric;
      if (outputs[0] < optimalVal) optimalVal = outputs[0];
      if (nInputsIn > 0 && printLevel_ >= 0) 
      {
        printf(" ===> output = %11.4e (best so far = %11.4e",
               outputs[0], optimalVal);
        if (outputs[0] == optimalVal) 
        {
          printf(" :");
          for (kk = 0; kk < nInputs; kk++)
            printf(" %d", (int) inputs[kk]);
        }
        printf(")\n");
      }
    }

    //**/ -----------------------------------------------------------
    //**/ update history for faster evaluation for revisits
    //**/ -----------------------------------------------------------
    if (nOutputs == 1)
    {
      if (nHist >= maxHist)
      {
        for (jj = 0; jj < maxHist/2; jj++)
          for (ii = 0; ii < nInputs+1; ii++)
            matHistory.setEntry(jj,ii,
                      matHistory.getEntry(jj+maxHist/2,ii));
        nHist = maxHist/2;
      }
      for (ii = 0; ii < nInputs; ii++)
        matHistory.setEntry(nHist,ii,inputs[ii]);
      matHistory.setEntry(nHist,nInputs,outputs[0]);
      nHist++;
    }
    return 0;
  }

  //**/ =============================================================
  //**/ option: G/I/A/D/E-optimal using the Fisher method and with 
  //**/ SCE optimization
  //**/ -------------------------------------------------------------
  //**/ Fisher 25: G; 26: I; 27: A; 28: D; 29: E 
  //**/ =============================================================
  else if (whichLocalFunction_ >= 25 && whichLocalFunction_ <= 29)
  {
    //**/ -----------------------------------------------------------
    //**/ error checking (the objective function must have size = 1
    //**/ even though the simulator has multiple outputs), except
    //**/ when called by odoeu_foptnb, which uses nOutputs=5. But, 
    //**/ if if nInputsIn == 0, it is asking for multiple metri
    //**/ evaluation so it is okay to for nOutputs>1
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0)
    {
      //**/ nOutputs=5 is allowed if function code = 25, meaning
      //**/ that odoeu_boptnbf is calling this function
      if (nOutputs == 5 && whichLocalFunction_ != 25)
      {
        printf("FuncIO ERROR: nOutputs has to be = 1 (unless code=25)\n");
        exit(1);
      }
      else if (nOutputs == 3 && whichLocalFunction_ != 25)
      {
        printf("FuncIO ERROR: nOutputs has to be = 1 (unless code=25)\n");
        exit(1);
      }
      //**/ otherwise, if nInputsIn > 0 and nOutputs != 1, there
      //**/ is a problem
      else if (nOutputs != 1 && whichLocalFunction_ != 25)
      {
        printf("FuncIO ERROR: nOutputs has to be = 1\n");
        exit(1);
      }
    }

    //**/ -----------------------------------------------------------
    //**/ initialization 
    //**/ -----------------------------------------------------------
    psVector vecXT, vecYT;
    if (ProblemInitialized == 0)
    {
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (Fisher): Initialization begins ..\n");

      //**/ --- read training sample ==> psuadeIO
      printf("The key to this class of methods is to create ");
      printf("the Fisher information\n");
      printf("matrices and then derives metrics from them. In ");
      printf("order to create the\n");
      printf("Fisher information matrix, the following information ");
      printf("are needed:\n");
      printf("- A training sample to compute the Fisher matrix\n");
      printf("  (and identify design and uncertain parameters)\n");
      printf("- A prior distribution for the uncertain parameters\n");
      printf("- A candidate set (or a selected subset) of ");
      printf("experimental designs\n");

      //**/ get the training sample
      snprintf(pString,100,
          "Enter name of the training sample (in PSUADE format): ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      PsuadeData *psuadeIO = new PsuadeData();
      status = psuadeIO->readPsuadeFile(fname);
      if (status != 0)
      {
        printf("FuncIO: ERROR encountered when reading the ");
        printf("training sample\n");
        printf("Possible reasons: \n");
        printf("- this file does not exist\n");
        printf("- this file is in wrong format (not PSUADE format)\n");
        exit(1);
      }
      psuadeIO->getParameter("input_ninputs", pdata);
      nInps = pdata.intData_;
      psuadeIO->getParameter("output_noutputs", pdata);
      nOuts = pdata.intData_;
      if (nOuts > 1)
      {
        printf("FuncIO ERROR: This method only works with nOutputs=1 ");
        printf("for training the\n");
        printf("              sample\n");
        printf("Suggestion: Delete all but one output and re-run.\n");
        exit(1);
      }

      //**/ --- user selects uncertain inputs ==> vecUInputs
      printDashes(PL_INFO,0);
      printf("Out of the %d inputs, some must be design parameters ",
             nInputs_);
      printf("and some must be\n");
      printf("uncertain parameters. In the following, please specify ");
      printf("which inputs\n");
      printf("are uncertain (and the rest will be design parameters).\n");
      vecUInputs.setLength(nInps);
      snprintf(pString,100,
         "Enter uncertain input number (1 - %d, 0 to terminate) : ",nInps);
      ii = 0;
      while (1)
      {
        kk = getInt(0, nInps, pString);
        if ((kk == 0 || kk > nInps) && ii > 0) break;
        vecUInputs[ii] = kk - 1;
        ii++;
      }
      vecUInputs.subvector(0, ii-1);
      int nUInps = vecUInputs.length();

      //**/ in order for the Fisher matrix to be well-conditioned, the
      //**/ following condition has to be met.
      if (nInputsIn != 0 && nInputsIn < nUInps)
      {
        printf("ERROR: Selected candidate subset must be larger ");
        printf("than the number of\n");
        printf("       uncertain parameters.\n");
        printf("- candidate set has %d members\n",nInputsIn);
        printf("- number of uncertain parameters = %d\n",nUInps);
        exit(1);
      }

      //**/ --- set uncertain parameter input to 1 in vecIT
      //**/ --- e.g. vecIT[kk]=1 if input kk is uncertain
      vecIT.setLength(nInps);
      kk = 1;
      for (ii = 0; ii < vecUInputs.length(); ii++)
      {
        vecIT[vecUInputs[ii]] = kk;
        kk++; 
      }

      //**/ --- read prior sample ==> matPriorSample
      printDashes(PL_INFO,0);
      printf("Uncertain parameters need a prior sample. The sample ");
      printf("file should\n");
      printf("have the following format:\n");
      printf("Line 1: <nSamples> <nInputs>\n");
      printf("Line 2: 1 <sample 1 values>\n");
      printf("Line 3: 2 <sample 2 values>\n");
      printf("Line 4: ...\n");
      snprintf(pString,100,"Enter the file name of your prior sample : ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      matPriorSample.setFormat(PS_MAT2D); // This may not be needed
      status = readIReadDataFile(fname, matPriorSample);
      if (status != 0)
      {
        printf("FuncIO: ERROR encountered when reading the prior sample\n");
        printf("Possible reasons: \n");
        printf("- this file does not exist\n");
        printf("- this file is in wrong format\n");
        exit(1);
      }
      if (matPriorSample.ncols() != vecUInputs.length())
      {
        printf("FuncIO ERROR: Prior sample nInputs=%d is not correct.\n",
               matPriorSample.ncols());
        printf("       Should be equal to %d.\n",vecUInputs.length());
        exit(1);
      }

      //**/ option to use smaller prior sample (this will not be 
      //**/ asked if this is called by odoeu_eval)
      if (nInputsIn > 0 && (matPriorSample.nrows() > 10))
      {
        printf("* Fisher-based methods are computationally ");
        printf("expensive, so you may want\n");
        printf("  to reduce the cost by collapsing the ");
        printf("prior sample into fewer sample\n");
        printf("  points (prior sample size = %d).\n",
               matPriorSample.nrows());
        snprintf(pString,100,
           "Collapse prior sample into smaller sample ? (y or n) "); 
        getString(pString, lineIn);
        if (lineIn[0] == 'y')
        {
          printf("The current size of the prior sample is %d.\n",
                 matPriorSample.nrows());
          snprintf(pString,100,
            "Use (1) the sample mean only or (2) a random sub-sample ? ");
          kk = getInt(1, 2, pString);
          if (kk == 1)
          {
            int nr = matPriorSample.nrows(), nc = matPriorSample.ncols();
            psMatrix tmpMat = matPriorSample;
            matPriorSample.setFormat(PS_MAT2D);
            matPriorSample.setDim(iOne, nc);
            for (ii = 0; ii < nc; ii++)
            {
              ddata = 0;
              for (jj = 0; jj < nr; jj++)
                ddata += tmpMat.getEntry(jj, ii);
              ddata /= (double) nr;
              matPriorSample.setEntry(0, ii, ddata); 
            }
          }
          else if (kk == 2)
          {
            snprintf(pString,100,"Enter sub-sample size (%d - %d) : ",
                    nInputs, matPriorSample.nrows()-1);
            int nSamFisher = getInt(nInputs,matPriorSample.nrows()-1,pString);
            
            int nr = matPriorSample.nrows(), nc = matPriorSample.ncols();
            psMatrix tmpMat = matPriorSample;
            matPriorSample.setFormat(PS_MAT2D);
            matPriorSample.setDim(nSamFisher, nc);
            int nCurrent = 0, nTrials=0, checkFlag=1;
            while (nCurrent < nSamFisher)
            {
              nTrials++;
              if (nTrials > 20*nr)
              {
                printf("INFO: Unable to draw a sub-sample with ");
                printf("unique sample points.\n");
                printf("      Stop searching for a unique sub-sample.\n");
                checkFlag = 0;
              }
              kk = PSUADE_rand() % nr;
              //**/ check that there are no duplicates
              jj = 0;
              if (checkFlag == 1)
              {
                for (ii = 0; ii < nCurrent-1; ii++)
                {  
                  for (jj = 0; jj < nc; jj++)
                  {
                    ddata = matPriorSample.getEntry(ii,jj);
                    dtmp  = tmpMat.getEntry(kk,jj);
                    if (dtmp != ddata) break;
                  }
                  //**/ give up if the same sample point
                  if (jj == nc) break;
                }
              }
              if (jj != nc)
              {
                for (jj = 0; jj < nc; jj++)
                {
                  ddata = tmpMat.getEntry(kk,jj);
                  matPriorSample.setEntry(nCurrent,jj,ddata);
                }
                nCurrent++;
              }
            }
          }
        }
      }

      //**/ --- read candidate set ==> matCandidates
      //**/ --- If no inputs is given (inputsIn = NULL), it means 
      //**/ --- the calling function is requesting using selected
      //**/ --- designs for analysis instead of parameters passed 
      //**/ --- to this function
      if (nInputs == 0 && nOutputs != 0)
      {
        printf("* Next please provide a SELECTED candidate ");
        printf("subset to be evaluated as a\n");
        printf("  whole (that is, not individually). The ");
        printf("file must be in the following\n");
        printf("  format:\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <selected design point 1 values>\n");
        printf("Line 3: <selected design point 2 values>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,
         "Enter the file name of the SELECTED candidate subset : ");
      }
      else
      {
        printf("* Next please provide a candidate set. The ");
        printf("file should be in the\n");
        printf("  following format:\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <candidate design 1 values>\n");
        printf("Line 3: <candidate design 2 values>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,
                "Enter the file name of your candidate design set : ");
      }
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      status = readIReadDataFile(fname, matCandidates);
      if (status != 0)
      {
        printf("FuncIO: ERROR encountered when reading candidate set\n");
        printf("Possible reasons: \n");
        printf("- this file does not exist\n");
        printf("- this file is in wrong format\n");
        exit(1);
      }
      int nCandidates = matCandidates.nrows();
      if (nInputsIn == 0 && nOutputs != 0)
      {
        printf("Size of the selected design set = %d\n", nCandidates);
        if (nCandidates < nUInps)
        {
          printf("ERROR: Selected design set should have > %d members.\n",
                 nUInps);
          exit(1);
        }
      }
      else printf("Size of the candidate set = %d\n", nCandidates);

      //**/ --- read evaluation set (for G and I) ==> matEvalSet
      if ((whichLocalFunction_ == 25 || whichLocalFunction_ == 26) &&
          !(nInputsIn == 0 && nOutputs == 0))
      {
        printf("An evaluation sample is needed to ");
        printf("compute the optimality metrics.\n");
        printf("This can be the same as the candidate set");
        printf(" (but not recommended).\n");
        printf("The file should be in the following format: \n");
        printf("Line 1: <numPoints> <nInputs>\n");
        printf("Line 2: 1 <sample 1 values> \n");
        printf("Line 3: 2 <sample 2 values> \n");
        printf("....\n");
        snprintf(pString,100,
                "Enter the file name of your evaluation set : ");
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        status = readIReadDataFile(fname, matEvalSet);
        if (status != 0)
        {
          printf("FuncIO ERROR when reading evaluation set\n");
          printf("Possible reasons: \n");
          printf("- this file does not exist\n");
          printf("- this file is in wrong format\n");
          exit(1);
        }
        if (matEvalSet.ncols() != 2*nOuts+nInps-nUInps &&
            matEvalSet.ncols() != nInps-nUInps)
        {
          printf("FuncIO ERROR: Evaluation set must have %d or %d columns.\n",
                 2*nOuts+nInps-nUInps,nInps-nUInps);
          exit(1);
        }
      }

      //**/ --- construct response surface (psuadeIO ==> rsPtr)
      //**/ --- RS is needed to compute Fisher matrix
      pData pInps, pOuts, pLBs, pUBs;
      psuadeIO->getParameter("method_nsamples", pdata);
      int nSamp = pdata.intData_;
      psuadeIO->getParameter("input_lbounds", pLBs);
      psuadeIO->getParameter("input_ubounds", pUBs);
      psuadeIO->getParameter("input_sample", pInps);
      psuadeIO->getParameter("output_sample", pOuts);
  
      int faFlag = 1, rsMethod=0;
      vecYT.setLength(nSamp);
      rsPtrs = new FuncApprox*[1];
      rsPtrs[0] = genFAInteractive(psuadeIO, faFlag);
      rsPtrs[0]->setBounds(pLBs.dbleArray_,pUBs.dbleArray_);
      rsPtrs[0]->setOutputLevel(0);
      for (jj = 0; jj < nSamp; jj++)
        vecYT[jj] = pOuts.dbleArray_[jj];
      status = rsPtrs[0]->initialize(pInps.dbleArray_,
                                     vecYT.getDVector());

      //**/ --- the rest of the initialization
      delete psuadeIO;
      ProblemInitialized = 1;
      matHistory.setDim(maxHist,nInputs+1);
      nHist = 0;
      optimalVal = PSUADE_UNDEFINED;
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (Fisher) Initialization complete.\n");
    }

    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL and outputs == NULL, evaluate each point
    //**/ in the set individually (support odoeu_feval)
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0 && nOutputs == 0)
    {
      int cc, nCandidates = matCandidates.nrows(), iOne=1, lcnt;
      int nUInps = vecUInputs.length();
      int priorNR = matPriorSample.nrows();
      double DMetric, AMetric, EMetric;
      psMatrix matCov, matCovInv, matGrad, matGradT;

      vecXT.setLength(nInps);
      vecYT.setLength(iOne);
      matGrad.setDim(nUInps, iOne);

      //**/ compute metric for each candidate`
      printf("Candidate D-metric\t   A-metric\t   E-metric\n");
      for (cc = 0; cc < nCandidates; cc++)
      {
        AMetric = EMetric = 0;
        DMetric = 1;

        //**/ stuff candidate coordinate in vecXT
        //**/ (from candidate matrix)
        lcnt = 0;
        for (jj = 0; jj < nInps; jj++)
        {
          if (vecIT[jj] < 1) /* vecIT[jj] = 0 for design input jj */
          {
            vecXT[jj] = matCandidates.getEntry(cc,lcnt);
            lcnt++;
          }
        }
        //**/ now stuff vecXT with each sample in the prior sample
        //**/ evaluate function and derivatives and add to matGrad
        for (ss = 0; ss < priorNR; ss++)
        {
          //**/ --- first stuff the prior sample into vecXT
          for (jj = 0; jj < nInps; jj++)
          {
            //**/ if parameter is uncertain, use prior sample
            if (vecIT[jj] >= 1)
            {
              ind = vecIT[jj] - 1;
              vecXT[jj] = matPriorSample.getEntry(ss,ind);
            }
          }
          //**/ --- evaluate function and compute derivatives too
          rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
          vecYT[0] = ddata;

          //**/ compute partial y(theta) /partial theta_j
          for (jj = 0; jj < nInps; jj++)
          {
            //**/ if parameter is uncertain, perturb and evaluate
            if (vecIT[jj] >= 1)
            {
              ind = vecIT[jj] - 1;
              dtmp = vecXT[jj];
              vecXT[jj] *= (1.0 + 1e-6);
              rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
              //**/ finite difference delta Y_kk wrt theta_jj
              ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
              vecXT[jj] = dtmp;
              dtmp = matGrad.getEntry(ind, 0);
              ddata += dtmp;
              matGrad.setEntry(ind, 0, ddata);
            }
          }
        } /* ss for priorNR */
        //**/ take the mean for col cc in matGrad so afterward
        //**/ now matGrad is a m x c matrix consisting of
        //**/ [mean(df(x1)/dtheta_1) .. mean(df(xc)/dtheta_1)]
        //**/ ...
        //**/ [mean(df(x1)/dtheta_m) .. mean(df(x2)/dtheta_j)]
        for (jj = 0; jj < nUInps; jj++)
        {
          ddata = matGrad.getEntry(jj, 0);
          ddata /= (double) priorNR;
          matGrad.setEntry(jj, 0, ddata);
        }
        //**/ --- now compute information matrix (Grad * Grad^T)
        //**/ --- covariance matrix ~ inv(Grad * Grad^T)
        matGradT = matGrad;
        matGradT.transpose();
        matGrad.matmult(matGradT, matCovInv);
        ddata = matCovInv.computeDeterminant();
        if (PABS(ddata) < 1e-16)
        {
          printf("\nINFO: Fisher matrix is singular ==> skip.\n");
          DMetric = PSUADE_UNDEFINED;
          AMetric = PSUADE_UNDEFINED;
          EMetric = PSUADE_UNDEFINED;
          GMetric = PSUADE_UNDEFINED;
          IMetric = PSUADE_UNDEFINED;
        }
        else
        {
          //**/ --- accumulate 1/determinant for inverse covariance
          //**/     (since Fisher^{-1} approximates covariance)
          DMetric = 1.0 / ddata;

          //**/ --- now get back the covariance matrix by inversion
          matCovInv.computeInverse(matCov);

          //**/ Since this is for individual, the matrix is 1 x 1
          AMetric = matCov.getEntry(0, 0);

          //**/ eigenvalue is matrix at (0,0)
          EMetric = matCov.getEntry(0, 0);
        }
        printf("%5d \t%12.4e\t%12.4e\t%12.4e\n",cc+1,
               DMetric,AMetric,EMetric);
      }
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ if nInputsIn = 0 and inputs = NULL, create an inputs of all 
    //**/ candidates because in this case, it is assumed that callers
    //**/ want to use the entire candidate set instead of 'inputs' 
    //**/ (e.g. called from odoeu_feval)
    //**/ Otherwise, copy inputsIn to vecInps
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0 && inputsIn == NULL)
    {
      nInputs = matCandidates.nrows();
      vecInps.setLength(nInputs);
      inputs = vecInps.getDVector();
      for (ii = 0; ii < nInputs; ii++) inputs[ii] = ii + 1;
    }
    else
    {
      vecInps.setLength(nInputsIn);
      inputs = vecInps.getDVector();
      for (ii = 0; ii < nInputs; ii++) inputs[ii] = inputsIn[ii];
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (duplication selection
    //**/ is not allowed). If so, just return a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = (int) vecInps[ii];
        kk  = (int) vecInps[jj];
        if (ind == kk) count++;
      }
    }
    //**/ -----------------------------------------------------------
    //**/ if the inputs is from the candidate set (that is, the whole
    //**/ set is to be evaluated, and it has duplicates, compress it)
    //**/ -----------------------------------------------------------
    if (count > 0 && nInputsIn == 0)
    {
      if (nInputsIn == 0)
      {
        sortDbleList(nInputs, vecInps.getDVector());
        count = 1;
        for (ii = 1; ii < nInputs; ii++)
        {
          if (vecInps[count-1] != vecInps[ii]) 
          {
            vecInps[count] = vecInps[ii];
            count++;
          }
        }
        vecInps.subvector(0, nInputs-1);
      }
    }
    //**/ -----------------------------------------------------------
    //**/ if inputs are from optimizer, return large values
    //**/ (selection has to be unique in this case)
    //**/ -----------------------------------------------------------
    else if (count > 0) 
    {
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      //printf(" ==> duplicate selection (not a user concern) : ");
      //for (ii = 0; ii < nInputs; ii++) printf("%d ",(int) vecInps[ii]);
      //printf(" ==> skip\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ error checking (whether inputs are valid) and display
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0 && printLevel_ >= 0) 
    {
      if (whichLocalFunction_ == 25)
      {
        if (nInputsIn > 0)
          printf("GOPTIMAL (Fisher) inputs: ");
        else
          printf("<All>OPTIMAL (Fisher) inputs: ");
      }
      else if (whichLocalFunction_ == 26)
        printf("IOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 27)
        printf("DOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 28)
        printf("AOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 29)
        printf("EOPTIMAL (Fisher) inputs: ");
    }
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) vecInps[ii];
      if (ind < 1 || ind > matCandidates.nrows())
      {
        printf("FuncIO ERROR: Wrong input values.\n");
        printf("              Check candidate set size consistency.\n");
        printf("The erroneous inputs are:\n");
        for (jj = 0; jj < nInputs; jj++)
          printf("Candidate %3d for evaluation = %d\n", jj+1, 
                 (int) vecInps[jj]);
        printf("But they should all be in the range of [1,%d]\n",
               matCandidates.nrows());
        exit(1);
      }
      if (nInputsIn > 0 && printLevel_ >= 0) printf("%5d ", ind);
    }
    if (nInputsIn > 0 && printLevel_ >= 0) printf("\n");

    //**/ -----------------------------------------------------------
    //**/ search history to see if this has been evaluated before
    //**/ -----------------------------------------------------------
    if (nOutputs == 1)
    {
      for (jj = 0; jj < nHist; jj++)
      {
        count = 0;
        for (ii = 0; ii < nInputs; ii++)
        {
          for (kk = 0; kk < nInputs; kk++)
            if (matHistory.getEntry(jj,kk) == inputs[ii])
              break;
          if (kk != nInputs) count++;
        }
        if (count == nInputs)
        {
          outputs[0] = matHistory.getEntry(jj,nInputs);
          if (printLevel_ > 0) 
            printf(" ===> output = %11.4e (revisit)\n", outputs[0]);
          return 0; 
        }
      }
    }

    //**/ -----------------------------------------------------------
    //**/ if nInputs = nOutputs = 0, it means odoeu_feval is calling
    //**/ this function to evaluate the GIDAE metrics for all the
    //**/ candidate points in the candidate set.
    //**/ vecInps (double) has the candidate indices (1-based)
    //**/ This code is also called by odoeu_foptn to evaluate a
    //**/ given subset of candidates
    //**/ -----------------------------------------------------------
    vecXT.setLength(nInps); /* nInps = nInputs in training sample */
    vecYT.setLength(nOuts); /* nOuts = 1 */
    int nUInps = vecUInputs.length(); /* number of uncertain inputs */

#if 1
    //**/ -----------------------------------------------------------
    //**/ this 'if' segment is a different implementation than the
    //**/ 'else' segment, in the sense that this segment computes
    //**/ M = E_{theta} [dY(c_1)/d theta ... dY(c_n)/d theta]
    //**/ where M is the fisher matrix, E_{theta} is mean wrt
    //**/ the uncertain inputs theta, c_i is candidate i, Y is
    //**/ simulation output of interest.
    //**/ -----------------------------------------------------------
    psVector vecEig;
    psMatrix matGrad, matGradT, matCovInv, matEig;
    int priorNR = matPriorSample.nrows(), cc, cand;
    int nCandidates = vecInps.length();
    matGrad.setDim(nUInps, nCandidates);
    double DMetric2=0,AMetric2=0,EMetric2=0;
    double GMetric2=0, IMetric2=0;

    for (cc = 0; cc < nCandidates; cc++)
    {
      cand = (int) (vecInps[cc]) - 1; 
      //**/ stuff candidate coordinate in vecXT
      //**/ (from candidate matrix)
      lcnt = 0;
      for (jj = 0; jj < nInps; jj++)
      { 
        if (vecIT[jj] < 1) /* vecIT[jj] = 0 for design input jj */
        {
          vecXT[jj] = matCandidates.getEntry(cand,lcnt);
          lcnt++;
        } 
      }
      //**/ now stuff vecXT with each sample in the prior sample
      //**/ evaluate function and derivatives and add to matGrad
      for (ss = 0; ss < priorNR; ss++)
      {
        //**/ --- first stuff the prior sample into vecXT
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, use prior sample
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            vecXT[jj] = matPriorSample.getEntry(ss,ind);
          }
        }
        //**/ --- evaluate function and compute derivatives too
        rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
        vecYT[0] = ddata;

        //**/ compute partial y(theta) /partial theta_j
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, perturb and evaluate
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            dtmp = vecXT[jj];
            vecXT[jj] *= (1.0 + 1e-6);
            rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
            //**/ finite difference delta Y_kk wrt theta_jj
            ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
            vecXT[jj] = dtmp;
            dtmp = matGrad.getEntry(ind, cc);
            ddata += dtmp; 
            matGrad.setEntry(ind, cc, ddata); 
          }
        }
      } /* ss for priorNR */
      //**/ take the mean for col cc in matGrad so afterward
      //**/ now matGrad is a m x c matrix consisting of
      //**/ [mean(df(x1)/dtheta_1) .. mean(df(xc)/dtheta_1)] 
      //**/ ...
      //**/ [mean(df(x1)/dtheta_m) .. mean(df(x2)/dtheta_j)] 
      for (jj = 0; jj < nUInps; jj++)
      {
        ddata = matGrad.getEntry(jj, cc);
        ddata /= (double) priorNR;
        matGrad.setEntry(jj, cc, ddata);
      } 
    }
    //**/ --- now compute information matrix (Grad * Grad^T) 
    //**/ --- covariance matrix ~ inv(Grad * Grad^T)
    matGradT = matGrad;
    matGradT.transpose();
    matGrad.matmult(matGradT, matCovInv);
    ddata = matCovInv.computeDeterminant(); 
    if (PABS(ddata) < 1e-16)
    {
      printf("\nINFO: Fisher matrix is singular ==> skip.\n");
      //printf("This means some uncertain parameters are insensitive, ");
      //printf("and Fisher-based\n");
      //printf("optimality does not work well, or maybe there are ");
      //printf("redundant samples\n");
      //printf("in the prior sample set.\n");
      //printf("Suggestion: Try Bayes-based optimality criteria.\n");
      //exit(1);
      DMetric2 = PSUADE_UNDEFINED;
      AMetric2 = PSUADE_UNDEFINED;
      EMetric2 = PSUADE_UNDEFINED;
      GMetric2 = PSUADE_UNDEFINED;
      IMetric2 = PSUADE_UNDEFINED;
    }
    else
    {
      //**/ --- accumulate 1/determinant for inverse covariance 
      //**/     (since Fisher^{-1} approximates covariance)
      DMetric2 = (1.0 / matCovInv.computeDeterminant()); 

      //**/ --- now get back the covariance matrix by inversion
      matCovInv.computeInverse(matCov);

      //**/ --- accumulate diagonal terms in covariance matrix
      //**/     (A metric is sum of variances)
      for (kk = 0; kk < nUInps; kk++)
        AMetric2 += matCov.getEntry(kk, kk);

      //**/ --- compute eigenvalues for covariance matrix
      //**/     (E metric is max eigenvalue)
      matCov.eigenSolve(matEig, vecEig, 1);
      ddata = vecEig[0];
      for (kk = 1; kk < nUInps; kk++)
        if (vecEig[kk] > ddata) ddata = vecEig[kk];
      EMetric2 = ddata;

      //**/ --- compute G and I metrics, if needed
      psVector vecCT, vecMCT;
      vecMCT.setLength(nUInps);
      GMetric2 = IMetric2 = 0;
      if (matEvalSet.nrows() > 0)
      {
        //**/ --- first stuff the mean of prior sample into vecXT
        vecXT.setLength(nInps);
        for (jj = 0; jj < nInps; jj++)
        {
          for (ss = 0; ss < priorNR; ss++)
          {
            if (vecIT[jj] >= 1)
            {
              ind = vecIT[jj] - 1;
              vecXT[jj] += matPriorSample.getEntry(ss,ind);
            }
          }
          vecXT[jj] /= (double) priorNR;
        }

        //**/ perform analysis for each evaluation point
        int nGood = 0;
        for (cc = 0; cc < matEvalSet.nrows(); cc++)
        {
          //**/ stuff candidate coordinate in vecXT
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          { 
            if (vecIT[jj] < 1)
            {
              vecXT[jj] = matEvalSet.getEntry(cc,lcnt);
              lcnt++;
            }
          }

          //**/ --- evaluate function value at the mid point
          rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
          vecYT[0] = ddata;

          //**/ compute partial y(theta) /partial theta_j
          vecCT.setLength(nUInps);
          for (jj = 0; jj < nInps; jj++)
          {
            //**/ if parameter is uncertain, perturb and evaluate
            if (vecIT[jj] >= 1)
            {
              ind = vecIT[jj] - 1;
              dtmp = vecXT[jj];
              vecXT[jj] *= (1.0 + 1e-6);
              rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
              //**/ finite difference delta Y_kk wrt theta_jj
              ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
              vecXT[jj] = dtmp;
              vecCT[ind] = ddata;
            }
          }
          //**/ take the mean for row cc in matGrad
          matCov.matvec(vecCT, vecMCT, 0);
          ddata = 0;
          for (jj = 0; jj < nUInps; jj++) 
            ddata += (vecCT[jj] * vecMCT[jj]);
          if (ddata > 0)
          {
            IMetric2 += ddata;
            if (ddata > GMetric2) GMetric2 = ddata;
            nGood++;
          }
        }
        if (nGood == 0)
        {
          if (printLevel_ > 0)
            printf("\nINFO: evaluation set not informative ==> skip.\n");
          IMetric2 = PSUADE_UNDEFINED;
          GMetric2 = PSUADE_UNDEFINED;
        }
        else
        {
          IMetric2 /= (double) nGood;
        }
      }
    }
#else
    //**/ -----------------------------------------------------------
    //**/ this 'else' segment is a different implementation than the
    //**/ 'if' segment, in the sense that this segment computes many
    //**/ M's = [dY(c_1)/d theta ... dY(c_n)/d theta] where M is the 
    //**/ fisher matrix, c_i is candidate i, and Y is simulation 
    //**/ output of interest. For each M, optimality metrics are
    //**/ computed and average over all prior sample points
    //**/ -----------------------------------------------------------
    psVector vecEig, vecCT, vecMCT;
    psMatrix matGrad, matGradT, matCovInv, matEig;
    int priorNR = matPriorSample.nrows(), cc, cand;
    int nCandidates = vecInps.length();
    double DMetric2=0,AMetric2=0,EMetric2=0;
    double GMetric2=0, IMetric2=0;
    vecMCT.setLength(nUInps);

    //**/ the process is repeated for each prior sample
    for (ss = 0; ss < priorNR; ss++)
    {
      //**/ --- first stuff the prior sample into vecXT
      for (jj = 0; jj < nInps; jj++)
      {
        //**/ if parameter is uncertain, use prior sample
        if (vecIT[jj] >= 1)
        {
          ind = vecIT[jj] - 1;
          vecXT[jj] = matPriorSample.getEntry(ss,ind);
        }
      }
      //**/ initialize matGrad
      matGrad.setDim(nUInps, nCandidates);

      //**/ fille matGrad with gradient wrt uncertain
      //**/ parameters for each candidate point
      for (cc = 0; cc < nCandidates; cc++)
      {
        cand = (int) (vecInps[cc]) - 1; 
        //**/ stuff candidate coordinate in vecXT
        //**/ (from candidate matrix)
        lcnt = 0;
        for (jj = 0; jj < nInps; jj++)
        {
          if (vecIT[jj] < 1) /* vecIT[jj] = 0 for design input jj */
          {
            vecXT[jj] = matCandidates.getEntry(cand,lcnt);
            lcnt++;
          } 
        }
        //**/ --- evaluate function and compute derivatives too
        rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
        vecYT[0] = ddata;

        //**/ compute partial y(theta) /partial theta_j
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, perturb and evaluate
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            dtmp = vecXT[jj];
            vecXT[jj] *= (1.0 + 1e-8);
            rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
            //**/ finite difference delta Y wrt theta_jj for 
            //**/ candidate cc
            ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
            vecXT[jj] = dtmp;
            matGrad.setEntry(ind, cc, ddata); 
          }
        }
      } /* cc for nCandidates */

      //**/ --- now compute information matrix (Grad * Grad^T) 
      //**/ --- covariance matrix ~ inv(Grad * Grad^T)
      matGradT = matGrad;
      matGradT.transpose();
      matGrad.matmult(matGradT, matCovInv);
      ddata = matCovInv.computeDeterminant(); 
      if (ddata != 0)
        DMetric2 += (1.0 / matCovInv.computeDeterminant()); 

      //**/ --- now get back the covariance matrix by inversion
      //**/ --- and compute A metric
      matCovInv.computeInverse(matCov);
      for (kk = 0; kk < nUInps; kk++)
        AMetric2 += matCov.getEntry(kk, kk);

      //**/ --- compute EMetric (minimize the maximum
      //**/ --- eigenvalue
      matCov.eigenSolve(matEig, vecEig, 1);
      ddata = vecEig[0];
      for (kk = 1; kk < nUInps; kk++)
        if (vecEig[kk] > ddata) ddata = vecEig[kk];
      EMetric2 += ddata;

      //**/ --- compute G and I metrics, if needed
      if (matEvalSet.nrows() > 0)
      {
        double GMetric2T = 0;
        for (cc = 0; cc < matEvalSet.nrows(); cc++)
        {
          //**/ stuff evaluation coordinate in vecXT
          //**/ uncertain input values already stuffed
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          { 
            if (vecIT[jj] < 1)
            {
              vecXT[jj] = matEvalSet.getEntry(cc,lcnt);
              lcnt++;
            }
          }
          //**/ --- evaluate function and compute derivatives too
          rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
          vecYT[0] = ddata;

          //**/ compute partial y(theta) /partial theta_j for
          //**/ evaluation point cc
          vecCT.setLength(nUInps);
          for (jj = 0; jj < nInps; jj++)
          {
            //**/ if parameter is uncertain, perturb and evaluate
            if (vecIT[jj] >= 1)
            {
              ind = vecIT[jj] - 1;
              dtmp = vecXT[jj];
              vecXT[jj] *= (1.0 + 1e-8);
              rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
              ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
              vecXT[jj] = dtmp;
              vecCT[ind] += ddata;
            }
          }
          //**/compute inner product c^t Cov c
          matCov.matvec(vecCT,vecMCT,0);
          ddata = 0;
          for (jj = 0; jj < nUInps; jj++) 
            ddata += (vecCT[jj] * vecMCT[jj]);
          IMetric2 += ddata;
          if (ddata > GMetric2T) GMetric2T = ddata;
        }
        GMetric2 += GMetric2T;
      }
    }
    IMetric2 /= (double) priorNR;
    GMetric2 /= (double) priorNR;
    DMetric2 /= (double) priorNR;
    AMetric2 /= (double) priorNR;
    EMetric2 /= (double) priorNR;
#endif

#if 0
    //**/ if it is called by odoeu_fevaln, let it
    //**/ display the result (so not to be done here)
    if (nInputsIn == 0)
    {
      printAsterisks(PL_INFO,0);
      printf("  *** Fisher metric summary: \n");
      printDashes(PL_INFO,0);
      printf("  G-metric = %e\n", GMetric2);
      printf("  I-metric = %e\n", IMetric2);
      printf("  D-metric = %e\n", DMetric2);
      printf("  A-metric = %e\n", AMetric2);
      printf("  E-metric = %e\n", EMetric2);
      printAsterisks(PL_INFO,0);
      if (nOutputs >= 1) outputs[0] = GMetric2;
      if (nOutputs >= 2) outputs[1] = IMetric2;
      if (nOutputs >= 3) outputs[2] = DMetric2;
      if (nOutputs >= 4) outputs[3] = AMetric2;
      if (nOutputs >= 5) outputs[4] = EMetric2;
      return 0;
    }
#endif

    //**/ --- return the proper metric
    if (whichLocalFunction_ == 25 && nOutputs == 3) 
    {
       outputs[0] = DMetric2;
       outputs[1] = AMetric2;
       outputs[2] = EMetric2;
    }
    else if (whichLocalFunction_ == 25 && nOutputs == 5) 
    {
       outputs[0] = GMetric2;
       outputs[1] = IMetric2;
       outputs[2] = DMetric2;
       outputs[3] = AMetric2;
       outputs[4] = EMetric2;
    }
    else
    {
      if (whichLocalFunction_ == 25) outputs[0] = GMetric2;
      if (whichLocalFunction_ == 26) outputs[0] = IMetric2;
      if (whichLocalFunction_ == 27) outputs[0] = DMetric2;
      if (whichLocalFunction_ == 28) outputs[0] = AMetric2;
      if (whichLocalFunction_ == 29) outputs[0] = EMetric2;
      if (outputs[0] < optimalVal) optimalVal = outputs[0];
      if (printLevel_ >= 0) 
      {
        printf(" ===> output = %11.4e (best so far = %11.4e",
               outputs[0], optimalVal);
        if (outputs[0] == optimalVal && inputs != NULL) 
        {
          printf(" :");
          for (kk = 0; kk < nInputs; kk++)
            printf(" %d", (int) inputs[kk]);
        }
        printf(")\n");
      }
    }

    //**/ --- update iteration history
    if (nOutputs == 1)
    {
      if (nHist >= maxHist)
      {
        for (jj = 0; jj < maxHist/2; jj++)
          for (ii = 0; ii < nInputs+1; ii++)
            matHistory.setEntry(jj,ii,
                      matHistory.getEntry(jj+maxHist/2,ii));
        nHist = maxHist/2;
      }
      for (ii = 0; ii < nInputs; ii++)
        matHistory.setEntry(nHist,ii,inputs[ii]);
      matHistory.setEntry(nHist,nInputs,outputs[0]);
      nHist++;
    }
    return 0;

#if 0
    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL, create an inputs of all candidates
    //**/ (because when nInputsIn == 0, it is assumed that callers
    //**/ want to use the candidate set instead of from 'inputs' -
    //**/ e.g. called from odoeu_eval)
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0)
    {
      nInputs = matCandidates.nrows();
      vecInps.setLength(nInputs);
      inputs = vecInps.getDVector();
      for (ii = 0; ii < nInputs; ii++) inputs[ii] = ii + 1;
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (duplication selection
    //**/ is not allowed). If so, just return a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = inputs[ii];
        kk  = inputs[jj];
        if (ind == kk) count++;
      }
    }
    if (count > 0)
    {
      if (nInputsIn == 0)
      {
        printf(" ==> ERROR: duplicate selection ");
        for (ii = 0; ii < nOutputs; ii++) outputs[ii] = 0;
        return 0;
      }
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      printf(" ==> duplicate selection ");
      for (ii = 0; ii < nInputs; ii++) printf("%d ", (int) inputs[ii]);
      printf(" ==> skip\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ error checking (whether inputs are valid) and display
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0 && printLevel_ >= 0) 
    {
      if (whichLocalFunction_ == 25)
        printf("GOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 26)
        printf("IOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 27)
        printf("DOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 28)
        printf("AOPTIMAL (Fisher) inputs: ");
      else if (whichLocalFunction_ == 29)
        printf("EOPTIMAL (Fisher) inputs: ");
    }
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) inputs[ii];
      if (ind < 1 || ind > matCandidates.nrows())
      {
        printf("ODOE Function Evaluator ERROR: wrong input values.\n");
        printf("     Check candidate set size consistency.\n");
        printf("The erroneous inputs are:\n");
        for (jj = 0; jj < nInputs; jj++)
          printf("Candidate %3d for evaluation = %d\n", jj+1, 
                 (int) inputs[jj]);
        printf("But they should all be in the range of [1,%d]\n",
               matCandidates.nrows());
        exit(1);
      }
      if (nInputsIn > 0 && printLevel_ >= 0) printf("%5d ", ind);
    }
    if (nInputsIn > 0 && printLevel_ >= 0) printf("\n");

    //**/ -----------------------------------------------------------
    //**/ check history to see whether this has been evaluated before
    //**/ if so, just fetch it from history record
    //**/ -----------------------------------------------------------
    for (jj = 0; jj < nHist; jj++)
    {
      count = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        for (kk = 0; kk < nInputs; kk++)
          if (matHistory.getEntry(jj,kk) == inputs[ii])
            break;
        if (kk != nInputs) count++;
      }
      if (count == nInputs)
      {
        outputs[0] = matHistory.getEntry(jj,nInputs);
        if (printLevel_ > 0) 
          printf(" ===> output = %11.4e (revisit)\n", outputs[0]);
        return 0; 
      }
    }

    //**/ -----------------------------------------------------------
    //**/ for deterministic Fisher-based optimal design
    //**/ NOTE (July 2021: G- and I-metrics are not valid in this 
    //**/                  context
    //**/ -----------------------------------------------------------
    vecXT.setLength(nInps);
    vecYT.setLength(nOuts);
    psVector vecEig;
    psMatrix matGrad, matGradT, matCovInv, matEig;
    nUInps = vecUInputs.length();
    int priorNR = matPriorSample.nrows();
    matGrad.setDim(nUInps, priorNR);
    double DMetric2=0,AMetric2=0,EMetric2=0,GMetric2=0,IMetric2=0;
    double Gmax;

    //**/ G-optimal max of inner products from all prior samples
    //**/ I-optimal sum of inner products from all prior samples
    //**/ D-optimal sum of determinants from all prior samples
    //**/ A-optimal sum of all input variances from all priors 
    //**/ E-optimal max of all input variances for each prior and
    //**/           take average
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ stuff candidate coordinate in vecXT
      //**/ (candidate index in inputs[ii])
      lcnt = 0;
      for (jj = 0; jj < nInps; jj++)
      { 
        //**/ if parameter is uncertain, use prior sample
        if (vecIT[jj] < 1)
        {
          ind = (int) (inputs[ii] - 1 + 1.0e-12);
          vecXT[jj] = matCandidates.getEntry(ind,lcnt);
          lcnt++;
        } 
      }
      //**/ now stuff vecXT with each sample in the prior sample
      //**/ evaluate function and derivatives and add to matGrad
      for (ss = 0; ss < priorNR; ss++)
      {
        //**/ --- first stuff the prior sample into vecXT
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, use prior sample
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            vecXT[jj] = matPriorSample.getEntry(ss,ind);
          }
        }
        //**/ --- evaluate function and compute derivatives too
        rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
        vecYT[0] = ddata;

        //**/ compute partial y(theta, eta_i) /partial theta_j
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, perturb and evaluate
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            dtmp = vecXT[jj];
            vecXT[jj] *= (1.0 + 1e-8);
            rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
            //**/ finite difference delta Y_kk wrt theta_jj
            ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
            if (ddata < 0) ddata = - ddata;
            matGrad.setEntry(ind, ss, ddata); 
            vecXT[jj] = dtmp;
          }
        }
      }
      //**/ --- now compute information matrix (Grad * Grad^T) 
      //**/ --- covariance matrix ~ inv(Grad * Grad^T)
      matGradT = matGrad;
      matGradT.transpose();
      matGrad.matmult(matGradT, matCovInv);
      for (kk = 0; kk < matCovInv.nrows(); kk++)
      {
        for (jj = 0; jj < matCovInv.ncols(); jj++)
        {
          ddata = matCovInv.getEntry(kk,jj);
          ddata /= (double) matPriorSample.nrows();
          matCovInv.setEntry(kk,jj,ddata);
        }
      }
      ddata = matCovInv.computeDeterminant(); 
      if (ddata == 0)
      {
        printf("\nINFO: Fisher determinant = 0 at prior sample %d\n",
               ss+1);
        printf("This means some uncertain parameters are insensitive, ");
        printf("and Fisher-based\n");
        printf("optimality does not work well, or maybe there are ");
        printf("redundant samples\n");
        printf("in the prior sample set.\n");
        printf("Suggestion: Try Bayes-based optimality criteria.\n");
        exit(1);
      }
      //**/ --- accumulate 1/determinant for inverse covariance 
      //**/     (since Fisher^{-1} approximates covariance)
      DMetric2 += (1.0 / matCovInv.computeDeterminant()); 

      //**/ --- now get back the covariance matrix by inversion
      matCovInv.computeInverse(matCov);

      //**/ --- accumulate diagonal terms in covariance matrix
      //**/     (A metric is sum of variances)
      for (kk = 0; kk < nUInps; kk++)
        AMetric2 += matCov.getEntry(kk, kk);

      //**/ --- compute eigenvalues for covariance matrix
      //**/     (E metric is max eigenvalue)
      matCov.eigenSolve(matEig, vecEig, 1);
      ddata = vecEig[0];
      for (kk = 1; kk < nUInps; kk++)
        if (vecEig[kk] > ddata) ddata = vecEig[kk];
      EMetric2 += ddata;

      //**/ compute G and I
      psVector vecDTheta;
      if (whichLocalFunction_ >= 25 && whichLocalFunction_ <= 26)
      {
        //**/ for each evaluation set point kk (prior sample ss): 
        //**/ - create a gradient vector dtheta
        //**/ - form inner product with dtheta F^{-1} dtheta
        Gmax = -PSUADE_UNDEFINED;
        vecXT.setLength(nInps);
        vecYT.setLength(nOuts);

        //**/ evaluate every one in the evaluation set
        for (kk = 0; kk < matEvalSet.nrows(); kk++)
        {
          //**/ fill vecXT with evaluation point
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          {
            //**/ get design parameter values from matCandidates
            if (vecIT[jj] < 1)
            {
              vecXT[jj] = matEvalSet.getEntry(kk,lcnt);
              lcnt++;
            }
          }

          //**/ for each of the prior sample point
          vecDTheta.setLength(nUInps);
          for (ss = 0; ss < matPriorSample.nrows(); ss++)
          {
            //**/ first stuff vecXT with the prior point
            for (jj = 0; jj < nInps; jj++)
            {
              if (vecIT[jj] >= 1)
              {
                ind = vecIT[jj] - 1;
                vecXT[jj] = matPriorSample.getEntry(ss,ind);
              }
            }

            //**/ --- evaluate function
            rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
            vecYT[0] = ddata;

            //**/ compute partial y(theta, eta_i) /partial theta_j
            for (jj = 0; jj < nInps; jj++)
            {
              //**/ if parameter is uncertain, perturb
              if (vecIT[jj] >= 1)
              {
                //**/ get the uncertain parameter index (0-based)
                ind = vecIT[jj] - 1;
                //**/ save the original value
                dtmp = vecXT[jj];
                //**/ perturb
                vecXT[jj] *= (1.0 + 1e-7);
                //**/ evaluate gradient for every output and sum
                rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),
                                         &ddata);
                //**/ finite difference delta Y_kk wrt theta_ind
                ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
                vecDTheta[ind] += ddata;
              }
            }
          }

          //**/ now vecDTheta is ready, compute inner product
          //**/ but first normalize it
          for (jj = 0; jj < vecDTheta.length(); jj++)
            vecDTheta[jj] /= (double) matPriorSample.nrows();

          psVector vecVT;
          matCov.matvec(vecDTheta,vecVT,0);
          ddata = 0.0;
          for (jj = 0; jj < nUInps; jj++)
            ddata += vecDTheta[jj] * vecVT[jj];
          //**/ negative because this quantity is actually 
          //**/ sigma^2 - innerproduct (so max is min)
          if (ddata > Gmax) Gmax = ddata; 
          IMetric2 += ddata;
        }
        GMetric2 += Gmax;
      }
    }
    GMetric2 /= (double) nInputs;
    IMetric2 /= (double) (nInputs * matEvalSet.nrows());
    DMetric2 /= (double) nInputs;
    AMetric2 /= (double) nInputs;
    EMetric2 /= (double) nInputs;

    //**/ --- return the proper metric
    if (whichLocalFunction_ == 25) outputs[0] = GMetric2;
    if (whichLocalFunction_ == 26) outputs[0] = IMetric2;
    if (whichLocalFunction_ == 27) outputs[0] = DMetric2;
    if (whichLocalFunction_ == 28) outputs[0] = AMetric2;
    if (whichLocalFunction_ == 29) outputs[0] = EMetric2;
    if (outputs[0] < optimalVal) optimalVal = outputs[0];
    if (nInputsIn > 0 && printLevel_ >= 0) 
    {
      printf(" ===> output = %11.4e (best so far = %11.4e",
             outputs[0], optimalVal);
      if (outputs[0] == optimalVal) 
      {
        printf(" :");
        for (kk = 0; kk < nInputs; kk++)
          printf(" %d", (int) inputs[kk]);
      }
      printf(")\n");
    }
    //**/ if nInputsIn = 0, it means this function is called
    //**/ from the interpreter to evaluate the metrics (i.e.
    //**/ odoeu_eval), and to expedite things, one call with
    //**/ code 25 will return 2 metrics and 26 with 3.
    if (nInputsIn == 0) 
    {
      if (whichLocalFunction_ == 25 && nOutputs >= 2)
      {
        outputs[0] = GMetric2;
        outputs[1] = IMetric2;
      }
      if (whichLocalFunction_ == 27 && nOutputs >= 3)
      {
        outputs[0] = DMetric2;
        outputs[1] = AMetric2;
        outputs[2] = EMetric2;
      }
      return 0;
    }

    //**/ --- update iteration history
    if (nHist >= maxHist)
    {
      for (jj = 0; jj < maxHist/2; jj++)
        for (ii = 0; ii < nInputs+1; ii++)
          matHistory.setEntry(jj,ii,
                    matHistory.getEntry(jj+maxHist/2,ii));
      nHist = maxHist/2;
    }
    for (ii = 0; ii < nInputs; ii++)
      matHistory.setEntry(nHist,ii,inputs[ii]);
    matHistory.setEntry(nHist,nInputs,outputs[0]);
    nHist++;
    return 0;
#endif
  }

  //**/ =============================================================
  //**/ option: ODOE MMD with SCE optimization
  //**/ =============================================================
  else if (whichLocalFunction_ == 30)
  {
    if (ProblemInitialized == 0)
    {
      if (printLevel_ > 0)
        printf("funcIO: ODOE_MMD Initialization begins ..\n");

      //**/ --- read the candidate set (from CommonUse or from a file
      if (psConfig_.MatCommonUse_.nrows() > 0)
      {
        printf("funcIO psLocalFunction: candidates in common matrix.\n");
        matCandidates = psConfig_.MatCommonUse_;
        nInps = matCandidates.ncols();
      }
      else
      {
        printf("If not, a candidate set file is to be loaded.\n");
        printf("This file should have either the\n");
        printf("(1) PSUADE iread format (iread-compatible) or\n");
        printf("(2) PSUADE data format (load-compatible)\n");
        snprintf(pString,100,
                "Enter the file name of your candidate set : ");
        getString(pString, fname);
        kk = strlen(fname);
        fname[kk-1] = '\0';
        FILE *fp = fopen(fname, "r");
        if (fp == NULL)
        {
          printf("FuncIO ERROR: candidate file not found.\n");
          exit(1);
        }
        fscanf(fp,"%s", pString);  
        fclose(fp);

        //**/ see if it is in PSUADE data format
        if (!strcmp(pString,"PSUADE_IO"))
        {
          //**/ if so, read sample file
          PsuadeData *psIO=NULL;
          psIO = new PsuadeData();
          status = psIO->readPsuadeFile(fname);
          if (status != 0)
          {
            printf("FuncIO ERROR when reading candidate set\n");
            exit(1);
          }
          psIO->getParameter("method_nsamples", pdata);
          int nSamp = pdata.intData_;
          psIO->getParameter("input_ninputs", pdata);
          nInps = pdata.intData_;
          psIO->getParameter("input_sample", pdata);

          //**/ populate matCandidates
          matCandidates.setDim(nSamp,nInps);
          for (ii = 0; ii < nSamp; ii++)
            for (jj = 0; jj < nInps; jj++)
              matCandidates.setEntry(ii,jj,pdata.dbleArray_[ii*nInps+jj]);

          //**/ get weights and put into VecCommonUse, if needed
          int hasWeights=0, outputID=0;
          psIO->getParameter("output_noutputs", pdata);
          nOuts = pdata.intData_;
          pdata.clean();
          psIO->getParameter("output_sample", pdata);
          printf("Select one of the following alternatives: \n");
          printf("(1) Unweighted MMD (no weight given to each sample)\n");
          printf("(2) Weighted MMD (use one of the outputs as weight)\n");
          snprintf(pString,100,"Enter your choice (1 or 2): ");
          hasWeights = getInt(1,2,pString);
          if (hasWeights == 2)
          {
            snprintf(pString,100,
              "Which output to use as weight (1 - %d): ",nOuts);
            outputID = getInt(1, nOuts, pString);
            outputID--;
            for (ii = 0; ii < nSamp; ii++)
              if (pdata.dbleArray_[ii*nOuts+outputID] <= 0.0)
                break;
            if (ii != nSamp)
            {
              printf("ERROR: some of the weights are not <= 0.\n");
              return -1;
            }
            psConfig_.VecCommonUse_.setLength(nSamp);
            for (ii = 0; ii < nSamp; ii++)
              psConfig_.VecCommonUse_[ii] = 
                         pdata.dbleArray_[ii*nOuts+outputID];
          }
          delete psIO;
        }
        else
        //**/ if in iread data format, read it (but no weights allowed)
        {
          status = readIReadDataFile(fname, matCandidates);
          if (status != 0)
          {
            printf("FuncIO ERROR: reading candidate set\n");
            exit(1);
          }
        }
      }
      int nCandidates = matCandidates.nrows();

      //**/ read the must-have candidate points
      nPreSelected = psConfig_.intCommonUse_;

      //**/ compute distances
      matDistances.setDim(nCandidates, nCandidates);
      for (ii = 0; ii < nCandidates; ii++)
      {
        for (jj = ii+1; jj < nCandidates; jj++)
        {
          ddata = 0.0;
          for (kk = 0; kk < matCandidates.ncols(); kk++)
          {
            dtmp = (matCandidates.getEntry(ii,kk) - 
                    matCandidates.getEntry(jj,kk));
            ddata += dtmp * dtmp;
          }
          if (psConfig_.VecCommonUse_.length() == nCandidates)
            ddata *= (psConfig_.VecCommonUse_[ii] *
                      psConfig_.VecCommonUse_[jj]);
          ddata = sqrt(ddata);
          matDistances.setEntry(ii, jj, ddata);
          matDistances.setEntry(jj, ii, ddata);
        }
      }
      ProblemInitialized = 1;
      if (printLevel_ > 0)
        printf("funcIO: ODOE_MMD Initialization complete.\n");
    }

    //**/ -----------------------------------------------------------
    //**/ check that the inputs are within ranges
    //**/ -----------------------------------------------------------
    if (printLevel_ > 1) printf("ODOE_MMD Function inputs: ");
    int inpCheck=0;
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) inputs[ii];
      if (ind < 1) ind = 1;
      if (ind > matCandidates.nrows())
        ind = matCandidates.nrows();
      if (printLevel_ > 1) printf("%6d ", ind);
      if (ind < 1 || ind > matCandidates.nrows()-nPreSelected)
      {
        printf("input %d = %d out of range [1,%d]\n",ii+1,ind,
               matCandidates.nrows()-nPreSelected);
        outputs[0] = PSUADE_UNDEFINED;
        return 0;
      }
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (duplication selection
    //**/ is not allowed). If so, just return a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = (int) vecInps[ii];
        kk  = (int) vecInps[jj];
        if (ind == kk) count++;
      }
    }
    if (count > 0) 
    {
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      //printf(" ==> duplicate selection (not a user concern) : ");
      //for (ii = 0; ii < nInputs; ii++) printf("%d ",(int) vecInps[ii]);
      //printf(" ==> skip\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ find minimum distances between selected points
    //**/ -----------------------------------------------------------
    double minDist = PSUADE_UNDEFINED;
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ find among candidates
      ind = int(inputs[ii]) - 1;
      if (ind < 0) ind = 0;
      for (jj = ii+1; jj < nInputs; jj++)
      {
        kk  = int(inputs[jj]) - 1;
        if (kk < 0) kk = 0;
        if (matDistances.getEntry(ind,kk) < minDist)
          minDist = matDistances.getEntry(ind,kk);
      }
      //**/ find with respect to pre-selected points
      for (jj = matCandidates.nrows()-nPreSelected; 
           jj < matCandidates.nrows(); jj++)
      {
        if (matDistances.getEntry(ind,jj) < minDist)
          minDist = matDistances.getEntry(ind,jj);
      }
    }
    outputs[0] = -minDist;
    if (printLevel_ > 1) printf(" ===> output = %e\n", minDist);
  }

  //**/ =============================================================
  //**/ option: ODOE MMV/MAV with SCE optimization
  //**/ =============================================================
  else if (whichLocalFunction_ == 31 || whichLocalFunction_ == 32)
  {
    if (ProblemInitialized == 0)
    {
      if (printLevel_ > 0)
      {
        if (whichLocalFunction_ == 31)
          printf("ODOE_MMV Initialization begins ..\n");
        else
          printf("ODOE_MAV Initialization begins ..\n");
      }
      //**/ --- read the candidate set (from CommonUse or from a file
      if (psConfig_.MatCommonUse_.nrows() > 0)
      {
        printf("funcIO psLocalFunction: candidates in common matrix.\n");
        matCandidates = psConfig_.MatCommonUse_;
        nInps = matCandidates.ncols();
      }
      else
      {
        printf("FuncIO ODOE_MM/AV ERROR: No candidate matrix found.\n");
        exit(1);
      }

      //**/ ask for a training sample for GP 
      printf("To run MMV/MAV, PSUADE needs to build a Gaussian ");
      printf("Process either from a\n");
      printf("training sample or from a vector of user-provided ");
      printf("hyperparameters.\n");
      snprintf(pString,100,"Use a training set ? (y or n) ");
      getString(pString, lineIn);
      if (lineIn[0] == 'y')
      {
        printf("This training sample must be in PSUADE format.\n");
        snprintf(pString,100,"Name of the training sample ? ");
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        PsuadeData *psuadeIO = new PsuadeData();
        status = psuadeIO->readPsuadeFile(fname);
        if (status != 0)
        {
          printf("FuncIO ERROR when reading training sample\n");
          printf("       Maybe this file is in wrong format?\n");
          exit(1);
        }
        psuadeIO->getParameter("input_ninputs", pdata);
        nInps = pdata.intData_;
        psuadeIO->getParameter("output_noutputs", pdata);
        nOuts = pdata.intData_;
        if (nOuts > 1)
        {
          printf("FuncIO ERROR: this method only works with nOutputs=1\n");
          printf("Suggestion: delete all but one output and re-run.\n");
          exit(1);
        }
        pData pInps, pOuts, pLBs, pUBs, pStas;
        psuadeIO->getParameter("method_nsamples", pdata);
        int nSamp = pdata.intData_;
        psuadeIO->getParameter("input_lbounds", pLBs);
        psuadeIO->getParameter("input_ubounds", pUBs);
        psuadeIO->getParameter("input_sample", pInps);
        psuadeIO->getParameter("output_sample", pOuts);
        psuadeIO->getParameter("output_states", pStas);
        gpMVSampler = new GPBasedSampling();
        gpMVSampler->setInputBounds(nInps, pLBs.dbleArray_,
                            pUBs.dbleArray_);
        gpMVSampler->setOutputParams(1);
        gpMVSampler->setSamplingParams(nSamp, -1, -1);
        gpMVSampler->loadSamples(nSamp,nInps,nOuts,pInps.dbleArray_,
                           pOuts.dbleArray_, pStas.intArray_);
        if (whichLocalFunction_ == 31) gpMVSampler->setMMV();
        else                           gpMVSampler->setMAV();
        gpMVSampler->initialize(1);
      }
      else
      {
        //**/ instantiate GP (which will read the training sample)
        gpMVSampler = new GPBasedSampling();
        if (whichLocalFunction_ == 31) gpMVSampler->setMMV();
        else                           gpMVSampler->setMAV();
        gpMVSampler->initialize(1);
      }

      ProblemInitialized = 1;
      if (printLevel_ > 0)
      {
        if (whichLocalFunction_ == 31)
          printf("funcIO ODOE_MMV Initialization complete.\n");
        else
          printf("funcIO ODOE_MAV Initialization complete.\n");
      }
    }

    //**/ -----------------------------------------------------------
    //**/ check that the inputs are within ranges
    //**/ -----------------------------------------------------------
    if (printLevel_ > 1) 
    {
      if (whichLocalFunction_ == 31)
        printf("ODOE_MMV Function inputs: ");
      else
        printf("ODOE_MAV Function inputs: ");
    }
    int inpCheck=0;
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) inputs[ii];
      if (ind < 1) ind = 1;
      if (ind > matCandidates.nrows())
        ind = matCandidates.nrows();
      if (printLevel_ > 1) printf("%6d ", ind);
      if (ind < 1 || ind > matCandidates.nrows())
      {
        if (whichLocalFunction_ == 31)
          printf("MMV input %d = %d out of range [1,%d]\n",ii+1,
                 ind, matCandidates.nrows());
        else
          printf("MAV input %d = %d out of range [1,%d]\n",ii+1,
                 ind, matCandidates.nrows());
        outputs[0] = PSUADE_UNDEFINED;
        return 0;
      }
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (duplication selection
    //**/ is not allowed). If so, just return a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = (int) inputs[ii];
        kk  = (int) inputs[jj];
        if (ind == kk) count++;
      }
    }
    if (count > 0) 
    {
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      //printf(" ==> duplicate selection (not a user concern) : ");
      //for (ii = 0; ii < nInputs; ii++) printf("%d ",(int) vecInps[ii]);
      //printf(" ==> skip\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ extract selection matrix
    //**/ -----------------------------------------------------------
    psMatrix samMatrix;
    samMatrix.setDim(nInputs,nInps);
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = int(inputs[ii]) - 1;
      if (ind < 0) ind = 0;
      for (jj = 0; jj < nInps; jj++)
        samMatrix.setEntry(ii,jj,matCandidates.getEntry(ind,jj));
    }
    outputs[0] = gpMVSampler->evaluateCandidateSet(matCandidates,
                                                   samMatrix);
    if (printLevel_ > 1) printf(" ===> output = %e\n", outputs[0]);
  }

  //**/ =============================================================
  //**/ option: ODOE G/I/A/D/E-optimal with SCE optimization
  //**/ (option 45-49, as opposed to LocalFunction 25-29, uses Fisher 
  //**/ with simulator provided by user instead of response surface)
  //**/ -------------------------------------------------------------
  //**/ Fisher 45: G; 46: I; 47: A; 48: D; 49: E 
  //**/ =============================================================
  else if (whichLocalFunction_ >= 45 && whichLocalFunction_ <= 49)
  {
    //**/ -----------------------------------------------------------
    //**/ error checking (the objective function must have size 1
    //**/ even though the simulator has multiple outputs
    //**/ But if nInputsIn == 0, it is asking for multiple metric
    //**/ evaluation so it is acceptable to have nOutputs > 1
    //**/ -----------------------------------------------------------
    if (nOutputs != 1 && nInputsIn > 0)
    {
      printf("FuncIO ERROR: nOutputs has to be = 1\n");
      exit(1);
    }
    psVector vecXT, vecYT;
    int nUInps, nSamFisher;

    //**/ -----------------------------------------------------------
    //**/ initialization 
    //**/ -----------------------------------------------------------
    if (ProblemInitialized == 0)
    {
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (Fisher-Jacobi): Initialization begins ..\n");

      PsuadeData *psuadeIO=NULL;
      if (whichLocalFunction_ == 45 || whichLocalFunction_ == 46)
      {
        //**/ --- option to use a response surface for evaluation ---
        printf("This method uses an actual simulator to compute the ");
        printf("simulation output\n");
        printf("(1 output only) as well as its derivatives with ");
        printf("respect to each input\n");
        printf("(for the uncertain inputs only).\n");
        printf("As for evaluating the G- and I-metric using an ");
        printf("evaluation set, you\n");
        printf("have the option to use a response surface built ");
        printf("from a training set\n");
        printf("(no derivative needed, simulation nOutput=1 only).\n");
        snprintf(pString,100,
          "Build response surface from training set ? (y or n) ");
        getString(pString, lineIn);
        if (lineIn[0] == 'y')
        {
          printf("You are now requested to provide a training sample.\n");
          printf("This training sample must be in PSUADE format.\n");
          printf("ALSO, THIS TRAINING SAMPLE MUST HAVE ITS DRIVER ");
          printf("FIELD POINTING TO A\n");
          printf("SIMULATION EXECUTABLE, AND THIS SIMULATOR HAS ");
          printf("TO COMPUTE 1 OUTPUT PLUS\n");
          printf("ITS DERIVATIVES WITH RESPECT TO THE INCERTAIN INPUTS.\n");
          snprintf(pString,100,
                 "Name of the training sample (in PSUADE format) ? ");
          getString(pString, fname);
          fname[strlen(fname)-1] = '\0';
          psuadeIO = new PsuadeData();
          status = psuadeIO->readPsuadeFile(fname);
          if (status != 0)
          {
            printf("FuncIO ERROR when reading training sample\n");
            printf("       Maybe this file is in wrong format?\n");
            exit(1);
          }
          psuadeIO->getParameter("input_ninputs", pdata);
          nInps = pdata.intData_;
          psuadeIO->getParameter("output_noutputs", pdata);
          nOuts = pdata.intData_;
          if (nOuts > 1)
          {
            printf("FuncIO ERROR: sample nOutputs = 1 only\n");
            exit(1);
          }
          //**/ --- use training sample to construct response surface
          //**/ --- psuadeIO ==> rsPtrs
          pData pInps, pOuts, pLBs, pUBs;
          psuadeIO->getParameter("method_nsamples", pdata);
          int nSamp = pdata.intData_;
          psuadeIO->getParameter("input_lbounds", pLBs);
          psuadeIO->getParameter("input_ubounds", pUBs);
          psuadeIO->getParameter("input_sample", pInps);
          psuadeIO->getParameter("output_sample", pOuts);
          int faFlag = 1, rsMethod;
          vecYT.setLength(nSamp);
          printf("FuncIO: constructing response surfaces\n");
          rsPtrs = new FuncApprox*[1];
          rsPtrs[0] = genFAInteractive(psuadeIO, faFlag);
          rsPtrs[0]->setBounds(pLBs.dbleArray_,pUBs.dbleArray_);
          rsPtrs[0]->setOutputLevel(0);
          for (jj = 0; jj < nSamp; jj++)
            vecYT[jj] = pOuts.dbleArray_[jj];
          status = rsPtrs[0]->initialize(pInps.dbleArray_,
                                         vecYT.getDVector());
        }
        else
        {
          printf("INFO: The evaluation set will be ");
          printf("evaluated by the simulator.\n");
        }
      }
      if (psuadeIO == NULL)
      {
        //**/ --- read information from psuade input file 
        printf("Please provide an PSUADE input file for ");
        printf("extracting user information:\n"); 
        printf("1. input dimension and input bounds\n");
        printf("2. simulation driver name (make sure 'driver' is set)\n");
        snprintf(pString,100,"Enter PSUADE input file name : ");
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        psuadeIO = new PsuadeData();
        status = psuadeIO->readPsuadeFile(fname);
        if (status != 0)
        {
          printf("FuncIO ERROR: when reading PSUADE input file.\n");
          printf("       Maybe this file is in wrong format?\n");
          exit(1);
        }
        psuadeIO->getParameter("input_ninputs", pdata);
        nInps = pdata.intData_;
        psuadeIO->getParameter("output_noutputs", pdata);
        nOuts = pdata.intData_;
        if (nOuts != nInps+1)
        {
          printf("FuncIO ERROR: derivative-based Fisher methods must have ");
          printf("nOuts=nInps+1.\n"); 
          exit(1);
        }
        rsPtrs = NULL;
      }

      //**/ --- user selects uncertain inputs ==> vecUInputs
      vecUInputs.setLength(nInps);
      snprintf(pString,100,
         "Enter uncertain input number (1 - %d, 0 to end) : ",nInps);
      ii = 0;
      while (1)
      {
        kk = getInt(0, nInps, pString);
        if (kk == 0 || kk > nInps) break;
        vecUInputs[ii] = kk - 1;
        ii++;
      }
      vecUInputs.subvector(0, ii-1);
      nUInps = vecUInputs.length();

      //**/ --- set uncertain parameter input to 1 in vecIT
      //**/ --- e.g. vecIT[kk]=1 if input kk is uncertain
      vecIT.setLength(nInps);
      kk = 1;
      for (ii = 0; ii < vecUInputs.length(); ii++)
      {
        vecIT[vecUInputs[ii]] = kk;
        kk++; 
      }

      //**/ --- read prior sample ==> matPriorSample
      printf("Uncertain parameters need a prior sample for inference.\n");
      printf("This file should have the following format:\n");
      printf("Line 1: <nSamples> <nInputs>\n");
      printf("Line 2: 1 <sample 1 values>\n");
      printf("Line 3: 2 <sample 2 values>\n");
      printf("Line 4: ...\n");
      snprintf(pString,100,"Enter the file name of your prior sample : ");
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      status = readIReadDataFile(fname, matPriorSample);
      if (status != 0)
      {
        printf("FuncIO ERROR: when reading prior sample\n");
        FILE *fchk = fopen(fname, "r");
        if (fchk == NULL)
             printf("       Prior sample file does not exist.\n");
        else 
        {
          fclose(fchk);
          printf("       Maybe this file is in wrong format?\n");
        }
        exit(1);
      }
      if (matPriorSample.nrows() < nInputs)
      {
        printf("FuncIO ERROR: prior sample size is too small.\n");
        printf("       Minimum sample size is %d.\n",nInputs);
        exit(1);
      }
      if (matPriorSample.ncols() != vecUInputs.length())
      {
        printf("FuncIO ERROR: prior sample nInputs is not correct.\n");
        printf("       Should be equal to %d.\n",vecUInputs.length());
        exit(1);
      }

      //**/ option to use smaller prior sample (this will not be 
      //**/ asked if this is called by odoeu_eval)
      if (nInputsIn > 0 && (matPriorSample.nrows() > nUInps))
      {
        printf("Fisher-based methods are computationally expensive, so ");
        printf("you may want\n");
        printf("to reduce the cost by collapsing the prior sample into ");
        printf("fewer sample\n");
        printf("points (has to be larger than the number of ");
        printf("uncertain parameters.)\n");
        snprintf(pString,100,
           "Collapse prior sample into smaller sample ? (y or n) \n"); 
        getString(pString, lineIn);
        if (lineIn[0] == 'y')
        {
          printf("The size of the prior sample is %d.\n",
                 matPriorSample.nrows());
          snprintf(pString,100,
            "Use (1) the full sample or (2) a random sub-sample ? ");
          kk = getInt(1, 2, pString);
          if (kk == 2)
          {
            snprintf(pString,100,"Enter sub-sample size (%d - %d) : ",
                    nInputs, matPriorSample.nrows()/2+1);
            nSamFisher = getInt(nInputs,matPriorSample.nrows()/2+1,pString);
            
            int nr = matPriorSample.nrows(), nc = matPriorSample.ncols();
            psMatrix tmpMat = matPriorSample;
            matPriorSample.setFormat(PS_MAT2D);
            matPriorSample.setDim(nSamFisher, nc);
            int nCurrent = 0, nTrials=0, checkFlag=1;
            while (nCurrent < nSamFisher)
            {
              nTrials++;
              if (nTrials > 20*nr)
              {
                printf("INFO: ERROR when drawing a unique sub-sample.\n");
                printf("      Stop checking for a unique sub-sample.\n");
                checkFlag = 0;
              }
              kk = PSUADE_rand() % nr;
              //**/ check that there are no duplicates
              jj = 0;
              if (checkFlag == 1)
              {
                for (ii = 0; ii < nCurrent-1; ii++)
                {  
                  for (jj = 0; jj < nc; jj++)
                  {
                    ddata = matPriorSample.getEntry(ii,jj);
                    dtmp  = tmpMat.getEntry(kk,jj);
                    if (dtmp != ddata) break;
                  }
                  //**/ give up if the same sample point
                  if (jj == nc) break;
                }
              }
              if (jj != nc)
              {
                for (jj = 0; jj < nc; jj++)
                {
                  ddata = tmpMat.getEntry(kk,jj);
                  matPriorSample.setEntry(nCurrent,jj,ddata);
                }
                nCurrent++;
              }
            }
          }
          else nSamFisher = matPriorSample.nrows(); 
        }
      }

      //**/ --- read candidate set ==> matCandidates
      //**/ --- If no inputs is given, it means the calling function
      //**/ --- is requesting using selected designs for analysis
      if (nInputs == 0)
      {
        printf("Next provide the file name of your selected designs.\n");
        printf("The file should be in the following format:\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <design 1 values>\n");
        printf("Line 3: <design 2 values>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,
          "Enter the file name of your selected candidatesubset : ");
      }
      else
      {
        printf("Next provide the file name of your candidate design set.\n");
        printf("The file should be in the following format:\n");
        printf("Line 1: <nSamples> <nInputs>\n");
        printf("Line 2: <design 1 values>\n");
        printf("Line 3: <design 2 values>\n");
        printf("Line 4: ...\n");
        snprintf(pString,100,
                 "Enter the file name of your candidate design set : ");
      }
      getString(pString, fname);
      fname[strlen(fname)-1] = '\0';
      status = readIReadDataFile(fname, matCandidates);
      if (status != 0)
      {
        printf("FuncIO ERROR: When reading candidate or selected set\n");
        printf("       Maybe this file is in wrong format?\n");
        exit(1);
      }
      int nCandidates = matCandidates.nrows();
      printf("Size of Candidate set = %d\n", nCandidates);

      //**/ --- read evaluation set (for G and I) ==> matEvalSet
      if (whichLocalFunction_ == 45 || whichLocalFunction_ == 46)
      {
        printf("An evaluation sample is needed to ");
        printf("compute the optimality metrics.\n");
        printf("This can be the same as the candidate set ");
        printf("(but not recommended).\n");
        printf("The file should be in the following format: \n");
        printf("Line 1: <numPoints> <nInputs>\n");
        printf("Line 2: 1 <sample 1 values> \n");
        printf("Line 3: 2 <sample 2 values> \n");
        printf("....\n");
        snprintf(pString,100,
                "Enter the file name of your evaluation set : ");
        getString(pString, fname);
        fname[strlen(fname)-1] = '\0';
        status = readIReadDataFile(fname, matEvalSet);
        if (status != 0)
        {
          printf("FuncIO ERROR: when reading evaluation set\n");
          printf("       Maybe this file is in wrong format?\n");
          exit(1);
        }
        if (matEvalSet.ncols() != nInps-nUInps)
        {
          printf("FuncIO ERROR: evaluation file must have %d columns.\n",
                 nInps-nUInps);
          exit(1);
        }
        if (matPriorSample.nrows()*matEvalSet.nrows() > 10000)
          printf("WARNING: Prior and/or evaluation sample too large ==> time consuming.\n");
      }
      //**/ --- get input bounds and driver --- 
      pData pAppFiles;
      psuadeIO->getParameter("app_files", pAppFiles);
      if (!strcmp(pAppFiles.strArray_[0], "NONE"))
      {
        printf("FuncIO ERROR: simulator driver is not defined.\n");
        exit(1);
      } 
      FisherFuncIO = new FunctionInterface();
      FisherFuncIO->loadInputData(nInps, NULL);
      FisherFuncIO->loadOutputData(nInps+1, NULL);
      FisherFuncIO->loadFunctionData(pAppFiles.nStrings_,
                                     pAppFiles.strArray_);
  
      //**/ --- the rest of the initialization
      delete psuadeIO;
      ProblemInitialized = 1;
      matHistory.setDim(maxHist,nInputs+1);
      nHist = 0;
      optimalVal = PSUADE_UNDEFINED;
      if (printLevel_ > 0)
        printf("ODOE_(X)OPTIMAL (Fisher-Jacobi): Initialization completed.\n");
    }

    //**/ -----------------------------------------------------------
    //**/ if inputs = NULL, create an inputs of all candidates
    //**/ (because when nInputsIn == 0, it is assumed that callers
    //**/ are providing selection via inputs - that is, calling from
    //**/ odoeu_eval)
    //**/ -----------------------------------------------------------
    if (nInputsIn == 0)
    {
      nInputs = matCandidates.nrows();
      vecInps.setLength(nInputs);
      inputs = vecInps.getDVector();
      for (ii = 0; ii < nInputs; ii++) inputs[ii] = ii + 1;
    }

    //**/ -----------------------------------------------------------
    //**/ check to see if there are duplicates (duplication selection
    //**/ is not allowed). If so, just return a large value
    //**/ -----------------------------------------------------------
    count = 0;
    for (ii = 0; ii < nInputs; ii++)
    {
      for (jj = ii+1; jj < nInputs; jj++)
      { 
        ind = inputs[ii];
        kk  = inputs[jj];
        if (ind == kk) count++;
      }
    }
    if (count > 0)
    {
      if (nInputsIn == 0)
      {
        printf(" ==> ERROR: duplicate selection ");
        for (ii = 0; ii < nOutputs; ii++) outputs[ii] = 0;
        return 0;
      }
      if (optimalVal < 0) outputs[0] = 0.1 * optimalVal;
      else                outputs[0] = 10.0 * optimalVal;
      printf(" ==> duplicate selection ");
      for (ii = 0; ii < nInputs; ii++) printf("%d ", (int) inputs[ii]);
      printf(" ==> skip\n");
      return 0;
    }

    //**/ -----------------------------------------------------------
    //**/ error checking (whether inputs are valid) and display
    //**/ -----------------------------------------------------------
    if (nInputsIn > 0 && printLevel_ >= 0) 
    {
      if (whichLocalFunction_ == 45)
        printf("GOPTIMAL (Fisher-Jacobi) inputs: ");
      else if (whichLocalFunction_ == 46)
        printf("IOPTIMAL (Fisher-Jacobi) inputs: ");
      else if (whichLocalFunction_ == 47)
        printf("DOPTIMAL (Fisher-Jacobi) inputs: ");
      else if (whichLocalFunction_ == 48)
        printf("AOPTIMAL (Fisher-Jacobi) inputs: ");
      else if (whichLocalFunction_ == 49)
        printf("EOPTIMAL (Fisher-Jacobi) inputs: ");
    }
    for (ii = 0; ii < nInputs; ii++)
    {
      ind = (int) inputs[ii];
      if (ind < 1 || ind > matCandidates.nrows())
      {
        printf("ODOE Function Evaluator ERROR: wrong input values.\n");
        printf("     Check candidate set size consistency.\n");
        printf("The erroneous inputs are:\n");
        for (jj = 0; jj < nInputs; jj++)
          printf("Candidate %3d for evaluation = %d\n", jj+1, 
                 (int) inputs[jj]);
        printf("But they should all be in the range of [1,%d]\n",
               matCandidates.nrows());
        exit(1);
      }
      if (nInputsIn > 0 && printLevel_ >= 0) printf("%5d ", ind);
    }
    if (nInputsIn > 0 && printLevel_ >= 0) printf("\n");

    //**/ -----------------------------------------------------------
    //**/ check history to see whether this has been evaluated before
    //**/ if so, just fetch it from history record
    //**/ -----------------------------------------------------------
    for (jj = 0; jj < nHist; jj++)
    {
      count = 0;
      for (ii = 0; ii < nInputs; ii++)
      {
        for (kk = 0; kk < nInputs; kk++)
          if (matHistory.getEntry(jj,kk) == inputs[ii])
            break;
        if (kk != nInputs) count++;
      }
      if (count == nInputs)
      {
        outputs[0] = matHistory.getEntry(jj,nInputs);
        if (printLevel_ > 0) 
          printf(" ===> output = %11.4e (revisit)\n", outputs[0]);
        return 0; 
      }
    }

    //**/ -----------------------------------------------------------
    //**/ for deterministic Fisher-based optimal design
    //**/ -----------------------------------------------------------
    vecXT.setLength(nInps);
    vecYT.setLength(nInps+1);
    psVector vecEig;
    psMatrix matGrad, matGradT, matCovInv, matEig;
    nUInps = vecUInputs.length();
    int iOne=1, priorNR = matPriorSample.nrows();
    matGrad.setDim(nUInps, priorNR);
    double DMetric2=0,AMetric2=0,EMetric2=0,GMetric2=0,IMetric2=0;
    double Gmax;

    //**/ G-optimal max of inner products from all prior samples
    //**/ I-optimal sum of inner products from all prior samples
    //**/ D-optimal sum of determinants from all prior samples
    //**/ A-optimal sum of all input variances from all priors 
    //**/ E-optimal max of all input variances for each prior and
    //**/           take average
    for (ii = 0; ii < nInputs; ii++)
    {
      //**/ stuff candidate coordinate in vecXT
      //**/ (candidate index in inputs[ii])  
      lcnt = 0;
      for (jj = 0; jj < nInps; jj++)
      {
        //**/ if parameter is uncertain, use prior sample
        if (vecIT[jj] < 1)
        {
          ind = (int) (inputs[ii] - 1 + 1.0e-12);
          vecXT[jj] = matCandidates.getEntry(ind,lcnt);
          lcnt++;
        }
      }

      //**/ now for each sample in the prior sample
      //**/ evaluate function and derivatives and add to matGrad
      for (ss = 0; ss < matPriorSample.nrows(); ss++)
      {
        //**/ --- first stuff the prior sample into vecXT
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, use prior sample
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            vecXT[jj] = matPriorSample.getEntry(ss,ind);
          }
        }
        //**/ --- evaluate function
        //**/ --- which should provide derivatives too
        //**/ --- so nOuts must be = nInps+1
        int doSim = 0;
        if (matFisherSimStore.nrows() == 0)
        {
          matFisherSimStore.setDim(matCandidates.nrows()*priorNR,nInps+1);
          doSim = 1;
        }
        ddata = 0;
        ind = (int) (inputs[ii] - 1 + 1.0e-12);
        for (jj = 0; jj < nInps+1; jj++)
          ddata += matFisherSimStore.getEntry(ind*priorNR+ss,jj);
        if (ddata == 0) doSim = 1;

        //**/ if RS is not to be used, use simulator directly
        if (doSim == 1)
        {
          if (whichLocalFunction_ >= 45 && whichLocalFunction_ <= 46)
            kk = ii*matPriorSample.nrows()*(matEvalSet.nrows()+1)+ss+1;
          else
            kk = ii*matPriorSample.nrows() + ss + 1;
          FisherFuncIO->evaluate(kk,nInps,vecXT.getDVector(),
                                 nInps+1, vecYT.getDVector(), 0);
          for (jj = 0; jj < nInps+1; jj++)
            matFisherSimStore.setEntry(ind*priorNR+ss,jj,vecYT[jj]); 
        }
        else
        {
          //**/ otherwise, use RS pre-evaluated data
          for (jj = 0; jj < nInps+1; jj++)
            vecYT[jj] = matFisherSimStore.getEntry(ind*priorNR+ss,jj); 
        }

        //**/ put derivative of uncertain inputs to matGrad
        for (jj = 0; jj < nInps; jj++)
        {
          //**/ if parameter is uncertain, use simulator derivatives
          if (vecIT[jj] >= 1)
          {
            ind = vecIT[jj] - 1;
            matGrad.setEntry(ind, ss, vecYT[jj+1]); 
          }
        }
      }

      //**/ --- now compute information matrix (Grad * Grad^T) 
      //**/ --- covariance matrix ~ inv(Grad * Grad^T)
      matGradT = matGrad;
      matGradT.transpose();
      matGrad.matmult(matGradT, matCovInv);
      for (kk = 0; kk < matCovInv.nrows(); kk++)
      {
        for (jj = 0; jj < matCovInv.ncols(); jj++)
        {
          ddata = matCovInv.getEntry(kk,jj);
          ddata /= (double) matPriorSample.nrows(); 
          matCovInv.setEntry(kk,jj,ddata);
        }
      }
      ddata = matCovInv.computeDeterminant(); 
      if (ddata == 0)
      {
        printf("\nINFO: Fisher determinant = 0 at prior sample %d\n",
               ss+1);
        printf("This means some uncertain parameters are insensitive,\n");
        printf("and Fisher-based optimality does not work well.\n");
        printf("Suggestion: Try Bayes-based optimality criteria.\n");
        exit(1);
      }
      //**/ --- accumulate 1/determinant for inverse covariance 
      //**/     (since Fisher^{-1} approximates covariance)
      DMetric2 += (1.0 / matCovInv.computeDeterminant()); 

      //**/ --- now get back the covariance matrix by inversion
      //**/ --- for computing A-metric
      matCovInv.computeInverse(matCov);

      //**/ --- accumulate diagonal terms in covariance matrix
      //**/     (A metric is sum of variances)
      for (jj = 0; jj < nUInps; jj++)
        AMetric2 += matCov.getEntry(jj, jj);

      //**/ --- compute eigenvalues for covariance matrix
      //**/     (E metric is max eigenvalue)
      matCov.eigenSolve(matEig, vecEig, 1);
      ddata = vecEig[0];
      for (jj = 1; jj < nUInps; jj++)
        if (vecEig[jj] > ddata) ddata = vecEig[jj];
      EMetric2 += ddata;

      //**/ compute G and I
      psVector vecDTheta;
      if (whichLocalFunction_ >= 45 && whichLocalFunction_ <= 46)
      {
        //**/ for each evaluation set point: 
        //**/ - create a gradient vector dtheta
        //**/ - form inner product with dtheta C^{-1} dtheta
        Gmax = -PSUADE_UNDEFINED;
        vecXT.setLength(nInps);
        vecYT.setLength(nInps+1);

        //**/ evaluate every one in the evaluation set
        for (kk = 0; kk < matEvalSet.nrows(); kk++)
        {
          //**/ fill vecXT with evaluation point 
          lcnt = 0;
          for (jj = 0; jj < nInps; jj++)
          {
            //**/ get design parameter values from matCandidates
            if (vecIT[jj] < 1)
            {
              vecXT[jj] = matEvalSet.getEntry(kk,lcnt);
              lcnt++;
            }
          }

          //**/ for each of the prior sample point
          vecDTheta.setLength(nUInps);
          for (ss = 0; ss < matPriorSample.nrows(); ss++)
          {
            //**/ first stuff vecXT with the prior point
            for (jj = 0; jj < nInps; jj++)
            {
              if (vecIT[jj] >= 1)
              {
                ind = vecIT[jj] - 1;
                vecXT[jj] = matPriorSample.getEntry(ss,ind);
              }
            }

            //**/ --- evaluate function
            int doSim = 0;
            if (matFisherEvalStore.nrows() == 0)
            {
              matFisherEvalStore.setDim(matEvalSet.nrows()*priorNR,nInps+1);
              doSim = 1;
            }
            ddata = 0;
            for (jj = 0; jj < nInps+1; jj++)
              ddata += matFisherEvalStore.getEntry(kk*priorNR+ss,jj);
            if (ddata == 0) doSim = 1;

            if (doSim == 1 && rsPtrs == NULL)
            {
              jj = (ii*(matEvalSet.nrows()+1)+kk+1)*matPriorSample.nrows()+ss+1;
              FisherFuncIO->evaluate(jj,nInps,vecXT.getDVector(),
                                     nInps+1, vecYT.getDVector(), 0);
              for (jj = 0; jj < nInps+1; jj++)
                 matFisherEvalStore.setEntry(kk*priorNR+ss,jj,vecYT[jj]);
            }
            else if (doSim == 0)
            {
              for (jj = 0; jj < nInps+1; jj++)
                 vecYT[jj] = matFisherEvalStore.getEntry(kk*priorNR+ss,jj);
            }
            else
            {
              rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),&ddata);
              vecYT[0] = ddata;
              for (jj = 0; jj < nInps; jj++)
              {
                //**/ if parameter is uncertain, perturb and evaluate
                if (vecIT[jj] >= 1)
                {
                  ind = vecIT[jj] - 1;
                  dtmp = vecXT[jj];
                  vecXT[jj] *= (1.0 + 1e-8);
                  rsPtrs[0]->evaluatePoint(iOne,vecXT.getDVector(),
                                           &ddata);
                  //**/ finite difference delta Y_kk wrt theta_jj
                  ddata = (ddata - vecYT[0]) / (vecXT[jj] - dtmp);
                  if (ddata < 0) ddata = - ddata;
                  vecXT[jj] = dtmp;
                  vecYT[jj+1] = ddata;
                }
              }
              for (jj = 0; jj < nInps+1; jj++)
                matFisherEvalStore.setEntry(kk*priorNR+ss,jj,vecYT[jj]);
            }

            //**/ add partial y(theta, eta_i) /partial theta_j
            for (jj = 0; jj < nInps; jj++)
            {
              //**/ if parameter is uncertain, perturb
              if (vecIT[jj] >= 1)
              {
                //**/ get the uncertain parameter index (0-based)
                ind = vecIT[jj] - 1;
                vecDTheta[ind] += vecYT[jj+1];
              }
            }
          } /* ss */ 

          //**/ now vecDTheta is ready, compute inner product
          //**/ but first normalize it
          for (jj = 0; jj < vecDTheta.length(); jj++)
            vecDTheta[jj] /= (double) matPriorSample.nrows();

          psVector vecVT;
          matCov.matvec(vecDTheta,vecVT,0);
          ddata = 0.0;
          for (jj = 0; jj < nUInps; jj++)
            ddata += vecDTheta[jj] * vecVT[jj];
          //**/ negative because this quantity is actually 
          //**/ sigma^2 - innerproduct (so max is min)
          if (ddata > Gmax) Gmax = ddata; 
          IMetric2 += ddata;
        }
        GMetric2 += Gmax;
      }
    }
    GMetric2 /= (double) nInputs;
    IMetric2 /= (double) (nInputs * matEvalSet.nrows());
    DMetric2 /= (double) nInputs;
    AMetric2 /= (double) nInputs;
    EMetric2 /= (double) nInputs;

    //**/ --- return the proper metric
    if (whichLocalFunction_ == 45) outputs[0] = GMetric2;
    if (whichLocalFunction_ == 46) outputs[0] = IMetric2;
    if (whichLocalFunction_ == 47) outputs[0] = DMetric2;
    if (whichLocalFunction_ == 48) outputs[0] = AMetric2;
    if (whichLocalFunction_ == 49) outputs[0] = EMetric2;
    if (outputs[0] < optimalVal) optimalVal = outputs[0];
    if (nInputsIn > 0 && printLevel_ >= 0) 
    {
      printf(" ===> output = %11.4e (best so far = %11.4e",
             outputs[0], optimalVal);
      if (outputs[0] == optimalVal) 
      {
        printf(" :");
        for (kk = 0; kk < nInputs; kk++)
          printf(" %d", (int) inputs[kk]);
      }
      printf(")");
    }
    printf("\n");
    //**/ if nInputsIn = 0, it means this function is called
    //**/ from the interpreter to evaluate the metrics (i.e.
    //**/ odoeu_eval), and to expedite things, one call with
    //**/ code 45 will return 2 metrics and 46 with 3.
    if (nInputsIn == 0) 
    {
      if (whichLocalFunction_ == 45 && nOutputs >= 2)
      {
        outputs[0] = GMetric2;
        outputs[1] = IMetric2;
      }
      if (whichLocalFunction_ == 47 && nOutputs >= 3)
      {
        outputs[0] = DMetric2;
        outputs[1] = AMetric2;
        outputs[2] = EMetric2;
      }
      return 0;
    }

    //**/ --- update iteration history
    if (nHist >= maxHist)
    {
      for (jj = 0; jj < maxHist/2; jj++)
        for (ii = 0; ii < nInputs+1; ii++)
          matHistory.setEntry(jj,ii,
                    matHistory.getEntry(jj+maxHist/2,ii));
      nHist = maxHist/2;
    }
    for (ii = 0; ii < nInputs; ii++)
      matHistory.setEntry(nHist,ii,inputs[ii]);
    matHistory.setEntry(nHist,nInputs,outputs[0]);
    nHist++;
    return 0;
  }

  //**/ =============================================================
  //**/ default option: response surface based optimization that 
  //**/ mimics the deterministic version of inference
  //**/ (This is called by setting the opt_driver field in the 
  //**/ APPLICATION section to PSUADE_OPT_RSLS)
  //**/ =============================================================
  else if (appOptFlag_ == 1)
  {
    //**/ -----------------------------------------------------------
    //**/ initialization 
    //**/ -----------------------------------------------------------
    if (ProblemInitialized == 0)
    {
      //**/ firstly, clean up
      if (printLevel_ > 0)
        printf("PSUADE_LOCAL: LS optimization initialization begins ..\n");
      if (rsPtrs != NULL)
      {
        for (ii = 0; ii < nOuts; ii++)
          if (rsPtrs[ii] != NULL) delete rsPtrs[ii];
        delete [] rsPtrs;
      }

      //**/ ask for training sample
      printf("FuncIO: This local function is for computing the ");
      printf("least-squares error\n");
      printf("        given a response surface for prediction ");
      printf("and an experimental\n");
      printf("        data set for comparison (Similar to MCMC ");
      printf("except that this is\n");
      printf("        deterministic and stochastic like MCMC).\n");
      printf("NOTE: The number of inputs in the training sample ");
      printf("should be the sum of\n");
      printf("      the number of optimization variables and ");
      printf("the number of DESIGN\n");
      printf("      parameters in the experimental data set.\n");
      printf("Please enter your training sample (PSUADE format) : "); 
      scanf("%s", fname);
      fgets(lineIn,5000,stdin); 
      FILE *fp = fopen(fname, "r");
      if (fp == NULL)
      {
        printf("ERROR: training sample file %s not found.\n", fname);
        printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
        exit(1);
      }
      PsuadeData *psIO = new PsuadeData();
      psIO->setOutputLevel(0);
      status = psIO->readPsuadeFile(fname);
      if (status != 0)
      {
        printf("ERROR: sample file %s probably in wrong format.\n", 
               fname);
        printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
        exit(0);
      }
      psIO->getParameter("input_ninputs", pdata);
      nInps = pdata.intData_;
      psIO->getParameter("output_noutputs", pdata);
      nOuts = pdata.intData_;
      if (nOuts == 0)
      {
        printf("ERROR: the training sample has no output.\n"); 
        printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
        exit(0);
      }
      psIO->getParameter("method_nsamples", pdata);
      int nSamp = pdata.intData_;
      if (nSamp == 0)
      {
        printf("ERROR: No sample in the training sample file.\n"); 
        printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
        exit(0);
      }

      //**/ build response surface
      psVector vecYT;
      pData pInps, pOuts, pLBs, pUBs;
      psIO->getParameter("input_lbounds", pLBs);
      psIO->getParameter("input_ubounds", pUBs);
      psIO->getParameter("input_sample", pInps);
      psIO->getParameter("output_sample", pOuts);
      rsPtrs = new FuncApprox*[nOuts];
      printf("FuncIO: constructing response surfaces\n");
      vecYT.setLength(nSamp);
      for (ii = 0; ii < nOuts; ii++)
      {
        rsPtrs[ii] = genFAInteractive(psIO, iOne);
        rsPtrs[ii]->setBounds(pLBs.dbleArray_,pUBs.dbleArray_);
        rsPtrs[ii]->setOutputLevel(0);
        for (jj = 0; jj < nSamp; jj++)
          vecYT[jj] = pOuts.dbleArray_[jj*nOuts+ii];
        status = rsPtrs[ii]->initialize(pInps.dbleArray_,
                                        vecYT.getDVector());
        if (status != 0)
        {
          printf("ERROR: Failed to build response surface for output %d\n",
                 ii+1);
          printf("Where: file %s line %d, exiting\n",__FILE__,__LINE__);
          exit(0);
        }
      }

      //**/ extract experimental data 
      printf("Next, enter the file that contains experimental data.\n");
      printf("The data file should be in the following format: \n");
      printf("Line 1: <number of data points> <nInputs> <nOutputs>\n");
      printf("Line 2: 1 <inputs for point 1> <output/scale pairs>\n");
      printf("Line 3: 2 <inputs for point 2> <output/scale pairs>\n");
      printf("... (which is the same as that for rsmcmc)\n");
      MCMCAnalyzer *mcmc = new MCMCAnalyzer();
      double dstat = mcmc->readSpecFile(nInps, nOuts, vecDParams,
                         matOptInps,matOptMeans,matOptStds,kk,0);
      if (printLevel_ > 3)
      {
        printf("  Experimental data : \n");
        for (ii = 0; ii < matOptMeans.nrows(); ii++)
        {
          if (matOptInps.nrows() > 0)
          {
            printf("  Input(s) = ");
            for (jj = 0; jj < matOptInps.ncols(); jj++)
              printf("%16.8e ", matOptInps.getEntry(ii,jj));
          }
          for (jj = 0; jj < matOptMeans.ncols(); jj++)
            printf(" mean = %16.8e, stdv = %16.8e\n",
                   matOptMeans.getEntry(ii,jj),
                   matOptStds.getEntry(ii,jj));
        }
      }
      if (dstat != 0)
      {
        printf("ERROR: when reading the experimental data file.\n");
        printf("Where: file %s line %d, exiting\n", __FILE__, __LINE__);
        exit(0);
      }
      delete mcmc;
      if (nInps + matOptInps.ncols() != nInputsIn)
      {
        printf("PSUADDE_LOCAL ERROR: nInputs mismatch\n");
        printf("   Training data nInputs = %d (1)\n", nInps); 
        printf("   Experiment    nInputs = %d (2)\n", 
               matOptInps.nrows()); 
        printf("   Optimization  nInputs = %d (3)\n", nInputsIn); 
        printf("(1) should be equal to (2) + (3)\n"); 
      }
      ProblemInitialized = 1;
      if (printLevel_ > 0)
        printf("PSUADE_LOCAL: LS optimization Initialization complete\n");
    }
    psVector vecXX;
    vecXX.setLength(nInps);
    int offset1, offset2;
    double loglikelihood=0, mse, YOut;
    if (printLevel_ > 2)
    {
      printf("  Current optimization inputs : \n");
      for (jj = 0; jj < nInps; jj++)
        if (vecDParams.length() == 0 || vecDParams[jj] == 0)
          printf("    Input %4d = %14.6e\n",jj+1,vecXX[jj]);
    } 
    for (ii = 0; ii < matOptMeans.nrows(); ii++)
    {
      offset1 = offset2 = 0;
      for (jj = 0; jj < nInps; jj++)
      {
        if (vecDParams.length() > 0 && vecDParams[jj] == 1)
        {
          vecXX[jj] = matOptInps.getEntry(ii, offset1);
          offset1++;
        }
        else
        {
          vecXX[jj] = inputs[offset2];
          offset2++;
        }
      }
      if (printLevel_ > 3)
      {
        //printf("  Current inputs : \n");
        //for (jj = 0; jj < nInps; jj++)
        //  printf("    Input %4d = %14.6e\n",jj+1,vecXX[jj]);
        printf("  Current outputs : \n");
      } 
      mse = 0;
      for (jj = 0; jj < nOuts; jj++)
      {
        YOut = rsPtrs[jj]->evaluatePoint(vecXX.getDVector());
        mse += pow(YOut-matOptMeans.getEntry(ii,jj),2.0)/
              (pow(matOptStds.getEntry(ii,jj),2.0));
        if (printLevel_ > 3)
          printf("  Sample %4d Output %3d = %14.6e, prediction = %14.6e\n",
                 ii+1,jj+1,matOptMeans.getEntry(ii,jj),YOut);
      }
      loglikelihood += mse;
    }
    outputs[0] = loglikelihood;
    if (printLevel_ > 1)
      printf("PSUADE_LOCAL squared error function value = %14.6e\n",
             loglikelihood);
    return 0;
  }
  else
  {
    printf("PSUADE_Local ERROR: no available function.\n");
    printf("NOTE: If you would like to insert your own ");
    printf("simulation model to PSUADE\n");
    printf("      for faster processing (since calling ");
    printf("user simulation model via\n");
    printf("      the driver interface may incur high I/O ");
    printf("cost), add your code to\n");
    printf("      line %d of file %s.\n",__LINE__,__FILE__);
    exit(1);
  }
#if 0
  {
    //**/ =============================================================
    //**/ default option: OUU example
    //**/ =============================================================
    double X1, X2, X3, X4, D, W1, W2, W3, T1, T2, Y, D1, D2, D3;
    double D4, W4, alpha=5, beta=1, delta=-10, gamma3, gamma4;

    //**/ --- Dowling's toy problem
    if (printLevel_ > 1)
      printf("Local function: Dowling's toy problem.\n");
    if (toyOption == -1)
    {
      printf("Available test problem :\n");
      printf("1. Dowling's toy problem - function in one-stage OUU.\n");
      printf("2. Dowling's toy problem - optimizer in two-stage OUU.\n");
      printf("Which problem to use ? (1 or 2) ");
      scanf("%d", &toyOption);
      if (toyOption != 2) toyOption = 1;
      if (toyOption == 1)
           printf("Dowling's toy function for single-stage OUU selected.\n");
      else printf("Dowling's toy optimizer for two-stage OUU selected.\n");
    }
    if (toyOption == 1)
    {
      D  = inputs[0];
      X1 = inputs[1];
      X2 = inputs[2];
      W1 = inputs[3];
      W2 = inputs[4];
      W3 = inputs[5];
      Y  = pow(X1 - D + W1, 2.0) + (1 + W1 * W1) * pow(X2 - D + W2, 2.0) +
           (W1 + W3) * X1 + (W2 + W3) * X2 + pow(W2*W2+(2+W3*W3)*D, 2.0);
      outputs[0] = Y;
    }
    else
    {
      if (nInputs != 12)
      {
        printf("psLocalFunction ERROR: nInputs do not match.\n");
        exit(1);
      }
      D1 = inputs[0];
      D2 = inputs[1];
      D3 = inputs[2];
      D4 = inputs[3];
      X1 = inputs[4];
      X2 = inputs[5];
      X3 = inputs[6];
      X4 = inputs[7];
      W1 = inputs[8];
      W2 = inputs[9];
      W3 = inputs[10];
      W4 = inputs[11];
      X1 = - W1;
      gamma3 = 1 + W3 * W3;
      gamma4 = 1 + W4 * W4;
      X2 = - (delta + beta*D2*gamma4 + beta*gamma4*W2)/(beta*gamma4);
      X3 = - (delta * W3 + D3*gamma3*gamma4 + gamma3*gamma4*W3) / 
             (gamma3 * gamma4);
      T1 = delta*gamma3 + beta*delta*W3*W3 + beta*gamma3*gamma4*W2;
      T2 = beta*gamma3*gamma4*W3*W3 + beta*D2 * gamma3 * gamma4;
      T1 = T1 + T2;
      T2 = beta * D4 * gamma3 * gamma4;
      T1 = T1 - T2;
      T2 = beta*delta*gamma3*gamma4 + beta*D3*gamma3*gamma4*W3;
      X4 = (T1 + T2) / (beta * gamma3 * gamma4 * gamma4);

      Y = (X1 + W1) * (X1 + W1);
      Y = Y + beta * pow(X2 + W2 + D2, 2.0);
      Y = Y + (1 + W3 * W3) * pow(X3 + D3 + W3, 2.0);
      T1 = pow(D4 + X2 + W3 * X3 + X4 * (1 + W4 * W4), 2.0);
      Y = Y + 1.0/(1+W4*W4) * T1;
      Y = Y - 2.0 * delta * X4;
      Y = Y + alpha * pow(D1+W1, 2.0);
      Y = Y + (10 - alpha) * D1 * D1;
      Y = Y + (10 - beta) * D2 * D2;
      Y = Y + 3 * D3 * D3;
      Y = Y + D4 * D4 * sqrt(1+W3*W3+W4*W4);
      outputs[0] = Y;
      if (printLevel_ > 1) printf(" ===> output = %e\n", Y);
    }
  }
#endif
  return 0;
}

// ************************************************************************
// Local function
// ------------------------------------------------------------------------
int FunctionInterface::psEnsembleLocalFunction(int nSamples, int nInputs, 
                              double *inputs, int nOutputs, double *outputs)
{
  int    ss, ii;
  double X1, X2, X3, X4, D, W1, W2, W3, T1, T2, Y, D1, D2, D3, D4, W4;
  double alpha=5, beta=1, delta=-10, gamma3, gamma4;
  static int initializeProblem=2;

  //**/ Dowling's toy problem
  if (printLevel_ > 0)
    printf("FuncIO local function: Dowling's toy problem.\n");
  if (initializeProblem == -1)
  {
    printf("Available test problem :\n");
    printf(" 1. OUU toy problem - function in single-stage OUU.\n");
    printf(" 2. OUU toy problem - optimizer in two-stage OUU.\n");
    printf("Which problem to use ? (1 or 2) ");
    scanf("%d", &initializeProblem);
    if (initializeProblem != 2) initializeProblem = 1;
    if (initializeProblem == 1)
         printf("OUU toy function for single-stage OUU selected.\n");
    else printf("OUU toy optimizer for two-stage OUU selected.\n");
  }
  if (initializeProblem == 1)
  {
    for (ss = 0; ss < nSamples; ss++)
    {
      D  = inputs[ss*nInputs];
      X1 = inputs[ss*nInputs+1];
      X2 = inputs[ss*nInputs+2];
      W1 = inputs[ss*nInputs+3];
      W2 = inputs[ss*nInputs+4];
      W3 = inputs[ss*nInputs+5];
      Y = pow(X1 - D + W1, 2.0) + (1 + W1 * W1) * pow(X2 - D + W2, 2.0) +
          (W1 + W3) * X1 + (W2 + W3) * X2 + pow(W2*W2+(2+W3*W3)*D, 2.0);
      outputs[ss] = Y;
    }
  }
  else
  {
    if (nInputs != 12)
    {
      printf("psEnsembleLocalFunction ERROR: nInputs do not match.\n");
      exit(1);
    }
    for (ss = 0; ss < nSamples; ss++)
    {
      D1  = inputs[ss*nInputs];
      D2 = inputs[ss*nInputs+1];
      D3 = inputs[ss*nInputs+2];
      D4 = inputs[ss*nInputs+3];
      X1 = inputs[ss*nInputs+4];
      X2 = inputs[ss*nInputs+5];
      X3 = inputs[ss*nInputs+6];
      X4 = inputs[ss*nInputs+7];
      W1 = inputs[ss*nInputs+8];
      W2 = inputs[ss*nInputs+9];
      W3 = inputs[ss*nInputs+10];
      W4 = inputs[ss*nInputs+11];
      X1 = - W1;
      gamma3 = 1 + W3 * W3;
      gamma4 = 1 + W4 * W4;
      X2 = - (delta + beta * D2 * gamma4 + beta * gamma4 * W2)/(beta*gamma4);
      X3 = - (delta * W3 + D3 * gamma3 * gamma4 + gamma3 * gamma4 * W3) / 
           (gamma3 * gamma4);
      T1 = delta * gamma3 + beta*delta*W3*W3 + beta * gamma3 * gamma4 * W2;
      T2 = beta * gamma3 * gamma4 * W3 * W3 + beta * D2 * gamma3 * gamma4;
      T1 = T1 + T2;
      T2 = beta * D4 * gamma3 * gamma4;
      T1 = T1 - T2;
      T2 = beta * delta * gamma3 * gamma4 + beta*D3 * gamma3 * gamma4 * W3;
      X4 = (T1 + T2) / (beta * gamma3 * gamma4 * gamma4);

      Y = (X1 + W1) * (X1 + W1);
      Y = Y + beta * pow(X2 + W2 + D2, 2.0);
      Y = Y + (1 + W3 * W3) * pow(X3 + D3 + W3, 2.0);
      T1 = pow(D4 + X2 + W3 * X3 + X4 * (1 + W4 * W4), 2.0);
      Y = Y + 1.0/(1+W4*W4) * T1;
      Y = Y - 2.0 * delta * X4;
      Y = Y + alpha * pow(D1+W1, 2.0);
      Y = Y + (10 - alpha) * D1 * D1;
      Y = Y + (10 - beta) * D2 * D2;
      Y = Y + 3 * D3 * D3;
      Y = Y + D4 * D4 * sqrt(1+W3*W3+W4*W4);
      outputs[ss] = Y;
    }
  }
  return 0;
}

// ************************************************************************
// read data file 
// ------------------------------------------------------------------------
double FunctionInterface::readDataFile(int nInputs, int nOutputs,  
                 psMatrix &matExpInps, psMatrix &matExpMeans, 
                 psMatrix &matExpStdvs, int printLevel)
{
  int    ii, jj, kk, dnSamples, dnInputs;
  double ddata, ddata2;
  char   lineIn[1001], cfname[1001], cword[101];
  FILE   *fp=NULL;

  //**/ read spec file for likelihood function
  printf("==> Enter the name of the data file: ");
  scanf("%s", cfname);
  fgets(lineIn, 1000, stdin);
  kk = strlen(cfname);
  if (kk <= 1000)
  {
    cfname[kk] = '\0';
    fp = fopen(cfname, "r");
    if (fp == NULL)
    {
      printf("FuncIO ERROR : cannot open data file %s.\n",cfname);
      return PSUADE_UNDEFINED;
    }
  }
  else
  {
    printf("FuncIO ERROR : file name too long.\n");
    return PSUADE_UNDEFINED;
  }

  sscanf(lineIn, "%s", cword); 
  int lineCnt = 0;
  if (!strcmp(cword, "PSUADE_BEGIN")) lineCnt++;
  lineCnt--;
  fclose(fp);
  fp = fopen(cfname, "r");
  for (ii = 0; ii < lineCnt; ii++) fgets(lineIn, 2000, fp);
  fscanf(fp, "%d %d %d", &dnSamples, &dnInputs, &kk);
  if (dnSamples <= 0)
  {
    printf("FuncIO ERROR: no. of data <= 0.\n");
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  printf("FuncIO INFO: Number of data points = %d\n",dnSamples);

  if (kk != nOutputs)
  {
    printf("FuncIO ERROR: nOutputs = %d in data file != %d\n",
           kk, nOutputs);
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  printf("Data file : Number of outputs = %d\n",nOutputs);

  if (dnInputs < 0)
  {
    printf("FuncIO ERROR: number of variables < 0.\n");
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  if (dnInputs != nInputs)
  {
    printf("FuncIO ERROR: number of inputs %d != %d\n",
           dnInputs,nInputs);
    fclose(fp);
    return PSUADE_UNDEFINED;
  }
  printf("Data File : Number of inputs = %d\n",dnInputs);
  matExpInps.setFormat(PS_MAT1D);
  matExpInps.setDim(dnSamples, dnInputs);
  matExpMeans.setFormat(PS_MAT1D);
  matExpStdvs.setFormat(PS_MAT1D);
  matExpMeans.setDim(dnSamples, nOutputs);
  matExpStdvs.setDim(dnSamples, nOutputs);
  for (ii = 0; ii < dnSamples; ii++)
  {
    fscanf(fp, "%d", &kk);
    if (kk != ii+1)
    {
      printf("FuncIO ERROR: Invalid data number %d ",kk);
      printf("     at line %d in the data file.\n",ii+2);
      printf("              Data number expected = %d\n",ii+1);
      printf("==> check line %d\n", ii+3);
      fclose(fp);
      return PSUADE_UNDEFINED;
    }
    if (printLevel > 0)
      printf("Data %d : \n", ii+1);
    for (jj = 0; jj < dnInputs; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      matExpInps.setEntry(ii, jj, ddata);
      if (printLevel > 0)
        printf("     Input %d = %e\n", jj+1, ddata);
    }
    for (jj = 0; jj < nOutputs; jj++)
    {
      fscanf(fp, "%lg %lg", &ddata, &ddata2);
      matExpMeans.setEntry(ii, jj, ddata);
      matExpStdvs.setEntry(ii, jj, ddata2);
      if (printLevel > 0)
        printf("     Output mean/stdev = %16.8e %16.8e\n",
               ddata, ddata2);
      if (ddata2 < 0.0)
      {
        fclose(fp);
        printf("FuncIO ERROR: std dev in data file <= 0.\n");
        printf("==> check the last entry in line %d\n", ii+3);
        return PSUADE_UNDEFINED;
      }
    }
  }
  fclose(fp);
  return 0;
}

// ************************************************************************
// create an FunctionInterface instantiation
// ------------------------------------------------------------------------
FunctionInterface *createFunctionInterface(PsuadeData *psuadeIO)
{
  int   nInputs, nOutputs;
  pData pPtr, pINames, pONames, pAppFiles;
  FunctionInterface *funcIO;

  //**/ -------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ -------------------------------------------------------------
  funcIO = new FunctionInterface();
  psuadeIO->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO->getParameter("input_names", pINames);
  psuadeIO->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psuadeIO->getParameter("output_names", pONames);
  psuadeIO->getParameter("app_files", pAppFiles);
  funcIO->loadInputData(nInputs, pINames.strArray_);
  funcIO->loadOutputData(nOutputs,pONames.strArray_);
  funcIO->loadFunctionData(pAppFiles.nStrings_, pAppFiles.strArray_);
  return funcIO;
}

// ************************************************************************
// create an FunctionInterface instantiation without setting up the
// driver (in case it is a response surface, this will save the time to
// set it up)
// ------------------------------------------------------------------------
FunctionInterface *createFunctionInterfaceSimplified(PsuadeData *psuadeIO)
{
  int   nInputs, nOutputs;
  pData pPtr, pINames, pONames, pAppFiles;
  FunctionInterface *funcIO;

  //**/ -------------------------------------------------------------
  // set up the FunctionInterface (for sample runs)
  //**/ -------------------------------------------------------------
  funcIO = new FunctionInterface();
  assert(psuadeIO->getParameter("input_ninputs", pPtr) == 0);
  nInputs = pPtr.intData_;
  assert(psuadeIO->getParameter("input_names", pINames) == 0);
  funcIO->loadInputData(nInputs, pINames.strArray_);
  assert(psuadeIO->getParameter("output_noutputs", pPtr) == 0);
  nOutputs = pPtr.intData_;
  assert(psuadeIO->getParameter("output_names", pONames) == 0);
  funcIO->loadOutputData(nOutputs, pONames.strArray_);
  assert(psuadeIO->getParameter("app_files", pAppFiles) == 0);
  strcpy(pAppFiles.strArray_[0], "NULL");
  funcIO->loadFunctionData(pAppFiles.nStrings_, pAppFiles.strArray_);
  return funcIO;
}

// ************************************************************************
// create an FunctionInterface instantiation using data file for
// response surface
// ------------------------------------------------------------------------
FunctionInterface *createFunctionInterfaceGivenAppDriver(int nInputs,
                                               int nOutputs, char *fname)
{
  int   ii, nAppFiles=5;
  char  pString[1000];
  psStrings strInpNames, strOutNames, strAppFiles;
  FunctionInterface *funcIO;

  strAppFiles.setNumStrings(nAppFiles);
  for (ii = 0; ii < nAppFiles; ii++)
  {
    strcpy(pString, "NULL");
    strAppFiles.loadOneString(ii, pString);
  }
  strAppFiles.loadOneString(0, fname);

  strInpNames.setNumStrings(nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    strcpy(pString, "XX");
    strInpNames.loadOneString(ii, pString);
  }

  strOutNames.setNumStrings(nOutputs);
  for (ii = 0; ii < nOutputs; ii++)
  {
    strcpy(pString, "YY");
    strOutNames.loadOneString(ii, pString);
  }

  funcIO = new FunctionInterface();
  funcIO->loadInputData(nInputs, strInpNames.getStrings());
  funcIO->loadOutputData(nOutputs,strOutNames.getStrings());
  funcIO->loadFunctionData(nAppFiles, strAppFiles.getStrings());

  return funcIO;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
FunctionInterface& FunctionInterface::operator=(const FunctionInterface &)
{
  printf("FuncIO operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}


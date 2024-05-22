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
// Functions for the class PsuadeBase
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#ifdef WINDOWS
#define UNICODE
#include <windows.h>
//extern void Sleep(unsigned long milliseconds);
#endif //WINDOWS

// ------------------------------------------------------------------------
// system includes
// ------------------------------------------------------------------------
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <unistd.h>

// ------------------------------------------------------------------------
// local includes : class definition and utilities
// ------------------------------------------------------------------------
#include "Psuade.h"
#include "PsuadeBase.h"
#include "dtype.h"
#include "sysdef.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"

// ------------------------------------------------------------------------
// local includes : sampling methods and input distributions
// ------------------------------------------------------------------------
#include "PDFManager.h"
#include "PDFBase.h"

// ------------------------------------------------------------------------
// local includes : function approximator
// ------------------------------------------------------------------------
#include "FuncApprox.h"

// ------------------------------------------------------------------------
// local includes : optimizers
// ------------------------------------------------------------------------
#include "Optimizers/Optimizer.h"

// ------------------------------------------------------------------------
// local includes : python
// ------------------------------------------------------------------------
#ifdef HAVE_PYTHON
#include "Python.h"
#endif

// ------------------------------------------------------------------------
// local defines 
// ------------------------------------------------------------------------
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PsuadeBase::PsuadeBase()
{
  outputLevel_   = 0;
  sampler_       = NULL;
  psuadeIO_      = NULL;
  useRSModel_    = 0;
  psuadeStop_    = 0;

  inputCMat_     = NULL;
  SamPDFFiles_   = NULL;
  
  nInputs_ = nSamples_ = nOutputs_ = 0;

#ifdef HAVE_PYTHON
  update_gui   = NULL;
  yesno_dialog = NULL;
#endif
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PsuadeBase::~PsuadeBase()
{
  cleanUp();

#ifdef HAVE_PYTHON
  if (update_gui != NULL) Py_DECREF(update_gui);
  update_gui = NULL;
  if (yesno_dialog != NULL) Py_DECREF(yesno_dialog);
  yesno_dialog = NULL;
#endif
}

// ************************************************************************
// get data and parameters from PsuadeData object
// ------------------------------------------------------------------------
int PsuadeBase::getInputFromFile(const char *fname, 
                                 const char *psuadeio_filename)
{
  int   status, nInputs, nOutputs, samplingMethod, nRefines, ii;
  int   randomize, nReps, nSamples, flag, iSum, usePDFs;
  pData pPtr,pSymTable,pLowerB,pUpperB,pOutputs,pInputs,pStates,pPDFs;
  PDFManager *pdfman;

  //**/ ----------------------------------------------------------------
  //**/ sanitize, create IO objects, and read data
  //**/ ----------------------------------------------------------------
  cleanUp();
  if (psuadeIO_ != NULL) delete psuadeIO_;
  psuadeIO_ = new PsuadeData();
  status = psuadeIO_->readPsuadeFile(fname);
  if (status != 0) 
  {
    printOutTS(PL_ERROR,
       "PsuadeBase ERROR: cannot read file %s or wrong format.\n",fname);
    return -1;
  }
  //**/ ----------------------------------------------------------------
  //**/ If supplied a second file name, read PSUADE_IO section from there
  //**/ Normally the second file is not used (for Python wrapper only)
  //**/ ----------------------------------------------------------------
  if (psuadeio_filename != NULL)
  {
    status = psuadeIO_->readPsuadeIO(psuadeio_filename);
    if (status != 0) 
    {
      printOutTS(PL_ERROR,
           "PsuadeBase ERROR: cannot read file %s or wrong format.\n",
           psuadeio_filename);
      return -1;
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ get input, output, method, and other info from psuadeIO 
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;
  printOutTS(PL_DETAIL, "PSUADE::getInputFromFile begins.\n");
  psuadeIO_->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO_->getParameter("input_symtable", pSymTable);
  psuadeIO_->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psuadeIO_->getParameter("method_nsamples", pPtr);
  nSamples = pPtr.intData_;
  psuadeIO_->getParameter("method_sampling", pPtr);
  samplingMethod = pPtr.intData_;
  psuadeIO_->getParameter("method_nreplications", pPtr);
  nReps = pPtr.intData_;
  psuadeIO_->getParameter("method_randomflag", pPtr);
  randomize = pPtr.intData_;
  psuadeIO_->getParameter("input_use_input_pdfs", pPtr);
  usePDFs = pPtr.intData_;
  psuadeIO_->getParameter("input_pdfs", pPDFs);
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  nRefines = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ if a restart file is read, get the samples and load it to sampler.
  //**/ Otherwise, load bounds, create and then get sample points.
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_lbounds", pLowerB);
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  psuadeIO_->getParameter("output_sample", pOutputs);

  if (pOutputs.dbleArray_ != NULL)
  {
    iSum = 0;
    if (pPDFs.intArray_ != NULL) 
    {
      for (ii = 0; ii < nInputs; ii++) iSum += pPDFs.intArray_[ii];
    }
    if ((iSum > 0) && (nRefines > 0) && (usePDFs == 1))
    {
      printOutTS(PL_ERROR,
                 "PSUADE ERROR: you have requested sample refinement.\n");
      printOutTS(PL_ERROR,
                 "       Sample refinement requires that all the input\n");
      printOutTS(PL_ERROR,"       probability distributions be uniform.\n");
      exit(1);
    }
    psuadeIO_->getParameter("input_sample", pInputs);
    psuadeIO_->getParameter("output_states", pStates);
    if ((nRefines == 0) && (samplingMethod == -1)) 
      samplingMethod = PSUADE_SAMP_MC;
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samplingMethod); 
    sampler_->setPrintLevel(outputLevel_);
    sampler_->setInputBounds(nInputs,pLowerB.dbleArray_,pUpperB.dbleArray_);
    sampler_->setInputParams(nInputs, NULL, NULL, pSymTable.intArray_);
    sampler_->setOutputParams(nOutputs);
    sampler_->setSamplingParams(nSamples, nReps, randomize);
    //**/if (samplingMethod == PSUADE_SAMP_SG)
    //**/{
    //**/   snprintf(sparam,100,"nLevels %d", nRefines); 
    //**/   sampler_->setParam(sparam);
    //**/}
    sampler_->initialize(1);
    sampler_->loadSamples(nSamples, nInputs, nOutputs, pInputs.dbleArray_,
                          pOutputs.dbleArray_, pStates.intArray_);
  }
  else
  {
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samplingMethod);
    sampler_->doSampling(psuadeIO_);
  }
  psuadeIO_->writePsuadeFile(NULL,1);
  return 0;
}

// ************************************************************************
// run integrated design and analysis
// ------------------------------------------------------------------------
int PsuadeBase::run()
{
  char  *appName, inString[200];
  FILE  *fp;
  pData pAppFiles, pPtr;

  //**/ ----------------------------------------------------------------
  //**/ make sure the PSUADE IO object has been given
  //**/ ----------------------------------------------------------------
  if (psuadeIO_ == NULL)
  {
    printOutTS(PL_ERROR, "PSUADE::run ERROR - no PsuadeData object.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ check whether rsModel is requested. If so, set useRSModel_
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("app_files", pAppFiles);
  appName = pAppFiles.strArray_[0];
  fp = fopen(appName, "r");
  if (fp != NULL)
  {
    fscanf(fp, "%10c", inString);
    if (!strncmp(inString, "PSUADE_IO",9)) useRSModel_ = 1;
    fclose(fp);
  }

  //**/ ----------------------------------------------------------------
  //**/ initialize the stop flag to 0. Set it to 1 if we need to stop
  //**/ ----------------------------------------------------------------
  psuadeStop_ = 0;

  //**/ ----------------------------------------------------------------
  //**/ call the appropriate function to run the simulations
  //**/ ----------------------------------------------------------------
  //psuadeIO_->getParameter("method_refine_type", pPtr);
  //int refineType = pPtr.intData_;
  psuadeIO_->getParameter("ana_method", pPtr);
  int anaMethod = pPtr.intData_;

  //**/ use nearest neighbors gradients for refinement analysis
  if (anaMethod == PSUADE_ANA_ARSMNN)
  {
    runAdaptiveNN();
    printf("INFO: Use MARS response surface for the adaptive sample.\n");
  }
  //**/ use MARS with bagging (if initial sample not METIS)
  else if (anaMethod == PSUADE_ANA_ARSMMB) 
  {
    psuadeIO_->getParameter("method_sampling", pPtr);
    int samMethod = pPtr.intData_;
    if (samMethod == PSUADE_SAMP_METIS)
         runAdaptiveErrBased1();
    else runAdaptiveErrBasedG();
    printf("Advice: Use MARS response surface for the adaptive sample.\n");
  }
  //**/ reliability analysis
  else if (anaMethod == PSUADE_ANA_REL) runAdaptivePRA();
  //**/ optimization using adaptive refinement 
  else if (anaMethod == PSUADE_ANA_AOPT) runAdaptiveOpt();
  //**/ adaptive response surface using derivative information
  else if (anaMethod == PSUADE_ANA_GLSA) runAdaptiveGradBased();
  //**/ fetch run type and call sequential or ensemble runs
  else
  {
    psuadeIO_->getParameter("app_runtype", pPtr);
    int runType = pPtr.intData_;
    //**/ type=16 ==> ensemble run mode
    if (((runType & 16) == 0) || (useRSModel_ == 1)) runUniform();
    else                                             runEnsemble();
  }

  //**/ ----------------------------------------------------------------
  //**/ when all is done, tell psuadeIO which may write results to file 
  //**/ ----------------------------------------------------------------
  if (psuadeIO_ != NULL) psuadeIO_->processOutputData();
  return 0;
}

// ************************************************************************
// interactive session
// ------------------------------------------------------------------------
int PsuadeBase::sessionInteractive()
{
  setPrintLevelTS(3);
  interpretInteractive();
  return 0;
}

// ************************************************************************
// parallel interactive session
// ------------------------------------------------------------------------
int PsuadeBase::sessionInteractiveParallel()
{
  setPrintLevelTS(3);
  interpretInteractiveParallel();
  return 0;
}

// ************************************************************************
// ************************************************************************

// ************************************************************************
// run the sample points on a single processor
// ------------------------------------------------------------------------
int PsuadeBase::runUniform()
{
  int    mm, oldNSamples, sampleID, refineFlag=1, count, status, askFlag=1;
  int    ii, iR, iteration, nReUsed, maxState, launchOnly, noAnalysis=0;
  int    refineRatio, parallelJobCount, limitedJobCount, jobsCompletedLast;
  double refineThreshold;
  char   winput[500];
  FILE   *fp;
  pData  pPtr, pLowerB, pUpperB, pAppFiles, pStates, pInpData, pOutData;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (outputLevel_ >= 4) printOutTS(PL_DETAIL, "PSUADE::run begins.\n");
  if (psuadeIO_ == NULL)
  {
    printOutTS(PL_ERROR, "PSUADE run: ERROR - no PsuadeData object.\n");
    exit(1);
  }

  //**/ -------------------------------------------------------------
  //**/ if generate input files only mode, just generate the files
  //**/ -------------------------------------------------------------
  int    runType, nSamples, nInputs, nOutputs, *sampleStates;
  double *sampleInputs, *sampleOutputs;
  FunctionInterface *funcIO=NULL;

  psuadeIO_->getParameter("app_runtype", pPtr);
  runType = pPtr.intData_;
  if (runType & 8)
  {
    printOutTS(PL_INFO, 
               "PSUADE run: INFO - GENERATE INPUT FILE ONLY MODE.\n");
    funcIO = createFunctionInterfaceSimplified(psuadeIO_);
    psuadeIO_->getParameter("input_sample", pInpData);
    sampleInputs = pInpData.dbleArray_;
    psuadeIO_->getParameter("output_sample", pOutData);
    sampleOutputs = pOutData.dbleArray_;
    psuadeIO_->getParameter("output_states", pStates);
    sampleStates = pStates.intArray_;
    psuadeIO_->getParameter("method_nsamples", pPtr);
    nSamples = pPtr.intData_;
    psuadeIO_->getParameter("input_ninputs", pPtr);
    nInputs = pPtr.intData_;
    psuadeIO_->getParameter("output_noutputs", pPtr);
    nOutputs = pPtr.intData_;
    if (sampler_ == NULL)
    {
      printOutTS(PL_ERROR,"PSUADE run: ERROR - no sampler.\n");
      exit(1);
    }
    sampler_->getSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                         sampleOutputs, sampleStates);
    for (sampleID = 0; sampleID < nSamples; sampleID++) 
      status = funcIO->evaluate(sampleID, nInputs,
                       &sampleInputs[sampleID*nInputs], nOutputs, 
                       &sampleOutputs[sampleID*nOutputs],1);   
    delete funcIO;
    return 0;
  }

  //**/ -------------------------------------------------------------
  //**/ get parameters from psuadeIO for experiment and analysis
  //**/ -------------------------------------------------------------
  psuadeIO_->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  nOutputs = pPtr.intData_;
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_randomflag", pPtr);
  int randomize = pPtr.intData_;
  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;
  psuadeIO_->getParameter("ana_graphicsflag", pPtr);
  int sampleGraphics = pPtr.intData_;
  psuadeIO_->getParameter("ana_method", pPtr);
  int analysisMethod = pPtr.intData_;
  psuadeIO_->getParameter("app_files", pAppFiles);
  char *appDriver = pAppFiles.strArray_[0];
  psuadeIO_->getParameter("ana_threshold", pPtr);
  double analysisThreshold = pPtr.dbleData_;
  psuadeIO_->getParameter("ana_opt_switch", pPtr);
  int optSwitch = pPtr.intData_;

  //**/ -------------------------------------------------------------
  //**/ runType 1                   : nondeterministic runs
  //**/ runType 2&!4 (launchOnly=1) : launch only
  //**/ runType 4    (launchOnly=1) : limited launch only
  //**/ runType 8    (launchOnly=0) : generate input file only
  //**/ -------------------------------------------------------------
  if ((runType & 2) && ((runType & 4) == 0))
  {
    launchOnly = 1;
    maxParallelJobs = 100000;
    if (outputLevel_ > 0) 
    {
      printOutTS(PL_INFO,
           "PSUADE run: launch_only, max parallel jobs set to 100000.\n");
      printOutTS(PL_INFO,
           "            launch_interval has been set to %d seconds\n",
           launchInterval);
    }
    outputLevel_ = 3;
  }
  else if (runType & 4)
  {
    launchOnly = 1;
    if (outputLevel_ > 0) 
    {
      printOutTS(PL_INFO,
         "PSUADE run: limited_launch_only mode, max parallel jobs = %d.\n",
         maxParallelJobs);
      printOutTS(PL_INFO,
         "            launch_interval has been set to %d seconds\n",
         launchInterval);
    }
  }
  else launchOnly = 0;

  //**/ -------------------------------------------------------------
  //**/ if no driver found, see if all sample points have been run
  //**/ -------------------------------------------------------------
  if (!strcmp(appDriver, "NONE"))
  {
    psuadeIO_->getParameter("input_sample", pInpData);
    sampleInputs = pInpData.dbleArray_;
    psuadeIO_->getParameter("output_sample", pOutData);
    sampleOutputs = pOutData.dbleArray_;
    psuadeIO_->getParameter("output_states", pStates);
    sampleStates = pStates.intArray_;
    psuadeIO_->getParameter("method_nsamples", pPtr);
    nSamples = pPtr.intData_;
    psuadeIO_->getParameter("input_ninputs", pPtr);
    nInputs = pPtr.intData_;
    psuadeIO_->getParameter("output_noutputs", pPtr);
    nOutputs = pPtr.intData_;
    count = 0;
    for (sampleID = 0; sampleID < nSamples; sampleID++)
      if (sampleStates[sampleID] == 1) count++;
    if (count < nSamples)
    {
      if (optSwitch == 1)
      {
        if (outputLevel_ > 1) 
        {
          printOutTS(PL_INFO,
               "PSUADE INFO: no driver given. The sample points\n");
          printOutTS(PL_INFO,
               "             will be used for optimization only.\n");
        }
        for (sampleID = 0; sampleID < nSamples; sampleID++)
        {
          if (sampleStates[sampleID] == 0)
          {
            for (ii = 0; ii < nOutputs; ii++)
              sampleOutputs[sampleID*nOutputs+ii] = PSUADE_UNDEFINED;
            sampleStates[sampleID] = 1;
          }
        }
      }
      else printOutTS(PL_WARN,"PSUADE WARNING: no driver given.\n");
      nRefinements = 0;
      noAnalysis = 1;
    }
    funcIO = createFunctionInterfaceSimplified(psuadeIO_);
  }
  else
  {
    psuadeIO_->getParameter("input_sample", pInpData);
    sampleInputs = pInpData.dbleArray_;
    psuadeIO_->getParameter("output_sample", pOutData);
    sampleOutputs = pOutData.dbleArray_;
    psuadeIO_->getParameter("output_states", pStates);
    sampleStates = pStates.intArray_;
    psuadeIO_->getParameter("method_nsamples", pPtr);
    nSamples = pPtr.intData_;
    psuadeIO_->getParameter("input_ninputs", pPtr);
    nInputs = pPtr.intData_;
    psuadeIO_->getParameter("output_noutputs", pPtr);
    nOutputs = pPtr.intData_;
    //**/ June 2016: do not check (this will create a problem if
    //**/ all PSUADE_IO points have been evaluated ==> no driver
    //**/ will be returned to perform the optimization
    //**/count = 0;
    //**/for (sampleID = 0; sampleID < nSamples; sampleID++)
    //**/   if (sampleStates[sampleID] == 1) count++;
    //**/if (count < nSamples || nRefinements > 0)
    //**/{
    //**/   if (outputLevel_ > 0) 
    //**/      printOutTS(PL_INFO, 
    //**/           "PSUADE run: creating interface to user driver.\n");
    //**/   funcIO = createFunctionInterface(psuadeIO_);
    //**/}
    //**/else funcIO = createFunctionInterfaceSimplified(psuadeIO_);
    if (outputLevel_ > 0) 
      printOutTS(PL_INFO, 
                 "PSUADE run: creating interface to user driver.\n");
    funcIO = createFunctionInterface(psuadeIO_);
  }

  //**/ -------------------------------------------------------------
  //**/ if a surrogate model has been given, reset to sequential mode
  //**/ -------------------------------------------------------------
  if (useRSModel_)
  {
    maxParallelJobs = 1;
    minJobWaitTime  = 0;
    maxJobWaitTime  = 0;
  }

  //**/ -------------------------------------------------------------
  //**/ set up analysis tools
  //**/ -------------------------------------------------------------
  analysisManager_.clearAnalyzers();
  analysisManager_.setup(psuadeIO_);

  //**/ -------------------------------------------------------------
  //**/ set up general run parameters
  //**/ -------------------------------------------------------------
  if ((sampleGraphics & 2) != 0) sampleGraphics = 1;
  else                           sampleGraphics = 0;
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);
  funcIO->setOutputLevel(outputLevel_);

  //**/ -------------------------------------------------------------
  //**/ loop over number of refinements 
  //**/ -------------------------------------------------------------
  if (outputLevel_ > 0) 
  {
    printOutTS(PL_INFO,"PSUADE run: output level = %d\n", outputLevel_);
    printOutTS(PL_INFO,
       "PSUADE run: max parallel jobs = %d\n", maxParallelJobs);
    printOutTS(PL_INFO,
       "PSUADE run: max job wait time = %d seconds\n", maxJobWaitTime);
    printOutTS(PL_INFO,
       "PSUADE run: min job wait time = %d seconds\n", minJobWaitTime);
    printOutTS(PL_INFO,
       "PSUADE run: launch interval   = %d seconds\n", launchInterval);
    printOutTS(PL_INFO,
       "PSUADE run: save frequency    = every %d runs\n", saveFrequency);
    printOutTS(PL_INFO,
       "NOTE: If evaluation should be fast but is slow, check ");
    printOutTS(PL_INFO,
       "save_frequency\n");
    printOutTS(PL_INFO,
       "      because it may be due to too much I/O.\n");
    printOutTS(PL_INFO,
       "NOTE: To dynamically change max jobs, create psuade_pmachine file\n");
    printOutTS(PL_INFO,
       "NOTE: To terminate gracefully, create psuade_stop file\n");
  }
  psIVector vecRefineNSamps;
  vecRefineNSamps.setLength(nRefinements+1);
  for (iR = 0; iR < nRefinements+1; iR++)
  {
    if (sampler_ == NULL)
    {
      printOutTS(PL_ERROR,"PSUADE run: ERROR - no sampler.\n");
      exit(1);
    }
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 0) 
    {
      printEquals(PL_INFO, 0);
      if (nRefinements > 0)
        printOutTS(PL_INFO, 
             "PSUADE run: refinement %d(out of %d), nSamples = %d\n",
             iR, nRefinements, nSamples);
      else
        printOutTS(PL_INFO, 
             "PSUADE run: nSamples = %d \n", nSamples);
    }

    //**/ ----------------------------------------------------------
    //**/ get run information for this refinement step (from sampler)
    //**/ and transform to the corresponding distribution functions
    //**/ ----------------------------------------------------------
    vecRefineNSamps[iR] = nSamples;
    psuadeIO_->getParameter("input_sample", pInpData);
    sampleInputs = pInpData.dbleArray_;
    psuadeIO_->getParameter("output_sample", pOutData);
    sampleOutputs = pOutData.dbleArray_;
    psuadeIO_->getParameter("output_states", pStates);
    sampleStates = pStates.intArray_;
    psuadeIO_->getParameter("method_nsamples", pPtr);
    nSamples = pPtr.intData_;
    psuadeIO_->getParameter("input_ninputs", pPtr);
    nInputs = pPtr.intData_;
    psuadeIO_->getParameter("output_noutputs", pPtr);
    nOutputs = pPtr.intData_;

    //**/ ----------------------------------------------------------
    //**/ state = 1 : job has been completed
    //**/ ----------------------------------------------------------
    int jobsCompleted = 0;
    for (sampleID = 0; sampleID < nSamples; sampleID++) 
    {
      if (sampleStates[sampleID] == 1) jobsCompleted++;
      else                             sampleStates[sampleID] = 0;
    }

    //**/ ----------------------------------------------------------
    //**/ run all samples
    //**/ ----------------------------------------------------------
    parallelJobCount = limitedJobCount = iteration = 0;
    while ((noAnalysis == 0) && (jobsCompleted < nSamples))
    {
      jobsCompletedLast = jobsCompleted;
      iteration++;
      for (sampleID = 0; sampleID < nSamples; sampleID++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          Py_DECREF(temp);
        }
#endif
        //**/ If anything has set the flag psuadeStop_ to 1, stop now
        if (psuadeStop_ == 1)
        {
          psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                         sampleStates,NULL); 
	  psuadeIO_->writePsuadeFile(NULL,0);
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
	}

        if ((sampleStates[sampleID] == 0) && 
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------
          //**/ if minJobWaitTime = 0, don't look for repeats, as
          //**/ this will take longer than actually running the jobs
          //**/ -------------------------------------------------
          if ((minJobWaitTime > 0) && (runType & 1) == 0)
            status = compareSamples(sampleID,nSamples,nInputs,
                                    sampleInputs, sampleStates);
          else status = -1;

          if (status < 0) /* repeated sample not found */
          {
            if ((runType & 4))
            {
              if (limitedJobCount < maxParallelJobs)
              {
                if (outputLevel_ > 2)
                  printOutTS(PL_INFO, "Limited launch: job = %6d\n",
                             sampleID+1);
                status = funcIO->evaluate(sampleID,nInputs,
                           &sampleInputs[sampleID*nInputs], nOutputs, 
                           &sampleOutputs[sampleID*nOutputs],0);   
                limitedJobCount++;
              }
              else
              {
                if (outputLevel_ > 2)
                  printOutTS(PL_INFO, 
                     "Limited launch: creating jobfile %6d\n",sampleID+1);
                status = funcIO->evaluate(sampleID,nInputs,
                            &sampleInputs[sampleID*nInputs], nOutputs, 
                            &sampleOutputs[sampleID*nOutputs],1);   
              }
            }
            else
            {
              if (outputLevel_ > 2)
                printOutTS(PL_INFO, "Launch: job = %6d\n",sampleID+1);
              status = funcIO->evaluate(sampleID,nInputs,
                           &sampleInputs[sampleID*nInputs], nOutputs, 
                           &sampleOutputs[sampleID*nOutputs],0);   
              if ((runType & 2) == 0) parallelJobCount++;
            }
          }
          else /* repeat found, copy the results or wait */
          {
            printOutTS(PL_INFO, 
                 "PSUADE run: sample %d repeated with sample %d.\n",
                 sampleID+1,status+1);
            if (outputLevel_ > 2)
            {
              for (mm = 0; mm < nInputs; mm++)
                printOutTS(PL_INFO, "(%4d,%4d) : %12.4e %12.4e\n",
                     status+1,sampleID+1,sampleInputs[status*nInputs+mm],
                     sampleInputs[sampleID*nInputs+mm]);
            }
            printOutTS(PL_INFO,"==>substitution in lieu of simulation.\n");
            printOutTS(PL_INFO,"Note: turn on nondeterministic otherwise.\n");

            if (sampleStates[status] == 1)
            {
              for (ii = 0; ii < nOutputs; ii++)
                sampleOutputs[sampleID*nOutputs+ii] =
                            sampleOutputs[status*nOutputs+ii];
              status = 0;
            }
            else status = -(status + 1);
          }

          //**/ -------------------------------------------------
          //**/ if sample run is completed, store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(sampleID, nOutputs,
                           &(sampleOutputs[sampleID*nOutputs]),
                           &(sampleStates[sampleID]));
            //**/ This will take much time if N is large (4/2009)
            //**/psuadeIO_->updateOutputSection(nSamples,nOutputs,
            //**/                    sampleOutputs,sampleStates,NULL);
            if (parallelJobCount > 0) parallelJobCount--;
            if (outputLevel_ > 3)
            {
              printOutTS(PL_INFO, "Completed: job = %6d\n", sampleID+1);
              for (mm = 0; mm < nInputs; mm++)
                printOutTS(PL_INFO, "\t\tinput data %3d = %e\n", mm+1,
                     sampleInputs[sampleID*nInputs+mm]);
              for (mm = 0; mm < nOutputs; mm++)
                printOutTS(PL_INFO, "\t\toutput data %2d = %e\n", mm+1, 
                     sampleOutputs[sampleID*nOutputs+mm]);
            }
            jobsCompleted++;
            if (((jobsCompleted-jobsCompletedLast) % saveFrequency)==0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                                   sampleOutputs,sampleStates,NULL); 
              psuadeIO_->writePsuadeFile(NULL,0);
            }

            if (outputLevel_ > 0)
            {
              if (((sampleID+1) % 100) == 0) 
                printOutTS(PL_INFO, 
                     "\nSample point %6d completed (out of %d).\n",
                            sampleID+1, nSamples);
              else if ((sampleID % 11) == 1) printOutTS(PL_INFO, ".");
                fflush(stdout);
            }
            if ((sampleID % 1) == 0) 
            {
              fp = fopen("psuade_stop","r");
              if (fp != NULL)
              {
                fclose(fp);
                printOutTS(PL_INFO,
                     "psuade_stop file found - terminate (1).\n");
                psuadeIO_->updateOutputSection(nSamples,nOutputs,
                                sampleOutputs,sampleStates,NULL); 
                psuadeIO_->writePsuadeFile(NULL,0);
		//throw Psuade_Stop_Exception();
                printf("INFO: psuade_stop found ==> terminate.\n");
                exit(0);
              }
            }
          }
          else // status > 0 or launchOnly = 1
          {
            sampleStates[sampleID] = status; /* running or waiting */
            if ((status > 0) && (launchOnly == 1)) 
            {
#ifdef WINDOWS
              Sleep(1000 * launchInterval);
#else
              sleep(launchInterval);
#endif
            }
          }
        }
        else if (sampleStates[sampleID] >= 2)
        {
          //**/ ----------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ ----------------------------------------------------
          status = funcIO->evaluate(sampleID,nInputs,
                          &sampleInputs[sampleID*nInputs], nOutputs, 
                          &sampleOutputs[sampleID*nOutputs],2);   
          if (status == 0) 
          {
            sampler_->storeResult(sampleID, nOutputs,
                        &(sampleOutputs[sampleID*nOutputs]),
                        &(sampleStates[sampleID]));
            //**/ This will take much time if N is large (4/2009)
            //**/psuadeIO_->updateOutputSection(nSamples,nOutputs,
            //**/                    sampleOutputs,sampleStates,NULL);
            jobsCompleted++;
            parallelJobCount--;
          }
          else 
          {
            sampleStates[sampleID]++;
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                   "Waiting for Job %d to complete (status = %d)\n",
                   sampleID+1, sampleStates[sampleID]);
          } 
          if ((minJobWaitTime > 0) &&
              ((sampleStates[sampleID]-2)*minJobWaitTime > maxJobWaitTime))
          {
            sampleStates[sampleID] = 0;
            parallelJobCount--;
            sampleID--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO,
                   "PSUADE run: sample %6d to be restarted.\n",
                   sampleID+1);
          }
        }
      }

      //**/ ----------------------------------------------------------
      //**/ update the results in the data file
      //**/ ----------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                     sampleStates,NULL);
      if (jobsCompletedLast < jobsCompleted)
      {
        fillInSamples(nSamples,nOutputs,sampleOutputs,sampleStates);
        psuadeIO_->writePsuadeFile(NULL,0);
      }

      //**/ ----------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ ----------------------------------------------------------
      maxState = 0;
      for (sampleID = 0; sampleID < nSamples; sampleID++)
        if (sampleStates[sampleID] > maxState)
          maxState = sampleStates[sampleID];
      if ((maxState > 100) && (minJobWaitTime <= 0))
      {
        minJobWaitTime  = 1;
        if (outputLevel_ > 0) 
          printOutTS(PL_INFO, 
               "PSUADE run: min job wait time re-set to %d.\n",
               minJobWaitTime);
      }
      if ((maxState > 200) && (minJobWaitTime >  0))
      {
        minJobWaitTime *= 2;
        if (outputLevel_ > 0) 
          printOutTS(PL_INFO, 
               "PSUADE run: min job wait time re-set to %d.\n",
               minJobWaitTime);
      }

      //**/ ----------------------------------------------------------
      //**/ launch_only or limited_launch_only, stop here.
      //**/ ----------------------------------------------------------
      if (launchOnly) break;

      //**/ ----------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ ----------------------------------------------------------
      if ((jobsCompleted < nSamples) && (minJobWaitTime > 0))
      {
        if (outputLevel_ > 0) 
          printOutTS(PL_INFO, 
               "PSUADE run: sleep for %d seconds.\n",minJobWaitTime);
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
        if (minJobWaitTime > 100)
        {
          fp = fopen("psuadeStatus", "a");
          if (fp != NULL)
          {
            fprintf(fp,"\nRefinement %d, iteration %d: \n",iR,iteration);
            for (ii = 0; ii < nSamples; ii++)
              fprintf(fp, "Sample %7d = %d\n",ii+1,sampleStates[ii]);
            fclose(fp);
          }
          printOutTS(PL_INFO, 
               "PSUADE sample state information in psuadeStatus.\n");
        }
      }

      //**/ ----------------------------------------------------------
      //**/ dynamically reconfiguring the max number of parallel jobs
      //**/ ----------------------------------------------------------
      fp = fopen("psuade_pmachine","r");
      if (fp != NULL)
      {
        fscanf(fp, "%d", &status);
        if ((status > 0) && (status < 200))
        {
          maxParallelJobs = status;
          if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
        }
        printOutTS(PL_INFO, 
             "PSUADE run: psuade_pmachine found, max jobs reset to %d.\n",
             maxParallelJobs);
        fclose(fp);
      }

      //**/ ----------------------------------------------------------
      //**/ dynamically restart certain jobs
      //**/ ----------------------------------------------------------
      fp = fopen("psuade_restart","r");
      if (fp != NULL)
      {
        fscanf(fp, "%d", &count);
        for (ii = 0; ii < count; ii++)
        {
	  //**/ Bill Oliver making sure count is a reasonable value.
          //**/ I am not sure this line is needed, but I need to 
          //**/ recall what I did (8/2013)
	  //**/ if (count > maxParallelJobs) break;
          fscanf(fp, "%d", &sampleID);
          if ((sampleID >= 1) && (sampleID <= nSamples)) 
          {
            sampleID--;
            if (sampleStates[sampleID] == 1) 
            {
              printOutTS(PL_INFO,
                 "Sample %d completed, no need to restart.\n",sampleID+1);
            }
            else
            {
              if ((sampleStates[sampleID] != 0) && (parallelJobCount > 0))
                parallelJobCount--;
              sampleStates[sampleID] = 0; 
              printOutTS(PL_INFO,"Sample %d to be restarted.\n",sampleID+1);
            }
          }
        }
        fclose(fp);
      }

      //**/ ----------------------------------------------------------
      //**/ dynamically terminating the session (as control-C does not
      //**/ do the job correctly)
      //**/ ----------------------------------------------------------
      fp = fopen("psuade_stop","r");
      if (fp != NULL)
      {
        fclose(fp);
        printOutTS(PL_INFO,"\npsuade_stop file found - terminate (2).\n");
	psuadeIO_->writePsuadeFile(NULL,0);
	//**/throw Psuade_Stop_Exception();
        printf("INFO: psuade_stop found ==> terminate.\n");
        exit(0);
      }
      if (outputLevel_ > 0) 
      {
        printOutTS(PL_INFO,"\nPSUADE run: jobs completed = %d(out of %d)\n",
                   jobsCompleted, nSamples);
        if ((jobsCompleted == 0) && (askFlag == 1) && (iteration > 10) &&
            (outputLevel_ >= 4))
        {
#ifdef HAVE_PYTHON
          //**/ Display a dialog box asking whether to continue
          if (yesno_dialog != NULL) 
          {
            PyObject* temp;
	    temp = PyObject_CallFunction( yesno_dialog, "ss",
		     "No Progress", "No progress has been made.\n"
		     "Something may be wrong.\n\n"
                     "Do you want to proceed anyway?" );
            //**/ If the answer is no, stop immediately
            if ((temp != NULL) && (PyObject_Not(temp) == 1)) 
            {
              Py_DECREF(temp);
              //throw Psuade_Stop_Exception();
              printf("INFO: terminate.\n");
              exit(0);
            }
	    else if (temp != NULL) Py_DECREF(temp);
          }
#else
	  //**/ Ask whether to continue on the console
	  printOutTS(PL_INFO, 
               "The following message is for precautionary purpose:\n");
	  printOutTS(PL_INFO, 
               "No progress has been made. Something may be wrong if\n");
	  printOutTS(PL_INFO, 
               "your job should take less than a second to run, and\n");
	  printOutTS(PL_INFO, 
               "in that case you should answer no to the following\n");
          printOutTS(PL_INFO, 
               "question. Otherwise, answer yes.\n");
          printOutTS(PL_INFO, 
               "You can turn off this checking by setting diagnostics\n");
          printOutTS(PL_INFO, 
               "level to less than 4. You can avoid this complaint by\n");
          printOutTS(PL_INFO, 
               "setting the minimum wait time to be your estimated\n");
          printOutTS(PL_INFO, "time per run.\n");
          printf("Do you want to proceed? (y or n) ");
          scanf("%s", winput);
          if (winput[0] == 'n') 
          {
            //**/throw Psuade_Stop_Exception();
            printf("INFO: terminate.\n");
            exit(0);
          }
#endif
          askFlag = 0;
        }
      }
    }
    if (noAnalysis)
      printf("PSUADE run: samples not run since there is no driver.\n");

    //**/ -------------------------------------------------------------
    //**/ limited_launch_only, stop here.
    //**/ -------------------------------------------------------------
    if (launchOnly && (jobsCompleted < nSamples))
    {
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO,"PSUADE run: terminate - launch only mode.\n");
      break;
    }

    //**/ -------------------------------------------------------------
    //**/ perform importance and other analyses
    //**/ (if no driver, will turn on refinement)
    //**/ -------------------------------------------------------------
    if (noAnalysis == 0)
    {
      refineFlag = analysisManager_.analyze(psuadeIO_,iR+1,
                                 vecRefineNSamps.getIVector(),-1);
    }
    else refineFlag = 1;

    //**/ -------------------------------------------------------------
    //**/ find minima in the design space (if found, status > 0)
    //**/ -------------------------------------------------------------
    status = 0;
    if (optSwitch == 1)
    {
      int nInitX = 0;
      psMatrix matOptData;
      psuadeIO_->getParameter("ana_opt_nlocalmin", pPtr);
      matOptData.setFormat(PS_MAT2D);
      matOptData.setDim(4, pPtr.intData_*nInputs);
      status = OptimizerSearch(psuadeIO_,funcIO,matOptData,nInitX);
    }
    if (status > 0) refineFlag = 0;

    //**/ -------------------------------------------------------------
    //**/ refine 
    //**/ -------------------------------------------------------------
    if ((refineFlag == 1) && (nRefinements > 0) && (iR < nRefinements))
    {
      //**/ sampleErrors needed in adaptive 
      double *sampleErrors = analysisManager_.getSampleErrors();
      refineRatio = 2;
      refineThreshold = analysisThreshold;
      sampler_->refine(refineRatio,randomize,refineThreshold,nSamples,
                       sampleErrors);

      //**/ create new arrays of sample inputs and outputs

      oldNSamples = nSamples;
      nSamples = sampler_->getNumSamples();
      if (nSamples == oldNSamples)
      {
        refineFlag = 0;
        break;
      }
      psVector  vecTX, vecTY;
      psIVector vecTS;
      vecTX.setLength(nSamples*nInputs);
      vecTY.setLength(nSamples*nOutputs);
      vecTS.setLength(nSamples);
            
      //**/ get the new set of sample inputs
      sampler_->getSamples(nSamples,nInputs,nOutputs,vecTX.getDVector(), 
                   vecTY.getDVector(), vecTS.getIVector());
      nReUsed = 0;
      for (sampleID = 0; sampleID < nSamples; sampleID++)
        if (vecTS[sampleID] == 1) nReUsed++;
      printOutTS(PL_INFO,
           "PSUADE run: refinement - number reused = %d(out of %d)\n",
           nReUsed, oldNSamples);

      psuadeIO_->updateMethodSection(-1,nSamples,-1,nRefinements-iR-1,-1);
      psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                        vecTX.getDVector(),NULL,NULL,NULL,NULL,NULL);
      psuadeIO_->updateOutputSection(nSamples,nOutputs,vecTY.getDVector(),
                                     vecTS.getIVector(),NULL);
      psuadeIO_->writePsuadeFile(NULL,0);
    }
    if (outputLevel_ > -1) 
    {
      if (nRefinements > 0) 
        printOutTS(PL_INFO, 
             "PSUADE run: refinements completed = %d (out of %d)\n",
             iR, nRefinements);
      printEquals(PL_INFO, 0);
    }
    if (refineFlag == 0) break;
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up 
  //**/ ----------------------------------------------------------------
  delete funcIO;
  if (outputLevel_ >= 4) printOutTS(PL_INFO, "PSUADE run: exiting...\n");
  return 0;
}

// ************************************************************************
// run the sample points as an ensemble
// ------------------------------------------------------------------------
int PsuadeBase::runEnsemble()
{
  int     iR, ii, iteration, sampleID, status, nReUsed, refineFlag;
  int     oldNSamples, refineRatio, parallelJobCount;
  double  refineThreshold;
  pData   pPtr, pAppFiles, pStates, pInpData, pOutData;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printf("INFO: You have turned on ensemble_run_mode. In this mode, the\n");
  printf("      ensemble_driver in the APPLICATION section will be used.\n");
  printf("      Your ensemble_driver executable will be run via\n");
  printf("           ensemble_driver psuadeEval.in psuadeEval.out\n");
  printf("      PSUADE writes to psuadeEval.in in the following format:\n");
  printf("      line 1: <nSamples>\n");
  printf("      line 2: parameter values for sample point 1\n");
  printf("      line 3: parameter values for sample point 2\n");
  printf("      .....\n\n");
  printf("      Your ensemble_driver is expected to write the sample\n");
  printf("      output values to the psuadeEval.out file.\n");
  printf("      To change the size of each ensemble, change the\n");
  printf("      max_parallel_jobs variable in the APPLICATION section.\n");
  printEquals(PL_INFO, 0);
  if (outputLevel_ >= 4) 
    printOutTS(PL_DETAIL,"PSUADE::ensemble run begins.\n");
  if (psuadeIO_ == NULL)
  {
    printOutTS(PL_ERROR,"PSUADE run ERROR - no PsuadeData object.\n");
    return 0;
  }

  //**/ -------------------------------------------------------------
  //**/ if generate input files only mode, just generate the files
  //**/ -------------------------------------------------------------
  psuadeIO_->getParameter("app_runtype", pPtr);
  int runType = pPtr.intData_;
  if ((runType & 16) == 1)
  {
    printOutTS(PL_INFO,"INFO: PSUADE is in ensemble run mode.\n");
    printOutTS(PL_INFO,"      All other run modes will be disabled.\n");
    printOutTS(PL_INFO,"      Optimization will be disabled.\n");
  }

  //**/ -------------------------------------------------------------
  //**/ if no driver found, see if all sample points have been run
  //**/ -------------------------------------------------------------
  psuadeIO_->getParameter("app_files", pAppFiles);
  char *ensembleDriver = pAppFiles.strArray_[5];
  if (!strcmp(ensembleDriver, "NONE"))
  {
    printOutTS(PL_ERROR,"PSUADE ERROR: no ensemble driver given.\n");
    return 0;
  }

  //**/ -------------------------------------------------------------
  //**/ get parameters from psuadeIO for experiment and analysis
  //**/ -------------------------------------------------------------
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_randomflag", pPtr);
  int randomize = pPtr.intData_;
  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;
  psuadeIO_->getParameter("ana_threshold", pPtr);
  double analysisThreshold = pPtr.dbleData_;
  psuadeIO_->getParameter("output_states", pStates);
  int *sampleStates = pStates.intArray_;
  int count = 0;
  for (sampleID = 0; sampleID < nSamples; sampleID++)
    if (sampleStates[sampleID] == 1) count++;
  if ((count == nSamples) && (nRefinements == 0))
  {
    printOutTS(PL_INFO,
         "PSUADE INFO: all sample points have been evaluated.\n");
    return 0;
  }
  if (outputLevel_ > 0) 
    printOutTS(PL_INFO,"PSUADE run: creating interface to user driver.\n");
  FunctionInterface *funcIO = createFunctionInterface(psuadeIO_);

  //**/ -------------------------------------------------------------
  //**/ set up analysis tools
  //**/ -------------------------------------------------------------
  analysisManager_.clearAnalyzers();
  analysisManager_.setup(psuadeIO_);

  //**/ -------------------------------------------------------------
  //**/ set up general run parameters
  //**/ -------------------------------------------------------------
  funcIO->setOutputLevel(outputLevel_);

  //**/ -------------------------------------------------------------
  //**/ loop over number of refinements 
  //**/ -------------------------------------------------------------
  double *sampleInputs=NULL, *sampleOutputs=NULL;
  psIVector vecRefineNSamps;
  psVector  vecXT, vecYT;
  vecRefineNSamps.setLength(nRefinements+1);
  vecXT.setLength(maxParallelJobs*nInputs);
  vecYT.setLength(maxParallelJobs*nOutputs);
  for (iR = 0; iR < nRefinements+1; iR++)
  {
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 0) 
    {
      printEquals(PL_INFO, 0);
      if (nRefinements > 0)
        printOutTS(PL_INFO,
             "PSUADE run: refinement %d(out of %d), nSamples = %d\n",
             iR, nRefinements, nSamples);
      else
        printOutTS(PL_INFO,"PSUADE run: running sample, nSamples = %d \n",
                   nSamples);
    }

    //**/ ----------------------------------------------------------
    //**/ get run information for this refinement step (from sampler)
    //**/ and transform to the corresponding distribution functions
    //**/ ----------------------------------------------------------
    vecRefineNSamps[iR] = nSamples;
    psuadeIO_->getParameter("input_sample", pInpData);
    sampleInputs = pInpData.dbleArray_;
    psuadeIO_->getParameter("output_sample", pOutData);
    sampleOutputs = pOutData.dbleArray_;
    psuadeIO_->getParameter("output_states", pStates);
    sampleStates = pStates.intArray_;
    psuadeIO_->getParameter("method_nsamples", pPtr);
    nSamples = pPtr.intData_;

    //**/ ----------------------------------------------------------
    //**/ find out how many jobs have been completed (job state=1
    //**/ or if any output is undefined)
    //**/ ----------------------------------------------------------
    int jobsCompleted = 0;
    for (sampleID = 0; sampleID < nSamples; sampleID++) 
    {
      if (sampleStates[sampleID] == 1) jobsCompleted++;
      else                             sampleStates[sampleID] = 0;
      for (ii = 0; ii < nOutputs; ii++) 
        if (sampleOutputs[sampleID*nOutputs+ii] >= PSUADE_UNDEFINED)
          break;
      if ((ii != nOutputs) && (sampleStates[sampleID] == 1))
      {
        jobsCompleted--;
        sampleStates[sampleID] = 0;
      }
    }

    //**/ ----------------------------------------------------------
    //**/ run all samples
    //**/ ----------------------------------------------------------
    iteration = 0;
    while (jobsCompleted < nSamples)
    {
      printOutTS(PL_INFO,
                 "PSUADE ensemble run begins (parallelism = %d)\n",
                 maxParallelJobs);
      while (jobsCompleted < nSamples)
      {
        iteration++;
        parallelJobCount = 0;
        for (sampleID = 0; sampleID < nSamples; sampleID++)
        {
          if ((sampleStates[sampleID] == 0) && 
              (parallelJobCount < maxParallelJobs))
          {
            for (ii = 0; ii < nInputs; ii++)
              vecXT[parallelJobCount*nInputs+ii] =
                        sampleInputs[sampleID*nInputs+ii];
            parallelJobCount++;
          }
        }
        status = funcIO->ensembleEvaluate(parallelJobCount,nInputs,
                vecXT.getDVector(),nOutputs,vecYT.getDVector(),iteration);
        parallelJobCount = 0;
        for (sampleID = 0; sampleID < nSamples; sampleID++)
        {
          if ((sampleStates[sampleID] == 0) && 
              (parallelJobCount < maxParallelJobs))
          {
            for (ii = 0; ii < nOutputs; ii++)
              sampleOutputs[sampleID*nOutputs+ii] = 
                         vecYT[parallelJobCount*nOutputs+ii];
            sampleStates[sampleID] = 1; 
            parallelJobCount++;
            sampler_->storeResult(sampleID, nOutputs,
                                &(sampleOutputs[sampleID*nOutputs]),
                                &(sampleStates[sampleID]));
            jobsCompleted++;
          }
        }
      }
      printOutTS(PL_INFO,"PSUADE ensemble run completed.\n");

      //**/ -------------------------------------------------
      //**/ save the result
      //**/ -------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                     sampleStates,NULL); 
      psuadeIO_->writePsuadeFile(NULL,0);
    }

    //**/ -------------------------------------------------------------
    //**/ perform analyses
    //**/ -------------------------------------------------------------
    refineFlag = analysisManager_.analyze(psuadeIO_,iR+1,
                                      vecRefineNSamps.getIVector(),-1);

    //**/ -------------------------------------------------------------
    //**/ refine 
    //**/ -------------------------------------------------------------
    if ((refineFlag == 1) && (nRefinements > 0) && (iR < nRefinements))
    {
      //**/ sampleErrors needed in adaptive 
      double *sampleErrors = analysisManager_.getSampleErrors();
      refineRatio = 2;
      refineThreshold = analysisThreshold;
      sampler_->refine(refineRatio,randomize,refineThreshold,nSamples,
                       sampleErrors);

      //**/ create new arrays of sample inputs and outputs
      oldNSamples = nSamples;
      nSamples = sampler_->getNumSamples();
      if (nSamples == oldNSamples)
      {
        refineFlag = 0;
        break;
      }
      psVector  vecTX, vecTY;
      psIVector vecTS;
      vecTX.setLength(nSamples*nInputs);
      vecTY.setLength(nSamples*nOutputs);
      vecTS.setLength(nSamples);
         
      //**/ get the new set of sample inputs
      sampler_->getSamples(nSamples,nInputs,nOutputs,vecTX.getDVector(), 
                           vecTY.getDVector(),vecTS.getIVector());
      nReUsed = 0;
      for (sampleID = 0; sampleID < nSamples; sampleID++)
        if (sampleStates[sampleID] == 1) nReUsed++;
      printOutTS(PL_INFO,
           "PSUADE refinement: nSamples reused = %d (out of %d)\n",
            nReUsed, oldNSamples);

      psuadeIO_->updateMethodSection(-1,nSamples,-1,nRefinements-iR-1,-1);
      psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                       vecTX.getDVector(),NULL,NULL,NULL,NULL,NULL);
      psuadeIO_->updateOutputSection(nSamples,nOutputs,vecTY.getDVector(),
                                     vecTS.getIVector(),NULL);
      psuadeIO_->writePsuadeFile(NULL,0);
    }
    if (outputLevel_ > -1) 
    {
      if (nRefinements > 0) 
        printOutTS(PL_INFO,
             "PSUADE run: refinements completed = %d (out of %d)\n",
             iR, nRefinements);
      printEquals(PL_INFO, 0);
    }
    if (refineFlag == 0) break;
  }

  //**/ ----------------------------------------------------------------
  //**/ clean up 
  //**/ ----------------------------------------------------------------
  delete funcIO;
  if (outputLevel_ >= 4) printOutTS(PL_INFO,"PSUADE run: exiting...\n");
  return 0;
}

// ************************************************************************
// run the special adaptive mode for response surface analysis 
// It works only with METIS sampling (if specified otherwise, it will be
// switched to METIS)
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptiveNN()
{
  int    ss, ii, jj, ivar1, ivar2, iOne=1, initFlag, currNJobs, nJobsDiff;
  int    parallelJobCount, status, maxState, jobsCompletedLast, marsMode=0;
  int    lastNSamples, nPtsPerDim=64, length, tstNInputs, tstNOutputs;
  double errAvg, refineThreshold=1.0, errMax, dtemp, errRMS, totalSum;
  char   systemCommand[100],cString[100],winput[500],*targv[6],sparam[501];
  pData  pPtr, pLowerB, pUpperB, pPDFs, pTstInputs, pTstOutputs;
  FILE   *fp;
  FuncApprox        *faPtr=NULL;
  FunctionInterface *funcIO=NULL;
  PsuadeData        *tstIO=NULL;
  psVector  vecSamInps, vecSamOuts;
  psIVector vecSamStas;
  psMatrix  matMarsX, matMarsY;

  //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ (This function is intended for uniform distribution only)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_pdfs", pPDFs);
  int *inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  int iSum = 0;
  for (ss = 0; ss < nInputs; ss++) iSum += inputPDFs[ss];
  if (iSum)
  {
    printOutTS(PL_ERROR, 
         "PSUADE ERROR: adaptiveRSM does not currently allow \n");
    printOutTS(PL_ERROR, 
         "       non-uniform probability distributions to be\n");
    printOutTS(PL_ERROR, "       defined in the INPUT SECTION.\n");
    printOutTS(PL_ERROR, "       Please fix it and then run again.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking and correcting
  //**/ (This function is intended for METIS design only)
  //**/ (If current sample is not METIS, generate a new one)
  //**/ ----------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,
    "PSUADE adaptiveNN: This adaptive sampling uses only METIS design.\n");
  printOutTS(PL_INFO,
    "This method performs adaptive sampling based on the difference (i.e.\n");
  printOutTS(PL_INFO,
    "gradient) between the sample in each METIS cell with that the its\n");
  printOutTS(PL_INFO,
    "neighbors. Higher scores are given to METIS cells with the largest\n");
  printOutTS(PL_INFO,
    "differences (i.e. max-max difference). The cells with the highest\n");
  printOutTS(PL_INFO,
    "scores are selected for bisection to create new sample points.\n"); 
  printOutTS(PL_INFO,
    "This method uses MARS to predict cell outputs.\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,"Note: Turn on rs_expert mode to set RS parameters.\n");
  printOutTS(PL_INFO,"      Turn on outputLevel>0 to get error histogram.\n");
  printEquals(PL_INFO,0);
  psuadeIO_->getParameter("method_refine_type", pPtr);
  int refineType = pPtr.intData_;
  psuadeIO_->getParameter("method_sampling", pPtr);
  int samMethod = pPtr.intData_;
  if (samMethod != PSUADE_SAMP_METIS)
  {
    printOutTS(PL_INFO,"PSUADE adaptiveNN: sampling defaulted to METIS.\n");
    samMethod = PSUADE_SAMP_METIS; 
    psuadeIO_->updateMethodSection(samMethod,-1,-1,-1,-1);
    initFlag = 1;
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = NULL;
  }
  else initFlag = 0;

  //**/ ----------------------------------------------------------------
  //**/ get parameters from psuadeIO for experiment and analysis
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  if (nOutputs > 1)
  {
    printOutTS(PL_INFO, "PSUADE adaptiveNN: nOutputs should be 1.\n");
    return 0;
  }
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  int refineSize = pPtr.intData_;

  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;
  psuadeIO_->getParameter("ana_rstype", pPtr);
  int rstype = pPtr.intData_;
  psuadeIO_->getParameter("ana_threshold", pPtr);
  double anaThreshold = pPtr.dbleData_;

  //**/ ----------------------------------------------------------------
  //**/ ask users if there is a test set
  //**/ ----------------------------------------------------------------
  int    tstNSamples = 0;
  double *tstSamInputs=NULL, *tstSamOutputs=NULL;
  printOutTS(PL_INFO,"You may test the quality of the response surface\n");
  printOutTS(PL_INFO,"using a test sample (in psuadeData format).\n");
  snprintf(cString,100,"Use a test sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter the test sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    tstIO = new PsuadeData();
    status = tstIO->readPsuadeFile(winput);
    if (status != 0) 
    {
      printOutTS(PL_ERROR,
           "PSUADE adaptiveNN ERROR: cannot read file %s or wrong format.\n",
            winput);
      exit(1);
    }
    tstIO->getParameter("method_nsamples", pPtr);
    tstNSamples = pPtr.intData_;
    tstIO->getParameter("input_ninputs", pPtr);
    tstNInputs = pPtr.intData_;
    tstIO->getParameter("output_noutputs", pPtr);
    tstNOutputs = pPtr.intData_;
    if (tstNInputs != nInputs)
    {
      printOutTS(PL_ERROR, 
           "PSUADE adaptiveNN ERROR : test sample nInputs != %d\n",
           nInputs);
      delete tstIO;
      return -1;
    }
    if (tstNOutputs > 1)
    {
      printOutTS(PL_INFO,
                 "PSUADE adaptiveNN ERROR: test sample nOutputs != 1.\n");
      delete tstIO;
      return -1;
    }
    tstIO->getParameter("input_sample", pTstInputs);
    tstSamInputs = pTstInputs.dbleArray_;
    tstIO->getParameter("output_sample", pTstOutputs);
    tstSamOutputs = pTstOutputs.dbleArray_;
  }
  int numMars = 50;
  if (psConfig_.RSExpertModeIsOn())
  {
    if (rstype == PSUADE_RS_MARSB)
    {
      snprintf(cString,100,"Number of MARS (default = 100, >2, <502) = ");
      numMars = getInt(3, 501, cString);
      snprintf(cString,100,"Use mean (0) or median (1) of MarsBag : ");
      marsMode = getInt(0, 1, cString);
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ generate initial sample, if needed
  //**/ refineType = 0: uniform refinement (when adaptive not set)
  //**/ for reliability, refinement size should be unlimited
  //**/ ----------------------------------------------------------------
  if (initFlag == 1)
  {
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samMethod); 
    sampler_->setPrintLevel(outputLevel_);
    sampler_->setInputBounds(nInputs, iLowerB, iUpperB);
    sampler_->setOutputParams(nOutputs);
    sampler_->setSamplingParams(nSamples, 1, 1);
    sampler_->initialize(0);
    nSamples = sampler_->getNumSamples();
    psuadeIO_->updateMethodSection(-1,nSamples,-1,-1,-1);
  }
  //**/ refineType can be set to be uniform (default) or adaptive (1)
  if (refineType == 0)
  {
    strcpy(sparam, "setUniformRefinement");
    refineSize = 100000;
  }
  else strcpy(sparam,"setAdaptiveRefinementBasedOnOutputs");
  sampler_->setParam(sparam);
  //**/ if uniform, unlimited refineSize
  //**/ if adaptive, refineSize is user-specified
  if (refineSize > 0)
  {
    snprintf(sparam,100,"setRefineSize %d", refineSize);
    sampler_->setParam(sparam);
    printOutTS(PL_INFO, "PSUADE adaptiveNN: refineSize = %d\n",refineSize);
  }
  int refineRatio = 2;
  int randomize = 1;

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  funcIO = createFunctionInterface(psuadeIO_);

  //**/ ----------------------------------------------------------------
  //**/ set up general run parameters
  //**/ ----------------------------------------------------------------
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ ----------------------------------------------------------------
  //**/ load the initial samples for MARS
  //**/ ----------------------------------------------------------------
  nSamples = sampler_->getNumSamples();
  length   = nSamples + nRefinements*refineSize;
  matMarsX.setFormat(PS_MAT2D);
  matMarsY.setFormat(PS_MAT2D);
  matMarsX.setDim(numMars,length*nInputs);
  matMarsY.setDim(numMars,length);

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  nSamples = -1;
  int jobsCompleted = 0;
  int refineIndex   = 0;
  int marsNSamples  = 0;

  while (1)
  {
    //**/ ----------------------------------------------------------------
    //**/ get the sample data, perform the necessary transforms
    //**/ ----------------------------------------------------------------
    lastNSamples = nSamples;
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 1)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, 
             "PSUADE adaptiveNN: current level    = %d (of %d)\n",
             refineIndex, nRefinements);
      printOutTS(PL_INFO, 
           "PSUADE adaptiveNN: current nSamples = %d\n",nSamples);
    }
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    double *dInps = vecSamInps.getDVector();
    double *dOuts = vecSamOuts.getDVector();
    int    *iStas = vecSamStas.getIVector();
    sampler_->getSamples(nSamples,nInputs,nOutputs,vecSamInps.getDVector(), 
                         vecSamOuts.getDVector(),vecSamStas.getIVector());
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                  vecSamInps.getDVector(),NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,
                 vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) if (vecSamStas[ss]==0) currNJobs++;
    if (psConfig_.AnaExpertModeIsOn() && (jobsCompleted < currNJobs))
    {
      printOutTS(PL_INFO, 
           "A sample has been created in the psuadeData file. \n");
      printOutTS(PL_INFO, 
           "At this point you can choose to run your model with\n");
      printOutTS(PL_INFO, 
           "this sample via psuade or by yourself (if the model\n");
      printOutTS(PL_INFO, 
           "is expensive to run, you want to choose the latter).\n");
      snprintf(systemCommand,100,"Run the model yourself (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        printOutTS(PL_INFO, 
          "You have chosen to run the sample yourself.\n");
        printOutTS(PL_INFO, 
          "The following steps are for ARSM refinements:\n");
        printOutTS(PL_INFO, 
          "(1) Rename psuadeData to something else. \n");
        printOutTS(PL_INFO, 
          "(2) Comment out num_refinements in this file.\n");
        printOutTS(PL_INFO, 
          "(3) Change sampling method to be MC in this file.\n");
        printOutTS(PL_INFO, 
          "(4) Comment out analysis method = ARSM in this file.\n");
        printOutTS(PL_INFO, 
          "(5) Run the sample in this file and collect outputs.\n");
        printOutTS(PL_INFO, 
          "(6) Restore num_refinements in this file.\n");
        printOutTS(PL_INFO, 
          "(7) Restore sampling method = METIS in this file.\n");
        printOutTS(PL_INFO, 
          "(8) Restore analysis method = ARSM in this file.\n");
        printOutTS(PL_INFO, 
          "(9) Finally, restart psuade with this file.\n");
        delete funcIO;
        return 0;
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          if (temp != NULL) Py_DECREF(temp);
        }
#endif
        //**/ If anything has set the flag psuadeStop_ to 1, stop now
        if (psuadeStop_ == 1)
        {
          psuadeIO_->writePsuadeFile(NULL,0);
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
          //throw Psuade_Stop_Exception();
        }

        if ((vecSamStas[ss] == 0) && 
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &dInps[ss*nInputs], 
                             nOutputs, &dOuts[ss*nOutputs],0);   

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dOuts[ss*nOutputs]),
                                  &(iStas[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            if ((nJobsDiff % saveFrequency) == 0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                           vecSamOuts.getDVector(),vecSamStas.getIVector(),
                           NULL);
              psuadeIO_->writePsuadeFile(NULL,0);
            }
            if (outputLevel_ > 0)
            {
              if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
                fflush(stdout);
            }
          }
          else
          {
            vecSamStas[ss] = status;
            parallelJobCount++;
          }
        }
        else if (vecSamStas[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &(dInps[ss*nInputs]), 
                              nOutputs, &(dOuts[ss*nOutputs]),2);   
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dOuts[ss*nOutputs]),
                                  &(iStas[ss]));
            jobsCompleted++;
            parallelJobCount--;
          }
          else vecSamStas[ss]++;

          if ((minJobWaitTime > 0) &&
              ((vecSamStas[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
            vecSamStas[ss] = 0;
            parallelJobCount--;
            ss--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                 "PSUADE adaptiveNN: sample %6d to be restarted.\n",ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,
                  vecSamOuts.getDVector(), vecSamStas.getIVector(),NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > maxState) maxState = vecSamStas[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > 10) vecSamStas[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO, 
             "\nPSUADE adaptiveNN: jobs completed = %d(of %d)\n",
             jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ response surface analysis
    //**/ ----------------------------------------------------------------
    errRMS = anaThreshold;
    if (lastNSamples > 0)
    {
      printOutTS(PL_INFO,"PSUADE adaptiveNN: response surface analysis.\n");
      printOutTS(PL_INFO,
             "   construct response surface with %d sample points.\n",
             lastNSamples);
      printOutTS(PL_INFO,
             "   test response surface with previous %d sample points.\n",
             nSamples-lastNSamples);
      faPtr = genFA(rstype, nInputs, iOne, lastNSamples);
      if (faPtr == NULL)
      {
        printOutTS(PL_ERROR,"ERROR: cannot create function approximator.\n");
        exit(1);
      }
      faPtr->setNPtsPerDim(nPtsPerDim);
      faPtr->setBounds(iLowerB, iUpperB);
      faPtr->setOutputLevel(outputLevel_);
      if (!psConfig_.RSExpertModeIsOn() &&
          (rstype == PSUADE_RS_MARS || rstype == PSUADE_RS_MARSB))
      {
        strcpy(cString, "mars_params");
        targv[0] = (char *) cString;
        ivar1 = lastNSamples;
        targv[1] = (char *) &ivar1;
        ivar2 = 2 * nInputs / 3 + 1;
        targv[2] = (char *) &ivar2;
        faPtr->setParams(3, targv);
        if (rstype == PSUADE_RS_MARSB)
        {
          strcpy(cString, "num_mars");
          targv[0] = (char *) cString;
          targv[1] = (char *) &numMars;
          faPtr->setParams(2, targv);
          if (marsMode == 1)
          {
            strcpy(cString, "median");
            targv[0] = (char *) cString;
            faPtr->setParams(1, targv);
          }
        }
      }
      faPtr->initialize(vecSamInps.getDVector(),vecSamOuts.getDVector());
      errMax = errAvg = errRMS = 0.0;
      totalSum = 0.0;
      for (ss = lastNSamples; ss < nSamples; ss++)
      {
        //**/ evaluate
        double outData = faPtr->evaluatePoint(&dInps[ss*nInputs]);
        if (outputLevel_ > 3)
        {
          printOutTS(PL_INFO,
               "Data %5d : predicted = %12.4e, (actual) = %12.4e,",
               ss, outData, vecSamOuts[ss]);
          printOutTS(PL_INFO,
               " diff = %12.4e\n",outData-vecSamOuts[ss]);
        }
        totalSum += PABS(vecSamOuts[ss]);
        dtemp   = outData - vecSamOuts[ss];
        errAvg += dtemp;
        errRMS += (dtemp * dtemp);
        dtemp   = PABS(dtemp);
        errMax = (dtemp > errMax) ? dtemp : errMax;
      }
      totalSum /= (double) (nSamples - lastNSamples);;
      errRMS    = sqrt(errRMS/(nSamples-lastNSamples));
      errAvg    = errAvg / (nSamples-lastNSamples);
      printOutTS(PL_INFO, 
            "     response surface unscaled max error = %e\n",errMax);
      printOutTS(PL_INFO, 
            "     response surface   scaled max error = %e\n",
            errMax/totalSum);
      printOutTS(PL_INFO, 
            "     response surface unscaled rms error = %e\n",errRMS);
      printOutTS(PL_INFO, 
            "     response surface   scaled rms error = %e\n",
            errRMS/totalSum);
      printOutTS(PL_INFO, 
            "     response surface unscaled avg error = %e\n",errAvg);
      printOutTS(PL_INFO, 
            "     response surface   scaled avg error = %e\n",
            errAvg/totalSum);
      delete faPtr;
    }
    if (tstNSamples > 0)
    {
      printEquals(PL_INFO, 0);
      faPtr = genFA(rstype, nInputs, iOne, nSamples);
      if (faPtr == NULL)
      {
	printOutTS(PL_ERROR,
             "function genFA returned NULL in file %s line %d, exiting\n", 
             __FILE__, __LINE__);
        exit(1);
      }
      faPtr->setNPtsPerDim(nPtsPerDim);
      faPtr->setBounds(iLowerB, iUpperB);
      faPtr->setOutputLevel(outputLevel_);

      if ((!psConfig_.RSExpertModeIsOn()) &&
          (rstype == PSUADE_RS_MARS || rstype == PSUADE_RS_MARSB))
      {
        strcpy(cString, "mars_params");
        targv[0] = (char *) cString;
        targv[1] = (char *) &nSamples;
        ivar2 = 2 * nInputs / 3 + 1;
        targv[2] = (char *) &ivar2;
        faPtr->setParams(3, targv);
        if (rstype == PSUADE_RS_MARSB)
        {
          strcpy(cString, "num_mars");
          targv[0] = (char *) cString;
          targv[1] = (char *) &numMars;
          faPtr->setParams(2, targv);
          if (marsMode == 1)
          {
            strcpy(cString, "median");
            targv[0] = (char *) cString;
            faPtr->setParams(1, targv);
          }
          //**/ prepare data
          for (ii = 0; ii < numMars; ii++)
          {
            for (ss = 0; ss < nSamples-marsNSamples; ss++)
            {
              if (marsNSamples == 0)
                   ivar1 = PSUADE_rand() % (nSamples-marsNSamples);
              else ivar1 = ss;
              ivar2 = ivar1 + marsNSamples;
              for (jj = 0; jj < nInputs; jj++)
                matMarsX.setEntry(ii,(marsNSamples+ss)*nInputs+jj,
                                  vecSamInps[ivar2*nInputs+jj]);
              matMarsY.setEntry(ii,marsNSamples+ss,vecSamOuts[ivar2]);
            }
          }
          marsNSamples = nSamples;
          strcpy(cString, "mars_sample");
          targv[0] = (char *) cString;
          targv[2] = (char *) &marsNSamples;
          double **dX = matMarsX.getMatrix2D();
          double **dY = matMarsX.getMatrix2D();
          for (ii = 0; ii < numMars; ii++)
          {
            targv[1] = (char *) &ii;
            targv[3] = (char *) dX[ii];
            targv[4] = (char *) dY[ii];
            faPtr->setParams(5, targv);
          }
        }
      }
      faPtr->initialize(vecSamInps.getDVector(),
                        vecSamOuts.getDVector());
      psVector vecTstOuts;
      vecTstOuts.setLength(tstNSamples);
      faPtr->evaluatePoint(tstNSamples,tstSamInputs,
                           vecTstOuts.getDVector());
      totalSum = errMax = errAvg = errRMS =0.0;
      for (ss = 0; ss < tstNSamples; ss++)
      {
        totalSum += PABS(vecTstOuts[ss]);
        dtemp   = vecTstOuts[ss] - tstSamOutputs[ss];
        errAvg += dtemp;
        errRMS  += dtemp * dtemp;
        dtemp   = PABS(dtemp);
        errMax  = (dtemp > errMax) ? dtemp : errMax;
      }
      errRMS = sqrt(errRMS / tstNSamples);
      totalSum /= (double) tstNSamples;
      errAvg  = errAvg / tstNSamples;
      printOutTS(PL_INFO,
            "     test sample RS unscaled max error = %e\n",errMax);
      printOutTS(PL_INFO,
            "     test sample RS   scaled max error = %e\n",
            errMax/totalSum);
      printOutTS(PL_INFO,
            "     test sample RS unscaled rms error = %e\n",errRMS);
      printOutTS(PL_INFO,
            "     test sample RS   scaled rms error = %e\n",
            errRMS/totalSum);
      printOutTS(PL_INFO,
            "     test sample RS unscaled avg error = %e\n",errAvg);
      printOutTS(PL_INFO,
            "     test sample RS   scaled avg error = %e\n",
            errAvg / totalSum);
      if (errRMS < anaThreshold || refineIndex >= nRefinements)
      {
        snprintf(cString,100,"arsm_nn_err.m");
        fp = fopen(cString, "w");
        if (fp != NULL)
        {
          fprintf(fp, "%% inputs, true outputs, predicted outputs\n");
          fprintf(fp, "A = [\n");
          for (ss = 0; ss < tstNSamples; ss++)
          {
            for (ii = 0; ii < nInputs; ii++)
              fprintf(fp, "%e ", tstSamInputs[ss*nInputs+ii]);
            fprintf(fp, "%e %e\n", tstSamOutputs[ss], vecTstOuts[ss]);
          }
          fprintf(fp, "];\n");
          fwritePlotCLF(fp);
          fprintf(fp, "m = %d;\n", nInputs);
          fprintf(fp, "X1 = A(:,1);\n");
          fprintf(fp, "X2 = A(:,2);\n");
          fprintf(fp, "Y  = A(:,m+1) - A(:,m+2);\n");
          fprintf(fp, "subplot(2,2,1)\n");
          fprintf(fp, "plot3(X1,X2,Y,'*','markerSize',13')\n");
          fwritePlotXLabel(fp, "Input 1");
          fwritePlotYLabel(fp, "Input 2");
          fwritePlotZLabel(fp, "Output");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp, "Errors w.r.t. Input 1 and 2");
          fprintf(fp, "subplot(2,2,2)\n");
          fprintf(fp, "plot(X1,Y,'*','markerSize',13')\n");
          fwritePlotXLabel(fp, "Input 1");
          fwritePlotYLabel(fp, "Output");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp, "Output vs Input 1");
          fprintf(fp, "subplot(2,2,4)\n");
          fprintf(fp, "plot(X2,Y,'*','markerSize',13')\n");
          fwritePlotXLabel(fp, "Input 2");
          fwritePlotYLabel(fp, "Output");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp, "Output vs Input 2");
          fprintf(fp, "AA = [\n");
          for (ss = 0; ss < nSamples; ss++)
          {
            for (ii = 0; ii < nInputs; ii++)
              fprintf(fp, "%e ", vecSamInps[ss*nInputs+ii]);
            fprintf(fp, "\n");
          }
          fprintf(fp, "];\n");
          fprintf(fp, "XX1 = AA(:,1);\n");
          fprintf(fp, "XX2 = AA(:,2);\n");
          fprintf(fp, "subplot(2,2,3)\n");
          fprintf(fp, "plot(XX1,XX2,'*','markerSize',13')\n");
          fwritePlotXLabel(fp, "Input 1");
          fwritePlotYLabel(fp, "Input 2");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp, "Input 1 vs 2 (original sample)");
          printOutTS(PL_INFO, "PSUADE adaptiveNN: error plots are in %s.\n",
                     cString);
          fclose(fp);
        }
      }
      delete faPtr;
    }

    //**/ ----------------------------------------------------------------
    //**/ update the sample data in the sampler
    //**/ ----------------------------------------------------------------
    refineIndex++;
    sampler_->loadSamples(nSamples,nInputs,nOutputs,vecSamInps.getDVector(), 
                          vecSamOuts.getDVector(), vecSamStas.getIVector());
    if (errRMS < anaThreshold)
    {
      printOutTS(PL_INFO,
           "PSUADE adaptiveNN: threshold reached (using unscaled rms).\n");
      break;
    }
    if (psConfig_.AnaExpertModeIsOn())
    {
      snprintf(systemCommand,100,"Do you want to quit now (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        break;
      }
    }
    if (refineIndex > nRefinements) break;

    //**/ ----------------------------------------------------------------
    //**/ request refinement 
    //**/ ----------------------------------------------------------------
    sampler_->refine(refineRatio,randomize,refineThreshold,0,NULL);
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete funcIO;
  if (tstIO != NULL) delete tstIO;
  return 0;
}

#if 1
// ************************************************************************
// run the special adaptive mode for response surface analysis 
// using MARS with bagging as response surface when METIS sampling is 
// specified (otherwise it will called ErrBasedG).
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptiveErrBased1()
{
  //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ (This function is intended for uniform distribution only)
  //**/ ----------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"PSUADE adaptive(1): This method uses ");
  printOutTS(PL_INFO,"adaptive sampling to create new\n");
  printOutTS(PL_INFO,
    "       new samples. Users are to first create a METIS sample (a\n");
  printOutTS(PL_INFO,
    "       space-filling design) of size N. Subsequently, the sample\n");
  printOutTS(PL_INFO,
    "       is run with the user-provided simulator to generate sample\n");
  printOutTS(PL_INFO,
    "       outputs, which are then used to create a response surface.\n");
  printOutTS(PL_INFO,
    "       (NOTE: this response surface should also estimate prediction\n");
  printOutTS(PL_INFO,
    "       errors.) The original METIS sample is then refined adaptively\n");
  printOutTS(PL_INFO,
    "       based on searching the design space for points that give the\n");
  printOutTS(PL_INFO,
    "       largest prediction errors.\n");
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"To be able to exercise more control of this method,\n");
  printOutTS(PL_INFO,"you can turn on rs_expert or ana_expert modes in\n");
  printOutTS(PL_INFO,"the PSUADE input file.\n");
  printOutTS(PL_INFO,"Turn on outputLevel (>0) to get more diagnostics.\n");
  printEquals(PL_INFO, 0);

  pData pPtr, pPDFs;
  psuadeIO_->getParameter("input_pdfs", pPDFs);
  int *inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  int ss, ii, iSum = 0;
  for (ss = 0; ss < nInputs; ss++) iSum += inputPDFs[ss];
  if (iSum)
  {
    printOutTS(PL_ERROR,
       "PSUADE ERROR: adaptive(1) does not currently allow non-uniform\n");
    printOutTS(PL_ERROR,
       "       probability distribution be defined in the INPUT section.\n");
    printOutTS(PL_ERROR,"       Please fix it and then run again.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ Users can choose random adaptivity for comparison purposes
  //**/ ----------------------------------------------------------------
  int  useRandomPts = 0, useRS=0;
  char cString[100], winput[100];

  printEquals(PL_INFO,0);
  printOutTS(PL_INFO,"Adaptive sampling options:\n");
  printOutTS(PL_INFO, 
    "1. Random selection of new aggregates (i.e. not based on errors)\n"); 
  printOutTS(PL_INFO, 
    "   (This option is provided for users to demonstrate that smart\n");
  printOutTS(PL_INFO, 
    "    sampling performs better.)\n"); 
  printOutTS(PL_INFO, "2. Error-based selection of new aggregates\n");
  printOutTS(PL_INFO,
    "   (This option performs smart sampling by first identifying a good\n");
  printOutTS(PL_INFO,
    "    aggregate, then splitting the aggregate, and finally selecting\n");
  printOutTS(PL_INFO, 
    "    a cell (each aggregate has many cells clustered together) as the\n");
  printOutTS(PL_INFO, 
    "    new sample point (mid-point of SOME RANDOM CELL IN THE AGGREGATE).\n");
  printOutTS(PL_INFO, 
    "3. Error-based selection of new aggregates\n");
  printOutTS(PL_INFO,
    "   (This option performs smart sampling by first identifying a good\n");
  printOutTS(PL_INFO,
    "    aggregate, then splitting the aggregate, and finally selecting\n");
  printOutTS(PL_INFO, 
    "    a cell (each aggregate has many cells clustered together) as the\n");
  printOutTS(PL_INFO, 
    "    new sample point (mid-point of THE CELL THAT HAS MAXIMUM ERRORS).\n");
  printOutTS(PL_INFO, 
    "    As such it requires interpolation for each cell and so expensive.\n");
  ii = 0;
  snprintf(cString,100,"Which option ? (1 - 3) ");
  while (ii <= 0 || ii > 3) ii = getInt(1,3,cString);
  if      (ii == 1) useRandomPts = 1;
  else if (ii == 3) useRS = 1;

  //**/ ----------------------------------------------------------------
  //**/ error checking and correcting
  //**/ (This function is intended for METIS design only)
  //**/ (If current sample is not METIS, generate a new one)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_refine_type", pPtr);
  int refineType = pPtr.intData_, initFlag=0;
  psuadeIO_->getParameter("method_sampling", pPtr);
  int samMethod = pPtr.intData_;
  if (samMethod != PSUADE_SAMP_METIS)
  {
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): sampling defaulted to METIS.\n");
    samMethod = PSUADE_SAMP_METIS; 
    psuadeIO_->updateMethodSection(samMethod,-1,-1,-1,-1);
    initFlag = 1;
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = NULL;
  }
  else initFlag = 0;

  //**/ ----------------------------------------------------------------
  //**/ get input/output/method parameters from psuadeIO 
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  if (nOutputs > 1)
  {
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1) ERROR: nOutputs should be equal to 1.\n");
    return 0;
  }
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  int refineSize = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ ask users if there is a test set
  //**/ ----------------------------------------------------------------
  int    tstNSamples=0, tstNInputs, tstNOutputs, status;
  double *tstSamInputs=NULL, *tstSamOutputs=NULL;
  pData      pTstInputs, pTstOutputs;
  PsuadeData *tstIO=NULL;

  printOutTS(PL_INFO,"You may test the quality of the response ");
  printOutTS(PL_INFO,"surface using a test sample\n");
  printOutTS(PL_INFO,"in PSUADE data format.\n");
  snprintf(cString,100,"Use a test sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter the test sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    tstIO = new PsuadeData();
    status = tstIO->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR,
         "PSUADE adaptive(1) ERROR: cannot read file %s or wrong format.\n",
         winput);
      exit(1);
    }
    tstIO->getParameter("method_nsamples", pPtr);
    tstNSamples = pPtr.intData_;
    tstIO->getParameter("input_ninputs", pPtr);
    tstNInputs = pPtr.intData_;
    tstIO->getParameter("output_noutputs", pPtr);
    tstNOutputs = pPtr.intData_;
    if (tstNInputs != nInputs)
    {
      printOutTS(PL_ERROR,
         "PSUADE adaptive(1) ERROR : test sample nInputs != %d\n",nInputs);
      return 0;
    }
    if (tstNOutputs > 1)
    {
      printOutTS(PL_INFO,
           "PSUADE adaptive(1) ERROR: test sample nOutputs != 1.\n");
      return 0;
    }
    tstIO->getParameter("input_sample", pTstInputs);
    tstSamInputs = pTstInputs.dbleArray_;
    tstIO->getParameter("output_sample", pTstOutputs);
    tstSamOutputs = pTstOutputs.dbleArray_;
    delete tstIO;
  }

  //**/ ----------------------------------------------------------------
  //**/ Users can choose to add another set of sample points
  //**/ ----------------------------------------------------------------
  int    auxNSamples=0, auxNInputs, auxNOutputs, length;
  double *auxSamInputs=NULL, *auxSamOutputs=NULL;
  pData  pAuxInputs, pAuxOutputs;
  PsuadeData *auxIO=NULL;

  printOutTS(PL_INFO,"You may add to the base METIS sample an ");
  printOutTS(PL_INFO,"auxiliary PRE-RUN sample.\n");
  printOutTS(PL_INFO,"The purpose may be, for example, to cover ");
  printOutTS(PL_INFO,"the corners using Factorial\n");
  printOutTS(PL_INFO,"or Fractional Factorial samples.\n");
  snprintf(cString,100,"Add an auxiliary sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter auxiliary sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    auxIO = new PsuadeData();
    status = auxIO->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR,
         "PSUADE adaptive(1) ERROR: cannot read file %s ",winput);
      printOutTS(PL_ERROR," or file in wrong format.\n");
      exit(1);
    }
    auxIO->getParameter("method_nsamples", pPtr);
    auxNSamples = pPtr.intData_;
    auxIO->getParameter("input_ninputs", pPtr);
    auxNInputs = pPtr.intData_;
    auxIO->getParameter("output_noutputs", pPtr);
    auxNOutputs = pPtr.intData_;
    if (auxNInputs != nInputs)
    {
      printOutTS(PL_ERROR,
      "PSUADE adaptive(1) ERROR : auxiliary nInputs != %d\n",nInputs);
      return 0;
    }
    if (auxNOutputs > 1)
    {
      printOutTS(PL_INFO,
           "PSUADE adaptive(1) ERROR: auxiliary sample nOutputs != 1.\n");
      return 0;
    }
    auxIO->getParameter("input_sample", pAuxInputs);
    auxSamInputs = pAuxInputs.dbleArray_;
    double *dAuxInps = auxSamInputs;
    length = (nSamples+auxNSamples+nRefinements*refineSize)*nInputs;
    auxSamInputs = new double[length];
    for (ss = 0; ss < auxNSamples*nInputs; ss++)
      auxSamInputs[ss] = dAuxInps[ss];
    auxIO->getParameter("output_sample", pAuxOutputs);
    length = (nSamples+auxNSamples+nRefinements*refineSize)*nOutputs;
    auxSamOutputs = new double[length];
    for (ss = 0; ss < auxNSamples; ss++)
      auxSamOutputs[ss] = pAuxOutputs.dbleArray_[ss];
    for (ss = 0; ss < auxNSamples; ss++)
    {
      if (auxSamOutputs[ss] == 1e35)
      {
        printOutTS(PL_INFO,"PSUADE adaptive(1) ERROR: some output in ");
        printOutTS(PL_INFO,"the auxiliary sample are\n");
        printOutTS(PL_INFO,"undefined.\n");
        return 0;
      }
    }
  }

  //**/ ----------------------------------------------------------------
  //**/ get other parameters from psuadeIO 
  //**/ ----------------------------------------------------------------
  pData  pLowerB, pUpperB;
  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;

  printDashes(PL_INFO, 0);
  printf("Four response surface options are available:\n");
  printf("0. Polynomial regression\n");
  printf("1. MARS with bootstrapping (recommended ");
  printf("for discontinuous functions)\n");
  printf("2. Gaussian process (not as good as MARSB)\n");
  printf("3. Kriging (not as good as MARSB)\n");
  snprintf(cString,100,"Which option ? (0 - 3) ");
  int rstype = getInt(0, 3, cString); 
  if      (rstype == 0) rstype = PSUADE_RS_REGR2; 
  else if (rstype == 1) rstype = PSUADE_RS_MARSB; 
  else if (rstype == 2) rstype = PSUADE_RS_GP3; 
  else if (rstype == 3) rstype = PSUADE_RS_KR; 

  int numMars=100, marsMode=0;
  if (psConfig_.RSExpertModeIsOn())
  {
    if (rstype == PSUADE_RS_MARSB)
    {
      snprintf(cString,100,"Number of MARS (default = 100, >2, <502) = ");
      numMars = getInt(3, 501, cString);
      snprintf(cString,100,"Use mean (0) or median (1) of MarsBag : ");
      marsMode = getInt(0, 1, cString);
    }
  }
  int polyOrder=2;
  if (rstype == PSUADE_RS_REGR2)
  {
    snprintf(cString,100,"polynomial order ? (1 - 4) = ");
    polyOrder = getInt(1, 4, cString);
    if      (polyOrder == 1) rstype = PSUADE_RS_REGR1;
    else if (polyOrder == 2) rstype = PSUADE_RS_REGR2;
    else if (polyOrder == 3) rstype = PSUADE_RS_REGR3;
    else if (polyOrder == 4) rstype = PSUADE_RS_REGR4;
  }
  psuadeIO_->getParameter("ana_threshold", pPtr);
  double anaThreshold = pPtr.dbleData_;

  //**/ ----------------------------------------------------------------
  //**/ generate initial sample, if needed
  //**/ refineType = 0: uniform refinement (when adaptive not set)
  //**/ for reliability, refinement size should be unlimited
  //**/ ----------------------------------------------------------------
  if (initFlag == 1)
  {
    //**/ if not METIS to begin with, create METIS
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samMethod); 
    sampler_->setPrintLevel(outputLevel_);
    sampler_->setInputBounds(nInputs, iLowerB, iUpperB);
    sampler_->setOutputParams(nOutputs);
    sampler_->setSamplingParams(nSamples, 1, 1);
    sampler_->initialize(0);
    nSamples = sampler_->getNumSamples();
    psuadeIO_->updateMethodSection(-1,nSamples,-1,-1,-1);
  }
  //**/ refineType can be set to be uniform (default) or adaptive (1)
  char sparam[501];
  if (refineType == 0)
  {
    strcpy(sparam, "setUniformRefinement");
    refineSize = 100000;
  }
  else strcpy(sparam, "setAdaptiveRefinementBasedOnErrors");
  sampler_->setParam(sparam);
  //**/ if uniform, it is the same as adaptive with unlimited refineSize
  //**/ if adaptive, refineSize is user-specified
  if (refineSize > 0)
  {
    snprintf(sparam,100,"setRefineSize %d", refineSize);
    sampler_->setParam(sparam);
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): refineSize = %d\n",refineSize);
  }
  int refineRatio = 2;

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  FunctionInterface *funcIO = createFunctionInterface(psuadeIO_);

  //**/ ----------------------------------------------------------------
  //**/ set up general run parameters
  //**/ ----------------------------------------------------------------
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ ----------------------------------------------------------------
  //**/ load the initial samples for MARS
  //**/ ----------------------------------------------------------------
  nSamples = sampler_->getNumSamples();
  psMatrix matMarsDataX, matMarsDataY;
  matMarsDataX.setFormat(PS_MAT2D);
  matMarsDataY.setFormat(PS_MAT2D);
  length = (nSamples+auxNSamples+nRefinements*refineSize);
  matMarsDataX.setDim(numMars, length*nInputs);
  matMarsDataY.setDim(numMars, length);
  for (ii = 0; ii < numMars; ii++)
  {
    for (ss = 0; ss < auxNSamples*nInputs; ss++)
      matMarsDataX.setEntry(ii,ss,auxSamInputs[ss]);
    for (ss = 0; ss < auxNSamples; ss++)
      matMarsDataY.setEntry(ii,ss,auxSamOutputs[ss]);
  }
  double **marsDataX = matMarsDataX.getMatrix2D();
  double **marsDataY = matMarsDataY.getMatrix2D();

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  int jobsCompleted = 0, jobsCompletedLast = 0, nPtsPerDim=64, iOne=1;
  int refineIndex  = 0, currNJobs, nJobsDiff, parallelJobCount, jj;
  int marsNSamples = auxNSamples, nSamples2, maxState, ivar1, ivar2;
  double refineThreshold=1.0, dtemp, totalSum, errMax, errAvg, errL2;
  char   sysCmd[100], *targv[6];
  FILE   *fp=NULL;
  FuncApprox *faPtr=NULL;
  psVector  vecSamInps, vecSamOuts, vecSamStds;
  psIVector vecSamStas;

  while (1)
  {
    //**/ ----------------------------------------------------------------
    //**/ fetch the sample from sampler, and write to a file (unevaluated)
    //**/ ----------------------------------------------------------------
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 1)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, 
         "PSUADE adaptive(1): current level    = %d (of %d)\n",
         refineIndex, nRefinements);
      printOutTS(PL_INFO, 
         "PSUADE adaptive(1): current nSamples = %d\n", nSamples);
    }
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    sampler_->getSamples(nSamples,nInputs,nOutputs,
                vecSamInps.getDVector(),vecSamOuts.getDVector(), 
                vecSamStas.getIVector());

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) 
      if (vecSamStas[ss]==0) currNJobs++;
    if (psConfig_.AnaExpertModeIsOn() && (jobsCompleted < currNJobs))
    {
      printOutTS(PL_INFO,"A sample has been created in the psuadeData ");
      printOutTS(PL_INFO,"file. At this point you\n");
      printOutTS(PL_INFO,"can choose to run your model with this sample ");
      printOutTS(PL_INFO,"by PSUADE or by yourself\n");
      printOutTS(PL_INFO,"(if the model is expensive to run, you may ");
      printOutTS(PL_INFO,"want to choose the latter).\n");
      snprintf(sysCmd,100,"Run the model yourself (y or n) ? ");
      getString(sysCmd, cString);
      if (cString[0] == 'y')
      {
        printOutTS(PL_INFO,
           "You have chosen to run the sample yourself.\n");
        printOutTS(PL_INFO, 
          "The following are the steps to continue ARSM refinements:\n");
        printOutTS(PL_INFO, 
          "(1) Rename psuadeData to something else (e.g. psData). \n");
        printOutTS(PL_INFO, 
          "(2) Run the sample in this file and collect outputs.\n");
        printOutTS(PL_INFO, 
          "(3) Replace the outputs in psData (do not change others).\n");
        printOutTS(PL_INFO, 
          "(4) Finally, restart psuade with this file (psData).\n");
        delete funcIO;
        return 0;
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    double *dInps = vecSamInps.getDVector();
    double *dOuts = vecSamOuts.getDVector();
    double *dStds;
    int    *iStas = vecSamStas.getIVector();
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          if (temp != NULL) Py_DECREF(temp);
        }
#endif
	//**/
	//**/ If anything has set the flag psuadeStop_ to 1, stop now
	//**/
        if (psuadeStop_ == 1)
        {
          psuadeIO_->writePsuadeFile(NULL,0);
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
          //throw Psuade_Stop_Exception();
        }

        if ((vecSamStas[ss] == 0) && (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &(dInps[ss*nInputs]), 
                            nOutputs,&(dOuts[ss*nOutputs]),0);   
          if (outputLevel_ > 2)
            printf("PSUADE adaptive(1): job submitted = %d, status = %d\n",
                   ss+1, status);

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dOuts[ss*nOutputs]),
                                  &(iStas[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            if ((nJobsDiff % saveFrequency) == 0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                           vecSamOuts.getDVector(),vecSamStas.getIVector(),
                           NULL);
              psuadeIO_->writePsuadeFile(NULL,0);
            }
            if (outputLevel_ > 0)
            {
               if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
               fflush(stdout);
            }
          }
          else
          {
            vecSamStas[ss] = status;
            parallelJobCount++;
          }
        }
        else if (vecSamStas[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &(dInps[ss*nInputs]), 
                              nOutputs, &(dOuts[ss*nOutputs]),2);   
          if (outputLevel_ > 2)
            printf("PSUADE adaptive(1): job checked = %d, status = %d\n",
                   ss+1, status);
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dOuts[ss*nOutputs]),
                                  &(iStas[ss]));
            jobsCompleted++;
            parallelJobCount--;
          }
          else vecSamStas[ss]++;

          if ((minJobWaitTime > 0) &&
              ((vecSamStas[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
            vecSamStas[ss] = 0;
            parallelJobCount--;
            ss--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                "PSUADE adaptive(1): sample %6d to be restarted.\n",ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,
                   vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > maxState) maxState = vecSamStas[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > 10) vecSamStas[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO, 
               "PSUADE adaptive(1): jobs completed = %d(of %d)\n",
               jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ response surface construction
    //**/ ----------------------------------------------------------------
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): response surface analysis.\n");
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): current level     = %d (of %d)\n",
           refineIndex, nRefinements);
    printOutTS(PL_INFO, "                    training nSamples = %d\n",
           nSamples + auxNSamples);

    faPtr = genFA(rstype, nInputs, iOne, nSamples+auxNSamples);
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ prepare MarsBag samples
    if (rstype == PSUADE_RS_MARSB)
    {
      for (ii = 0; ii < numMars; ii++)
      {
        for (ss = 0; ss < nSamples-marsNSamples+auxNSamples; ss++)
        {
          if (marsNSamples == auxNSamples)
            ivar1 = PSUADE_rand() % (nSamples-marsNSamples+auxNSamples);
          else ivar1 = ss;
          for (jj = 0; jj < nInputs; jj++)
            marsDataX[ii][(marsNSamples+ss)*nInputs+jj] =
               vecSamInps[(ivar1+marsNSamples-auxNSamples)*nInputs+jj];
          marsDataY[ii][marsNSamples+ss] = 
               vecSamOuts[ivar1+marsNSamples-auxNSamples];
        }
      }
      marsNSamples = nSamples + auxNSamples;

      strcpy(cString, "mars_params");
      targv[0] = (char *) cString;
      ivar1 = (nSamples + auxNSamples);
      targv[1] = (char *) &ivar1;
      ivar2 = 2 * nInputs / 3 + 1;
      targv[2] = (char *) &ivar2;
      faPtr->setParams(3, targv);
      strcpy(cString, "num_mars");
      targv[0] = (char *) cString;
      targv[1] = (char *) &numMars;
      faPtr->setParams(2, targv);
      if (marsMode == 1)
      {
        strcpy(cString, "median");
        targv[0] = (char *) cString;
        faPtr->setParams(1, targv);
      }
      strcpy(cString, "mars_sample");
      targv[0] = (char *) cString;
      targv[2] = (char *) &marsNSamples;
      for (ii = 0; ii < numMars; ii++)
      {
        targv[1] = (char *) &ii;
        targv[3] = (char *) marsDataX[ii];
        targv[4] = (char *) marsDataY[ii];
        faPtr->setParams(5, targv);
      }
    }
    faPtr->initialize(vecSamInps.getDVector(),vecSamOuts.getDVector());
    
    //**/ ----------------------------------------------------------------
    //**/ if there is a test set, compute its error statistics
    //**/ using the response surface from last batch
    //**/ ----------------------------------------------------------------
    if (tstNSamples > 0)
    {
      printEquals(PL_INFO, 0);
      psVector vecTstOuts;
      vecTstOuts.setLength(tstNSamples);
      faPtr->evaluatePoint(tstNSamples,tstSamInputs,vecTstOuts.getDVector());
      totalSum = errMax = errAvg = errL2 =0.0;
      for (ss = 0; ss < tstNSamples; ss++)
      {
        totalSum += PABS(vecTstOuts[ss]);
        dtemp = vecTstOuts[ss] - tstSamOutputs[ss];
        errAvg += dtemp;
        dtemp = PABS(dtemp);
        errL2  += dtemp * dtemp;
        errMax = (dtemp > errMax) ? dtemp : errMax;
      }
      errL2 = sqrt(errL2 / tstNSamples);
      totalSum /= (double) tstNSamples;
      errAvg  = errAvg / tstNSamples;
      printOutTS(PL_INFO,"RS Analysis of Test Sample with Current Sample:\n"); 
      printOutTS(PL_INFO,"Sample size of test sample = %d\n", tstNSamples);
      printOutTS(PL_INFO, 
        "     Test sample RS unscaled max error = %e\n",errMax);
      printOutTS(PL_INFO, 
        "     Test sample RS   scaled max error = %e\n",errMax/totalSum);
      printOutTS(PL_INFO, 
        "     Test sample RS unscaled rms error = %e\n",errL2);
      printOutTS(PL_INFO, 
        "     Test sample RS   scaled rms error = %e\n",errL2/totalSum);
      printOutTS(PL_INFO, 
        "     Test sample RS unscaled avg error = %e\n",errAvg);
      printOutTS(PL_INFO, 
        "     test sample RS   scaled avg error = %e\n",errAvg/totalSum);
      //**/if (errL2 < anaThreshold || refineIndex >= nRefinements)
      if (outputLevel_ > 0)
      {
        if (plotScilab())
          snprintf(cString,100,"arsm_testset_prediction_errors.sci");
        else
          snprintf(cString,100,"arsm_testset_prediction_errors.m");
        fp = fopen(cString, "w");
        strcpy(winput, "inputs, true outputs, predicted outputs");
        fwriteComment(fp, winput);
        fwritePlotCLF(fp);
        fprintf(fp, "A = [\n");
        for (ss = 0; ss < tstNSamples; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%e ", tstSamInputs[ss*nInputs+ii]);
          fprintf(fp, "%e %e\n", tstSamOutputs[ss], vecTstOuts[ss]);
        }
        fprintf(fp, "];\n");
        fprintf(fp, "m = %d;\n", nInputs);
        fprintf(fp, "X1 = A(:,1);\n");
        fprintf(fp, "X2 = A(:,2);\n");
        fprintf(fp, "Y  = A(:,m+1) - A(:,m+2);\n");
        fprintf(fp, "subplot(2,2,1)\n");
        if (plotScilab())
             fprintf(fp, "scatter3(X1,X2,Y,'*')\n");
        else fprintf(fp, "plot3(X1,X2,Y,'*','markersize',13)\n");
        fwritePlotXLabel(fp, "Input 1");
        fwritePlotYLabel(fp, "Input 2");
        fwritePlotZLabel(fp, "Errors");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Errors wrt Input 1 and 2");
        fprintf(fp, "subplot(2,2,2)\n");
        if (plotScilab())
             fprintf(fp, "plot(X1,Y,'*')\n");
        else fprintf(fp, "plot(X1,Y,'*','markersize',13)\n");
        fwritePlotXLabel(fp, "Input 1");
        fwritePlotYLabel(fp, "Errors");
        fwritePlotAxes(fp); 
        fwritePlotTitle(fp, "Errors vs Input 1");
        fprintf(fp, "subplot(2,2,4)\n");
        if (plotScilab())
             fprintf(fp, "plot(X2,Y,'*')\n");
        else fprintf(fp, "plot(X2,Y,'*','markersize',13)\n");
        fwritePlotXLabel(fp, "Input 2");
        fwritePlotYLabel(fp, "Errors");
        fwritePlotAxes(fp); 
        fwritePlotTitle(fp, "Errors vs Input 2");
        fprintf(fp, "AA = [\n");
        for (ss = 0; ss < nSamples; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%e ", vecSamInps[ss*nInputs+ii]);
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        fprintf(fp, "XX1 = AA(:,1);\n");
        fprintf(fp, "XX2 = AA(:,2);\n");
        fprintf(fp, "subplot(2,2,3)\n");
        fprintf(fp, "plot(XX1,XX2,'*')\n");
        fwritePlotXLabel(fp, "Input 1");
        fwritePlotYLabel(fp, "Input 2");
        fwritePlotAxes(fp); 
        fwritePlotTitle(fp, "Input 1 vs 2 (original sample)");
        printOutTS(PL_INFO, "PSUADE adaptive(1): error plots are in %s.\n",
               cString);
        fclose(fp);
      }
    }
    printEquals(PL_INFO, 0);

    //**/ ----------------------------------------------------------------
    //**/ check termination is reached
    //**/ ----------------------------------------------------------------
    refineIndex++;
    if (refineIndex > nRefinements)
    {
      printOutTS(PL_INFO, 
           "PSUADE adaptive(1): number of refinements %d reached.\n",
           nRefinements);
      delete faPtr;
      break;
    }

    //**/ ----------------------------------------------------------------
    //**/ creating sample errors if useRS == 0
    //**/ ----------------------------------------------------------------
    psVector vecSamErrors;
    if (useRS == 0)
    {
      //**/ --------------------------------------------------------------
      //**/ refine one level uniformly
      //**/ --------------------------------------------------------------
      Sampling *samplerAux = (Sampling *) SamplingCreateFromID(samMethod);
      samplerAux->setInputBounds(nInputs, iLowerB, iUpperB);
      samplerAux->setPrintLevel(outputLevel_);
      samplerAux->setOutputParams(nOutputs);
      samplerAux->setSamplingParams(nSamples, -1, -1);
      samplerAux->initialize(1);
      samplerAux->loadSamples(nSamples, nInputs, nOutputs, 
                   vecSamInps.getDVector(), vecSamOuts.getDVector(), 
                   vecSamStas.getIVector());
      //**/ tell sampler to use psuadeMetisInfo.tmp instead
      strcpy(sparam, "changeInfoName");
      samplerAux->setParam(sparam);
      strcpy(sparam, "setUniformRefinement");
      samplerAux->setParam(sparam);
      printOutTS(PL_INFO, 
         "PSUADE adaptive(1): uniform refinement.\n");
      samplerAux->refine(refineRatio,iOne,0,0,NULL);
      nSamples2 = samplerAux->getNumSamples();
      if (nSamples2 != 2 * nSamples)
      {
        printOutTS(PL_INFO,"PSUADE adaptive(1): Castastropic error.\n");
        printOutTS(PL_INFO,
           "       refined sample size != 2 * original size\n");
        printOutTS(PL_INFO,"       Please consult developers.\n");
        exit(1);
      }
      psVector  vecSamInps2, vecSamOuts2;
      psIVector vecSamStas2;
      vecSamInps2.setLength(nSamples2*nInputs);
      vecSamOuts2.setLength(nSamples2*nOutputs);
      vecSamStas2.setLength(nSamples2);
      vecSamErrors.setLength(nSamples2);
      samplerAux->getSamples(nSamples2, nInputs, nOutputs, 
            vecSamInps2.getDVector(), vecSamOuts2.getDVector(), 
            vecSamStas2.getIVector());
      //**/ evaluate the unevaluated new points
      double *dInps2 = vecSamInps2.getDVector();
      double *dOuts2 = vecSamOuts2.getDVector();
      faPtr->evaluatePointFuzzy(nSamples2-nSamples,&(dInps2[nInputs*nSamples]), 
                         &(dOuts2[nSamples]), vecSamErrors.getDVector());
    }

    //**/ ----------------------------------------------------------------
    //**/ refinement
    //**/ ----------------------------------------------------------------
    if (useRS == 1)
    {
      targv[0] = (char *) faPtr;
      strcpy(cString, "setRSPTR");
      targv[0] = (char *) cString;
      targv[1] = (char *) faPtr;
      ii = 2;
      sampler_->setParam(ii, targv);
    }
    printOutTS(PL_INFO, 
         "PSUADE adaptive(1): calling refine function.\n");
    sampler_->refine(refineRatio,useRandomPts,refineThreshold,nSamples,
                     vecSamErrors.getDVector());
    printOutTS(PL_INFO, 
         "PSUADE adaptive(1): returned from refine function.\n");

    //**/ ----------------------------------------------------------------
    //**/ fetch the new points and compute its error statistics
    //**/ using the response surface from last batch
    //**/ ----------------------------------------------------------------
    int nSamplesOld = nSamples;
    nSamples = sampler_->getNumSamples();
    vecSamInps.setLength(nSamples*nInputs);
    vecSamOuts.setLength(nSamples*nOutputs);
    vecSamStas.setLength(nSamples);
    vecSamStds.setLength(nSamples);
    sampler_->getSamples(nSamples, nInputs, nOutputs, 
               vecSamInps.getDVector(), vecSamOuts.getDVector(), 
               vecSamStas.getIVector());
    dInps = vecSamInps.getDVector();
    dOuts = vecSamOuts.getDVector();
    iStas = vecSamStas.getIVector();
    dStds = vecSamStds.getDVector();
    faPtr->evaluatePointFuzzy(nSamples-nSamplesOld,
            &(dInps[nInputs*nSamplesOld]), &(dOuts[nSamplesOld]), 
            &(dStds[nSamplesOld]));

    //**/ compute statistics (on s.d.) of the new points
    totalSum = errMax = errAvg = errL2 =0.0;
    for (ss = nSamplesOld; ss < nSamples; ss++)
    {
      totalSum += PABS(vecSamStds[ss]);
      errMax = (PABS(vecSamStds[ss])>errMax) ? 
                PABS(vecSamStds[ss]) : errMax;
      errAvg += vecSamStds[ss];
      errL2  += pow(vecSamStds[ss], 2.0e0);
    }
    errL2 = sqrt(errL2 / (nSamples - nSamplesOld));
    totalSum /= (nSamples - nSamplesOld);
    errAvg  = sqrt(errAvg/(nSamples-nSamplesOld));
    printOutTS(PL_INFO,"RS Analysis of Newly Created Refined Sample:\n"); 
    printOutTS(PL_INFO, 
       "     Refined sample RS unscaled max error = %e\n",errMax);
    printOutTS(PL_INFO, 
       "     Refined sample RS scaled   max error = %e\n",errMax/totalSum);
    printOutTS(PL_INFO, 
       "     Refined sample RS unscaled rms error = %e\n",errL2);
    printOutTS(PL_INFO, 
       "     Refined sample RS scaled   rms error = %e\n",errL2/totalSum);
    printOutTS(PL_INFO, 
       "     Refined sample RS unscaled avg error = %e\n",errAvg);
    printOutTS(PL_INFO,
       "     Refined sample RS scaled   avg error = %e\n",errAvg/totalSum);

    //**/ ----------------------------------------------------------------
    //**/ update and convergence check
    //**/ ----------------------------------------------------------------
    sampler_->loadSamples(nSamples, nInputs, nOutputs, 
                 vecSamInps.getDVector(), vecSamOuts.getDVector(), 
                 vecSamStas.getIVector());
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                vecSamInps.getDVector(),NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,
                vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL);
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);

    if (errL2 < anaThreshold)
    {
      printOutTS(PL_INFO,"PSUADE adaptive(1): threshold reached.\n");
      printOutTS(PL_INFO,"                    unscaled rms = %en",errL2);
      printOutTS(PL_INFO, 
              "                    threshold    = %e\n", anaThreshold);
      delete faPtr;
      break;
    }
    if (psConfig_.AnaExpertModeIsOn())
    {
      //**/snprintf(sysCmd,100,"Want to quit now (y or n) ? ");
      //**/getString(sysCmd, cString);
      cString[0] = 'n';
      if (cString[0] == 'y')
      {
        delete faPtr;
        break;
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ clean up 
    //**/ ----------------------------------------------------------------
    delete faPtr;
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete funcIO;
  if (auxIO != NULL)
  {
    if (auxSamInputs  != NULL) delete [] auxSamInputs;
    if (auxSamOutputs != NULL) delete [] auxSamOutputs;
    delete auxIO;
  }
  matMarsDataX.clean();
  matMarsDataY.clean();
  return 0;
}
#endif

#if 0
// ************************************************************************
// obsolete
// ************************************************************************
// run the special adaptive mode for response surface analysis 
// using MARS with bagging as response surface when METIS sampling is 
// specified (otherwise it will called ErrBasedG).
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptiveErrBased1()
{
  int    ii, jj, ss, initFlag,
  int    loopFlag, currNJobs, nJobsDiff, status, maxState;
  int    parallelJobCount, jobsCompletedLast, rstype, length, iOne=1;
  int    nPtsPerDim=64, ivar1, ivar2, useRandomPts=0, nSamples2;
  int    auxNSamples=0, auxNInputs, auxNOutputs, tstNInputs, tstNOutputs;
  double refineThreshold=1.0, dtemp, errMax=0.0, errAvg=0.0, errL2=0.0;
  double anaThreshold, totalSum=0.0;
  double *auxSamInputs=NULL, *auxSamOutputs=NULL, *tstOutputs;
  char   systemCommand[100], cString[100], winput[100], *targv[6];
  char   sparam[501];
  FILE   *fp;
  pData  pPtr, pLowerB, pUpperB, pPDFs, pAuxInputs, pAuxOutputs;
  pData  pTstInputs, pTstOutputs;
  FuncApprox *faPtr=NULL;
  Sampling   *samplerAux=NULL;
  PsuadeData *auxIO=NULL;
  FunctionInterface *funcIO=NULL;
  psVector  vecSamInps, vecSamOuts;
  psIVector vecSamStas;

  //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ (This function is intended for uniform distribution only)
  //**/ ----------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,
    "PSUADE adaptive(1): Use METIS with MarsBagging/GP/Kriging.\n");
  printOutTS(PL_INFO,
    "This method performs adaptive sampling based on prediction errors at\n");
  printOutTS(PL_INFO,
    "each METIS cell using MARSBag/GP/Kriging prediction error analysis.\n");
  printOutTS(PL_INFO,
    "The cells with largest prediction errors are selected for refinement.\n");
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,"To be able to exercise more control of this method,\n");
  printOutTS(PL_INFO,"you can turn on interactive and/or expert modes in\n");
  printOutTS(PL_INFO,"the PSUADE input file.\n");
  printOutTS(PL_INFO,"Turn on outputLevel (>0) to get error histogram.\n");
  printOutTS(PL_INFO,"Turn on interactive to have stepwise control.\n");
  printEquals(PL_INFO, 0);

  psuadeIO_->getParameter("input_pdfs", pPDFs);
  int *inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  int iSum = 0;
  for (ss = 0; ss < nInputs; ss++) iSum += inputPDFs[ss];
  if (iSum)
  {
    printOutTS(PL_ERROR,
       "PSUADE ERROR: adaptive(1) does not currently allow\n");
    printOutTS(PL_ERROR,
       "       non-uniform probability distribution to be\n");
    printOutTS(PL_ERROR,"       defined in the INPUT SECTION.\n");
    printOutTS(PL_ERROR,"       Please fix it and then run again.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ Users can choose random adaptivity for comparison purposes
  //**/ ----------------------------------------------------------------
  printOutTS(PL_INFO, 
    "The following option allows comparing this prediction error-based\n");
  printOutTS(PL_INFO, 
    "strategy with mere random adaptivity. To run this prediction error-\n");
  printOutTS(PL_INFO, "based strategy, answer 'n' below'.\n");
  snprintf(cString,100,"Use random points for refinement ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y') useRandomPts = 1;

  //**/ ----------------------------------------------------------------
  //**/ error checking and correcting
  //**/ (This function is intended for METIS design only)
  //**/ (If current sample is not METIS, generate a new one)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_refine_type", pPtr);
  int refineType = pPtr.intData_;
  psuadeIO_->getParameter("method_sampling", pPtr);
  int samMethod = pPtr.intData_;
  if (samMethod != PSUADE_SAMP_METIS)
  {
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): sampling defaulted to METIS.\n");
    samMethod = PSUADE_SAMP_METIS; 
    psuadeIO_->updateMethodSection(samMethod,-1,-1,-1,-1);
    initFlag = 1;
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = NULL;
  }
  else initFlag = 0;

  //**/ ----------------------------------------------------------------
  //**/ get input/output/method parameters from psuadeIO 
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  if (nOutputs > 1)
  {
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1) ERROR: nOutputs should be 1.\n");
    return 0;
  }
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  int refineSize = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ ask users if there is a test set
  //**/ ----------------------------------------------------------------
  int    tstNSamples=0;
  double *tstSamInputs=NULL, *tstSamOutputs=NULL;
  PsuadeData *tstIO=NULL;
  printOutTS(PL_INFO,"You may test the quality of the response surface\n");
  printOutTS(PL_INFO,"using a test sample (in psuadeData format).\n");
  snprintf(cString,100,"Use a test sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter the test sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    tstIO = new PsuadeData();
    status = tstIO->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR,
         "PSUADE adaptive(1) ERROR: cannot read file %s or wrong format.\n",
         winput);
      exit(1);
    }
    tstIO->getParameter("method_nsamples", pPtr);
    tstNSamples = pPtr.intData_;
    tstIO->getParameter("input_ninputs", pPtr);
    tstNInputs = pPtr.intData_;
    tstIO->getParameter("output_noutputs", pPtr);
    tstNOutputs = pPtr.intData_;
    if (tstNInputs != nInputs)
    {
      printOutTS(PL_ERROR,
         "PSUADE adaptive(1) ERROR : test sample nInputs != %d\n",nInputs);
      return 0;
    }
    if (tstNOutputs > 1)
    {
      printOutTS(PL_INFO,
           "PSUADE adaptive(1) ERROR: test sample nOutputs != 1.\n");
      return 0;
    }
    tstIO->getParameter("input_sample", pTstInputs);
    tstSamInputs = pTstInputs.dbleArray_;
    tstIO->getParameter("output_sample", pTstOutputs);
    tstSamOutputs = pTstOutputs.dbleArray_;
  }

  //**/ ----------------------------------------------------------------
  //**/ Users can choose to add another set of sample points
  //**/ ----------------------------------------------------------------
  printOutTS(PL_INFO,
       "You may add to the base sample an auxiliary sample which\n");
  printOutTS(PL_INFO,
       "covers the corners (for example, factorial or fractional\n");
  printOutTS(PL_INFO,"factorial).\n");
  snprintf(cString,100,"Add an auxiliary sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter auxiliary sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    auxIO = new PsuadeData();
    status = auxIO->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR,
        "PSUADE adaptive(1) ERROR: cannot read file %s or wrong format.\n",
        winput);
      exit(1);
    }
    auxIO->getParameter("method_nsamples", pPtr);
    auxNSamples = pPtr.intData_;
    auxIO->getParameter("input_ninputs", pPtr);
    auxNInputs = pPtr.intData_;
    auxIO->getParameter("output_noutputs", pPtr);
    auxNOutputs = pPtr.intData_;
    if (auxNInputs != nInputs)
    {
      printOutTS(PL_ERROR,
      "PSUADE adaptive(1) ERROR : auxiliary nInputs != %d\n",nInputs);
      return 0;
    }
    if (auxNOutputs > 1)
    {
      printOutTS(PL_INFO,
           "PSUADE adaptive(1) ERROR: auxiliary sample nOutputs != 1.\n");
      return 0;
    }
    auxIO->getParameter("input_sample", pAuxInputs);
    auxSamInputs = pAuxInputs.dbleArray_;
    double *dAuxInps = auxSamInputs;
    length = (nSamples+auxNSamples+nRefinements*refineSize)*nInputs;
    auxSamInputs = new double[length];
    for (ss = 0; ss < auxNSamples*nInputs; ss++)
      auxSamInputs[ss] = dAuxInps[ss];
    auxIO->getParameter("output_sample", pAuxOutputs);
    length = (nSamples+auxNSamples+nRefinements*refineSize)*nOutputs;
    auxSamOutputs = new double[length];
    for (ss = 0; ss < auxNSamples; ss++)
      auxSamOutputs[ss] = pAuxOutputs.dbleArray_[ss];
  }

  //**/ ----------------------------------------------------------------
  //**/ get other parameters from psuadeIO 
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;
  //**/ force to use MARSB
  //psuadeIO_->getParameter("ana_rstype", pPtr);
  //rstype = pPtr.intData_;
  //if ((rstype != PSUADE_RS_MARSB) && (rstype != PSUADE_RS_GP1) &&
  //    (rstype != PSUADE_RS_TGP)   && (rstype != PSUADE_RS_KR))
  //{
     printOutTS(PL_INFO, 
        "PSUADE adaptive(1) INFO: RS type set to MarsBag.\n");
     rstype = PSUADE_RS_MARSB; 
  //}
  int numMars=100;
  int marsMode=0;
  if (psConfig_.RSExpertModeIsOn())
  {
    if (rstype == PSUADE_RS_MARSB)
    {
      snprintf(cString,100,"Number of MARS (default = 100, >2, <502) = ");
      numMars = getInt(3, 501, cString);
      snprintf(cString,100,"Use mean (0) or median (1) of MarsBag : ");
      marsMode = getInt(0, 1, cString);
    }
  }
  psuadeIO_->getParameter("ana_threshold", pPtr);
  anaThreshold = pPtr.dbleData_;

  //**/ ----------------------------------------------------------------
  //**/ generate initial sample, if needed
  //**/ refineType = 0: uniform refinement (when adaptive not set)
  //**/ for reliability, refinement size should be unlimited
  //**/ ----------------------------------------------------------------
  if (initFlag == 1)
  {
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samMethod); 
    sampler_->setPrintLevel(outputLevel_);
    sampler_->setInputBounds(nInputs, iLowerB, iUpperB);
    sampler_->setOutputParams(nOutputs);
    sampler_->setSamplingParams(nSamples, 1, 1);
    sampler_->initialize(0);
    nSamples = sampler_->getNumSamples();
    psuadeIO_->updateMethodSection(-1,nSamples,-1,-1,-1);
  }
  //**/ refineType can be set to be uniform (default) or adaptive (1)
  if (refineType == 0)
  {
    strcpy(sparam, "setUniformRefinement");
    refineSize = 100000;
  }
  else strcpy(sparam, "setAdaptiveRefinementBasedOnErrors");
  sampler_->setParam(sparam);
  //**/ if uniform, unlimited refineSize
  //**/ if adaptive, refineSize is user-specified
  if (refineSize > 0)
  {
    snprintf(sparam,100,"setRefineSize %d", refineSize);
    sampler_->setParam(sparam);
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): refineSize = %d\n",refineSize);
  }
  int refineRatio = 2;
  int randomize = 1;

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  funcIO = createFunctionInterface(psuadeIO_);

  //**/ ----------------------------------------------------------------
  //**/ set up general run parameters
  //**/ ----------------------------------------------------------------
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ ----------------------------------------------------------------
  //**/ load the initial samples for MARS
  //**/ ----------------------------------------------------------------
  nSamples = sampler_->getNumSamples();
  psMatrix matMarsDataX, matMarsDataY;
  matMarsDataX.setFormat(PS_MAT2D); 
  matMarsDataY.setFormat(PS_MAT2D); 
  length = (nSamples+auxNSamples+nRefinements*refineSize);
  matMarsDataX.setDim(numMars, length*nInputs);
  matMarsDataY.setDim(numMars, length);
  double **marsDataX = matMarsDataX.getMatrix2D();
  double **marsDataY = matMarsDataY.getMatrix2D();
  for (ii = 0; ii < numMars; ii++)
  {
    for (ss = 0; ss < auxNSamples*nInputs; ss++)
      marsDataX[ii][ss] = auxSamInputs[ss];
    for (ss = 0; ss < auxNSamples; ss++)
      marsDataY[ii][ss] = auxSamOutputs[ss];
  }

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  int jobsCompleted = 0;
  int refineLevel = 0;
  int marsNSamples = auxNSamples;

  while (1)
  {
    //**/ ----------------------------------------------------------------
    //**/ get the sample data, perform the necessary transforms
    //**/ ----------------------------------------------------------------
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 1)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, 
         "PSUADE adaptive(1): current level    = %d (of %d)\n",
         refineLevel, nRefinements);
      printOutTS(PL_INFO, 
         "PSUADE adaptive(1): current nSamples = %d\n", nSamples);
    }
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    sampler_->getSamples(nSamples,nInputs,nOutputs,vecSamInps.getDVector(), 
                         vecSamOuts.getDVector(), vecSamStas.getIVector());
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                     vecSamInps.getDVector(),NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,vecSamOuts.getDVector(),
                                   vecSamStas.getIVector(),NULL);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) 
      if (vecSamStas[ss]==0) currNJobs++;
    if (psConfig_.AnaExpertModeIsOn() && (jobsCompleted < currNJobs))
    {
      printOutTS(PL_INFO, 
           "A sample has been created in the psuadeData file. \n");
      printOutTS(PL_INFO, 
           "At this point you can choose to run your model with\n");
      printOutTS(PL_INFO, 
           "this sample via psuade or by yourself (if the model\n");
      printOutTS(PL_INFO, 
           "is expensive to run, you want to choose the latter).\n");
      snprintf(systemCommand,100,"Run the model yourself (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        printOutTS(PL_INFO,
           "You have chosen to run the sample yourself.\n");
        printOutTS(PL_INFO, 
          "The following are the steps to continue ARSM refinements:\n");
        printOutTS(PL_INFO, 
          "(1) Rename psuadeData to something else (e.g. psData). \n");
        printOutTS(PL_INFO, 
          "(2) Run the sample in this file and collect outputs.\n");
        printOutTS(PL_INFO, 
          "(3) Replace the outputs in psData (do not change others).\n");
        printOutTS(PL_INFO, 
          "(4) Finally, restart psuade with this file (psData).\n");
        delete funcIO;
        return 0;
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    double *dInps = vecSamInps.getDVector();
    double *dOuts = vecSamOuts.getDVector();
    int    *iStas = vecSamStas.getIVector();
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          if (temp != NULL) Py_DECREF(temp);
        }
#endif
	//**/
	//**/ If anything has set the flag psuadeStop_ to 1, stop now
	//**/
        if (psuadeStop_ == 1)
        {
          psuadeIO_->writePsuadeFile(NULL,0);
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
          //throw Psuade_Stop_Exception();
        }

        if ((vecSamStas[ss] == 0) && 
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &(dInps[ss*nInputs]), 
                            nOutputs,&(dOuts[ss*nOutputs]),0);   
          if (outputLevel_ > 2)
            printf("PSUADE adaptive(1): job submitted = %d, status = %d\n",
                   ss+1, status);

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dOuts[ss*nOutputs]),
                                  &(iStas[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            if ((nJobsDiff % saveFrequency) == 0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                           vecSamOuts.getDVector(),vecSamStas.getIVector(),
                           NULL);
              psuadeIO_->writePsuadeFile(NULL,0);
            }
            if (outputLevel_ > 0)
            {
               if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
               fflush(stdout);
            }
          }
          else
          {
            vecSamStas[ss] = status;
            parallelJobCount++;
          }
        }
        else if (vecSamStas[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &(dInps[ss*nInputs]), 
                              nOutputs, &(dOuts[ss*nOutputs]),2);   
          if (outputLevel_ > 2)
            printf("PSUADE adaptive(1): job checked = %d, status = %d\n",
                   ss+1, status);
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dOuts[ss*nOutputs]),
                                  &(iStas[ss]));
            jobsCompleted++;
            parallelJobCount--;
          }
          else vecSamStas[ss]++;

          if ((minJobWaitTime > 0) &&
              ((vecSamStas[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
            vecSamStas[ss] = 0;
            parallelJobCount--;
            ss--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                 "PSUADE adaptive(1): sample %6d to be restarted.\n",ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,
                   vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > maxState) maxState = vecSamStas[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > 10) vecSamStas[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO, 
               "PSUADE adaptive(1): jobs completed = %d(of %d)\n",
               jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ refine one level uniformly
    //**/ ----------------------------------------------------------------
    samplerAux = (Sampling *) SamplingCreateFromID(samMethod);
    samplerAux->setInputBounds(nInputs, iLowerB, iUpperB);
    samplerAux->setOutputParams(nOutputs);
    samplerAux->setSamplingParams(nSamples, -1, -1);
    samplerAux->initialize(1);
    samplerAux->loadSamples(nSamples, nInputs, nOutputs, 
                 vecSamInps.getDVector(), vecSamOuts.getDVector(), 
                 vecSamStas.getIVector());
    //**/ tell sampler to use psuadeMetisInfo.tmp instead
    strcpy(sparam, "changeInfoName");
    samplerAux->setParam(sparam);
    strcpy(sparam, "setUniformRefinement");
    samplerAux->setParam(sparam);
    samplerAux->refine(refineRatio,randomize,0,0,NULL);
    nSamples2 = samplerAux->getNumSamples();
    if (nSamples2 != 2 * nSamples)
    {
      printOutTS(PL_INFO,"PSUADE adaptive(1): Castastropic error.\n");
      printOutTS(PL_INFO,
         "       refined sample size != 2 * original size\n");
      printOutTS(PL_INFO,"       Please consult developers.\n");
      delete samplerAux;
      if (sampler_ != NULL) SamplingDestroy(sampler_);
      sampler_ = NULL;
      samplerAux = NULL;
      delete funcIO;
      return 0;
    }
    psVector  vecSamInps2, vecSamOuts2, vecSamStds2;
    psIVector vecSamStas2;
    vecSamInps2.setLength(nSamples2*nInputs);
    vecSamOuts2.setLength(nSamples2*nOutputs);
    vecSamStas2.setLength(nSamples2);
    vecSamStds2.setLength(nSamples2);
    samplerAux->getSamples(nSamples2, nInputs, nOutputs, 
          vecSamInps2.getDVector(), vecSamOuts2.getDVector(), 
          vecSamStas2.getIVector());

    //**/ ----------------------------------------------------------------
    //**/ response surface analysis
    //**/ ----------------------------------------------------------------
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): response surface analysis.\n");
    printOutTS(PL_INFO, 
       "PSUADE adaptive(1): current level     = %d (of %d)\n",
           refineLevel, nRefinements);
    printOutTS(PL_INFO, "                    training nSamples = %d\n",
           nSamples + auxNSamples);
    printOutTS(PL_INFO, "                    test set nSamples = %d\n",
           nSamples2-nSamples);

    faPtr = genFA(rstype, nInputs, iOne, nSamples+auxNSamples);
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);

    //**/ prepare MarsBag samples
    for (ii = 0; ii < numMars; ii++)
    {
      for (ss = 0; ss < nSamples-marsNSamples+auxNSamples; ss++)
      {
        if (marsNSamples == auxNSamples)
             ivar1 = PSUADE_rand() % (nSamples-marsNSamples+auxNSamples);
        else ivar1 = ss;
        for (jj = 0; jj < nInputs; jj++)
          marsDataX[ii][(marsNSamples+ss)*nInputs+jj] =
               vecSamInps[(ivar1+marsNSamples-auxNSamples)*nInputs+jj];
        marsDataY[ii][marsNSamples+ss] = 
               vecSamOuts[ivar1+marsNSamples-auxNSamples];
      }
    }
    marsNSamples = nSamples + auxNSamples;

    if ((!psConfig_.RSExpertModeIsOn()) && (rstype == PSUADE_RS_MARSB))
    {
      strcpy(cString, "mars_params");
      targv[0] = (char *) cString;
      ivar1 = (nSamples + auxNSamples);
      targv[1] = (char *) &ivar1;
      ivar2 = 2 * nInputs / 3 + 1;
      targv[2] = (char *) &ivar2;
      faPtr->setParams(3, targv);
      strcpy(cString, "num_mars");
      targv[0] = (char *) cString;
      targv[1] = (char *) &numMars;
      faPtr->setParams(2, targv);
      if (marsMode == 1)
      {
        strcpy(cString, "median");
        targv[0] = (char *) cString;
        faPtr->setParams(1, targv);
      }
      strcpy(cString, "mars_sample");
      targv[0] = (char *) cString;
      targv[2] = (char *) &marsNSamples;
      for (ii = 0; ii < numMars; ii++)
      {
        targv[1] = (char *) &ii;
        targv[3] = (char *) marsDataX[ii];
        targv[4] = (char *) marsDataY[ii];
        faPtr->setParams(5, targv);
      }
    }
    faPtr->initialize(vecSamInps.getDVector(),vecSamOuts.getDVector());

    //**/ evaluate the unevaluated new points
    double *dInps2 = vecSamInps2.getDVector();
    double *dOuts2 = vecSamOuts2.getDVector();
    int    *iStas2 = vecSamStas2.getIVector();
    double *dStds2 = vecSamStds2.getDVector();
    faPtr->evaluatePointFuzzy(nSamples2-nSamples,&(dInps2[nInputs*nSamples]), 
                         &(dOuts2[nSamples]), &(dStds2[nSamples]));

    //**/ if no test set, compute statistics (on s.d.) of the new points
    if (tstNSamples == 0)
    {
      // extract the magnitude of each sample points
      totalSum = errMax = errAvg = errL2 =0.0;
      for (ss = nSamples; ss < nSamples2; ss++)
      {
        totalSum += PABS(dOuts2[ss]);
        errMax = (PABS(vecSamStds2[ss])>errMax) ? 
                  PABS(vecSamStds2[ss]) : errMax;
        errAvg += vecSamStds2[ss];
        errL2  += pow(vecSamStds2[ss], 2.0e0);
      }
      errL2 = sqrt(errL2 / (nSamples2 - nSamples));
      totalSum /= (nSamples2 - nSamples);
      errAvg  = sqrt(errAvg/(nSamples2-nSamples));
      printOutTS(PL_INFO, 
         "     response surface unscaled max error = %e\n",errMax);
      printOutTS(PL_INFO, 
         "     response surface   scaled max error = %e\n",errMax/totalSum);
      printOutTS(PL_INFO, 
         "     response surface unscaled rms error = %e\n",errL2);
      printOutTS(PL_INFO, 
         "     response surface   scaled rms error = %e\n",errL2/totalSum);
      printOutTS(PL_INFO, 
         "     response surface unscaled avg error = %e\n",errAvg);
      printOutTS(PL_INFO,
         "     response surface   scaled avg error = %e\n",errAvg/totalSum);
    }

    //**/ if there is a test set, compute its error statistics
    //**/ using the response surface from last batch
    if (tstNSamples > 0)
    {
      printEquals(PL_INFO, 0);
      tstOutputs = new double[tstNSamples];
      faPtr->evaluatePoint(tstNSamples, tstSamInputs, tstOutputs);
      totalSum = errMax = errAvg = errL2 =0.0;
      for (ss = 0; ss < tstNSamples; ss++)
      {
        totalSum += PABS(tstOutputs[ss]);
        dtemp = tstOutputs[ss] - tstSamOutputs[ss];
        errAvg += dtemp;
        dtemp = PABS(dtemp);
        errL2  += dtemp * dtemp;
        errMax = (dtemp > errMax) ? dtemp : errMax;
      }
      errL2 = sqrt(errL2 / tstNSamples);
      totalSum /= (double) tstNSamples;
      errAvg  = errAvg / tstNSamples;
      printOutTS(PL_INFO, 
        "     test sample RS unscaled max error = %e\n",errMax);
      printOutTS(PL_INFO, 
        "     test sample RS   scaled max error = %e\n",errMax/totalSum);
      printOutTS(PL_INFO, 
        "     test sample RS unscaled rms error = %e\n",errL2);
      printOutTS(PL_INFO, 
        "     test sample RS   scaled rms error = %e\n",errL2/totalSum);
      printOutTS(PL_INFO, 
        "     test sample RS unscaled avg error = %e\n",errAvg);
      printOutTS(PL_INFO, 
        "     test sample RS   scaled avg error = %e\n",errAvg/totalSum);
      //**/if (errL2 < anaThreshold || refineLevel >= nRefinements)
      if (outputLevel_ > 0)
      {
        snprintf(cString,100,"arsm_marsb_err.m");
        fp = fopen(cString, "w");
        fprintf(fp, "%% inputs, true outputs, predicted outputs\n");
        fwritePlotCLF(fp);
        fprintf(fp, "A = [\n");
        for (ss = 0; ss < tstNSamples; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%e ", tstSamInputs[ss*nInputs+ii]);
          fprintf(fp, "%e %e\n", tstSamOutputs[ss], tstOutputs[ss]);
        }
        fprintf(fp, "];\n");
        fprintf(fp, "m = %d;\n", nInputs);
        fprintf(fp, "X1 = A(:,1);\n");
        fprintf(fp, "X2 = A(:,2);\n");
        fprintf(fp, "Y  = A(:,m+1) - A(:,m+2);\n");
        fprintf(fp, "subplot(2,2,1)\n");
        fprintf(fp, "plot3(X1,X2,Y,'*','markerSize',13')\n");
        fwritePlotXLabel(fp, "Input 1");
        fwritePlotYLabel(fp, "Input 2");
        fwritePlotZLabel(fp, "Output");
        fwritePlotAxes(fp);
        fwritePlotTitle(fp, "Errors w.r.t. Input 1 and 2");
        fprintf(fp, "subplot(2,2,2)\n");
        fprintf(fp, "plot(X1,Y,'*','markerSize',13')\n");
        fwritePlotXLabel(fp, "Input 1");
        fwritePlotYLabel(fp, "Output");
        fwritePlotAxes(fp); 
        fwritePlotTitle(fp, "Output vs Input 1");
        fprintf(fp, "subplot(2,2,4)\n");
        fprintf(fp, "plot(X2,Y,'*','markerSize',13')\n");
        fwritePlotXLabel(fp, "Input 2");
        fwritePlotYLabel(fp, "Output");
        fwritePlotAxes(fp); 
        fwritePlotTitle(fp, "Output vs Input 2");
        fprintf(fp, "AA = [\n");
        for (ss = 0; ss < nSamples; ss++)
        {
          for (ii = 0; ii < nInputs; ii++)
            fprintf(fp, "%e ", vecSamInps[ss*nInputs+ii]);
          fprintf(fp, "\n");
        }
        fprintf(fp, "];\n");
        fprintf(fp, "XX1 = AA(:,1);\n");
        fprintf(fp, "XX2 = AA(:,2);\n");
        fprintf(fp, "subplot(2,2,3)\n");
        fprintf(fp, "plot(XX1,XX2,'*','markerSize',13')\n");
        fwritePlotXLabel(fp, "Input 1");
        fwritePlotYLabel(fp, "Input 2");
        fwritePlotAxes(fp); 
        fwritePlotTitle(fp, "Input 1 vs 2 (original sample)");
        printOutTS(PL_INFO, "PSUADE adaptive(1): error plots are in %s.\n",
               cString);
        fclose(fp);
      }
      delete [] tstOutputs;
    }
    printEquals(PL_INFO, 0);

    //**/ ----------------------------------------------------------------
    //**/ update the sample outputs to the sampler
    //**/ ----------------------------------------------------------------
    refineLevel++;
    sampler_->loadSamples(nSamples,nInputs,nOutputs,vecSamInps.getDVector(), 
                          vecSamOuts.getDVector(),vecSamStas.getIVector());
    if (errL2 < anaThreshold)
    {
      printOutTS(PL_INFO,"PSUADE adaptive(1): threshold reached.\n");
      printOutTS(PL_INFO,"                    unscaled rms = %en",errL2);
      printOutTS(PL_INFO, 
              "                    threshold    = %e\n", anaThreshold);
      delete faPtr;
      break;
    }
    if (psConfig_.AnaExpertModeIsOn())
    {
      //**/snprintf(systemCommand,100,"Want to quit now (y or n) ? ");
      //**/getString(systemCommand, cString);
      cString[0] = 'n';
      if (cString[0] == 'y')
      {
        delete faPtr;
        break;
      }
    }
    if (refineLevel > nRefinements)
    {
      printOutTS(PL_INFO, 
           "PSUADE adaptive(1): number of refinements %d reached.\n",
           nRefinements);
      delete faPtr;
      break;
    }

    //**/ ----------------------------------------------------------------
    //**/ request refinement with prediction error information from the
    //**/ uniformly refined sample
    //**/ ----------------------------------------------------------------
    if (useRandomPts == 1)
    {
      psIVector vecIT;
      vecIT.setLength(nSamples);
      generateRandomIvector(nSamples, vecIT.getIVector());
      for (ss = nSamples-1; ss >= nSamples-refineSize; ss--) 
        vecSamStds2[vecIT[ss]] = 1.0;
      for (ss = 0;  ss < nSamples-refineSize; ss++) 
        vecSamStds2[vecIT[ss]] = 0.0;
    }
    else
    {
      for (ss = 0; ss < nSamples; ss++) 
        vecSamStds2[ss] = PABS(vecSamStds2[ss+nSamples]);
    }
    //targv[0] = (char *) faPtr;
    //strcpy(cString, "setRSPTR");
    //targv[0] = (char *) cString;
    //targv[1] = (char *) faPtr;
    //ii = 2;
    //sampler_->setParam(ii, targv);
    //**/ the std dev information is from the new points
    sampler_->refine(refineRatio,randomize,refineThreshold,
                     nSamples,vecSamStds2.getDVector());
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ clean up 
    //**/ ----------------------------------------------------------------
    delete samplerAux;
    delete faPtr;
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete funcIO;
  if (auxIO != NULL)
  {
    if (auxSamInputs  != NULL) delete [] auxSamInputs;
    if (auxSamOutputs != NULL) delete [] auxSamOutputs;
    delete auxIO;
  }
  if (tstIO != NULL) delete tstIO;
  return 0;
}
#endif

// ************************************************************************
// run the special adaptive mode for response surface analysis 
// using MARS with bagging as response surface (bootstrap) on GMETIS or
// other than METIS sampling
// This method uses arbitrary initial set of sample points.
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptiveErrBasedG()
{
  int    refineLevel, ss, loopFlag, currNJobs, nJobsDiff, iOne=1, ivar1; 
  int    nPtsPerDim=64, length, parallelJobCount, ivar2, status, maxState;
  int    jobsCompletedLast, auxNSamples=0; 
  int    auxNInputs, auxNOutputs, tstNSamples=0,tstNInputs, tstNOutputs;
  double refineThreshold=1.0, dtemp;
  double totalSum=0.0, errMax=0.0, errAvg=0.0, errL2=0.0;
  double *auxSamInputs=NULL, *auxSamOutputs=NULL;
  double *tstSamInputs=NULL, *tstSamOutputs=NULL, *tstOutputs;
  char   systemCommand[100], cString[100], winput[500];
  char   *targv[3], sparam[501];
  FILE   *fp;
  pData  pPtr, pLowerB, pUpperB, pPDFs, pInpData, pOutData, pStates;
  pData  pAuxInputs, pAuxOutputs, pTstInputs, pTstOutputs;
  FuncApprox *faPtr=NULL;
  Sampling   *samplerAux=NULL;
  PsuadeData *auxIO=NULL, *tstIO=NULL;
  FunctionInterface *funcIO=NULL;

  //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ (This function is intended for uniform distribution only)
  //**/ ----------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"PSUADE adaptive(G): Use GMetis with MarsBagging.\n");
  printOutTS(PL_INFO,
       "Adaptive sampling based on predicted errors from the MARS-based\n");
  printOutTS(PL_INFO,
       "(with bagging) response surfaces (differs from RSMMB\n");
  printOutTS(PL_INFO,
       "by allowing initial sample to be arbitrary instead of Metis).\n");
  printDashes(PL_INFO, 0);
  printOutTS(PL_INFO,
       "To run this method, an initial sample can be provided.\n");
  printOutTS(PL_INFO,"The sample should be somewhat space-filling.\n");
  printEquals(PL_INFO, 0);
  printOutTS(PL_INFO,
       "To be able to exercise more control of this method,\n");
  printOutTS(PL_INFO,"you can turn on interactive and/or expert modes in\n");
  printOutTS(PL_INFO,"the input file.\n");
  printOutTS(PL_INFO,"Turn on outputLevel (>0) to get error histogram.\n");
  printEquals(PL_INFO, 0);

  psuadeIO_->getParameter("input_pdfs", pPDFs);
  int *inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  int iSum = 0;
  for (ss = 0; ss < nInputs; ss++) iSum += inputPDFs[ss];
  if (iSum)
  {
    printOutTS(PL_ERROR,
      "PSUADE adaptive(G) ERROR: does not currently allow \n");
    printOutTS(PL_ERROR,
      "       non-uniform probability distribution to be\n");
    printOutTS(PL_ERROR,"       defined in the INPUT SECTION.\n");
    printOutTS(PL_ERROR,"       Please fix it and then run again.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ get parameters from psuadeIO for experiment and analysis
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  if (nOutputs > 1)
  {
    printOutTS(PL_INFO, "PSUADE adaptive(G): nOutputs should be 1.\n");
    printOutTS(PL_INFO, 
         "       INFO: use 'write' in interactive model to select\n");
    printOutTS(PL_INFO, "             1 output only and re-run this.\n");
    return 0;
  }
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  int refineSize = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ ask users if there is a test set
  //**/ ----------------------------------------------------------------
  printOutTS(PL_INFO,"You may test the quality of the response surface\n");
  printOutTS(PL_INFO,"using a test sample (in psuadeData format).\n");
  snprintf(cString,100,"Use a test sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter the test sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    tstIO = new PsuadeData();
    status = tstIO->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR, 
        "PSUADE adaptive(G) ERROR: cannot read file %s or wrong format.\n",
        winput);
      exit(1);
    }
    tstIO->getParameter("method_nsamples", pPtr);
    tstNSamples = pPtr.intData_;
    tstIO->getParameter("input_ninputs", pPtr);
    tstNInputs = pPtr.intData_;
    tstIO->getParameter("output_noutputs", pPtr);
    tstNOutputs = pPtr.intData_;
    if (tstNInputs != nInputs)
    {
      printOutTS(PL_ERROR,
           "PSUADE adaptive(G) ERROR: test sample nInputs != %d\n",nInputs);
      return 0;
    }
    if (tstNOutputs > 1)
    {
      printOutTS(PL_INFO,
           "PSUADE adaptive(G) ERROR: test sample nOutputs != 1.\n");
      return 0;
    }
    tstIO->getParameter("input_sample", pTstInputs);
    tstSamInputs = pTstInputs.dbleArray_;
    tstIO->getParameter("output_sample", pTstOutputs);
    tstSamOutputs = pTstOutputs.dbleArray_;
  }

  //**/ ----------------------------------------------------------------
  //**/ Users can choose to add another set of sample points
  //**/ ----------------------------------------------------------------
  printOutTS(PL_INFO, 
       "You may add to the base sample an auxiliary sample which\n");
  printOutTS(PL_INFO, 
       "covers the corners (for example, factorial or fractional\n");
  printOutTS(PL_INFO, "factorial).\n");
  snprintf(cString,100,"Add an auxiliary sample ? (y or n) ");
  getString(cString, winput);
  if (winput[0] == 'y')
  {
    snprintf(cString,100,"Enter auxiliary sample file name : ");
    getString(cString, winput);
    ss = strlen(winput);
    winput[ss-1] = '\0';
    auxIO = new PsuadeData();
    status = auxIO->readPsuadeFile(winput);
    if (status != 0)
    {
      printOutTS(PL_ERROR, 
        "PSUADE adaptive(G) ERROR: cannot read file %s or wrong format.\n",
        winput);
      exit(1);
    }
    auxIO->getParameter("method_nsamples", pPtr);
    auxNSamples = pPtr.intData_;
    auxIO->getParameter("input_ninputs", pPtr);
    auxNInputs = pPtr.intData_;
    auxIO->getParameter("output_noutputs", pPtr);
    auxNOutputs = pPtr.intData_;
    if (auxNInputs != nInputs)
    {
      printOutTS(PL_ERROR,
         "PSUADE adaptive(G) ERROR : auxiliary nInputs != %d\n",nInputs);
      exit(1);
    }
    if (auxNOutputs > 1)
    {
      printOutTS(PL_INFO, 
           "PSUADE adaptive(G): auxiliary sample nOutputs != 1.\n");
      printOutTS(PL_INFO, 
           "       INFO: use 'write' in interactive model to select\n");
      printOutTS(PL_INFO, 
           "             1 output only and re-run this.\n");
      return 0;
    }
    printOutTS(PL_INFO,
        "auxiliary sample has sample size = %d\n", auxNSamples);

    auxIO->getParameter("input_sample", pAuxInputs);
    length = (nSamples+auxNSamples+nRefinements*refineSize)*nInputs;
    auxSamInputs = new double[length];
    for (ss = 0; ss < auxNSamples*nInputs; ss++)
      auxSamInputs[ss] = pAuxInputs.dbleArray_[ss];
    auxIO->getParameter("output_sample", pAuxOutputs);
    length = (nSamples+auxNSamples+nRefinements*refineSize)*nOutputs;
    auxSamOutputs = new double[length];
    for (ss = 0; ss < auxNSamples; ss++) 
      auxSamOutputs[ss] = pAuxOutputs.dbleArray_[ss];
  }

  //**/ ----------------------------------------------------------------
  //**/ get other parameters from psuadeIO 
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_sample", pInpData);
  double *sampleInputs = pInpData.dbleArray_;
  psuadeIO_->getParameter("output_sample", pOutData);
  double *sampleOutputs = pOutData.dbleArray_;
  psuadeIO_->getParameter("output_states", pStates);
  int *sampleStates = pStates.intArray_;

  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;
  psuadeIO_->getParameter("ana_rstype", pPtr);
  int rstype = pPtr.intData_;
  int marsMode=0;
  int numMars=100;
  if (psConfig_.RSExpertModeIsOn())
  {
    printf("PSUADE adaptive(G): Select response surface for fitting:\n");
    printf("marsb: MARS with bagging\n");
    printf("gp   : Gaussian process\n");
    snprintf(cString,100,"Select response surface type: ");
    getString(cString, winput);
    if      (!strcmp(winput, "marsb")) rstype = PSUADE_RS_MARSB; 
    else if (!strcmp(winput, "gp"))    rstype = PSUADE_RS_GP3; 
    else
    {
      printOutTS(PL_INFO,"INFO: RS type defaulted to MarsBag.\n");
      rstype = PSUADE_RS_MARSB; 
    }
    if (!strcmp(winput, "marsb")) 
    {
      snprintf(cString,100,"Number of MARS (default = 100, > 2, < 502) = ");
      numMars = getInt(3, 501, cString);
      snprintf(cString,100,"Use mean (0) or median (1) of MarSBag : ");
      marsMode = getInt(0, 1, cString);
    }
  }
  else if (rstype != PSUADE_RS_MARSB)
  {
    printOutTS(PL_INFO,"PSUADE adaptive(G): RS type defaulted to MarsBag.\n");
    rstype = PSUADE_RS_MARSB; 
  }
  psuadeIO_->getParameter("ana_threshold", pPtr);
  double anaThreshold = pPtr.dbleData_;
  int refineRatio = 2;
  int randomize = 1;

  //**/ ----------------------------------------------------------------
  //**/ set up GMetis and load data
  //**/ ----------------------------------------------------------------
  if (sampler_ != NULL) SamplingDestroy(sampler_);
  int samMethod = PSUADE_SAMP_GMETIS;
  sampler_ = (Sampling *) SamplingCreateFromID(samMethod);
  sampler_->setPrintLevel(outputLevel_);
  sampler_->setInputBounds(nInputs, iLowerB, iUpperB);
  sampler_->setOutputParams(nOutputs);
  //**/ do not set nSamples, GMetis will ask for number of aggregates
  //**/ revert (3/2014)
  sampler_->setSamplingParams(nSamples, 1, 1);
  //**/ delete the GMetis information file
  strcpy(sparam, "reset");
  sampler_->setParam(sparam);
  //**/ refineType can be set to be adaptive (1)
  strcpy(sparam, "setAdaptiveRefinementBasedOnErrors");
  sampler_->setParam(sparam);
  //**/ if adaptive, refineSize is user-specified
  if (refineSize > 0)
  {
    snprintf(sparam,100,"setRefineSize %d", refineSize);
    sampler_->setParam(sparam);
    printOutTS(PL_INFO, "PSUADE adaptive(G): refineSize = %d\n",refineSize);
  }
  sampler_->initialize(1);
  sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs,
                        sampleOutputs, sampleStates);
 
  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  funcIO = createFunctionInterface(psuadeIO_);

  //**/ ----------------------------------------------------------------
  //**/ set up general run parameters
  //**/ ----------------------------------------------------------------
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  int jobsCompleted = 0;
  refineLevel = 0;
  nSamples = -1;

  while (1)
  {
    //**/ ----------------------------------------------------------------
    //**/ get the sample data, perform the necessary transforms
    //**/ ----------------------------------------------------------------
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 1)
    {
      printEquals(PL_INFO, 0);
      printOutTS(PL_INFO, 
           "PSUADE adaptive(G): current level    = %d (of %d)\n",
           refineLevel, nRefinements);
      printOutTS(PL_INFO, 
           "PSUADE adaptive(G): current nSamples = %d\n",nSamples);
    }
    psVector  vecSamInps, vecSamOuts;
    psIVector vecSamStas;
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    double *dPtrX = vecSamInps.getDVector();
    double *dPtrY = vecSamOuts.getDVector();
    int    *iPtrS = vecSamStas.getIVector();

    sampler_->getSamples(nSamples,nInputs,nOutputs,vecSamInps.getDVector(), 
                         vecSamOuts.getDVector(), vecSamStas.getIVector());
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                          vecSamInps.getDVector(),NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,vecSamOuts.getDVector(),
                                   vecSamStas.getIVector(),NULL);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) if (vecSamStas[ss]==0) currNJobs++;
    if (psConfig_.AnaExpertModeIsOn() && (jobsCompleted < currNJobs))
    {
      printOutTS(PL_INFO, 
           "A sample has been created in the psuadeData file. \n");
      printOutTS(PL_INFO, 
           "At this point you can choose to run your model with\n");
      printOutTS(PL_INFO, 
           "this sample via psuade or by yourself (if the model\n");
      printOutTS(PL_INFO, 
           "is expensive to run, you want to choose the latter).\n");
      snprintf(systemCommand,100,"Run the model yourself (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        printOutTS(PL_INFO, 
          "You have chosen to run the sample yourself.\n");
        printOutTS(PL_INFO, 
          "The following are steps to continue ARSM refinement:\n");
        printOutTS(PL_INFO, 
          "(1) Rename psuadeData to something else (e.g. psData). \n");
        printOutTS(PL_INFO, 
          "(2) Run the sample in this file and collect outputs.\n");
        printOutTS(PL_INFO, 
          "(3) Replace the outputs in psData (outputs only).\n");
        printOutTS(PL_INFO, 
          "(4) Finally, restart psuade with this file (psData).\n");
        delete funcIO;
        return 0;
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          if (temp != NULL) Py_DECREF(temp);
        }
#endif
        //**/ If anything has set the flag psuadeStop_ to 1, stop now
        if (psuadeStop_ == 1)
	{
	  psuadeIO_->writePsuadeFile(NULL,0);
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
	  //throw Psuade_Stop_Exception();
	}

        if ((vecSamStas[ss] == 0) && 
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &dPtrX[ss*nInputs], 
                                    nOutputs, &dPtrY[ss*nOutputs],0);   
          if (outputLevel_ > 2)
            printf("PSUADE adaptive(G): job submitted = %d, status = %d\n",
                   ss+1, status);

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dPtrY[ss*nOutputs]),
                                  &(iPtrS[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            if ((nJobsDiff % saveFrequency) == 0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                  vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL);
              psuadeIO_->writePsuadeFile(NULL,0);
            }
            if (outputLevel_ > 0)
            {
              if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
              fflush(stdout);
            }
          }
          else
          {
            vecSamStas[ss] = status;
            parallelJobCount++;
          }
        }
        else if (vecSamStas[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs, &dPtrX[ss*nInputs], 
                                    nOutputs, &dPtrY[ss*nOutputs],2);   
          if (outputLevel_ > 2)
            printf("PSUADE adaptive(G): job checked = %d, status = %d\n",
                         ss+1, status);
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs, &(dPtrY[ss*nOutputs]),
                                  &(iPtrS[ss]));
            jobsCompleted++;
            parallelJobCount--;
          }
          else vecSamStas[ss]++;

          if ((minJobWaitTime > 0) &&
              ((vecSamStas[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
            vecSamStas[ss] = 0;
            parallelJobCount--;
            ss--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                   "PSUADE adaptive(G): sample %6d to be restarted.\n",
                   ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,
                   vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > maxState) maxState = vecSamStas[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (vecSamStas[ss] > 10) vecSamStas[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO,
                "PSUADE adaptive(G): jobs completed = %d(of %d)\n",
                jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ refine one level
    //**/ ----------------------------------------------------------------
    printOutTS(PL_INFO,
               "Perform uniform refinement from the current sample.\n");
    samplerAux = (Sampling *) SamplingCreateFromID(samMethod);
    samplerAux->setInputBounds(nInputs, iLowerB, iUpperB);
    samplerAux->setOutputParams(nOutputs);
    samplerAux->setPrintLevel(outputLevel_);
    samplerAux->setSamplingParams(nSamples, -1, -1);
    samplerAux->initialize(1);
    samplerAux->loadSamples(nSamples,nInputs,nOutputs,
                  vecSamInps.getDVector(), vecSamOuts.getDVector(), 
                  vecSamStas.getIVector());
    strcpy(sparam, "changeInfoName");
    samplerAux->setParam(sparam);
    strcpy(sparam, "setUniformRefinement");
    samplerAux->setParam(sparam);
    samplerAux->refine(refineRatio,randomize,0,0,NULL);
    int nSamples2 = samplerAux->getNumSamples();
    psVector  vecSamInps2, vecSamOuts2, vecSamStds2;
    psIVector vecSamStas2;
    vecSamInps2.setLength(nSamples2 * nInputs);
    vecSamOuts2.setLength(nSamples2 * nOutputs);
    vecSamStas2.setLength(nSamples2);
    vecSamStds2.setLength(nSamples2);
    double *dPtrX2 = vecSamInps2.getDVector();
    double *dPtrY2 = vecSamOuts2.getDVector();
    double *dPtrS2 = vecSamStds2.getDVector();
    samplerAux->getSamples(nSamples2, nInputs, nOutputs, 
                   vecSamInps2.getDVector(), vecSamOuts2.getDVector(), 
                   vecSamStas2.getIVector());

    //**/ ----------------------------------------------------------------
    //**/ response surface analysis
    //**/ ----------------------------------------------------------------
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"PSUADE adaptive(G): response surface analysis.\n");
    printOutTS(PL_INFO,
           "PSUADE adaptive(G): current level     = %d (of %d)\n",
           refineLevel,nRefinements);
    printOutTS(PL_INFO,"                    training nSamples = %d\n",
           nSamples+auxNSamples);
    printOutTS(PL_INFO,"                    test set nSamples = %d\n",
           nSamples2-nSamples);

    faPtr = genFA(rstype, nInputs, iOne, nSamples+auxNSamples);
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    if ((!psConfig_.RSExpertModeIsOn()) && (rstype == PSUADE_RS_MARSB))
    {
      strcpy(cString, "mars_params");
      targv[0] = (char *) cString;
      ivar1 = (nSamples + auxNSamples);
      targv[1] = (char *) &ivar1;
      ivar2 = 2 * nInputs / 3 + 1;
      printOutTS(PL_INFO, "Set degree of interaction = 3\n");
      ivar2 = 3;
      targv[2] = (char *) &ivar2;
      faPtr->setParams(3, targv);
      strcpy(cString, "num_mars");
      targv[0] = (char *) cString;
      targv[1] = (char *) &numMars;
      faPtr->setParams(2, targv);
      if (marsMode == 1)
      {
        strcpy(cString, "median");
        targv[0] = (char *) cString;
        faPtr->setParams(1, targv);
      }
    }
    if (auxIO != NULL)
    {
      for (ss = 0; ss < nSamples*nInputs; ss++)
        auxSamInputs[auxNSamples*nInputs+ss] = vecSamInps[ss];
      for (ss = 0; ss < nSamples; ss++)
        auxSamOutputs[auxNSamples+ss] = vecSamOuts[ss];
      faPtr->initialize(auxSamInputs,auxSamOutputs);
    }
    else faPtr->initialize(vecSamInps.getDVector(),vecSamOuts.getDVector());

    for (ss = 0; ss < nSamples; ss++) vecSamStds2[ss] = 0;
    faPtr->evaluatePointFuzzy(nSamples2-nSamples,&dPtrX2[nInputs*nSamples], 
                              &dPtrY2[nSamples], &dPtrS2[nSamples]);

    // extract the magnitude of each sample points
    totalSum = errMax = errAvg = errL2 =0.0;
    for (ss = nSamples; ss < nSamples2; ss++)
    {
      totalSum += PABS(vecSamOuts2[ss]);
      errMax = (vecSamStds2[ss] > errMax) ? vecSamStds2[ss] : errMax;
      errAvg += vecSamStds2[ss];
      errL2  += pow(vecSamStds2[ss], 2.0e0);
    }
    errL2 = sqrt(errL2 / (nSamples2 - nSamples));
    totalSum /= (nSamples2 - nSamples);
    errAvg  = errAvg / (nSamples2 - nSamples);
    printOutTS(PL_INFO, 
       "     response surface unscaled max error = %e\n",errMax);
    printOutTS(PL_INFO, 
       "     response surface   scaled max error = %e\n",errMax/totalSum);
    printOutTS(PL_INFO, 
       "     response surface unscaled rms error = %e\n",errL2);
    printOutTS(PL_INFO, 
       "     response surface   scaled rms error = %e\n",errL2/totalSum);
    printOutTS(PL_INFO, 
       "     response surface unscaled avg error = %e\n",errAvg);
    printOutTS(PL_INFO, 
       "     response surface   scaled avg error = %e\n",errAvg/totalSum);
    if (tstNSamples > 0)
    {
      printEquals(PL_INFO, 0);
      tstOutputs = new double[tstNSamples];
      faPtr->evaluatePoint(tstNSamples, tstSamInputs, tstOutputs);
      totalSum = errMax = errAvg = errL2 =0.0;
      for (ss = 0; ss < tstNSamples; ss++)
      {
        totalSum += PABS(tstOutputs[ss]);
        dtemp = tstOutputs[ss] - tstSamOutputs[ss];
        errAvg += dtemp;
        dtemp = PABS(dtemp);
        errL2  += dtemp * dtemp;
        errMax = (dtemp > errMax) ? dtemp : errMax;
      }
      errL2 = sqrt(errL2 / tstNSamples);
      totalSum /= (double) tstNSamples;
      errAvg  = errAvg / tstNSamples;
      printOutTS(PL_INFO, 
         "     test sample RS unscaled max error = %e\n",errMax);
      printOutTS(PL_INFO, 
         "     test sample RS   scaled max error = %e\n",errMax/totalSum);
      printOutTS(PL_INFO, 
         "     test sample RS unscaled rms error = %e\n",errL2);
      printOutTS(PL_INFO, 
         "     test sample RS   scaled rms error = %e\n",errL2/totalSum);
      printOutTS(PL_INFO, 
         "     test sample RS unscaled avg error = %e\n",errAvg);
      printOutTS(PL_INFO, 
         "     test sample RS   scaled avg error = %e\n",errAvg/totalSum);
      if (outputLevel_ > 0)
      {
        snprintf(cString,100,"rsag_err_hist_%d.m", refineLevel);
        fp = fopen(cString, "w");
        if (fp != NULL)
        {
          fprintf(fp, "%% first column: true outputs\n");
          fprintf(fp, "A = [\n");
          for (ss = 0; ss < tstNSamples; ss++)
            fprintf(fp, "%e %e\n", tstSamOutputs[ss], tstOutputs[ss]);
          fprintf(fp, "];\n");
          fwritePlotCLF(fp);
          fprintf(fp, "hist(A(:,1)-A(:,2), 10)\n");
          fwritePlotAxes(fp);
          fwritePlotTitle(fp, "Distribution of Prediction Errors");
          fwritePlotXLabel(fp, "Output Value");
          snprintf(winput,100,"Count (total = %d)", tstNSamples);
          fwritePlotYLabel(fp, winput);
          fprintf(fp,"E = A(:,1)-A(:,2);\n");
          fprintf(fp,"hist(E)\n");
          fprintf(fp,"m = mean(E);\n");
          fprintf(fp,"std  = sqrt(sum((E - m) .* (E - m)) / length(E))\n");
          fclose(fp);
          printOutTS(PL_INFO,
               "PSUADE adaptive(G): error distribution plot is in %s.\n",
               cString);
        }
      }
      delete [] tstOutputs;
    }

    //**/ ----------------------------------------------------------------
    //**/ update the sample data in the sampler
    //**/ ----------------------------------------------------------------
    refineLevel++;
    sampler_->loadSamples(nSamples,nInputs,nOutputs,vecSamInps.getDVector(), 
                          vecSamOuts.getDVector(), vecSamStas.getIVector());
    if (errL2 < anaThreshold)
    {
      printOutTS(PL_INFO, 
         "PSUADE adapive(G): threshold reached (based on unscaled L2).\n");
      printOutTS(PL_INFO, "         Error     = %e\n",errL2);
      printOutTS(PL_INFO, "         Threshold = %e\n",anaThreshold);
      break;
    }
    if (psConfig_.AnaExpertModeIsOn())
    {
      snprintf(systemCommand,100,"Do you want to quit now (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        break;
      }
    }
    if (refineLevel > nRefinements)
    {
      printOutTS(PL_INFO, 
           "PSUADE adapive(G): number of refinements %d reached.\n",
           nRefinements);
      break;
    }

    //**/ ----------------------------------------------------------------
    //**/ request refinement 
    //**/ ----------------------------------------------------------------
    for (ss = 0; ss < nSamples2-nSamples; ss++)
      vecSamStds2[ss] = PABS(vecSamStds2[ss+nSamples]);
    sampler_->refine(refineRatio,randomize,refineThreshold,
                     nSamples2-nSamples,vecSamStds2.getDVector());
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ clean up 
    //**/ ----------------------------------------------------------------
    delete samplerAux;
    delete faPtr;
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete funcIO;
  if (auxIO != NULL)
  {
    if (auxSamInputs  != NULL) delete [] auxSamInputs;
    if (auxSamOutputs != NULL) delete [] auxSamOutputs;
    delete auxIO;
  }
  if (tstIO != NULL) delete tstIO;
  return 0;
}

// ************************************************************************
// run the special adaptive mode for risk analysis (with METIS)
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptivePRA()
{
  int    initFlag, count, askFlag=0, loopFlag, currNJobs, *sampleStates;
  int    nJobsDiff, parallelJobCount, status, maxState, jobsCompletedLast;
  int    ss, curVol, totVol, numSuccess=0;
  double refineThreshold=1.0, curVal=1.0, lastVal, ddata;
  char   cString[100], sparam[501];
  pData  pPtr, pLowerB, pUpperB, pPDFs;
  FunctionInterface *funcIO=NULL;

  //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ (This function is intended for uniform distribution only)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_pdfs", pPDFs);
  int *inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  int iSum = 0;
  for (ss = 0; ss < nInputs; ss++) iSum += inputPDFs[ss];
  if (iSum)
  {
    printOutTS(PL_ERROR, 
         "PSUADE adaptivePRA ERROR: does not currently allow \n");
    printOutTS(PL_ERROR, 
         "       non-uniform probability distribution to be\n");
    printOutTS(PL_ERROR, "       defined in the INPUT SECTION.\n");
    printOutTS(PL_ERROR, "       Please fix it and then run again.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking and correcting
  //**/ (This function is intended for METIS design only)
  //**/ (If current sample is not METIS, generate a new one)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_refine_type", pPtr);
  int refineType = pPtr.intData_;
  psuadeIO_->getParameter("method_sampling", pPtr);
  int samMethod = pPtr.intData_;
  if (samMethod != PSUADE_SAMP_METIS)
  {
    printOutTS(PL_INFO, 
         "PSUADE adaptivePRA: sampling defaulted to METIS.\n");
    samMethod = PSUADE_SAMP_METIS; 
    psuadeIO_->updateMethodSection(samMethod,-1,-1,-1,-1);
    initFlag = 1;
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = NULL;
  }
  else initFlag = 0;

  //**/ ----------------------------------------------------------------
  //**/ get parameters from psuadeIO for experiment and analysis
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("ana_threshold", pPtr);
  double anaThreshold = pPtr.dbleData_;
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  if (nOutputs > 1)
  {
    printOutTS(PL_INFO, "PSUADE adaptivePRA: nOutputs should be 1.\n");
    return 0;
  }
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  int refineSize = pPtr.intData_;

  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ generate initial sample, if needed
  //**/ refineType = 0: uniform refinement (when adaptive not set)
  //**/ for reliability, refinement size should be unlimited
  //**/ ----------------------------------------------------------------
  if (initFlag == 1)
  {
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samMethod); 
    sampler_->setPrintLevel(outputLevel_);
    sampler_->setInputBounds(nInputs, iLowerB, iUpperB);
    sampler_->setOutputParams(nOutputs);
    sampler_->setSamplingParams(nSamples, 1, 1);
    sampler_->initialize(0);
    nSamples = sampler_->getNumSamples();
    psuadeIO_->updateMethodSection(-1,nSamples,-1,-1,-1);
  }
  //**/ refineType can be set to be uniform (default) or adaptive (1)
  if (refineType == 0)
  {
    strcpy(sparam, "setUniformRefinement");
    refineSize = 100000;
  }
  else strcpy(sparam, "setAdaptiveRefinementBasedOnOutputs");
  sampler_->setParam(sparam);
  //**/ if uniform, unlimited refineSize
  //**/ if adaptive, refineSize is user-specified
  if (refineSize > 0)
  {
    snprintf(sparam,100,"setRefineSize %d", refineSize);
    sampler_->setParam(sparam);
    printOutTS(PL_INFO, "PSUADE adaptivePRA: refineSize = %d\n",refineSize);
  }
  int refineRatio = 2;
  int randomize = 1;

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  funcIO = createFunctionInterface(psuadeIO_);

  //**/ ----------------------------------------------------------------
  //**/ ask for number of uniform refinements
  //**/ ----------------------------------------------------------------
  printOutTS(PL_INFO, 
       "PSUADE adaptivePRA: initial nSamples = %d\n", nSamples);
  printOutTS(PL_INFO, 
       "PSUADE adaptivePRA: number of refinements = %d\n", nRefinements);
  printOutTS(PL_INFO, 
       "This function finds the first division between FAIL/NoFAIL ");
  printOutTS(PL_INFO, "and focuses\n");
  printOutTS(PL_INFO, 
       "the adaptive sampling on the interface. Hence, it is ");
  printOutTS(PL_INFO,"important that\n");
  printOutTS(PL_INFO, 
       "the initial nSamples is sufficiently large to uncover all ");
  printOutTS(PL_INFO,"FAIL/NoFail\n"); 
  printOutTS(PL_INFO,"interfaces.\n");
  int nUniform = 0;

  //**/ ----------------------------------------------------------------
  //**/ set up general run parameters
  //**/ ----------------------------------------------------------------
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  double relMax = 0, relMin = 0.0;
  int jobsCompleted = 0;
  int refineLevel = 0;
  nSamples = -1;

  while (1)
  {
    //**/ ----------------------------------------------------------------
    //**/ get the sample data, perform the necessary transforms
    //**/ ----------------------------------------------------------------
    nSamples = sampler_->getNumSamples();
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO, 
         "PSUADE adaptivePRA: current level    = %d (of %d)\n",
         refineLevel, nRefinements);
    printOutTS(PL_INFO, 
         "PSUADE adaptivePRA: current nSamples = %d\n",nSamples);
    psVector  vecSamInps, vecSamOuts, vecYT;
    psIVector vecSamStas;
    vecSamInps.setLength(nSamples*nInputs);
    vecSamOuts.setLength(nSamples*nOutputs);
    vecSamStas.setLength(nSamples);
    double *sampleInputs  = vecSamInps.getDVector();
    double *sampleOutputs = vecSamOuts.getDVector();
    int    *sampleStates  = vecSamStas.getIVector();
    sampler_->getSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                         sampleOutputs, sampleStates);
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                               sampleInputs,NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                   sampleStates,NULL);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) if (sampleStates[ss]==0) currNJobs++;

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          if (temp != NULL) Py_DECREF(temp);
        }
#endif
        //**/ If anything has set the flag psuadeStop_ to 1, stop now
	if (psuadeStop_ == 1)
	{
	  psuadeIO_->writePsuadeFile(NULL,0);
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
	  //throw Psuade_Stop_Exception();
	}

        if ((sampleStates[ss] == 0) && 
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs,
                        &sampleInputs[ss*nInputs], nOutputs, 
                        &sampleOutputs[ss*nOutputs],0);   

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs,
                            &(sampleOutputs[ss*nOutputs]),
                            &(sampleStates[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            if ((nJobsDiff % saveFrequency) == 0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                                   sampleOutputs,sampleStates,NULL);
              psuadeIO_->writePsuadeFile(NULL,0);
            }
            if (outputLevel_ > 0)
            {
              if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
              fflush(stdout);
            }
          }
          else
          {
            sampleStates[ss] = status;
            parallelJobCount++;
          }
        }
        else if (sampleStates[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs,
                             &sampleInputs[ss*nInputs], nOutputs, 
                             &sampleOutputs[ss*nOutputs],2);   
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs,
                              &(sampleOutputs[ss*nOutputs]),
                              &(sampleStates[ss]));
            jobsCompleted++;
            parallelJobCount--;
          }
          else sampleStates[ss]++;

          if ((minJobWaitTime > 0) &&
              ((sampleStates[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
            sampleStates[ss] = 0;
            parallelJobCount--;
            ss--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                   "PSUADE adaptivePRA: sample %6d to be restarted.\n",
                   ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                     sampleStates,NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (sampleStates[ss] > maxState) maxState = sampleStates[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (sampleStates[ss] > 10) sampleStates[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO,
             "\nAdaptivePRA: jobs completed = %d(of %d)\n",
             jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ request Metis to tabulate results 
    //**/ ----------------------------------------------------------------
    refineLevel++;
    sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                          sampleOutputs, sampleStates);

    //**/ check that all outputs are 0/1 or others
    for (ss = 0; ss < nSamples; ss++)
    {
      if ((sampleOutputs[ss] != 0) && (sampleOutputs[ss] != 1))
        break;
    }

    //**/ if the outputs are not 0/1, a few things needs to be done
    //**/ (1) ask users for safety margin
    //**/ (2) send the pass/fail flags to sampler (replace sampleOutputs)
    if (ss != nSamples)
    {
      printOutTS(PL_INFO, 
         "AdaptivePRA INFO: non 0/1 outputs (transform to 0/1).\n");
      if (askFlag == 0)
      {
        printOutTS(PL_INFO, 
           "          To do so, need lower/upper safety margins.\n");
        while (relMin >= relMax)
        {
          printf("Safety region is defined to be inside [lower, upper].\n");
          snprintf(cString,100,"Upper bound for safety region (-999 if none): ");
          relMax = getDouble(cString);
          if (relMax == -999) relMax = 1.0e10;
          snprintf(cString,100,"Lower bound for safety region (-999 if none): ");
          relMin = getDouble(cString);
          if (relMin == -999) relMin = -1.0e10;
          if (relMin >= relMax) printOutTS(PL_INFO, "INVALID bounds.\n");
        } 
        askFlag = 1;
      }
      vecYT.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++)
      {
        if (sampleOutputs[ss] < relMin || sampleOutputs[ss] > relMax)
             vecYT[ss] = 0;
        else vecYT[ss] = 1;
      }
      sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                            vecYT.getDVector(), sampleStates);
    }

    //**/ With the help of sampler, compute reliability
    strcpy(sparam, "calVolumes");
    curVol = sampler_->setParam(sparam);
    strcpy(sparam, "totalVolumes");
    totVol = sampler_->setParam(sparam);
    lastVal = curVal;
    curVal = 1.0 * curVol / totVol;
    if (refineLevel == 1) lastVal = curVal;
    printOutTS(PL_INFO,
       "CURRENT RELIABLITY (SUCCESS) = %8.5f%% (%d/%d*100)\n",
       100*curVal,curVol,totVol);
    if ((refineLevel > 1) && (curVal != 1.0) && (curVal != 0.0) &&
        (curVal+lastVal != 0.0))
      printOutTS(PL_INFO, "Convergence check = %e (%e), trial %d (of 2)\n",
            PABS(2.0*(curVal-lastVal)/(curVal+lastVal)),anaThreshold,
            numSuccess+1);
    if ((refineLevel > 1) && (curVal != 1.0) && (curVal != 0.0) &&
        (curVal+lastVal != 0.0) &&
        (PABS(2.0*(lastVal-curVal)/(lastVal+curVal)) < anaThreshold))
    {
      if (curVal != 1.0)
      {
        if (numSuccess <= 0) numSuccess++;
        else
        {
          printOutTS(PL_INFO, "Convergence check: %e <? %e\n",
               PABS(2.0*(lastVal-curVal)/(lastVal+curVal)),anaThreshold);
          printOutTS(PL_INFO, 
               "CONVERGED: RELIABILITY (SUCCESS) = %8.5f%%\n",
               100*curVal);
          break;
        }
      }
    }
    else numSuccess = 0;

    //**/ restore the sampleOutputs (in case they have been changed)
    sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                          sampleOutputs, sampleStates);

    //**/ if refined enough, move on
    if (refineLevel > nRefinements) break;

    //**/ ----------------------------------------------------------------
    //**/ request refinement 
    //**/ ----------------------------------------------------------------
    if (refineLevel <= nUniform)
    {
      printOutTS(PL_INFO,
          "AdaptivePRA INFO: user requested uniform refinement.\n");
      strcpy(sparam, "setUniformRefinement");
      sampler_->setParam(sparam);
      printOutTS(PL_INFO, "PRA user-set uniform refinement begins.\n");
      sampler_->refine(refineRatio,randomize,refineThreshold,0,NULL);
      printOutTS(PL_INFO, "PRA user-set uniform refinement completes.\n");
    }
    //**/ if sampleOutputs are not 0/1 and reliability = 0 or 1
    //**/ we need adaptive refinement near the threshold (based on error)
    //**/ and we need to set the refineSize (if we don't impose threshold)
    else if ((relMax != relMin) && ((curVal == 0.0) || (curVal == 1.0)))
    {
      printOutTS(PL_INFO,
         "AdaptivePRA INFO: sample outputs are not 0 or 1\n");
      printOutTS(PL_INFO,
         "      Current reliability == %4.2f => adaptive refinement\n");
      printOutTS(PL_INFO, 
         "      near thresholds (%e,%e).\n",relMin,relMax);
      psVector vecSamErrs;
      vecSamErrs.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++) 
      {
        ddata = sampleOutputs[ss];
        //**/ the closer it is to the limits, the higher the error
        if      (ddata == relMin) ddata = PSUADE_UNDEFINED;
        else if (ddata == relMax) ddata = PSUADE_UNDEFINED;
        else if (PABS(ddata-relMax) < PABS(ddata-relMin))
                                  ddata = 1.0 / PABS(ddata - relMax);
        else                      ddata = 1.0 / PABS(ddata - relMin);
        vecSamErrs[ss] = ddata;
      }
      printOutTS(PL_INFO, 
          "PRA adaptive refinement (Output!=0/1; currRA==0/1) begins.\n");
      strcpy(sparam, "setAdaptiveRefinementBasedOnErrors");
      sampler_->setParam(sparam);
      sampler_->refine(refineRatio,randomize,refineThreshold,
                       nSamples,vecSamErrs.getDVector());
      printOutTS(PL_INFO, 
          "PRA adaptive refinement (Output!=0/1; currRA==0/1) completes.\n");
    }
    //**/ if sampleOutputs are not 0/1 and reliability not 0 or 1
    //**/ convert the outputs to 0/1 based on pass/fail
    //**/ apply adaptive refinement
    else if ((relMax != relMin) && (curVal != 0.0) && (curVal != 1.0))
    {
      printOutTS(PL_INFO,
         "AdaptivePRA INFO: sample outputs are not 0/1\n");
      printOutTS(PL_INFO,
         "      Current reliability != 0 nor 1 => adaptive refinement\n");
      printOutTS(PL_INFO, 
         "      (Convert outputs to 0/1 before refinement).\n");
      printOutTS(PL_INFO, 
         "      (Call refinement based on gradients between neighbors)\n");
      //**/ convert outputs to 0/1
      vecYT.setLength(nSamples);
      for (ss = 0; ss < nSamples; ss++)
      {
        if (sampleOutputs[ss] < relMin || sampleOutputs[ss] > relMax)
             vecYT[ss] = 0;
        else vecYT[ss] = 1;
      }
      sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                            vecYT.getDVector(), sampleStates);
      strcpy(sparam, "setAdaptiveRefinementBasedOnOutputs");
      sampler_->setParam(sparam);
      snprintf(sparam,100,"setRefineSize 100000");
      sampler_->setParam(sparam);
      printOutTS(PL_INFO, 
          "PRA adaptive refinement (Output!=0/1;currRA!=0/1) begins.\n");
      sampler_->refine(refineRatio,randomize,refineThreshold,0,NULL);
      printOutTS(PL_INFO, 
          "PRA adaptive refinement (Output!=0/1;currRA!=0/1) completes.\n");
      snprintf(sparam,100,"setRefineSize %d", refineSize);
      sampler_->setParam(sparam);
      vecYT = vecSamOuts;
      count = nSamples;
      nSamples = sampler_->getNumSamples();
      vecSamInps.setLength(nSamples * nInputs);
      vecSamOuts.setLength(nSamples * nOutputs);
      vecSamStas.setLength(nSamples);
      sampleInputs  = vecSamInps.getDVector();
      sampleOutputs = vecSamOuts.getDVector();
      sampleStates  = vecSamStas.getIVector();
      sampler_->getSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                           sampleOutputs, sampleStates);
      for (ss = 0; ss < count*nOutputs; ss++) 
        sampleOutputs[ss] = vecYT[ss];
      sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                            sampleOutputs, sampleStates);
    }
    //**/ if sampleOutputs are 0/1 and reliability = 0 or 1
    else if ((relMax == relMin) && ((curVal == 0.0) || (curVal == 1.0)))
    {
      printOutTS(PL_INFO,
         "AdaptivePRA INFO: sample outputs are 0/1\n");
      printOutTS(PL_INFO, 
         "     Current reliability = 0 or 1 => uniform refinement.\n");
      strcpy(sparam, "setUniformRefinement");
      sampler_->setParam(sparam);
      printOutTS(PL_INFO, 
           "PRA uniform refinement (Output==0/1;currRA==0/1) begins.\n");
      sampler_->refine(refineRatio,randomize,refineThreshold,0,NULL);
      printOutTS(PL_INFO, 
           "PRA uniform refinement (Output==0/1;currRA==0/1) completes.\n");
    }
    //**/ if sampleOutputs are 0/1 and reliability anything else
    else
    {
      printOutTS(PL_INFO, 
         "AdaptivePRA INFO: sample outputs are 0/1\n");
      printOutTS(PL_INFO, 
         "     Current reliability != 0 nor 1 => adaptive ");
      printOutTS(PL_INFO, "refinement near output\n");
      printOutTS(PL_INFO, "     equal to 0/1 interface.\n");
      strcpy(sparam, "setAdaptiveRefinementBasedOnOutputs");
      sampler_->setParam(sparam);
      snprintf(sparam,100,"setRefineSize 100000");
      sampler_->setParam(sparam);
      printOutTS(PL_INFO, 
          "PRA uniform refinement (Output==0/1;currRA==0/1) begins.\n");
      sampler_->refine(refineRatio,randomize,refineThreshold,
                       nSamples,NULL);
      printOutTS(PL_INFO, 
          "PRA uniform refinement (Output==0/1;currRA!=0/1) completes.\n");
    }

    //**/ ----------------------------------------------------------------
    //**/ update data archive
    //**/ ----------------------------------------------------------------
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete funcIO;
  return 0;
}

// ************************************************************************
// run the special adaptive mode for optimization (with METIS)
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptiveOpt()
{
  int    ss, ii, initFlag, refineLevel, loopFlag, currNJobs, nJobsDiff;
  int    parallelJobCount, status, maxState, jobsCompletedLast;
  int    nUniform, *inputPDFs;
  double refineThreshold=1.0;
  char   cString[100], sparam[501];
  pData  pPtr, pLowerB, pUpperB, pPDFs;
  FunctionInterface *funcIO=NULL;

   //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ (This function is intended for uniform distribution only)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("input_pdfs", pPDFs);
  inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  int iSum = 0;
  for (ss = 0; ss < nInputs; ss++) iSum += inputPDFs[ss];
  if (iSum)
  {
    printOutTS(PL_ERROR, 
         "PSUADE ERROR: adaptiveOpt does not currently allow \n");
    printOutTS(PL_ERROR, 
         "       non-uniform probability distribution to be \n");
    printOutTS(PL_ERROR, "       defined in the INPUT SECTION.\n");
    printOutTS(PL_ERROR, "       Please fix it and then run again.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking and correcting
  //**/ (This function is intended for METIS design only)
  //**/ (If current sample is not METIS, generate a new one)
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_refine_type", pPtr);
  int refineType = pPtr.intData_;
  psuadeIO_->getParameter("method_sampling", pPtr);
  int samMethod = pPtr.intData_;
  if (samMethod != PSUADE_SAMP_METIS)
  {
    printOutTS(PL_INFO, 
         "PSUADE adaptiveOpt: sampling defaulted to METIS.\n");
    samMethod = PSUADE_SAMP_METIS; 
    psuadeIO_->updateMethodSection(samMethod,-1,-1,-1,-1);
    initFlag = 1;
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = NULL;
  }
  else initFlag = 0;

  //**/ ----------------------------------------------------------------
  //**/ get parameters from psuadeIO for experiment and analysis
  //**/ ----------------------------------------------------------------
  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("input_ninputs", pPtr);
  nInputs = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  if (nOutputs > 1)
  {
    printOutTS(PL_INFO, "PSUADE adaptiveOpt: nOutputs should be 1.\n");
    return 0;
  }
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefinements = pPtr.intData_;
  psuadeIO_->getParameter("method_refine_size", pPtr);
  int refineSize = pPtr.intData_;

  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("app_maxparalleljobs", pPtr);
  int maxParallelJobs = pPtr.intData_;
  psuadeIO_->getParameter("app_minjobwaittime", pPtr);
  int minJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_maxjobwaittime", pPtr);
  int maxJobWaitTime = pPtr.intData_;
  psuadeIO_->getParameter("app_launchinterval", pPtr);
  int launchInterval = pPtr.intData_;
  psuadeIO_->getParameter("app_savefrequency", pPtr);
  int saveFrequency = pPtr.intData_;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;

  //**/ ----------------------------------------------------------------
  //**/ generate initial sample, if needed
  //**/ refineType = 0: uniform refinement (when adaptive not set)
  //**/ for reliability, refinement size should be unlimited
  //**/ ----------------------------------------------------------------
  if (initFlag == 1)
  {
    if (sampler_ != NULL) SamplingDestroy(sampler_);
    sampler_ = (Sampling *) SamplingCreateFromID(samMethod); 
    sampler_->setPrintLevel(outputLevel_);
    sampler_->setInputBounds(nInputs, iLowerB, iUpperB);
    sampler_->setOutputParams(nOutputs);
    sampler_->setSamplingParams(nSamples, 1, 1);
    sampler_->initialize(0);
    nSamples = sampler_->getNumSamples();
    psuadeIO_->updateMethodSection(-1,nSamples,-1,-1,-1);
  }
  //**/ refineType can be set to be uniform (default) or adaptive (1)
  if (refineType == 0)
  {
    strcpy(sparam, "setUniformRefinement");
    refineSize = 100000;
  }
  else strcpy(sparam, "setAdaptiveRefinementBasedOnOutputs");
  sampler_->setParam(sparam);
  //**/ if uniform, unlimited refineSize
  //**/ if adaptive, refineSize is user-specified
  if (refineSize > 0)
  {
    snprintf(sparam,100,"setRefineSize %d", refineSize);
    sampler_->setParam(sparam);
    printOutTS(PL_INFO, "PSUADE adaptiveOpt: refineSize = %d\n",refineSize);
  }
  int refineRatio = 2;
  int randomize = 1;

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  funcIO = createFunctionInterface(psuadeIO_);

  //**/ ----------------------------------------------------------------
  //**/ ask for number of uniform refinements
  //**/ ----------------------------------------------------------------
  printOutTS(PL_INFO, 
       "PSUADE adaptiveOpt: initial nSamples = %d\n", nSamples);
  printOutTS(PL_INFO, 
       "PSUADE adaptiveOpt: number of refinements = %d\n", nRefinements);
  printOutTS(PL_INFO, 
       "This function focuses the adaptive sampling on the regions\n");
  printOutTS(PL_INFO, 
       "where most of the small objective function values are located.\n");
  printOutTS(PL_INFO, 
       "It is important that the initial nSamples is sufficient large\n");
  printOutTS(PL_INFO, 
        "to uncover all potential troughs. Alternatively, you can\n");
  printOutTS(PL_INFO, 
       "enforce the number of uniform refinements before adaptive\n");
  printOutTS(PL_INFO, 
       "sampling is applied. You have the opportunity to do it now.\n");
  snprintf(cString,100,"Number of initial uniform refinements : (0 - %d): ",
          nRefinements);
  nUniform = getInt(0, nRefinements, cString);
  snprintf(cString,100,"Number of minimum to track : (1 - 10): ");
  int nMins = getInt(1, 10, cString);

  //**/ ----------------------------------------------------------------
  //**/ set up general run parameters
  //**/ ----------------------------------------------------------------
  if (maxParallelJobs > 1) funcIO->setAsynchronousMode();
  funcIO->setLaunchInterval(launchInterval);

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  int jobsCompleted = 0;
  refineLevel = 0;
  nSamples = -1;

  while (1)
  {
    //**/ ----------------------------------------------------------------
    //**/ get the sample data, perform the necessary transforms
    //**/ ----------------------------------------------------------------
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 1)
    {
      printAsterisks(PL_INFO, 0);
      printOutTS(PL_INFO, 
           "PSUADE adaptiveOpt: current level    = %d (of %d)\n",
           refineLevel, nRefinements);
      printOutTS(PL_INFO, 
           "PSUADE adaptiveOpt: current nSamples = %d\n",nSamples);
    }
    psVector  vecSamInps, vecSamOuts;
    psIVector vecSamStas;
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    double *sampleInputs  = vecSamInps.getDVector();
    double *sampleOutputs = vecSamOuts.getDVector();
    int    *sampleStates  = vecSamStas.getIVector();
    sampler_->getSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                         sampleOutputs, sampleStates);
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                                  sampleInputs,NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                   sampleStates,NULL);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) if (sampleStates[ss]==0) currNJobs++;

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
#ifdef HAVE_PYTHON
        //**/ Yield control to python so it can update the GUI
        PyObject* temp;
        if (update_gui != NULL) 
        {
          temp = PyObject_CallObject(update_gui, NULL);
          if (temp != NULL) Py_DECREF(temp);
        }
#endif
        //**/ If anything has set the flag psuadeStop_ to 1, stop now
	if (psuadeStop_ == 1)
	{
	  psuadeIO_->writePsuadeFile(NULL,0);
          delete funcIO;
          printf("INFO: psuadeStop_ is turned on ==> terminate.\n");
          exit(0);
	  //throw Psuade_Stop_Exception();
	}

        if ((sampleStates[ss] == 0) && 
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs,
                            &sampleInputs[ss*nInputs], nOutputs, 
                            &sampleOutputs[ss*nOutputs],0);   

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs,
                       &(sampleOutputs[ss*nOutputs]),
                       &(sampleStates[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            if ((nJobsDiff % saveFrequency) == 0)
            {
              psuadeIO_->updateOutputSection(nSamples,nOutputs,
                               sampleOutputs,sampleStates,NULL);
              psuadeIO_->writePsuadeFile(NULL,0);
            }
            if (outputLevel_ > 0)
            {
              if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
              fflush(stdout);
            }
          }
          else
          {
            sampleStates[ss] = status;
            parallelJobCount++;
          }
        }
        else if (sampleStates[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to 
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs,
                              &sampleInputs[ss*nInputs], nOutputs, 
                              &sampleOutputs[ss*nOutputs],2);   
          if (status == 0) 
          {
            sampler_->storeResult(ss, nOutputs,
                             &(sampleOutputs[ss*nOutputs]),
                             &(sampleStates[ss]));
            jobsCompleted++;
            parallelJobCount--;
          }
          else sampleStates[ss]++;

          if ((minJobWaitTime > 0) &&
              ((sampleStates[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
            sampleStates[ss] = 0;
            parallelJobCount--;
            ss--; /* roll back to the sample to be restarted */
            if (outputLevel_ > 0) 
              printOutTS(PL_INFO, 
                   "PSUADE adaptiveOpt: sample %6d to be restarted.\n",
                   ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                     sampleStates,NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (sampleStates[ss] > maxState) maxState = sampleStates[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (sampleStates[ss] > 10) sampleStates[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0) 
        printOutTS(PL_INFO, 
             "\nPSUADE adaptiveOpt: jobs completed = %d(of %d)\n",
             jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ request Metis to tabulate results 
    //**/ ----------------------------------------------------------------
    refineLevel++;
    sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs, 
                          sampleOutputs, sampleStates);

    //**/ (1) ask users for safety margin
    //**/ (2) send the pass/fail flags to sampler (replace sampleOutputs)
    psVector  vecYT;
    psIVector vecSortList;
    vecYT.setLength(nSamples);
    vecSortList.setLength(nSamples);
    for (ss = 0; ss < nSamples; ss++) vecYT[ss] = sampleOutputs[ss];
    for (ss = 0; ss < nSamples; ss++) vecSortList[ss] = ss;
    sortDbleList2a(nSamples,vecYT.getDVector(),vecSortList.getIVector());
    for (ss = 0; ss < nMins; ss++)
      printOutTS(PL_INFO, 
           "PSUADE adaptiveOpt: min %2d = %e\n", ss+1, vecYT[ss]);
    printOutTS(PL_INFO, "PSUADE adaptiveOpt: min at \n");
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO, "   Input %2d = %e\n", ii+1,
           sampleInputs[vecSortList[0]*nInputs+ii]);

    //**/ if refined enough, move on
    if (refineLevel > nRefinements) break;

    //**/ ----------------------------------------------------------------
    //**/ request refinement 
    //**/ ----------------------------------------------------------------
    if (refineLevel <= nUniform)
    {
      printOutTS(PL_INFO, "INFO: user requested uniform refinement.\n");
      strcpy(sparam,"setUniformRefinement");
      sampler_->setParam(sparam);
      sampler_->refine(refineRatio,randomize,refineThreshold,0,NULL);
    }
    //**/ we need adaptive refinement near the threshold (based on error)
    //**/ and we need to set the refineSize (if we don't impose threshold)
    else
    {
      psVector vecSamErrs;
      vecSamErrs.setLength(nSamples);
      double dmin = PSUADE_UNDEFINED;
      for (ss = 0; ss < nSamples; ss++) 
      if (sampleOutputs[ss] < dmin) dmin = sampleOutputs[ss];
      for (ss = 0; ss < nSamples; ss++)
      {
        vecSamErrs[ss] = sampleOutputs[ss] - dmin;
        if (sampleOutputs[ss] == 0.0) vecSamErrs[ss] = PSUADE_UNDEFINED;
        else                          vecSamErrs[ss] = 1.0 / vecSamErrs[ss];
      }
      strcpy(sparam,"setAdaptiveRefinementBasedOnErrors");
      sampler_->setParam(sparam);
      sampler_->refine(refineRatio,randomize,refineThreshold,
                       nSamples,vecSamErrs.getDVector());
    }

    //**/ ----------------------------------------------------------------
    //**/ update data archive
    //**/ ----------------------------------------------------------------
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up 
  //**/ -------------------------------------------------------------------
  delete funcIO;
  return 0;
}

#if 0
// ************************************************************************
// run two-level multi-level response surfaces (experimental, not done)
// ------------------------------------------------------------------------
int PsuadeBase::runAdaptive2Level()
{
  int    samMethod, initFlag, refineLevel, kk, status, iOne=1;
  char   systemCommand[100], cString[100], winput[500];
  char   *targv[6], sparam[501];
  pData  pPtr, pLowerB, pUpperB, pPDFs, pTstInputs, pTstOutputs;
  FILE   *fp;
  FuncApprox        *faPtr=NULL;
  FunctionInterface *funcIO=NULL;

  //**/ display banner
  printAsterisks(PL_INFO, 0);
  printOutTS(PL_INFO,"PSUADE adaptive(2L): GMetis with MarsBagging.\n");
  printf("This procedure constructs a 2-level response surface.\n");
  printf("To run this procedure, prepare the following: \n");
  printf("1. a PSUADE file for the expense simulations, which should\n");
  printf("   have an PSUADE_IO section with completed simulations\n");
  printf("2. a PSUADE file for the cheaper simulations, which should\n");
  printf("   have an PSUADE_IO section with completed simulations\n");
  printf("NOTE: normally sample size 1 is smaller than sample size 2\n");
  printf("      (expensive simulations incur high computational cost\n");
  printf("NOTE: The 2 input files must have the same set of input\n");
  printf("      parameters in the same order and ranges.\n");
  printf("NOTE: Only one output will be allowed. If you have multiple\n");
  printf("      outputs in your files, the first one will be used.\n");
  printEquals(PL_INFO, 0);

  //**/ read in PSUADE input files for the two models
  snprintf(cString,100,"Name of the expensive simulation PSUADE file: ");
  getString(cString, winput);
  kk = strlen(winput);
  winput[kk-1] = '\0';
  PsuadeData *fineIO = new PsuadeData();
  status = fineIO->readPsuadeFile(winput);
  if (status != 0)
  {
    printf("run2LevelRS ERROR: cannot read file %s or wrong format.\n",
           winput);
    exit(1);
  }
  snprintf(cString,100,"Name of the cheaper simulation PSUADE file: ");
  getString(cString, winput);
  kk = strlen(winput);
  winput[kk-1] = '\0';
  PsuadeData *coarseIO = new PsuadeData();
  status = coarseIO->readPsuadeFile(winput);
  if (status != 0)
  {
    printf("run2LevelRS ERROR: cannot read file %s or wrong format.\n",
           winput);
    exit(1);
  }

  //**/ read in sample information
  fineIO->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  fineIO->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  coarseIO->getParameter("input_ninputs", pPtr);
  kk = pPtr.intData_;
  if (kk != nInputs)
  {
    printf("run2LevelRS ERROR: the 2 files have different nInputs.\n");
    delete fineIO;
    delete coarseIO;
    exit(1);
  }
  fineIO->getParameter("output_noutputs", pPtr);
  double fNOutputs = pPtr.intData_;
  coarseIO->getParameter("output_noutputs", pPtr);
  double cNOutputs = pPtr.intData_;
  fineIO->getParameter("method_nsamples", pPtr);
  double fNSamples = pPtr.intData_;
  coarseIO->getParameter("method_nsamples", pPtr);
  double cNSamples = pPtr.intData_;
  if (fNSamples <= 0)
  {
    printf("run2LevelRS ERROR: no samples for expensive simulator.\n");
    delete fineIO;
    delete coarseIO;
    exit(1);
  }
  if (cNSamples <= 0)
  {
    printf("run2LevelRS ERROR: no samples for cheaper simulator.\n");
    delete fineIO;
    delete coarseIO;
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ request more sample information
  //**/ ----------------------------------------------------------------
  printf("Current sample size for expensive simulation = %d\n",fNSamples);
  snprintf(cString,100,
     "Specify the maximum number of expensive simulations (> %d, <=1000)",
     fNSamples);
  int fMaxSamples = getInt(fNSamples+1, 1000, cString);
  printf("Current sample size for the cheaper simulation = %d\n",cNSamples);
  snprintf(cString,100,
     "Specify the maximum number of cheaper simulations (> %d, <=100000)",
     cNSamples);
  int cMaxSamples = getInt(cNSamples+1, 100000, cString);

  //**/ ----------------------------------------------------------------
  //**/ fetch sample inputs and outputs
  //**/ ----------------------------------------------------------------
  double *fSamInputs = new double[fMaxSamples*nInputs];
  fineIO->getParameter("input_sample", pPtr);
  for (ii = 0; ii < fNSamples*nInputs; i++) 
    fSamInputs[ii] = pPtr.dbleArray_[ii];
  fineIO->getParameter("output_sample", pPtr);
  for (ii = 0; ii < fNSamples; i++) 
    fSamOutputs[ii] = pPtr.dbleArray_[ii*fNOutputs];
  double *cSamInputs = new double[cMaxSamples*nInputs];
  coarseIO->getParameter("input_sample", pPtr);
  for (ii = 0; ii < cNSamples*nInputs; i++) 
    cSamInputs[ii] = pPtr.dbleArray_[ii];
  coarseIO->getParameter("output_sample", pPtr);
  for (ii = 0; ii < cNSamples; i++) 
    cSamOutputs[ii] = pPtr.dbleArray_[ii*cNOutputs];
  fineIO->getParameter("input_lbounds", pPtr);
  double *iLowerB = pPtr.dbleArray_;
  pPtr.dbleArray_ = NULL;
  fineIO->getParameter("input_ubounds", pPtr);
  double *iUpperB = pPtr.dbleArray_;
  pPtr.dbleArray_ = NULL;
  int nRefinements = 2;

  //**/ ----------------------------------------------------------------
  //**/ set up GMetis and load the current sample set
  //**/ ----------------------------------------------------------------
  samMethod = PSUADE_SAMP_GMETIS;
  Sampling *sampler = (Sampling *) SamplingCreateFromID(samMethod);
  sampler->setPrintLevel(outputLevel_);
  sampler->setInputBounds(nInputs, iLowerB, iUpperB);
  sampler->setOutputParams(iOne);
  sampler->setSamplingParams(nSamples, 1, 1);
  strcpy(sparam, "reset");
  sampler->setParam(sparam);
  strcpy(sparam, "setAdaptiveRefinementBasedOnErrors");
  sampler_->setParam(sparam);
  snprintf(sparam,100,"setRefineSize 5");
  sampler->setParam(sparam);
  sampler->initialize(1);
  sampler->loadSamples(nSamples, nInputs, iOne, sampleInputs,
                       sampleOutputs, sampleStates);

  //**/ ----------------------------------------------------------------
  //**/ set up the FunctionInterface (for sample runs)
  //**/ ----------------------------------------------------------------
  cfuncIO = createFunctionInterface(coarseIO);
  ffuncIO = createFunctionInterface(fineIO);

  //**/ ----------------------------------------------------------------
  //**/ iterate until termination criterion is met
  //**/ ----------------------------------------------------------------
  int jobsCompleted = 0;
  refineLevel = 0;
  nSamples = -1;

  while (1)
  {
    //**/ --------------------------------------------------------------
    //**/ get the sample data, perform the necessary transforms
    //**/ --------------------------------------------------------------
    nSamples = sampler_->getNumSamples();
    if (outputLevel_ > 1)
    {
       printEquals(PL_INFO, 0);
       printOutTS(PL_INFO,
            "PSUADE adaptive(G): current level    = %d (of %d)\n",
            refineLevel, nRefinements);
       printOutTS(PL_INFO,
            "PSUADE adaptive(G): current nSamples = %d\n",nSamples);
    }
    psVector  vecSamInps, vecSamOuts;
    psIVector vecSamStas;
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    double *sampleInputs  = vecSamInps.getDVector();
    double *sampleOutputs = vecSamOuts.getDVector();
    int    *sampleStates  = vecSamStas.getIVector();
    sampler_->getSamples(nSamples, nInputs, nOutputs, sampleInputs,
                         sampleOutputs, sampleStates);
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                                  sampleInputs,NULL,NULL,NULL,NULL,NULL);
    psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                   sampleStates,NULL);

    //**/ ----------------------------------------------------------------
    //**/ count number of jobs (state=0) to be run
    //**/ ----------------------------------------------------------------
    currNJobs = jobsCompleted;
    for (ss = 0; ss < nSamples; ss++) if (sampleStates[ss]==0) currNJobs++;
    if (psConfig_.AnaExpertModeIsOn() && (jobsCompleted < currNJobs))
    {
      printOutTS(PL_INFO,
           "A sample has been created in the psuadeData file. \n");
      printOutTS(PL_INFO,
           "At this point you can choose to run your model with\n");
      printOutTS(PL_INFO,
           "this sample via psuade or by yourself (if the model\n");
      printOutTS(PL_INFO,
           "is expensive to run, you want to choose the latter).\n");
      snprintf(systemCommand,100,"Run the model yourself (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        printOutTS(PL_INFO,
          "You have chosen to run the sample yourself.\n");
        printOutTS(PL_INFO,
          "The following are steps to continue ARSM refinement:\n");
        printOutTS(PL_INFO,
          "(1) Rename psuadeData to something else (e.g. psData). \n");
        printOutTS(PL_INFO,
          "(2) Run the sample in this file and collect outputs.\n");
        printOutTS(PL_INFO,
          "(3) Replace the outputs in psData (outputs only).\n");
        printOutTS(PL_INFO,
          "(4) Finally, restart psuade with this file (psData).\n");
        delete funcIO;
        return 0;
      }
    }

    //**/ ----------------------------------------------------------------
    //**/ run selected sample points (with state = 0)
    //**/ ----------------------------------------------------------------
    parallelJobCount = 0;
    while (jobsCompleted < currNJobs)
    {
      jobsCompletedLast = jobsCompleted;
      for (ss = 0; ss < nSamples; ss++)
      {
        if ((sampleStates[ss] == 0) &&
            (parallelJobCount < maxParallelJobs))
        {
          //**/ -------------------------------------------------------
          //**/ run the job
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs,&sampleInputs[ss*nInputs], 
                                    iOne,&sampleOutputs[ss*nOutputs],0);

          //**/ -------------------------------------------------------
          //**/ if sample run is completed (status=0), store the result
          //**/ if not, increment counter and wait
          //**/ -------------------------------------------------------
          if (status == 0)
          {
            sampler_->storeResult(ss, nOutputs,
                      &(sampleOutputs[ss*nOutputs]),&(sampleStates[ss]));
            jobsCompleted++;
            nJobsDiff = jobsCompleted - jobsCompletedLast;
            psuadeIO_->updateOutputSection(nSamples,nOutputs,
                                sampleOutputs,sampleStates,NULL);
            psuadeIO_->writePsuadeFile(NULL,0);
            if (outputLevel_ > 0)
            {
              if ((jobsCompleted % 11) == 1) printOutTS(PL_INFO, ".");
                     fflush(stdout);
            }
          }
          else
          {
            sampleStates[ss] = status;
            parallelJobCount++;
          }
        }
        else if (sampleStates[ss] >= 2)
        {
          //**/ -------------------------------------------------------
          //**/ state >= 2 : examine if the job has been completed
          //**/ If so, store the results. If not, see if it needs to
          //**/ be restarted.
          //**/ -------------------------------------------------------
          status = funcIO->evaluate(ss,nInputs,
                         &sampleInputs[ss*nInputs], nOutputs,
                         &sampleOutputs[ss*nOutputs],2);
          if (status == 0)
          {
             sampler_->storeResult(ss, nOutputs,
                              &(sampleOutputs[ss*nOutputs]),
                              &(sampleStates[ss]));
             jobsCompleted++;
             parallelJobCount--;
          }
          else sampleStates[ss]++;

          if ((minJobWaitTime > 0) &&
              ((sampleStates[ss]-2)*minJobWaitTime>maxJobWaitTime))
          {
             sampleStates[ss] = 0;
             parallelJobCount--;
             ss--; /* roll back to the sample to be restarted */
             if (outputLevel_ > 0)
                printOutTS(PL_INFO,
                     "PSUADE adaptive(G): sample %6d to be restarted.\n",
                     ss+1);
          }
        }
      }

      //**/ -------------------------------------------------------------
      //**/ update the results in the data file
      //**/ -------------------------------------------------------------
      psuadeIO_->updateOutputSection(nSamples,nOutputs,sampleOutputs,
                                     sampleStates,NULL);
      psuadeIO_->writePsuadeFile(NULL,0);

      //**/ -------------------------------------------------------------
      //**/ increase job wait time if the jobs are too slow
      //**/ -------------------------------------------------------------
      maxState = 0;
      for (ss = 0; ss < nSamples; ss++)
        if (sampleStates[ss] > maxState) maxState = sampleStates[ss];
      if ((maxState > 100) && (minJobWaitTime <= 0)) minJobWaitTime *= 2;
      for (ss = 0; ss < nSamples; ss++)
        if (sampleStates[ss] > 10) sampleStates[ss] /= 2;

      //**/ -------------------------------------------------------------
      //**/ if not all have been completed, wait for some time
      //**/ -------------------------------------------------------------
      if ((jobsCompleted < currNJobs) && (minJobWaitTime > 0))
      {
#ifdef WINDOWS
        Sleep(1000 * minJobWaitTime);
#else
        sleep(minJobWaitTime);
#endif
      }
      if (outputLevel_ > 0)
        printOutTS(PL_INFO,
             "PSUADE adaptive(G): jobs completed = %d(of %d)\n",
             jobsCompleted, currNJobs);
    }

    //**/ ----------------------------------------------------------------
    //**/ refine one level
    //**/ ----------------------------------------------------------------
    printOutTS(PL_INFO,"Perform uniform refinement of the current sample.\n");
    samplerAux = (Sampling *) SamplingCreateFromID(samMethod);
    samplerAux->setInputBounds(nInputs, iLowerB, iUpperB);
    samplerAux->setOutputParams(nOutputs);
    samplerAux->setSamplingParams(nSamples, -1, -1);
    samplerAux->initialize(1);
    samplerAux->loadSamples(nSamples, nInputs, nOutputs, sampleInputs,
                            sampleOutputs, sampleStates);
    strcpy(sparam, "changeInfoName");
    samplerAux->setParam(sparam);
    strcpy(sparam, "setUniformRefinement");
    samplerAux->setParam(sparam);
    samplerAux->refine(refineRatio,randomize,0,0,NULL);
    nSamples2 = samplerAux->getNumSamples();
    psVector  vecSamX2, vecSamY2, vecSamU2;
    psIVector vecSamS2;
    vecSamX2.setLength(nSamples2 * nInputs);
    vecSamY2.setLength(nSamples2 * nOutputs);
    vecSamS2.setLength(nSamples2);
    vecSamU2.setLength(nSamples2);
    double *samInputs2  = vecSamX2.getDVector();
    double *samOutputs2 = vecSamY2.getDVector();
    int    *samStates2  = vecSamS2.getIVector();
    double *samStds2    = vecSamU2.getDVector();

    samplerAux->getSamples(nSamples2, nInputs, nOutputs, samInputs2,
                           samOutputs2, samStates2);

    //**/ ----------------------------------------------------------------
    //**/ response surface analysis
    //**/ ----------------------------------------------------------------
    printAsterisks(PL_INFO, 0);
    printOutTS(PL_INFO,"PSUADE adaptive(G): response surface analysis.\n");
    printOutTS(PL_INFO,
           "PSUADE adaptive(G): current level     = %d (of %d)\n",
           refineLevel,nRefinements);
    printOutTS(PL_INFO,"                    training nSamples = %d\n",
           nSamples+auxNSamples);
    printOutTS(PL_INFO,"                    test set nSamples = %d\n",
           nSamples2-nSamples);

    faPtr = genFA(rstype, nInputs, iOne, nSamples+auxNSamples);
    faPtr->setNPtsPerDim(nPtsPerDim);
    faPtr->setBounds(iLowerB, iUpperB);
    faPtr->setOutputLevel(outputLevel_);
    if ((!psConfig_.RSExpertModeIsOn()) && (rstype == PSUADE_RS_MARSB))
    {
      strcpy(cString, "mars_params");
      targv[0] = (char *) cString;
      ivar1 = (nSamples + auxNSamples);
      targv[1] = (char *) &ivar1;
      ivar2 = 2 * nInputs / 3 + 1;
      printOutTS(PL_INFO, "Set degree of interaction = 3\n");
      ivar2 = 3;
      targv[2] = (char *) &ivar2;
      faPtr->setParams(3, targv);
      strcpy(cString, "num_mars");
      targv[0] = (char *) cString;
      targv[1] = (char *) &numMars;
      faPtr->setParams(2, targv);
      if (marsMode == 1)
      {
        strcpy(cString, "median");
        targv[0] = (char *) cString;
        faPtr->setParams(1, targv);
      }
    }
    if (auxIO != NULL)
    {
      for (ss = 0; ss < nSamples*nInputs; ss++)
        auxSamInputs[auxNSamples*nInputs+ss] = sampleInputs[ss];
      for (ss = 0; ss < nSamples; ss++)
        auxSamOutputs[auxNSamples+ss] = sampleOutputs[ss];
      faPtr->initialize(auxSamInputs,auxSamOutputs);
    }
    else faPtr->initialize(sampleInputs,sampleOutputs);

    for (ss = 0; ss < nSamples; ss++) samStds2[ss] = 0;
    faPtr->evaluatePointFuzzy(nSamples2-nSamples,
                           &samInputs2[nInputs*nSamples],
                           &samOutputs2[nSamples], &samStds2[nSamples]);

    // extract the magnitude of each sample points
    totalSum = errMax = errAvg = errL2 =0.0;
    for (ss = nSamples; ss < nSamples2; ss++)
    {
      totalSum += PABS(samOutputs2[ss]);
      errMax = (samStds2[ss] > errMax) ? samStds2[ss] : errMax;
      errAvg += samStds2[ss];
      errL2  += pow(samStds2[ss], 2.0e0);
    }
    errL2 = sqrt(errL2 / (nSamples2 - nSamples));
    totalSum /= (nSamples2 - nSamples);
    errAvg  = errAvg / (nSamples2 - nSamples);
    printOutTS(PL_INFO,
       "     response surface unscaled max error = %e\n",errMax);
    printOutTS(PL_INFO,
       "     response surface   scaled max error = %e\n",errMax/totalSum);
    printOutTS(PL_INFO,
       "     response surface unscaled rms error = %e\n",errL2);
    printOutTS(PL_INFO,
       "     response surface   scaled rms error = %e\n",errL2/totalSum);
    printOutTS(PL_INFO,
       "     response surface unscaled avg error = %e\n",errAvg);
    printOutTS(PL_INFO, 
       "     response surface   scaled avg error = %e\n",errAvg/totalSum);

    //**/ ----------------------------------------------------------------
    //**/ update the sample data in the sampler
    //**/ ----------------------------------------------------------------
    refineLevel++;
    sampler_->loadSamples(nSamples, nInputs, nOutputs, sampleInputs,
                          sampleOutputs, sampleStates);
    if (errL2 < anaThreshold)
    {
      printOutTS(PL_INFO,
         "PSUADE adapive(G): threshold reached (based on unscaled L2).\n");
      break;
    }
    if (psConfig_.AnaExpertModeIsOn())
    {
      snprintf(systemCommand,100,"Do you want to quit now (y or n) ? ");
      getString(systemCommand, cString);
      if (cString[0] == 'y')
      {
        break;
      }
    }
    if (refineLevel > nRefinements)
    {
      printOutTS(PL_INFO,
           "PSUADE adapive(G): number of refinements %d reached.\n",
           nRefinements);
      break;
    }

    //**/ ----------------------------------------------------------------
    //**/ request refinement
    //**/ ----------------------------------------------------------------
    for (ss = 0; ss < nSamples2-nSamples; ss++)
      samStds2[ss] = PABS(samStds2[ss+nSamples]);
    sampler_->refine(refineRatio,randomize,refineThreshold,
                     nSamples2-nSamples,samStds2);
    psuadeIO_->updateMethodSection(-1,-1,-1,nRefinements-1,-1);
    psuadeIO_->writePsuadeFile(NULL,0);

    //**/ ----------------------------------------------------------------
    //**/ clean up
    //**/ ----------------------------------------------------------------
    delete samplerAux;
    delete faPtr;
  }

  //**/ -------------------------------------------------------------------
  //**/ clean up
  //**/ -------------------------------------------------------------------
  return 0;
}
#endif

// ************************************************************************
// deallocate variables 
// ------------------------------------------------------------------------
void PsuadeBase::cleanUp()
{
  int ii;
  if (sampler_  != NULL) SamplingDestroy(sampler_); 
  if (psuadeIO_ != NULL) delete psuadeIO_;
  sampler_ = NULL;
  psuadeIO_ = NULL;
  VecSamInputs_.clean();
  VecSamOutputs_.clean();
  VecSamStates_.clean();
  VecILowerBs_.clean();
  VecIUpperBs_.clean();
  VecInpPDFs_.clean();
  VecInpMeans_.clean();
  VecInpStds_.clean();
  StrInpNames_.clean();
  StrOutNames_.clean();
  if (inputCMat_  != NULL) delete inputCMat_;
  if (SamPDFFiles_ != NULL)
  {
    for (ii = 0; ii < nInputs_; ii++)
      if (SamPDFFiles_[ii] != NULL) delete [] SamPDFFiles_[ii];
    delete [] SamPDFFiles_;
  }
  VecTagArray_.clean();
  VecDataReg_.clean();
  VecSamPDFIndices_.clean();
  inputCMat_     = NULL;
  SamPDFFiles_   = NULL;
  nInputs_ = nSamples_ = nOutputs_ = 0;
}

// ************************************************************************
// fill in samples (for duplicates)
// ------------------------------------------------------------------------
int PsuadeBase::fillInSamples(int nSamples, int nOutputs, 
                              double *sampleOutputs, int *sampleStates)
{
  int sampleID, index, ii;

  for (sampleID = 0; sampleID < nSamples; sampleID++)
  {
    if (sampleStates[sampleID] < 0)
    {
      index = -(sampleStates[sampleID] + 1);
      if (sampleStates[index] == 1)
      {
        for (ii = 0; ii < nOutputs; ii++)
          sampleOutputs[sampleID*nOutputs+ii] =  
                  sampleOutputs[index*nOutputs+ii];  
        sampleStates[sampleID] = 1;  
      }
    }
  } 
  return 0;
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
PsuadeBase& PsuadeBase::operator=(const PsuadeBase &)
{
  printOutTS(PL_ERROR,
       "PsuadeBase operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}


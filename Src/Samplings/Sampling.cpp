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
// Functions for the class Sampling
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************

#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "sysdef.h"
#include "Sampling.h"
#include "MCSampling.h"
#include "FactorialSampling.h"
#include "LHSampling.h"
#include "OASampling.h"
#include "OALHSampling.h"
#include "MOATSampling.h"
#include "SobolSampling.h"
#include "LPtauSampling.h"
#include "MetisSampling.h"
#include "FASTSampling.h"
#include "SFASTSampling.h"
#include "BoxBehnkenSampling.h"
#include "PlackettBurmanSampling.h"
#include "FractFactSampling.h"
#include "CentralCompositeSampling.h"
#include "LSASampling.h"
#include "UserMetisSampling.h"
#include "GMOATSampling.h"
#include "GMetisSampling.h"
#include "SparseGridSampling.h"
#include "DiscreteSampling.h"
#include "RFractFactSampling.h"
#include "PrintingTS.h"
#include "PsuadeUtil.h"

// ------------------------------------------------------------------------
// local includes : The OTHER way to do sampling...
// ------------------------------------------------------------------------
#include "PDFManager.h"
#include "PDFBase.h"
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
Sampling::Sampling()
{
  printLevel_    = 0;
  samplingID_    = -1;
  nSamples_      = 0;
  nInputs_       = 0;
  nOutputs_      = 0;
  randomize_     = 0;
  nReplications_ = 1;
  vecInpSettings_ = NULL;
}

// ************************************************************************
// Copy Constructor created by Bill Oliver 
// ------------------------------------------------------------------------
Sampling::Sampling(const Sampling & smp)
{
  printLevel_ = smp.printLevel_;
  samplingID_ = smp.samplingID_;
  nSamples_ = smp.nSamples_;
  nInputs_ = smp.nInputs_;
  nOutputs_ = smp.nOutputs_;
  randomize_ = smp.randomize_;
  nReplications_ = smp.nReplications_;
  vecLBs_ = smp.vecLBs_;
  vecUBs_ = smp.vecUBs_;
  vecSamInps_ = smp.vecSamInps_;
  vecSamOuts_ = smp.vecSamOuts_;
  vecSamStas_ = smp.vecSamStas_;
  vecSymTable_ = smp.vecSymTable_;
  vecInpNumSettings_ = smp.vecInpNumSettings_;
  if (smp.vecInpSettings_ != NULL)
  {
    vecInpSettings_ = new psVector[nInputs_];
    for (int ii = 0; ii < nInputs_; ii++)
      vecInpSettings_[ii] = smp.vecInpSettings_[ii];
  }
  maxNumSettings_ = smp.maxNumSettings_;
} 

// ************************************************************************
// Overload operator=  by Bill Oliver 
// ------------------------------------------------------------------------
Sampling & Sampling::operator=(const Sampling & smp)
{
  if (this == & smp) return *this;

  printLevel_ = smp.printLevel_;
  samplingID_ = smp.samplingID_;
  nSamples_ = smp.nSamples_;
  nInputs_ = smp.nInputs_;
  nOutputs_ = smp.nOutputs_;
  randomize_ = smp.randomize_;
  nReplications_ = smp.nReplications_;
  vecLBs_ = smp.vecLBs_;
  vecUBs_ = smp.vecUBs_;
  vecSamInps_ = smp.vecSamInps_;
  vecSamOuts_ = smp.vecSamOuts_;
  vecSamStas_ = smp.vecSamStas_;
  vecSymTable_ = smp.vecSymTable_;
  vecInpNumSettings_ = smp.vecInpNumSettings_;
  if (smp.vecInpSettings_ != NULL)
  {
    vecInpSettings_ = new psVector[nInputs_];
    for (int ii = 0; ii < nInputs_; ii++)
      vecInpSettings_[ii] = smp.vecInpSettings_[ii];
  }
  maxNumSettings_ = smp.maxNumSettings_;
  return *this;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
Sampling::~Sampling()
{
  if (vecInpSettings_ != NULL)
  {
    for (int ii = 0; ii < nInputs_; ii++)
      vecInpSettings_[ii].clean();
    delete [] vecInpSettings_;
  }
}

// ************************************************************************
// This function fully constructs a sample from psuadeIO_.
// All the common functionality is done in the Sampling class, subclasses 
// should overwrite this function. Currently this function does not support 
// adaptive sampling or refinement, although it probably could.  
// (The psuadeIO object has all the info for it)
// @param psuadeIO_: A Psuade Data file, should have Input/Output variables 
// and Method defined.
// ------------------------------------------------------------------------
int Sampling::doSampling(PsuadeData* psuadeIO_) 
{
  pData  pPtr, pSymTable, pLowerB, pUpperB, pOutputs, pInputs;
  pData  pStates, pPDFs, pSettings;

  int outputLevel_ = 0;
  psuadeIO_->getParameter("ana_diagnostics", pPtr);
  outputLevel_ = pPtr.intData_;

  psuadeIO_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  psuadeIO_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  psuadeIO_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  psuadeIO_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  psuadeIO_->getParameter("input_symtable", pSymTable);
  int *symTable = pSymTable.intArray_;

  psuadeIO_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  psuadeIO_->getParameter("method_randomflag", pPtr);
  int randomize = pPtr.intData_;
  psuadeIO_->getParameter("method_nreplications", pPtr);
  int nReps = pPtr.intData_;  

  psuadeIO_->getParameter("method_sampling", pPtr);
  int samplingMethod = pPtr.intData_;
  psuadeIO_->getParameter("input_use_input_pdfs", pPtr);
  int usePDFs = pPtr.intData_;
  psuadeIO_->getParameter("input_pdfs", pPDFs);
  int *inputPDFs = pPDFs.intArray_;
  psuadeIO_->getParameter("method_nrefinements", pPtr);
  int nRefines = pPtr.intData_;

  psuadeIO_->getParameter("input_settings", pSettings);
  int haveSettings = pSettings.nInts_;

  //**/ Blow away any existing samples:
  psuadeIO_->resetSamples();

  //**/ Set up the Sampler
  setPrintLevel(outputLevel_);
  setInputBounds(nInputs, iLowerB, iUpperB);
  setOutputParams(nOutputs);
  setInputParams(nInputs, NULL, NULL, symTable);
  setSamplingParams(nSamples, nReps, randomize);
  if (haveSettings) 
  {
    setInputParams(nInputs, pSettings.intArray_,
                   pSettings.dbleArray2D_, NULL);
  }

  //**/ There are two ways to sample, the normal way, and the PDF way
  //**/ Figure out if we're supposed to do the PDF way
  int  iSum = 0;
  for (int ii = 0; ii < nInputs; ii++) iSum += inputPDFs[ii];
  bool doingPDFs = false;
  if (iSum > 0 && usePDFs) doingPDFs = true;
  
  //**/ I wish PDFs were part of the normal sampling class hierarchy
  if (doingPDFs) 
  {
    if (nRefines > 0)
    {
      printf("PSUADE ERROR: Both sample refinement and PDFs ");
      printf("are requested. Sample\n");
      printf("       refinement requires that all input ");
      printf("probability distributions be\n");
      printf("       uniform. You need to either reset the ");
      printf("number of refinements to\n");
      printf("       be zero, or not take out the probability ");
      printf("distributions.\n");
      exit(1);
    }

    printf("Sampling INFO: Creating a sample where some uncertain ");
    printf("parameters are\n");
    printf("               not uniformly distributed.\n");
    PDFManager *pdfman;
    pdfman = new PDFManager();
    pdfman->initialize(psuadeIO_);
    pdfman->genSample();
    delete pdfman;
  
    pData pPtr;
    psuadeIO_->getParameter("input_sample", pPtr);
    double *sampleInputs = pPtr.dbleArray_;
    psuadeIO_->getParameter("output_sample", pPtr);
    double *sampleOutputs = pPtr.dbleArray_;
    psuadeIO_->getParameter("output_states", pPtr);
    int *sampleStates = pPtr.intArray_;

    initialize(1);
    loadSamples(nSamples, nInputs, nOutputs, sampleInputs,
                 sampleOutputs, sampleStates);
  }
  else
  {
    printf("Sampling INFO: Creating a sample assuming all uncertain ");
    printf("parameters are\n");
    printf("               uniformly distributed.\n");

    psVector  vecSamInps, vecSamOuts;
    psIVector vecSamStas;

    initialize(0);
    nSamples = getNumSamples(); /* may have been modified */
    vecSamInps.setLength(nSamples * nInputs);
    vecSamOuts.setLength(nSamples * nOutputs);
    vecSamStas.setLength(nSamples);
    getSamples(nSamples, nInputs, nOutputs, vecSamInps.getDVector(),
               vecSamOuts.getDVector(), vecSamStas.getIVector());
    psuadeIO_->updateInputSection(nSamples,nInputs,NULL,NULL,NULL,
                     vecSamInps.getDVector(),NULL,NULL,NULL,NULL,NULL); 
    psuadeIO_->updateOutputSection(nSamples,nOutputs,
                   vecSamOuts.getDVector(),vecSamStas.getIVector(),NULL); 
  }
  return 0;
}

// ************************************************************************
// get print level
// ------------------------------------------------------------------------
int Sampling::setPrintLevel(int level)
{
  printLevel_ = level;
  return 0;
}

// ************************************************************************
// setInputBounds : set up the input lower and upper bounds
// ------------------------------------------------------------------------
int Sampling::setInputBounds(int nInputs, double *lower, double *upper)
{
  if (nInputs_ != 0 && nInputs != nInputs_)
  {
    printf("Sampling::setInputBounds ERROR : nInputs mismatch.\n");
    exit(1);
  }
  nInputs_ = nInputs;
  vecLBs_.setLength(nInputs_);
  vecUBs_.setLength(nInputs_);
  for (int jj = 0; jj < nInputs_; jj++) 
  {
    vecLBs_[jj] = lower[jj];
    vecUBs_[jj] = upper[jj];
    if (lower[jj] >= upper[jj]) 
    {
      printf("Sampling::setInputBounds ERROR : lbound >= ubound.\n");
      printf("          Input %d : %e %e\n", jj+1, lower[jj], upper[jj]);
      exit(1);
    }    
    if (vecLBs_[jj] >= vecUBs_[jj]) 
    {
      printf("Sampling::setInputBounds ERROR : lbound >= ubound.\n");
      printf("          Input %d : %e %e\n",jj+1,vecLBs_[jj],vecUBs_[jj]);
      exit(1);
    }    
  }
  return 0;
}

// ************************************************************************
// setSamplingParams : set up other sampling parameters
// ------------------------------------------------------------------------
int Sampling::setSamplingParams(int nSamples, int nReps, int randomize)
{
  if (nSamples_ != 0 && nSamples_ != nSamples)
  {
    printf("Sampling::setSamplingParams ERROR : nSamples mismatch.\n");
    printf("          nSamples = %d versus %d\n", nSamples, nSamples_);
    exit(1);
  }
  nSamples_ = nSamples;
  if (nReps > 0) nReplications_ = nReps;
  if (randomize > 0) randomize_ = randomize;
  vecSamInps_.setLength(0);
  return 0;
}

// ************************************************************************
// setInputParams : set up the input settings
// ------------------------------------------------------------------------
int Sampling::setInputParams(int, int *, double **, int *)
{
  return -1;
}

// ************************************************************************
// setOutputParams : set up the output settings
// ------------------------------------------------------------------------
int Sampling::setOutputParams(int nOutputs)
{
  if (nOutputs_ != 0 && nOutputs_ != nOutputs)
  {
    printf("Sampling::setOutputParams ERROR : nOutputs mismatch.\n");
    exit(1);
  }
  nOutputs_ = nOutputs;
  return 0;
}

// ************************************************************************
// load samples
// ------------------------------------------------------------------------
int Sampling::loadSamples(int nSamples, int nInputs, int nOutputs,
                       double *samples, double *outputs, int *states)
{
  int ii, jj, offset;

  //**/ ----------------------------------------------------------------
  //**/ clean up and then error checking 
  //**/ ----------------------------------------------------------------
  if (nSamples != nSamples_) 
  {
    printf("Sampling::loadSamples WARNING: mismatched nSamples. \n");
    printf("          nSamples %d reset to incoming nSamples %d. \n",
           nSamples_, nSamples);
    nSamples_ = nSamples;
  }
  if (nInputs != nInputs_) 
  {
    printf("Sampling::loadSamples WARNING: invalid nInputs (%d,%d).\n",
           nInputs, nInputs_);
    exit(1);
  }
  if (nOutputs != nOutputs_) 
  {
    printf("Sampling::loadSamples WARNING: invalid nOutputs. \n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ load samples
  //**/ ----------------------------------------------------------------
  allocSampleData();
  for (ii = 0; ii < nSamples_; ii++) 
  {
    offset = ii * nInputs_;
    for (jj = 0; jj < nInputs_; jj++) 
      vecSamInps_[offset+jj] = samples[offset+jj];
    vecSamStas_[ii] = states[ii];
  }
  for (ii = 0; ii < nSamples_*nOutputs_; ii++)
    vecSamOuts_[ii] = outputs[ii];
  return 0;
}

// ************************************************************************
// get number of samples
// ------------------------------------------------------------------------
int Sampling::getNumSamples()
{
  return nSamples_;
}

// ************************************************************************
// get number of inputs
// ------------------------------------------------------------------------
int Sampling::getNumInputs()
{
  return nInputs_;
}

// ************************************************************************
// get number of outputs
// ------------------------------------------------------------------------
int Sampling::getNumOutputs()
{
  return nOutputs_;
}

// ************************************************************************
// get all sample points 
// ------------------------------------------------------------------------
int Sampling::getSamples(int nSamples, int nInputs, int nOutputs,
                         double *sampData, double *outputs, int *states)
{
  int ii, jj, offset, offset2;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples != nSamples_) 
  {
    printf("Sampling::getSamples ERROR : invalid nSamples. \n");
    printf("Sampling::loadSamples WARNING: invalid nInputs (%d,%d).\n",
           nSamples,nSamples_);
    exit(1);
  }
  if (nInputs != nInputs_) 
  {
    printf("Sampling::getSamples ERROR : invalid nInputs. \n");
    exit(1);
  }
  if (nOutputs_ != nOutputs) 
  {
    printf("Sampling::getSamples ERROR : invalid nOutputs. \n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ copy sample data
  //**/ ----------------------------------------------------------------
  if (vecSamInps_.length() == nSamples*nInputs)
  {
    for (ii = 0; ii < nSamples; ii++) 
    {
      offset = ii * nInputs;
      offset2 = ii * nOutputs;
      for (jj = 0; jj < nInputs; jj++) 
        sampData[offset+jj] = vecSamInps_[offset+jj];
      if (vecSamOuts_.length() == nSamples*nOutputs)
      {
        for (jj = 0; jj < nOutputs; jj++) 
          outputs[offset2+jj] = vecSamOuts_[offset2+jj];
      }
      if (vecSamStas_.length() == nSamples) states[ii] = vecSamStas_[ii];
    }
    return nSamples;
  }
  return 0;
}

// ************************************************************************
// get all sample points (this is for library call - no state information)
// ------------------------------------------------------------------------
int Sampling::getSamples(int nSamples, int nInputs, int nOutputs,
                         double *sampData, double *outputs)
{
  int ii, jj, offset, offset2;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  if (nSamples != nSamples_) 
  {
    printf("Sampling::getSamples ERROR : invalid nSamples. \n");
    exit(1);
  }
  if (nInputs != nInputs_) 
  {
    printf("Sampling::getSamples ERROR : invalid nInputs. \n");
    exit(1);
  }
  if (nOutputs_ != nOutputs) 
  {
    printf("Sampling::getSamples ERROR : invalid nOutputs. \n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ copy sample data
  //**/ ----------------------------------------------------------------
  if (vecSamInps_.length() == nSamples*nInputs)
  {
    for (ii = 0; ii < nSamples; ii++) 
    {
      offset = ii * nInputs;
      offset2 = ii * nOutputs;
      for (jj = 0; jj < nInputs; jj++) 
        sampData[offset+jj] = vecSamInps_[offset+jj];
      if (vecSamOuts_.length() == nSamples*nOutputs)
      {
        for (jj = 0; jj < nOutputs; jj++) 
          outputs[offset2+jj] = vecSamOuts_[offset2+jj];
      }
    }
    return nSamples;
  }
  return 0;
}

// ************************************************************************
// store sample outputs
// ------------------------------------------------------------------------
int Sampling::storeResult(int sampleID, int nOutputs, double *outputs, 
                          int *state)
{
  int jj;

  //**/ ----------------------------------------------------------------
  //**/ error checking
  //**/ ----------------------------------------------------------------
  //if (sampleID < 0 || sampleID >= nSamples_) 
  //{
  //  printf("Sampling::storeResult ERROR : invalid sample ID. \n");
  //  exit(1);
  //}
  //if (nOutputs != nOutputs_) 
  //{
  //  printf("Sampling::storeResult ERROR : invalid nOutputs. \n");
  //  exit(1);
  //}

  //**/ ----------------------------------------------------------------
  //**/ store results
  //**/ ----------------------------------------------------------------
  if (vecSamOuts_.length() == nSamples_*nOutputs_)
  {
    for (jj = 0; jj < nOutputs_; jj++)
      vecSamOuts_[sampleID*nOutputs_+jj] = outputs[jj];
    vecSamStas_[sampleID] = 1;
    (*state) = 1;
    return 0;
  }
  return 0;
}

// ************************************************************************
// perform refinement 
// ------------------------------------------------------------------------
int Sampling::refine(int, int, double, int, double *)
{
  return -1;
}

// ************************************************************************
// perform repair 
// ------------------------------------------------------------------------
int Sampling::repair(char *, int)
{
  return -1;
}

// ************************************************************************
// allocate sample matrix, output, and states
// ------------------------------------------------------------------------
int Sampling::allocSampleData()
{
  int ii;
  vecSamInps_.setLength(nSamples_*nInputs_);
  vecSamOuts_.setLength(nSamples_*nOutputs_);
  vecSamStas_.setLength(nSamples_);
  for (ii = 0; ii < nSamples_*nOutputs_; ii++)
    vecSamOuts_[ii] = PSUADE_UNDEFINED;
  return 0;
}

// ************************************************************************
// set parameter
// ------------------------------------------------------------------------
int Sampling::setParam(char *)
{
  return -1;
}

// ************************************************************************
// set parameter
// ------------------------------------------------------------------------
int Sampling::setParam(int, char **)
{
  return -1;
}

// ************************************************************************
// create a sampling scheme from identifier
// ------------------------------------------------------------------------
Sampling *SamplingCreateFromID(int samplingMethod)
{
  Sampling *sampler;
  char     sparam[101];

  switch (samplingMethod)
  {
    case PSUADE_SAMP_MC: 
         sampler = (Sampling *) new MCSampling();
         break;
    case PSUADE_SAMP_FACT:
         sampler = (Sampling *) new FactorialSampling();
         break;
    case PSUADE_SAMP_LHS:
         sampler = (Sampling *) new LHSampling();
         break;
    case PSUADE_SAMP_OA:
         sampler = (Sampling *) new OASampling();
         break;
    case PSUADE_SAMP_OALH:
         sampler = (Sampling *) new OALHSampling();
         break;
    case PSUADE_SAMP_MOAT:
         sampler = (Sampling *) new MOATSampling();
         break;
    case PSUADE_SAMP_SOBOL:
         sampler = (Sampling *) new SobolSampling();
         break;
    case PSUADE_SAMP_LPTAU:
         sampler = (Sampling *) new LPtauSampling();
         break;
    case PSUADE_SAMP_METIS:
         sampler = (Sampling *) new MetisSampling();
         break;
    case PSUADE_SAMP_FAST:
         sampler = (Sampling *) new FASTSampling();
         break;
    case PSUADE_SAMP_BBD:
         sampler = (Sampling *) new BoxBehnkenSampling();
         break;
    case PSUADE_SAMP_PBD:
         sampler = (Sampling *) new PlackettBurmanSampling();
         break;
    case PSUADE_SAMP_FF4:
         sampler = (Sampling *) new FractFactSampling();
         strcpy(sparam, "setResolution 4");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_FF5:
         sampler = (Sampling *) new FractFactSampling();
         strcpy(sparam, "setResolution 5");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCI4:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 4");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 0");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCI5:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 5");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 0");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCIF:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 0");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 0");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCF4:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 4");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 1");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCF5:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 5");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 1");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCFF:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 0");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 1");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCC4:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 4");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 2");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCC5:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 5");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 2");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_CCCF:
         sampler = (Sampling *) new CentralCompositeSampling();
         strcpy(sparam, "setResolution 0");
         sampler->setParam(sparam);
         strcpy(sparam, "setScheme 2");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_SFAST:
         sampler = (Sampling *) new SFASTSampling();
         break;
    case PSUADE_SAMP_UMETIS:
         sampler = (Sampling *) new UserMetisSampling();
         break;
    case PSUADE_SAMP_GMOAT:
         sampler = (Sampling *) new GMOATSampling();
         break;
    case PSUADE_SAMP_GMETIS:
         sampler = (Sampling *) new GMetisSampling();
         break;
    case PSUADE_SAMP_SG:
         sampler = (Sampling *) new SparseGridSampling();
         break;
    case PSUADE_SAMP_DISCRETE:
         sampler = (Sampling *) new DiscreteSampling();
         break;
    case PSUADE_SAMP_LSA:
         sampler = (Sampling *) new LSASampling();
         break;
    case PSUADE_SAMP_RFF4:
         sampler = (Sampling *) new RFractFactSampling();
         strcpy(sparam, "setResolution 4");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_RFF5:
         sampler = (Sampling *) new RFractFactSampling();
         strcpy(sparam, "setResolution 5");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_SOBOL2:
         sampler = (Sampling *) new SobolSampling();
         strcpy(sparam, "setOrder 2");
         sampler->setParam(sparam);
         break;
    case PSUADE_SAMP_SOBOLG:
         sampler = (Sampling *) new SobolSampling();
         strcpy(sparam, "setOrder 3");
         sampler->setParam(sparam);
         break;
    default :
         printf("PSUADE::SamplingCreateFromID ERROR - ");
         printf("invalid sampling method (%d).\n",samplingMethod);
         exit(1);
         break;
  }
  return sampler;
}

// ************************************************************************
// destroy a sampling object 
// ------------------------------------------------------------------------
int SamplingDestroy(Sampling *sampler)
{
  MCSampling               *MCSampler;
  FactorialSampling        *FactSampler;
  LHSampling               *LHSampler;
  OASampling               *OASampler;
  OALHSampling             *OALHSampler;
  MOATSampling             *MOATSampler;
  SobolSampling            *SobolSampler;
  LPtauSampling            *LPtauSampler;
  MetisSampling            *MetisSampler;
  FASTSampling             *FastSampler;
  SFASTSampling            *SFastSampler;
  BoxBehnkenSampling       *BBDSampler;
  PlackettBurmanSampling   *PBDSampler;
  FractFactSampling        *FFSampler;
  CentralCompositeSampling *CCDSampler;
  UserMetisSampling        *UMetisSampler;
  GMOATSampling            *GMOATSampler;
  GMetisSampling           *GMetisSampler;
  SparseGridSampling       *SparseGridSampler;
  DiscreteSampling         *DiscreteSampler;
  LSASampling              *LSASampler;
  RFractFactSampling       *RFFSampler;

  switch (sampler->samplingID_)
  {
    case PSUADE_SAMP_MC: 
         MCSampler = (MCSampling *) sampler;
         delete MCSampler;
         break;
    case PSUADE_SAMP_FACT:
         FactSampler = (FactorialSampling *) sampler;
         delete FactSampler;
         break;
    case PSUADE_SAMP_LHS:
         LHSampler = (LHSampling *) sampler;
         delete LHSampler;
         break;
    case PSUADE_SAMP_OA:
         OASampler = (OASampling *) sampler;
         delete OASampler;
         break;
    case PSUADE_SAMP_OALH:
         OALHSampler = (OALHSampling *) sampler;
         delete OALHSampler;
         break;
    case PSUADE_SAMP_MOAT:
          MOATSampler = (MOATSampling *) sampler;
          delete MOATSampler;
          break;
    case PSUADE_SAMP_SOBOL:
         SobolSampler = (SobolSampling *) sampler;
         delete SobolSampler;
         break;
    case PSUADE_SAMP_LPTAU:
         LPtauSampler = (LPtauSampling *) sampler;
         delete LPtauSampler;
         break;
    case PSUADE_SAMP_METIS:
         MetisSampler = (MetisSampling *) sampler;
         delete MetisSampler;
         break;
    case PSUADE_SAMP_FAST:
         FastSampler = (FASTSampling *) sampler;
         delete FastSampler;
         break;
    case PSUADE_SAMP_BBD:
         BBDSampler = (BoxBehnkenSampling *) sampler;
         delete BBDSampler;
         break;
    case PSUADE_SAMP_PBD:
         PBDSampler = (PlackettBurmanSampling *) sampler;
         delete PBDSampler;
         break;
    case PSUADE_SAMP_FF4:
         FFSampler = (FractFactSampling *) sampler;
         delete FFSampler;
         break;
    case PSUADE_SAMP_FF5:
         FFSampler = (FractFactSampling *) sampler;
         delete FFSampler;
         break;
    case PSUADE_SAMP_CCI4:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCI5:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCIF:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCF4:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCF5:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCFF:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCC4:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCC5:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_CCCF:
         CCDSampler = (CentralCompositeSampling *) sampler;
         delete CCDSampler;
         break;
    case PSUADE_SAMP_SFAST:
         SFastSampler = (SFASTSampling *) sampler;
         delete SFastSampler;
         break;
    case PSUADE_SAMP_UMETIS:
         UMetisSampler = (UserMetisSampling *) sampler;
         delete UMetisSampler;
         break;
    case PSUADE_SAMP_GMOAT:
         GMOATSampler = (GMOATSampling *) sampler;
         delete GMOATSampler;
         break;
    case PSUADE_SAMP_GMETIS:
         GMetisSampler = (GMetisSampling *) sampler;
         delete GMetisSampler;
         break;
    case PSUADE_SAMP_SG:
         SparseGridSampler = (SparseGridSampling *) sampler;
         delete SparseGridSampler;
         break;
    case PSUADE_SAMP_DISCRETE:
         DiscreteSampler = (DiscreteSampling *) sampler;
         delete DiscreteSampler;
         break;
    case PSUADE_SAMP_LSA:
         LSASampler = (LSASampling *) sampler;
         delete LSASampler;
         break;
    case PSUADE_SAMP_RFF4:
         RFFSampler = (RFractFactSampling *) sampler;
         delete RFFSampler;  
         break;
    case PSUADE_SAMP_RFF5:
         RFFSampler = (RFractFactSampling *) sampler;
         delete RFFSampler;
         break;
    case PSUADE_SAMP_SOBOL2:
         SobolSampler = (SobolSampling *) sampler;
         delete SobolSampler;
         break;
    case PSUADE_SAMP_SOBOLG:
         SobolSampler = (SobolSampling *) sampler;
         delete SobolSampler;
         break;
  }
  return 0;
}

// ************************************************************************
// compute sample quality
// ------------------------------------------------------------------------
int SamplingQuality(int nSamples, int nInputs, double *sampleInputs,
                    double *lbnds, double *ubnds, double *quality, 
                    int printFlag)
{
  int    ss1, ss2, ii, numPerms, minIndex;
  double distance, minDistance, dtemp, minMinDist, avgDist;
  psVector vecDT;

  //**/ find minimum of the minimum distance between points
  minMinDist =  1.0e35;
  for (ss1 = 0; ss1 < nSamples; ss1++)
  {
    minDistance = 1.0e35;
    for (ss2 = 0; ss2 < nSamples; ss2++)
    {
      if (ss1 != ss2)
      {
        distance = 0.0;
        for (ii = 0; ii < nInputs; ii++)
        {
          dtemp = sampleInputs[ss1*nInputs+ii] - 
                  sampleInputs[ss2*nInputs+ii];
          distance += pow(dtemp/(ubnds[ii]-lbnds[ii]), 2.0);
        }
        distance = sqrt(distance);
        if (distance < minDistance) minDistance = distance;
      }
    }
    if (minDistance < minMinDist) minMinDist = minDistance;
  }
  quality[0] = minMinDist;

  //**/ find average of minimum distance between corners and samples
  if (nInputs > 30)
  {
    printf("Avg-avg distance is not calculated for nInputs > 30.\n");
    quality[1] = -1;
  }
  else
  {
    numPerms = 1;
    for (ii = 0; ii < nInputs; ii++) numPerms *= 2;
    vecDT.setLength(nInputs);
    for (ss1 = 0; ss1 < numPerms; ss1++)
    {
      for (ii = 0; ii < nInputs; ii++) 
      {
        if (ss1 & (1 << ii)) vecDT[ii] = 1.0;
        else                 vecDT[ii] = 0.0;
      }
      minDistance = 1.0e35;
      for (ss2 = 0; ss2 < nSamples; ss2++)
      {
        distance = 0.0;
        for (ii = 0; ii < nInputs; ii++)
        {
          dtemp = (sampleInputs[ss2*nInputs+ii] - lbnds[ii]) /
                  (ubnds[ii] - lbnds[ii]);
          dtemp -= vecDT[ii];
          distance += dtemp * dtemp;
        }
        distance = sqrt(distance);
        if (distance < minDistance) minDistance = distance;
      }  
      avgDist += minDistance;
    }
    quality[1] = minDistance / (double) numPerms;
  }
 
  //**/ find average of distances between points
  avgDist =  0.0;
  for (ss1 = 0; ss1 < nSamples; ss1++)
  {
    for (ss2 = 0; ss2 < nSamples; ss2++)
    {
      if (ss1 != ss2)
      {
        distance = 0.0;
        for (ii = 0; ii < nInputs; ii++)
        {
          dtemp = sampleInputs[ss1*nInputs+ii] - 
                  sampleInputs[ss2*nInputs+ii];
          distance += pow(dtemp/(ubnds[ii]-lbnds[ii]), 2.0);
        }
        distance = sqrt(distance);
        avgDist += distance;
      }
    }
  }
  avgDist /= (1.0 * (nSamples) * (nSamples - 1.0));
  quality[2] = avgDist;

  //**/ find the shortest distance between each point to the rest
  for (ss1 = 0; ss1 < nSamples; ss1++)
  {
    minDistance = 1.0e35;
    minIndex = -1;
    for (ss2 = 0; ss2 < nSamples; ss2++)
    {
      if (ss1 != ss2)
      {
        distance = 0.0;
        for (ii = 0; ii < nInputs; ii++)
        {
          dtemp = sampleInputs[ss1*nInputs+ii] - 
                  sampleInputs[ss2*nInputs+ii];
          distance += pow(dtemp/(ubnds[ii]-lbnds[ii]), 2.0);
        }
        distance = sqrt(distance);
        if (distance < minDistance) 
        {
          minDistance = distance;
          minIndex = ss2;
        }
      }
    }
    if (printFlag > 0)
      printf("Sample %6d: min distance is %10.4e from sample %d\n",
             ss1+1,minDistance,minIndex+1);
  }
  return 0;
}

// ************************************************************************
// ask user for a sampling method 
// ------------------------------------------------------------------------
int getSamplingMethod(char *pString)
{
  printDashes(PL_INFO, 0);
  printf("Available sampling methods: \n");
  printDashes(PL_INFO, 0);
  printf("1. Monte Carlo sampling\n");
  printf("2. Full Factorial sampling\n");
  printf("3. Latin hypercube sampling\n");
  printf("4. Orthogonal array sampling\n");
  printf("5. Morris sampling\n");
  printf("6. Quasi Monte Carlo (LPTAU) sampling\n");
  printf("7. Full space-filling (METIS) sampling\n");
  printf("8. Sparse grid sampling\n");
  int samType = getInt(1, 8, pString);
  if      (samType == 1) samType = PSUADE_SAMP_MC;
  else if (samType == 2) samType = PSUADE_SAMP_FACT;
  else if (samType == 3) samType = PSUADE_SAMP_LHS;
  else if (samType == 4) samType = PSUADE_SAMP_OA;
  else if (samType == 5) samType = PSUADE_SAMP_MOAT;
  else if (samType == 6) samType = PSUADE_SAMP_LPTAU;
  else if (samType == 7) samType = PSUADE_SAMP_METIS;
  else if (samType == 8) samType = PSUADE_SAMP_SG;
  return samType;
}

// ************************************************************************
// check sample smoothness with a sample itself 
// ------------------------------------------------------------------------
int CheckSampleSmoothness(psVector VecX, psVector VecY, psVector VecXL,
         psVector VecXU, int nNeigh, psVector &VecMeas,psVector &VecSD,int)
{
  int       nSamples, nInputs, ii, ss, ss2, ind, count, nNeighbors;
  double    range, dist;
  psVector  VecXN, VecDist, VecDTmp;
  psIVector VecITmp, VecList;

  //**/ ---------------------------------------------------------------
  //**/ normalize sample inputs
  //**/ ---------------------------------------------------------------
  nNeighbors = nNeigh;
  nSamples = VecY.length();
  nInputs  = VecX.length() / nSamples;
  VecXN.setLength(VecX.length());
  for (ii = 0; ii < nInputs; ii++)
  {
    range = 1.0 / (VecXU[ii] - VecXL[ii]);
    for (ss = 0; ss < nSamples; ss++)
      VecXN[ss*nInputs+ii] = (VecX[ss*nInputs+ii] - VecXL[ii]) * range;
  }

  //**/ ---------------------------------------------------------------
  //**/ find pairwise distances 
  //**/ ---------------------------------------------------------------
  VecDist.setLength(nSamples * nSamples);
  for (ss = 0; ss < nSamples; ss++)
  {
    for (ss2 = 0; ss2 < ss-1; ss2++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs; ii++)
        dist += pow(VecXN[ss*nInputs+ii] - VecXN[ss2*nInputs+ii], 2.0);
      dist = sqrt(dist);
      VecDist[ss*nSamples+ss2] = dist;
      VecDist[ss2*nSamples+ss] = dist;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ generate statistics for each sample point
  //**/ ---------------------------------------------------------------
  VecList.setLength(nNeighbors);
  VecDTmp.setLength(nSamples);
  VecITmp.setLength(nSamples);
  VecMeas.setLength(nSamples);
  VecSD.setLength(nSamples);
  for (ss = 0; ss < nSamples; ss++)
  {
    for (ss2 = 0; ss2 < nSamples; ss2++)
    {
      VecDTmp[ss2] = VecDist[ss*nSamples+ss2];
      VecITmp[ss2] = ss2;
    }
    sortDbleList2a(nSamples, VecDTmp.getDVector(), VecITmp.getIVector());
    count = 0;
    ii = 1;
    while (count < nNeighbors && ii < nSamples)
    {
      ind = VecITmp[ii];
      dist = VecDist[ss*nSamples+ind];
      if (dist > 0)
      {
        VecMeas[ss] += PABS((VecY[ss] - VecY[ind]) / dist);
        count++;
        if (count == 1) VecSD[ss] = dist;
      }
      ii++;
    }
    VecMeas[ss] /= (double) nNeighbors; 
  }
  return 0;
}

// ************************************************************************
// check sample smoothness across samples 
// ------------------------------------------------------------------------
int CheckSampleSmoothness2(psVector VecX, psVector VecY, psVector VecXL,
            psVector VecXU, psVector VecX2, psVector VecY2, int nNeigh, 
            psVector &VecMeas, psVector &VecSD, int printLevel)
{
  int       nSamples,nSamples2,nInputs,ii,ss,ss2,ind,count,nNeighbors;
  double    range, dist, ddata;
  psVector  VecXN, VecXN2, VecDist, VecDTmp;
  psIVector VecITmp, VecList;
  FILE      *fp=NULL;

  //**/ ---------------------------------------------------------------
  //**/ normalize sample inputs
  //**/ ---------------------------------------------------------------
  if (printLevel > 0) fp = fopen("ssc_details", "w");
  nNeighbors = nNeigh;
  nSamples  = VecY.length();
  nInputs   = VecX.length() / nSamples;
  nSamples2 = VecX2.length() / nInputs;
  VecXN.setLength(VecX.length());
  VecXN2.setLength(VecX2.length());
  for (ii = 0; ii < nInputs; ii++)
  {
    range = 1.0 / (VecXU[ii] - VecXL[ii]);
    for (ss = 0; ss < nSamples; ss++)
      VecXN[ss*nInputs+ii] = 
         (VecX[ss*nInputs+ii] - VecXL[ii]) * range;
    for (ss = 0; ss < nSamples2; ss++)
      VecXN2[ss*nInputs+ii] = 
         (VecX2[ss*nInputs+ii] - VecXL[ii]) * range;
  }

  //**/ ---------------------------------------------------------------
  //**/ find pairwise distances 
  //**/ ---------------------------------------------------------------
  VecDist.setLength(nSamples * nSamples2);
  for (ss = 0; ss < nSamples; ss++)
  {
    for (ss2 = 0; ss2 < nSamples2; ss2++)
    {
      dist = 0.0;
      for (ii = 0; ii < nInputs; ii++)
        dist += pow(VecXN[ss*nInputs+ii] - VecXN2[ss2*nInputs+ii], 2.0);
      VecDist[ss2*nSamples+ss] = dist;
    }
  }

  //**/ ---------------------------------------------------------------
  //**/ generate statistics for each sample point
  //**/ ---------------------------------------------------------------
  VecList.setLength(nNeighbors);
  VecDTmp.setLength(nSamples);
  VecITmp.setLength(nSamples);
  VecMeas.setLength(nSamples2);
  VecSD.setLength(nSamples2);
  if (fp != NULL) 
    fprintf(fp,"%%Each sample lists its neighbor, distance, gradient\n");
  for (ss2 = 0; ss2 < nSamples2; ss2++)
  {
    if (fp != NULL) fprintf(fp, "Sample %d: \n", ss2+1);
    for (ss = 0; ss < nSamples; ss++)
    {
      VecDTmp[ss] = VecDist[ss2*nSamples+ss];
      VecITmp[ss] = ss;
    }
    sortDbleList2a(nSamples, VecDTmp.getDVector(), VecITmp.getIVector());
    count = 0;
    ii = 1;
    while (count < nNeighbors && ii < nSamples)
    {
      ind = VecITmp[ii];
      dist = VecDist[ss2*nSamples+ind];
      if (dist > 0)
      {
        ddata = PABS((VecY2[ss2] - VecY[ind]) / dist);
        if (fp != NULL) 
          fprintf(fp, " %8d  %12.4e  %12.4e %12.4e %12.4e\n",ind+1,
                  dist,ddata,VecY[ind],VecY2[ss2]);
        VecMeas[ss2] += ddata; 
        count++;
        if (count == 1) VecSD[ss2] = dist;
      }
      ii++;
    }
    VecMeas[ss2] /= (double) nNeighbors; 
  }
  if (fp != NULL) 
  {
    fclose(fp);
    printf("ssc_details has more detailed ssc information\n");
  }
  return 0;
}


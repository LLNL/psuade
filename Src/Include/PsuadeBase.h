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
// Definition for the PsuadeBase class
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#ifndef __PSUADEBASEH__
#define __PSUADEBASEH__

#include "psMatrix.h"
#include "PsuadeData.h"
#include "FunctionInterface.h"
#include "Sampling.h"
#include "AnalysisManager.h"
#include "Exceptions.h"
#include "PsuadeSession.h"

// ************************************************************************
// class definition 
// ************************************************************************
class PsuadeBase 
{
  int outputLevel_;
  int useRSModel_;
  Sampling *sampler_;

public:
  psVector  VecSamInputs_;
  psVector  VecSamOutputs_;
  psIVector VecSamStates_;
  psVector  VecILowerBs_;
  psVector  VecIUpperBs_;
  psIVector VecInpPDFs_;
  psVector  VecInpMeans_;
  psVector  VecInpStds_;
  psMatrix *inputCMat_;
  psIVector VecTagArray_;
  psVector  VecDataReg_;
  psIVector VecSamPDFIndices_;
  psStrings StrInpNames_;
  psStrings StrOutNames_;
  int  nInputs_;
  int  nSamples_;
  int  nOutputs_;
  int  nReplications_;
  char **SamPDFFiles_;
  int  SamplingMethod_;

public:
  int psuadeStop_;
  PsuadeData  *psuadeIO_;
  AnalysisManager analysisManager_;

#ifdef HAVE_PYTHON
  PyObject* update_gui;
  PyObject* yesno_dialog;
#endif

public:

  // Constructor 
  PsuadeBase();

  // Destructor 
  ~PsuadeBase();

  // get input data from a given file in PSUADE's standard format
  // @param filename - name of file from which data are to be read
  // @param psuadeio_filename - optional - file to read psuade_io data from
  int  getInputFromFile(const char *filename,
                        const char *psuadeio_filename=NULL);

  // run the computer experiments
  int  run();

  // run in interactive mode
  int  sessionInteractive();

  // run in parallel interactive mode
  int  sessionInteractiveParallel();

  // assign operator 
  PsuadeBase& operator=(const PsuadeBase &);

private:

  // run different modes (PsuadeBase)
  int    runUniform();
  int    runEnsemble();

  // run adaptive sampling methods (PsuadeBase)
  int    runAdaptiveNN();
  int    runAdaptiveErrBased1();
  int    runAdaptiveErrBasedG();
  int    runAdaptivePRA();
  int    runAdaptiveOpt();
  int    runAdaptiveGradBased();

  // functions in Interpreter/RSInterpreter/ParallelInterpreter files
  int    interpretInteractive();
  int    interpretInteractiveParallel();
  void   getSampleFromPsuadeIO();
  void   displayHelpMenu(char *, char *);
  void   collapseSample();
  int    checkResidentSampleOutputs();

  // function in the RSInterpreter file
  int    RSBasedAnalysis(char *, PsuadeSession *);

  // functions in the ODEInterpreter file
  int    ODOEAnalysis(char *);
  void   displayHelpMenuODOE(int);

  // functions in GenDriver
  int    setupGuide();
  int    genBatchFile(int);
  int    genDriver(int);
  int    genKPCAMcmcWorkflow();
  int    genSetup(int, char *);

  // function used by runAdaptiveGradBased 
  int    evaluateFull(int, double *, FunctionInterface *, double *,
                      int, double *, double *, double *);

  // miscellaneous functions in PsuadeBase
  int    fillInSamples(int, int, double *, int *);
  void   updateSampleInputBounds();
  void   cleanUp();

#if 0
  // obsolete
  void   pgPlotResponseSurface();
#endif

};

#endif // __PSUADEBASEH__


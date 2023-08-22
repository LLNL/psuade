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
// Functions for PsuadeData 
// AUTHOR : CHARLES TONG
// DATE   : 2003
// ************************************************************************
#include <stdio.h>
#include <assert.h>
#include <algorithm>
#include <string>
#include "dtype.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "Globals.h"
#include "PsuadeData.h"
#include "Psuade.h"
#include "PsuadeConfig.h"
#include "PrintingTS.h"

// ************************************************************************
// local defines 
// ------------------------------------------------------------------------ 
#define psmin(a,b) (((a)<(b)) ? (a) : (b))
#define psmax(a,b) (((a)<(b)) ? (b) : (a))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------ 
PsuadeData::PsuadeData()
{
#ifdef PS_DEBUG
  printf("PsuadeData constructor begins ...\n");
#endif
  //**/------------------------------------------------------------------ 
  //**/ user may have set up other than default output file name
  //**/------------------------------------------------------------------ 
  strcpy(psuadeFileName_, psOutputFilename_);

  //**/------------------------------------------------------------------ 
  //**/ initialize internal parameters
  //**/ The other internal structures do not need to be initialized (they
  //**/ are already done when they are declared)
  //**/------------------------------------------------------------------ 
  outputLevel_ = 0;
  writeCnt_ = 0;
#ifdef PS_DEBUG
  printf("PsuadeData constructor ends.\n");
#endif
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------ 
PsuadeData::~PsuadeData()
{ 
  //**/------------------------------------------------------------------ 
  //**/ nothing needs to be cleaned (exit anyway)
  //**/------------------------------------------------------------------ 
}

// ************************************************************************
// read from a Psuade input file
// ------------------------------------------------------------------------ 
int PsuadeData::readPsuadeFile(const char *fname)
{
  int   lineLeng=2000, status;
  char  lineIn[3000], winput[2001];
  const char* keywords[] = {"PSUADE", "INPUT", "OUTPUT", "METHOD",
                            "APPLICATION", "ANALYSIS", "END"};
  FILE* fIn;

#ifdef PS_DEBUG
  printf("PsuadeData readPsuadeFile begins\n");
#endif
  // -------------------------------------------------------------------- 
  //  clean up first (in case it has been used)
  // -------------------------------------------------------------------- 
  pInput_.reset();
  pOutput_.reset();
  pMethod_.reset();
  pApplication_.reset();
  pAnalysis_.reset();

  // -------------------------------------------------------------------- 
  //  error checking 
  // -------------------------------------------------------------------- 
  fIn = fopen(fname, "r");
  if (fIn == NULL) 
  {
    printf("readPsuadeFile ERROR: file %s not found.\n",fname);
    printf("You are currently in directory: ");
    fflush(stdout);
#ifdef WINDOWS
    system("dir");
#else
    system("pwd");
#endif
    printf("\n");
    fflush(stdout);
    return -1;
  }
  fclose(fIn);

  // -------------------------------------------------------------------- 
  //  read input and output data from the given file (PSUADE_IO), if any
  // -------------------------------------------------------------------- 
  status = readPsuadeIO(fname);
  if (status != 0) return -1;

  // -------------------------------------------------------------------- 
  //  read in parameters
  // -------------------------------------------------------------------- 
  fIn = fopen(fname, "r");
  if (fIn == NULL)
  {
    printf("PsuadeData ERROR: Cannot open file %s.\n",fname);
    return -1;
  }
  while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
  {
    sscanf(lineIn,"%s", winput);
    if (strcmp(winput, keywords[0]) == 0) break;
  }
  if (feof(fIn) != 0)
  {
    printf("readPsuadeFile ERROR: keyword %s not found in file %s.\n",
           keywords[0], fname);
    fclose(fIn);
    return -1;
  }
  while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
  {
    sscanf(lineIn,"%s", winput);
    if (strcmp(winput, keywords[1]) == 0) 
      status = readInputSection(fIn);
    else if (strcmp(winput, keywords[2]) == 0) 
      status = readOutputSection(fIn);
    else if (strcmp(winput, keywords[3]) == 0) 
      status = readMethodSection(fIn);
    else if (strcmp(winput, keywords[4]) == 0) 
      status = readApplicationSection(fIn);
    else if (strcmp(winput, keywords[5]) == 0) 
      status = readAnalysisSection(fIn);
    else if (strcmp(winput, keywords[6]) == 0) break;
    else if (strcmp(winput,"#") == 0) { /* comment line */ }
    else if (winput[0] == '#') { /* comment line */ }
    else 
    {
      printf("readPsuadeFile ERROR: \n");
      printf("\t\t Unrecognized line - %s in file %s\n", lineIn, fname);
      printf("\t\t Looking for one of the following section keywords:\n");
      printf("\t\t - INPUT\n");
      printf("\t\t - OUTPUT\n");
      printf("\t\t - METHOD\n");
      printf("\t\t - APPLICATION\n");
      printf("\t\t - ANALYSIS\n");
      printf("\t\t Or END\n");
      fclose(fIn);
      return -1;
    }
    if (status != 0) 
    {
      fclose(fIn);
      return -1;
    }
  }
  fclose(fIn);
#ifdef PS_DEBUG
  printf("PsuadeData readPsuadeFile ends\n");
#endif
  return 0;
}

// ************************************************************************
// writeToFile
// ------------------------------------------------------------------------ 
void PsuadeData::writePsuadeFile(const char *inFilename, int flag)
{
  int  errFlag = 0;
  char command[1000], fname[1000];
  FILE *fOut;
  
  if (outputLevel_ > 4)
    printf("PsuadeData writePsuadeFile begins\n");
  //----------------------------------------------------------------- 
  //  Error checking.  If all the PSUADE_IO info are missing, it's OK.
  //  If we have some but not all of it, we have an issue 
  //----------------------------------------------------------------- 
  if (pInput_.VecSamInps_.length() == pInput_.nInputs_*pMethod_.nSamples_)
    errFlag++;
  if (pOutput_.VecSamOuts_.length() == 
      pOutput_.nOutputs_*pMethod_.nSamples_) errFlag++;
  if (pOutput_.VecSamStas_.length() == pMethod_.nSamples_) errFlag++;
  if (errFlag != 0 && errFlag != 3)
  {
    if (pInput_.VecSamInps_.length() == 0) 
      printf("writePsuadeFile ERROR: no input\n");
    else if (pOutput_.VecSamOuts_.length() == 0) 
      printf("writePsuadeFile ERROR: no outputs\n");
    else if (pOutput_.VecSamStas_.length() == 0) 
      printf("writePsuadeFile ERROR: no sample states\n");
    printf("CATASTROPHIC ERROR!\n");
    exit(1);
  }

  // -------------------------------------------------------------------- 
  //  if archive file exists, store it somewhere else first
  // -------------------------------------------------------------------- 
  if (inFilename != NULL) 
    strcpy(fname, inFilename);
  else if (strcmp(psuadeFileName_,"NULL")) 
    strcpy(fname,psuadeFileName_);
  else 
  {
    printf("PsuadeData writePsuadeFile ERROR: no file given to write.\n");
    return;
  }
  
  // -------------------------------------------------------------------- 
  // open archive file
  // -------------------------------------------------------------------- 
  fOut = fopen(fname, "w");
  if (fOut == NULL)
  {
    printf("PsuadeData writePsuadeFile ERROR: cannot open file %s\n",fname);
    exit(1);
  } 

  // -------------------------------------------------------------------- 
  //  write sample data to archive file 
  // -------------------------------------------------------------------- 
  writePsuadeIO(fOut, 0);

  // -------------------------------------------------------------------- 
  //  write parameters 
  // -------------------------------------------------------------------- 
  fprintf(fOut, "PSUADE\n");
  writeInputSection(fOut);
  writeOutputSection(fOut);
  writeMethodSection(fOut);
  writeApplicationSection(fOut);
  writeAnalysisSection(fOut);
  fprintf(fOut, "END\n");
  fclose(fOut);
  if (outputLevel_ > 3 && flag > 0)
    printf("\nPsuadeData writePsuadeFile: data has been written to %s.\n",
           fname);
  if (outputLevel_ > 4)
    printf("PsuadeData writePsuadeFile ends\n");
}

// ************************************************************************
// request data from this object 
// ------------------------------------------------------------------------ 
int PsuadeData::getParameter(const char *keyword, pData &pd)
{ 
  pd.clean();
  if (!strncmp(keyword,"input_",6))
       return(getInputParameter(keyword, pd));
  else if (!strncmp(keyword,"output_",7))
       return(getOutputParameter(keyword, pd));
  else if (!strncmp(keyword,"method_",7))
       return(getMethodParameter(keyword, pd));
  else if (!strncmp(keyword,"app_",4))
       return(getApplicationParameter(keyword,pd));
  else if (!strncmp(keyword,"ana_",4))
       return(getAnalysisParameter(keyword,pd));
  return 0;
}

// ************************************************************************
// request for INPUT section parameter 
// ------------------------------------------------------------------------ 
int PsuadeData::getInputParameter(const char *keyword, pData &pd)
{ 
  int  ii, jj, kk, retflag=0;
  char winput[200];

  if (!strcmp(keyword, "input_ninputs")) pd.intData_ = pInput_.nInputs_;
  else if (!strcmp(keyword, "input_lbounds"))
  {
    pd.dbleArray_ = new double[pInput_.nInputs_];
    for (ii = 0; ii < pInput_.nInputs_; ii++)
      pd.dbleArray_[ii] = pInput_.VecInpLBds_[ii];
    pd.nDbles_ = pInput_.nInputs_;
  }
  else if (!strcmp(keyword, "input_ubounds"))
  {
    pd.dbleArray_ = new double[pInput_.nInputs_];
    for (ii = 0; ii < pInput_.nInputs_; ii++)
      pd.dbleArray_[ii] = pInput_.VecInpUBds_[ii];
    pd.nDbles_ = pInput_.nInputs_;
  }
  else if (!strcmp(keyword, "input_names"))
  {
    if (pd.strArray_ != NULL)
    {
      for (ii = 0; ii < pd.nStrings_; ii++) delete [] pd.strArray_[ii];
      delete [] pd.strArray_;
    }
    pd.strArray_ = new char*[pInput_.nInputs_];
    for (ii = 0; ii < pInput_.nInputs_; ii++)
    {
      pd.strArray_[ii] = new char[1000];
      if (pInput_.StrInpNames_.numStrings() > 0) 
        strcpy(pd.strArray_[ii], pInput_.StrInpNames_.getOneString(ii));
      else
      {
        snprintf(winput,100,"X%d", ii+1);
        strcpy(pd.strArray_[ii], winput);
      }
    }
    pd.nStrings_ = pInput_.nInputs_;
  }
  else if (!strcmp(keyword, "input_pdfs"))
  {
    if (pInput_.VecInpPDFs_.length() > 0)
    {
      pd.intArray_ = new int[pInput_.nInputs_];
      for (ii = 0; ii < pInput_.nInputs_; ii++)
        pd.intArray_[ii] = pInput_.VecInpPDFs_[ii];
      pd.nInts_ = pInput_.nInputs_;
    }
  }
  else if (!strcmp(keyword, "input_means"))
  {
    if (pInput_.VecInpMeans_.length() > 0)
    {
      pd.dbleArray_ = new double[pInput_.nInputs_];
      for (ii = 0; ii < pInput_.nInputs_; ii++)
        pd.dbleArray_[ii] = pInput_.VecInpMeans_[ii];
      pd.nDbles_ = pInput_.nInputs_;
    }
  }
  else if (!strcmp(keyword, "input_stdevs"))
  {
    if (pInput_.VecInpStdvs_.length() > 0)
    {
      pd.dbleArray_ = new double[pInput_.nInputs_];
      for (ii = 0; ii < pInput_.nInputs_; ii++)
        pd.dbleArray_[ii] = pInput_.VecInpStdvs_[ii];
      pd.nDbles_ = pInput_.nInputs_;
    }
  }
  else if (!strcmp(keyword, "input_aux"))
  {
    if (pInput_.VecInpAuxs_.length() > 0)
    {
      pd.dbleArray_ = new double[pInput_.nInputs_];
      for (ii = 0; ii < pInput_.nInputs_; ii++)
        pd.dbleArray_[ii] = pInput_.VecInpAuxs_[ii];
      pd.nDbles_ = pInput_.nInputs_;
    } 
  }
  else if (!strcmp(keyword, "input_cor_matrix"))
  {
    pd.psObject_ = (void *) &(pInput_.corMatrix_);
  }
  else if (!strcmp(keyword, "input_symtable"))
  {
    if (pInput_.VecInpSyms_.length() > 0)
    {
      pd.intArray_ = new int[pInput_.nInputs_];
      for (ii = 0; ii < pInput_.nInputs_; ii++)
        pd.intArray_[ii] = pInput_.VecInpSyms_[ii];
      pd.nInts_ = pInput_.nInputs_;
    }
  }
  else if (!strcmp(keyword, "input_settings"))
  {
    if (pInput_.VecInpNumVals_.length() > 0)
    {
      pd.nInts_ = pInput_.nInputs_;
      pd.intArray_ = new int[pInput_.nInputs_];
      pd.dbleArray2D_ = new double*[pInput_.nInputs_];
      for (ii = 0; ii < pInput_.nInputs_; ii++)
      {
        jj = pInput_.VecInpNumVals_[ii];
        pd.intArray_[ii] = jj;
        if (jj > 0)
        {
          pd.dbleArray2D_[ii] = new double[jj];
          for (kk = 0; kk < jj; kk++)
            pd.dbleArray2D_[ii][kk] = 
              pInput_.MatInpVals_.getEntry(ii,kk);
        }
        else pd.dbleArray2D_[ii] = NULL;
      }
    }
  }
  else if (!strcmp(keyword, "input_sample"))
  {
    if (pInput_.VecSamInps_.length()==pInput_.nInputs_*pMethod_.nSamples_)
    {
      pd.dbleArray_ = new double[pInput_.nInputs_*pMethod_.nSamples_];
      for (ii = 0; ii < pInput_.nInputs_*pMethod_.nSamples_; ii++)
        pd.dbleArray_[ii] = pInput_.VecSamInps_[ii];
      pd.nDbles_ = pInput_.nInputs_ * pMethod_.nSamples_;
    }
  }
  else if (!strcmp(keyword, "input_sample_files"))
  {
    if (pd.strArray_ != NULL)
    {
      for (ii = 0; ii < pd.nStrings_; ii++) delete [] pd.strArray_[ii];
      delete [] pd.strArray_;
      pd.nStrings_ = 0;
      pd.strArray_ = NULL;
    }
    pd.strArray_ = new char*[pInput_.nInputs_];
    for (ii = 0; ii < pInput_.nInputs_; ii++)
    {
      pd.strArray_[ii] = new char[1000];
      strcpy(pd.strArray_[ii],pInput_.StrSamFileNames_.getOneString(ii));
    }
    pd.nStrings_ = pInput_.nInputs_;
  }
  else if (!strcmp(keyword, "input_sample_indices"))
  {
    pd.intArray_ = new int[pInput_.nInputs_];
    for (ii = 0; ii < pInput_.nInputs_; ii++)
      pd.intArray_[ii] = pInput_.VecInpSInds_[ii];
    pd.nInts_ = pInput_.nInputs_;
  }
  else if (!strcmp(keyword, "input_use_input_pdfs"))
    pd.intData_ = pInput_.useInputPDFs_;
  else retflag = 1;
  return retflag;
}

// ************************************************************************
// request for OUTPUT section parameters 
// ------------------------------------------------------------------------ 
int PsuadeData::getOutputParameter(const char *keyword, pData &pd)
{ 
  int  ii, retflag=0;
  char winput[200];

  if (!strcmp(keyword,"output_noutputs")) pd.intData_ = pOutput_.nOutputs_;
  else if (!strcmp(keyword, "output_names"))
  {
    pd.strArray_ = new char*[pOutput_.nOutputs_];
    for (ii = 0; ii < pOutput_.nOutputs_; ii++)
    {
      pd.strArray_[ii] = new char[100];
      if (pOutput_.StrOutNames_.numStrings() == pOutput_.nOutputs_)
        strcpy(pd.strArray_[ii], pOutput_.StrOutNames_.getOneString(ii));
      else
      {
        snprintf(winput,100,"Y%d", ii+1);
        strcpy(pd.strArray_[ii], winput);
      }
    }
    pd.nStrings_ = pOutput_.nOutputs_;
  }
  else if (!strcmp(keyword, "output_sample"))
  {
    if (pOutput_.VecSamOuts_.length() == 
        pOutput_.nOutputs_*pMethod_.nSamples_)
    {
      pd.dbleArray_ = new double[pOutput_.nOutputs_*pMethod_.nSamples_];
      for (ii = 0; ii < pOutput_.nOutputs_*pMethod_.nSamples_; ii++)
        pd.dbleArray_[ii] = pOutput_.VecSamOuts_[ii];
      pd.nDbles_ = pOutput_.nOutputs_ * pMethod_.nSamples_;
    }
  }
  else if (!strcmp(keyword, "output_states"))
  {
    if (pOutput_.VecSamStas_.length() == pMethod_.nSamples_)
    {
      pd.intArray_ = new int[pMethod_.nSamples_];
      for (ii = 0; ii < pMethod_.nSamples_; ii++)
        pd.intArray_[ii] = pOutput_.VecSamStas_[ii];
      pd.nInts_ = pMethod_.nSamples_;
    }
  }
  else retflag = 1;
  return retflag;
}

// ************************************************************************
// request for METHOD section parameters 
// ------------------------------------------------------------------------ 
int PsuadeData::getMethodParameter(const char *keyword, pData &pd)
{
  int  retflag=0;

  if (!strcmp(keyword, "method_sampling"))
    pd.intData_ = pMethod_.samplingMethod_;
  else if (!strcmp(keyword, "method_nsamples"))
    pd.intData_ = pMethod_.nSamples_;
  else if (!strcmp(keyword, "method_nreplications"))
    pd.intData_ = pMethod_.nReplications_;
  else if (!strcmp(keyword, "method_nrefinements"))
    pd.intData_ = pMethod_.nRefinements_;
  else if (!strcmp(keyword, "method_refnrefinements"))
    pd.intData_ = pMethod_.refNumRefinements_;
  else if (!strcmp(keyword, "method_refine_type"))
    pd.intData_ = pMethod_.refinementType_;
  else if (!strcmp(keyword, "method_refine_size"))
    pd.intData_ = pMethod_.refinementSize_;
  else if (!strcmp(keyword, "method_randomflag"))
    pd.intData_ = pMethod_.sampleRandomize_;
  else retflag = 1;
  return retflag;
}

// ************************************************************************
// request for APPLICATION section parameters 
// ------------------------------------------------------------------------ 
int PsuadeData::getApplicationParameter(const char *keyword, pData &pd)
{ 
  int  ii, retflag=0;

  if (!strcmp(keyword, "app_files"))
  {
    pd.strArray_ = new char*[7];
    for (ii = 0; ii < 7; ii++) pd.strArray_[ii] = new char[500];
    strcpy(pd.strArray_[0], pApplication_.appDriver_);
    strcpy(pd.strArray_[1], pApplication_.inputTemplate_);
    strcpy(pd.strArray_[2], pApplication_.outputTemplate_);
    strcpy(pd.strArray_[3], pApplication_.optDriver_);
    strcpy(pd.strArray_[4], pApplication_.auxOptDriver_);
    strcpy(pd.strArray_[5], pApplication_.ensembleDriver_);
    strcpy(pd.strArray_[6], pApplication_.ensembleOptDriver_);
    pd.nStrings_ = 7;
  }
  else if (!strcmp(keyword, "app_maxparalleljobs"))
    pd.intData_ = pApplication_.maxParallelJobs_;
  else if (!strcmp(keyword, "app_minjobwaittime"))
    pd.intData_ = pApplication_.minJobWaitTime_;
  else if (!strcmp(keyword, "app_maxjobwaittime"))
    pd.intData_ = pApplication_.maxJobWaitTime_;
  else if (!strcmp(keyword, "app_runtype"))
    pd.intData_ = pApplication_.runType_;
  else if (!strcmp(keyword, "app_launchinterval"))
    pd.intData_ = pApplication_.launchInterval_;
  else if (!strcmp(keyword, "app_savefrequency"))
    pd.intData_ = pApplication_.saveFrequency_;
  else if (!strcmp(keyword, "app_usefunction"))
    pd.intData_ = pApplication_.useFunction_;
  else retflag = 1;
   return retflag;
}

// ************************************************************************
// request for ANALYSIS section parameters 
// ------------------------------------------------------------------------ 
int PsuadeData::getAnalysisParameter(const char *keyword, pData &pd)
{ 
  int  ii, retflag=0;

  if (!strcmp(keyword, "ana_opt_switch"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[0];
  else if (!strcmp(keyword, "ana_opt_method"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[1];
  else if (!strcmp(keyword, "ana_opt_nlocalmin"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[2];
  else if (!strcmp(keyword, "ana_opt_rstype"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[3];
  else if (!strcmp(keyword, "ana_opt_printlevel"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[4];
  else if (!strcmp(keyword, "ana_opt_outputid"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[6];
  else if (!strcmp(keyword, "ana_opt_numfmin"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[5];
  else if (!strcmp(keyword, "ana_opt_fmin"))
    pd.dbleData_ = pAnalysis_.optimizeDbleOptions_[0];
  else if (!strcmp(keyword, "ana_opt_cutoff"))
    pd.dbleData_ = pAnalysis_.optimizeDbleOptions_[1];
  else if (!strcmp(keyword, "ana_opt_maxfeval"))
    pd.intData_ = pAnalysis_.optimizeIntOptions_[7];
  else if (!strcmp(keyword, "ana_opt_deltax"))
    pd.dbleData_ = pAnalysis_.optimizeDbleOptions_[3];
  else if (!strcmp(keyword, "ana_opt_tolerance"))
    pd.dbleData_ = pAnalysis_.optimizeDbleOptions_[2];
  else if (!strcmp(keyword, "ana_method"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[0];
  else if (!strcmp(keyword, "ana_outputid"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[1];
  else if (!strcmp(keyword, "ana_rstype"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[2];
  else if (!strcmp(keyword, "ana_poly_order"))
    pd.intData_ = pAnalysis_.legendreOrder_;
  else if (!strcmp(keyword, "ana_mars_nbasis"))
    pd.intData_ = pAnalysis_.marsNbasis_;
  else if (!strcmp(keyword, "ana_mars_interaction"))
    pd.intData_ = pAnalysis_.marsNdegrees_;
  else if (!strcmp(keyword, "ana_num_mars"))
    pd.intData_ = pAnalysis_.marsNum_;
  else if (!strcmp(keyword, "ana_kri_mode"))
    pd.intData_ = pAnalysis_.kriMode_;
  else if (!strcmp(keyword, "ana_kri_tol"))
    pd.dbleData_ = pAnalysis_.kriTol_;
  else if (!strcmp(keyword, "ana_rsindexfile"))
  {
    pd.nStrings_ = 1;
    pd.strArray_ = new char*[1];
    pd.strArray_[0] = new char[200];
    strcpy(pd.strArray_[0], pAnalysis_.rsIndexFile_);
  }
  else if (!strcmp(keyword, "ana_rsindex_sample_file"))
  {
    pd.nStrings_ = 1;
    pd.strArray_ = new char*[1];
    pd.strArray_[0] = new char[200];
    strcpy(pd.strArray_[0], pAnalysis_.rsIndexSampleFile_);
  }
  else if (!strcmp(keyword, "ana_diagnostics"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[3];
  else if (!strcmp(keyword, "ana_graphicsflag"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[4];
  else if (!strcmp(keyword, "ana_transform"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[5];
  else if (!strcmp(keyword, "ana_regressionwgtid"))
    pd.intData_ = pAnalysis_.analysisIntOptions_[6];
  else if (!strcmp(keyword, "ana_threshold"))
    pd.dbleData_ = pAnalysis_.analysisDbleOptions_[0];
  else if (!strcmp(keyword, "ana_opt_smtargetfile"))
  {
    pd.nStrings_ = 1;
    pd.strArray_ = new char*[1];
    pd.strArray_[0] = new char[200];
    strcpy(pd.strArray_[0], pAnalysis_.specsFile_);
  }
  else if (!strcmp(keyword, "ana_num_rsfilters"))
    pd.intData_ = pAnalysis_.numRSFilters_;
  else if (!strcmp(keyword, "ana_rsfilterlbounds"))
  {
    if (pAnalysis_.numRSFilters_ > 0)
    {
      pd.dbleArray_ = new double[pAnalysis_.numRSFilters_];
      for (ii = 0; ii < pAnalysis_.numRSFilters_; ii++)
        pd.dbleArray_[ii] = pAnalysis_.RSFilters_[ii]->FilterLBound_;
    }
  }
  else if (!strcmp(keyword, "ana_rsfilterubounds"))
  {
    if (pAnalysis_.numRSFilters_ > 0)
    {
      pd.dbleArray_ = new double[pAnalysis_.numRSFilters_];
      for (ii = 0; ii < pAnalysis_.numRSFilters_; ii++)
        pd.dbleArray_[ii] = pAnalysis_.RSFilters_[ii]->FilterUBound_;
    }
  }
  else if (!strcmp(keyword, "ana_rsfilterdatafile"))
  {
    if (pAnalysis_.numRSFilters_ > 0)
    {
      pd.nStrings_ = pAnalysis_.numRSFilters_;
      pd.strArray_ = new char*[pAnalysis_.numRSFilters_];
      for (ii = 0; ii < pAnalysis_.numRSFilters_; ii++)
      {
        pd.strArray_[ii] = new char[500];
        strcpy(pd.strArray_[ii],
               pAnalysis_.RSFilters_[ii]->FilterDataFile_);
      }
    }
  }
  else if (!strcmp(keyword, "ana_rsfilterindexfile"))
  {
    if (pAnalysis_.numRSFilters_ > 0)
    {
      pd.nStrings_ = pAnalysis_.numRSFilters_;
      pd.strArray_ = new char*[pAnalysis_.numRSFilters_];
      for (ii = 0; ii < pAnalysis_.numRSFilters_; ii++)
      {
        pd.strArray_[ii] = new char[500];
        strcpy(pd.strArray_[ii],
               pAnalysis_.RSFilters_[ii]->FilterIndexFile_);
      }
    }
  }
  else if (!strcmp(keyword, "ana_num_moatfilters"))
    pd.intData_ = pAnalysis_.numMOATFilters_;
  else if (!strcmp(keyword, "ana_moatfilterlbounds"))
  {
    if (pAnalysis_.numMOATFilters_ > 0)
    {
      pd.dbleArray_ = new double[pAnalysis_.numMOATFilters_];
      for (ii = 0; ii < pAnalysis_.numMOATFilters_; ii++)
        pd.dbleArray_[ii] = pAnalysis_.MOATFilters_[ii]->FilterLBound_;
    }
  }
  else if (!strcmp(keyword, "ana_moatfilterubounds"))
  {
    if (pAnalysis_.numMOATFilters_ > 0)
    {
      pd.dbleArray_ = new double[pAnalysis_.numMOATFilters_];
      for (ii = 0; ii < pAnalysis_.numMOATFilters_; ii++)
        pd.dbleArray_[ii] = pAnalysis_.MOATFilters_[ii]->FilterUBound_;
    }
  }
  else if (!strcmp(keyword, "ana_moatfilterdatafile"))
  {
    if (pAnalysis_.numMOATFilters_ > 0)
    {
      pd.nStrings_ = pAnalysis_.numMOATFilters_;
      pd.strArray_ = new char*[pAnalysis_.numMOATFilters_];
      for (ii = 0; ii < pAnalysis_.numMOATFilters_; ii++)
      {
        pd.strArray_[ii] = new char[500];
        strcpy(pd.strArray_[ii],
               pAnalysis_.MOATFilters_[ii]->FilterDataFile_);
      }
    }
  }
  else if (!strcmp(keyword, "ana_moatfilterindexfile"))
  {
    if (pAnalysis_.numMOATFilters_ > 0)
    {
      pd.nStrings_ = pAnalysis_.numMOATFilters_;
      pd.strArray_ = new char*[pAnalysis_.numMOATFilters_];
      for (ii = 0; ii < pAnalysis_.numMOATFilters_; ii++)
      {
        pd.strArray_[ii] = new char[500];
        strcpy(pd.strArray_[ii],
               pAnalysis_.MOATFilters_[ii]->FilterIndexFile_);
      }
    }
  }
  else if (!strcmp(keyword, "ana_use_input_pdfs"))
    pd.intData_ = pAnalysis_.useInputPDFs_;
  else retflag = 1;
  return retflag;
}

// ************************************************************************
// Generate basic input section (No samples)
// ------------------------------------------------------------------------ 
void PsuadeData::createInputSection(int nInputs, int *symTable,
                            double *lowerB, double *upperB, char **names)
{
  int ii;
  char pString[10000];

  //**/  ----- clean up if nInputs different -----
  if (pInput_.nInputs_ != nInputs && nInputs > 0)
  {
    pInput_.StrInpNames_.clean();
    pInput_.VecInpLBds_.clean();
    pInput_.VecInpUBds_.clean();
    pInput_.VecInpSyms_.clean();
    pInput_.VecInpPDFs_.clean();
    pInput_.VecInpMeans_.clean();
    pInput_.VecInpStdvs_.clean();
    pInput_.VecInpAuxs_.clean();
    pInput_.VecInpNumVals_.clean();
    pInput_.MatInpVals_.clean();
    pInput_.StrSamFileNames_.clean();
    pInput_.VecInpSInds_.clean();
  }

  //**/  ----- update with same nInputs -----
  if (pInput_.nInputs_ == nInputs && nInputs > 0)
  {
    if (names != NULL) 
    {
      pInput_.StrInpNames_.load(nInputs, (const char **) names);
    }
    if (lowerB != NULL) 
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpLBds_[ii] = lowerB[ii]; 
    if (upperB != NULL) 
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpUBds_[ii] = upperB[ii]; 
    if (symTable != NULL)
    {
      if (pInput_.VecInpSyms_.length() == 0)
        pInput_.VecInpSyms_.setLength(nInputs);
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpSyms_[ii] = symTable[ii];
    }
    return;
  }

  //**/  ----- update with different nInputs -----
  if (pInput_.nInputs_ != nInputs && nInputs > 0)
  {
    pInput_.nInputs_ = nInputs;
    pInput_.StrInpNames_.setNumStrings(nInputs);
    if (names != NULL) 
    {
      pInput_.StrInpNames_.load(nInputs, (const char **) names);
    }
    else
    {
      for (ii = 0; ii < nInputs; ii++)
      {
        snprintf(pString,100,"X%d", ii+1);
        pInput_.StrInpNames_.loadOneString(ii, pString);
      }
    }
    pInput_.VecInpLBds_.setLength(nInputs);
    if (lowerB != NULL)
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpLBds_[ii] = lowerB[ii]; 
    else
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpLBds_[ii] = 0.0; 
    pInput_.VecInpUBds_.setLength(nInputs);
    if (upperB != NULL)
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpUBds_[ii] = upperB[ii]; 
    else
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpUBds_[ii] = 1.0;
    pInput_.VecInpSyms_.clean();
    if (symTable != NULL)
    {
      pInput_.VecInpSyms_.setLength(nInputs);
      for (ii = 0; ii < nInputs; ii++)
        pInput_.VecInpSyms_[ii] = symTable[ii];
    }
    pInput_.useInputPDFs_  = false;
    pInput_.VecInpPDFs_.setLength(nInputs);
    pInput_.VecInpMeans_.setLength(nInputs);
    pInput_.VecInpStdvs_.setLength(nInputs);
    pInput_.VecInpAuxs_.setLength(nInputs);
    pInput_.VecInpSInds_.setLength(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      pInput_.VecInpPDFs_[ii] = 0;
      pInput_.VecInpMeans_[ii] = 0.0;
      pInput_.VecInpStdvs_[ii] = 0.0;
      pInput_.VecInpAuxs_[ii] = 0.0;
      pInput_.VecInpSInds_[ii] = 0;
    }
    pInput_.corMatrix_.setDim(nInputs, nInputs);
    for (ii = 0; ii < nInputs; ii++)
      pInput_.corMatrix_.setEntry(ii, ii, 1.0e0);
    pInput_.StrSamFileNames_.setNumStrings(nInputs);
    for (ii = 0; ii < nInputs; ii++)
    {
      strcpy(pString, "NONE");
      pInput_.StrSamFileNames_.loadOneString(ii, pString);
    }
  }
  return;
}

// ************************************************************************
// update the input section
// ------------------------------------------------------------------------ 
void PsuadeData::updateInputSection(int nSamples,int nInputs,int *symTable,
                    double *lowerB, double *upperB, double *sampleInputs, 
                    char **names,int *iPDFs, double *iMeans, double *iStds,
                    psMatrix *corMatrix)
{
  int ii;

  createInputSection(nInputs, symTable, lowerB, upperB, names);

  //**/  ----- load the sample inputs -----
  pMethod_.nSamples_ = nSamples;
  pInput_.VecSamInps_.setLength(nSamples * nInputs);
  for (ii = 0; ii < nInputs*nSamples; ii++) 
    pInput_.VecSamInps_[ii] = sampleInputs[ii]; 
  if (iPDFs != NULL)
  {
    for (ii = 0; ii < nInputs; ii++) 
      pInput_.VecInpPDFs_[ii] = iPDFs[ii];
  }
  if (iMeans != NULL)
  {
    for (ii = 0; ii < nInputs; ii++) 
      pInput_.VecInpMeans_[ii] = iMeans[ii];
  }
  if (iStds != NULL)
  {
    for (ii = 0; ii < nInputs; ii++) 
      pInput_.VecInpStdvs_[ii] = iStds[ii];
  }
  if (corMatrix != NULL)
  {
    pInput_.corMatrix_.load(*corMatrix);
  }
} 

// ************************************************************************
// update the fixed parameters 
// ------------------------------------------------------------------------ 
void PsuadeData::updateFixedParameters(int nFixed, char **names, 
                                       double *values)
{
  char pString[1001]; 
  pInput_.nFixedInps_ = nFixed;
  pInput_.StrFixedInpNames_.setNumStrings(nFixed);
  pInput_.VecFixedInpVals_.setLength(nFixed);
  snprintf(pString,100,"num_fixed = %d", pInput_.nFixedInps_);
  psConfig_.putParameter(pString);
  for (int ii = 0; ii < pInput_.nFixedInps_; ii++)
  {
    if (names != NULL && names[ii] != NULL)
      pInput_.StrFixedInpNames_.loadOneString(ii, names[ii]);
    else
    { 
      snprintf(pString,100,"Xfixed%d", ii+1);
      pInput_.StrFixedInpNames_.loadOneString(ii, pString);
    }
    pInput_.VecFixedInpVals_[ii] = values[ii];
    snprintf(pString,1000,"fixed-%d %s = %24.16e", ii+1,
             pInput_.StrFixedInpNames_.getOneString(ii),
             pInput_.VecFixedInpVals_[ii]);
    psConfig_.putParameter(pString);
  }
}

// ************************************************************************
// create the output section (No sample points or states)
// ------------------------------------------------------------------------ 
void PsuadeData::createOutputSection(int nOutputs, char **names)
{
  pOutput_.nOutputs_ = nOutputs;
  if (names != NULL) 
    pOutput_.StrOutNames_.load(nOutputs, (const char **) names);
} 

// ************************************************************************
// update the output section
// ------------------------------------------------------------------------ 
void PsuadeData::updateOutputSection(int nSamples, int nOutputs,  
                      double *sampleOutputs,int *sampleStates,char **names)
{
  int ii, ss;

  createOutputSection(nOutputs, names);

  pMethod_.nSamples_ = nSamples;
  pOutput_.VecSamOuts_.setLength(nSamples*nOutputs);
  pOutput_.VecSamStas_.setLength(nSamples);
  for (ss = 0; ss < nOutputs*nSamples; ss++)
    pOutput_.VecSamOuts_[ss] = sampleOutputs[ss]; 
  for (ss = 0; ss < nSamples; ss++)
    pOutput_.VecSamStas_[ss] = sampleStates[ss]; 
} 

// ************************************************************************
// Clears out any existing samples so they can be replaced.
// ------------------------------------------------------------------------ 
void PsuadeData::resetSamples() 
{
  pInput_.VecSamInps_.clean();
  pOutput_.VecSamOuts_.clean();
  pOutput_.VecSamStas_.clean();
}

// ************************************************************************
// update method information 
// ------------------------------------------------------------------------ 
void PsuadeData::updateMethodSection(int method, int nSamples, int nReps,
                                     int nRefine, int randomFlag)
{
  if (method     >= 0) pMethod_.samplingMethod_  = method;
  if (nSamples   >= 0) pMethod_.nSamples_        = nSamples;
  if (nReps      >= 0) pMethod_.nReplications_   = nReps;
  if (nRefine    >= 0) pMethod_.nRefinements_    = nRefine;
  if (randomFlag >= 0) pMethod_.sampleRandomize_ = randomFlag;
}

// ************************************************************************
// update application information 
// ------------------------------------------------------------------------ 
void PsuadeData::updateApplicationSection(char *appDriver, char *optDriver,
                               char *auxOptDriver, char *ensembleDriver,
                               char *ensembleOptDriver, int maxJobs) 
{
  if (appDriver != NULL) strcpy(pApplication_.appDriver_, appDriver);
  if (optDriver != NULL) strcpy(pApplication_.optDriver_, optDriver);
  if (auxOptDriver != NULL) strcpy(pApplication_.auxOptDriver_,auxOptDriver);
  if (ensembleDriver != NULL) 
    strcpy(pApplication_.ensembleDriver_,ensembleDriver);
  if (ensembleOptDriver != NULL) 
    strcpy(pApplication_.ensembleOptDriver_,ensembleOptDriver);
  if (maxJobs > 0) pApplication_.maxParallelJobs_ = maxJobs;
}

// ************************************************************************
// update analysis information 
// ------------------------------------------------------------------------ 
void PsuadeData::updateAnalysisSection(int method,int transform,int rstype,
                                       int diag, int outputID, int usePDFs)
{
  if (method    >=  0) pAnalysis_.analysisIntOptions_[0] = method;
  if (transform >=  0) pAnalysis_.analysisIntOptions_[5] = transform;
  if (rstype    >=  0) pAnalysis_.analysisIntOptions_[2] = rstype;
  if (diag      >= -2) pAnalysis_.analysisIntOptions_[3] = diag;
  if (outputID  >=  0) pAnalysis_.analysisIntOptions_[1] = outputID;
  if (usePDFs   >=  0) pAnalysis_.useInputPDFs_ = usePDFs;
}

// ************************************************************************
// update optimization information 
// ------------------------------------------------------------------------ 
void PsuadeData::updateOptimizationSection(int method, int nLocalMin,
                                 double tolerance, int maxIter, int pLevel)
{
  if (method >= 0 && method <= 21)
  {
    pAnalysis_.optimizeIntOptions_[1] = method;
    pAnalysis_.optimizeIntOptions_[0] = 1;
  }
  else 
    printf("updateOptimizationSection: method %d not updated.\n",
           method);
  if (tolerance >= 0) pAnalysis_.optimizeDbleOptions_[2] = tolerance;
  if (nLocalMin > 0)  pAnalysis_.optimizeIntOptions_[2]  = nLocalMin;
  if (maxIter > 1)    pAnalysis_.optimizeIntOptions_[7]  = maxIter;
  if (pLevel >= 0)    pAnalysis_.optimizeIntOptions_[4]  = pLevel;
}

// ************************************************************************
// get sample information
// ------------------------------------------------------------------------ 
int PsuadeData::getSession(PsuadeSession *session)
{
  int ii, nn;
  session->nSamples_ = pMethod_.nSamples_;
  session->nInputs_ = pInput_.nInputs_;
  nn = session->nSamples_ * session->nInputs_; 
  if (nn > 0)
  {
    session->vecSamInputs_.setLength(nn);
    for (ii = 0; ii < nn; ii++)
      session->vecSamInputs_[ii] = pInput_.VecSamInps_[ii];
  }
  else session->vecSamInputs_.clean();

  session->nOutputs_ = pOutput_.nOutputs_;
  nn = session->nSamples_ * session->nOutputs_; 
  if (nn > 0)
  {
    session->vecSamOutputs_.setLength(nn);
    for (ii = 0; ii < nn; ii++)
      session->vecSamOutputs_[ii] = pOutput_.VecSamOuts_[ii];
  }
  else session->vecSamOutputs_.clean();

  nn = session->nSamples_; 
  if (nn > 0)
  {
    session->vecSamStates_.setLength(nn);
    for (ii = 0; ii < nn; ii++)
      session->vecSamStates_[ii] = pOutput_.VecSamStas_[ii];
  }
  else session->vecSamStates_.clean();

  session->vecInpPDFs_.setLength(pInput_.nInputs_);
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    if (pInput_.VecInpPDFs_.length() > 0)
         session->vecInpPDFs_[ii] = pInput_.VecInpPDFs_[ii];
    else session->vecInpPDFs_[ii] = 0.0;
  }
  session->vecInpMeans_.setLength(pInput_.nInputs_);
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    if (pInput_.VecInpMeans_.length() > 0)
         session->vecInpMeans_[ii] = pInput_.VecInpMeans_[ii];
    else session->vecInpMeans_[ii] = 0.0;
  }
  session->vecInpStds_.setLength(pInput_.nInputs_);
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    if (pInput_.VecInpStdvs_.length() > 0)
         session->vecInpStds_[ii] = pInput_.VecInpStdvs_[ii];
    else session->vecInpStds_[ii] = 0.0;
  }

  session->vecInpLBounds_.setLength(pInput_.nInputs_);
  session->vecInpUBounds_.setLength(pInput_.nInputs_);
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    session->vecInpLBounds_[ii] = pInput_.VecInpLBds_[ii];
    session->vecInpUBounds_[ii] = pInput_.VecInpUBds_[ii];
  }
  session->corMatrix_.load(pInput_.corMatrix_);
  session->owned_ = 1;
  return 0;
}

// ************************************************************************
// set output level for error output
// ------------------------------------------------------------------------ 
void PsuadeData::setOutputLevel(int level)
{
  outputLevel_ = level;
  if (outputLevel_ < 0) outputLevel_ = 1;
}

// ************************************************************************
// process output data - write to matlab file
// ------------------------------------------------------------------------ 
void PsuadeData::processOutputData()
{
  int  ss, ii;
  FILE *fp=NULL;

  if ((pAnalysis_.fileWriteFlag_ & 1) != 0)
  {
    fp = fopen("psuade_matlab.m", "w");
    if ( fp == NULL ) return;
    fprintf(fp, "XY = [\n");
    for (ss = 0; ss < pMethod_.nSamples_; ss++) 
    {
      for (ii = 0; ii < pInput_.nInputs_; ii++) 
        fprintf(fp, "   %24.16e\n", 
                pInput_.VecSamInps_[ss*pInput_.nInputs_+ii]);
      for (ii = 0; ii < pOutput_.nOutputs_; ii++) 
        fprintf(fp, "   %24.16e\n", 
                pOutput_.VecSamOuts_[ss*pOutput_.nOutputs_+ii]);
    }
    fprintf(fp, "];\n");
    fprintf(fp, "X = [\n");
    for (ii = 0; ii < pInput_.nInputs_*pMethod_.nSamples_; ii++) 
      fprintf(fp, "   %24.16e\n", pInput_.VecSamInps_[ii]);
    fprintf(fp, "];\n");
    fprintf(fp, "Y = [\n");
    for (ii = 0; ii < pOutput_.nOutputs_*pMethod_.nSamples_; ii++) 
      fprintf(fp, "   %24.16e\n", pOutput_.VecSamOuts_[ii]);
    fprintf(fp, "];\n");
    for (ii = 0; ii < pInput_.nInputs_; ii++) 
      fprintf(fp, "X%d = X(%d:%d:%d);\n", ii+1, ii+1, pInput_.nInputs_,
              pMethod_.nSamples_*pInput_.nInputs_);
    for (ii = 0; ii < pOutput_.nOutputs_; ii++) 
      fprintf(fp, "Y%d = Y(%d:%d:%d);\n", ii+1, ii+1,
              pOutput_.nOutputs_, pMethod_.nSamples_*pOutput_.nOutputs_);
    fclose(fp);
  }
}

// ************************************************************************
// auxiliary get functions
// ************************************************************************

// ************************************************************************
//  Input Getters
// ************************************************************************
char** PsuadeData::getInput_inputNames()
{
  char **retVal = NULL, **inpStrings;
  if (pInput_.StrInpNames_.numStrings() > 0) 
  {
    retVal = new char*[pInput_.nInputs_];
    inpStrings = pInput_.StrInpNames_.getStrings();
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
    {
      retVal[ii] = new char[strlen(inpStrings[ii]+2)];
      strcpy(retVal[ii], inpStrings[ii]);
    }
  }
  return retVal;
}
int PsuadeData::getInput_nInputs()
{
  return pInput_.nInputs_;
}
double *PsuadeData::getInput_inputLBounds()
{
  double *retVal = NULL;
  if (pInput_.VecInpLBds_.length() > 0)
  {
    retVal = new double[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpLBds_[ii];
  } 
  return retVal;
}
double *PsuadeData::getInput_inputUBounds() 
{
  double *retVal = NULL;
  if (pInput_.VecInpUBds_.length() > 0)
  {
    retVal = new double[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpUBds_[ii];
  } 
  return retVal;
}
int *PsuadeData::getInput_inputPDFs() 
{
  int *retVal = NULL;
  if (pInput_.VecInpPDFs_.length() > 0) 
  {
    retVal = new int[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpPDFs_[ii];
  } 
  return retVal;
}
double* PsuadeData::getInput_inputMeans() 
{
  double *retVal = NULL;
  if (pInput_.VecInpMeans_.length() > 0) 
  {
    retVal = new double[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpMeans_[ii];
  } 
  return retVal;
}
double *PsuadeData::getInput_inputStdevs() 
{
  double *retVal = NULL;
  if (pInput_.VecInpStdvs_.length() > 0) 
  {
    retVal = new double[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpStdvs_[ii];
  } 
  return retVal;
}
double *PsuadeData::getInput_inputAuxs() 
{
  double *retVal = NULL;
  if (pInput_.VecInpAuxs_.length() > 0) 
  {
    retVal = new double[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpAuxs_[ii];
  } 
  return retVal;
}
int *PsuadeData::getInput_symbolTable() 
{
  int *retVal = NULL;
  if (pInput_.VecInpSyms_.length() > 0) 
  {
    retVal = new int[pInput_.nInputs_];
    for (int ii = 0; ii < pInput_.nInputs_; ii++)
      retVal[ii] = pInput_.VecInpSyms_[ii];
  } 
  return retVal;
}
double *PsuadeData::getInput_sampleInputs() 
{
  int    len = pInput_.nInputs_ * pMethod_.nSamples_;
  double *retVal = NULL;
  if (pInput_.VecSamInps_.length() == len) 
  {
    retVal = new double[len];
    for (int ii = 0; ii < len; ii++)
      retVal[ii] = pInput_.VecSamInps_[ii];
  } 
  return retVal;
}
psMatrix PsuadeData::getInput_corMatrix() 
{
  return pInput_.corMatrix_;
}
int PsuadeData::getInput_useInputPDFs() 
{
  return pInput_.useInputPDFs_;
}

// ************************************************************************
//  Output Getters
// ************************************************************************
int PsuadeData::getOutput_nOutputs() 
{
  return pOutput_.nOutputs_;
}
char** PsuadeData::getOutput_outputNames() 
{
  char** retVal = NULL;
  if (pOutput_.StrOutNames_.numStrings() > 0) 
  {
    retVal = new char*[pOutput_.nOutputs_];
    for (int ii = 0; ii < pOutput_.nOutputs_; ii++)
    {
      retVal[ii] = strdup(pOutput_.StrOutNames_.getOneString(ii));
    }
  }
  return retVal;
}
double *PsuadeData::getOutput_sampleOutputs() 
{
  int    len = pOutput_.nOutputs_ * pMethod_.nSamples_;
  double *retVal = NULL;
  if (pOutput_.VecSamOuts_.length() == len) 
  {
    retVal = new double[len];
    for (int ii = 0; ii < len; ii++)
      retVal[ii] = pOutput_.VecSamOuts_[ii];
  } 
  return retVal;
}
int *PsuadeData::getOutput_sampleStates() 
{
  int *retVal = NULL;
  if (pOutput_.VecSamStas_.length() == pMethod_.nSamples_) 
  {
    retVal = new int[pMethod_.nSamples_];
    for (int ii = 0; ii < pMethod_.nSamples_; ii++)
      retVal[ii] = pOutput_.VecSamStas_[ii];
  } 
  return retVal;
}

// ************************************************************************
//  Method Getters
// ************************************************************************
int PsuadeData::getMethod_samplingMethod() 
{
  return pMethod_.samplingMethod_;
}
int PsuadeData::getMethod_nSamples() 
{
  return pMethod_.nSamples_;
}
int PsuadeData::getMethod_nReplications() 
{
  return pMethod_.nReplications_;
}
int PsuadeData::getMethod_nRefinements() 
{
  return pMethod_.nRefinements_;
}
int PsuadeData::getMethod_refNumRefinements() 
{
  return pMethod_.refNumRefinements_;
}
int PsuadeData::getMethod_refinementType() 
{
  return pMethod_.refinementType_;
}
int PsuadeData::getMethod_refinementSize() 
{
  return pMethod_.refinementSize_;
}
int PsuadeData::getMethod_sampleRandomize() 
{
  return pMethod_.sampleRandomize_;
}

// ************************************************************************
//  Application Getters
// ************************************************************************
char* PsuadeData::getApplication_appDriver() 
{
  return strdup(pApplication_.appDriver_);
}
char* PsuadeData::getApplication_rsDriver() 
{
  return strdup(pApplication_.rsDriver_);
}
char* PsuadeData::getApplication_optDriver() 
{
  return strdup(pApplication_.optDriver_);
}
char* PsuadeData::getApplication_auxOptDriver() 
{
  return strdup(pApplication_.auxOptDriver_);
}
char* PsuadeData::getApplication_ensembleDriver() 
{
  return strdup(pApplication_.ensembleDriver_);
}
char* PsuadeData::getApplication_ensembleOptDriver() 
{
  return strdup(pApplication_.ensembleOptDriver_);
}
char* PsuadeData::getApplication_inputTemplate() 
{
  return strdup(pApplication_.inputTemplate_);
}
char* PsuadeData::getApplication_outputTemplate() 
{
  return strdup(pApplication_.outputTemplate_);
}
int PsuadeData::getApplication_maxParallelJobs() 
{
  return pApplication_.maxParallelJobs_;
}
int PsuadeData::getApplication_maxJobWaitTime() 
{
  return pApplication_.maxJobWaitTime_;
}
int PsuadeData::getApplication_minJobWaitTime() 
{
  return pApplication_.minJobWaitTime_;
}
int PsuadeData::getApplication_runType() 
{
  return pApplication_.runType_;
}
int PsuadeData::getApplication_useFunction() 
{
  return pApplication_.useFunction_;
}
int PsuadeData::getApplication_launchInterval() 
{
  return pApplication_.launchInterval_;
}
int PsuadeData::getApplication_saveFrequency() 
{
  return pApplication_.saveFrequency_;
}

// ************************************************************************
//  Analysis Getters
// ************************************************************************
int PsuadeData::getAnalysis_fileWriteFlag() 
{
  return pAnalysis_.fileWriteFlag_;
}
int PsuadeData::getAnalysis_useInputPDFs() 
{
  return pAnalysis_.useInputPDFs_;
}
int PsuadeData::getAnalysis_legendreOrder() 
{
  return pAnalysis_.legendreOrder_;
}
int PsuadeData::getAnalysis_marsNbasis() 
{
  return pAnalysis_.marsNbasis_;
}
int PsuadeData::getAnalysis_marsNdegrees() 
{
  return pAnalysis_.marsNdegrees_;
}
int PsuadeData::getAnalysis_marsNum() 
{
  return pAnalysis_.marsNum_;
}
int PsuadeData::getAnalysis_kriMode() 
{
  return pAnalysis_.kriMode_;
}
double PsuadeData::getAnalysis_kriTol() 
{
  return pAnalysis_.kriTol_;
}
int PsuadeData::getAnalysis_optSaveHistory() 
{
  return pAnalysis_.optSaveHistory_;
}
int PsuadeData::getAnalysis_optUseHistory() 
{
  return pAnalysis_.optUseHistory_;
}
char* PsuadeData::getAnalysis_specsFile() 
{
  return strdup(pAnalysis_.specsFile_);
}
char* PsuadeData::getAnalysis_rsIndexFile() 
{
  return strdup(pAnalysis_.rsIndexFile_);
}
char* PsuadeData::getAnalysis_rsIndexSampleFile() 
{
  return strdup(pAnalysis_.rsIndexSampleFile_);
}

int* PsuadeData::getAnalysis_analysisIntOptions() 
{
  int* retVal = new int[10];
  std::copy(pAnalysis_.analysisIntOptions_, 
            pAnalysis_.analysisIntOptions_+10, retVal);
  return retVal;
}

double* PsuadeData::getAnalysis_analysisDbleOptions() 
{
  double* retVal = new double[10];
  std::copy(pAnalysis_.analysisDbleOptions_, 
            pAnalysis_.analysisDbleOptions_+10, retVal);
  return retVal;
}

int* PsuadeData::getAnalysis_optimizeIntOptions() 
{
  int* retVal = new int[10];
  std::copy(pAnalysis_.optimizeIntOptions_, 
            pAnalysis_.optimizeIntOptions_+10, retVal);
  return retVal;
}

double* PsuadeData::getAnalysis_optimizeDbleOptions() 
{
  double* retVal = new double[10];
  std::copy(pAnalysis_.optimizeDbleOptions_, 
            pAnalysis_.optimizeDbleOptions_+10, retVal);
  return retVal;
}

// ************************************************************************
// internal functions
// ************************************************************************

// ************************************************************************
// A function for reading PSUADE IO data
// ------------------------------------------------------------------------ 
int PsuadeData::readPsuadeIO(const char *fname) 
{
  int    nInputs, nOutputs, nSamples, ss, ii, idata, idata2, status;
  double ddata;
  char   lineInput[500], keyword[500];
  FILE   *fIn;

  fIn = fopen(fname, "r");
  if (fIn == NULL) return -1;
  fgets(lineInput, 500, fIn);
  sscanf(lineInput, "%s", keyword);
  while (keyword[0] == '#')
  {
    fgets(lineInput, 500, fIn);
    sscanf(lineInput, "%s", keyword);
  }
  if (!strcmp(keyword, "PSUADE_IO")) /* data is in this section */
  {
    fscanf(fIn, "%d %d %d\n", &nInputs, &nOutputs, &nSamples);
    if (nInputs <= 0 || nOutputs <= 0 || nSamples <= 0)
    {
      printf("readPsuadeIO ERROR: invalid parameters.\n");
      printf(" nSamples = %d\n", nSamples);
      printf(" nInputs  = %d\n", nInputs);
      printf(" nOutputs = %d\n", nOutputs);
      printf("The first line after PSUADE should have 3 integers:\n");
      printf("<nInputs> <nOutputs> <nSamples>\n");
      fclose(fIn);
      return -1;
    }
    pInput_.nInputs_ = nInputs;
    pOutput_.nOutputs_ = nOutputs;
    pMethod_.nSamples_ = nSamples;
    pInput_.VecSamInps_.setLength(nInputs*nSamples);
    pOutput_.VecSamOuts_.setLength(nOutputs*nSamples);
    pOutput_.VecSamStas_.setLength(nSamples);
    for (ss = 0; ss < nSamples; ss++)
    {
      fscanf(fIn,"%d %d", &idata, &idata2);
      if (idata != (ss+1))
      {
        printf("readPsuadeIO ERROR: Reading sample %d\n",ss+1);
        printf("             Sample number read = %d.\n",idata);
        printf("Advice: Check format correctness in PSUADE_IO section.\n");
        fclose(fIn);
        return -1;
      }
      if (idata2 != 1) idata2 = 0;
      pOutput_.VecSamStas_[ss] = idata2; 
      for (ii = 0; ii < nInputs; ii++) 
      {
        status = fscanf(fIn,"%lg",&ddata);
        pInput_.VecSamInps_[ss*nInputs+ii] = ddata;
        if (status == 0)
        {
          printf("readPsuadeIO ERROR when reading sample %d input value.\n",
                 ss+1);
          exit(1);
        }
        else if (isnan(ddata))
        {
          printf("readPsuadeIO ERROR: an input for sample %d is NaN.\n",
                 ss+1);
          exit(1);
        }
      }
      for (ii = 0; ii < nOutputs; ii++) 
      {
        status = fscanf(fIn,"%lg",&ddata);
        pOutput_.VecSamOuts_[ss*nOutputs+ii] = ddata;
        if (status == 0)
        {
          printf("readPsuadeIO ERROR when reading sample %d output value.\n",
                 ss+1);
          exit(1);
        }
        else if (isnan(ddata))
        {
          printf("readPsuadeIO ERROR: an output for sample %d is NaN.\n",
                 ss+1);
          exit(1);
        }
      }
    }
    if (outputLevel_ > 0)
    {
      printf("readPsuadeIO: read sample data completed.\n");
      printf("   nInputs, nOutputs, nSamples = %d %d %d\n", nInputs, 
             nOutputs, nSamples);
    }
  }
  else
  {
    if (outputLevel_ > 0)
      printf("readPsuadeIO: PSUADE_IO section absent.\n");
    fclose(fIn);
    return 0;
  }
  fclose(fIn);
  return 0;
}

// ************************************************************************
// A function for reading PSUADE parameters from an input file.
// ------------------------------------------------------------------------ 
//  INPUT
//     dimension 3
//     variable  1 X1name = lbound1 ubound1 
//     variable  2 X2name = lbound2 ubound2 
//     variable  3 X3name = lbound3 ubound3 
//     PDF       1 U/N/L  mean stdev
//  END
// ------------------------------------------------------------------------ 
int PsuadeData::readInputSection(FILE *fp) 
{
  int    ii, ind, idata, itmp, lineLeng=1000, nInputs=0;
  double ddata;
  char   line[1001], winput[1001], winput2[1001], winput3[1001];
  const char *keywords[] = {"dimension", "variable", "PDF", "COR", "NAME", 
                            "num_fixed", "fixed", "discrete", "END"};
  FILE *fp2=NULL;

  //**/  ----------------------------------------------------------------
  //**/  if file exists, read input data
  //**/  ----------------------------------------------------------------
  if (outputLevel_ > 1) printf("PSUADE: Entering readInputSection\n");
  if (fp == NULL)
  {
    printf("readInputSection ERROR: file pointer = NULL.\n");
    return -1;
  }
  while ((fgets(line, lineLeng, fp) != NULL) && (feof(fp) == 0))
  {
    strcpy(winput, "#");
    sscanf(line,"%s", winput);
    if (strcmp(winput, keywords[0]) == 0) /* dimension */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
        sscanf(line,"%s %s %s", winput, winput3, winput2);
      for (ii = 0; ii < strlen(winput2); ii++)  
      {
        if (winput2[ii] < '0' || winput2[ii] > '9')
        {
          printf("INPUT SECTION syntax ERROR: invalid dimension\n");
          printf("Line read = %s\n",line);
          printf("CORRECT Syntax: (an example with 1 input)\n");
          printf("    dimension = 1\n");
          printf("    variable 1 X1 = 0 1\n");
          return -1;
        }
      }
      sscanf(winput2, "%d", &nInputs);
      if (nInputs <= 0)
      {
        printf("INPUT SECTION ERROR: nInputs <= 0.\n");
        printf("CORRECT Syntax: (an example with 1 input)\n");
        printf("    dimension = 1\n");
        printf("    variable 1 X1 = 0 1\n");
        return -1;
      } 
      if (outputLevel_ > 1) 
        printf("INPUT SECTION: nInputs read = %d\n",nInputs);
      if (pInput_.nInputs_ != 0 && pInput_.nInputs_ != nInputs)
      {
        printf("INPUT SECTION ERROR: nInputs mismatch.\n");
        printf("  nInputs in INPUT and PSUADE_IO sections are different.\n");
        return -1;
      }
      //**/ clean up first
      pInput_.StrSamFileNames_.clean();
      pInput_.VecInpSInds_.clean();
      pInput_.VecInpLBds_.clean();
      pInput_.VecInpUBds_.clean();
      pInput_.StrInpNames_.clean();
      pInput_.VecInpPDFs_.clean();
      pInput_.VecInpMeans_.clean();
      pInput_.VecInpStdvs_.clean();
      pInput_.VecInpAuxs_.clean();
      //**/ memory allocation and initialization
      pInput_.nInputs_ = nInputs;
      pInput_.VecInpLBds_.setLength(nInputs);
      pInput_.VecInpUBds_.setLength(nInputs);
      pInput_.StrInpNames_.setNumStrings(nInputs);
      pInput_.VecInpPDFs_.setLength(nInputs);
      pInput_.VecInpMeans_.setLength(nInputs);
      pInput_.VecInpStdvs_.setLength(nInputs);
      pInput_.VecInpAuxs_.setLength(nInputs);
      pInput_.VecInpSInds_.setLength(nInputs);
      pInput_.StrSamFileNames_.setNumStrings(nInputs);
      for (ii = 0; ii < nInputs; ii++)
      {
        pInput_.VecInpLBds_[ii] = 0.0;
        pInput_.VecInpUBds_[ii] = 1.0;
	pInput_.VecInpPDFs_[ii] = 0;
	pInput_.VecInpMeans_[ii] = 0.0;
	pInput_.VecInpStdvs_[ii] = 0.0;
	pInput_.VecInpAuxs_[ii] = 0.0;
	pInput_.VecInpSInds_[ii] = 0;
        strcpy(winput, "NONE");
        pInput_.StrSamFileNames_.loadOneString(ii, winput);
      }
      pInput_.corMatrix_.setDim(nInputs, nInputs);
      ddata = 1.0;
      for (ii = 0; ii < nInputs; ii++)
        pInput_.corMatrix_.setEntry(ii, ii, ddata);
    }
    else if (strcmp(winput, keywords[1]) == 0) /* variable */
    {
      if (nInputs <= 0)
      {
        printf("INPUT SECTION ERROR: input dimension not defined yet.\n");
        printf("    Make sure to declare dimension first, e.g.\n");
        printf("CORRECT Syntax: (an example with 1 input)\n");
        printf("    INPUT\n");
        printf("       dimension = 1\n");
        printf("       varialbe 1 X = 0 1\n");
        printf("    END\n");
        return -1;
      } 
      sscanf(line,"%s %s", winput, winput);
      for (ii = 0; ii < strlen(winput); ii++)  
      {
        if (winput[ii] < '0' || winput[ii] > '9')
        {
          printf("INPUT SECTION syntax ERROR: invalid variable number\n");
          printf("Line read = %s\n",line);
          printf("CORRECT Syntax: (an example with 1 input)\n");
          printf("    dimension = 1\n");
          printf("    variable 1 X1 = 0 1\n");
          return -1;
        }
      }
      sscanf(winput, "%d", &idata); 
      if (idata < 1 || idata > pInput_.nInputs_)
      {
        printf("INPUT SECTION ERROR: invalid input number %d\n",idata);
        printf("Line read = %s\n",line);
        printf("    input number should be between 1 and %d\n",nInputs);
        return -1;
      }
      idata--;
      sscanf(line,"%s %d %s", winput, &itmp, winput2);
      if (pInput_.StrInpNames_.getOneString(idata) != NULL)
      {
        printf("Line read = %s\n",line);
        printf("INPUT SECTION ERROR: \n");
        printf("\t\tinput number %d may have repeated definitions.\n", 
               idata+1);
        return -1;
      }
      pInput_.StrInpNames_.loadOneString(idata, winput2);

      sscanf(line,"%s %d %s %s", winput, &itmp, winput2, winput3);
      if (strcmp(winput3, "=") != 0 )
      {
        printf("INPUT SECTION ERROR: invalid format for input %d\n",itmp);
        printf("Line read = %s\n",line);
        printf("CORRECT Syntax:   variable 1 X1 = 0.0 1.0\n");
        return -1;
      }
      sscanf(line,"%s %d %s %s %lg %lg", winput, &itmp, winput2,
             winput3, &(pInput_.VecInpLBds_[idata]),
             &(pInput_.VecInpUBds_[idata]));
      if (pInput_.VecInpLBds_[idata] >= pInput_.VecInpUBds_[idata])
      {
        printf("INPUT SECTION ERROR: \n");
        printf("\t\tinput lbound >= ubound (%d %e %e).\n", idata+1,
               pInput_.VecInpLBds_[idata], pInput_.VecInpUBds_[idata]);
        printf("Line read = %s\n",line);
        return -1;
      }
    }
    else if (strcmp(winput, keywords[2]) == 0) /* PDF */
    {
      if (nInputs <= 0)
      {
        printf("INPUT SECTION ERROR: input dimension not defined yet.\n");
        printf("    Make sure to declare dimension first, e.g.\n");
        printf("CORRECT Syntax: (an example with 1 input)\n");
        printf("    INPUT\n");
        printf("       dimension = 1\n");
        printf("       varialbe 1 X = 0 1\n");
        printf("    END\n");
        return -1;
      } 
      sscanf(line,"%s %d", winput, &idata);
      if (idata <= 0 || idata > pInput_.nInputs_)
      {
        printf("INPUT SECTION ERROR: invalid input number %d.\n",idata);
        printf("Line read = %s\n",line);
        printf("     input number should be between 1 and %d\n",nInputs);
        return -1;
      } 
      idata--;
      sscanf(line,"%s %d %s", winput, &itmp, winput2);
      if ( !strcmp(winput2, "U"))
      {
        printf("Line read = %s\n",line);
        printf("INPUT SECTION INFO: PDF type U(ser) specified\n");
        printf("                    All inputs must be of this type.\n");
               pInput_.VecInpPDFs_[idata] = PSUADE_PDF_USER;
        pInput_.useInputPDFs_ = 1;
#if 0
        //**/ U is no longer uniform, but user (Mar 2016)
        pInput_.VecInpPDFs_[idata] = 0;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        if (pInput_.VecInpMeans_[idata] != pInput_.VecInpLBds_[idata])
        {
          printf("INPUT SECTION ERROR: for uniform distributions, the\n");
          printf("    first data should be the same as the lower bound\n");
          printf("    in the variable definition.\n");
          printf("    lower bound defined  = %e\n", 
                 pInput_.VecInpMeans_[idata]);
          printf("    lower bound expected = %e\n", 
                 pInput_.VecInpLBds_[idata]);
          printf("NOTE: The default is uniform distribution with lower\n");
          printf("      and upper bounds as defined in the variable\n");
          printf("      definitions.\n");
          return -1;
        }
        if (pInput_.VecInpStdvs_[idata] != pInput_.VecInpUBds_[idata])
        {
          printf("INPUT SECTION ERROR: for uniform distributions, for\n");
          printf("    second data should be the same as the upper bound\n");
          printf("    in the variable definition.\n");
          printf("    upper bound defined  = %e\n", 
                 pInput_.VecInpStdvs_[idata]);
          printf("    upper bound expected = %e\n", 
                 pInput_.VecInpUBds_[idata]);
          printf("NOTE: The default is uniform distribution with lower\n");
          printf("      and upper bounds as defined in the variable\n");
          printf("      definitions.\n");
          return -1;
        }
#endif
      }
      else if ( !strcmp(winput2, "N")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_NORMAL;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "L")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_LOGNORMAL;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "T")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_TRIANGLE;
        sscanf(line,"%s %d %s %lg %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]),
               &(pInput_.VecInpAuxs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "B")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_BETA;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "W")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_WEIBULL;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "G")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_GAMMA;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "IG")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_INVGAMMA;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "C")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_CAUCHY;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "E")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_EXPONENTIAL;
        sscanf(line,"%s %d %s %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]));
        pInput_.VecInpStdvs_[idata] = 0.0;
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "S")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_SAMPLE;
        winput3[0] = '\0';
        sscanf(line,"%s %d %s %s %d",winput,&itmp,winput2,winput3,&ind);
        if (pInput_.StrSamFileNames_.numStrings() == 0)
        {
          printf("INPUT SECTION ERROR: nInputs has not been read yet.\n");
          return -1;
        } 
        if ((fp2=fopen(winput3,"r")) == NULL)
        {
          printf("INPUT SECTION ERROR: no S type sample file %s found.\n",
                 winput3);
          return -1;
        } 
        else
        {
          fgets(winput2, 500, fp2);
          if (!strcmp(winput2,"PSUADE_BEGIN")) fgets(winput2, 500, fp2);
          fclose(fp2);
          sscanf(winput2,"%d %d",&itmp,&ii);
          if (itmp >= 100000 && ii <= 10)
          {
            pInput_.VecInpPDFs_[idata] = PSUADE_PDF_SAMPLEHIST;
            printf("PDF for input %d: switch from S to S2\n",idata+1); 
          }
        }
        winput2[strlen(winput3)] = '\0';
        pInput_.StrSamFileNames_.loadOneString(idata,winput3);
        if (ind < 0 || ind > nInputs)
        {
          printf("INPUT SECTION ERROR: invalid PDF type S sample index %d\n",
                 ind);
          return -1;
        } 
        else ind--;
        if (ind < 0)
        {
          printf("INPUT SECTION INFO: PDF type S sample index not given.\n");
          printf("                    Index is set to the default = 1.\n");
          ind = 0;
        }
        pInput_.VecInpSInds_[idata] = ind;
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "F")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_F;
        sscanf(line,"%s %d %s %lg %lg",winput,&itmp,winput2,
               &(pInput_.VecInpMeans_[idata]),
               &(pInput_.VecInpStdvs_[idata]));
        pInput_.useInputPDFs_ = 1;
      }
      else if ( !strcmp(winput2, "S2")) 
      {
        pInput_.VecInpPDFs_[idata] = PSUADE_PDF_SAMPLEHIST;
        winput3[0] = '\0';
        sscanf(line,"%s %d %s %s %d",winput,&itmp,winput2,winput3,&ind);
        if (pInput_.StrSamFileNames_.numStrings() == 0)
        {
          printf("INPUT SECTION ERROR: nInputs has not been read yet.\n");
          printf("Line read = %s\n",line);
          return -1;
        } 
        if ((fp2=fopen(winput3,"r")) == NULL)
        {
          printf("INPUT SECTION ERROR: S2 type sample file %s not found\n",
                 winput3);
          return -1;
        } 
        else
        {
          fgets(winput2, 500, fp2);
          if (!strcmp(winput2,"PSUADE_BEGIN")) fgets(winput2, 500, fp2);
          fclose(fp2);
          sscanf(winput2,"%d %d",&itmp,&ii);
          if (itmp < 100000 || ii > 10)
          {
            pInput_.VecInpPDFs_[idata] = PSUADE_PDF_SAMPLE;
            printf("INFO: PDF for input %d: switch from S2 to S\n",idata+1); 
          }
        }
        winput3[strlen(winput3)] = '\0';
        pInput_.StrSamFileNames_.loadOneString(idata, winput3);
        if (ind < 0 || ind > nInputs)
        {
          printf("INPUT SECTION ERROR: invalid PDF type S2 index %d.\n",
                 ind);
          return -1;
        } 
        else ind--;
        if (ind < 0)
        {
          printf("INPUT SECTION INFO: PDF type S2 sample index not given.\n");
          printf("              Index set is set to the default = 1.\n");
          ind = 0;
        }
        pInput_.VecInpSInds_[idata] = ind;
        pInput_.useInputPDFs_ = 1;
      }
      else
      {
        printf("INPUT SECTION ERROR: input distribution %c not recognized.\n",
               winput2[0]);
        return -1;
      } 
    } 
    else if (strcmp(winput, keywords[3]) == 0) /* correlation */
    {
      if (nInputs <= 0)
      {
        printf("INPUT SECTION ERROR: nInputs not defined.\n");
        printf("    Make sure to declare dimension first, e.g.\n");
        printf("    INPUT\n");
        printf("       dimension = 2\n");
        printf("       varialbe 1 X = 0 1\n");
        printf("       varialbe 2 Y = 0 1\n");
        printf("       COR 1 2 0.1\n");
        printf("    END\n");
        return -1;
      } 
      sscanf(line,"%s %d %d %lg", winput, &idata, &itmp, &ddata);
      idata--;
      itmp--;
      if (idata < 0 || idata >= pInput_.nInputs_ ||
          itmp < 0 || itmp >= pInput_.nInputs_)
      {
        printf("INPUT SECTION ERROR: invalid input numbers (%d,%d)\n",
               idata+1, itmp+1);
        printf("    input numbers should be between 1 and %d\n",nInputs);
        return -1;
      }
      if (idata == itmp && ddata != 1.0)
      { 
        printf("INPUT SECTION ERROR: Cor(%d,%d) should be = 1.\n",
               idata+1, idata+1);
        return -1;
      }
      if (idata != itmp && (ddata <= -1.0 || ddata >= 1.0))
      { 
        printf("INPUT SECTION ERROR: |Cor(%d,%d)| should be < 1.\n",
               idata+1, itmp+1);
        return -1;
      }
      pInput_.corMatrix_.setEntry(idata, itmp, ddata);
      pInput_.corMatrix_.setEntry(itmp, idata, ddata);
    }
    else if (strcmp(winput, keywords[4]) == 0) /* NAME */
    {
      /* display name for the variable - not used by psuade */
      printf("INPUT SECTION INFO: NAME field not implemented.\n");
    }
    else if (strcmp(winput, keywords[5]) == 0) /* num_fixed */
    { 
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2, &idata);
      else sscanf(line,"%s %d", winput, &idata);
      if (idata <= 0)
      {
        printf("INPUT SECTION ERROR: num_fixed <= 0.\n");
        return -1;
      } 
      if (outputLevel_ > 1) 
        printf("readInput: nFixed read = %d\n",idata);
      if (pInput_.nFixedInps_ != 0 && pInput_.nFixedInps_ != idata)
      {
        printf("INPUT SECTION ERROR: num_fixed mismatch.\n");
        printf("              Check num_fixed in INPUT section.\n");
        return -1;
      }
      pInput_.VecFixedInpVals_.setLength(idata);
      pInput_.nFixedInps_ = idata;
      pInput_.StrFixedInpNames_.setNumStrings(idata);
      snprintf(winput,100,"num_fixed = %d", pInput_.nFixedInps_);
      psConfig_.putParameter(winput);
    }
    else if (strcmp(winput, keywords[6]) == 0) /* fixed */
    {
      if (pInput_.nFixedInps_ <= 0)
      {
        printf("INPUT SECTION ERROR: num_fixed not set.\n");
        return -1;
      } 
      sscanf(line,"%s %d", winput, &idata);
      idata--;
      if (idata < 0 || idata >= pInput_.nFixedInps_)
      {
        printf("INPUT SECTION ERROR : invalid fixed input number %d\n",
               idata+1);
        printf("    input number should be between 1 and %d\n",
               pInput_.nFixedInps_);
        return -1;
      }
      sscanf(line,"%s %d %s", winput, &itmp, winput2);
      if (pInput_.StrFixedInpNames_.getOneString(idata) != NULL)
      {
        printf("INPUT SECTION WARNING: \n");
        printf("\t\tfixedinput number %d may have been re-defined.\n",
               idata+1);
      }
      pInput_.StrFixedInpNames_.loadOneString(idata, winput2);

      sscanf(line,"%s %d %s %s", winput, &itmp, winput2, winput3);
      if (strcmp(winput3, "=") != 0 )
      {
        printf("INPUT SECTION ERROR: fixed input format for input %d\n",
               itmp);
        printf("    syntax: fixed 1 X1 = 0.0\n");
        return -1;
      }
      sscanf(line,"%s %d %s %s %lg", winput, &itmp, winput2,
             winput3, &(pInput_.VecFixedInpVals_[idata]));
      snprintf(winput,1000,"fixed-%d %s = %24.16e", itmp, 
               pInput_.StrFixedInpNames_.getOneString(idata),
              pInput_.VecFixedInpVals_[idata]);
      psConfig_.putParameter(winput);
    }
    else if (strcmp(winput, keywords[7]) == 0) /* discrete */
    {
      sscanf(line,"%s %d", winput, &idata);
      if (idata < 1 || idata > pInput_.nInputs_)
      {
        printf("INPUT SECTION ERROR: invalid input number %d\n",idata);
        printf("    input number should be between 1 and %d\n",nInputs);
        return -1;
      }
      snprintf(winput2,100,"iDiscrete%d", idata);
      psConfig_.putParameter(winput2);
    }
    else if (strcmp(winput, keywords[8]) == 0) /* END */
    {
      break;
    }
    else if (strcmp(winput,"#") == 0) 
    {
      /* comment line */
    }
    else if (winput[0] == '#') /* comments */
    {
      /* comment line */
    }
    else 
    {
      printf("INPUT SECTION ERROR: unrecognized line: %s\n",line);
      return -1;
    }
  }
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    if (pInput_.StrInpNames_.getOneString(ii) == NULL)
    {
      printf("INPUT SECTION ERROR: input %d has not defined\n",ii+1);
      snprintf(winput,100,"X%d", ii+1);
      pInput_.StrInpNames_.loadOneString(ii, winput);
      pInput_.VecInpLBds_[ii] = 0;
      pInput_.VecInpUBds_[ii] = 1;
      return -1;
    }
  }
  //**/ if one input is of type user, all will be set the same
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    if (pInput_.VecInpPDFs_[ii] == PSUADE_PDF_USER)
    {
      for (ind = 0; ind < pInput_.nInputs_; ind++)
      {
        if (pInput_.VecInpPDFs_[ind] != PSUADE_PDF_USER)
        {
          printf("INPUT SECTION ERROR: PDF type U, if specified,\n");
          printf("              should be used for all inputs.\n");
          return -1;
        }
      }
    }
  }
  if (feof(fp) != 0)
  {
    printf("INPUT SECTION ERROR: END not found.\n");
    return -1;
  }
  if (outputLevel_ > 1) printf("PSUADE: Exiting readInputSection\n");
  return 0;
}

// ************************************************************************
// A function for reading PSUADE parameters from an input file.
// ------------------------------------------------------------------------ 
//  OUTPUT
//     dimension 3
//     variable  1  Y1name 
//     variable  2  Y2name
//     variable  3  Y3name
//  END
// ------------------------------------------------------------------------ 
int PsuadeData::readOutputSection(FILE *fp) 
{
  int  ii, idata, itmp, lineLeng=200, nOutputs;
  char line[200], winput[200], winput2[200], winput3[200];
  const char *keywords[] = {"dimension", "variable", "NAME", "END"}; 

  //**/  ----------------------------------------------------------------
  //**/  if file exists, read input data
  //**/  ----------------------------------------------------------------
  if (fp == NULL)
  {
    printf("readOutputSection ERROR: file pointer = null\n");
    return -1;
  }
  while ((fgets(line, lineLeng, fp) != NULL) && (feof(fp) == 0))
  {
    strcpy(winput, "#");
    sscanf(line,"%s", winput);
    if (strcmp(winput, keywords[0]) == 0) /* dimension */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
        sscanf(line,"%s %s %s", winput, winput3, winput2);
      for (ii = 0; ii < strlen(winput2); ii++)  
      {
        if (winput2[ii] < '0' || winput2[ii] > '9')
        {
          printf("OUTPUT SECTION syntax ERROR: invalid dimension\n");
          printf("Line read = %s\n",line);
          printf("CORRECT Syntax: (an example with 1 input)\n");
          printf("    dimension = 1\n");
          printf("    variable 1 Y\n");
          return -1;
        }
      }
      sscanf(winput2, "%d", &nOutputs);
      if (nOutputs <= 0)
      {
        printf("Line read: %s\n", line);
        printf("OUTPUT SECTION ERROR: nOutputs %d <= 0.\n",nOutputs);
        return -1;
      } 
      if (pOutput_.nOutputs_ != 0 && pOutput_.nOutputs_ != nOutputs)
      {
        printf("Line read: %s\n", line);
        printf("OUTPUT SECTION ERROR: nOutputs mismatch. nOutputs in\n");
        printf("       OUTPUT & PSUADE_IO sections are different.\n");
        return -1;
      } 
      pOutput_.nOutputs_ = nOutputs;
      pOutput_.StrOutNames_.setNumStrings(pOutput_.nOutputs_);
    } 
    else if ( strcmp(winput, keywords[1]) == 0 ) /* variable */
    {
      if (pOutput_.nOutputs_ <= 0)
      {
        printf("OUTPUT SECTION ERROR: nOutputs not set. Format:\n");
        printf("    OUTPUT\n");
        printf("       dimension = 1\n");
        printf("       variable 1 Y\n");
        printf("    END\n");
        return -1;
      } 
      sscanf(line,"%s %s", winput, winput2);
      for (ii = 0; ii < strlen(winput2); ii++)  
      {
        if (winput2[ii] < '0' || winput2[ii] > '9')
        {
          printf("OUTPUT SECTION syntax ERROR: invalid variable index\n");
          printf("Line read = %s\n",line);
          printf("CORRECT Syntax: (an example with 1 input)\n");
          printf("    dimension = 1\n");
          printf("    variable 1 Y\n");
          return -1;
        }
      }
      sscanf(winput2,"%d", &idata);
      idata--;
      if (idata < 0 || idata >= pOutput_.nOutputs_)
      {
        printf("OUTPUT SECTION ERROR: invalid output no. %d.\n",
               idata+1);
        printf("    Output number should be between 1 and %d\n", 
               pOutput_.nOutputs_);
        return -1;
      } 
      sscanf(line,"%s %d %s", winput, &itmp, winput2);
      pOutput_.StrOutNames_.loadOneString(idata, winput2);
    }
    else if (strcmp(winput, keywords[2]) == 0) /* NAME */
    {
      /* display name for the variable - not used by psuade */
      printf("readOutputSection INFO: NAME option not implemented.\n");
    }
    else if (strcmp(winput, keywords[3]) == 0) /* END */
    {
      break;
    }
    else if (strcmp(winput,"#") == 0) /* comments */
    {
      /* comment line */
    }
    else if (winput[0] == '#') /* comments */
    {
      /* comment line */
    }
    else 
    {
      printf("OUTPUT SECTION ERROR: unrecognized line - %s\n",line);
      return -1;
    }
  }
  if (feof(fp) != 0)
  {
    printf("OUTPUT SECTION ERROR: END not found.\n");
    return -1;
  }
  for (ii = 0; ii < pOutput_.nOutputs_; ii++)
  {
    if (pOutput_.StrOutNames_.getOneString(ii) == NULL)
    {
      printf("OUTPUT SECTION WARNING: variable %d undeclared.\n",
             ii+1);
      snprintf(winput,100,"Y%d", ii+1);
      pOutput_.StrOutNames_.loadOneString(ii, winput);
      return -1;
    } 
  } 
  return 0;
}

// ************************************************************************
// A function for reading PSUADE parameters from an input file.
// ------------------------------------------------------------------------ 
//  METHOD
//     sampling <MC,FACT,LH,OA,OALH,MOAT,SOBOL,LPTAU,
//               METIS,FAST,BBD,PBD,FF4,FF5,CCI4,CCI5,CCIF,CCF4,
//               CCF5,CCFF,CCC4,CCC5,CCCF,SFAST,UMETIS,GMOAT,SG,RFF4,RFF5>
//     randomize 
//     randomize_more (truly randomize for replicated Latin hypercube) 
//     num_samples 100
//     num_replications 3
//     num_refinements 3
//     reference_num_refinements (for purposes of analysis)
//     input_settings ID NUM DATA...
//  END
// ------------------------------------------------------------------------ 
int PsuadeData::readMethodSection(FILE *fp) 
{
  int  methodID, ii, idata, idata2, lineLeng=200, nSamples;
  double ddata;
  char line[200], winput[200], winput2[200], winput3[200];
  const char *keywords[] = {"sampling", "randomize", "randomize_more",
                            "num_samples", "num_replications", 
                            "num_refinements", "reference_num_refinements",
                            "refinement_type", "input_settings",
                            "random_seed", "refinement_size", 
                            "num_symbols", "END"}; 
  int  nMethods=32;
  const char *methods[] = {"MC","FACT","LH","OA","OALH","MOAT","SOBOL",
                           "LPTAU", "METIS","FAST","BBD","PBD","FF4","FF5",
                           "CCI4","CCI5", "CCIF","CCF4","CCF5","CCFF",
                           "CCC4","CCC5","CCCF", "SFAST","UMETIS","GMOAT",
                           "GMETIS","SPARSEGRID","DISCRETE", "LSA", "RFF4",
                           "RFF5"};

  //**/  ----------------------------------------------------------------
  //**/  if file exists, read input data
  //**/  ----------------------------------------------------------------
  if (fp == NULL)
  {
    printf("METHOD SECTION ERROR: file pointer = null\n");
    return -1;
  }
  while ((fgets(line, lineLeng, fp) != NULL) && (feof(fp) == 0))
  {
    strcpy(winput, "#");
    sscanf(line,"%s", winput);
    if (strcmp(winput, keywords[0]) == 0) /* sampling */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") != 0)
      {
        printf("METHOD SECTION ERROR: sampling format.\n");
        printf("Syntax: e.g. sampling = LPTAU\n");
        return -1;
      } 
      sscanf(line,"%s %s %s", winput, winput2, winput3);
      for (methodID = 0; methodID < nMethods; methodID++)
      {
        if (strcmp(winput3, methods[methodID]) == 0)
        {
          pMethod_.samplingMethod_ = PSUADE_SAMP_MC + methodID;
          break;
        }
      }
      if (methodID >= nMethods) 
      {
        printf("METHOD SECTION ERROR: invalid method %s.\n",winput3);
        return -1;
      }
    }   
    else if (strcmp(winput, keywords[1]) == 0) /* randomize */
    {
      pMethod_.sampleRandomize_ = 1;
      snprintf(winput2,100,"randomize");
      psConfig_.putParameter(winput2);
    }
    else if (strcmp(winput, keywords[2]) == 0) /* random perturbation */
    {
      pMethod_.sampleRandomize_ = 3;
    }
    else if (strcmp(winput, keywords[3]) == 0) /* number of samples */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2, &nSamples);
      else sscanf(line,"%s %d", winput, &nSamples);
      if (nSamples <= 0)
      {
        printf("METHOD SECTION ERROR: nSamples %d <= 0.\n",nSamples);
        return -1;
      }
      if (pMethod_.nSamples_ != 0 && pMethod_.nSamples_ != nSamples) 
      {
        printf("METHOD SECTION WARNING: nSamples mismatch. nSamples\n");
        printf("       in METHOD & PSUADE_IO sections are different.\n");
      } 
      pMethod_.nSamples_ = nSamples;
    }
    else if (strcmp(winput, keywords[4]) == 0) /* number of replications */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2,&pMethod_.nReplications_);
      else sscanf(line,"%s %d", winput, &pMethod_.nReplications_);
      if (pMethod_.nReplications_ <= 0)
      {
        printf("METHOD SECTION ERROR: nReplications <= 0.\n");
        return -1;
      } 
    }
    else if (strcmp(winput, keywords[5]) == 0) /* number of refinements */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2,&pMethod_.nRefinements_);
      else sscanf(line,"%s %d", winput, &pMethod_.nRefinements_);
      if (pMethod_.nRefinements_ < 0) 
      {
        printf("METHOD SECTION ERROR: nRefinements < 0.\n");
        return -1;
      } 
    }
    else if (strcmp(winput, keywords[6]) == 0) /*reference no. refinements */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d",winput,winput2,&pMethod_.refNumRefinements_);
      else sscanf(line,"%s %d", winput, &pMethod_.refNumRefinements_);
      if (pMethod_.refNumRefinements_ < 0) 
      {
        printf("METHOD SECTION ERROR: ref. nRefinements < 0.\n");
        return -1;
      } 
    }
    else if (strcmp(winput, keywords[7]) == 0) /* refinement type */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %s", winput, winput2, winput3);
      else sscanf(line,"%s %s", winput, winput3);
      if (!strcmp(winput3, "adaptive")) pMethod_.refinementType_ = 1;
    }
    else if (strcmp(winput, keywords[8]) == 0) /* input settings */
    {
      sscanf(line,"%s %d %d", winput, &idata, &idata2);

      //**/ error checking
      if (idata < 1 || idata > pInput_.nInputs_ ||
          pInput_.nInputs_ == 0 || idata2 <= 0)
      {
        printf("METHOD SECTION ERROR: invalid input %d",idata);
        printf(" in input_settings.\n");
        return -1;
      } 
      if (idata2 <= 0 || idata2 > 1000)
      {
        printf("METHOD SECTION ERROR: number of setting too large\n");
        printf("    Your number of settings = %d, max = 1000\n",idata2);
        return -1;
      } 

      //**/ set up matrix and vector to keep track of counts
      if (pInput_.MatInpVals_.nrows() == 0)
        pInput_.MatInpVals_.setDim(pInput_.nInputs_,1000);
      if (pInput_.VecInpNumVals_.length() == 0)
      {
        pInput_.VecInpNumVals_.setLength(pInput_.nInputs_);
        for (ii = 0; ii < pInput_.nInputs_; ii++)
          pInput_.VecInpNumVals_[ii] = 0;
      }

      //**/ store settings
      pInput_.VecInpNumVals_[idata-1] = idata2;
      for (ii = 0; ii < idata2; ii++) 
      {
        fscanf(fp, "%lg", &ddata);
        pInput_.MatInpVals_.setEntry(idata-1, ii, ddata); 
      }
    }
    else if (strcmp(winput, keywords[9]) == 0) /* random seed */
    {
      sscanf(line,"%s %s", winput, winput2);
      long rseed;
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %ld", winput, winput2, &rseed);
      else sscanf(line,"%s %ld", winput, &rseed);
      if (rseed <= 0) rseed = -1; 
      psConfig_.setRandomSeed(rseed); 
    }
    else if (strcmp(winput, keywords[10]) == 0) /* refine size */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d",winput,winput2,&pMethod_.refinementSize_);
      else sscanf(line,"%s %d", winput, &pMethod_.refinementSize_);
      if (pMethod_.refinementSize_ <= 0) pMethod_.refinementSize_ = 100000; 
    }
    else if (strcmp(winput, keywords[11]) == 0) /* input symbols */
    {
      sscanf(line,"%s %d %s %d", winput, &idata, winput2, &idata2);
      if (idata < 1 || idata > pInput_.nInputs_ ||
          pInput_.nInputs_ == 0 || idata2 <= 0)
      {
        printf("METHOD SECTION ERROR: invalid input %d",idata);
        printf(" in num_symbols.\n");
        return -1;
      } 
      if (pInput_.VecInpSyms_.length() == 0)
      {
        pInput_.VecInpSyms_.setLength(pInput_.nInputs_);
        for (ii = 0; ii < pInput_.nInputs_; ii++)
          pInput_.VecInpSyms_[ii] = 0;
      }
      pInput_.VecInpSyms_[idata-1] = idata2;
    }
    else if (strcmp(winput, keywords[12]) == 0) /* END */
    {
      break;
    }
    else if (strcmp(winput,"#") == 0) /* comments */
    {
      /* comment line */
    }
    else if (winput[0] == '#') /* comments */
    {
      /* comment line */
    }
    else 
    {
      printf("METHOD SECTION ERROR: unrecognized line - %s\n",line);
      return -1;
    }
  }
  if (pMethod_.samplingMethod_ == 1 && pMethod_.nReplications_ > 1)
  {
    printf("METHOD SECTION ERROR: \n");
    printf("\t\tFor Factorial sampling, nReplications should be 1.\n");
    return -1;
  } 
  idata = pMethod_.nSamples_ / pMethod_.nReplications_ * 
          pMethod_.nReplications_;
  if (idata != pMethod_.nSamples_)
  {
    printf("METHOD SECTION ERROR: \n");
    printf("\t\tnSamples should be multiples of nReplications.\n");
    return -1;
  } 
  if (feof(fp) != 0)
  {
    printf("METHOD SECTION ERROR: END not found.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// A function for reading PSUADE parameters from an input file.
// ------------------------------------------------------------------------ 
//  APPLICATION
//     driver = </home/user/apps/runfile, rsModel>
//     opt_driver = </home/user/apps/runfile, rsModel>
//     aux_opt_driver = (for multilevel optimization)
//     input_template = /home/user/apps/input_template
//     output_template = out_template
//     max_parallel_jobs = <d>
//     max_job_wait_time = <d>
//     min_job_wait_time = <d>
//     nondeterministic
//     launch_only
//     limited_launch_only
//     launch_interval = <d>
//     save_frequency = <d>
//     function = <s>
//     use_function
//     module = <s>
//     launch_function = <s>
//     use_launch_script
//     rs_driver = rsModel
//     use_rs
//  END
// ------------------------------------------------------------------------ 
int PsuadeData::readApplicationSection(FILE *fp) 
{
  int  lineLeng=2000, leng, ii;
  char line[2000], winput[2000], winput2[2000], winput3[2000];
  const char *keywords[] = {"driver","opt_driver","aux_opt_driver",
              "output_template","input_template", "max_parallel_jobs", 
              "max_job_wait_time", "min_job_wait_time","nondeterministic", 
              "launch_interval", "save_frequency", "launch_only", 
              "limited_launch_only", "gen_inputfile_only",
              "ensemble_run_mode", "function", "use_function", "module", 
              "launch_function", "use_launch_script", "rs_driver", 
              "use_rs", "ensemble_driver", "ensemble_opt_driver", "END"};
  FILE *fp2;

  //**/  ----------------------------------------------------------------
  //**/  if file exists, read input data
  //**/  ----------------------------------------------------------------
  if (fp == NULL)
  {
    printf("APPLICATION SECTION ERROR: file pointer = null\n");
    return -1;
  }
  while ((fgets(line, lineLeng, fp) != NULL) && (feof(fp) == 0))
  {
    strcpy(winput, "#");
    sscanf(line,"%s", winput);
    if (strcmp(winput, keywords[0]) == 0) /* driver */
    {
      //**/ Get "driver" as winput, then skip the '=' then get the 
      //**/ rest of the line
      sscanf(line,"%s %*[=] %199[^\n] ", winput, winput2);

      // If there are any quotes, expunge them
      std::string pathname = winput2;
      pathname.erase(std::remove(pathname.begin(), pathname.end(), '"'), 
                     pathname.end());
      strcpy(pApplication_.appDriver_, pathname.c_str()); 
      leng = strlen(pApplication_.appDriver_);
      for (ii = 0; ii < leng; ii++)
        if (pApplication_.appDriver_[ii] == '\r')
          pApplication_.appDriver_[ii] = '\0';
      if (strcmp(pApplication_.appDriver_, "NONE") &&
          strcmp(pApplication_.appDriver_, "PSUADE_LOCAL"))
      {
        leng = strlen(pApplication_.appDriver_);
        for (ii = 0; ii < leng; ii++)
          if (pApplication_.appDriver_[ii] == ' ') break;
        if (ii == leng)
        {
          fp2 = fopen(pApplication_.appDriver_, "r");
          if (fp2 == NULL)
          {
            printf("APPLICATION SECTION WARNING: ");
            printf("app driver %s not found.\n",pApplication_.appDriver_);
          }
          else fclose(fp2);
        }
        else 
        {
          printf("INFO: make sure driver '%s' exists and executable.\n",
                 pApplication_.appDriver_);
          winput[0] = '0';
          while (winput[0] != 'n' && winput[0] != 'y')
          {
            printf("Proceed? (y or n) ");
	    scanf("%s", winput);
          }
          if (winput[0] != 'y') exit(1);
          fgets(line, lineLeng, stdin);
        }
      }
    }   
    else if (strcmp(winput, keywords[1]) == 0) /* optimization driver */
    {
      sscanf(line,"%s %s", winput, winput2);
      //**/ Get "opt_driver" as winput, then skip the '=' then get the 
      //**/ rest of the line
      sscanf(line,"%s %*[=] %199[^\n] ", winput, winput2);

      // If there are any quotes, expunge them
      std::string pathname = winput2;
      pathname.erase(std::remove(pathname.begin(), pathname.end(), '"'), 
                     pathname.end());
      strcpy(pApplication_.optDriver_, pathname.c_str()); 
      leng = strlen(pApplication_.optDriver_);
      for (ii = 0; ii < leng; ii++)
        if (pApplication_.optDriver_[ii] == '\r')
          pApplication_.optDriver_[ii] = '\0';
      if (strcmp(pApplication_.optDriver_, "NONE") &&
          strcmp(pApplication_.optDriver_, "PSUADE_LOCAL"))
      {
        leng = strlen(pApplication_.optDriver_);
        for (ii = 0; ii < leng; ii++)
          if (pApplication_.optDriver_[ii] == ' ') break;
        if (ii == leng)
        {
          fp2 = fopen(pApplication_.optDriver_, "r");
          if (fp2 == NULL)
          {
            printf("APPLICATION SECTION WARNING: ");
            printf("opt driver %s not found.\n",
                   pApplication_.optDriver_);
          }
          else fclose(fp2);
        }
        else
        {
          printf("INFO: make sure file '%s' exists and executable.\n",
                 pApplication_.optDriver_);
          winput[0] = '0';
          while (winput[0] != 'n' && winput[0] != 'y')
          {
            printf("Proceed? (y or n) ");
	    scanf("%s", winput);
          }
          if (winput[0] != 'y') exit(1);
          fgets(line, lineLeng, stdin);
        }
      }
    }   
    else if (strcmp(winput, keywords[2]) == 0) /* aux opt driver */
    {
      sscanf(line,"%s %s", winput, winput2);
      //**/ Get "aux_opt_driver" as winput, then skip the '=' then get the 
      //**/ rest of the line
      sscanf(line,"%s %*[=] %199[^\n] ", winput, winput2);

      // If there are any quotes, expunge them
      std::string pathname = winput2;
      pathname.erase(std::remove(pathname.begin(), pathname.end(), '"'), 
                     pathname.end());
      strcpy(pApplication_.auxOptDriver_, pathname.c_str()); 
      leng = strlen(pApplication_.auxOptDriver_);
      for (ii = 0; ii < leng; ii++)
        if (pApplication_.auxOptDriver_[ii] == '\r')
          pApplication_.auxOptDriver_[ii] = '\0';
      if (strcmp(pApplication_.auxOptDriver_, "NONE"))
      {
        leng = strlen(pApplication_.auxOptDriver_);
        for (ii = 0; ii < leng; ii++)
          if (pApplication_.auxOptDriver_[ii] == ' ') break;
        if (ii == leng)
        {
          fp2 = fopen(pApplication_.auxOptDriver_, "r");
          if (fp2 == NULL)
          {
            printf("APPLICATION SECTION WARNING: opt_driver ");
            printf("%s not found.\n",pApplication_.auxOptDriver_);
          }
          else fclose(fp2);
        }
        else 
        {
          printf("INFO: be sure aux_opt_driver '%s' is executable.\n",
                 pApplication_.auxOptDriver_);
          winput[0] = '0';
          while (winput[0] != 'n' && winput[0] != 'y')
          {
            printf("Proceed? (y or n) ");
	    scanf("%s", winput);
          }
          if (winput[0] != 'y') exit(1);
          fgets(line, lineLeng, stdin);
        }
      }
    }   
    //**/ turn off for now to rid of strstr problems
    else if (strcmp(winput, keywords[3]) == 0) /* output template */
    {
      //**/ obsolete
      //**/ sscanf(line,"%s %s", winput, winput2);
      //**/ if (strcmp(winput2, "=") == 0 )
      //**/    sscanf(line,"%s %s %s", winput, winput2, winput3);
      //**/ else sscanf(line,"%s %s", winput, winput3);
      //**/ strncpy(pApplication_.outputTemplate_,winput3,strlen(winput3)+1); 
      printf("ERROR: output template feature is no longer supported.\n");
    }
    else if (strcmp(winput, keywords[4]) == 0) /* input template */
    {
      //**/ obsolete
      //**/ sscanf(line,"%s %s", winput, winput2);
      //**/ if (strcmp(winput2, "=") == 0 )
      //**/    sscanf(line,"%s %s %s", winput, winput2, winput3);
      //**/ else sscanf(line,"%s %s", winput, winput3);
      //**/ strncpy(pApplication_.inputTemplate_,winput3,strlen(winput3)+1); 
      //**/ if (strcmp(pApplication_.inputTemplate_, "NONE"))
      //**/ {
      //**/    if (outputLevel_ > 1)
      //**/    {
      //**/       fp2 = fopen(pApplication_.inputTemplate_, "r");
      //**/       if (fp2 == NULL)
      //**/       {
      //**/          printf("APPLICATION SECTION WARNING: ");
      //**/          printf("input template %s not found.\n",
      //**/                 pApplication_.inputTemplate_);
      //**/       }
      //**/       else fclose(fp2);
      //**/    }
      //**/ }
      printf("ERROR: input template feature is no longer supported.\n");
    }
    else if (strcmp(winput, keywords[5]) == 0) /* max parallel jobs */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
        sscanf(line,"%s %s %d", winput, winput2,
               &pApplication_.maxParallelJobs_);
      else sscanf(line,"%s %d",winput,&pApplication_.maxParallelJobs_);

      if (pApplication_.maxParallelJobs_ < 1)
        pApplication_.maxParallelJobs_ = 1;
      if (pApplication_.maxParallelJobs_ > 200) 
      {
        printf("APPLICATION SECTION WARNING: parallel jobs = %d.\n",
               pApplication_.maxParallelJobs_);
        printf("   Max parallel jobs too large : reset to 20.\n");
        pApplication_.maxParallelJobs_ = 20; 
      }
    }
    else if (strcmp(winput, keywords[6]) == 0) /* maximum job wait time */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2,
                     &pApplication_.maxJobWaitTime_);
      else sscanf(line,"%s %d",winput,&pApplication_.maxJobWaitTime_);
    }
    else if (strcmp(winput, keywords[7]) == 0) /* minimum job wait time */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2,
                  &pApplication_.minJobWaitTime_);
      else sscanf(line,"%s %d",winput,&pApplication_.minJobWaitTime_);
    }
    else if (strcmp(winput, keywords[8]) == 0) /* nondeterministic */
    {
      pApplication_.runType_ |= 1;
      break;
    }
    else if (strcmp(winput, keywords[9]) == 0) /* launch interval */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2,
                  &pApplication_.launchInterval_);
      else sscanf(line,"%s %d",winput,&pApplication_.launchInterval_);
      if (pApplication_.launchInterval_ < 0)
         pApplication_.launchInterval_ = 0;
    }
    else if (strcmp(winput, keywords[10]) == 0) /* save frequency */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (strcmp(winput2, "=") == 0 )
           sscanf(line,"%s %s %d", winput, winput2,
                  &pApplication_.saveFrequency_);
      else sscanf(line,"%s %d",winput,&pApplication_.saveFrequency_);
      if (pApplication_.saveFrequency_ < 1)
        pApplication_.saveFrequency_ = 10;
    }
    else if (strcmp(winput, keywords[11]) == 0) /* launch only */
    {
      pApplication_.runType_ |= 2;
    }
    else if (strcmp(winput, keywords[12]) == 0) /* limited launch only */
    {
      pApplication_.runType_ |= 4;
    }
    else if (strcmp(winput, keywords[13]) == 0) /* generate input file only */
    {
      pApplication_.runType_ |= 8;
    }
    else if (strcmp(winput, keywords[14]) == 0) /* ensemble run mode */
    {
      pApplication_.runType_ |= 16;
    }
    else if (strcmp(winput, keywords[15]) == 0) /* function */
    {
      /* python driver function, not used by psuade */
    }
    else if (strcmp(winput, keywords[16]) == 0) /* use_function */
    {
      pApplication_.useFunction_ = 1;
    }
    else if (strcmp(winput, keywords[17]) == 0) /* module */
    {
      /* python launch script module, not used by psuade */
    }
    else if (strcmp(winput, keywords[18]) == 0) /* launch_function */
    {
      /* python launch script function name, not used by psuade */
    }
    else if (strcmp(winput, keywords[19]) == 0) /* use_launch_script */
    {
      pApplication_.useFunction_ = 2;
    }
    else if (strcmp(winput, keywords[20]) == 0) /* rs_driver */
    {
      sscanf(line,"%s %s", winput, winput2);
      //**/ Get "rs_driver" as winput, then skip the '=' then get the 
      //**/ rest of the line
      scanf(line,"%s %*[=] %199[^\n] ", winput, winput2);

      // If there are any quotes, expunge them
      std::string pathname = winput2;
      pathname.erase(std::remove(pathname.begin(), pathname.end(), '"'), 
                     pathname.end());
      strcpy(pApplication_.rsDriver_, pathname.c_str()); 
      leng = strlen(pApplication_.rsDriver_);
      for (ii = 0; ii < leng; ii++)
        if (pApplication_.rsDriver_[ii] == '\r')
          pApplication_.rsDriver_[ii] = '\0';
      if (strcmp(pApplication_.rsDriver_, "NONE"))
      {
        leng = strlen(pApplication_.rsDriver_);
        for (ii = 0; ii < leng; ii++)
          if (pApplication_.rsDriver_[ii] == ' ') break;
        if (ii == leng)
        {
          fp2 = fopen(pApplication_.rsDriver_, "r");
          if (fp2 == NULL)
          {
            printf("APPLICATION SECTION WARNING: ");
            printf("rs driver %s not found.\n",pApplication_.rsDriver_);
          }
          else fclose(fp2);
        }
        else
        {
          printf("INFO: make sure rs_driver '%s' exists.\n",
                 pApplication_.rsDriver_);
          winput[0] = '0';
          while (winput[0] != 'n' && winput[0] != 'y')
          {
            printf("Proceed? (y or n) ");
            scanf("%s", winput);
          }
          if (winput[0] != 'y') exit(1);
          fgets(line, lineLeng, stdin);
        }
      }
    }   
    else if (strcmp(winput, keywords[21]) == 0) /* use_rs */
    {
      pApplication_.useFunction_ = 3;
    }
    else if (strcmp(winput, keywords[22]) == 0) /* ensemble driver */
    {
      sscanf(line,"%s %s", winput, winput2);
      //**/ Get "ensemble_driver" as winput, then skip the '=' then get 
      //**/ the rest of the line
      sscanf(line,"%s %*[=] %199[^\n] ", winput, winput2);

      // If there are any quotes, expunge them
      std::string pathname = winput2;
      pathname.erase(std::remove(pathname.begin(), pathname.end(), '"'), 
                     pathname.end());
      strcpy(pApplication_.ensembleDriver_, pathname.c_str()); 
      if (outputLevel_ > 1)
        printf("ensemble driver = %s\n", pApplication_.ensembleDriver_);
      leng = strlen(pApplication_.ensembleDriver_);
      for (ii = 0; ii < leng; ii++)
        if (pApplication_.ensembleDriver_[ii] == '\r')
          pApplication_.ensembleDriver_[ii] = '\0';
      if (strcmp(pApplication_.ensembleDriver_, "NONE"))
      {
        leng = strlen(pApplication_.ensembleDriver_);
        for (ii = 0; ii < leng; ii++)
          if (pApplication_.ensembleDriver_[ii] == ' ') break;
        if (ii == leng)
        {
          fp2 = fopen(pApplication_.ensembleDriver_, "r");
          if (fp2 == NULL)
          {
            printf("APPLICATION SECTION WARNING: ");
            printf("ensemble driver %s not found.\n",
                   pApplication_.ensembleDriver_);
          }
          else fclose(fp2);
        }
        else
        {
          printf("INFO: make sure ensemble_driver '%s' is executable.\n",
                 pApplication_.ensembleDriver_);
          winput[0] = '0';
          while (winput[0] != 'n' && winput[0] != 'y')
          {
            printf("Proceed? (y or n) ");
            scanf("%s", winput);
          }
          if (winput[0] != 'y') exit(1);
          fgets(line, lineLeng, stdin);
        }
      }
    }   
    else if (strcmp(winput, keywords[23]) == 0) /* ensemble opt driver */
    {
      sscanf(line,"%s %s", winput, winput2);
      //**/ Get "ensemble_opt_driver" as winput, then skip the '=' then 
      //**/ get the rest of the line
      sscanf(line,"%s %*[=] %199[^\n] ", winput, winput2);

      // If there are any quotes, expunge them
      std::string pathname = winput2;
      pathname.erase(std::remove(pathname.begin(), pathname.end(), '"'), 
                     pathname.end());
      strcpy(pApplication_.ensembleOptDriver_, pathname.c_str()); 
      if (outputLevel_ > 1)
        printf("ensemble opt driver = %s\n", 
               pApplication_.ensembleOptDriver_);
      leng = strlen(pApplication_.ensembleOptDriver_);
      for (ii = 0; ii < leng; ii++)
        if (pApplication_.ensembleOptDriver_[ii] == '\r')
          pApplication_.ensembleOptDriver_[ii] = '\0';
      if (strcmp(pApplication_.ensembleOptDriver_, "NONE"))
      {
        leng = strlen(pApplication_.ensembleOptDriver_);
        for (ii = 0; ii < leng; ii++)
          if (pApplication_.ensembleOptDriver_[ii] == ' ') break;
        if (ii == leng)
        {
          fp2 = fopen(pApplication_.ensembleOptDriver_, "r");
          if (fp2 == NULL)
          {
            printf("APPLICATION SECTION WARNING: ");
            printf("ensemble opt driver %s not found.\n",
                   pApplication_.ensembleOptDriver_);
          }
          else fclose(fp2);
        }
        else
        {
          printf("INFO: make sure ensemble_opt_driver '%s' can be run.\n",
                 pApplication_.ensembleOptDriver_);
          winput[0] = '0';
          while (winput[0] != 'n' && winput[0] != 'y')
          {
            printf("Proceed? (y or n) ");
	    scanf("%s", winput);
          }
          if (winput[0] != 'y') exit(1);
          fgets(line, lineLeng, stdin);
        }
      }
    }   
    else if (strcmp(winput, keywords[24]) == 0) /* END */
    {
      break;
    }
    else if (strcmp(winput,"#") == 0) 
    {
      /* comment line */
    }
    else if (winput[0] == '#') /* comments */
    {
      /* comment line */
    }
    else 
    {
      printf("APPLICATION SECTION ERROR: \n");
      printf("\t\t unrecognized line - %s\n", line);
      return -1;
    }
  }

  // useFunction_: 0- driver, 1- python-generated function driver,
  // 2- python launch script, 3- response surface from psuadeBase file
  if ( pApplication_.useFunction_ == 1 )
    strcpy( pApplication_.appDriver_, "psuadeFunctionDriver" );
  else if ( pApplication_.useFunction_ == 2 )
    strcpy( pApplication_.appDriver_, "psuadeLaunchDriver" );
  else if ( pApplication_.useFunction_ == 3 )
    strcpy( pApplication_.appDriver_, pApplication_.rsDriver_ );

  if ( pApplication_.minJobWaitTime_ > 0 &&
       pApplication_.minJobWaitTime_ >= pApplication_.maxJobWaitTime_ )
  {
    pApplication_.maxJobWaitTime_ = pApplication_.minJobWaitTime_ * 5;
    if (outputLevel_ > 1)
    {
      printf("APPLICATION SECTION INFO: max job wait time\n");
      printf("            has been reset to 5*(min job wait time)\n");
    }
  }
  if (feof(fp) != 0)
  {
    printf("APPLICATION SECTION ERROR: END not found.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// A function for reading PSUADE analysis parameters from an input file.
// ------------------------------------------------------------------------ 
//  ANALYSIS
//     analyzer method = <Moment,MainEffect,TwoParamEffect,ANOVA,GLSA,..>
//     analyzer outputID = <d>
//     analyzer threshold = <f>
//     optimization method = <crude,txmath,appspack,minpack,cobyla,sm,mm> 
//     optimization num_local_minima = <d> (or num_starts 
//     optimization use_response_surface
//     optimization fmin = <f>
//     optimization num_fmin = <d>
//     optimization cutoff = <f>
//     optimization output_level = <d>
//     optimization output_id = <d>
//     graphics 
//     sample_graphics 
//     file_write matlab
//     diagnostics X
//  END
// ------------------------------------------------------------------------ 
//  optimizeIntOptions[0] = turn on/off optimization
//  optimizeIntOptions[1] = choose optimization method
//  optimizeIntOptions[2] = store num_local_minima or num_starts
//  optimizeIntOptions[3] = use response surface for optimization
//  optimizeIntOptions[4] = output level
//  optimizeIntOptions[5] = num of fmin to calculate
//  optimizeIntOptions[6] = optimization output ID
//  optimizeDbleOptions[0] = fmin
//  optimizeDbleOptions[1] = cutoff point for optimization
//  optimizeDbleOptions[2] = tolerance for optimization
//  analysisIntOptions[0] = analysis method
//  analysisIntOptions[1] = output ID
//  analysisIntOptions[2] = rstype
//  analysisIntOptions[3] = diagnostics
//  analysisIntOptions[4] = graphics
//  analysisDbleOptions[0] = threshold
// ------------------------------------------------------------------------ 
int PsuadeData::readAnalysisSection(FILE *fp) 
{
  int    lineLeng=200, outputID, rstype, transform, ii, idata;
  double threshold, lbound, ubound;
  char   line[200], winput[200], winput2[200], winput3[200], winput4[200];
  char   winput5[200];
  const char *keywords[] = {
             "analyzer", "optimization", "graphics", "sample_graphics",
             "file_write", "printlevel", "use_config_file", "interactive", 
             "ana_expert", "rs_expert", "opt_expert", "sam_expert", 
             "io_expert", "rs_max_pts", "scilab", "use_input_pdfs",
             "constraint_op_and", "python_override", "python_interpreter",
             "master", "END", "diagnostics", "create_config", "mcmc_gibbs",
             "rs_codegen"};
  const char *analysisOptions[] = {
             "method", "threshold", "output_id", "rstype", "transform", 
             "regression_wgt_id", "rs_constraint", "moat_constraint", 
             "rs_index_file", "rs_legendre_order", "rs_mars_num_bases", 
             "rs_mars_interaction","rs_num_mars","rs_kriging_mode",
             "rs_kriging_tol", "rs_index_sample_file", "rs_codegen"};
  const char   *deprecateOptions[] = {
             "rsfilter", "moat_filter", "fileWrite", "rsMaxPts", 
             "sampleGraphics"};
  const char *analysisMethods[] = {
             "Moment", "MainEffect", "TwoParamEffect", "ANOVA", "GLSA", 
             "RSFA","MOAT", "Sobol", "Correlation", "Integration",
             "FAST", "FF", "PCA", "FORM", "RSMSobol1", "RSMSobol2", 
             "RSMSobolTSI", "Bootstrap", "RSMSobolG", "ARSMNN",
             "ARSM","REL","AOPT", "GOWER", "DELTA", "ETA", "LSA"};
  const char *resSurfTypes[] = {
             "MARS","linear","quadratic","cubic", "quartic", "ANN", 
             "selective_regression", "GP1", "GP3", "SVM", "PWL", "TGP",
             "MARSBag","sum_of_trees","Legendre","user_regression",
             "sparse_grid_regression", "Kriging", "splines", "KNN", "RBF",
             "Acosso", "Bssanova", "psuade_regression", "RBFBag", "PLS",
             "MRBF", "MGP3", "MMARS", "MTGP", "MNN", "HLEG", "HGP3"};
  const char *transformTypes[] = {"logx","logy"};
  const char *optimizeOptions[] = {
             "method", "fmin", "num_local_minima", "use_response_surface", 
             "cutoff", "tolerance", "print_level", "num_fmin", "output_id",
             "max_feval", "deltax", "target_file", "save_history", 
             "use_history", "num_starts"};
  const char *optimizeSchemes[] = {
             "crude", "txmath", "appspack", "minpack", "cobyla", "sm", "mm",
             "mm_adaptive", "bobyqa", "sce", "moo", "ouu", "ouu1", "ouu2",
             "lincoa", "newuoa", "ouu_unconstr", "ouu_bndconstr",
             "ouu_ineq_constr", "lbfgs", "ouu_lbfgs", "nomad", "ouu_minlp"};
  psuadeFilter **filters;
  FILE   *fconf;

  //**/  ----------------------------------------------------------------
  //**/  if file exists, read input data
  //**/  ----------------------------------------------------------------
  if (fp == NULL)
  {
    printf("ANALYSIS SECTION ERROR: file pointer = null\n");
    return -1;
  }
  while ((fgets(line, lineLeng, fp) != NULL) && (feof(fp) == 0))
  {
    strcpy(winput, "#");
    sscanf(line,"%s", winput);

    if (strcmp(winput, keywords[0]) == 0)  /* analyzer */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (!strcmp(winput2,analysisOptions[0])) /* == analysis method == */
      {
        sscanf(line,"%s %s %s %s", winput, winput2, winput3, winput4);
        if (!strcmp(winput4,analysisMethods[0]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_MOMENT;
        else if (!strcmp(winput4,analysisMethods[1]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_ME;
        else if (!strcmp(winput4,analysisMethods[2]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_IE;
        else if (!strcmp(winput4,analysisMethods[3]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_ANOVA;
        else if (!strcmp(winput4,analysisMethods[4]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_GLSA;
        else if (!strcmp(winput4,analysisMethods[5]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_RSFA;
        else if (!strcmp(winput4,analysisMethods[6]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_MOAT;
        else if (!strcmp(winput4,analysisMethods[7]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_SOBOL;
        else if (!strcmp(winput4,analysisMethods[8]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_CORRELATION;
        else if (!strcmp(winput4,analysisMethods[9]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_INTEGRATION;
        else if (!strcmp(winput4,analysisMethods[10]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_FAST;
        else if (!strcmp(winput4,analysisMethods[11]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_FF;
        else if (!strcmp(winput4,analysisMethods[12]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_PCA;
        else if (!strcmp(winput4,analysisMethods[13]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_FORM;
        else if (!strcmp(winput4,analysisMethods[14]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_RSSOBOL1;
        else if (!strcmp(winput4,analysisMethods[15]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_RSSOBOL2;
        else if (!strcmp(winput4,analysisMethods[16]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_RSSOBOLTSI;
        else if (!strcmp(winput4,analysisMethods[17]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_BSTRAP;
        else if (!strcmp(winput4,analysisMethods[18]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_RSSOBOLG;
        else if (!strcmp(winput4,analysisMethods[19]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_ARSMNN;
        else if (!strcmp(winput4,analysisMethods[20]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_ARSMMB;
        else if (!strcmp(winput4,analysisMethods[21]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_REL;
        else if (!strcmp(winput4,analysisMethods[22]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_AOPT;
        else if (!strcmp(winput4,analysisMethods[23]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_GOWER;
        else if (!strcmp(winput4,analysisMethods[24]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_DTEST;
        else if (!strcmp(winput4,analysisMethods[25]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_ETEST;
        else if (!strcmp(winput4,analysisMethods[26]))
           pAnalysis_.analysisIntOptions_[0] |= PSUADE_ANA_LSA;
        else
        {
          printf("ANALYSIS SECTION ERROR: invalid method %s\n",winput4);
          return -1;
        }
      }
      else if (!strcmp(winput2,analysisOptions[1]))/* ==analysis thresh== */
      {
        sscanf(line,"%s %s %s %lg", winput, winput2, winput3, &threshold);
        if (threshold < 0.0 || threshold > 1.0)
        {
          printf("ANALYSIS SECTION ERROR: invalid threshold %e.\n",
                 threshold);
          printf("           (threshold has to be > 0.0 and < 1.0.\n");
          return -1;
        }
        pAnalysis_.analysisDbleOptions_[0] = threshold;
      }
      else if (!strcmp(winput2,analysisOptions[2]))/* == analysis outID == */
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, &outputID);
        if (outputID <= 0 || outputID > pOutput_.nOutputs_)
        {
          printf("ANALYSIS SECTION ERROR: invalid output ID %d\n",
                 outputID);
          printf("           output ID has to be between 1 and %d\n",
                 pOutput_.nOutputs_);
          return -1;
        }
        pAnalysis_.analysisIntOptions_[1] = outputID - 1;
      }
      else if (!strcmp(winput2,analysisOptions[3]))/* ==analysis rs type== */
      {
        sscanf(line,"%s %s %s %s", winput, winput2, winput3, winput4);
        rstype = 0;
        if (!strcmp(winput4,resSurfTypes[0])) 
             rstype = PSUADE_RS_MARS;
        else if (!strcmp(winput4,resSurfTypes[1])) 
             rstype = PSUADE_RS_REGR1;
        else if (!strcmp(winput4,resSurfTypes[2])) 
             rstype = PSUADE_RS_REGR2;
        else if (!strcmp(winput4,resSurfTypes[3])) 
             rstype = PSUADE_RS_REGR3;
        else if (!strcmp(winput4,resSurfTypes[4])) 
             rstype = PSUADE_RS_REGR4;
        else if (!strcmp(winput4,resSurfTypes[5])) 
        {
          rstype = PSUADE_RS_ANN;
          printf("ANN is currently not supported.\n");
          printf("    Set to default = Kriging.\n");
          rstype = PSUADE_RS_KR;
        }
        else if (!strcmp(winput4,resSurfTypes[6])) 
             rstype = PSUADE_RS_REGRS;
        else if (!strcmp(winput4,resSurfTypes[7])) 
             rstype = PSUADE_RS_GP1;
        else if (!strcmp(winput4,resSurfTypes[8])) 
             rstype = PSUADE_RS_GP3;
        else if (!strcmp(winput4,resSurfTypes[9])) 
             rstype = PSUADE_RS_SVM;
        else if (!strcmp(winput4,resSurfTypes[10])) 
             rstype = PSUADE_RS_PWL;
        else if (!strcmp(winput4,resSurfTypes[11])) 
             rstype = PSUADE_RS_TGP;
        else if (!strcmp(winput4,resSurfTypes[12])) 
             rstype = PSUADE_RS_MARSB;
        else if (!strcmp(winput4,resSurfTypes[13])) 
             rstype = PSUADE_RS_SOTS;
        else if (!strcmp(winput4,resSurfTypes[14])) 
             rstype = PSUADE_RS_REGRL;
        else if (!strcmp(winput4,resSurfTypes[15])) 
             rstype = PSUADE_RS_REGRU;
        else if (!strcmp(winput4,resSurfTypes[16])) 
             rstype = PSUADE_RS_REGSG;
        else if (!strcmp(winput4,resSurfTypes[17])) 
             rstype = PSUADE_RS_KR;
        else if (!strcmp(winput4,resSurfTypes[18])) 
             rstype = PSUADE_RS_SPLINES;
        else if (!strcmp(winput4,resSurfTypes[19])) 
             rstype = PSUADE_RS_KNN;
        else if (!strcmp(winput4,resSurfTypes[20])) 
             rstype = PSUADE_RS_RBF;
        else if (!strcmp(winput4,resSurfTypes[21])) 
             rstype = PSUADE_RS_ACOSSO;
        else if (!strcmp(winput4,resSurfTypes[22]))
             rstype = PSUADE_RS_BSSANOVA;
        else if (!strcmp(winput4,resSurfTypes[23])) 
             rstype = PSUADE_RS_LOCAL;
        else if (!strcmp(winput4,resSurfTypes[24])) 
             rstype = PSUADE_RS_RBFB;
        else if (!strcmp(winput4,resSurfTypes[25])) 
             rstype = PSUADE_RS_PLS;
        else if (!strcmp(winput4,resSurfTypes[26])) 
             rstype = PSUADE_RS_MRBF;
        else if (!strcmp(winput4,resSurfTypes[27])) 
             rstype = PSUADE_RS_MGP3;
        else if (!strcmp(winput4,resSurfTypes[28])) 
             rstype = PSUADE_RS_MMARS;
        else if (!strcmp(winput4,resSurfTypes[29])) 
             rstype = PSUADE_RS_MTGP;
        else if (!strcmp(winput4,resSurfTypes[30])) 
             rstype = PSUADE_RS_MDNN;
        else if (!strcmp(winput4,resSurfTypes[31])) 
             rstype = PSUADE_RS_HLEG;
        else if (!strcmp(winput4,resSurfTypes[32])) 
             rstype = PSUADE_RS_HGP3;
        else
        {
          printf("ANALYSIS SECTION ERROR: invalid RS type %s\n",winput4);
          return -1;
        }
        pAnalysis_.analysisIntOptions_[2] = rstype;
      }
      else if (!strcmp(winput2,analysisOptions[4]))/* ==analysis xsform == */
      {
        sscanf(line,"%s %s %s %s", winput, winput2, winput3, winput4);
        transform = 0;
        if      (!strcmp(winput4,transformTypes[0])) transform |= 1;
        else if (!strcmp(winput4,transformTypes[1])) transform |= 2;
        else
        {
          printf("ANALYSIS SECTION ERROR: invalid transform %s\n",winput4);
          return -1;
        }
        pAnalysis_.analysisIntOptions_[5] |= transform;
      }
      else if (!strcmp(winput2,analysisOptions[5]))/*=analysis regr wgt =*/
      {
        sscanf(line,"%s %s %s %d",winput,winput2,winput3,
               &pAnalysis_.analysisIntOptions_[6]);
        pAnalysis_.analysisIntOptions_[6]--;
        printf("ANALYSIS SECTION INFO: regression wgt outputID = %d.\n",
               pAnalysis_.analysisIntOptions_[6]+1);
        printf("        Please make sure it is correct.\n");
      }
      else if (strcmp(winput2, analysisOptions[6]) == 0 ||
               strcmp(winput2, deprecateOptions[0]) == 0) /* rs_constraint */
      {
        filters = pAnalysis_.RSFilters_;
        ii = pAnalysis_.numRSFilters_ + 1;
        pAnalysis_.RSFilters_ = new psuadeFilter*[ii];
        for (ii = 0; ii < pAnalysis_.numRSFilters_; ii++)
          pAnalysis_.RSFilters_[ii] = filters[ii];
        if (filters != NULL) delete [] filters;
        pAnalysis_.numRSFilters_++;
        ii = pAnalysis_.numRSFilters_ - 1;
        pAnalysis_.RSFilters_[ii] = new psuadeFilter();
        sscanf(line,"%s %s %s %s %s %lg %lg", winput, winput2, winput3,
               winput4, winput5, &lbound, &ubound);
        strcpy(pAnalysis_.RSFilters_[ii]->FilterDataFile_, winput4);
        strcpy(pAnalysis_.RSFilters_[ii]->FilterIndexFile_, winput5);
        pAnalysis_.RSFilters_[ii]->FilterLBound_ = lbound;
        pAnalysis_.RSFilters_[ii]->FilterUBound_ = ubound;
        if (winput3[0] != '=')
        {
          printf("analyzer rs_contraint format ERROR: \n");
          printf("   format:  analyzer rs_contraint = ");
          printf("dataFile indexFile lbound ubound\n");
          return -1;
        }
      }
      else if (strcmp(winput2, analysisOptions[7]) == 0 || 
               strcmp(winput2, deprecateOptions[1]) == 0) 
      {
        filters = pAnalysis_.MOATFilters_;
        ii = pAnalysis_.numMOATFilters_ + 1;
        pAnalysis_.MOATFilters_ = new psuadeFilter*[ii];
        for (ii = 0; ii < pAnalysis_.numMOATFilters_; ii++)
          pAnalysis_.MOATFilters_[ii] = filters[ii];
        if (filters != NULL) delete [] filters;
        pAnalysis_.numMOATFilters_++;
        ii = pAnalysis_.numMOATFilters_ - 1;
        pAnalysis_.MOATFilters_[ii] = new psuadeFilter();
        sscanf(line,"%s %s %s %s %s %lg %lg", winput, winput2, winput3,
               winput4, winput5, &lbound, &ubound);
        strcpy(pAnalysis_.MOATFilters_[ii]->FilterDataFile_, winput4);
        strcpy(pAnalysis_.MOATFilters_[ii]->FilterIndexFile_, winput5);
        pAnalysis_.MOATFilters_[ii]->FilterLBound_ = lbound;
        pAnalysis_.MOATFilters_[ii]->FilterUBound_ = ubound;
        if (winput3[0] != '=')
        {
          printf("analyzer moat_constraint format ERROR: \n");
          printf("   format:  analyzer moat_constraint = ");
          printf("dataFile indexFile lbound ubound\n");
          return -1;
        }
      }
      else if (strcmp(winput2, analysisOptions[8]) == 0) /* rs index file */
      {
        sscanf(line,"%s %s %s %s", winput, winput2, winput3, 
               pAnalysis_.rsIndexFile_);
      }
      else if (strcmp(winput2, analysisOptions[9]) == 0) /* legendre_order */
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, 
               &pAnalysis_.legendreOrder_);
        snprintf(winput,100,"Legendre_order = %d", pAnalysis_.legendreOrder_);
        psConfig_.putParameter(winput);
      }
      else if (strcmp(winput2, analysisOptions[10]) == 0) /* _mars_nbases */
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, 
               &pAnalysis_.marsNbasis_);
        snprintf(winput,100,"MARS_num_bases = %d", pAnalysis_.marsNbasis_);
        psConfig_.putParameter(winput);
      }
      else if (strcmp(winput2, analysisOptions[11]) == 0) 
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, 
               &pAnalysis_.marsNdegrees_);
        snprintf(winput,100,"MARS_interaction = %d", pAnalysis_.marsNdegrees_);
        psConfig_.putParameter(winput);
      }
      else if (strcmp(winput2, analysisOptions[12]) == 0) /* rs_num_mars*/
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, 
               &pAnalysis_.marsNum_);
        if (pAnalysis_.marsNum_ < 2)
        {
          pAnalysis_.marsNum_ = 51;
          printf("ANALYSIS SECTION ERROR: invalid number of MARS (>= 2).\n");
        }
        else 
        {
          snprintf(winput,100,"MARS_num = %d", pAnalysis_.marsNum_);
          psConfig_.putParameter(winput);
        }
      }
      else if (strcmp(winput2, analysisOptions[13]) == 0) /* kriging_mode */
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, 
               &pAnalysis_.kriMode_);
        if (pAnalysis_.kriMode_ < 1 || pAnalysis_.kriMode_ > 3)
        {
          pAnalysis_.kriMode_ = -1;
          printf("ANALYSIS SECTION ERROR: invalid Kriging mode (1-3 only).\n");
        }
        else
        {
          snprintf(winput,100,"KRI_mode = %d", pAnalysis_.kriMode_);
          psConfig_.putParameter(winput);
        }
      }
      else if (strcmp(winput2, analysisOptions[14]) == 0) /* _kriging_tol */
      {
        sscanf(line,"%s %s %s %lg", winput, winput2, winput3, 
               &pAnalysis_.kriTol_);
        if (pAnalysis_.kriTol_ <= 1.0e-14 || pAnalysis_.kriTol_ > 0.1)
        {
          pAnalysis_.kriTol_ = -1;
          printf("ANALYSIS SECTION ERROR: invalid Kriging tolerance.\n");
          printf("         Should be in the range of 1e-14 and 1e-1.\n");
        }
        else
        {
          snprintf(winput,100,"KRI_tol = %e", pAnalysis_.kriTol_);
          psConfig_.putParameter(winput);
        }
      }
      else if (strcmp(winput2, analysisOptions[15]) == 0) /* rs index sample */
      {
        sscanf(line,"%s %s %s %s", winput, winput2, winput3, 
               pAnalysis_.rsIndexSampleFile_);
      }
      else 
      {
        printf("ANALYSIS SECTION ERROR: ");
        printf("unrecognized analyzer option - %s\n", winput2);
        return -1;
      }
    }
    else if (strcmp(winput, keywords[1]) == 0)  /* optimization */
    {
      sscanf(line,"%s %s", winput, winput2);
        
      if (!strcmp(winput2, optimizeOptions[0])) /* opt method */
      {
        sscanf(line,"%s %s %s %s",winput,winput2,winput3,winput4);
        if (!strcmp(winput4, optimizeSchemes[0])) /* crude method */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 0;
        }
        else if (!strcmp(winput4, optimizeSchemes[1])) /* txmath */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 1;
        }
        else if (!strcmp(winput4, optimizeSchemes[2])) /* appspack */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 2;
        }
            else if (!strcmp(winput4, optimizeSchemes[3])) /* minpack */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 3;
        }
        else if (!strcmp(winput4, optimizeSchemes[4])) /* cobyla */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 4;
        }
        else if (!strcmp(winput4, optimizeSchemes[5])) /* sm */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 5;
        }
        else if (!strcmp(winput4, optimizeSchemes[6])) /* mm */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 6;
        }
        else if (!strcmp(winput4, optimizeSchemes[7])) /* adaptive mm */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 7;
        }
        else if (!strcmp(winput4, optimizeSchemes[8])) /* bobyqa */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 8;
        }
        else if (!strcmp(winput4, optimizeSchemes[9])) /* SCE */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 9;
        }
        else if (!strcmp(winput4, optimizeSchemes[10])) /* MOO */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 10;
        }
        else if (!strcmp(winput4, optimizeSchemes[11])) /* OUU */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 11;
        }
        else if (!strcmp(winput4, optimizeSchemes[12])) /* OUU1 */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 12;
        }
        else if (!strcmp(winput4, optimizeSchemes[13])) /* OUU2 */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 13;
        }
        else if (!strcmp(winput4, optimizeSchemes[14])) /* Lincoa */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 14;
        }
        else if (!strcmp(winput4, optimizeSchemes[15])) /* Newuoa */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 15;
        }
        else if (!strcmp(winput4, optimizeSchemes[16])) /* ouu_newuoa */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 16;
        }
        else if (!strcmp(winput4, optimizeSchemes[17])) /* ouu_bobyqa */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 11;
          //**/ ouu_bobyqa = ouu
        }
        else if (!strcmp(winput4, optimizeSchemes[18])) /* ouu_cobyla */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 18;
        }
        else if (!strcmp(winput4, optimizeSchemes[19])) /* lbfgs */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 19;
        }
        else if (!strcmp(winput4, optimizeSchemes[20])) /* ouu_lbfgs */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 20;
        }
        else if (!strcmp(winput4, optimizeSchemes[21])) /* nomad */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 21;
        }
        else if (!strcmp(winput4, optimizeSchemes[22])) /* ouu/minlp */
        {
          pAnalysis_.optimizeIntOptions_[0] = 1;
          pAnalysis_.optimizeIntOptions_[1] = 22;
        }
        else
        {
          printf("ANALYSIS SECTION ERROR: invalid opt scheme %s\n",winput4);
          return -1;
        }
      }
      else if (!strcmp(winput2, optimizeOptions[1])) /* opt fmin */
      {
        sscanf(line,"%s %s %s %lg",winput,winput2,winput3,
               &(pAnalysis_.optimizeDbleOptions_[0]));
      }
      else if (!strcmp(winput2, optimizeOptions[2])) /* num_local_min */
      {
        sscanf(line,"%s %s %s %d",winput,winput2,winput3,
               &(pAnalysis_.optimizeIntOptions_[2]));
        if (pAnalysis_.optimizeIntOptions_[2] < 1)
        {
          printf("ANALYSIS SECTION ERROR: num_local_minima < 1.\n");
          pAnalysis_.optimizeIntOptions_[2] = 1;
          return -1;
        }
        if (pAnalysis_.optimizeIntOptions_[2] > pMethod_.nSamples_)
        {
          printf("ANALYSIS SECTION ERROR: num_local_minima = %d.\n",
                 pAnalysis_.optimizeIntOptions_[2]);
          printf("                      but nSamples = %d.\n",
                 pMethod_.nSamples_);
          printf("NOTE: nSamples has to be >= num_local_minima.\n");
          return -1;
        }
      }
      else if (!strcmp(winput2, optimizeOptions[3])) /* opt RS */
      {
        pAnalysis_.optimizeIntOptions_[3] = 1;
      }
      else if (!strcmp(winput2, optimizeOptions[4])) /* opt cutoff */
      {
        sscanf(line,"%s %s %s %lg",winput,winput2,winput3,
               &(pAnalysis_.optimizeDbleOptions_[1]));
      }
      else if (!strcmp(winput2, optimizeOptions[5])) /* opt tolerance */
      {
        sscanf(line,"%s %s %s %lg",winput,winput2,winput3,
               &(pAnalysis_.optimizeDbleOptions_[2]));
      }
      else if (!strcmp(winput2, optimizeOptions[6])) /* opt outlevel */
      {
        sscanf(line,"%s %s %s %d",winput,winput2,winput3,
               &(pAnalysis_.optimizeIntOptions_[4]));
      }
      else if (!strcmp(winput2, optimizeOptions[7])) /* opt num_fmin */
      {
        sscanf(line,"%s %s %s %d",winput,winput2,winput3,
               &(pAnalysis_.optimizeIntOptions_[5]));
        if (pAnalysis_.optimizeIntOptions_[5] <= 0)
        {
          printf("ANALYSIS SECTION ERROR: opt nfmin <= 0.\n");
          return -1;
        }
      }
      else if (!strcmp(winput2, optimizeOptions[8])) /* opt output ID */
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, &outputID);
        if (outputID <= 0 || outputID > pOutput_.nOutputs_)
        {
          printf("ANALYSIS SECTION ERROR: invalid outputID %d\n",
                 outputID);
          return -1;
        }
        pAnalysis_.optimizeIntOptions_[6] = outputID - 1;
      }
      else if (!strcmp(winput2, optimizeOptions[9])) /* opt max feval */
      {
        sscanf(line,"%s %s %s %d", winput, winput2, winput3, 
               &(pAnalysis_.optimizeIntOptions_[7]));
      }
      else if (!strcmp(winput2, optimizeOptions[10])) /* opt delta X */
      {
        sscanf(line,"%s %s %s %lg", winput, winput2, winput3, 
               &(pAnalysis_.optimizeDbleOptions_[3]));
      }
      else if (!strcmp(winput2, optimizeOptions[11])) /* opt TARGET FILE */
      {
        sscanf(line,"%s %s %s %s", winput, winput2, winput3, 
               pAnalysis_.specsFile_);
      }
      else if (!strcmp(winput2, optimizeOptions[12])) /* opt save history */
      {
        pAnalysis_.optSaveHistory_ = 1;
        snprintf(winput,100,"opt_save_history");
        psConfig_.putParameter(winput);
      }
      else if (!strcmp(winput2, optimizeOptions[13])) /* opt use history */
      {
        pAnalysis_.optUseHistory_ = 1;
        snprintf(winput,100,"opt_use_history");
        psConfig_.putParameter(winput);
      }
      else if (!strcmp(winput2, optimizeOptions[14])) /* no of restarts */
      {
        //**/ this option is the same of num_local_minima. It
        //**/ is included because this command name is more
        //**/ appropriate
        sscanf(line,"%s %s %s %d",winput,winput2,winput3,
               &(pAnalysis_.optimizeIntOptions_[2]));
        if (pAnalysis_.optimizeIntOptions_[2] < 1)
        {
          printf("ANALYSIS SECTION ERROR: num_starts < 1.\n");
          pAnalysis_.optimizeIntOptions_[2] = 1;
          return -1;
        }
        if (pAnalysis_.optimizeIntOptions_[2] > pMethod_.nSamples_)
        {
          printf("ANALYSIS SECTION ERROR: num_starts = %d.\n",
                 pAnalysis_.optimizeIntOptions_[2]);
          printf("                      but nSamples = %d.\n",
                 pMethod_.nSamples_);
          printf("NOTE: nSamples has to be >= num_starts ");
          printf("to give enough initial points.\n");
          return -1;
        }
      }
      else
      {
        printf("ANALYSIS SECTION ERROR: unrecognized optimize options %s.\n",
               winput2);
        return -1;
      }
    }
    else if (strcmp(winput, keywords[2]) == 0) /* graphics */
    {
      pAnalysis_.analysisIntOptions_[4] |= 1;
    }
    else if (strcmp(winput, keywords[3]) == 0 ||
             strcmp(winput, deprecateOptions[4]) == 0) /* sample Graphics */
    {
      pAnalysis_.analysisIntOptions_[4] |= 2;
    }
    else if (strcmp(winput, keywords[4]) == 0 ||
             strcmp(winput, deprecateOptions[2]) == 0) /* file write */
    {
      sscanf(line,"%s %s", winput, winput2);
      if (!strcmp(winput2, "matlab")) pAnalysis_.fileWriteFlag_ |= 1;
    }
    //set the diagnostics level; allow printlevel
    //added 4/2/14 - Jim MCEnerney
    else if (strcmp(winput, keywords[5]) == 0 || 
             strcmp(winput, keywords[21]) == 0) /* diagnostics */
    {
      sscanf(line,"%s %d", winput, &(pAnalysis_.analysisIntOptions_[3]));
      //force to range defined by PL_MIN to PL_MAX in sysdef.h
      pAnalysis_.analysisIntOptions_[3] = 
                     psmax(PL_MIN, pAnalysis_.analysisIntOptions_[3]);
      pAnalysis_.analysisIntOptions_[3] = 
                     psmin(PL_MAX, pAnalysis_.analysisIntOptions_[3]);
      outputLevel_ = pAnalysis_.analysisIntOptions_[3];
      setPrintLevelTS(outputLevel_);
    }
    else if (!strcmp(winput, keywords[6])) /* use configure file */
    {
      sscanf(line,"%s %s %s", winput, winput2, winput3); 
      if (strcmp(winput3, "NONE"))
      {
        fconf = fopen(winput3,"r");
        if (fconf == NULL)
        {
          printf("ANALYSIS SECTION ERROR: config file %s not found.\n",
                 winput3);
        }
        else
        {
          fclose(fconf); 
          psConfig_.addFromFile(winput3);
        }
      }
    }
    else if (strcmp(winput, keywords[7]) == 0) /* interactive */
    {
      //**/deprecated
      psConfig_.AnaExpertModeOn();
    }
    else if (strcmp(winput, keywords[8]) == 0) /* analysis expert mode */
    {
      psConfig_.AnaExpertModeOn();
    }
    else if (strcmp(winput, keywords[9]) == 0)/*response surface expert mode*/
    {
      psConfig_.RSExpertModeOn();
    }
    else if (strcmp(winput, keywords[10]) == 0)/*optimization expert mode*/
    {
      psConfig_.OptExpertModeOn();
    }
    else if (strcmp(winput, keywords[11]) == 0)/*sampling expert mode*/
    {
      psConfig_.SamExpertModeOn();
    }
    else if (strcmp(winput, keywords[12]) == 0)/* IO expert mode*/
    {
      psConfig_.IOExpertModeOn();
    }
    else if (strcmp(winput, keywords[13]) == 0 ||
             strcmp(winput, deprecateOptions[3]) == 0) /* RS max points */
    {
      sscanf(line,"%s %s %d", winput, winput2, &idata); 
      if (idata <= 1) 
      {
        psConfig_.RSMaxPts_ = 5000;
        printf("Max sample size for response surface is set to %d\n",
               psConfig_.RSMaxPts_);
      }
      else psConfig_.RSMaxPts_ = idata;
    }
    else if (strcmp(winput, keywords[14]) == 0) /* scilab mode */
    {
      psConfig_.PlotTool_ = 1;
    }
    else if (strcmp(winput, keywords[15]) == 0) /* use_input_pdfs */
    {
      pAnalysis_.useInputPDFs_ = 1;
      printf("NOTE: The use_input_pdfs option has been enabled.\n");
      printf("Meaning: Whenever PDFs are specified in the ");
      printf("INPUT section, they will\n");
      printf("         be used in analysis if relevant. To ");
      printf("not use PDFs in analysis,\n");
      printf("         you will have to comment out the PDF ");
      printf("definitions.\n");
    }
    else if (strcmp(winput, keywords[16]) == 0) /* constraint_op_and */
    {
      psConfig_.RSConstraintSetOp_ = 1;
    }
    else if (strcmp(winput, keywords[17]) == 0) /* Python override */
    {
      psPythonOverride_ = 1;
    }
    else if (strcmp(winput, keywords[18]) == 0) /* Python interpreter */
    {
      if (psPythonInterpreter_ != NULL) delete [] psPythonInterpreter_;
      psPythonInterpreter_ = new char[10000];
      sscanf(line,"%s %s %s", winput, winput2, psPythonInterpreter_); 
      printf("Python interpreter has been set to %s.\n",
             psPythonInterpreter_);
    }
    else if (strcmp(winput, keywords[19]) == 0) /* master */
    {
      psConfig_.MasterModeOn();
    }
    else if (strcmp(winput, keywords[20]) == 0) /* END */
    {
      break;
    }
    else if (strcmp(winput, keywords[22]) == 0) /* Create config */
    {
      psConfig_.reset();
      break;
    }
    else if (strcmp(winput, keywords[23]) == 0) /* set MCMC to use Gibbs */
    {
      snprintf(winput,100,"MCMC_gibbs");
      psConfig_.putParameter(winput);
      break;
    }
    else if (strcmp(winput, keywords[24]) == 0) /* rs_codegen */
    {
      psConfig_.RSCodeGenOn();
    }
    else if (strcmp(winput,"#") == 0) 
    {
      /* comment line */
    }
    else if (winput[0] == '#') /* comments */
    {
      /* comment line */
    }
    else 
    {
      printf("ANALYSIS SECTION ERROR: ");
      printf("\t\t unrecognized line - %s\n", line);
      break;
    }
  }
  if (feof(fp) != 0)
  {
    printf("ANALYSIS SECTION ERROR: END not found.\n");
    return -1;
  }
  return 0;
}

// ************************************************************************
// A function for writing PSUADE data to an output file.
//**/  (flag to indicate whether to treat the samples as not done)
// ------------------------------------------------------------------------ 
void PsuadeData::writePsuadeIO(FILE *fOut, int flag) 
{
  int  ss, ii;
  char cString[1000], lineIn[1000];

  if (pMethod_.nSamples_ < 0) return;
  if (pMethod_.nSamples_ > 200000 && (!psConfig_.IOExpertModeIsOn()))
  {
    printf("INFO: Data too large to be written (%d).\n",
           pMethod_.nSamples_);
    printf("Write to file anyway ? (y or n) \n");
    scanf("%s", cString);
    printf("\n");
    fgets(lineIn,500,stdin);
    printf("To enforce file write without asking, set io_expert mode.\n");
    if (cString[0] != 'y') return;
  }

  //**/ Only actually write PSUADE_IO if we have all the data to do so.
  //**/ We do make files without generating the sampling somtimes.
  int inpLen = pInput_.nInputs_ * pMethod_.nSamples_;
  int outLen = pOutput_.nOutputs_ * pMethod_.nSamples_;
  if ((pInput_.VecSamInps_.length() == inpLen) && 
      (pOutput_.VecSamOuts_.length() == outLen) &&
      (pOutput_.VecSamStas_.length() == pMethod_.nSamples_))
  {
    fprintf(fOut,"PSUADE_IO (Note : inputs not true inputs if pdf ~=U)\n");
    fprintf(fOut,"%d %d %d\n", pInput_.nInputs_, pOutput_.nOutputs_, 
            pMethod_.nSamples_);
    for (ss = 0; ss < pMethod_.nSamples_; ss++)
    {
      if (flag == 0)
           fprintf(fOut,"%d %d\n", ss+1, pOutput_.VecSamStas_[ss]);
      else fprintf(fOut,"%d 0\n", ss+1);
      for (ii = 0; ii < pInput_.nInputs_; ii++)
        fprintf(fOut,"%24.16e\n", 
                pInput_.VecSamInps_[ss*pInput_.nInputs_+ii]);
      for (ii = 0; ii < pOutput_.nOutputs_; ii++)
        fprintf(fOut,"%24.16e\n",
                pOutput_.VecSamOuts_[ss*pOutput_.nOutputs_+ii]);
    }
    fprintf(fOut,"PSUADE_IO\n");
  }
}

// ************************************************************************
// A function for writing input section to a file
// ------------------------------------------------------------------------ 
void PsuadeData::writeInputSection(FILE *fOut) 
{
  int    ii, jj;
  double ddata;

  fprintf(fOut,"INPUT\n");
  fprintf(fOut,"   dimension = %d\n", pInput_.nInputs_);
  fflush(fOut);
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    fprintf(fOut,"   variable %d %s ",ii+1, 
            pInput_.StrInpNames_.getOneString(ii));
    fprintf(fOut," = %24.16e %24.16e\n",pInput_.VecInpLBds_[ii],
            pInput_.VecInpUBds_[ii]);
  }
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    if ((pInput_.VecInpPDFs_.length() > 0) && 
        (pInput_.VecInpPDFs_[ii] == PSUADE_PDF_NORMAL))
    {
      fprintf(fOut,"   PDF %d N %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_LOGNORMAL)
    {
      fprintf(fOut,"   PDF %d L %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_TRIANGLE)
    {
      fprintf(fOut,"   PDF %d T %12.5e %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii],
              pInput_.VecInpAuxs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_BETA)
    {
      fprintf(fOut,"   PDF %d B %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_WEIBULL)
    {
      fprintf(fOut,"   PDF %d W %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_GAMMA)
    {
      fprintf(fOut,"   PDF %d G %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_INVGAMMA)
    {
      fprintf(fOut,"   PDF %d IG %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_CAUCHY)
    {
      fprintf(fOut,"   PDF %d C %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_EXPONENTIAL)
    {
      fprintf(fOut,"   PDF %d E %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_SAMPLE)
    {
      fprintf(fOut,"   PDF %d S %s %d\n", ii+1,
              pInput_.StrSamFileNames_.getOneString(ii), 
              pInput_.VecInpSInds_[ii]+1);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_F)
    {
      fprintf(fOut,"   PDF %d F %12.5e %12.5e\n", ii+1,
              pInput_.VecInpMeans_[ii], pInput_.VecInpStdvs_[ii]);
    }
    else if ((pInput_.VecInpPDFs_.length() > 0) && 
              pInput_.VecInpPDFs_[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      fprintf(fOut,"   PDF %d S2 %s %d\n", ii+1,
              pInput_.StrSamFileNames_.getOneString(ii), 
              pInput_.VecInpSInds_[ii]);
    }
    else
    {
      //**/ fprintf(fOut,"   PDF %d U\n", ii+1);
    }
  }
  for (ii = 0; ii < pInput_.nInputs_; ii++)
  {
    for (jj = ii+1; jj < pInput_.nInputs_; jj++)
    {
      ddata = pInput_.corMatrix_.getEntry(ii, jj);
      if (ddata != 0.0)
        fprintf(fOut,"   COR %d %d %12.5e\n", ii+1, jj+1, ddata);
    }
  }
  if (pInput_.nFixedInps_ > 0)
  {
    fprintf(fOut,"   num_fixed = %d\n",pInput_.nFixedInps_); 
    for (ii = 0; ii < pInput_.nFixedInps_; ii++)
    {
      fprintf(fOut,"   fixed %d %s = %24.16e\n",ii+1, 
              pInput_.StrFixedInpNames_.getOneString(ii),
              pInput_.VecFixedInpVals_[ii]);
    }
  }
  fprintf(fOut,"#  PDF <inpNum> N  <mean> <std>\n");
  fprintf(fOut,"#  PDF <inpNum> L  <logmean> <std>\n");
  fprintf(fOut,"#  PDF <inpNum> T  <center> <halfbasewidth>\n");
  fprintf(fOut,"#  PDF <inpNum> B  <alpha> <beta>\n");
  fprintf(fOut,"#  PDF <inpNum> G  <alpha> <beta>\n");
  fprintf(fOut,"#  PDF <inpNum> W  <lambda> <K>\n");
  fprintf(fOut,"#  PDF <inpNum> IG <alpha> <beta>\n");
  fprintf(fOut,"#  PDF <inpNum> C  <X0> <gamma>\n");
  fprintf(fOut,"#  PDF <inpNum> E  <lambda>\n");
  fprintf(fOut,"#  PDF <inpNum> F  <D1> <D2>\n");
  fprintf(fOut,"#  PDF <inpNum> S  <filename> <index>\n");
  fprintf(fOut,"#  NOTE: <filename> in iwrite format\n");
  fprintf(fOut,"#  COR <inpNum> <inpNum> <value>\n");
  fprintf(fOut,"#  num_fixed = <count>\n");
  fprintf(fOut,"#  fixed <num> = <value>\n");
  fprintf(fOut,"END\n");
}

// ************************************************************************
// A function for writing output section to a file
// ------------------------------------------------------------------------ 
void PsuadeData::writeOutputSection(FILE *fOut) 
{
  int ii;

  fprintf(fOut,"OUTPUT\n");
  fprintf(fOut,"   dimension = %d\n", pOutput_.nOutputs_);
  for (ii = 0; ii < pOutput_.nOutputs_; ii++)
    fprintf(fOut,"   variable %d %s\n", ii+1, 
            pOutput_.StrOutNames_.getOneString(ii));
  fprintf(fOut,"END\n");
}

// ************************************************************************
// A function for writing method section to a file
// ------------------------------------------------------------------------ 
void PsuadeData::writeMethodSection(FILE *fOut) 
{
  int ii, kk;

  fprintf(fOut,"METHOD\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_MC)
       fprintf(fOut,"   sampling = MC\n");
  else fprintf(fOut,"#  sampling = MC\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_FACT)
       fprintf(fOut,"   sampling = FACT\n");
  else fprintf(fOut,"#  sampling = FACT\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_LHS)
       fprintf(fOut,"   sampling = LH\n");
  else fprintf(fOut,"#  sampling = LH\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_OA)
       fprintf(fOut,"   sampling = OA\n");
  else fprintf(fOut,"#  sampling = OA\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_OALH)
       fprintf(fOut,"   sampling = OALH\n");
  else fprintf(fOut,"#  sampling = OALH\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_MOAT)
       fprintf(fOut,"   sampling = MOAT\n");
  else fprintf(fOut,"#  sampling = MOAT\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_SOBOL)
       fprintf(fOut,"   sampling = SOBOL\n");
  else fprintf(fOut,"#  sampling = SOBOL\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_LPTAU)
       fprintf(fOut,"   sampling = LPTAU\n");
  else fprintf(fOut,"#  sampling = LPTAU\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_METIS)
       fprintf(fOut,"   sampling = METIS\n");
  else fprintf(fOut,"#  sampling = METIS\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_FAST)
       fprintf(fOut,"   sampling = FAST\n");
  else fprintf(fOut,"#  sampling = FAST\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_BBD)
       fprintf(fOut,"   sampling = BBD\n");
  else fprintf(fOut,"#  sampling = BBD\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_PBD)
       fprintf(fOut,"   sampling = PBD\n");
  else fprintf(fOut,"#  sampling = PBD\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_FF4)
       fprintf(fOut,"   sampling = FF4\n");
  else fprintf(fOut,"#  sampling = FF4\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_FF5)
       fprintf(fOut,"   sampling = FF5\n");
  else fprintf(fOut,"#  sampling = FF5\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCI4)
       fprintf(fOut,"   sampling = CCI4\n");
  else fprintf(fOut,"#  sampling = CCI4\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCI5)
       fprintf(fOut,"   sampling = CCI5\n");
  else fprintf(fOut,"#  sampling = CCI5\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCIF)
       fprintf(fOut,"   sampling = CCIF\n");
  else fprintf(fOut,"#  sampling = CCIF\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCF4)
       fprintf(fOut,"   sampling = CCF4\n");
  else fprintf(fOut,"#  sampling = CCF4\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCF5)
       fprintf(fOut,"   sampling = CCF5\n");
  else fprintf(fOut,"#  sampling = CCF5\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCFF)
       fprintf(fOut,"   sampling = CCFF\n");
  else fprintf(fOut,"#  sampling = CCFF\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCC4)
       fprintf(fOut,"   sampling = CCC4\n");
  else fprintf(fOut,"#  sampling = CCC4\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCC5)
       fprintf(fOut,"   sampling = CCC5\n");
  else fprintf(fOut,"#  sampling = CCC5\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_CCCF)
       fprintf(fOut,"   sampling = CCCF\n");
  else fprintf(fOut,"#  sampling = CCCF\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_SFAST)
       fprintf(fOut,"   sampling = SFAST\n");
  else fprintf(fOut,"#  sampling = SFAST\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_UMETIS)
       fprintf(fOut,"   sampling = UMETIS\n");
  else fprintf(fOut,"#  sampling = UMETIS\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_GMOAT)
       fprintf(fOut,"   sampling = GMOAT\n");
  else fprintf(fOut,"#  sampling = GMOAT\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_GMETIS)
       fprintf(fOut,"   sampling = GMETIS\n");
  else fprintf(fOut,"#  sampling = GMETIS\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_SG)
       fprintf(fOut,"   sampling = SPARSEGRID\n");
  else fprintf(fOut,"#  sampling = SPARSEGRID\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_LSA)
       fprintf(fOut,"   sampling = LSA\n");
  else fprintf(fOut,"#  sampling = LSA\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_RFF4)
       fprintf(fOut,"   sampling = RFF4\n");
  else fprintf(fOut,"#  sampling = RFF4\n");
  if (pMethod_.samplingMethod_ == PSUADE_SAMP_RFF5)
       fprintf(fOut,"   sampling = RFF5\n");
  else fprintf(fOut,"#  sampling = RFF5\n");
  fprintf(fOut,"   num_samples = %d\n", pMethod_.nSamples_);
  fprintf(fOut,"   num_replications = %d\n", pMethod_.nReplications_);

  fprintf(fOut,"   num_refinements = %d\n", pMethod_.nRefinements_);
  fprintf(fOut,"   refinement_size = %d\n", pMethod_.refinementSize_);
  fprintf(fOut,"   reference_num_refinements = %d\n", 
          pMethod_.refNumRefinements_);
  if (pMethod_.refinementType_ == 1)
     fprintf(fOut,"   refinement_type = adaptive\n"); 
  else
     fprintf(fOut,"#  refinement_type = adaptive\n"); 
  if (pInput_.VecInpSyms_.length() > 0) 
  {
    for (ii = 0; ii < pInput_.nInputs_; ii++)
      fprintf(fOut,"   num_symbols %d = %d\n",ii+1, 
              pInput_.VecInpSyms_[ii]);
  }
  if (pMethod_.sampleRandomize_ == 0)
  {
    fprintf(fOut,"#  randomize\n");
    fprintf(fOut,"#  randomize_more\n");
  }
  else if (pMethod_.sampleRandomize_ == 1)
  {
    fprintf(fOut,"   randomize\n");
    fprintf(fOut,"#  randomize_more\n");
  }
  else if (pMethod_.sampleRandomize_ == 3)
  {
    fprintf(fOut,"   randomize\n");
    fprintf(fOut,"   randomize_more\n");
  }
  if (pInput_.VecInpNumVals_.length() > 0) 
  {
    for (ii = 0; ii < pInput_.nInputs_; ii++)
    {
      if (pInput_.VecInpNumVals_[ii] > 0) 
      {
        fprintf(fOut,"   input_settings %4d %4d\n", ii+1,
                pInput_.VecInpNumVals_[ii]);
        for (kk = 0; kk < pInput_.VecInpNumVals_[ii]; kk++)
          fprintf(fOut,"                  %12.4e\n", 
                  pInput_.MatInpVals_.getEntry(ii, kk));
      }
    }
  }
  else
  {
    fprintf(fOut,"#  an example of settings: input 1 with 3 settings\n");
    fprintf(fOut,"#  input_settings 1 3\n");
    fprintf(fOut,"#                 0.0\n"); 
    fprintf(fOut,"#                 0.5\n"); 
    fprintf(fOut,"#                 1.0\n"); 
  } 
  if (psConfig_.getRandomSeed() != -1)
       fprintf(fOut,"   random_seed = %ld\n", psConfig_.getRandomSeed());
  else fprintf(fOut,"#  random_seed = 2147483647\n");
  fprintf(fOut,"END\n");
}

// ************************************************************************
// A function for writing application section to a file
// ------------------------------------------------------------------------ 
void PsuadeData::writeApplicationSection(FILE *fOut) 
{
  fprintf(fOut,"APPLICATION\n");
  if (strcmp(pApplication_.appDriver_, "NONE"))
       fprintf(fOut,"   driver = %s\n", pApplication_.appDriver_);
  else fprintf(fOut,"   driver = NONE\n");
  if (strcmp(pApplication_.optDriver_, "NONE"))
       fprintf(fOut,"   opt_driver = %s\n", pApplication_.optDriver_);
  else fprintf(fOut,"   opt_driver = NONE\n");
  if (strcmp(pApplication_.auxOptDriver_, "NONE"))
       fprintf(fOut,"   aux_opt_driver = %s\n",pApplication_.auxOptDriver_);
  else fprintf(fOut,"   aux_opt_driver = NONE\n");
  if (strcmp(pApplication_.ensembleDriver_, "NONE"))
       fprintf(fOut,"   ensemble_driver = %s\n",pApplication_.ensembleDriver_);
  else fprintf(fOut,"   ensemble_driver = NONE\n");
  if (strcmp(pApplication_.ensembleOptDriver_, "NONE"))
       fprintf(fOut,"   ensemble_opt_driver = %s\n",
               pApplication_.ensembleOptDriver_);
  else fprintf(fOut,"   ensemble_opt_driver = NONE\n");
  //**/disable to rid of strstr problem
  //**/if (strcmp(pApplication_.inputTemplate_, "NONE"))
  //**/fprintf(fOut,"   input_template = %s\n",pApplication_.inputTemplate_);
  //**/else fprintf(fOut,"#  input_template = NONE\n");
  //**/if (strcmp(pApplication_.outputTemplate_, "NONE"))
  //**/fprintf(fOut,"   output_template = %s\n",pApplication_.outputTemplate_);
  //**/else fprintf(fOut,"#  output_template = NONE\n");
  if (pApplication_.maxParallelJobs_ != 1)
       fprintf(fOut,"   max_parallel_jobs = %d\n",
               pApplication_.maxParallelJobs_);
  else fprintf(fOut,"#  max_parallel_jobs = 1\n");
  if (pApplication_.minJobWaitTime_ != 1)
       fprintf(fOut,"   min_job_wait_time = %d\n",
               pApplication_.minJobWaitTime_);
  else fprintf(fOut,"#  min_job_wait_time = 1\n");
  if (pApplication_.maxJobWaitTime_ != 1)
       fprintf(fOut,"   max_job_wait_time = %d\n",
               pApplication_.maxJobWaitTime_);
  else fprintf(fOut,"#  max_job_wait_time = 1000000\n");
  if (pApplication_.runType_ & 1) fprintf(fOut,"   nondeterministic\n");
  else                            fprintf(fOut,"#  nondeterministic\n");
  if (pApplication_.runType_ & 2) fprintf(fOut,"   launch_only\n");
  else                            fprintf(fOut,"#  launch_only\n");
  if (pApplication_.runType_ & 4) fprintf(fOut,"   limited_launch_only\n");
  else                            fprintf(fOut,"#  limited_launch_only\n");
  if (pApplication_.runType_ & 8) fprintf(fOut,"   gen_inputfile_only\n");
  else                            fprintf(fOut,"#  gen_inputfile_only\n");
  if (pApplication_.runType_ & 16) fprintf(fOut,"   ensemble_run_mode\n");
  else                             fprintf(fOut,"#  ensemble_run_mode\n");
  if (pApplication_.launchInterval_ != 1)
       fprintf(fOut,"   launch_interval = %d\n",pApplication_.launchInterval_);
  else fprintf(fOut,"#  launch_interval = 1\n");
  if (pApplication_.saveFrequency_ != 1000000)
       fprintf(fOut,"   save_frequency = %d\n",pApplication_.saveFrequency_);
  else fprintf(fOut,"#  save_frequency = 1000000\n");
  fprintf(fOut,"END\n");
}

// ************************************************************************
// A function for writing analysis section to a file
// ------------------------------------------------------------------------ 
void PsuadeData::writeAnalysisSection(FILE *fOut) 
{
  int ii;

  fprintf(fOut,"ANALYSIS\n");
  fprintf(fOut, "##**********************************************\n");
  fprintf(fOut, "## Moment - basic statistics\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_MOMENT) != 0) 
       fprintf(fOut,"   analyzer method = Moment\n"); 
  else fprintf(fOut,"#  analyzer method = Moment\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## MainEffect - raw data main effect analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_ME) != 0) 
       fprintf(fOut,"   analyzer method = MainEffect\n"); 
  else fprintf(fOut,"#  analyzer method = MainEffect\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## TwoParamEffect - raw data pairwise analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_IE) != 0) 
       fprintf(fOut,"   analyzer method = TwoParamEffect\n"); 
  else fprintf(fOut,"#  analyzer method = TwoParamEffect\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ANOVA - analysis of variance\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_ANOVA) != 0) 
       fprintf(fOut,"   analyzer method = ANOVA\n"); 
  else fprintf(fOut,"#  analyzer method = ANOVA\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## GLSA - gradient-based sensitivity analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_GLSA)!= 0) 
       fprintf(fOut,"   analyzer method = GLSA\n");
  else fprintf(fOut,"#  analyzer method = GLSA\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## RSFA - response surface analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_RSFA) != 0)
       fprintf(fOut,"   analyzer method = RSFA\n");
  else fprintf(fOut,"#  analyzer method = RSFA\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## MOAT - Morris screening analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_MOAT) != 0)
       fprintf(fOut,"   analyzer method = MOAT\n");
  else fprintf(fOut,"#  analyzer method = MOAT\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## SOBOL - Sobol' analysis on Sobol' samples\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_SOBOL) != 0)
       fprintf(fOut,"   analyzer method = Sobol\n");
  else fprintf(fOut,"#  analyzer method = Sobol\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## Correlation - classical correlation analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_CORRELATION) != 0)
       fprintf(fOut,"   analyzer method = Correlation\n");
  else fprintf(fOut,"#  analyzer method = Correlation\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## Integration - find area under response surface\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_INTEGRATION) != 0)
       fprintf(fOut,"   analyzer method = Integration\n");
  else fprintf(fOut,"#  analyzer method = Integration\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## FAST - total sensitivity analysis using FAST samples\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_FAST) != 0)
       fprintf(fOut,"   analyzer method = FAST\n");
  else fprintf(fOut,"#  analyzer method = FAST\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## FF - screening using fractional factorial samples\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_FF) != 0)
       fprintf(fOut,"   analyzer method = FF\n");
  else fprintf(fOut,"#  analyzer method = FF\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## PCA - principal component analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_PCA) != 0)
       fprintf(fOut,"   analyzer method = PCA\n");
  else fprintf(fOut,"#  analyzer method = PCA\n");
  //**/ never tested (3/2017)
  //if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_FORM) != 0)
  //     fprintf(fOut,"   analyzer method = FORM\n");
  //else fprintf(fOut,"#  analyzer method = FORM\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## RSMSobol1 - response surface based main effect\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_RSSOBOL1) != 0)
       fprintf(fOut,"   analyzer method = RSMSobol1\n");
  else fprintf(fOut,"#  analyzer method = RSMSobol1\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## RSMSobol2 - response surface based pairwise effect\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_RSSOBOL2) != 0)
       fprintf(fOut,"   analyzer method = RSMSobol2\n");
  else fprintf(fOut,"#  analyzer method = RSMSobol2\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## RSMSobolTSI - response surface based total effect\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_RSSOBOLTSI) != 0)
       fprintf(fOut,"   analyzer method = RSMSobolTSI\n");
  else fprintf(fOut,"#  analyzer method = RSMSobolTSI\n");
  //if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_BSTRAP) != 0)
  //     fprintf(fOut,"   analyzer method = Bootstrap\n");
  //else fprintf(fOut,"#  analyzer method = Bootstrap\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## RSMSobolG - response surface based group effect\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_RSSOBOLG) != 0)
       fprintf(fOut,"   analyzer method = RSMSobolG\n");
  else fprintf(fOut,"#  analyzer method = RSMSobolG\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ARSM - adaptive NN-based response surface analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_ARSMNN) != 0)
       fprintf(fOut,"   analyzer method = ARSMNN\n");
  else fprintf(fOut,"#  analyzer method = ARSMNN\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ARSM - adaptive MARS-based response surface analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_ARSMMB) != 0)
       fprintf(fOut,"   analyzer method = ARSM\n");
  else fprintf(fOut,"#  analyzer method = ARSM\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## REL - reliability analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_REL) != 0)
       fprintf(fOut,"   analyzer method = REL\n");
  else fprintf(fOut,"#  analyzer method = REL\n");
  //if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_AOPT) != 0)
  //     fprintf(fOut,"   analyzer method = AOPT\n");
  //else fprintf(fOut,"#  analyzer method = AOPT\n");
  //if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_GOWER) != 0)
  //     fprintf(fOut,"   analyzer method = GOWER\n");
  //else fprintf(fOut,"#  analyzer method = GOWER\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## DELTA - Delta test for parameter screening\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_DTEST) != 0)
       fprintf(fOut,"   analyzer method = DELTA\n");
  else fprintf(fOut,"#  analyzer method = DELTA\n");
  //if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_ETEST) != 0)
  //     fprintf(fOut,"   analyzer method = ETA\n");
  //else fprintf(fOut,"#  analyzer method = ETA\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## LSA - local sensitivity analysis\n");
  if ((pAnalysis_.analysisIntOptions_[0] & PSUADE_ANA_LSA) != 0)
       fprintf(fOut,"   analyzer method = LSA\n");
  else fprintf(fOut,"#  analyzer method = LSA\n");
  fprintf(fOut, "##**********************************************\n");

  if (pAnalysis_.analysisIntOptions_[1]+1 <= pOutput_.nOutputs_)
       fprintf(fOut,"   analyzer output_id  = %d\n", 
               pAnalysis_.analysisIntOptions_[1]+1);
  else fprintf(fOut,"   analyzer output_id  = 1\n");

  fprintf(fOut, "##**********************************************\n");
  fprintf(fOut, "##RS: MARS - multivariate adaptive regression\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MARS)
       fprintf(fOut,"   analyzer rstype = MARS\n");
  else fprintf(fOut,"#  analyzer rstype = MARS\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: linear - linear regression\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGR1)
       fprintf(fOut,"   analyzer rstype = linear\n"); 
  else fprintf(fOut,"#  analyzer rstype = linear\n"); 
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: quadratic - quadratic regression\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGR2)
       fprintf(fOut,"   analyzer rstype = quadratic\n");
  else fprintf(fOut,"#  analyzer rstype = quadratic\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: cubic - third-order polynomial\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGR3)
       fprintf(fOut,"   analyzer rstype = cubic\n");
  else fprintf(fOut,"#  analyzer rstype = cubic\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: quartic - fourth-order polynomial\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGR4)
       fprintf(fOut,"   analyzer rstype = quartic\n");
  else fprintf(fOut,"#  analyzer rstype = quartic\n");
  //**/if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_ANN)
  //**/     fprintf(fOut,"   analyzer rstype = ANN\n");
  //**/else fprintf(fOut,"#  analyzer rstype = ANN\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: selective - selected polynomial order terms\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGRS)
       fprintf(fOut,"   analyzer rstype = selective_regression\n");
  else fprintf(fOut,"#  analyzer rstype = selective_regression\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: GP1 - Gaussian process by MacKay\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_GP1)
       fprintf(fOut,"   analyzer rstype = GP1\n");
  else fprintf(fOut,"#  analyzer rstype = GP1\n");
  fprintf(fOut, "##RS: GP3 - Gaussian process by Tong\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_GP3)
       fprintf(fOut,"   analyzer rstype = GP3\n");
  else fprintf(fOut,"#  analyzer rstype = GP3\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: SVM - support vector machine\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_SVM)
       fprintf(fOut,"   analyzer rstype = SVM\n");
  else fprintf(fOut,"#  analyzer rstype = SVM\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: PWL - piecewise linear approximation\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_PWL)
       fprintf(fOut,"   analyzer rstype = PWL\n");
  else fprintf(fOut,"#  analyzer rstype = PWL\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: TGP - treed Gaussian process by Lee et al.\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_TGP)
       fprintf(fOut,"   analyzer rstype = TGP\n");
  else fprintf(fOut,"#  analyzer rstype = TGP\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: MARSBag - MARS with bagging\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MARSB)
       fprintf(fOut,"   analyzer rstype = MARSBag\n");
  else fprintf(fOut,"#  analyzer rstype = MARSBag\n");
  //**/if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_EARTH)
  //**/     fprintf(fOut,"   analyzer rstype = EARTH\n");
  //**/else fprintf(fOut,"#  analyzer rstype = EARTH\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: sum_of_trees based on repeated bisections\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_SOTS)
       fprintf(fOut,"   analyzer rstype = sum_of_trees\n");
  else fprintf(fOut,"#  analyzer rstype = sum_of_trees\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: Legendre - Legendre polynomial regression\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGRL)
       fprintf(fOut,"   analyzer rstype = Legendre\n");
  else fprintf(fOut,"#  analyzer rstype = Legendre\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: user - user provides basis functions\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGRU)
       fprintf(fOut,"   analyzer rstype = user_regression\n");
  else fprintf(fOut,"#  analyzer rstype = user_regression\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: sparse_grid - only with special quadrature pts \n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_REGSG)
       fprintf(fOut,"   analyzer rstype = sparse_grid_regression\n");
  else fprintf(fOut,"#  analyzer rstype = sparse_grid_regression\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: Krigining - using 2nd order correlation\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_KR)
       fprintf(fOut,"   analyzer rstype = Kriging\n");
  else fprintf(fOut,"#  analyzer rstype = Kriging\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: splines - splines on 2D/3D grid samples\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_SPLINES)
       fprintf(fOut,"   analyzer rstype = splines\n");
  else fprintf(fOut,"#  analyzer rstype = splines\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: KNN - k-nearest neighbors\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_KNN)
       fprintf(fOut,"   analyzer rstype = KNN\n");
  else fprintf(fOut,"#  analyzer rstype = KNN\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: RBF - radial basis function\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_RBF)
       fprintf(fOut,"   analyzer rstype = RBF\n");
  else fprintf(fOut,"#  analyzer rstype = RBF\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: ACOSSO - Curt Storlie's ACOSSO\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_ACOSSO)
       fprintf(fOut,"   analyzer rstype = Acosso\n");
  else fprintf(fOut,"#  analyzer rstype = Acosso\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: BSSANOVA - Curt Storlie's BSSANOVA\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_BSSANOVA)
       fprintf(fOut,"   analyzer rstype = Bssanova\n");
  else fprintf(fOut,"#  analyzer rstype = Bssanova\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: psuade_regression - PSUADE's internal function\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_LOCAL)
       fprintf(fOut,"   analyzer rstype = psuade_regression\n");
  else fprintf(fOut,"#  analyzer rstype = psuade_regression\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: RBFBag - RBF with bootstraps\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_RBFB)
       fprintf(fOut,"   analyzer rstype = RBFBag\n");
  else fprintf(fOut,"#  analyzer rstype = RBFBag\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: PLS - partial least squares (correlated inputs)\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_PLS)
       fprintf(fOut,"   analyzer rstype = PLS\n");
  else fprintf(fOut,"#  analyzer rstype = PLS\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: MRBF - multiple RBF\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MRBF)
       fprintf(fOut,"   analyzer rstype = MRBF\n");
  else fprintf(fOut,"#  analyzer rstype = MRBF\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: MGP3 - multiple GP3\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MGP3)
       fprintf(fOut,"   analyzer rstype = MGP3\n");
  else fprintf(fOut,"#  analyzer rstype = MGP3\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: MMARS - multiple MARS\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MMARS)
       fprintf(fOut,"   analyzer rstype = MMARS\n");
  else fprintf(fOut,"#  analyzer rstype = MMARS\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: MTGP - multiple TGP\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MTGP)
       fprintf(fOut,"   analyzer rstype = MTGP\n");
  else fprintf(fOut,"#  analyzer rstype = MTGP\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##RS: MNN - multi-domain Neural Network\n");
  if (pAnalysis_.analysisIntOptions_[2] == PSUADE_RS_MDNN)
       fprintf(fOut,"   analyzer rstype = MNN\n");
  else fprintf(fOut,"#  analyzer rstype = MNN\n");
  fprintf(fOut, "##==============================================\n");
  if (pAnalysis_.legendreOrder_ > 0)
       fprintf(fOut,"   analyzer rs_legendre_order = %d\n", 
               pAnalysis_.legendreOrder_);
  else fprintf(fOut,"#  analyzer rs_legendre_order = -1\n");
  if (pAnalysis_.marsNbasis_ > 0)
       fprintf(fOut,"   analyzer rs_mars_num_bases = %d\n", 
               pAnalysis_.marsNbasis_);
  else fprintf(fOut,"#  analyzer rs_mars_num_bases = -1\n");
  if (pAnalysis_.marsNdegrees_ > 0)
       fprintf(fOut,"   analyzer rs_mars_interaction = %d\n", 
               pAnalysis_.marsNdegrees_);
  else fprintf(fOut,"#  analyzer rs_mars_interaction = -1\n");
  if (pAnalysis_.marsNum_ > 0)
       fprintf(fOut,"   analyzer rs_num_mars = %d\n", pAnalysis_.marsNum_);
  else fprintf(fOut,"#  analyzer rs_num_mars = -1\n");
  if (pAnalysis_.kriMode_ > 0)
       fprintf(fOut,"   analyzer rs_kriging_mode = %d\n", pAnalysis_.kriMode_);
  else fprintf(fOut,"#  analyzer rs_kriging_mode = -1\n");
  if (pAnalysis_.kriMode_ > 0)
       fprintf(fOut,"   analyzer rs_kriging_tol = %e\n", pAnalysis_.kriTol_);
  else fprintf(fOut,"#  analyzer rs_kriging_tol = -1\n");
  if (pAnalysis_.optSaveHistory_ == 1)
       fprintf(fOut,"   analyzer opt_save_history\n");
  else fprintf(fOut,"#  analyzer opt_save_history\n");
  if (pAnalysis_.optUseHistory_ == 1)
       fprintf(fOut,"   analyzer opt_use_history\n");
  else fprintf(fOut,"#  analyzer opt_use_history\n");
  if (pAnalysis_.analysisIntOptions_[6] >= 0 && 
      pAnalysis_.analysisIntOptions_[6] < pOutput_.nOutputs_)
       fprintf(fOut,"   analyzer regression_wgt_id = %d\n",
          pAnalysis_.analysisIntOptions_[6]+1);
  else fprintf(fOut,"#  analyzer regression_wgt_id = -1\n");
  if ((pAnalysis_.analysisIntOptions_[4] & PSUADE_GRAPHICS) != 0) 
       fprintf(fOut,"   graphics\n");
  else fprintf(fOut,"#  graphics\n");
  if ((pAnalysis_.analysisIntOptions_[4] & PSUADE_SAMPLE_GRAPHICS) != 0)
       fprintf(fOut,"   sample_graphics\n");
  else fprintf(fOut,"#  sample_graphics\n");
  fprintf(fOut,"   analyzer threshold = %e\n", 
          pAnalysis_.analysisDbleOptions_[0]);
  if (psConfig_.RSMaxPts_ != 5000)
       fprintf(fOut,"   rs_max_pts = %d\n", psConfig_.RSMaxPts_);
  else fprintf(fOut,"#  rs_max_pts = 5000\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## rs_constraint - for constrained UA/SA analysis\n");
  if (pAnalysis_.numRSFilters_ > 0)
  {
    for (ii = 0; ii < pAnalysis_.numRSFilters_; ii++)
    {
      fprintf(fOut,"   analyzer rs_constraint = %s %s %e %e\n",
              pAnalysis_.RSFilters_[ii]->FilterDataFile_,
              pAnalysis_.RSFilters_[ii]->FilterIndexFile_,
              pAnalysis_.RSFilters_[ii]->FilterLBound_,
              pAnalysis_.RSFilters_[ii]->FilterUBound_);
    }
  }
  else
  {
    fprintf(fOut,
       "#  analyzer rs_constraint = psData indexFile Lbnd Ubnd\n");
  }

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## moat_constraint - for constrained MOAT analysis\n");
  if (pAnalysis_.numMOATFilters_ > 0)
  {
     for (ii = 0; ii < pAnalysis_.numMOATFilters_; ii++)
     {
       fprintf(fOut,"   analyzer moat_constraint = %s %s %e %e\n",
               pAnalysis_.MOATFilters_[ii]->FilterDataFile_,
               pAnalysis_.MOATFilters_[ii]->FilterIndexFile_,
               pAnalysis_.MOATFilters_[ii]->FilterLBound_,
               pAnalysis_.MOATFilters_[ii]->FilterUBound_);
     }
  }
  else
  {
    fprintf(fOut,
      "## analyzer moat_constraint = psData indexFile Lbnd Ubnd\n");
  }

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## rs_index_file - use rs but fix some inputs\n");
  if (strcmp(pAnalysis_.rsIndexFile_, "NONE"))
     fprintf(fOut,"   analyzer rs_index_file = %s\n",
             pAnalysis_.rsIndexFile_);
  else
     fprintf(fOut,"#  analyzer rs_index_file = <indexFile>\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## rs_index_sample_file - sample for some inputs\n");
  if (strcmp(pAnalysis_.rsIndexSampleFile_, "NONE"))
     fprintf(fOut,"   analyzer rs_index_sample_file = %s\n",
             pAnalysis_.rsIndexSampleFile_);
  else
     fprintf(fOut,"#  analyzer rs_index_sample_file = <sampleFile>\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## crude - optimize in raw data or rs spaces\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) && 
      (pAnalysis_.optimizeIntOptions_[1] == 0)) 
       fprintf(fOut, "   optimization method = crude\n");
  else fprintf(fOut, "#  optimization method = crude\n");

  //**/if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
  //**/    (pAnalysis_.optimizeIntOptions_[1] == 1)) 
  //**/     fprintf(fOut, "   optimization method = txmath\n");
  //**/else fprintf(fOut, "#  optimization method = txmath\n");

  //**/if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
  //**/    (pAnalysis_.optimizeIntOptions_[1] == 2)) 
  //**/     fprintf(fOut, "   optimization method = appspack\n");
  //**/else fprintf(fOut, "#  optimization method = appspack\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## minpack - optimize with user provided gradients\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 3)) 
       fprintf(fOut, "   optimization method = minpack\n");
  else fprintf(fOut, "#  optimization method = minpack\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## sm - space mapping optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 5)) 
       fprintf(fOut, "   optimization method = sm\n");
  else fprintf(fOut, "#  optimization method = sm\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## mm - manifold mapping optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 6)) 
       fprintf(fOut, "   optimization method = mm\n");
  else fprintf(fOut, "#  optimization method = mm\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## mm - adaptive manifold mapping optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 7)) 
       fprintf(fOut, "   optimization method = mm_adaptive\n");
  else fprintf(fOut, "#  optimization method = mm_adaptive\n");

  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## cobyla: nonlinear inequality-constrained opt\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 4)) 
       fprintf(fOut, "   optimization method = cobyla\n");
  else fprintf(fOut, "#  optimization method = cobyla\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## bobyqa: bound-constrained optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 8)) 
       fprintf(fOut, "   optimization method = bobyqa\n");
   else fprintf(fOut, "#  optimization method = bobyqa\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## lincoa: linear inequality constrained opt\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 14)) 
       fprintf(fOut, "   optimization method = lincoa\n");
  else fprintf(fOut, "#  optimization method = lincoa\n");
#ifdef HAVE_NEWUOA
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## newuoa: unconstrained optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 15)) 
        fprintf(fOut, "   optimization method = newuoa\n");
  else fprintf(fOut, "#  optimization method = newuoa\n");
#endif
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## lbfgs - with derivatives\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 19))
       fprintf(fOut, "   optimization method = lbfgs\n");
  else fprintf(fOut, "#  optimization method = lbfgs\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "##  sce: genetic algorithm-type optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 9))
       fprintf(fOut, "   optimization method = sce\n");
  else fprintf(fOut, "#  optimization method = sce\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## moo - multi-objective optimization\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 10))
       fprintf(fOut, "   optimization method = moo\n");
  else fprintf(fOut, "#  optimization method = moo\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ouu - optimization under uncertainty\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 11))
       fprintf(fOut, "   optimization method = ouu\n");
  else fprintf(fOut, "#  optimization method = ouu\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ouu_unconstr - ouu with no constraints \n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 16))
       fprintf(fOut, "   optimization method = ouu_unconstr\n");
  else fprintf(fOut, "#  optimization method = ouu_unconstr\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ouu_ineq_constr - ouu with inequality constraints \n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 18))
       fprintf(fOut, "   optimization method = ouu_ineq_constr\n");
  else fprintf(fOut, "#  optimization method = ouu_ineq_constr\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ouu_lbfgs - ouu with derivatives\n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 19))
       fprintf(fOut, "   optimization method = ouu_lbfgs\n");
  else fprintf(fOut, "#  optimization method = ouu_lbfgs\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## nomad - mixed integer \n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 21))
       fprintf(fOut, "   optimization method = nomad\n");
  else fprintf(fOut, "#  optimization method = nomad\n");
  fprintf(fOut, "##==============================================\n");
  fprintf(fOut, "## ouu - mixed integer \n");
  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[1] == 22))
       fprintf(fOut, "   optimization method = ouu_minlp\n");
  else fprintf(fOut, "#  optimization method = ouu_minlp\n");
  //**/ Dec, 2015 - hide from users
  //**/ Dec, 2015 - hide from users
  //**/if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
  //**/    (pAnalysis_.optimizeIntOptions_[1] == 12)) 
  //**/     fprintf(fOut, "   optimization method = ouu1\n");
  //**/else fprintf(fOut, "#  optimization method = ouu1\n");

  //**/if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
  //**/    (pAnalysis_.optimizeIntOptions_[1] == 13)) 
  //**/     fprintf(fOut, "   optimization method = ouu2\n");
  //**/else fprintf(fOut, "#  optimization method = ouu2\n");
  fprintf(fOut, "#***********************************************\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[2] > 0)) 
       fprintf(fOut, "   optimization num_starts = %d\n",
               pAnalysis_.optimizeIntOptions_[2]);
  else fprintf(fOut, "#  optimization num_starts = 0\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[3] == 1)) 
       fprintf(fOut, "   optimization use_response_surface\n");
  else fprintf(fOut, "#  optimization use_response_surface\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[4] > 0)) 
       fprintf(fOut, "   optimization print_level = %d\n", 
               pAnalysis_.optimizeIntOptions_[4]);
  else fprintf(fOut, "#  optimization print_level = 0\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[5] > 0)) 
       fprintf(fOut, "   optimization num_fmin = %d\n", 
               pAnalysis_.optimizeIntOptions_[5]);
  else fprintf(fOut, "#  optimization num_fmin = 0\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeIntOptions_[6] >= 0)) 
       fprintf(fOut, "   optimization output_id = %d\n", 
               pAnalysis_.optimizeIntOptions_[6]+1);
  else fprintf(fOut, "#  optimization output_id = 0\n");

  if (pAnalysis_.optimizeIntOptions_[0] > 0)
       fprintf(fOut, "   optimization max_feval = %d\n", 
               pAnalysis_.optimizeIntOptions_[7]);
  else fprintf(fOut, "#  optimization max_feval = 10000\n");

  if (pAnalysis_.optimizeIntOptions_[0] > 0)
       fprintf(fOut, "   optimization deltax = %e\n",
               pAnalysis_.optimizeDbleOptions_[3]);
  else fprintf(fOut, "#  optimization deltax = 1.0e-6\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeDbleOptions_[0] != -1.0e50)) 
       fprintf(fOut, "   optimization fmin = %e\n",
               pAnalysis_.optimizeDbleOptions_[0]);
  else fprintf(fOut, "#  optimization fmin = not defined\n");

  if ((pAnalysis_.optimizeIntOptions_[0] > 0) &&
      (pAnalysis_.optimizeDbleOptions_[1] != 1.0e50)) 
       fprintf(fOut, "   optimization cutoff = %e\n",
               pAnalysis_.optimizeDbleOptions_[1]);
  else fprintf(fOut, "#  optimization cutoff = not defined\n");

  if (pAnalysis_.optimizeIntOptions_[0] > 0)
       fprintf(fOut, "   optimization tolerance = %e\n",
               pAnalysis_.optimizeDbleOptions_[2]);
  else fprintf(fOut, "#  optimization tolerance = not defined\n");

  if (pAnalysis_.analysisIntOptions_[3] == 0)
       fprintf(fOut,"#  printlevel \n"); 
  else fprintf(fOut,"   printlevel  %d\n",pAnalysis_.analysisIntOptions_[3]); 

  if ((pAnalysis_.fileWriteFlag_ & 1) != 0)
       fprintf(fOut,"   file_write matlab\n");
  else fprintf(fOut,"#  file_write matlab\n");
  if (pAnalysis_.useInputPDFs_ == 1)
       fprintf(fOut,"   use_input_pdfs\n");
  else fprintf(fOut,"#  use_input_pdfs\n");
  if (psConfig_.RSConstraintSetOp_ == 1)
       fprintf(fOut,"   constraint_op_and\n");
  else fprintf(fOut,"#  constraint_op_and\n");
  if (psConfig_.IOExpertModeIsOn())
     fprintf(fOut,"   io_expert\n");
  if (psConfig_.RSExpertModeIsOn())
     fprintf(fOut,"   rs_expert\n");
  if (psConfig_.AnaExpertModeIsOn())
     fprintf(fOut,"   ana_expert\n");
  if (psConfig_.OptExpertModeIsOn())
     fprintf(fOut,"   opt_expert\n");
  if (psConfig_.SamExpertModeIsOn())
     fprintf(fOut,"   sam_expert\n");
  if (psConfig_.PlotTool_ == 1)
     fprintf(fOut,"   scilab\n");
  fprintf(fOut,"END\n");
}

// ************************************************************************
// Return the state of a single sample, from pOutput_.VecSamStas_ array
// If given id = -1, return the number of samples
// ------------------------------------------------------------------------ 
int PsuadeData::getSampleState(int sampleid)
{
  if (sampleid == -1)
     return pMethod_.nSamples_;
  else
     return pOutput_.VecSamStas_[sampleid];
}

// ************************************************************************
// Return the value of a single sample input, from pInput_.VecSamInps_ 
// array. If given id = -1, return the number of inputs
// ------------------------------------------------------------------------ 
double PsuadeData::getSampleInput(int inputNumber, int sampleid)
{
  if (sampleid == -1)
    return pInput_.nInputs_;
  else
    return pInput_.VecSamInps_[sampleid*pInput_.nInputs_+inputNumber];
}

// ************************************************************************
// Return the value of a single sample output, from pOutput_.VecSamOuts_ 
// array. If given id = -1, return the number of outputs
// ------------------------------------------------------------------------ 
double PsuadeData::getSampleOutput(int outputNumber, int sampleid)
{
  if (sampleid == -1)
    return pOutput_.nOutputs_;
  else
    return pOutput_.VecSamOuts_[sampleid*pOutput_.nOutputs_+outputNumber];
}

// ************************************************************************
// ************************************************************************
// Subclass function definition
// ************************************************************************
// ************************************************************************

// ************************************************************************
// functions for psuadeInputSection
// ------------------------------------------------------------------------ 
psuadeInputSection::psuadeInputSection(): nFixedInps_(0),
                    nInputs_(0), useInputPDFs_(0) 
{
}

// ------------------------------------------------------------------------ 
psuadeInputSection::~psuadeInputSection()
{ 
  reset();
}

// ------------------------------------------------------------------------ 
void psuadeInputSection::reset()
{ 
  StrSamFileNames_.clean();
  StrInpNames_.clean();
  MatInpVals_.clean();
  VecInpSInds_.clean();
  VecSamInps_.clean();
  VecInpLBds_.clean();
  VecInpUBds_.clean();
  VecInpPDFs_.clean();
  VecInpMeans_.clean();
  VecInpStdvs_.clean();
  VecInpAuxs_.clean();
  VecInpSyms_.clean();
  VecInpNumVals_.clean();
  corMatrix_.setDim(0,0);
  nInputs_ = 0;
  useInputPDFs_ = 0;
}

// ************************************************************************
// functions for psuadeOutputSection
// ------------------------------------------------------------------------ 
psuadeOutputSection::psuadeOutputSection() : nOutputs_(0)
{
}

// ------------------------------------------------------------------------ 
psuadeOutputSection::~psuadeOutputSection()
{ 
   reset();
}

// ------------------------------------------------------------------------ 
void psuadeOutputSection::reset()
{ 
  nOutputs_ = 0;
  StrOutNames_.clean();
  VecSamOuts_.clean();
  VecSamStas_.clean();
}

// ************************************************************************
// functions for psuadeMethodSection
// ------------------------------------------------------------------------ 
psuadeMethodSection::psuadeMethodSection()
{ 
  reset();
}

// ------------------------------------------------------------------------ 
psuadeMethodSection::~psuadeMethodSection()
{ 
}

// ------------------------------------------------------------------------ 
void psuadeMethodSection::reset()
{ 
  nSamples_          = 0;
  samplingMethod_    = PSUADE_SAMP_MC;
  nReplications_     = 1;
  nRefinements_      = 0;
  refinementType_    = 0;
  refinementSize_    = 10000000;
  refNumRefinements_ = 0;
  sampleRandomize_   = 0;
}

// ************************************************************************
// functions for psuadeApplicationSection
// ------------------------------------------------------------------------ 
psuadeApplicationSection::psuadeApplicationSection()
{ 
  reset();
}

// ------------------------------------------------------------------------ 
psuadeApplicationSection::~psuadeApplicationSection()
{ 
}

// ------------------------------------------------------------------------ 
void psuadeApplicationSection::reset()
{ 
  strcpy(appDriver_, "NONE");
  strcpy(rsDriver_, "NONE");
  strcpy(optDriver_, "NONE");
  strcpy(auxOptDriver_, "NONE");
  strcpy(ensembleDriver_, "NONE");
  strcpy(ensembleOptDriver_, "NONE");
  strcpy(inputTemplate_, "NONE");
  strcpy(outputTemplate_, "NONE");
  maxParallelJobs_ = 1;
  minJobWaitTime_  = 1;
  maxJobWaitTime_  = 1000000;
  launchInterval_  = 1;
  saveFrequency_   = 1000000;
  runType_         = 0;
  useFunction_     = 0;
}

// ************************************************************************
// functions for psuadeAnalysisSection
// ------------------------------------------------------------------------ 
psuadeAnalysisSection::psuadeAnalysisSection()
{ 
  numRSFilters_   = 0;
  numMOATFilters_ = 0;
  RSFilters_   = NULL;
  MOATFilters_ = NULL;
  useInputPDFs_ = 0;
  legendreOrder_ = -1;
  marsNbasis_ = -1;
  marsNdegrees_ = -1;
  marsNum_ = -1;
  kriMode_ = -1;
  kriTol_ = -1;
  optSaveHistory_ = 0;
  optUseHistory_ = 0;
  reset();
}

// ------------------------------------------------------------------------ 
psuadeAnalysisSection::~psuadeAnalysisSection()
{ 
}

// ------------------------------------------------------------------------ 
void psuadeAnalysisSection::reset()
{ 
  for (int ii = 0; ii < 10; ii++)
  {
    optimizeIntOptions_[ii]  = 0;
    optimizeDbleOptions_[ii] = 0.0;
    analysisIntOptions_[ii]  = 0;
    analysisDbleOptions_[ii] = 0.0;
  }
  /* analysis method (default is none) */
  analysisIntOptions_[0]  = 0;
  /* analysis output ID */
  analysisIntOptions_[1]  = 0;
  /* analysis RS type (default MARS) */
  analysisIntOptions_[2]  = 0;
  /* analysis output level */
  analysisIntOptions_[3]  = 0;
  /* analysis sample graphics flags */
  analysisIntOptions_[4]  = 0;
  /* analysis sample transformation flags */
  analysisIntOptions_[5]  = 0;
  /* analysis regression weight output id */
  analysisIntOptions_[6]  = -1;
  /* analysis threshold */
  analysisDbleOptions_[0] = 1.0;
  /* turn off optimization */
  optimizeIntOptions_[0]  = 0;
  /* optimization method - none */
  optimizeIntOptions_[1]  = 0;
  /* number of local minima to be stored */
  optimizeIntOptions_[2]  = 1;
  /* use RS for optimization (default : no) */
  optimizeIntOptions_[3]  = 0;
  /* optimization output level */
  optimizeIntOptions_[4]  = 0;
  /* num_fmin to be found */
  optimizeIntOptions_[5]  = 1;
  /* output ID to be analyzed */
  optimizeIntOptions_[6]  = 0;
  /* maximum number of function evaluation */
  optimizeIntOptions_[7]  = 10000;
  /* function mimumum */
  optimizeDbleOptions_[0] = 0.0;
  /* cut off point for optimization */
  optimizeDbleOptions_[1] = 1.0e50;
  /* optimization tolerance */
  optimizeDbleOptions_[2] = 1.0e-6;
  /* optimization deltaX */
  optimizeDbleOptions_[3] = 1.0e-6;
  fileWriteFlag_ = 0;
  strcpy(specsFile_, "NONE");
  strcpy(rsIndexFile_, "NONE");
  strcpy(rsIndexSampleFile_, "NONE");
  if (numRSFilters_ > 0)
  {
    for (int jj = 0; jj < numRSFilters_; jj++)
    {
//      if (RSFilters_[jj]->FilterDataFile_ != NULL)
//        delete [] RSFilters_[jj]->FilterDataFile_;
//      if (RSFilters_[jj]->FilterIndexFile_ != NULL)
//        delete [] RSFilters_[jj]->FilterIndexFile_;
      if (RSFilters_[jj] != NULL) delete RSFilters_[jj];
    }
  }
  if (numMOATFilters_ > 0)
  {
    for (int kk = 0; kk < numMOATFilters_; kk++)
    {
//      if (MOATFilters_[kk]->FilterDataFile_ != NULL)
//        delete [] MOATFilters_[kk]->FilterDataFile_;
//      if (MOATFilters_[kk]->FilterIndexFile_ != NULL)
//        delete [] MOATFilters_[kk]->FilterIndexFile_;
      if (MOATFilters_[kk] != NULL) delete MOATFilters_[kk];
    }
  }
}

// ************************************************************************
// equal operator
// ------------------------------------------------------------------------
PsuadeData& PsuadeData::operator=(const PsuadeData &)
{
  printf("PsuadeData operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}


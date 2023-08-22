// ************************************************************************
// Copyright (c) 2007   Lawrence Livermore National Security, LLC. 
// Produced at the Lawrence Livermore National Laboratory.
// Written by the Charles Tong <tong10@llnl.gov>.
//
// CODE UCRL-CODE-235523 All rights reserved.
//
// This file is part of PSUADE.
//
// Please see the GPL.pdf file for the copyright notice,
//
// This program is free software; you can redistribute it and/or modify it 
// under the terms of the GNU General Public License (as published by the 
// Free Software Foundation) version 2.1 dated February 1991.
//
// This program is distributed in the hope that it will be useful, but 
// WITHOUT ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY 
// or FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of the 
// GNU General Public License for more details.
//
// You should have received a copy of the GNU General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// Functions for the class PDFManager
// AUTHOR : CHARLES TONG
// DATE   : 2008
// ************************************************************************
#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include "PDFManager.h"
#include "PsuadeData.h"
#include "PsuadeUtil.h"
#include "sysdef.h"
#include "PDFNormal.h"
#include "PDFLogNormal.h"
#include "PDFMVNormal.h"
#include "PDFMVLogNormal.h"
#include "PDFTriangle.h"
#include "PDFBeta.h"
#include "PDFWeibull.h"
#include "PDFGamma.h"
#include "PDFInvGamma.h"
#include "PDFCauchy.h"
#include "PDFExponential.h"
#include "PDFSample.h"
#include "PDFF.h"
#include "PDFSampleHist.h"
#include "PDFUser.h"

#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// Constructor 
// ------------------------------------------------------------------------
PDFManager::PDFManager()
{
  nInputs_ = 0;
  //**/ indicate which PDFptrs each input is assigned to 
  //**/ (some inputs may belong to the same pointer)
  vecPdfMap_.clean(); 
  //**/ PDF pointers (if uniform, NULL)
  //**/ (pointers to different pdfs)
  PDFptrs_ = NULL;
  PDFMVNormalPtr_ = NULL;
  PDFMVLogNormalPtr_ = NULL;

  //**/ PsuadeData pointer given in the initialize function
  PSIOptr_ = NULL;

  //**/ store the input indices in group normal
  nGNormal_ = 0;
  vecgNormalInputs_.clean();
  //**/ store the input indices in group lognormal
  nGLognormal_ = 0;
  vecgLNormalInputs_.clean();
  //**/ user PDF information
  vecUsrPDFFlags_.clean();
  printLevel_ = 0;
}
   
// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PDFManager::~PDFManager()
{
  if (PDFptrs_ != NULL)
  {
    for (int ii = 0; ii < nInputs_; ii++)
      if (PDFptrs_[ii] != NULL) delete PDFptrs_[ii];
    delete [] PDFptrs_;
  }
  if (PDFMVNormalPtr_ != NULL) delete PDFMVNormalPtr_;
  if (PDFMVLogNormalPtr_ != NULL) delete PDFMVLogNormalPtr_;
  vecPdfMap_.clean(); 
  vecgNormalInputs_.clean();
  vecgLNormalInputs_.clean();
  vecUsrPDFFlags_.clean();
}

// ************************************************************************
// initialize function 
// ------------------------------------------------------------------------
int PDFManager::initialize(PsuadeData *psData)
{
  int     ii, *inputPDFs, count, samplingMethod=0, *indices;
  double  *inputMeans, *inputStdevs;
  char    **snames;
  pData   pPtr, pPDFs, pMeans, pStdevs, pCorMat, pSIndices, pPDFNames;
  psMatrix *corMatrix;

  //**/--------------------------------------------------------
  //**/ clean up
  //**/--------------------------------------------------------
  if (PDFptrs_ != NULL)
  {
    for (ii = 0; ii < nInputs_; ii++)
      if (PDFptrs_[ii] != NULL) delete PDFptrs_[ii];
    delete [] PDFptrs_;
  }
  if (PDFMVNormalPtr_ != NULL) delete PDFMVNormalPtr_;
  if (PDFMVLogNormalPtr_ != NULL) delete PDFMVLogNormalPtr_;
  vecPdfMap_.clean(); 
  vecgNormalInputs_.clean();
  vecgLNormalInputs_.clean();
  nInputs_ = nGNormal_ = nGLognormal_ = 0;
  PDFptrs_ = NULL;
  PDFMVNormalPtr_ = NULL;
  PDFMVLogNormalPtr_ = NULL;
  PSIOptr_ = NULL;

  //**/--------------------------------------------------------
  //**/ error checking
  //**/--------------------------------------------------------
  if (psData == NULL)
  {
     printf("PDFManager ERROR: no PsuadeData object given.\n");
     exit(1);
  } 

  //**/--------------------------------------------------------
  //**/ store away the pointer for the PsuadeData object
  //**/ fetch information from the PsuadeData object
  //**/--------------------------------------------------------
  PSIOptr_ = psData;
  psData->getParameter("ana_diagnostics", pPtr);
  printLevel_ = pPtr.intData_;
  psData->getParameter("input_ninputs", pPtr);
  nInputs_ = pPtr.intData_;
  psData->getParameter("input_pdfs", pPDFs);
  inputPDFs = pPDFs.intArray_;
  psData->getParameter("input_means", pMeans);
  inputMeans = pMeans.dbleArray_;
  psData->getParameter("input_stdevs", pStdevs);
  inputStdevs = pStdevs.dbleArray_; 
  psData->getParameter("input_sample_files", pPDFNames);
  snames = pPDFNames.strArray_;
  psData->getParameter("input_sample_indices", pSIndices);
  indices = pSIndices.intArray_;
  psData->getParameter("input_cor_matrix", pCorMat);
  //**/ since corMatrix is to be kept, null the pointer 
  corMatrix = (psMatrix *) pCorMat.psObject_; 
  pCorMat.psObject_ = NULL; 
  count = 0;
  for (ii = 0; ii < nInputs_; ii++) count += inputPDFs[ii];
  if (count > 0) 
  {
    printf("PDFManager INFO: some distributions are not UNIFORM.\n");
    psData->getParameter("method_sampling", pPtr);
    samplingMethod = pPtr.intData_;
    if (samplingMethod != PSUADE_SAMP_MC)  
    {
      //printf("INFO: if you want to draw a sample from non-uniform\n");
      //printf("distributions, set the sampling method to be Monte\n");
      //printf("Carlo (MC). If you have created a space-filling sample\n");
      //printf("and would like to convert it to the desired PDFs, add\n");
      //printf("the PDF information in the INPUT section of your sample\n");
      //printf("file, launch PSUADE in command line mode, load the\n");
      //printf("sample file, use 'pdfconvert' to convert your sample,\n");
      //printf("and write the converted sample to another PSUADE data\n");
      //printf("file.\n");
      printf("PDFManager INFO: non-uniform distribution detected.\n");
      printf("                 Switching to Monte Carlo.\n");
      samplingMethod = PSUADE_SAMP_MC;
    }
  }
  initialize(nInputs_,inputPDFs,inputMeans,inputStdevs,*corMatrix,
             snames,indices);
  return 0;
}

// ************************************************************************
// another initialize function 
// snames[ii] - if input ii is user sample, this points to the sample file
// indices[ii] - if input ii is user sample, this points to input number
//               in the sample file
// ------------------------------------------------------------------------
int PDFManager::initialize(int nInputs, int *inputPDFs, double *inputMeans, 
                           double *inputStdevs, psMatrix &corMatrix, 
                           char **snames, int *indices)
{
  int ii, jj, kk, scount, errFlag, normalFlag, lognormalFlag, corErrFlag;
  double mean, stdev, ddata;
  char   pString[1001];
  FILE   *fp;
  psMatrix  corMat;
  psVector  vecMeans;
  psIVector vecInds;

  //**/ ----------------------------------------------------------------
  //**/ copy data
  //**/ ----------------------------------------------------------------
  nInputs_ = nInputs;
  corMat_.load(corMatrix);
  if (corMatrix.nrows() != corMatrix.ncols())
  {
    printf("PDFManager ERROR: correlation matrix is not square.\n");
    exit(1);
  }
  if (corMatrix.nrows() != nInputs)
  {
    printf("PDFManager ERROR: correlation matrix dimension mismatch.\n"); 
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ error checking 
  //**/ ----------------------------------------------------------------
  errFlag = corErrFlag = 0;
  double corCheck=0.0;
  for (ii = 0; ii < nInputs_; ii++)
  {
    for (jj = 0; jj < nInputs_; jj++)
    {
      //*/ first check that only same distributions are allowed to 
      //*/ have correlation != 0
      if (inputPDFs[ii] != inputPDFs[jj] && 
          corMat_.getEntry(ii,jj) != 0.0)
      {
        printf("PDFManager ERROR: correlation != 0 for different PDFs.\n");
        errFlag = 1;
      }
      //*/ check that only normal and lognormal distributions are 
      //*/ allowed to have correlation != 0
      if (inputPDFs[ii] != PSUADE_PDF_LOGNORMAL && 
          inputPDFs[ii] != PSUADE_PDF_NORMAL && ii != jj && 
          corMat_.getEntry(ii,jj) != 0.0)
      {
        printf("PDFManager ERROR: correlation != 0 for non-normal PDFs\n");
        errFlag = 1;
      }
      //*/ check that correlation cannot be less than -1 or > 1
      if (corMat_.getEntry(ii,jj) < -1 || corMat_.getEntry(ii,jj) > 1)
      {
        printf("PDFManager ERROR: correlation should be in [-1,1].\n");
        errFlag = 1;
      }
      //*/ check that covariance matrix is symmetric
      ddata = PABS(corMat_.getEntry(ii,jj)-corMat_.getEntry(jj,ii));
      corCheck += PABS(corMat_.getEntry(ii,jj)-corMat_.getEntry(jj,ii));
      if (jj > ii && ddata > 1.0e-12)
      {
        printf("Correlation matrix (%d,%d) != (%d,%d) ? %e %e\n",
               ii+1,jj+1,jj+1,ii+1,corMat_.getEntry(ii,jj),
               corMat_.getEntry(jj,ii));
        ddata = 0.5*(corMat_.getEntry(ii,jj)+corMat_.getEntry(jj,ii));
        corMat_.setEntry(ii,jj,ddata);
        corMat_.setEntry(jj,ii,ddata);
        corErrFlag++;
      }
    }
  }
  if (corErrFlag > 0)
  {
    printf("PDFManager INFO: Correlation matrix not symmetric (%d).\n",
           corErrFlag);
    printf("                 It has been symmetrized.\n");
    printf("norm(difference) = %e\n", corCheck);
  }
  vecUsrPDFFlags_.setLength(nInputs_);

  for (ii = 0; ii < nInputs_; ii++)
  {
    vecUsrPDFFlags_[ii] = -1;
    if (inputPDFs[ii] == PSUADE_PDF_SAMPLE)
    {
      if (snames == NULL)
      {
        printf("PDFManager ERROR: no sample file provided.\n"); 
        exit(1);
      }
      for (jj = 0; jj < ii; jj++)
      {
        if (!strcmp(snames[ii],snames[jj]))
        {
          vecUsrPDFFlags_[ii] = jj;
          break;
        }
      }
      if (vecUsrPDFFlags_[ii] == -1) vecUsrPDFFlags_[ii] = ii;
    }
    if (inputPDFs[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      if (snames == NULL)
      {
        printf("PDFManager ERROR: no sample file provided.\n"); 
        exit(1);
      }
      for (jj = 0; jj < ii; jj++)
      {
        if (!strcmp(snames[ii],snames[jj]))
        {
          vecUsrPDFFlags_[ii] = jj;
          break;
        }
      }
      if (vecUsrPDFFlags_[ii] == -1) vecUsrPDFFlags_[ii] = ii;
    }
  }
  assert(errFlag == 0); 

  //**/ ----------------------------------------------------------------
  //**/ searching for groupings (even if covariance = 0)
  //**/ ----------------------------------------------------------------
  nGNormal_ = nGLognormal_ = 0;
  vecgNormalInputs_.setLength(nInputs_);
  vecgLNormalInputs_.setLength(nInputs_);
  vecPdfMap_.setLength(nInputs_);
  normalFlag = lognormalFlag = 0;
  for (ii = 0; ii < nInputs_; ii++)
  {
    vecPdfMap_[ii] = inputPDFs[ii];
    if (inputPDFs[ii] == PSUADE_PDF_NORMAL)
    {
      for (jj = 0; jj < nInputs_; jj++)
        if (ii != jj && corMat_.getEntry(ii,jj) != 0.0) break;
      if (jj < nInputs_)
      {
        vecPdfMap_[ii] = 1000 + PSUADE_PDF_NORMAL;
        vecgNormalInputs_[nGNormal_++] = ii;
      }
    }
    if (inputPDFs[ii] == PSUADE_PDF_LOGNORMAL)
    {
      for (jj = 0; jj < nInputs_; jj++)
        if (ii != jj && corMat_.getEntry(ii,jj) != 0.0) break;
      if (jj < nInputs_)
      {
        vecPdfMap_[ii] = 1000 + PSUADE_PDF_LOGNORMAL;
        vecgLNormalInputs_[nGLognormal_++] = ii;
      }
    }
  } 

  //**/ ----------------------------------------------------------------
  //**/ inject standard deviations into the correlation matrix
  //**/ ----------------------------------------------------------------
  if (corMat_.nrows() != nInputs) corMat_.setDim(nInputs, nInputs);
  for (ii = 0; ii < nInputs_; ii++)
  {
    stdev = inputStdevs[ii]; 
    corMat_.setEntry(ii, ii, stdev);
  }

  //**/ ----------------------------------------------------------------
  //**/ construct the corresponding PDF objects
  //**/ PDFptrs[nInputs] and PDFptrs[nInputs+1] reserved for group
  //**/ normal and group lognormal
  //**/ ----------------------------------------------------------------
  scount = 0;
  PDFptrs_ = new PDFBase*[nInputs_];
  normalFlag = lognormalFlag = 0;
  for (ii = 0; ii < nInputs_; ii++) PDFptrs_[ii] = NULL;
  for (ii = 0; ii < nInputs_; ii++)
  {
    mean = inputMeans[ii]; 
    stdev = inputStdevs[ii]; 
    if (vecPdfMap_[ii] == PSUADE_PDF_NORMAL)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFNormal(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_LOGNORMAL)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFLogNormal(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_TRIANGLE)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFTriangle(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_BETA)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFBeta(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_WEIBULL)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFWeibull(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_GAMMA)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFGamma(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_INVGAMMA)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFInvGamma(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_CAUCHY)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFCauchy(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_EXPONENTIAL)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFExponential(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLE)
    {
      if (vecUsrPDFFlags_[ii] == ii)
      {
        if (indices == NULL)
        {
          printf("PDFManager ERROR: null index array.\n"); 
          exit(1);
        }
        //**/ compress sample input indices
        vecInds.setLength(nInputs_);
        scount = 0;
        for (jj = 0; jj < nInputs_; jj++)
          if (vecUsrPDFFlags_[jj] == ii) vecInds[scount++] = indices[jj];
        //**/ instantiate 
        PDFptrs_[ii] = 
           (PDFBase *) new PDFSample(scount,snames[ii],vecInds.getIVector());
        //**/ error checking 
        fp = fopen(snames[ii], "r");
        if (fp == NULL)
        {
          printf("PDFManager ERROR: cannot open sample file %s.\n",
                 snames[ii]);
          exit(1);
        }
        fscanf(fp, "%s", pString);
        if (strcmp(pString, "PSUADE_BEGIN"))
        {
          fclose(fp);
          fp = fopen(snames[ii], "r");
        }
        fscanf(fp, "%d %d", &jj, &kk);
        if (kk < scount)
        {
          printf("PDFManager ERROR: ");
          printf("sample file %s does not have enough inputs.\n",
                 snames[ii]);
          printf("                  sample file has %d inputs\n",kk);
          printf("                  PDFManager needs %d inputs\n",scount);
          exit(1);
        }
        fclose(fp);
      }
      else PDFptrs_[ii] = NULL;
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_F)
    {
      PDFptrs_[ii] = (PDFBase *) new PDFF(mean, stdev);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      if (vecUsrPDFFlags_[ii] == ii)
      {
        //**/ compress sample input indices
        vecInds.setLength(nInputs_);
        scount = 0;
        for (jj = 0; jj < nInputs_; jj++)
          if (vecUsrPDFFlags_[jj] == ii) vecInds[scount++] = indices[jj];
        //**/ instantiate 
        PDFptrs_[ii] = 
           (PDFBase *) new PDFSampleHist(snames[ii],scount,
                                         vecInds.getIVector());
        //**/ error checking 
        fp = fopen(snames[ii], "r");
        if (fp == NULL)
        {
          printf("PDFManager ERROR: cannot open sample file %s.\n",
                 snames[ii]);
          exit(1);
        }
        fscanf(fp, "%s", pString);
        if (strcmp(pString, "PSUADE_BEGIN"))
        {
           fclose(fp);
           fp = fopen(snames[ii], "r");
        }
        fscanf(fp, "%d %d", &jj, &kk);
        if (kk < scount)
        {
          printf("PDFManager ERROR: ");
          printf("sample file %s does not have enough inputs.\n",
                 snames[ii]);
          printf("                  sample file has %d inputs\n",kk);
          printf("                  PDFManager needs %d inputs\n",scount);
          exit(1);
        }
        fclose(fp);
      }
      else PDFptrs_[ii] = NULL;
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_NORMAL)
    {
      if (normalFlag == 0)
      {
        corMat.submatrix(corMat_,nGNormal_,vecgNormalInputs_.getIVector());
        vecMeans.setLength(nGNormal_);
        for (jj = 0; jj < nGNormal_; jj++)
          vecMeans[jj] = inputMeans[vecgNormalInputs_[jj]];
        PDFMVNormalPtr_ = new PDFMVNormal(vecMeans, corMat);
        normalFlag = 1;
      }
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_LOGNORMAL)
    {
      if (lognormalFlag == 0)
      {
        corMat.submatrix(corMat_, nGLognormal_, 
                         vecgLNormalInputs_.getIVector());
        vecMeans.setLength(nGLognormal_);
        for (jj = 0; jj < nGLognormal_; jj++)
          vecMeans[jj] = inputMeans[vecgLNormalInputs_[jj]];
        PDFMVLogNormalPtr_ = new PDFMVLogNormal(vecMeans, corMat);
        lognormalFlag = 1;
      }
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_USER)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        if (vecPdfMap_[ii] != PSUADE_PDF_USER)
        {
          printf("PDFManager ERROR: if U(ser) PDF is selected,\n");
          printf("           it must be used on all inputs.\n");
          exit(1);
        }
      }
      PDFptrs_[0] = (PDFBase *) new PDFUser(nInputs_);
      break;
    }
  }
  return 0;
}

// ************************************************************************
// forward transformation 
// ------------------------------------------------------------------------
int PDFManager::getPDF(int nSamples, psVector &vecIn, psVector &vecOut,
                       psVector &vecLower, psVector &vecUpper)
{
  int  ss, ii, jj, nInputs;
  char cString[1001];
  psVector vecLocalIn, vecLocalOut, vecData1, vecData2;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (nInputs_ <= 0)
  {
    printf("PDFManager evaluate ERROR: nInputs <= 0.\n");
    exit(1);
  }
  nInputs = vecLower.length();
  if (nInputs_ != nInputs)
  {
    printf("PDFManager evaluate ERROR: nInputs mismatch.\n");
    exit(1);
  }
  if (nInputs_ != vecUpper.length() || nInputs_ != vecLower.length())
  {
    printf("PDFManager evaluate ERROR: vecBounds length mismatch.\n");
    exit(1);
  }
  if (nSamples > vecIn.length()/nInputs)
  {
    printf("PDFManager evaluate ERROR: vecIn length mismatch.\n");
    exit(1);
  }
  if (nSamples > vecOut.length()/nInputs)
  {
    printf("PDFManager evaluate ERROR: vecOut length mismatch.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ no transformation
  //**/ ----------------------------------------------------------------
  if (PDFptrs_ == NULL)
  {
    for (ii = 0; ii < nSamples; ii++)
      for (jj = 0; jj < nInputs_; jj++)
        vecOut[ii*nInputs_+jj] = 1.0 / (vecUpper[jj] - vecLower[jj]);
    return 0;
  }

  //**/ ----------------------------------------------------------------
  //**/ perform transformation
  //**/ ----------------------------------------------------------------
  vecData1.setLength(nSamples*nInputs_);
  vecData2.setLength(nSamples*nInputs_);
  for (ii = 0; ii < nSamples*nInputs_; ii++) vecData2[ii] = 0.0;
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (vecPdfMap_[ii] == PSUADE_PDF_UNIFORM)
    {
      if (printLevel_ > 2) printf("Input %d: uniform distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = 1.0 / (vecUpper[ii] - vecLower[ii]);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_NORMAL)
    {
      if (printLevel_ > 2) printf("Input %d: normal distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_LOGNORMAL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: lognormal distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_TRIANGLE)
    {
      if (printLevel_ > 2) 
        printf("Input %d: triangle distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_BETA)
    {
      if (printLevel_ > 2) 
        printf("Input %d: beta distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_WEIBULL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: Weibull distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_GAMMA)
    {
      if (printLevel_ > 2) 
        printf("Input %d: gamma distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_INVGAMMA)
    {
      if (printLevel_ > 2) 
        printf("Input %d: inverse gamma distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
         vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_CAUCHY)
    {
      if (printLevel_ > 2) 
        printf("Input %d: inverse Cauchy distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_EXPONENTIAL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: exponential distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLE)
    {
      if (printLevel_ > 2) printf("Input %d: S distribution.\n",ii+1);
      jj = 0;
      for (ss = 0; ss < ii; ss++)
        if (vecUsrPDFFlags_[ii] == vecUsrPDFFlags_[ss]) jj++;
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      snprintf(cString,100,"setInput %d", jj);
      PDFptrs_[ii]->setParam(cString);
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_F)
    {
      if (printLevel_ > 2) printf("Input %d: F distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      if (printLevel_ > 2) printf("Input %d: S2 distribution.\n",ii+1);
      jj = 0;
      for (ss = 0; ss < ii; ss++)
        if (vecUsrPDFFlags_[ii] == vecUsrPDFFlags_[ss]) jj++;
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs+ii];
      snprintf(cString,100,"setInput %d", jj);
      PDFptrs_[ii]->setParam(cString);
      PDFptrs_[ii]->getPDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_NORMAL)
    {
      printf("ERROR: getPDF for multivariate normal not available.\n");
      exit(1);
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_LOGNORMAL)
    {
      printf("ERROR: getPDF for multivariate lognormal not available.\n");
      exit(1);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_USER)
    {
      printf("ERROR: getPDF for user distribution not available.\n");
      exit(1);
    }
  }
  return 0;
}

// ************************************************************************
// generate sample
// ------------------------------------------------------------------------
int PDFManager::genSample(int nSamples, psVector &vecOut, 
                          psVector &vecLower, psVector &vecUpper)
{
  int    ss, ii, jj, normalFlag, lognormalFlag, ind, nInputs;
  int    scount;
  double range, low, high;
  psVector  vecLocalOut, vLower, vUpper, vecData;
  psIVector vecCounts;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (nInputs_ <= 0)
  {
    printf("PDFManager genSample ERROR: nInputs <= 0.\n");
    exit(1);
  }
  nInputs = vecLower.length();
  if (nInputs_ != nInputs)
  {
    printf("PDFManager genSample ERROR: nInputs mismatch.\n");
    exit(1);
  }
  if (nInputs_ != vecUpper.length() || nInputs_ != vecLower.length())
  {
    printf("PDFManager genSample ERROR: vecBounds length mismatch.\n");
    exit(1);
  }
  if (nSamples > vecOut.length()/nInputs_)
  {
    printf("PDFManager genSample ERROR: vecOut length mismatch (%d vs %d).\n",
           nSamples, vecOut.length()/nInputs);
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ generate sample
  //**/ ----------------------------------------------------------------
  scount = 0;
  vecCounts.setLength(nInputs_);
  normalFlag = lognormalFlag = 0;
  vecData.setLength(nSamples*nInputs);
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (vecPdfMap_[ii] == PSUADE_PDF_UNIFORM)
    {
      //**/ uniform, just copy
      range = vecUpper[ii] - vecLower[ii];
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = PSUADE_drand()*range+vecLower[ii];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_TRIANGLE ||
        vecPdfMap_[ii] == PSUADE_PDF_BETA ||
        vecPdfMap_[ii] == PSUADE_PDF_WEIBULL || 
        vecPdfMap_[ii] == PSUADE_PDF_NORMAL ||
        vecPdfMap_[ii] == PSUADE_PDF_LOGNORMAL ||
        vecPdfMap_[ii] == PSUADE_PDF_GAMMA ||
        vecPdfMap_[ii] == PSUADE_PDF_INVGAMMA ||
        vecPdfMap_[ii] == PSUADE_PDF_CAUCHY ||
        vecPdfMap_[ii] == PSUADE_PDF_EXPONENTIAL ||
        vecPdfMap_[ii] == PSUADE_PDF_F)
    {
      low = vecLower[ii];
      high = vecUpper[ii];
      PDFptrs_[ii]->genSample(nSamples,vecData.getDVector(),&low,&high);
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData[ss];
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_NORMAL)
    {
      if (normalFlag == 0)
      {
        vLower.setLength(nGNormal_);
        vUpper.setLength(nGNormal_);
        vecLocalOut.setLength(nSamples*nGNormal_);
        for (jj = 0; jj < nGNormal_; jj++)
        {
          ind = vecgNormalInputs_[jj];
          vLower[jj] = vecLower[ind];
          vUpper[jj] = vecUpper[ind];
        }
        PDFMVNormalPtr_->genSample(nSamples,vecLocalOut,vLower,vUpper);
        for (jj = 0; jj < nGNormal_; jj++)
        {
          ind = vecgNormalInputs_[jj];
          for (ss = 0; ss < nSamples; ss++)
          {
            vecOut[ss*nInputs+ind] = vecLocalOut[ss*nGNormal_+jj];
            if (vecOut[ss*nInputs_+ind] > vecUpper[ind])
              vecOut[ss*nInputs_+ind] = vecUpper[ind];
            if (vecOut[ss*nInputs_+ind] < vecLower[ind])
              vecOut[ss*nInputs_+ind] = vecLower[ind];
          }
        }
        normalFlag = 1;
      }
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_LOGNORMAL)
    {
      if (lognormalFlag == 0)
      {
        vLower.setLength(nGLognormal_);
        vUpper.setLength(nGLognormal_);
        vecLocalOut.setLength(nSamples*nGLognormal_);
        for (jj = 0; jj < nGLognormal_; jj++)
        {
          ind = vecgLNormalInputs_[jj];
          vLower[jj] = vecLower[ind];
          vUpper[jj] = vecUpper[ind];
        }
        PDFMVLogNormalPtr_->genSample(nSamples,vecLocalOut,vLower,vUpper);
        for (jj = 0; jj < nGLognormal_; jj++)
        {
          ind = vecgLNormalInputs_[jj];
          for (ss = 0; ss < nSamples; ss++)
          {
            vecOut[ss*nInputs+ind] = vecLocalOut[ss*nGLognormal_+jj];
            if (vecOut[ss*nInputs_+ind] > vecUpper[ind])
              vecOut[ss*nInputs_+ind] = vecUpper[ind];
            if (vecOut[ss*nInputs_+ind] < vecLower[ind])
              vecOut[ss*nInputs_+ind] = vecLower[ind];
          }
        }
        lognormalFlag = 1;
      }
    }
  }
   
  //**/ ----------------------------------------------------------------
  //**/ special processing for user-generated sample
  //**/ ----------------------------------------------------------------
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLE)
    {
      if (vecUsrPDFFlags_[ii] == ii)
      {
        vecCounts[0] = ii;
        scount = 1;
        for (jj = ii+1; jj < nInputs_; jj++)
          if (vecUsrPDFFlags_[jj] == ii) vecCounts[scount++] = jj;
        PDFptrs_[ii]->genSample(nSamples,vecData.getDVector(),NULL,NULL);
        for (jj = 0; jj < scount; jj++)
        {
          ind = vecCounts[jj];
          for (ss = 0; ss < nSamples; ss++)
            vecOut[ss*nInputs_+ind] = vecData[scount*ss+jj];
        }
      }
    }
  }
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      if (vecUsrPDFFlags_[ii] == ii)
      {
        vecCounts[0] = ii;
        scount = 1;
        for (jj = ii+1; jj < nInputs_; jj++)
          if (vecUsrPDFFlags_[jj] == ii) vecCounts[scount++] = jj;
        PDFptrs_[ii]->genSample(nSamples,vecData.getDVector(),NULL,NULL);
        for (jj = 0; jj < scount; jj++)
        {
          ind = vecCounts[jj];
          for (ss = 0; ss < nSamples; ss++)
            vecOut[ss*nInputs_+ind] = vecData[scount*ss+jj];
        }
      }
    }
  }
  //**/ for user type, get sample for all inputs
  if (vecPdfMap_[0] == PSUADE_PDF_USER)
  {
    PDFptrs_[0]->genSample(nSamples,vecOut.getDVector(),NULL,NULL);
  }
  return 0;
}

// ************************************************************************
// generate sample (and put it back to the PsuadeData object)
// ------------------------------------------------------------------------
int PDFManager::genSample()
{
  int       ss, ii;
  pData     pPtr, pLowerB, pUpperB;
  psVector  vecLocalIn,vecLocalOut,vecUpper,vecLower,vecSamIns,vecSamOuts;
  psIVector vecSamStates;

  //**/--------------------------------------------------------
  //**/ error checking
  //**/--------------------------------------------------------
  if (printLevel_ > 2) printf("PDFManager genSample\n");
  if (PSIOptr_ == NULL)
  {
    printf("PDFManager ERROR: no PsuadeData object given.\n");
    exit(1);
  } 

  //**/--------------------------------------------------------
  //**/ if no PDFs defined, do nothing
  //**/--------------------------------------------------------
  if (PDFptrs_ == NULL) 
  {
    printf("PDFManager ERROR: no PDF initialization yet.\n");
    return 0;
  }

  //**/--------------------------------------------------------
  //**/ store away the pointer for the PsuadeData object
  //**/ fetch information from the PsuadeData object
  //**/--------------------------------------------------------
  PSIOptr_->getParameter("method_nsamples", pPtr);
  int nSamples = pPtr.intData_;
  PSIOptr_->getParameter("input_ninputs", pPtr);
  int nInputs = pPtr.intData_;
  PSIOptr_->getParameter("output_noutputs", pPtr);
  int nOutputs = pPtr.intData_;
  PSIOptr_->getParameter("input_lbounds", pLowerB);
  double *iLowerB = pLowerB.dbleArray_;
  PSIOptr_->getParameter("input_ubounds", pUpperB);
  double *iUpperB = pUpperB.dbleArray_;

  vecSamIns.setLength(nSamples*nInputs);
  vecSamOuts.setLength(nSamples*nOutputs);
  vecSamStates.setLength(nSamples);

  vecUpper.load(nInputs_, iUpperB);
  vecLower.load(nInputs_, iLowerB);
  vecLocalOut.setLength(nSamples*nInputs_);
  genSample(nSamples, vecLocalOut, vecLower, vecUpper);
  for (ss = 0; ss < nSamples; ss++)
  {
    for (ii = 0; ii < nInputs; ii++)
      vecSamIns[ss*nInputs+ii] = vecLocalOut[ss*nInputs+ii];
    for (ii = 0; ii < nOutputs; ii++)
      vecSamOuts[ss*nOutputs+ii] = PSUADE_UNDEFINED;
    vecSamStates[ss] = 0;
  }
  PSIOptr_->updateInputSection(nSamples,nInputs_,NULL,NULL,NULL,
                  vecSamIns.getDVector(),NULL,NULL,NULL,NULL,NULL);
  PSIOptr_->updateOutputSection(nSamples,nOutputs,vecSamOuts.getDVector(),
                                vecSamStates.getIVector(),NULL);
  return 0;
}

// ************************************************************************
// look up the cumulative density function
// ------------------------------------------------------------------------
int PDFManager::invCDF(int nSamples, psVector &vecIn, psVector &vecOut, 
                       psVector &vecLower, psVector &vecUpper)
{
  int  ss, ii, jj, nInputs;
  char cString[1001];
  psVector vecLocalIn, vecLocalOut, vecData1, vecData2;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (nInputs_ <= 0)
  {
    printf("PDFManager invCDF ERROR: nInputs <= 0.\n");
    exit(1);
  }
  nInputs = vecLower.length();
  if (nInputs_ != nInputs)
  {
    printf("PDFManager invCDF ERROR: nInputs mismatch.\n");
    exit(1);
  }
  if (nInputs_ != vecUpper.length() || nInputs_ != vecLower.length())
  {
    printf("PDFManager invCDF ERROR: vecBounds length mismatch.\n");
    exit(1);
  }
  if (nSamples > vecIn.length()/nInputs_)
  {
    printf("PDFManager invCDF ERROR: vecIn length mismatch.\n");
    exit(1);
  }
  if (nSamples != vecOut.length()/nInputs_)
  {
    printf("PDFManager invCDF ERROR: vecOut length mismatch.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ no transformation
  //**/ ----------------------------------------------------------------
  if (PDFptrs_ == NULL)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecIn[ss*nInputs_+ii];
    }
    return 0;
  }

  //**/ ----------------------------------------------------------------
  //**/ perform transformation
  //**/ ----------------------------------------------------------------
  vecData1.setLength(nSamples*nInputs);
  vecData2.setLength(nSamples*nInputs);
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (vecPdfMap_[ii] == PSUADE_PDF_UNIFORM)
    {
      if (printLevel_ > 2) 
        printf("Input %d: uniform distribution.\n",ii+1);
      //**/ uniform, just copy
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = vecIn[ss*nInputs+ii];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_NORMAL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: normal distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_LOGNORMAL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: lognormal distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_TRIANGLE)
    {
      if (printLevel_ > 2) 
        printf("Input %d: triangle distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_BETA)
    {
      if (printLevel_ > 2) 
        printf("Input %d: beta distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_WEIBULL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: Weibull distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_GAMMA)
    {
      if (printLevel_ > 2) 
        printf("Input %d: gamma distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_INVGAMMA)
    {
      if (printLevel_ > 2) 
        printf("Input %d: inverse gamma distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_CAUCHY)
    {
      if (printLevel_ > 2) 
        printf("Input %d: inverse Cauchy distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_EXPONENTIAL)
    {
      if (printLevel_ > 2) 
        printf("Input %d: exponential distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLE)
    {
      if (printLevel_ > 2) 
        printf("Input %d: S distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      jj = 0;
      for (ss = 0; ss < ii; ss++)
        if (vecUsrPDFFlags_[ii] == vecUsrPDFFlags_[ss]) jj++;
      snprintf(cString,100,"setInput %d", jj);
      PDFptrs_[ii]->setParam(cString);
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_F)
    {
      if (printLevel_ > 2) printf("Input %d: F distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      if (printLevel_ > 2) printf("Input %d: S2 distribution.\n",ii+1);
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      jj = 0;
      for (ss = 0; ss < ii; ss++)
        if (vecUsrPDFFlags_[ii] == vecUsrPDFFlags_[ss]) jj++;
      snprintf(cString,100,"setInput %d", jj);
      PDFptrs_[ii]->setParam(cString);
      PDFptrs_[ii]->invCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_NORMAL)
    {
      printf("invCDF for multivariate normal is not available.\n");
      exit(1);
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_LOGNORMAL)
    {
      printf("invCDF for multivariate lognormal is not available.\n");
      exit(1);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_USER)
    {
      for (jj = 0; jj < nInputs_; jj++)
      {
        if (vecPdfMap_[ii] != PSUADE_PDF_USER)
        {
          printf("PDFManager ERROR: if U(ser) PDF is selected,\n");
          printf("           it must be used on all inputs.\n");
          exit(1);
        }
        PDFptrs_[0]->invCDF(nSamples,vecData1.getDVector(),
                            vecData2.getDVector());
        break;
      }
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_USER)
    {
      printf("invCDF for user distribution is not available.\n");
      exit(1);
    }
    //**/ clipping with lower and upper bounds
    for (ss = 0; ss < nSamples; ss++)
    {
      if (vecOut[ss*nInputs_+ii] < vecLower[ii])
        vecOut[ss*nInputs_+ii] = vecLower[ii];
      else if (vecOut[ss*nInputs_+ii] > vecUpper[ii])
        vecOut[ss*nInputs_+ii] = vecUpper[ii];
    }
  }
  return 0;
}

// ************************************************************************
// look up the cumulative density function
// ------------------------------------------------------------------------
int PDFManager::getCDF(int nSamples, psVector &vecIn, psVector &vecOut, 
                       psVector &vecLower, psVector &vecUpper)
{
  int    ss, ii, jj, nInputs;
  double range;
  char   cString[1001];
  psVector vecLocalIn, vecLocalOut, vecData1, vecData2;

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (nInputs_ <= 0)
  {
    printf("PDFManager getCDF ERROR: nInputs <= 0.\n");
    exit(1);
  }
  nInputs = vecLower.length();
  if (nInputs_ != nInputs)
  {
    printf("PDFManager getCDF ERROR: nInputs mismatch.\n");
    exit(1);
  }
  if (nSamples != vecIn.length()/nInputs_)
  {
    printf("PDFManager getCDF ERROR: vecIn length mismatch.\n");
    exit(1);
  }
  if (nSamples > vecOut.length()/nInputs_)
  {
    printf("PDFManager getCDF ERROR: vecOut length mismatch.\n");
    exit(1);
  }

  //**/ ----------------------------------------------------------------
  //**/ no transformation
  //**/ ----------------------------------------------------------------
  if (PDFptrs_ == NULL)
  {
    for (ii = 0; ii < nInputs_; ii++)
    {
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecIn[ss*nInputs_+ii];
    }
    return 0;
  }

  //**/ ----------------------------------------------------------------
  //**/ perform transformation
  //**/ ----------------------------------------------------------------
  vecData1.setLength(nSamples*nInputs);
  vecData2.setLength(nSamples*nInputs);
  for (ii = 0; ii < nInputs_; ii++)
  {
    if (vecPdfMap_[ii] == PSUADE_PDF_UNIFORM)
    {
      //**/ uniform, just copy
      range = vecUpper[ii] - vecLower[ii];
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs+ii] = (vecIn[ss*nInputs+ii]-vecLower[ii])/range;
    }
    if (vecPdfMap_[ii] == PSUADE_PDF_TRIANGLE ||
        vecPdfMap_[ii] == PSUADE_PDF_BETA ||
        vecPdfMap_[ii] == PSUADE_PDF_WEIBULL ||
        vecPdfMap_[ii] == PSUADE_PDF_NORMAL ||
        vecPdfMap_[ii] == PSUADE_PDF_LOGNORMAL ||
        vecPdfMap_[ii] == PSUADE_PDF_GAMMA ||
        vecPdfMap_[ii] == PSUADE_PDF_INVGAMMA ||
        vecPdfMap_[ii] == PSUADE_PDF_CAUCHY ||
        vecPdfMap_[ii] == PSUADE_PDF_EXPONENTIAL ||
        vecPdfMap_[ii] == PSUADE_PDF_F)
    {
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      PDFptrs_[ii]->getCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLE)
    {
      jj = 0;
      for (ss = 0; ss < ii; ss++)
        if (vecUsrPDFFlags_[ii] == vecUsrPDFFlags_[ss]) jj++;
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      snprintf(cString,100,"setInput %d", jj);
      PDFptrs_[ii]->setParam(cString);
      PDFptrs_[ii]->getCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_SAMPLEHIST)
    {
      jj = 0;
      for (ss = 0; ss < ii; ss++)
        if (vecUsrPDFFlags_[ii] == vecUsrPDFFlags_[ss]) jj++;
      for (ss = 0; ss < nSamples; ss++)
        vecData1[ss] = vecIn[ss*nInputs_+ii];
      snprintf(cString,100,"setInput %d", jj);
      PDFptrs_[ii]->setParam(cString);
      PDFptrs_[ii]->getCDF(nSamples,vecData1.getDVector(),
                           vecData2.getDVector());
      for (ss = 0; ss < nSamples; ss++)
        vecOut[ss*nInputs_+ii] = vecData2[ss];
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_NORMAL)
    {
      printf("getCDF for multivariate normal is not available.\n");
      exit(1);
    }
    else if (vecPdfMap_[ii] == 1000+PSUADE_PDF_LOGNORMAL)
    {
      printf("getCDF for multivariate lognormal is not available.\n");
      exit(1);
    }
    else if (vecPdfMap_[ii] == PSUADE_PDF_USER)
    {
      printf("getCDF for user distribution is not available.\n");
      exit(1);
    }
  }
  return 0;
}


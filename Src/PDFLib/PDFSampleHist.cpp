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
// Functions for the user-defined distribution
// AUTHOR : CHARLES TONG
// DATE   : 2015
// ************************************************************************
#include <stdio.h>
#include <math.h>
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PDFSampleHist.h"
#include "PrintingTS.h"
#define PABS(x) (((x) >= 0) ? x : -(x))
#ifdef HAVE_METIS
extern "C" {
void METIS_PartGraphRecursive(int *, int *, int *, int *, int *,
                              int *, int *, int *, int *, int *, int *);
}
#endif

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PDFSampleHist::PDFSampleHist(char *fname, int scount, int *indices)
{
  int    ii, jj, nn, nInps;
  double ddata, dmin, dmax;
  char   pString[1001], filename[1001];
  FILE   *fp=NULL;
  psVector  vecOneS;
  psIVector vecIncr, vecGraphI, vecGraphJ;
 
  //**/ -------------------------------------------------------------
  //**/ check sample file
  //**/ -------------------------------------------------------------
  if (fname == NULL || !strcmp(fname, "NONE"))
  {
    printf("PDFSampleHist: Expecting a sample file having the ");
    printf("following format: \n");
    printf("line 1: (optional) PSUADE_BEGIN\n");
    printf("line 2: <number of sample points> <number of inputs>\n");
    printf("line 3: (optional) : '#' followed by input names\n");
    printf("line 4: 1 sample point 1 inputs \n");
    printf("line 5: 2 sample point 2 inputs \n");
    printf("line 6: 3 sample point 3 inputs \n");
    printf("...\n");
    printf("line n: (optional) PSUADE_END\n");
    snprintf(pString,100,"Enter name of sample file : ");
    getString(pString, filename);
    nn = strlen(filename);
    if (nn > 1000)
    {
      printf("PDFSampleHist ERROR: file name too long.\n");
      exit(1);
    }
    filename[nn-1] = '\0';
  }
  else strcpy(filename, fname);

  //**/ -------------------------------------------------------------
  //**/ read in sample
  //**/ -------------------------------------------------------------
  nSamples_ = 0;
  vecSamples_.clean();
  nInputs_ = scount;
  fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("PDFSampleHist ERROR: Cannot open sample file %s.\n",
           filename);
    exit(1);
  }
  fscanf(fp, "%s", pString);
  if (strcmp(pString, "PSUADE_BEGIN"))
  {
    fclose(fp);
    fp = fopen(filename, "r");
  } 
  fscanf(fp, "%d %d", &nSamples_, &nInps);
  if (nSamples_ < 100000)
  {
    printf("PDFSampleHist ERROR: Sample file has nSamples < 100000.\n");
    exit(1);
  }
  if (nInps < 1 || nInps > 10)
  {
    printf("PDFSampleHist ERROR: Sample file has nInputs <= 0 or > 10.\n");
    exit(1);
  }
  //**/ if sample file has different nInputs than the declared nInputs, 
  //**/ and no incoming index list is given, flag an error
  if (nInputs_ != nInps && indices == NULL)
  {
    printf("PDFSampleHist ERROR: nInputs does not match.\n");
    printf("          nInputs in your sample file    = %d\n",nInps);
    printf("          nInputs from psuade input file = %d\n",nInputs_);
    exit(1);
  }
  if (indices != NULL)
  {
    for (ii = 0; ii < scount; ii++)
    {
      if (indices[ii] < 0 || indices[ii] >= nInps)
      {
        printf("PDFSampleHist ERROR: Sample index > nInputs.\n");
        printf("              sample index requested         = %d\n",
               indices[ii]+1);
        printf("              nInputs in your sample file    = %d\n",
               nInps);
        exit(1);
      } 
    }
  }
  //**/ detect comment line
  fgets(pString, 1000, fp);
  while (1)
  {
    nn = getc(fp);
    if (nn == '#') fgets(pString, 1000, fp);
    else
    {
      ungetc(nn, fp);
      break;
    }
  }
  vecSamples_.setLength(nSamples_*nInputs_);
  vecOneS.setLength(nInps);
  for (ii = 0; ii < nSamples_; ii++)
  {
    fscanf(fp, "%d", &nn);
    if (nn != (ii+1))
    {
      printf("PDFSampleHist ERROR: invalid sample number.\n");
      printf("              Expected: %d\n", ii+1);
      printf("              Read:     %d\n", nn);
      printf("Advice: check your data format and line number %d.\n\n",
             ii+2);
      printf("Correct Format: \n");
      printf("line 1: (optional) PSUADE_BEGIN\n");
      printf("line 2: <number of sample points> <number of inputs>\n");
      printf("line 3: (optional) : '#' followed by input names\n");
      printf("line 4: 1 sample point 1 inputs \n");
      printf("line 5: 2 sample point 2 inputs \n");
      printf("line 6: 3 sample point 3 inputs \n");
      printf("...\n");
      printf("line n: (optional) PSUADE_END\n");
      fclose(fp);
      exit(1);
    } 
    for (jj = 0; jj < nInps; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      vecOneS[jj] = ddata;
    }
    for (jj = 0; jj < nInputs_; jj++)
    {
      nn = indices[jj];
      vecSamples_[ii*nInputs_+jj] = vecOneS[nn] ;
    }
    fgets(pString, 1000, fp);
  }
  fscanf(fp, "%s", pString);
  //**/fscanf(fp, "%s", pString);
  //**/if (strcmp(pString, "PSUADE_END"))
  //**/{
  //**/  printf("PDFSampleHist ERROR: File should end with PSUADE_END\n");
  //**/  exit(1);
  //**/} 
  fclose(fp);
  printOutTS(PL_INFO,
      "PDFSampleHist INFO: Sample file '%s' has been read.\n",fname);
  printOutTS(PL_INFO,"   Sample size   = %d\n", nSamples_);
  printOutTS(PL_INFO,"   No. of inputs = %d\n", nInputs_);
  if (indices != NULL)
  {
    for (ii = 0; ii < nInputs_; ii++)
      printOutTS(PL_INFO,
        "   Input %d has column %d in the sample file.\n",ii+1, 
        indices[ii]+1);
  }
  //**/ -------------------------------------------------------------
  //**/ generate pdf
  //**/ -------------------------------------------------------------
  //**/ find upper and lower bounds
  vecLowerBs_.setLength(nInputs_);
  vecUpperBs_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    dmin = dmax = vecSamples_[ii];
    for (nn = 1; nn < nSamples_; nn++)
    {
      ddata = vecSamples_[nn*nInputs_+ii];
      if (ddata < dmin) dmin = ddata;
      if (ddata > dmax) dmax = ddata;
    }
    vecLowerBs_[ii] = dmin - 0.01 * (dmax - dmin);
    vecUpperBs_[ii] = dmax + 0.01 * (dmax - dmin);
    if (vecLowerBs_[ii] == vecUpperBs_[ii])
    {
      printf("PDFSampleHist ERROR: Upper bound=lower bound for input %d.\n",
             ii+1);
      exit(1);
    }
  }
  //**/ generate a grid
  int ss, nnz=0, itmp;
  vecIncr.setLength(nInputs_+1);
  if (nInputs_ == 1 ) n1d_ = 2000;
  if (nInputs_ == 2 ) n1d_ = 1024;
  if (nInputs_ == 3 ) n1d_ = 100;
  if (nInputs_ == 4 ) n1d_ = 36;
  if (nInputs_ == 5 ) n1d_ = 16;
  if (nInputs_ == 6 ) n1d_ = 11;
  if (nInputs_ == 7 ) n1d_ = 8;
  if (nInputs_ == 8 ) n1d_ = 6;
  if (nInputs_ == 9 ) n1d_ = 5;
  if (nInputs_ == 10) n1d_ = 4;
  nCells_ = 1;
  vecIncr[0] = nCells_;
  for (ii = 1; ii <= nInputs_; ii++)
  {
    nCells_ *= n1d_;
    vecIncr[ii] = nCells_;
  }
  vecGraphI.setLength(nCells_+1);
  vecGraphJ.setLength(nCells_*nInputs_*2+1);
  vecGraphI[0] = nnz;
  for (jj = 0; jj < nCells_; jj++)
  {
    itmp = jj;
    for (ii = 0; ii < nInputs_; ii++)
    {
      ss = itmp % n1d_;
      itmp = itmp / n1d_;
      if (ss > 0     ) vecGraphJ[nnz++] = jj - vecIncr[ii];
      if (ss < n1d_-1) vecGraphJ[nnz++] = jj + vecIncr[ii];
    }
    vecGraphI[jj+1] = nnz;
  }
  //**/ subdivide the grid into 100 subdomains
  vecCellsOccupied_.setLength(nCells_);
  nRegions_ = 100;
#ifdef HAVE_METIS
  int wgtflag=0, numflag=0, edgeCut=0;
  int *options = new int[1];
  options[0] = 0;
  METIS_PartGraphRecursive(&nCells_, vecGraphI.getIVector(), 
         vecGraphJ.getIVector(), NULL, NULL, &wgtflag,&numflag,
         &nRegions_,options,&edgeCut,vecCellsOccupied_.getIVector());
  delete [] options;
#else
  printf("PDFSampleHist ERROR : METIS not installed.\n");
  exit(1);
#endif
  //**/ now cellsOccupied[i] has subdomain number
  //**/ next map sample points to subdomains
  vecRegionProbs_.setLength(nRegions_);
  vecSampleMap_.setLength(nSamples_);
  for (ii = 0; ii < nRegions_; ii++) vecRegionProbs_[ii] = 0.0;
  for (ss = 0; ss < nSamples_; ss++)
  {
    itmp = 0;
    for (ii = nInputs_-1; ii >= 0; ii--)
    {
      itmp = itmp * n1d_;
      ddata = vecSamples_[ss*nInputs_+ii];
      ddata = (ddata-vecLowerBs_[ii])/(vecUpperBs_[ii]-vecLowerBs_[ii]);
      if (ddata == 1.0) jj = n1d_ - 1;
      else              jj = (int) (ddata * n1d_);
      itmp += jj;
    }
    if (itmp < 0 || itmp >= nCells_)
    {
      printf("PDFSampleHist::refine INTERNAL ERROR.\n");
      printf("               Consult PSUADE developer.\n");
    }
    //**/ fetch which subdomain this grid cell belongs to
    jj = vecCellsOccupied_[itmp];
    vecSampleMap_[ss] = jj;
    vecRegionProbs_[jj] += 1.0;
  }
  ddata = 0.0;
  for (ss = 0; ss < nRegions_; ss++) ddata += vecRegionProbs_[ss];
  for (ss = 0; ss < nRegions_; ss++) vecRegionProbs_[ss] /= ddata;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PDFSampleHist::~PDFSampleHist()
{
  vecSamples_.clean();
  vecLowerBs_.clean();
  vecUpperBs_.clean();
  vecCellsOccupied_.clean();
  vecSampleMap_.clean();
  vecRegionProbs_.clean();
}

// ************************************************************************
// forward transformation to range
// ------------------------------------------------------------------------
int PDFSampleHist::getPDF(int length, double *inData, double *outData)
{
  int    ss, ii, jj, itmp;
  double ddata;
  if (psConfig_.PDFDiagnosticsIsOn())
    printf("PDFSampleHist: getPDF begins (length = %d)\n",length);
  for (ss = 0; ss < length; ss++)
  {
    itmp = 0;
    for (ii = nInputs_-1; ii >= 0; ii--)
    {
      itmp = itmp * n1d_;
      ddata = inData[ss*nInputs_+ii];
      ddata = (ddata-vecLowerBs_[ii])/(vecUpperBs_[ii]-vecLowerBs_[ii]);
      if (ddata == 1.0) jj = n1d_ - 1;
      else              jj = (int) (ddata * n1d_);
      itmp += jj;
    }
    if (itmp < 0 || itmp >= nCells_) outData[ii] = 0.0;
    else
    {
      jj = vecCellsOccupied_[itmp];
      vecSampleMap_[ss] = jj;
      outData[ii] = vecRegionProbs_[jj];
    }
  }
  if (psConfig_.PDFDiagnosticsIsOn()) 
    printf("PDFSampleHist: getPDF ends.\n");
  return 0;
}

// ************************************************************************
// look up cumulative density
// ------------------------------------------------------------------------
int PDFSampleHist::getCDF(int length, double *inData, double *outData)
{
  printf("PDFSampleHist::getCDF not available.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = 0;
  return -1;
}

// ************************************************************************
// transformation to range
// ------------------------------------------------------------------------
int PDFSampleHist::invCDF(int length, double *inData, double *outData)
{
  printf("PDFSampleHist::invCDF not available.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = 0;
  return -1;
}

// ************************************************************************
// generate a sample
// ------------------------------------------------------------------------
int PDFSampleHist::genSample(int length,double *outData,double *, double *)
{
  int ii, jj, ind;

  //**/ -------------------------------------------------------------
  //**/ generate sample
  //**/ -------------------------------------------------------------
  if (psConfig_.PDFDiagnosticsIsOn())
     printf("PDFSampleHist: genSample begins (length = %d)\n",length);
  for (ii = 0; ii < length; ii++)
  {
    ind = PSUADE_rand() % nSamples_;
    for (jj = 0; jj < nInputs_; jj++)
      outData[ii*nInputs_+jj] = vecSamples_[ind*nInputs_+jj];
  }
  if (psConfig_.PDFDiagnosticsIsOn()) 
    printf("PDFSampleHist: genSample ends.\n");
  return 0;
}

// ************************************************************************
// get mean
// ------------------------------------------------------------------------
double PDFSampleHist::getMean()
{
  printf("PDFSampleHist::getMean not available for this distribution.\n");
  return 0.0;
}

// ************************************************************************
// get mean
// ------------------------------------------------------------------------
int PDFSampleHist::setParam(char *sparam)
{
  char winput[1001];
  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "setInput"))
  {
    sscanf(sparam, "%s %d", winput, &whichInput_);
    if (whichInput_ < 0) whichInput_ = 0;
  }
  return 0;
}


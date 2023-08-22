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
// DATE   : 2013
// ************************************************************************
#include <stdio.h>
#include <math.h>
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PDFSample.h"
#include "PrintingTS.h"
#include "Psuade.h"
#define PABS(x) (((x) >= 0) ? x : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PDFSample::PDFSample(int scount, char *fname, int *indices)
{
  int    ii, jj, nn, nInps;
  double ddata, dmin, dmax;
  char   pString[1001], filename[1001], *cString;
  FILE   *fp=NULL;
  psVector vecOneS;

  //**/ -------------------------------------------------------------
  //**/ check sample file
  //**/ -------------------------------------------------------------
  if (fname == NULL || !strcmp(fname, "NONE"))
  {
    printf("PDFSample constructor: expecting a sample file.\n");
    printf("                        having the following format: \n");
    printf("line 1: (optional) PSUADE_BEGIN\n");
    printf("line 2: <number of sample points> <number of inputs>\n");
    printf("line 3: (optional) : '#' followed by input names\n");
    printf("line 4: 1 sample point 1 inputs \n");
    printf("line 5: 2 sample point 2 inputs \n");
    printf("line 6: 3 sample point 3 inputs \n");
    printf("...\n");
    printf("line n: (optional) PSUADE_END\n");
    printf("line  : (optional) #perturb (turn on perturbation)\n");
    snprintf(pString,100,"Enter name of sample file : ");
    getString(pString, filename);
    nn = strlen(filename);
    if (nn > 1000)
    {
      printf("PDFSample constructor ERROR: file name too long.\n");
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
  perturb_ = 0;
  fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("PDFSample ERROR: cannot open sample file %s.\n",filename);
    exit(1);
  }
  fscanf(fp, "%s", pString);
  if (strcmp(pString, "PSUADE_BEGIN"))
  {
    fclose(fp);
    fp = fopen(filename, "r");
  } 
  fscanf(fp, "%d %d", &nSamples_, &nInps);
  if (nSamples_ < 1)
  {
    printf("PDFSample ERROR: sample file has nSamples <= 0.\n");
    exit(1);
  }
  if (nInps < 1)
  {
    printf("PDFSample ERROR: sample file has nInputs <= 0.\n");
    exit(1);
  }
  //**/ if sample file has different nInputs than the declared nInputs, 
  //**/ and no incoming index list is given, flag an error
  if (nInputs_ != nInps && indices == NULL)
  {
    printf("PDFSample ERROR: nInputs does not match.\n");
    printf("           nInputs in your sample file    = %d\n",nInps);
    printf("           nInputs from psuade input file = %d\n",nInputs_);
    exit(1);
  }
  if (indices != NULL)
  {
    for (ii = 0; ii < scount; ii++)
    {
      if (indices[ii] < 0 || indices[ii] >= nInps)
      {
        printf("PDFSample ERROR: sample index > nInputs in sample file.\n");
        printf("          sample index requested         = %d\n",
               indices[ii]+1);
        printf("          nInputs in your sample file    = %d\n",nInps);
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
      printf("PDFSample ERROR: invalid sample number.\n");
      printf("           Expected: %d\n", ii+1);
      printf("           Read:     %d\n", nn);
      printf("Advice: check your data format and line number %d.\n\n",ii+2);
      printf("Correct Format: \n");
      printf("line 1: (optional) PSUADE_BEGIN\n");
      printf("line 2: <number of sample points> <number of inputs>\n");
      printf("line 3: (optional) : '#' followed by input names\n");
      printf("line 4: 1 sample point 1 inputs \n");
      printf("line 5: 2 sample point 2 inputs \n");
      printf("line 6: 3 sample point 3 inputs \n");
      printf("...\n");
      printf("line n: (optional) PSUADE_END\n");
      printf("line  : (optional) #perturb (turn on perturbation)\n");
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
  if (!strcmp(pString, "#perturb")) perturb_ = 1;
  if (perturb_ == 0)
  {
    cString = psConfig_.getParameter("randomize");
    if (cString != NULL) perturb_ = 1;
  }

  //**/fscanf(fp, "%s", pString);
  //**/if (strcmp(pString, "PSUADE_END"))
  //**/{
  //**/   printf("PDFSample ERROR: file should end with PSUADE_END.\n");
  //**/   exit(1);
  //**/} 
  fclose(fp);
  printOutTS(PL_INFO,"PDFSample INFO: sample file '%s' has been read.\n", 
             fname);
  printOutTS(PL_INFO,"   sample size   = %d\n", nSamples_);
  printOutTS(PL_INFO,"   no. of inputs = %d\n", nInputs_);
  if (perturb_ == 1)
  {
    printOutTS(PL_INFO,"   Perturbation has been turned on ");
    printOutTS(PL_INFO,"(sample drawn will be perturbed).\n");
    printOutTS(PL_INFO,
         "   The amount of perturbation is 0.01 of input range.\n");
  }
  if (indices != NULL)
  {
    for (ii = 0; ii < nInputs_; ii++)
      printOutTS(PL_INFO,"   Input %d has column %d in the sample file.\n",
                 ii+1, indices[ii]+1);
  }
  vecRanges_.setLength(nInputs_);
  for (ii = 0; ii < nInputs_; ii++)
  {
    dmin = dmax = vecSamples_[ii];
    for (nn = 1; nn < nSamples_; nn++)
    {
      ddata = vecSamples_[nn*nInputs_+ii];
      if (ddata < dmin) dmin = ddata;
      if (ddata > dmax) dmax = ddata;
    }
    vecRanges_[ii] = dmax - dmin;
  }
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PDFSample::~PDFSample()
{
  vecSamples_.clean();
  vecRanges_.clean();
}

// ************************************************************************
// forward transformation to range
// ------------------------------------------------------------------------
int PDFSample::getPDF(int length, double *inData, double *outData)
{
  printf("PDFSample::getPDF not available.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = 0;
  return -1;
}

// ************************************************************************
// look up cumulative density
// ------------------------------------------------------------------------
int PDFSample::getCDF(int length, double *inData, double *outData)
{
  printf("PDFSample::getCDF not available.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = 0;
  return -1;
}

// ************************************************************************
// transformation to range
// ------------------------------------------------------------------------
int PDFSample::invCDF(int length, double *inData, double *outData)
{
  printf("PDFSample WARNING::invCDF not available - no change.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = inData[ii];
  return -1;
}

// ************************************************************************
// generate a sample
// ------------------------------------------------------------------------
int PDFSample::genSample(int length,double *outData,double *, double *)
{
  int ii, jj, ind;

  //**/ -------------------------------------------------------------
  //**/ generate sample
  //**/ -------------------------------------------------------------
  if (psConfig_.PDFDiagnosticsIsOn())
    printf("PDFSample: genSample begins (length = %d)\n",length);
  if (perturb_ == 0)
  {
    //**/ if length > nSamples, make sure the sample gets all instances
    for (ii = 0; ii < length; ii++)
    {
      if (ii < nSamples_) ind = ii;
      else                ind = PSUADE_rand() % nSamples_;
      for (jj = 0; jj < nInputs_; jj++)
        outData[ii*nInputs_+jj] = vecSamples_[ind*nInputs_+jj];
    }
  }
  else
  {
    for (ii = 0; ii < length; ii++)
    {
      ind = PSUADE_rand() % nSamples_;
      for (jj = 0; jj < nInputs_; jj++)
        outData[ii*nInputs_+jj] = vecSamples_[ind*nInputs_+jj] +
                   0.01 * (PSUADE_drand() - 0.5) * vecRanges_[jj];
    }
  }
  if (psConfig_.PDFDiagnosticsIsOn()) 
    printf("PDFSample: genSample ends.\n");
  return 0;
}

// ************************************************************************
// get mean
// ------------------------------------------------------------------------
double PDFSample::getMean()
{
  printf("PDFSample::getMean not available for this distribution.\n");
  return 0.0;
}

// ************************************************************************
// get mean
// ------------------------------------------------------------------------
int PDFSample::setParam(char *sparam)
{
  char winput[1001];
  sscanf(sparam, "%s", winput);
  if (!strcmp(winput, "setInput"))
  {
    sscanf(sparam, "%s %d", winput, &whichInput_);
    if (whichInput_ < 0) whichInput_ = 0;
  }
  else if (!strcmp(winput, "perturb"))
  {
    perturb_ = 1;
  }
  return 0;
}


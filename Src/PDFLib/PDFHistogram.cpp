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
// Functions for the histogram-based distribution
// AUTHOR : CHARLES TONG
// DATE   : 2015
//**/ This is currently not intended to be a PDF class. Rather it is
//**/ a class to support the generation of scenarios in OUU
// ************************************************************************
#include <stdio.h>
#include <math.h>
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PDFHistogram.h"
#include "PrintingTS.h"
#include "psVector.h"
#define PABS(x) (((x) >= 0) ? x : -(x))

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PDFHistogram::PDFHistogram(char *fname, int nInps, int *indices, int *incrs)
{
  int    ii, jj, ss, nn, nSamples, nInputs;
  double ddata, dmin, dmax;
  char   pString[1001], filename[1001];
  FILE   *fp=NULL;
  psVector vecOneS;

  //**/ -------------------------------------------------------------
  //**/ check sample file
  //**/ -------------------------------------------------------------
  if (fname == NULL || !strcmp(fname, "NONE"))
  {
    printf("PDFHistogram constructor: expecting a sample file.\n");
    printf("                          having the following format: \n");
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
      printf("PDFHistogram constructor ERROR: file name too long.\n");
      exit(1);
    }
    filename[nn-1] = '\0';
  }
  else strcpy(filename, fname);

  //**/ -------------------------------------------------------------
  //**/ read in sample
  //**/ -------------------------------------------------------------
  fp = fopen(filename, "r");
  if (fp == NULL)
  {
    printf("PDFHistogram ERROR: cannot open sample file %s.\n",
           filename);
    exit(1);
  }
  fscanf(fp, "%s", pString);
  if (strcmp(pString, "PSUADE_BEGIN"))
  {
    fclose(fp);
    fp = fopen(filename, "r");
  } 
  fscanf(fp, "%d %d", &nSamples, &nInputs);
  if (nSamples < 100)
  {
    printf("PDFHistogram ERROR: sample file needs to be >= 100.\n");
    fclose(fp);
    exit(1);
  }
  if (nInputs < 1)
  {
    printf("PDFHistogram ERROR: sample file has nInputs <= 0.\n");
    fclose(fp);
    exit(1);
  }
  //**/ if sample file has different nInputs than the declared nInputs, 
  //**/ and no incoming index list is given, flag an error
  if (nInputs != nInps)
  {
    printf("PDFHistogram ERROR: nInputs does not match.\n");
    printf("          nInputs in your sample file    = %d\n",nInputs);
    printf("          nInputs from psuade input file = %d\n",nInps);
    fclose(fp);
    exit(1);
  }
  if (indices != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
    {
      if (indices[ii] < 0 || indices[ii] >= nInputs)
      {
        printf("PDFHistogram ERROR: wrong sample index.\n");
        printf("             sample index requested         = %d\n",
               indices[ii]+1);
        fclose(fp);
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
  vecSamples_.setLength(nSamples*nInputs);
  vecOneS.setLength(nInps);
  for (ii = 0; ii < nSamples; ii++)
  {
    fscanf(fp, "%d", &nn);
    if (nn != (ii+1))
    {
      printf("PDFHistogram ERROR: invalid sample number.\n");
      printf("             Expected: %d\n", ii+1);
      printf("             Read:     %d\n", nn);
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
      fclose(fp);
      exit(1);
    } 
    for (jj = 0; jj < nInps; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      vecOneS[jj] = ddata;
    }
    for (jj = 0; jj < nInputs; jj++)
    {
      if (indices != NULL) nn = indices[jj];
      else                 nn = jj;
      vecSamples_[ii*nInputs+jj] = vecOneS[nn] ;
    }
    fgets(pString, 1000, fp);
  }
  fscanf(fp, "%s", pString);
  //**/fscanf(fp, "%s", pString);
  //**/if (strcmp(pString, "PSUADE_END"))
  //**/{
  //**/   printf("PDFHistogram ERROR: file should end with PSUADE_END.\n");
  //**/   exit(1);
  //**/} 
  fclose(fp);
  printOutTS(PL_INFO,"PDFHistogram INFO: sample file '%s' has been read.\n", 
             fname);
  printOutTS(PL_INFO,"   Sample size   = %d\n", nSamples);
  printOutTS(PL_INFO,"   No. of inputs = %d\n", nInputs);
  if (indices != NULL)
  {
    for (ii = 0; ii < nInputs; ii++)
      printOutTS(PL_INFO,"   Input %d has column %d in the sample file.\n",
                 ii+1, indices[ii]+1);
  }

  //**/ -------------------------------------------------------------
  //**/ generate lower and upper bounds
  //**/ -------------------------------------------------------------
  vecLowerBs_.setLength(nInputs);
  vecUpperBs_.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++)
  {
    dmin = dmax = vecSamples_[ii];
    for (nn = 1; nn < nSamples; nn++)
    {
      ddata = vecSamples_[nn*nInputs+ii];
      if (ddata < dmin) dmin = ddata;
      if (ddata > dmax) dmax = ddata;
    }
    vecLowerBs_[ii] = dmin - 0.01 * (dmax - dmin);
    vecUpperBs_[ii] = dmax + 0.01 * (dmax - dmin);
    if (vecLowerBs_[ii] == vecUpperBs_[ii])
    {
      printf("PDFHistogram ERROR: upper bound=lower bound for input %d.\n",
             ii+1);
      exit(1);
    }
  }

  //**/ -------------------------------------------------------------
  //**/ get partitions
  //**/ -------------------------------------------------------------
  vecIncrs_.setLength(nInps);
  for (ii = 0; ii < nInputs; ii++)
  {
    if (incrs[ii] <= 0)
    {
      printf("PDFHistogram ERROR: invalid partition.\n");
      exit(1);
    }
    vecIncrs_[ii] = incrs[ii];
  }
   
  //**/ -------------------------------------------------------------
  //**/ build histogram
  //**/ -------------------------------------------------------------
  psIVector vecMinBs, vecMaxBs;
  initHistogram();
  vecMinBs.setLength(nInputs);
  vecMaxBs.setLength(nInputs);
  for (ii = 0; ii < nInputs; ii++) vecMinBs[ii] = 2 * vecIncrs_[ii]; 
  for (ii = 0; ii < nInputs; ii++) vecMaxBs[ii] = 0;
  for (ss = 0; ss < nSamples; ss++)
  {
    for (ii = 0; ii < nInputs; ii++) 
    {
      ddata = vecSamples_[ss*nInputs+ii];
      ddata = (ddata - vecLowerBs_[ii]) / 
              (vecUpperBs_[ii]-vecLowerBs_[ii]);
      if (ddata == 1.0) jj = vecIncrs_[ii] - 1;
      else              jj = (int) (ddata * vecIncrs_[ii]);
      vecIndSet_[ii] = jj;
      if (jj < vecMinBs[ii]) vecMinBs[ii] = jj;
      if (jj > vecMaxBs[ii]) vecMaxBs[ii] = jj;
    }
    mergeHistogram(vecIndSet_.getIVector(),ss);
  }
  finalizeHistogram();
  for (ii = 0; ii < nInputs; ii++) 
  {
    printf("PDF box range for Input %4d = [%d , %d]\n",ii+1,
           vecMinBs[ii],vecMaxBs[ii]);
  }
}

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PDFHistogram::PDFHistogram(int nSamp,int nInps,double *samInputs,
                           int *incrs, int flag)
{
  int    ii, jj, ss, maxCnt, maxInd;
  double ddata, dmin, dmax;
  char   pString[1001];

  //**/ -------------------------------------------------------------
  //**/ error checking
  //**/ -------------------------------------------------------------
  if (nSamp < 100)
  {
    printf("PDFHistogram ERROR: sample file needs to be >= 100.\n");
    exit(1);
  }
  if (nInps < 1)
  {
    printf("PDFHistogram ERROR: sample file has nInputs <= 0.\n");
    exit(1);
  }
  //**/ -------------------------------------------------------------
  //**/ copy sample
  //**/ -------------------------------------------------------------
  vecSamples_.setLength(nSamp*nInps);
  for (ii = 0; ii < nSamp*nInps; ii++) vecSamples_[ii] = samInputs[ii];
  printOutTS(PL_INFO,"PDFHistogram  Sample size   = %d\n", nSamp);
  printOutTS(PL_INFO,"PDFHistogram  No. of inputs = %d\n", nInps);

  //**/ -------------------------------------------------------------
  //**/ generate lower and upper bounds
  //**/ -------------------------------------------------------------
  vecLowerBs_.setLength(nInps);
  vecUpperBs_.setLength(nInps);
  for (ii = 0; ii < nInps; ii++)
  {
    dmin = dmax = vecSamples_[ii];
    for (ss = 1; ss < nSamp; ss++)
    {
      ddata = vecSamples_[ss*nInps+ii];
      if (ddata < dmin) dmin = ddata;
      if (ddata > dmax) dmax = ddata;
    }
    vecLowerBs_[ii] = dmin - 0.01 * (dmax - dmin);
    vecUpperBs_[ii] = dmax + 0.01 * (dmax - dmin);
    if (vecLowerBs_[ii] == vecUpperBs_[ii])
    {
      printf("PDFHistogram ERROR: upper bound=lower bound for input %d.\n",
             ii+1);
      exit(1);
    }
  }

  //**/ -------------------------------------------------------------
  //**/ get partitions
  //**/ -------------------------------------------------------------
  vecIncrs_.setLength(nInps);
  for (ii = 0; ii < nInps; ii++)
  {
    if (incrs[ii] <= 0)
    {
      printf("PDFHistogram ERROR: invalid partition.\n");
      exit(1);
    }
    vecIncrs_[ii] = incrs[ii];
  }
  vecHistMaps_ = NULL;
   
  //**/ -------------------------------------------------------------
  //**/ build histogram
  //**/ -------------------------------------------------------------
  initHistogram();
  for (ss = 0; ss < nSamp; ss++)
  {
    for (ii = 0; ii < nInps; ii++) 
    {
      ddata = vecSamples_[ss*nInps+ii];
      ddata = (ddata - vecLowerBs_[ii]) / 
              (vecUpperBs_[ii]-vecLowerBs_[ii]);
      if (ddata == 1.0) jj = vecIncrs_[ii] - 1;
      else              jj = (int) (ddata * vecIncrs_[ii]);
      vecIndSet_[ii] = jj;
    }
    mergeHistogram(vecIndSet_.getIVector(),ss);
  }
  finalizeHistogram();

  //**/ -------------------------------------------------------------
  //**/ compute and check means
  //**/ -------------------------------------------------------------
  vecMeans_.setLength(nInps);
  for (ii = 0; ii < nInps; ii++) 
  {
    vecMeans_[ii] = 0.0;
    for (ss = 0; ss < nSamp; ss++)
      vecMeans_[ii] += vecSamples_[ss*nInps+ii];
    vecMeans_[ii] /= (double) nSamp;
  }
  for (ii = 0; ii < nInps; ii++) 
  {
    ddata = vecMeans_[ii];
    ddata = (ddata - vecLowerBs_[ii]) / (vecUpperBs_[ii]-vecLowerBs_[ii]);
    if (ddata == 1.0) jj = vecIncrs_[ii] - 1;
    else              jj = (int) (ddata * vecIncrs_[ii]);
    vecIndSet_[ii] = jj;
  }
  mergeHistogram(vecIndSet_.getIVector(),-1);

  //**/ -------------------------------------------------------------
  //**/ write to file
  //**/ -------------------------------------------------------------
  if (flag == 1)
  {
    FILE *fp = fopen("psuade_pdfhist_sample","w");
    if (fp == NULL)
    {
      printf("PDFHistogram ERROR: no histogram file created.\n");
      return;
    }
    fprintf(fp, "%d %d\n", nHist_, nInps);
    for (ii = 0; ii < nHist_; ii++)
    {
      for (jj = 0; jj < nInps; jj++)
        fprintf(fp, "%16.8e ", vecHistMeds_[ii*nInps+jj]);
      fprintf(fp, "%16.8e\n", 1.0*vecHistCnts_[ii]/nSamp);
    }      
    fclose(fp);
    printf("PDFHistogram: a compressed sample is now ready in\n");
    printf(" psuade_pdfhist_sample.\n");
    maxCnt = vecHistCnts_[0];
    maxInd = 0;
    for (ss = 1; ss < nHist_; ss++) 
    {
      if (vecHistCnts_[ss] > maxCnt) 
      {
        maxCnt = vecHistCnts_[ss];
        maxInd = ss;
      }
    }
    printf("PDFHistogram: the mode is at sample number %d\n",maxInd+1);
  }
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PDFHistogram::~PDFHistogram()
{
  if (vecHistMaps_ != NULL)
  {
    for (int ii = 0; ii < nHist_; ii++) vecHistMaps_[ii].clean();
    delete [] vecHistMaps_;
  }
}

// ************************************************************************
// forward transformation to range
// ------------------------------------------------------------------------
int PDFHistogram::getPDF(int length, double *inData, double *outData)
{
  int    ss, ii, jj, nSamples, nInputs;
  double ddata;
  if (psConfig_.PDFDiagnosticsIsOn())
    printf("PDFHistogram: getPDF begins (length = %d)\n",length);
  nInputs = vecLowerBs_.length();
  for (ss = 0; ss < length; ss++)
  {
    for (ii = 0; ii < nInputs; ii++) 
    {
      ddata = inData[ss*nInputs+ii];
      ddata = (ddata - vecLowerBs_[ii])/(vecUpperBs_[ii]-vecLowerBs_[ii]);
      if (ddata == 1.0) jj = vecIncrs_[ii] - 1;
      else              jj = (int) (ddata * vecIncrs_[ii]);
      vecIndSet_[ii] = jj;
    }
    outData[ss] = findProbability(vecIndSet_);
  }
  if (psConfig_.PDFDiagnosticsIsOn()) printf("PDFHistogram: getPDF ends.\n");
  return 0;
}

// ************************************************************************
// look up cumulative density
// ------------------------------------------------------------------------
int PDFHistogram::getCDF(int length, double *inData, double *outData)
{
  printf("PDFHistogram::getCDF not available.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = inData[ii];
  return -1;
}

// ************************************************************************
// transformation to range
// ------------------------------------------------------------------------
int PDFHistogram::invCDF(int length, double *inData, double *outData)
{
  printf("PDFHistogram::invCDF not available.\n");
  for (int ii = 0; ii < length; ii++) outData[ii] = inData[ii];
  return -1;
}

// ************************************************************************
// generate a sample
// ------------------------------------------------------------------------
int PDFHistogram::genSample(int length,double *outData, double *lowers, 
                            double *uppers)
{
  int    ii, kk, ind, count, nInputs;
  double ddata, dtemp;

  //**/ -------------------------------------------------------------
  //**/ generate sample
  //**/ -------------------------------------------------------------
  if (psConfig_.PDFDiagnosticsIsOn())
    printf("PDFHistogram: genSample begins (length = %d)\n",length);
  if (lowers == NULL || uppers == NULL)
  {
    printf("PDFHist genSample ERROR - lower/upper bound unavailable.\n"); 
    exit(1);
  }
  count = 0;
  nInputs = vecLowerBs_.length();
  while (count < length)
  {
    ddata = PSUADE_drand();
    //**/ search histogram to locate the bin
    ind = searchHistogram(ddata);
    //**/ select a random point from the bin
    for (ii = 0; ii < nInputs; ii++)
    {
      kk = vecHistCells_[ind*nInputs+ii];
      ddata = (PSUADE_drand() + kk) / vecIncrs_[ii];
      if (lowers != NULL && uppers != NULL)
      {
        dtemp = uppers[ii] - lowers[ii];
        outData[count*nInputs+ii] = ddata * dtemp + lowers[ii];
      }
      else
      {
        dtemp = vecUpperBs_[ii] - vecLowerBs_[ii];
        outData[count*nInputs+ii] = ddata * dtemp + vecLowerBs_[ii];
      }
    }
    count++;
  }
  if (psConfig_.PDFDiagnosticsIsOn()) 
    printf("PDFHistogram: genSample ends.\n");
  return 0;
}

// ************************************************************************
// get mean
// ------------------------------------------------------------------------
double PDFHistogram::getMean()
{
  printf("PDFHistogram::getMean not relevant for this distribution.\n");
  printf("              Use getMeans.\n");
  return 0.0;
}

// ************************************************************************
// get means
// ------------------------------------------------------------------------
int PDFHistogram::getMeans(double *means)
{
  int ii, nInputs;
  nInputs = vecLowerBs_.length();
  if (vecMeans_.length() != 0 && means != NULL)
    for (ii = 0; ii < nInputs; ii++) means[ii] = vecMeans_[ii];
  else if (means != NULL)
    for (ii = 0; ii < nInputs; ii++) means[ii] = 0.0;
  return 0.0;
}

// ************************************************************************
// search the histogram CDF using a random number, return cell index
// ------------------------------------------------------------------------
int PDFHistogram::searchHistogram(double prob)
{
  int    ii;
  double ddata;
  if (prob < vecHistCDF_[0]) return 0; 
  if (prob >= vecHistCDF_[nHist_-1]) return (nHist_-1); 
  for (ii = 1; ii < nHist_; ii++)
    if (prob <= vecHistCDF_[ii]) return (ii);
  return 0;
}

// ************************************************************************
// initialize histogram
// ------------------------------------------------------------------------
void PDFHistogram::initHistogram()
{
  vecIndSet_.setLength(vecLowerBs_.length());
  if (vecHistMaps_ != NULL)
  {
    for (int ii = 0; ii < nHist_; ii++) vecHistMaps_[ii].clean();
    delete [] vecHistMaps_;
  }
  nHist_ = 0;
  vecHistMaps_ = NULL;
}

// ************************************************************************
// merge histogram
// ------------------------------------------------------------------------
void PDFHistogram::mergeHistogram(int *indexSet, int samNum)
{
  int ii, jj, index, nInputs;
  psVector vecDble;
  psIVector vecInt;

  nInputs = vecLowerBs_.length();
  for (ii = 0; ii < nHist_; ii++)
  {
    for (jj = 0; jj < nInputs; jj++)
      if (vecHistCells_[ii*nInputs+jj] != indexSet[jj]) break;
    if (jj == nInputs) break;
  }
  index = ii;
  if (samNum == -1)
  {
    if (index == nHist_) 
      printf("PDFHistogram INFO: the sample mean is outside the hull.\n"); 
    else
    {
      printf("PDFHistogram INFO: the sample mean is inside the hull.\n"); 
      if (index == 0) printf("Weight = %e\n", vecHistCDF_[index]);
      else            
        printf("Weight = %e\n",vecHistCDF_[index]-vecHistCDF_[index-1]);
    }
    return;
  }
  //*** new entry
  if (index == nHist_)
  {
    if (nHist_ % 1000 == 0)
    {
      vecInt = vecHistCells_;
      vecHistCells_.setLength((nHist_+1000)*nInputs);
      for (ii = 0; ii < nHist_*nInputs; ii++) 
        vecHistCells_[ii] = vecInt[ii];
      vecInt = vecHistCnts_;
      vecHistCnts_.setLength(nHist_+1000);
      for (ii = 0; ii < nHist_; ii++) vecHistCnts_[ii] = vecInt[ii];
      for (ii = nHist_; ii < nHist_+1000; ii++) vecHistCnts_[ii] = 0;
      psIVector *tmpIVecs = vecHistMaps_;
      vecHistMaps_ = new psIVector[nHist_+1000];
      for (ii = 0; ii < nHist_; ii++) vecHistMaps_[ii] = tmpIVecs[ii];
      if (tmpIVecs != NULL) delete [] tmpIVecs;
      vecDble = vecHistMeds_;
      vecHistMeds_.setLength((nHist_+1000)*nInputs);
      for (ii = 0; ii < nHist_*nInputs; ii++) 
        vecHistMeds_[ii] = vecDble[ii];
      for (ii = nHist_*nInputs; ii < (nHist_+1000)*nInputs; ii++) 
        vecHistMeds_[ii] = 0;
    }
    vecHistMaps_[nHist_].setLength(100); 
    vecHistMaps_[nHist_][0] = samNum;
    vecHistCnts_[nHist_]++;
    for (ii = 0; ii < nInputs; ii++) 
    {
      vecHistCells_[nHist_*nInputs+ii] = indexSet[ii]; 
      vecHistMeds_[nHist_*nInputs+ii] = vecSamples_[samNum*nInputs+ii]; 
    }
    nHist_++;
  }
  else
  //**/ existing entry
  {
    if (vecHistCnts_[index] % 100 == 0)
    {
      psIVector tmpIVec = vecHistMaps_[index];
      vecHistMaps_[index].setLength(vecHistCnts_[index]+100); 
      for (ii = 0; ii < vecHistCnts_[index]; ii++) 
        vecHistMaps_[index][ii] = tmpIVec[ii];
    }
    vecHistMaps_[index][vecHistCnts_[index]] = samNum;
    vecHistCnts_[index]++;
    for (ii = 0; ii < nInputs; ii++) 
      vecHistMeds_[index*nInputs+ii] += vecSamples_[samNum*nInputs+ii]; 
  }
}

// ************************************************************************
// finalize histogram
// ------------------------------------------------------------------------
void PDFHistogram::finalizeHistogram()
{
  int ss, ii, nInputs, nSamples;
  nInputs = vecLowerBs_.length();
  nSamples = vecSamples_.length() / nInputs;
  for (ss = 0; ss < nHist_; ss++) 
    for (ii = 0; ii < nInputs; ii++) 
      vecHistMeds_[ss*nInputs+ii] /= (double) vecHistCnts_[ss];
  vecHistCDF_.setLength(nHist_); 
  for (ss = 0; ss < nHist_; ss++)
     vecHistCDF_[ss] = 1.0 * vecHistCnts_[ss] / nSamples; 
  for (ss = 1; ss < nHist_; ss++)
     vecHistCDF_[ss] += vecHistCDF_[ss-1];
}

// ************************************************************************
// find probability
// ------------------------------------------------------------------------
double PDFHistogram::findProbability(psIVector vecIndSet)
{
  int    ss, ii, jj, nInputs, nSamples;
  double ddata;
  nInputs = vecLowerBs_.length();
  nSamples = vecSamples_.length() / nInputs;
  for (ss = 0; ss < nHist_; ss++)
  {
    for (ii = 0; ii < nInputs; ii++) 
      if (vecHistCells_[ss*nInputs+ii] != vecIndSet[ii]) break;
    if (ii == nInputs) break;
  }
  return (1.0*vecHistCnts_[ss]/nSamples);
}


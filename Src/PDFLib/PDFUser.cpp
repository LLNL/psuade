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
// DATE   : 2016
// ************************************************************************
#include <stdio.h>
#include <math.h>
#include "sysdef.h"
#include "Psuade.h"
#include "PsuadeUtil.h"
#include "PDFUser.h"
#include "PrintingTS.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------
PDFUser::PDFUser(int ninputs)
{
  int    ii, leng;
  char   pString[1001];
  FILE   *fp=NULL;

  //**/ -------------------------------------------------------------
  //**/ check sample file
  //**/ -------------------------------------------------------------
  printAsterisks(PL_INFO, 0);
  printf("Please provide an executable for generating a sample.\n");
  printf("PSUADE will call your executable using the following form:\n");
  printf("\n    exec <nSamples> <nInputs> <filename>\n\n");
  printf("where <nSamples> is the desired sample size\n");
  printf("      <nInputs> is the number of parameters, and\n");
  printf("      <filename> is the name of a file into which the sample\n");
  printf("                 points are to be written to, one sample point\n");
  printf("                 per row.\n");
  printEquals(PL_INFO, 0);
  printf("Name of user executable file (with execute permission on): ");
  scanf("%299[^\n]", samGenerator_);
  leng = strlen(samGenerator_);
  for (ii = 0; ii < leng; ii++)
    if (samGenerator_[ii] == '\r') samGenerator_[ii] = '\0';
  printf("PDFUser INFO: user file = %s\n",samGenerator_);
  fp = fopen(samGenerator_, "r");
  if (fp == NULL)
  {
    printf("PDFUser ERROR: user executable %s not found.\n",samGenerator_);
    exit(1);
  }
  fclose(fp);
  //struct stat sb;
  //if (stat(regFile_, &sb) != 0 || !(sb.st_mode & S_IXUSR))
  //{
  //   printf("PDFUser WARNING : User executable does not have\n");
  //   printf("                  execute permission.\n");
  //}
  fgets(pString, 1000, stdin);
  nInputs_ = ninputs;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------
PDFUser::~PDFUser()
{
}

// ************************************************************************
// generate a sample
// ------------------------------------------------------------------------
int PDFUser::genSample(int length,double *outData, double *, double *)
{
  int  ii, jj, cnt;
  char sysCmd[1001];
  FILE *fp;

  //**/ -------------------------------------------------------------
  //**/ generate sample
  //**/ -------------------------------------------------------------
  if (psConfig_.PDFDiagnosticsIsOn())
    printf("PDFUser: genSample begins (length = %d)\n",length);
  snprintf(sysCmd,1000,"%s %d %d psPDF",samGenerator_,length,nInputs_);
  system(sysCmd);
  fp = fopen("psPDF", "r");
  if (fp == NULL)
  {
    printf("PDFUser ERROR: user-generated sample file not found.\n");
    exit(1);
  }
  cnt = 0;
  while (feof(fp) == 0 && cnt < length*nInputs_)
  {
    fscanf(fp, "%lg", &outData[cnt]);
    cnt++;
  }
  fclose(fp);
  if (cnt != length*nInputs_)
  {
    printf("PDFUser WARNING: user sample generator did not generate\n");
    printf("                 enough sample points.\n");
    printf("        Number of data expected = %d\n",length*nInputs_);
    printf("        Number of data read     = %d\n",cnt);
    exit(1);
  }
  if (psConfig_.PDFDiagnosticsIsOn()) printf("PDFUser: genSample ends.\n");
  return 0;
}


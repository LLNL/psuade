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
// Functions for PsuadeConfig 
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "sysdef.h"
#include "PsuadeConfig.h"
#include "PsuadeUtil.h"

// ************************************************************************
// constructor 
// ------------------------------------------------------------------------ 
PsuadeConfig::PsuadeConfig()
{
  printLevel_ = 0;
  //**/ reset should not change this. otherwise nothing interactive
  Interactive_ = 0;
  reset();
}

// ************************************************************************
// copy constructor by Bill Oliver
// ------------------------------------------------------------------------
PsuadeConfig::PsuadeConfig(const PsuadeConfig & ps)
{
  //**/ clean up
  printLevel_ = 0;
  reset();

  //**/ initialize 
  nLinesUsed_ = ps.nLinesUsed_;
  printLevel_ = ps.printLevel_;
  StrFileData_ = ps.StrFileData_; 
  IOExpertMode_ = ps.IOExpertMode_;
  RSExpertMode_ = ps.RSExpertMode_;
  RSExpertSave_ = ps.RSExpertSave_;
  SamExpertMode_ = ps.SamExpertMode_;
  AnaExpertMode_ = ps.AnaExpertMode_;
  AnaExpertSave_ = ps.AnaExpertSave_;
  OptExpertMode_ = ps.OptExpertMode_;
  PDFDiagMode_ = ps.PDFDiagMode_;
  RSCodeGen_ = ps.RSCodeGen_;
  RSCodeGenSave_ = ps.RSCodeGenSave_;
  MasterMode_ = ps.MasterMode_;
  Interactive_ = ps.Interactive_;
  InteractiveSave_ = ps.InteractiveSave_;
  OUULibMode_ = ps.OUULibMode_;
  GMMode_ = ps.GMMode_;
  PlotTool_ = ps.PlotTool_;
  RandomSeed_ = ps.RandomSeed_;
  RSMaxPts_ = ps.RSMaxPts_;
  RSConstraintSetOp_ = ps.RSConstraintSetOp_;
}

// ************************************************************************
// copy constructor 
// ------------------------------------------------------------------------
PsuadeConfig & PsuadeConfig::operator=(const PsuadeConfig & ps)
{
  if (this == &ps) return *this;

  //**/ clean up
  printLevel_ = 0;
  reset();

  //**/ copy 
  nLinesUsed_ = ps.nLinesUsed_;
  printLevel_ = ps.printLevel_;
  StrFileData_ = ps.StrFileData_; 
  IOExpertMode_ = ps.IOExpertMode_;
  RSExpertMode_ = ps.RSExpertMode_;
  RSExpertSave_ = ps.RSExpertSave_;
  SamExpertMode_ = ps.SamExpertMode_;
  AnaExpertMode_ = ps.AnaExpertMode_;
  AnaExpertSave_ = ps.AnaExpertSave_;
  OptExpertMode_ = ps.OptExpertMode_;
  PDFDiagMode_ = ps.PDFDiagMode_;
  RSCodeGen_ = ps.RSCodeGen_;
  RSCodeGenSave_ = ps.RSCodeGenSave_;
  MasterMode_ = ps.MasterMode_;
  Interactive_ = ps.Interactive_;
  InteractiveSave_ = ps.InteractiveSave_;
  OUULibMode_ = ps.OUULibMode_;
  GMMode_ = ps.GMMode_;
  PlotTool_ = ps.PlotTool_;
  RandomSeed_ = ps.RandomSeed_;
  RSMaxPts_ = ps.RSMaxPts_;
  RSConstraintSetOp_ = ps.RSConstraintSetOp_;
  return *this;
}

// ************************************************************************
// destructor 
// ------------------------------------------------------------------------ 
PsuadeConfig::~PsuadeConfig()
{ 
  StrFileData_.clean();
}

// ************************************************************************
// initialize 
// ------------------------------------------------------------------------ 
void PsuadeConfig::initialize(char *fname, int printLevel)
{
  int  ii, lineLeng=1000, lineCnt;
  char lineIn[1001], winput[1001];
  FILE *fIn;

  //**/ ----------------------------------------------------------------- 
  //**/  initialize
  //**/ ----------------------------------------------------------------- 
  reset();

  //**/ ----------------------------------------------------------------- 
  //**/  check whether file exists
  //**/ ----------------------------------------------------------------- 
  fIn = fopen(fname, "r");
  if (fIn == NULL) 
  {
    printf("PsuadeConfig ERROR:: configure file %s not found.\n",fname);
    exit(1);
  }

  //**/ ----------------------------------------------------------------- 
  //**/  check whether keyword found
  //**/ ----------------------------------------------------------------- 
  while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
  {
    sscanf(lineIn, "%s", winput);
    if (strcmp(winput, "PSUADE_CONFIG") == 0) break;
  }
  if (feof(fIn) != 0)
  {
    printf("PsuadeConfig ERROR:: keyword PSUADE_CONFIG not found.\n");
    fclose(fIn);
    exit(1);
  }
  //**/ ----------------------------------------------------------------- 
  //**/  check whether end keyword found
  //**/ ----------------------------------------------------------------- 
  lineCnt = 0;
  while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
  {
    sscanf(lineIn,"%s", winput);
    if   (strcmp(winput, "PSUADE_END") == 0) break;
    else lineCnt++;
  }
  if (strcmp(winput, "PSUADE_END") != 0)
  {
    printf("PsuadeConfig ERROR:: keyword PSUADE_END not found.\n");
    fclose(fIn);
    exit(1);
  }
  fclose(fIn);

  //**/ ----------------------------------------------------------------- 
  //**/  read lines into buffer
  //**/ ----------------------------------------------------------------- 
  nLinesUsed_ = 0;
  int nLines = 1000; 
  if (lineCnt > nLines) nLines = lineCnt + 1000;
  StrFileData_.setNumStrings(nLines);
  strcpy(winput, "NONE");
  for (ii = 0; ii < nLines; ii++)
    StrFileData_.loadOneString(ii, winput);

  if (lineCnt == 0) return;
  else
  {
    fIn = fopen(fname, "r");
    while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
    {
      sscanf(lineIn,"%s", winput);
      if (strcmp(winput, "PSUADE_CONFIG") == 0) break;
    }
    for (ii = 0; ii < lineCnt; ii++)
    {
      fgets(winput, lineLeng, fIn);
      StrFileData_.loadOneString(ii, winput);
    }
    nLinesUsed_ = lineCnt;
    fclose(fIn);
  }
  printLevel_ = printLevel;
}

// ************************************************************************
// reset
// ------------------------------------------------------------------------ 
void PsuadeConfig::reset()
{
  nLinesUsed_ = 0;
  StrFileData_.clean();
  //**/ print level
  printLevel_ = 0;
  //**/ turn on/off I/O expert mode 
  IOExpertMode_ = 0;
  //**/ turn on/off response surface expert mode 
  RSExpertMode_ = 0;
  RSExpertSave_ = -1;
  //**/ turn on/off sampling expert mode
  SamExpertMode_ = 0;
  SamExpertSave_ = -1;
  //**/ turn on/off analysis expert mode
  AnaExpertMode_ = 0;
  AnaExpertSave_ = -1;
  //**/ turn on/off optimization expert mode
  OptExpertMode_ = 0;
  //**/ turn on/off probability distribution diagnostics mode
  PDFDiagMode_ = 0;
  //**/ turn on/off stand-alone response surface interpolation code generator
  RSCodeGen_ = 0;
  RSCodeGenSave_ = -1;
  //**/ turn on/off master mode
  MasterMode_ = 0;
  //**/ turn on/off interactive mode (deprecated)
  //**/ interactive should not be touched (6/2020)
  //Interactive_ = 0;
  InteractiveSave_ = -1;
  //**/ (NOT USED)
  OUULibMode_ = 1;
  //**/ turn on/off grandmaster mode
  GMMode_ = 0;
  //**/ switch between Matlab/Scilab (default = 0 ==> matlab)
  PlotTool_ = 0;
  //**/ user can override internally generated random seed
  RandomSeed_ = -1;
  //**/ maximum of sample point allowed for RS (warning for excessive time)
  RSMaxPts_ = 10000;
  //**/ control for how to handle multiple RS constraints (intersect or union)
  RSConstraintSetOp_ = 0;
  //**/ overall diagnostics
  Diagnostics_ = 0;
}

// ************************************************************************
// request data from this object 
// ------------------------------------------------------------------------ 
char *PsuadeConfig::getParameter(const char *keyword)
{ 
  int  ii;
  char firstWord[100];

  for (ii = 0; ii < nLinesUsed_; ii++)
  {
    sscanf(StrFileData_.getOneString(ii), "%s", firstWord);
    if (strcmp(keyword,firstWord) == 0)
    {
      if (printLevel_ > 1)
        printf("PsuadeConfig::parameter found: %s\n",
               StrFileData_.getOneString(ii));
      return StrFileData_.getOneString(ii);
    }
  }
  return NULL;
}

// ************************************************************************
// add new data to this object 
// ------------------------------------------------------------------------ 
void PsuadeConfig::putParameter(const char *putLine)
{ 
  int ii, leng;
  char keyword[1000], firstWord[1000], pString[1000];

  if (putLine == NULL) return;
  //**/ error checking
  if (nLinesUsed_ >= StrFileData_.numStrings())
    StrFileData_.addMoreStrings(1000);

  //**/ see if the parameter has been set - if so, replace
  sscanf(putLine, "%s", keyword);
  for (ii = 0; ii < nLinesUsed_; ii++)
  {
    sscanf(StrFileData_.getOneString(ii), "%s", firstWord);
    if (strcmp(keyword,firstWord) == 0)
    {
      leng = strlen(putLine);
      strncpy(StrFileData_.getOneString(ii), putLine, leng);
      break;
    }
  }
  //**/ if not, add a new line
  if (ii == nLinesUsed_)
  {
    leng = strlen(putLine);
    StrFileData_.loadOneString(nLinesUsed_, putLine);
    nLinesUsed_++;
  }
  return;
}

// ************************************************************************
// remove data from this object 
// ------------------------------------------------------------------------ 
int PsuadeConfig::removeParameter(const char *keyword)
{ 
  int  ii, jj, index;
  char firstWord[100];

  ii = 0;
  while (ii < nLinesUsed_)
  {
    sscanf(StrFileData_.getOneString(ii), "%s", firstWord);
    if (strcmp(keyword,firstWord) == 0) 
    {
      for (jj = ii+1; jj < nLinesUsed_; jj++)
        StrFileData_.loadOneString(jj-1,StrFileData_.getOneString(jj));
      nLinesUsed_--;
    }
    else ii++;
  }
  return 0;
}

// ************************************************************************
// remove data from this object based on substring match
// ------------------------------------------------------------------------ 
int PsuadeConfig::removeParameter2(const char *keyword)
{ 
  int  ii, jj, index, leng;
  char firstWord[100];

  leng = strlen(keyword);
  ii = 0;
  while (ii < nLinesUsed_)
  {
    sscanf(StrFileData_.getOneString(ii), "%s", firstWord);
    if (strncmp(keyword,firstWord,leng) == 0) 
    {
      for (jj = ii+1; jj < nLinesUsed_; jj++)
        StrFileData_.loadOneString(jj-1,StrFileData_.getOneString(jj));
      nLinesUsed_--;
    }
    else ii++;
  }
  return 0;
}

// ************************************************************************
// write to file 
// ------------------------------------------------------------------------ 
void PsuadeConfig::writeToFile(char *fname)
{
  int  ii;
  FILE *fOut;

  //**/ ----------------------------------------------------------------- 
  //**/  check whether file exists
  //**/ ----------------------------------------------------------------- 
  fOut = fopen(fname, "w");
  if (fOut == NULL) 
  {
    printf("PsuadeConfig ERROR:: cannot write to configure file %s.\n",
           fname);
    exit(1);
  }

  //**/ ----------------------------------------------------------------- 
  //**/  check for errors
  //**/ ----------------------------------------------------------------- 
  fprintf(fOut, "PSUADE_CONFIG\n");
  for (ii = 0; ii < nLinesUsed_; ii++) 
    fprintf(fOut, "%s\n", StrFileData_.getOneString(ii));
  fprintf(fOut, "PSUADE_END\n");
  fclose(fOut);
}

// ************************************************************************
// print content of the config object
// ------------------------------------------------------------------------ 
void PsuadeConfig::print()
{
  printAsterisks(PL_INFO, 0);
  printf("************* PSUADE configuration information\n");
  printEquals(PL_INFO, 0);
  for (int ii = 0; ii < nLinesUsed_; ii++)
  {
    if (strcmp(StrFileData_.getOneString(ii), "NONE")) 
      printf("%5d: %s\n", ii+1, StrFileData_.getOneString(ii));
  }
  printAsterisks(PL_INFO, 0);
}

// ************************************************************************
// add information from a configure file
// ------------------------------------------------------------------------ 
void PsuadeConfig::addFromFile(char *fname)
{
  int  ii, lineLeng=500, lineCnt;
  char lineIn[501], winput[501];
  FILE *fIn;

  //**/ ----------------------------------------------------------------- 
  //**/  check whether file exists
  //**/ ----------------------------------------------------------------- 
  fIn = fopen(fname, "r");
  if (fIn == NULL) 
  {
    printf("PsuadeConfig ERROR:: configure file %s not found.\n",fname);
    exit(1);
  }

  //**/ ----------------------------------------------------------------- 
  //**/  check whether keyword found
  //**/ ----------------------------------------------------------------- 
  while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
  {
    sscanf(lineIn, "%s", winput);
    if (strcmp(winput, "PSUADE_CONFIG") == 0) break;
  }
  if (feof(fIn) != 0)
  {
    printf("PsuadeConfig ERROR:: keyword PSUADE_CONFIG not found.\n");
    fclose(fIn);
    exit(1);
  }
  //**/ ----------------------------------------------------------------- 
  //**/  check whether end keyword found
  //**/ ----------------------------------------------------------------- 
  lineCnt = 0;
  while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
  {
    sscanf(lineIn,"%s", winput);
    if   (strcmp(winput, "PSUADE_END") == 0) break;
    else lineCnt++;
  }
  if (strcmp(winput, "PSUADE_END") != 0)
  {
    printf("PsuadeConfig ERROR:: keyword PSUADE_END not found.\n");
    fclose(fIn);
    exit(1);
  }
  fclose(fIn);

  //**/ ----------------------------------------------------------------- 
  //**/  read lines into buffer
  //**/ ----------------------------------------------------------------- 
  if (lineCnt+nLinesUsed_ > StrFileData_.numStrings()) 
  {
    int nLines = StrFileData_.numStrings();
    StrFileData_.addMoreStrings(lineCnt+1000);
    strcpy(winput, "NONE");
    for (ii = nLines; ii < nLines+lineCnt+1000; ii++) 
      StrFileData_.loadOneString(ii, winput);
  }
  if (lineCnt == 0) return;
  else
  {
    fIn = fopen(fname, "r");
    while ((fgets(lineIn, lineLeng, fIn) != NULL) && (feof(fIn) == 0))
    {
      sscanf(lineIn,"%s", winput);
      if (strcmp(winput, "PSUADE_CONFIG") == 0) break;
    }
    for (ii = 0; ii < lineCnt; ii++)
    {
      fgets(winput, lineLeng, fIn);
      StrFileData_.loadOneString(nLinesUsed_+ii, winput);
    }
    nLinesUsed_ += lineCnt;
    fclose(fIn);
  }
}

// ************************************************************************
// set RSExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSExpertModeOn()
{
  RSExpertMode_ = 1;
}

// ************************************************************************
// turn off RSExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSExpertModeOff()
{
  RSExpertMode_ = 0;
}

// ************************************************************************
// check to see if RSExpert mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::RSExpertModeIsOn()
{
  if (RSExpertMode_ == 0) return false;
  else                    return true;
}

// ************************************************************************
// save RSExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSExpertModeSaveAndReset()
{
  if (printLevel_ > 4) 
    printf("RSExpertModeSaveAndReset %d\n",RSExpertMode_);
  if (RSExpertSave_ != -1)
    printf("RSExpertModeSaveAndReset WARNING: this has been called once.\n");
  RSExpertSave_ = RSExpertMode_;
  RSExpertMode_ = 0;
}

// ************************************************************************
// restore RSExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSExpertModeRestore()
{
  RSExpertMode_ = RSExpertSave_;
  RSExpertSave_ = -1;
  if (printLevel_ > 4) 
    printf("RSExpertModeRestore %d\n",RSExpertMode_);
}

// ************************************************************************
// set AnaExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::AnaExpertModeOn()
{
  AnaExpertMode_ = 1;
}

// ************************************************************************
// turn off AnaExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::AnaExpertModeOff()
{
  AnaExpertMode_ = 0;
}

// ************************************************************************
// check to see if AnaExpert mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::AnaExpertModeIsOn()
{
  if (AnaExpertMode_ == 0) return false;
  else                     return true;
}

// ************************************************************************
// save AnaExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::AnaExpertModeSaveAndReset()
{
  if (printLevel_ > 4) 
    printf("AnaExpertModeSaveAndReset %d\n",AnaExpertMode_);
  if (AnaExpertSave_ != -1)
    printf("AnaExpertModeSaveAndReset WARNING: this has been called once.\n");
  AnaExpertSave_ = AnaExpertMode_;
  AnaExpertMode_ = 0;
}

// ************************************************************************
// set IOExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::IOExpertModeOn()
{
  IOExpertMode_ = 1;
}

// ************************************************************************
// turn off IOExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::IOExpertModeOff()
{
  IOExpertMode_ = 0;
}

// ************************************************************************
// check to see if IOExpert mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::IOExpertModeIsOn()
{
  if (IOExpertMode_ == 0) return false;
  else                    return true;
}

// ************************************************************************
// restore AnaExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::AnaExpertModeRestore()
{
  AnaExpertMode_ = AnaExpertSave_;
  AnaExpertSave_ = -1;
  if (printLevel_ > 4) 
    printf("AnaExpertModeRestore %d\n",AnaExpertMode_);
}

// ************************************************************************
// set SamExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::SamExpertModeOn()
{
  SamExpertMode_ = 1;
}

// ************************************************************************
// turn off SamExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::SamExpertModeOff()
{
  SamExpertMode_ = 0;
}

// ************************************************************************
// check to see if SamExpert mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::SamExpertModeIsOn()
{
  if (SamExpertMode_ == 0) return false;
  else                     return true;
}

// ************************************************************************
// save SamExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::SamExpertModeSaveAndReset()
{
  if (printLevel_ > 4) 
    printf("SamExpertModeSaveAndReset %d\n",SamExpertMode_);
  if (SamExpertSave_ != -1)
    printf("SamExpertModeSaveAndReset WARNING: this has been called once.\n");
  SamExpertSave_ = SamExpertMode_;
  SamExpertMode_ = 0;
}

// ************************************************************************
// restore SamExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::SamExpertModeRestore()
{
  SamExpertMode_ = SamExpertSave_;
  SamExpertSave_ = -1;
}

// ************************************************************************
// set OptExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::OptExpertModeOn()
{
  OptExpertMode_ = 1;
}

// ************************************************************************
// turn off OptExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::OptExpertModeOff()
{
  OptExpertMode_ = 0;
}

// ************************************************************************
// check to see if OptExpert mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::OptExpertModeIsOn()
{
  if (OptExpertMode_ == 0) return false;
  else                     return true;
}

// ************************************************************************
// set Interactive mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::InteractiveOn()
{
  Interactive_ = 1;
}

// ************************************************************************
// turn off Interactive mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::InteractiveOff()
{
  Interactive_ = 0;
}

// ************************************************************************
// check to see if Interactive mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::InteractiveIsOn()
{
  if (Interactive_ == 0) return false;
  else                   return true;
}

// ************************************************************************
// save Interactive mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::InteractiveSaveAndReset()
{
  InteractiveSave_ = Interactive_;
  Interactive_ = 0;
  if (printLevel_ > 4) 
    printf("InteractiveSaveAndReset %d\n",Interactive_);
}

// ************************************************************************
// restore Interactive mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::InteractiveRestore()
{
  Interactive_ = InteractiveSave_;
  InteractiveSave_ = -1;
  if (printLevel_ > 4) 
    printf("InteractiveRestore %d\n",Interactive_);
}

// ************************************************************************
// set master mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::MasterModeOn()
{
  MasterMode_ = 1;
}

// ************************************************************************
// turn off master mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::MasterModeOff()
{
  MasterMode_ = 0;
}

// ************************************************************************
// check to see if master mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::MasterModeIsOn()
{
  if (MasterMode_ == 0) return false;
  else                  return true;
}

// ************************************************************************
// set PDF diagnostics mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::PDFDiagnosticsOn()
{
  PDFDiagMode_ = 1;
}

// ************************************************************************
// turn off PDF diagnostics mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::PDFDiagnosticsOff()
{
  PDFDiagMode_ = 0;
}

// ************************************************************************
// check to see if PDF diagnostics mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::PDFDiagnosticsIsOn()
{
  if (PDFDiagMode_ == 0) return false;
  else                   return true;
}

// ************************************************************************
// set OUULib mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::OUULibModeOn()
{
  OUULibMode_ = 1;
}

// ************************************************************************
// turn off OUULib mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::OUULibModeOff()
{
  OUULibMode_ = 0;
}

// ************************************************************************
// check to see if OUULib mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::OUULibModeIsOn()
{
  if (OUULibMode_ == 0) return false;
  else                  return true;
}

// ************************************************************************
// set GM mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::GMModeOn()
{
  GMMode_ = 1;
}

// ************************************************************************
// turn off GM mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::GMModeOff()
{
  GMMode_ = 0;
}

// ************************************************************************
// check to see if GM mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::GMModeIsOn()
{
  if (GMMode_ == 0) return false;
  else              return true;
}

// ************************************************************************
// turn on diagnostics mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::DiagnosticsOn()
{
  Diagnostics_ = 1;
}

// ************************************************************************
// turn off diagnostics mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::DiagnosticsOff()
{
  Diagnostics_ = 0;
}

// ************************************************************************
// check to see if diagnostics mode is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::DiagnosticsIsOn()
{
  if (Diagnostics_ == 0) return false;
  else                   return true;
}

// ************************************************************************
// set random seed
// ------------------------------------------------------------------------ 
void PsuadeConfig::setRandomSeed(long rseed)
{
  RandomSeed_ = rseed;
}

// ************************************************************************
// get random seed
// ------------------------------------------------------------------------ 
long PsuadeConfig::getRandomSeed()
{
  return RandomSeed_;
}

// ************************************************************************
// set RS code gen flag 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSCodeGenOn()
{
  RSCodeGen_ = 1;
}

// ************************************************************************
// turn on RS code gen flag 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSCodeGenOff()
{
  RSCodeGen_ = 0;
}

// ************************************************************************
// check to see if RS code gen flag is on
// ------------------------------------------------------------------------ 
bool PsuadeConfig::RSCodeGenIsOn()
{
  if (RSCodeGen_ == 0) return false;
  else                 return true;
}

// ************************************************************************
// save RSCodeGen flag
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSCodeGenSaveAndReset()
{
  RSCodeGenSave_ = RSCodeGen_;
  RSCodeGen_ = 0;
}

// ************************************************************************
// restore SamExpert mode 
// ------------------------------------------------------------------------ 
void PsuadeConfig::RSCodeGenRestore()
{
  RSCodeGen_ = RSCodeGenSave_;
  RSCodeGenSave_ = 0;
}

// ************************************************************************
// friend function (create a function approximator given a file name)
// perform PDF transformation
// check invalid sample points
// RS type from file 
// ------------------------------------------------------------------------
extern "C"
int genConfigFileTemplate(char *fname)
{
  FILE *fp;
  fp = fopen(fname, "w");
  if (fp != NULL)
  {
    fprintf(fp,"# Use this file if you need to set certain RS or analyis\n");
    fprintf(fp,"# non-default options that will require turning on expert\n");
    fprintf(fp,"# modes which may ask users to enter options repeatedly\n");
    fprintf(fp,"# thus make life cumbersom. Current options are listed in\n");
    fprintf(fp,"# the following:\n");
    fprintf(fp,"# * Uncomment and add information to activate them.\n");
    fprintf(fp,"# * To use this file in the batch mode, add the following\n");
    fprintf(fp,"#   line in ANALYSIS section\n");
    fprintf(fp,"#       use_configure_file = <this file name>\n");
    fprintf(fp,"PSUADE_CONFIG\n");
    fprintf(fp,"## Normalize inputs for some response surface methods\n");
    fprintf(fp,"#normalize_input (take out the # to turn on)\n");
    fprintf(fp,"## Normalize outputs for some response surface methods\n");
    fprintf(fp,"#normalize_output (take out the # to turn on)\n");
    fprintf(fp,"## MARS parameters (take out the # to turn on)\n");
    fprintf(fp,"#RS_MARS_num_bases = 50\n");
    fprintf(fp,"#RS_MARS_interaction = 2\n");
    fprintf(fp,"## SVM parameters (remove # to turn on)\n");
    fprintf(fp,"#RS_SVM_gamma = 1.0 (1e-6 - 1.0)\n");
    fprintf(fp,"#RS_SVM_tol = 1.0 (1e-6 - 1.0)\n");
    fprintf(fp,"#RS_SVM_kernel = 1 (1:linear, 2:cubic, 3:RBF, 4:sigmoid)\n");
    fprintf(fp,"## RBF parameters (remove # to turn on)\n");
    fprintf(fp,"#RS_RBF_kernel = 0 (0/1/2/3:multi-quad, invM-Q, Gauss, spline)\n");
    fprintf(fp,"#RS_RBF_scale = x (Gaussian scale)\n");
    fprintf(fp,"#RS_RBF_thresh = x (SVD threshold)\n");
    fprintf(fp,"## Kriging parameters (remove # to turn on)\n");
    fprintf(fp,"#RS_KRI_mode = 2\n");
    fprintf(fp,"#RS_KRI_tol = 1.0e-6\n");
    fprintf(fp,"#RS_KRI_DATA_STDEV_FILE = <add a file here>\n");
    fprintf(fp,"#RS_KRI_LENG_SCALE 1 = 0.1\n");
    fprintf(fp,"#RS_KRI_LENG_SCALE 2 = 0.1\n");
    fprintf(fp,"## Legendre parameters (remove # to turn on)\n");
    fprintf(fp,"#RS_Legendre_order = 1\n");
    fprintf(fp,"## RBF,GP override - no multi-domain for large sample\n");
    fprintf(fp,"RS_no_multi_domain\n");
    fprintf(fp,"## multi-domain response surface method parameters\n");
    fprintf(fp,"RS_MRBF_max_sam_per_group = 1000\n");
    fprintf(fp,"RS_MGP_max_sam_per_group = 1000\n");
    fprintf(fp,"## MOAT parameter (number of levels)\n");
    fprintf(fp,"#MOAT_P = 4\n");
    fprintf(fp,"## GMOAT parameter (number of levels)\n");
    fprintf(fp,"#GMOAT_P = 4\n");
    fprintf(fp,"## RS-based Sobol index parameters\n");
    fprintf(fp,"#RSMSobol1_nsubsamples = 1000\n");
    fprintf(fp,"#RSMSobol1_nlevels = 200\n");
    fprintf(fp,"#RSMSobol2_nsubsamples = 1000\n");
    fprintf(fp,"#RSMSobol2_nlevels = 200\n");
    fprintf(fp,"#RSMSoboltsi_nsubsamples = 1000\n");
    fprintf(fp,"#RSMSoboltsi_nlevels = 100\n");
    fprintf(fp,"#RSMSoboltG_nsubsamples_ingroup = 500\n");
    fprintf(fp,"#RSMSoboltG_nsubsamples_outgroup = 2000\n");
    //**/ probably not needed (Sept 2013)
    //**/fprintf(fp, "#RSFA_cross_validation\n");
    //**/fprintf(fp, "#RSFA_cv_ngroups = 10\n");
    //**/fprintf(fp, "#RSFA_test_datafile = testfile\n");
    //**/fprintf(fp, "#RSFA_test_errorfile = errfile\n");
    //**/fprintf(fp, "#RSMSobol1_pdffile = pdffile\n");
    //**/fprintf(fp, "#RSMSobol1_scatterfile = file\n");
    //**/fprintf(fp, "#MOAT_repair_file = file\n");
    //**/fprintf(fp, "#MOAT_partition_file = file\n");
    //**/fprintf(fp, "#GMOAT_partition_file = file\n");
    //**/fprintf(fp, "#METIS_expand_ratio = 0.1\n");
    fprintf(fp, "PSUADE_END\n");
    fclose(fp);
    return 0;
  }
  else return -1;
}


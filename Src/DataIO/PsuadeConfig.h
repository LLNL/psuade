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
// Definition for the class PsuadeConfig
// AUTHOR : CHARLES TONG
// DATE   : 2006
// ************************************************************************
#ifndef __PSUADECONFIGH__
#define __PSUADECONFIGH__

#include "psMatrix.h"
#include "psStrings.h"

// ************************************************************************
// ************************************************************************
// main class definition 
// ************************************************************************
// ************************************************************************

class PsuadeConfig 
{
  int  printLevel_;
  int  nLinesUsed_;
  psStrings StrFileData_;

public:
  int  RSExpertMode_;
  int  RSExpertSave_;
  int  AnaExpertMode_;
  int  AnaExpertSave_;
  int  IOExpertMode_;
  int  SamExpertMode_;
  int  SamExpertSave_;
  int  OptExpertMode_;
  int  Interactive_;
  int  InteractiveSave_;
  int  PDFDiagMode_;
  int  MasterMode_;
  int  OUULibMode_;
  int  GMMode_;
  long RandomSeed_;
  int  RSMaxPts_;        // max no. of points for response surface
  int  PlotTool_;        // 0 - matlab, 1 - scilab
  int  RSCodeGen_;
  int  RSCodeGenSave_;
  int  RSConstraintSetOp_;
  int  Diagnostics_;
  psMatrix MatCommonUse_;
  psVector VecCommonUse_;
  int  intCommonUse_;

public:

  // Constructor
  PsuadeConfig();

  // Copy Contructor by Bill Oliver
  PsuadeConfig(const PsuadeConfig & ps);

  // Copy Contructor by Bill Oliver
  PsuadeConfig & operator=(const PsuadeConfig & ps);

  // Destructor
  ~PsuadeConfig();

  // Initialize
  // fname : configure file name
  // printLevel : diagnostics print level
  void initialize(char *fname, int printLevel);

  // Get parameter 
  void reset();

  // Get parameter 
  char *getParameter(const char *);

  // put parameter 
  void putParameter(const char *);

  // remove parameter 
  int removeParameter(const char *);

  // remove parameter based on substring match
  int removeParameter2(const char *);

  // write file
  // fname : configure file name
  void writeToFile(char *fname);

  // add from file
  // fname : configure file name
  void addFromFile(char *fname);

  // print 
  void print();

  // RSExpertMode
  bool RSExpertModeIsOn();
  void RSExpertModeOn();
  void RSExpertModeOff();
  void RSExpertModeSaveAndReset();
  void RSExpertModeRestore();

  // AnaExpertMode
  bool AnaExpertModeIsOn();
  void AnaExpertModeOn();
  void AnaExpertModeOff();
  void AnaExpertModeSaveAndReset();
  void AnaExpertModeRestore();

  // IOExpertMode
  bool IOExpertModeIsOn();
  void IOExpertModeOn();
  void IOExpertModeOff();

  // SamExpertMode
  bool SamExpertModeIsOn();
  void SamExpertModeOn();
  void SamExpertModeOff();
  void SamExpertModeSaveAndReset();
  void SamExpertModeRestore();

  // OptExpertMode
  bool OptExpertModeIsOn();
  void OptExpertModeOn();
  void OptExpertModeOff();

  // Interactive
  bool InteractiveIsOn();
  void InteractiveOn();
  void InteractiveOff();
  void InteractiveSaveAndReset();
  void InteractiveRestore();

  // Master mode
  bool MasterModeIsOn();
  void MasterModeOn();
  void MasterModeOff();

  // PDF diagnostics mode
  bool PDFDiagnosticsIsOn();
  void PDFDiagnosticsOn();
  void PDFDiagnosticsOff();

  // OUULib mode
  bool OUULibModeIsOn();
  void OUULibModeOn();
  void OUULibModeOff();

  // GM mode
  bool GMModeIsOn();
  void GMModeOn();
  void GMModeOff();

  // diagnostics mode
  bool DiagnosticsIsOn();
  void DiagnosticsOn();
  void DiagnosticsOff();

  // random seed
  void setRandomSeed(long);
  long getRandomSeed();

  // RS CodeGen flag 
  bool RSCodeGenIsOn();
  void RSCodeGenOn();
  void RSCodeGenOff();
  void RSCodeGenSaveAndReset();
  void RSCodeGenRestore();
};

// ************************************************************************
// friend function
// ------------------------------------------------------------------------
extern "C"
{
  int genConfigFileTemplate(char *filename);
}

#endif // __PSUADECONFIGH__


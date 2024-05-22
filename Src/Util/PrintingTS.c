/* ************************************************************************
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
// Utility routines for printing output to stdout, stderr, and log files.
//=======================================================================*/
#include "PrintingTS.h"

#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/time.h>
#include <time.h>
#include <stdio.h>
#include <sys/stat.h>
#include <dirent.h>
#include <libgen.h>

/*=========================================================================
  The overall print level we're running at 
 *=======================================================================*/
static int s_printLevel = 1;

/*=========================================================================
  Do thread safe printing? 
 *=======================================================================*/
static int doPrintTS = 1;

/*=========================================================================
  Do thread safe Logging printing? 
 *=======================================================================*/
static int doLogTS = 0;

/*=========================================================================
  MPI rank of symponent process. 
 *=======================================================================*/
static int localRank = -1;

/*=========================================================================
  Log file pointer. 
 *=======================================================================*/
static FILE* fpLog;

/*=========================================================================
  Lock that serializes output. 
 *=======================================================================*/
//static pthread_mutex_t printLock = PTHREAD_MUTEX_INITIALIZER;

/* ************************************************************************
  initializePrintingTS
  Create directory to hold log files if needed; open log file.
  
  This routine recursively removes a logging directory if it already exists
  and then creates a new empty top level logging directory with the same
  name.  It also constructs the log file name and opens it.
  
  doPrintTS gets set to 1/True.
  
  \param[in] printLevel  : The level of output to display and log.
  \param[in] logFileName : Log file name.
  \param[in] myRank      : Rank that is given.
  -----------------------------------------------------------------------*/
void
initializePrintingTS(int printLevel, const char* logFileName, int myRank)
{
  char commandString[128];
  char logFileNameFull[256];

  // pthread_mutex_lock(&printLock);

  localRank = myRank;

  s_printLevel = printLevel;

  if (logFileName) 
  {
    sprintf(logFileNameFull, "%s.%05d", logFileName, myRank);
    fpLog = fopen(logFileNameFull, "w");
  }

  if (fpLog != NULL) doLogTS = 1;

  //  pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
   finalizePrintingTS: Closes the log file.
  -----------------------------------------------------------------------*/
void finalizePrintingTS(void)
{
  // pthread_mutex_lock(&printLock);

  doPrintTS = 0;
  if (fpLog != NULL) { fclose(fpLog); fpLog = NULL; }

  //  pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
   printErrTS
   Prints error message to log file & stderr.
   \param[in] format : Text to print.
  -----------------------------------------------------------------------*/
void printErrTS(int printLevel, const char* format, ...)
{
  char buffer[4096];
  va_list args;

  // pthread_mutex_lock(&printLock);

  if(printLevel <= s_printLevel) 
  {
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);
    if(doLogTS) 
    {
      fputs(buffer, fpLog);
      fflush(fpLog);
    }
    if (doPrintTS) 
    {    
      fputs(buffer, stderr);
      fflush(stderr);
    }    
    va_end(args);
  }

  // pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 * printLogTS: Prints log message to log file.
 * \param[in] format : Text to print.
  -----------------------------------------------------------------------*/
void printLogTS(int printLevel, const char* format, ...)
{
  char buffer[4096];
  va_list args;

  // pthread_mutex_lock(&printLock);

  if (printLevel <= s_printLevel && doLogTS)
  {
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);

    fputs(buffer, fpLog);
    fflush(fpLog);

    va_end(args);
  }
 
  //  pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 * printOutTS: Prints message to log file & stdout.
 * \param[in] format : Text to print.
 *-----------------------------------------------------------------------*/
void printOutTS(int printLevel, const char* format, ...)
{
  char buffer[4096];
  va_list args;

  // pthread_mutex_lock(&printLock);

  if (printLevel <= s_printLevel) 
  {
    va_start(args, format);
    vsnprintf(buffer, sizeof(buffer), format, args);

    if(doLogTS) 
    {
      fputs(buffer, fpLog);
      fflush(fpLog);
    }

    if (doPrintTS) 
    {
      fputs(buffer, stdout);
      fflush(stdout);
    }
    va_end(args);
  }
  // pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 *  isPrintTSOn: Is thread safe printing on?
 *-----------------------------------------------------------------------*/
int isPrintTSOn(void)
{
  /* I think we can get away without locking here. */
  return doPrintTS;
}

/* ************************************************************************
 *  turnPrintTSOff: Turn thread safe printing off.
 *-----------------------------------------------------------------------*/
void turnPrintTSOff(void)
{
  // pthread_mutex_lock(&printLock);

  doPrintTS = 0;

  // pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 *  turnPrintTSOn: Turn thread safe printing on.
 *-----------------------------------------------------------------------*/
void turnPrintTSOn(void)
{
  // pthread_mutex_lock(&printLock);

  doPrintTS = 1;

  // pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 *  isLogTSOn: Is thread safe printing on?
 *-----------------------------------------------------------------------*/
int isLogTSOn(void)
{
  /* I think we can get away without locking here. */

  return doLogTS;
}

/* ************************************************************************
 *  turnLogTSOff: Turn thread safe printing off.
 *-----------------------------------------------------------------------*/
void turnLogTSOff(void)
{
  // pthread_mutex_lock(&printLock);

  doLogTS = 0;

  // pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 *  turnLogTSOn: Turn thread safe printing on.
 *-----------------------------------------------------------------------*/
void turnLogTSOn(void)
{
  // pthread_mutex_lock(&printLock);

  doLogTS = 1;

  // pthread_mutex_unlock(&printLock);
}

/* ************************************************************************
 *  getPrintLevel: Return the current printLevel (See description in 
 *                 initilaizePrintingTS)
 *-----------------------------------------------------------------------*/
int getPrintLevelTS(void)
{
  /* I think we can get away without locking here. */

  return s_printLevel;
}

/* ************************************************************************
 *  setPrintLevel: Set the printlevel, usually done at initialization. 
 *                 Default is 0.
 *-----------------------------------------------------------------------*/
void setPrintLevelTS(int printLevel)
{
  // pthread_mutex_lock(&printLock);

  s_printLevel = printLevel;

  // pthread_mutex_unlock(&printLock);
}


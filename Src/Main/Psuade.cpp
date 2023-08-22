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
// This is the main program for experimenting with different samplings
// AUTHOR : CHARLES TONG
// DATE   : 2003
//**/----------------------------------------------------------------------
//**/ Feb 2013: add psuade --info option
//**/----------------------------------------------------------------------
// ************************************************************************

// -------------------------------------------------------------------------
// internal and external library declarations
// -------------------------------------------------------------------------
#ifdef HAVE_MPICH
#include <mpi.h>
#endif
#include <math.h>
#include <unistd.h>
#include <stdio.h>
#include <iostream>
#include <string>
//**/#include <time.h>
using namespace std;
#include "PsuadeBase.h"
#include "CommManager.h"
#include "PsuadeCmakeConfig.h"
#include "PsuadeData.h"
#include "dtype.h"
#include "Exceptions.h"
#include "PsuadeUtil.h"
#include "PrintingTS.h"
#include "Globals.h"
#include "PsuadeConfig.h"

// ************************************************************************
//**/ The following segment of code is included to rid of the complaint on 
//**/ Sun4 about an undefined _MAIN_ using -lg++ and -lF77 load flags - 
//**/ should be changed if fixed
// ------------------------------------------------------------------------
#ifdef SUN4
extern "C" 
{
  void MAIN_(int argc, void** argv) { }
}
#endif
// ************************************************************************
// MPI-related set up
// ------------------------------------------------------------------------
extern "C" 
{
  int RunParallel(const char *);
  int RunParallelLocal(const char *);
}

#ifdef  HAVE_MPICH
#ifndef HAVE_MPI_USR_FCN
extern "C" 
{
  int UserFunction(MPI_Comm mpiComm, int index, char *workdir)
  {
    (void) mpiComm;
    (void) index;
    (void) workdir;
    return -1;
  }
}
#endif
#endif

// ************************************************************************
//  Options
//
//! \brief A data structure that encapsulates all command line options.
//!
//! The options object takes care of parsing the command line options.  It 
//! assumes default values, does basic error checking and prints a usage 
//! message if anything weird is found.  Lifted from the Coop project. -Jim 
//----------------------------------------------------------------------------
struct Options
{
public:

  //! \brief Turn on verbose output [1].
  //------------------------------------
  int doVerbose;

  //! \brief parallel Mode (MPI)
  //----------------------------------------------
  int parallelMode;

  //! \brief local parallel Mode (MPI)
  //----------------------------------------------
  int localParallelMode;

  //! \brief Input file name (positional argument)
  //----------------------------------------------
  string inputFilename;

  //! \brief Output file name (-o)
  //----------------------------------------------
  string outputFilename;

  //! \brief AKA argv[0]; path to the psuade executable.
  //-------------------------------------------------------
  string programName;

  //--------------------------------------------------------------------------
  //  Options
  //
  //! \brief Default constructor; set everything to reasonable defaults.
  //--------------------------------------------------------------------------
  Options(void) : doVerbose(0), parallelMode(0), localParallelMode(0), 
      inputFilename(""), outputFilename("psuadeData"), programName("psuade")
  { }

  //--------------------------------------------------------------------------
  //  Options
  //
  //! \brief This constructor loads defaults & command line options.
  //!
  //! \param[in] argc : Number of arguments.
  //! \param[in] argv : The actual arguments.
  //--------------------------------------------------------------------------
  Options(int argc, char** argv) : doVerbose(0), parallelMode(0), 
      localParallelMode(0), inputFilename(""), outputFilename("psuadeData"), 
      programName("psuade")
  {
    programName = string(argv[0]);
    bool userSetOutputName = false;
    bool userSetInputName = false;

    const string HELP        = "--help";
    const string HELP_S      = "-h";
    const string VERBOSE     = "--verbose";
    const string VERBOSE_S   = "-v";
    const string INFO        = "--info";
    const string INFO_S      = "-i";
    const string PARALLEL    = "--parallel";
    const string PARALLEL_S  = "-mp";
    const string L_PARALLEL  = "--local_parallel";
    const string L_PARALLEL_S= "-mpp";
    const string OUTPUTFILE  = "--output_file=";
    const string OUTPUTFILE_S= "-o";

    int argp = 1;
    int argn = argc;

    for (int ii = 1; ii < argc; ++ii)
    {
      argp = ii;
      argn = argc - ii;
      // Stop on first non-option. (Hopefully the filename)
      if (argv[argp][0] != '-') { break; }  

      if (argv[argp] == HELP || argv[argp] == HELP_S)
      {
        usage();
        exit(0);
      }
      else if (argv[argp] == INFO || argv[argp] == INFO_S)
      {
        info();
        exit(0);
      }
      else if (argv[argp] == VERBOSE || argv[argp] == VERBOSE_S)
      {
        doVerbose = 1;
      }
      else if (argv[argp] == PARALLEL || argv[argp] == PARALLEL_S)
      {
#ifdef HAVE_PARALLEL
        parallelMode = 1;
#else
        printf("ERROR: PARALLEL run requested with %s, but psuade was\n", 
               argv[argp]);
        printf("       not built with PARALLEL_BUILD.\n");
        exit(1);
#endif /*HAVE_PARALLEL*/
      }
      else if (argv[argp] == L_PARALLEL || argv[argp] == L_PARALLEL_S)
      {
#ifdef HAVE_PARALLEL
        localParallelMode = 1;
#else
        printf("ERROR: LOCAL PARALLEL run requested with %s, but psuade\n", 
               argv[argp]);
        printf("       was not built with PARALLEL_BUILD.\n");
        exit(1);
#endif /*HAVE_PARALLEL*/
      }
      else if (OUTPUTFILE.find(argv[argp], 0, OUTPUTFILE.size()) == 0)
      {
        outputFilename = argv[argp] + OUTPUTFILE.size();
        //**/The output filename is actually stored as a global...
        psOutputFilename_ = strdup(outputFilename.c_str());  
        userSetOutputName = true;
      }
      else if (OUTPUTFILE_S.find(argv[argp], 0, OUTPUTFILE_S.size()) == 0)
      {
        outputFilename = argv[argp] + OUTPUTFILE_S.size();
        //**/The output filename is actually stored as a global...
        psOutputFilename_ = strdup(outputFilename.c_str()); 
        userSetOutputName = true;
      }
      else
      {
        printf("Unknown option: %s\n", argv[argp]);
        usage();
        exit(1);
      }
      argp++;
    }
    //-----------------------------
    // Sanity checks
    //-----------------------------
    if (parallelMode && localParallelMode) 
    {
      printf("ERROR: cannot use both Parallel and Local Parallel Modes.\n");
      usage();
      exit(1);
    }
    
    //--------------------------------------------------------------------
    // Check for a inputfile to run.  If the options check consumed all of
    // the arguments, there are none left to run as a inputfile.
    //--------------------------------------------------------------------
    if (argp < argc)
    {
      inputFilename = argv[argp];
      //**/The global inputfile name is just used for sanity checking
      psInputFilename_ = strdup(inputFilename.c_str()); 
      userSetInputName = true;
      argp++;
    }

    //**/If we have too many positional command line args, barf and error out.
    if (argp < argc)  
    {
      printf("ERROR: Too many positional command line arguments!\n");;
      printf("       PSUADE only takes 1 argument: an input file %s.\n", 
             psInputFilename_);
      printf("       The following arguments are invalid\n");
      for (;argp < argc; argp++) printf("   %s\n",argv[argp]);
      printf("\n");
      usage();
      exit(1);
    }
  }
  // End Options(argc, argv).

  //--------------------------------------------------------------------------
  //  usage
  //
  //! \brief Basic usage and die routine.
  //!
  //! \return void
  //--------------------------------------------------------------------------
  void
  usage(void)
  {
    cerr << endl <<
    "Usage: " << endl <<
    "  " << programName << " [OPTIONS]... [INPUT FILE] \n" <<
    "  OPTIONS DESCRIPTION\n" 
    "  --help, -h         : Prints this message\n "
    "  --info, -i         : Prints information on this build of psuade\n"
    "  --verbose, -v      : Turn on verbose mode (support is spotty)\n"   
    "  --parallel, -mp    : Run PSUADE in parallel mode (with mpi)\n"
    "  --local_parallel, -mpp : Run PSUADE in local parallel mode\n"
    "  --output_file=, -o : Output file to write to, defaults to psuadeData\n"
    "\n"
    "  Normally PSUADE is run in the following modes: \n"  
    "  Normal      : psuade <--output_file=...> inputFile \n"
    "  Interactive : psuade\n"
    "  MP          : mpirun -np x psuade <--output_file=...> inputFile\n"
    "  MPP         : srun -N x -n y psuade <--output_file=...> inputFile\n" <<
    endl;
  }

  //--------------------------------------------------------------------------
  //  info
  //
  //! \brief Prints out information about this build of PSUADE.  
  //!        Version number and which modules are installed
  //!
  //! \return void
  //--------------------------------------------------------------------------
  void
  info(void)
  {
    printf("PSUADE version %d.%d.%d\n", psuade_VERSION_MAJOR, 
           psuade_VERSION_MINOR, psuade_VERSION_PATCH);

#ifdef HAVE_MARS
    printf("MARS     installed... true\n");
#endif
#ifdef HAVE_TPROS
    printf("TPROS    installed... true\n");
#endif
#ifdef HAVE_GPMC
    printf("GPMC     installed... true\n");
#endif
#ifdef HAVE_SVM
    printf("SVM      installed... true\n");
#endif
#ifdef HAVE_TGP
    printf("TGP      installed... true\n");
#endif
#ifdef HAVE_EARTH
    printf("EARTH    installed... true\n");
#endif
#ifdef HAVE_METIS
    printf("METIS    installed... true\n");
#endif
#ifdef HAVE_BOBYQA
    printf("BOBYQA   installed... true\n");
#endif
#ifdef HAVE_COBYLA
    printf("COBYLA   installed... true\n");
#endif
#ifdef HAVE_MINPACK
    printf("MINPACK  installed... true\n");
#endif
#ifdef HAVE_APPSPACK
    printf("APPSPACK installed... true\n");
#endif
#ifdef HAVE_TXMATH
    printf("TXMATH   installed... true\n");
#endif
#ifdef HAVE_PARALLEL
    printf("Parallel version\n");
#endif
#ifdef HAVE_MPICH
    printf("Use MPICH\n");
#endif
  }
}
;  // End struct Options.

// ************************************************************************
// Main driver
// ------------------------------------------------------------------------
int main(int argc, char** argv)
{
  int ind, status=0, mypid=0, nprocs=1, continueFlag=1, mode=0, root=0;
  PsuadeBase *psuade;

  //**/-----------------------------------------------------------
  //**/ optional tracking usage
  //**/----------------------------------------------------------- 
  //**/time_t     t;
  //**/struct tm  *tm;
  //**/char *inputString;
  //**/FILE *fp;
  //**/inputString = getenv("USER");
  //**/if (inputString != NULL)
  //**/{
  //**/  fp = fopen(".data", "a");
  //**/  if (fp != NULL)
  //**/  {
  //**/    t = time(NULL);
  //**/    tm = localtime(&t);
  //**/    fprintf(fp, "%s %s", inputString, asctime(tm));
  //**/    fclose(fp);
  //**/  }
  //**/}
  // --------------------------------------------------------------
  // if made for parallel processing, get machine parameters
  // -------------------------------------------------------------- 
  psCommMgr_ = new CommManager(argc, (void **) argv);
  mypid      = psCommMgr_->getPID();
  nprocs     = psCommMgr_->getNumProcs();

  // --------------------------------------------------------------
  // initialize print level and turn on interactive and screen dump
  // -------------------------------------------------------------- 
  initializePrintingTS(2, NULL, mypid);

  // -------------------------------------------------------------- 
  // initialize random number generator (may be initialized again
  // later if user specifies a random seed
  // -------------------------------------------------------------- 
  PSUADE_randInit(121019121);

  // --------------------------------------------------------------
  // instantiate configure object to accept options
  // -------------------------------------------------------------- 
  psConfig_.reset();
  psConfig_.InteractiveOn();

  // --------------------------------------------------------------
  // scan options
  // -------------------------------------------------------------- 
  Options opts(argc, argv);

  // --------------------------------------------------------------
  // if in parallel mode (if -mp is specified), run parallel jobs
  // -------------------------------------------------------------- 
  if (opts.parallelMode)
  {
    if(opts.inputFilename == "") 
    {
      psuade = new PsuadeBase();
      psuade->sessionInteractiveParallel();
    }
    else RunParallel(opts.inputFilename.c_str());
  } 
  else if (opts.localParallelMode) 
  {
    if(opts.inputFilename == "") 
    {
      printf("ERROR: local parallel mode requires and input file\n");
      opts.usage();
      exit(1);
    }
    RunParallelLocal(opts.inputFilename.c_str());
    continueFlag = 0;
  } 
  else 
  //**/ ----------------------------------------------------------------- 
  //**/ Normal serial Run
  //**/ ----------------------------------------------------------------- 
  { 
    printAsterisks(PL_INFO, 0);
    printf("*      Welcome to PSUADE (version %d.%d.%d)\n", 
           psuade_VERSION_MAJOR,psuade_VERSION_MINOR,psuade_VERSION_PATCH);

    printAsterisks(PL_INFO, 0);

    psuade = new PsuadeBase();
    try 
    {
      if (opts.inputFilename != "")
      {
        status = psuade->getInputFromFile(opts.inputFilename.c_str());
        if (status != 0) exit(status);
        psuade->run();
      } 
      else 
      {
        psuade->sessionInteractive();
      }
    }
    catch ( Psuade_Stop_Exception ) 
    {
      exit(1);
    }

    //**/ -------------------------------------------------------------- 
    //**/ clean up
    //**/ -------------------------------------------------------------- 
    delete psuade;
  }

  //**/ ----------------------------------------------------------------- 
  //**/ more clean up
  //**/ ----------------------------------------------------------------- 
  if (psCommMgr_ != NULL) delete psCommMgr_;
  char cStr[100];
  strcpy(cStr, ".psuade_mars");
  FILE *fp = fopen(cStr, "r");
  if (fp != NULL)
  {
    fclose(fp);
    unlink(cStr);
  }
  strcpy(cStr, ".psuade_gp3");
  fp = fopen(cStr, "r");
  if (fp != NULL)
  {
    fclose(fp);
    unlink(cStr);
  }
}

// ************************************************************************
// run jobs in MPI parallel mode
//**/ syntax: psuade -mp infile 
//**/         where infile should have format described in this function
// ------------------------------------------------------------------------
int RunParallel(const char *inFileName)
{
  int  mypid=0, nprocs=1, root=0, iOne=1, sampleID, lineLeng=200;
  int  nSamples=0, strLeng, *statusArray=NULL, nActive, index;
  int  procStep=1, hasArg=0, hasAux=0, hasExec=0, start=1, cch;
  const char *keywords[]={"workdir", "executable", "aux_exec", 
                          "num_samples", "proc_step", "argument", 
                          "PSUADE_PARALLEL","num_parallel","sample_start"}; 
  char lineIn[501], inString[500], inString2[500], auxExecutable[500];
  char workdir[500], executable[500], runLine[501], execArg[500];
  FILE *inFile, *argFile, *fp;
                                                                               
  mypid  = psCommMgr_->getPID();
  nprocs = psCommMgr_->getNumProcs();

  //**/ Read instructions from the input file
  if (mypid == 0)
  {
    inFile = fopen(inFileName, "r");
    if (inFile == NULL)
    {
      printf("RunParallel read ERROR - file %s = NULL\n",inFileName);
      exit(1);
    }
    fgets(lineIn, lineLeng, inFile);
    sscanf(lineIn, "%s", inString);
    if (strcmp(inString, "PSUADE_PARALLEL")) 
    {
      printf("RunParallel ERROR - keyword missing.\n");
      exit(1);
    }
    while ((fgets(lineIn,lineLeng,inFile) != NULL) && 
           ((cch=fgetc(inFile)) != EOF))
    {
      strcpy(inString, "#");
      sscanf(lineIn,"%s", inString);
      if (strcmp(inString, keywords[0]) == 0) // work directory 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %s", inString, inString2, workdir);
        else sscanf(lineIn, "%s %s", inString, workdir);
      }
      else if (strcmp(inString, keywords[1]) == 0) // executable 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %s", inString, inString2, executable);
        else sscanf(lineIn, "%s %s", inString, executable);
        fp = fopen(executable, "r");
        if (fp == NULL)
        {
          printf("RunParallel ERROR: executable %s not found.\n",
                 executable);
          strcpy(executable, "true");
        }
        else
        {
          fclose(fp);
          hasExec = 1;
        }
      }
      else if (strcmp(inString, keywords[2]) == 0) // aux executable 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0)
             sscanf(lineIn, "%s %s %s", inString, inString2, auxExecutable);
        else sscanf(lineIn, "%s %s", inString, auxExecutable);
        fp = fopen(auxExecutable, "r");
        if (fp == NULL)
        {
          printf("RunParallel ERROR: auxiliary executable %s not found\n",
                 auxExecutable);
          strcpy(auxExecutable, "true");
        }
        else
        {
          fclose(fp);
          hasAux = 1;
        }
      }
      else if (strcmp(inString, keywords[3]) == 0 || 
               strcmp(inString, keywords[7]) == 0) // deg of parallelism 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %d", inString, inString2, &nSamples);
        else sscanf(lineIn, "%s %d", inString, &nSamples);
        if (nSamples <= 0)
        {
          printf("RunParallel ERROR - nSamples %d <= 0.\n",nSamples);
          exit(1);
        }
      }
      else if (strcmp(inString, keywords[4]) == 0) // processor step 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %d", inString, inString2, &procStep);
        else sscanf(lineIn, "%s %d", inString, &procStep);
        if (procStep <= 0)
        {
          printf("RunParallel ERROR - procStep %d <= 0.\n",procStep);
          exit(1);
        }
      }
      else if (strcmp(inString, keywords[5]) == 0) // executable argument
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %s", inString, inString2, execArg);
        else sscanf(lineIn, "%s %s", inString, execArg);
        fp = fopen(execArg, "r");
        if (fp == NULL)
        {
          printf("RunParallel ERROR: argument file %s not found.\n",
                 execArg);
          exit(1);
        }
        else
        {
          fclose(fp);
          hasArg = 1;
        }
      }
      else if (strcmp(inString, keywords[6]) == 0) // end 
      {
        break;
      }
      else if (strcmp(inString, keywords[8]) == 0) // start sample
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %d", inString, inString2, &start);
        else sscanf(lineIn, "%s %d", inString, &start);
        if (start <= 0)
        {
          printf("RunParallel ERROR: invalid start sample <= 0.\n");
          exit(1);
        }
      }
      else if (inString[0] == '#') /* comments */
      {
        /* comment line */
      }
      else
      {
        printf("RunParallel ERROR: unrecognized line - %s\n",lineIn);
        printf("Correct format: \n");
        printf("PSUADE_PARALLEL\n");
        printf("workdir = <your designated work directory prefix>\n");
        printf("executable = <name of executable (absolute path)>\n");
        printf("argument = <name of argument (absolute path)>\n");
        printf("aux_exec = <name of second executable (absolute path)>\n");
        printf("num_parallel = <number of parallel simulations>\n");
        printf("proc_step = <which processors to run>\n");
        printf("start_sample = <which sample point to start>\n");
        printf("PSUADE_PARALLEL\n");
        exit(1);
      }
    }
    fclose(inFile);
  }

  //**/ find out which directory is active (has not been run) 
  if (mypid == 0)
  {
    statusArray = new int[nSamples];
    for (sampleID = 0; sampleID < nSamples; sampleID++)
      statusArray[sampleID] = -1;
    nActive = 0;
    for (sampleID = start-1; sampleID < nSamples; sampleID++)
    {
      snprintf(runLine,500,"%s.%d", workdir, sampleID+1);
      inFile = fopen(runLine, "r");
      if (inFile == NULL)
      {
        printf("RunParallel ERROR: %s does not exist.\n",runLine);
        exit(1);
      }
      fclose(inFile);
      snprintf(runLine,500,"%s.%d/completed", workdir, sampleID+1);
      inFile = fopen(runLine, "r");
      if (inFile == NULL) statusArray[nActive++] = sampleID;
      else
      {
        printf("RunParallel: file 'completed' found in %s.%d.\n",
               workdir, sampleID+1);
        fclose(inFile);
      }
    }
    printf("RunParallel INFO: number of jobs to be run = %d.\n",
           nActive);
  }

  //**/ transfer data to other processors
  if (mypid == 0) strLeng = strlen(workdir) + 1;
  else            strLeng = 10;
  workdir[strLeng] = '\0';
  psCommMgr_->bcast((void *) &strLeng, iOne, INT, root);
  psCommMgr_->bcast((void *) workdir, strLeng, CHAR, root);
  if (mypid == 0) strLeng = strlen(executable) + 1;
  else            strLeng = 10;
  executable[strLeng] = '\0';
  psCommMgr_->bcast((void *) &strLeng, iOne, INT, root);
  psCommMgr_->bcast((void *) executable, strLeng, CHAR, root);
  psCommMgr_->bcast((void *) &hasExec, iOne, INT, root);
  psCommMgr_->bcast((void *) &hasArg, iOne, INT, root);
  if (hasArg == 1)
  {
    if (mypid == 0) strLeng = strlen(execArg) + 1;
    else            strLeng = 10;
    execArg[strLeng] = '\0';
    psCommMgr_->bcast((void *) &strLeng, iOne, INT, root);
    psCommMgr_->bcast((void *) execArg, strLeng, CHAR, root);
  }
  psCommMgr_->bcast((void *) &hasAux, iOne, INT, root);
  if (hasAux == 1)
  {
    if (mypid == 0) strLeng = strlen(auxExecutable) + 1;
    else            strLeng = 10;
    auxExecutable[strLeng] = '\0';
    psCommMgr_->bcast((void *) &strLeng, iOne, INT, root);
    psCommMgr_->bcast((void *) auxExecutable, strLeng, CHAR, root);
  }
  psCommMgr_->bcast((void *) &nActive, iOne, INT, root);
  if (nActive > 0 && mypid != 0) statusArray = new int[nActive]; 
  psCommMgr_->bcast((void *) statusArray, nActive, INT, root);

  //**/ run all jobs
  for (sampleID = 0; sampleID < nActive; sampleID++)
  {
    index = statusArray[sampleID];
    if (sampleID % (nprocs/procStep) == (mypid/procStep))
    {
      if (hasExec == 1)
      {
        printf("Processor %d running job %d\n", mypid, index+1);
        if (hasArg == 1)
          snprintf(runLine,500,"(cd %s.%d; %s %s)", workdir, index+1, 
                  executable, execArg);
        else
          snprintf(runLine,500,"(cd %s.%d; %s %d)", workdir, index+1, 
                  executable, index+1);
      }
      printf("%s\n", runLine);
      system("echo 'Run directory = '; pwd");
      system(runLine);
      if (hasAux == 1)
      {
        snprintf(runLine,500,"(cd %s.%d; %s)", workdir, index+1, 
                 auxExecutable);
        printf("%s\n", runLine);
        system(runLine);
      }
    }
  }
  if (statusArray != NULL) delete [] statusArray;
  return 0;
}

// ************************************************************************
// run jobs in MPI parallel mode when the simulation code has been compiled
// into psuade (This mode helps alleviating multiple calls to queuing
// system)
// syntax: psuade -mpp infile
// where infile has the following syntax:
//     PSUADE_PARALLEL
//       workdir = xxx
//       num_sample = xxx
//       nprocs_per_job = xxx
//     PSUADE_PARALLEL
// Note: workdir.x (x = 1 to num_sample) should have been set up prior
//       to this call. Also, the work directories that have been 
//       completed should have a file called 'completed'
// ------------------------------------------------------------------------
int RunParallelLocal(const char *inFileName)
{
  int  mypid=0, nprocs=1, root=0, iOne=1, sampleID, ip, ii, lineLeng=200;
  int  nSamples=0, strLeng, *statusArray, nActive, index, npPerJob=0, cch;
  const char *keywords[] = {"workdir", "num_samples", "nprocs_per_job", 
                            "PSUADE_PARALLEL"}; 
  char lineIn[501], inString[500], inString2[500], outString[500];
  char workdir[500];
  FILE *inFile;
                                                                                
  mypid  = psCommMgr_->getPID();
  nprocs = psCommMgr_->getNumProcs();

  //**/ Read instructions from the input file
  if (mypid == 0)
  {
    inFile = fopen(inFileName, "r");
    if (inFile == NULL)
    {
      printf("RunParallelLocal read ERROR - file %s = NULL\n",inFileName);
      exit(1);
    }
    fgets(lineIn, lineLeng, inFile);
    sscanf(lineIn, "%s", inString);
    if (strcmp(inString, "PSUADE_PARALLEL")) 
    {
      printf("RunParallelLocal ERROR - keyword missing.\n");
      exit(1);
    }
    while ((fgets(lineIn, lineLeng, inFile) != NULL) && 
           ((cch=fgetc(inFile)) != EOF))
    {
      strcpy(inString, "#");
      sscanf(lineIn,"%s", inString);
      if (strcmp(inString, keywords[0]) == 0) // work directory 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %s", inString, inString2, workdir);
        else sscanf(lineIn, "%s %s", inString, workdir);
      }
      else if (strcmp(inString, keywords[1]) == 0) // nSamples 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %d", inString, inString2, &nSamples);
        else sscanf(lineIn, "%s %d", inString, &nSamples);
        if (nSamples <= 0)
        {
          printf("RunParallelLocal ERROR - nSamples %d <= 0.\n",
                 nSamples);
          exit(1);
        }
      }
      else if (strcmp(inString, keywords[2]) == 0) // number of procs 
      {
        sscanf(lineIn, "%s %s", inString, inString2);
        if (strcmp(inString2, "=") == 0 )
             sscanf(lineIn, "%s %s %d", inString, inString2, &npPerJob);
        else sscanf(lineIn, "%s %d", inString, &npPerJob);
        if (npPerJob <= 0)
        {
          printf("RunParallelLocal ERROR - npPerJob %d <= 0.\n",
                 npPerJob);
          exit(1);
        }
      }
      else if (strcmp(inString, keywords[3]) == 0) // end 
      {
        break;
      }
      else if (inString[0] == '#') /* comments */
      {
        /* comment line */
      }
      else
      {
        printf("RunParallelLocal ERROR - unrecognized line - %s\n",lineIn);
        exit(1);
      }
    }
    fclose(inFile);
  }
  if (nSamples <= 0 || npPerJob <= 0)
  {
    printf("RunParallelLocal ERROR: nSamples or npPerJob not specified.\n");
    exit(1);
  }

  //**/ find out which directory is active (has not been run) 
  //**/ (look for <workdir>.x where x goes from 1 to nSamples)
  if (mypid == 0)
  {
    statusArray = new int[nSamples];
    for (sampleID = 0; sampleID < nSamples; sampleID++)
      statusArray[sampleID] = -1;
    nActive = 0;
    for (sampleID = 0; sampleID < nSamples; sampleID++)
    {
      snprintf(outString,500,"%s.%d", workdir, sampleID+1);
      inFile = fopen(outString, "r");
      if (inFile == NULL)
      {
        printf("RunParallelLocal ERROR: %s does not exist.\n",outString);
        exit(1);
      }
      fclose(inFile);
      snprintf(outString,500,"%s.%d/completed", workdir, sampleID+1);
      inFile = fopen(outString, "r");
      if (inFile == NULL) statusArray[nActive++] = sampleID;
      if (inFile != NULL) fclose(inFile);
    }
  }

  //**/ transfer data to other processors
  if (mypid == 0) strLeng = strlen(workdir) + 1;
  else            strLeng = 10;
  workdir[strLeng] = '\0';
  psCommMgr_->bcast((void *) &strLeng, iOne, INT, root);
  psCommMgr_->bcast((void *) workdir, strLeng, CHAR, root);
  psCommMgr_->bcast((void *) &nActive, iOne, INT, root);
  if (nActive > 0 && mypid != 0) statusArray = new int[nActive]; 
  psCommMgr_->bcast((void *) statusArray, nActive, INT, root);
  psCommMgr_->bcast((void *) &npPerJob, iOne, INT, root);

#ifdef HAVE_MPICH
  //**/ set up and run jobs
  MPI_Comm  newComm;
  MPI_Group baseGroup, newGroup;
  int  myGroup=-1, nGroups, *pgroup;

  nGroups = nprocs / npPerJob;
  if (nGroups <= 0)
  {
    printf("RunParallelLocal ERROR: not enough processors.\n");
    exit(1);
  }
  pgroup  = new int[npPerJob];
  for (ip = 0; ip < nGroups; ip++)
  {
    for (ii = 0; ii < npPerJob; ii++) pgroup[ii] = ip * npPerJob + ii;
    for (ii = 0; ii < npPerJob; ii++)
    {
      if (mypid == pgroup[ii])
      {
        myGroup = ip;
        MPI_Comm_group(MPI_COMM_WORLD, &baseGroup);
        MPI_Group_incl(baseGroup, npPerJob, pgroup, &newGroup);
        MPI_Comm_create(MPI_COMM_WORLD,newGroup,&newComm);
        break;
      }
    }
  }
  for (sampleID = 0; sampleID < nActive; sampleID++)
  {
    index = statusArray[sampleID];
    if (sampleID % nGroups == myGroup)
    {
      printf("Processor %d running job %d\n", mypid, index+1);
      UserFunction(newComm, index+1, workdir);
    }
  }
  delete [] pgroup;
  if (statusArray != NULL) delete [] statusArray;
#else
  printf("RunParallelLocal ERROR: MPI build not enabled.\n");
  exit(1);
#endif
  return 0;
}


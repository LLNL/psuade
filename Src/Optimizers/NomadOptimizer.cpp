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
// Functions for the class NomadOptimizer
// AUTHOR : CHARLES TONG
// DATE   : 2016
// ************************************************************************
#include <stdio.h>
#include <stdlib.h>
#include "NomadOptimizer.h"
#include "Psuade.h"
#include "PsuadeUtil.h"

#ifdef HAVE_NOMAD
#include "../../External/NOMAD/src/nomad.hpp"
#include "../../External/NOMAD/src/Display.hpp"
using namespace NOMAD;
#endif

oData *psNomadoData_=NULL;
int    psNomadnOutputs_=-1;
double psNomadOptY_=1e50;
//**/ these variables are for tracking convergence
double psNomadTolerance_=1.0e-5;
psVector VecPsNomadObjFcnStore_;
//**/ other useful flags
int    psNomadPrintLevel_=-1;
int    psNomadStop_=0;
int    psNomadCurrDriver_=-1;
//**/ for storing history data
#define psNomadMaxSaved_ 10000
int     psNomadSaveHistory_=0;
int     psNomadNSaved_=0;
double  psNomadSaveX_[psNomadMaxSaved_*10];
double  psNomadSaveY_[psNomadMaxSaved_*10];
#define PABS(x)  ((x) > 0 ? x : -(x))

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
NomadOptimizer::NomadOptimizer()
{
  if (psConfig_.InteractiveIsOn())
  {
    printAsterisks(PL_INFO, 0);
    printf("*   NOMAD Optimizer Usage Information\n");
    printEquals(PL_INFO, 0);
    printf("* - To run this optimizer, first make sure opt_driver has\n");
    printf("*   been initialized to point to your optimization objective\n");
    printf("*   function evaluator\n");
    printf("* - Set maximum number of iterations in PSUADE input file\n");
    printf("* - Set num_local_minima to perform multi-start optimization\n");
    printf("* - Set optimization print_level to give additonal outputs\n");
// printf("* - In Opt EXPERT mode, the optimization history log will be\n");
// printf("*   turned on automatically. Previous psuade_nomad_history\n");
// printf("*   file will also be reused.\n");
    printAsterisks(PL_INFO, 0);
  }
  nInputs_ = 0;
  inputTypes_ = NULL;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
NomadOptimizer::~NomadOptimizer()
{
  if (inputTypes_ != NULL) delete [] inputTypes_;
}

// ************************************************************************
// set discrete variables 
// ------------------------------------------------------------------------
void NomadOptimizer::setDiscreteVariable(int index)
{
  char winput[101];
  if (index <= 0)
  {
    printf("NOMAD setDiscreteVariable ERROR: variable number <= 0.\n");
    exit(1);
  }
  snprintf(winput,100,"iDiscrete%d", index);
  psConfig_.putParameter(winput);
}

// ************************************************************************
// optimize (this function should be used in the library mode)
// ------------------------------------------------------------------------
void NomadOptimizer::optimize(int nInputs, double *XValues, double *lbds,
                          double *ubds, int nOuts, int maxfun, double tol)
{
  psVector vecOptX;
  if (nInputs <= 0)
  {
    printf("NomadOptimizer ERROR: nInputs <= 0.\n");
    exit(1);
  }
  //**/ create and load up the oData object before calling optimize
  oData *odata = new oData();
  odata->outputLevel_ = 0;
  odata->nInputs_ = nInputs;
  vecOptX.setLength(nInputs);
  odata->optimalX_ = vecOptX.getDVector();
  odata->initialX_ = XValues;
  odata->lowerBounds_ = lbds;
  odata->upperBounds_ = ubds;
  odata->tolerance_ = tol;
  if (odata->tolerance_ <= 0) odata->tolerance_ = 1e-6;
  odata->nOutputs_ = nOuts;
  odata->outputID_ = 0;
  odata->maxFEval_ = maxfun;
  odata->tolerance_ = tol;
  odata->setOptDriver_ = 0;
  //**/ for this function to work, setObjectiveFunction has to be called
  //**/ beforehand
  odata->optFunction_ = objFunction_;
  odata->funcIO_ = NULL;
  optimize(odata);
  odata->initialX_ = NULL;
  odata->lowerBounds_ = NULL;
  odata->upperBounds_ = NULL;
  odata->optFunction_ = NULL;
  VecOptX_.setLength(nInputs);
  for (int ii = 0; ii < nInputs; ii++)
    XValues[ii] = VecOptX_[ii] = odata->optimalX_[ii];
  odata->optimalX_ = NULL;
}

// ************************************************************************
// optimize
// ------------------------------------------------------------------------
void NomadOptimizer::optimize(oData *odata)
{
#ifdef HAVE_NOMAD
  int    ii, kk, nInps, nOuts, printLevel, maxfun, idata, option;
  double ddata, tol;
  char   pString[101], *cString;
  FILE   *fp;
  psVector vecXT;

  //**/ ----------------------------------------------------------
  //**/ ------ fetch optimization information ------
  //**/ ----------------------------------------------------------
  printLevel = odata->outputLevel_;
  nInps  = odata->nInputs_;
  nOuts  = odata->nOutputs_;
  maxfun = odata->maxFEval_;
  psNomadTolerance_ = odata->tolerance_;
  double *lbounds = odata->lowerBounds_;
  double *ubounds = odata->upperBounds_;
  vecXT.setLength(nInps);
  for (ii = 0; ii < nInps; ii++) vecXT[ii] = odata->initialX_[ii];
  for (ii = 0; ii < nInps; ii++) odata->optimalX_[ii] = vecXT[ii];
  odata->optimalY_ = 1.0e50;
  //**/ this is done to save the driver switching for multiple opt
  if ((odata->setOptDriver_ & 1))
  {
    printf("NOMAD: setting optimization simulation driver.\n");
    psNomadCurrDriver_ = odata->funcIO_->getDriver();
    odata->funcIO_->setDriver(1);
  }

  //**/ ----------------------------------------------------------
  //**/ retrieve history and look for discrete variables
  //**/ ----------------------------------------------------------
  cString = psConfig_.getParameter("opt_save_history");
  if (cString != NULL) psNomadSaveHistory_ = 1;
  cString = psConfig_.getParameter("opt_use_history");
  if (cString != NULL)
  {
    printf("NOMAD: use history has been turned on.\n");
    fp = fopen("psuade_nomad_history","r");
    if (fp != NULL)
    {
      psNomadNSaved_ = 0;
      while (feof(fp) == 0)
      {
        fscanf(fp, "%d %d", &ii, &kk);
        if (ii != 999 || kk != nInps) break;
        else
        {
          for (ii = 0; ii < nInps; ii++)
            fscanf(fp, "%lg",&psNomadSaveX_[psNomadNSaved_*nInps+ii]);
          for (ii = 0; ii < nOuts; ii++)
            fscanf(fp, "%lg",&psNomadSaveY_[psNomadNSaved_*nOuts+ii]);
          psNomadNSaved_++;
        }
        if (((psNomadNSaved_+1)*nInps > psNomadMaxSaved_*10) ||
            ((psNomadNSaved_+1)*nOuts > psNomadMaxSaved_*10))
          break;
      }
      fclose(fp);
    }
  }
  kk = 0;
  for (ii = 0; ii < nInps; ii++)
  {
    snprintf(pString,100,"iDiscrete%d", ii+1);
    cString = psConfig_.getParameter(pString);
    if (cString != NULL) kk++;
  }
  if (kk > 0)
  {
    if (inputTypes_ != NULL) delete [] inputTypes_;
    inputTypes_ = new int[nInps];
    for (ii = 0; ii < nInps; ii++) 
    {
      snprintf(pString,100,"iDiscrete%d", ii+1);
      cString = psConfig_.getParameter(pString);
      if (cString != NULL)
      {
        inputTypes_[ii] = 2;
        if (printLevel > 0)
          printf("NOMAD input %4d is discrete\n",ii+1);
      }
      else inputTypes_[ii] = 1;
    }
    option = 2;
  }

  //**/ ----------------------------------------------------------
  //**/ ------ set up nomad ------
  //**/ ----------------------------------------------------------
  psNomadPrintLevel_ = printLevel;
  Display dout ( std::cout );
  dout.precision( DISPLAY_PRECISION_STD );
  Parameters nomadp(dout);
  nomadp.set_DISPLAY_DEGREE(0);

  //**/ input set up: dimension and types
  nomadp.set_DIMENSION(nInps);
  if (psConfig_.InteractiveIsOn() && nInputs_ == 0 && inputTypes_ == NULL)
  {
    printf("NOMAD can solve either \n");
    printf("1. continuous (which is slow compared to bobyqa) or \n");
    printf("2. mixed-integer optimization.\n");
    snprintf(pString,100,"Please select (1) or (2) : ");
    option = getInt(1, 2, pString);

    //**/ if mixed integer optimization is desired, ask more questions
    if (option == 2)
    {
      inputTypes_ = new int[nInps];
      printf("NOMAD can handle 3 input types: \n");
      printf("1. Real (or R) - real/continuous\n");
      printf("2. Int  (or I) - integer\n");
      printf("3. Bin  (or B) - binary\n");
      //printf("4. Cat  (or C) - categorical\n");
      for (ii = 0; ii < nInps; ii++)
      {
        snprintf(pString,100,"Please enter type for input %d : ",ii+1);
        inputTypes_[ii] = getInt(1, 3, pString);
        idata = (int) lbounds[ii];
        if (inputTypes_[ii] == 3)
        {
          if (lbounds[ii] != 0 || ubounds[ii] != 1)
          {
            printf("ERROR: input type has been set to binary but ");
            printf("the bounds are not [0,1].\n");
            printf("INFO: bounds reset to [0,1]\n"); 
            lbounds[ii] = 0;
            ubounds[ii] = 1;
          }
        }
        if (inputTypes_[ii] == 2)
        {
          idata = (int) lbounds[ii];
          kk    = (int) ubounds[ii];
          if ((lbounds[ii]-1.0*idata) != 0 || (ubounds[ii]-1.0*kk) != 0)
          {
            printf("ERROR: input type has been set to integer but ");
            printf("the bounds are not integers..\n");
            printf("       Please set the bounds correctly.\n");
            exit(1);
          }
        }
      }
    }
    if (nOuts > 1)
    {
      printf("The number of outputs = %d\n", nOuts);
      printf("NOMAD will treat the first one as objective function\n");
      printf("and the rest as inequality constraints.\n");
      printf("NOTE: For NOMAD to accept a design point, all inequality\n");
      printf("      constraints have to be <= 0.\n");
    }
  }
  nInputs_ = nInps;

  //**/ input set up: bounds and initial guess
  Point lbnds(nInps), ubnds(nInps), X0(nInps);
  for (ii = 0; ii < nInps; ii++)
  {
    lbnds[ii] = lbounds[ii];
    ubnds[ii] = ubounds[ii];
    if (inputTypes_ != NULL)
    {
      switch(inputTypes_[ii])  
      {
        case 1: nomadp.set_BB_INPUT_TYPE (ii, CONTINUOUS); 
                X0[ii] = vecXT[ii];
                break;
        case 2: nomadp.set_BB_INPUT_TYPE (ii, INTEGER); 
                X0[ii] = (int) (vecXT[ii] + 0.5);
                break;
        case 3: nomadp.set_BB_INPUT_TYPE (ii, BINARY); 
                idata = (int) vecXT[ii];
                if (idata == 0) X0[ii] = 0;
                else            X0[ii] = 1;
                break;
        case 4: nomadp.set_BB_INPUT_TYPE (ii, CATEGORICAL); 
                X0[ii] = (int) vecXT[ii];
                break;
      }
    }
    else X0[ii] = vecXT[ii];
  }
  nomadp.set_LOWER_BOUND (lbnds);
  nomadp.set_UPPER_BOUND (ubnds);
  nomadp.set_X0(X0);

  //**/ set up nomad output (it can handle constraints)
  psNomadnOutputs_ = nOuts;
  vector<bb_output_type> bbot (nOuts);
  //**/ first is objective function
  bbot[0] = OBJ;
  //**/ The others are inequality constraints. Nomad manual
  //**/ recommends the following choices for others
  //**/  EB - constraints always need to be satisfied
  //**/       note: this is very limiting though
  //**/  PB - constraints need only be satisfied at the solution
  for (ii = 1; ii < nOuts; ii++) bbot[ii] = PB;
  nomadp.set_BB_OUTPUT_TYPE ( bbot );

  //**/ set other nomad parameters 
  nomadp.set_SPECULATIVE_SEARCH ( true );
  if (psConfig_.InteractiveIsOn() && psConfig_.OptExpertModeIsOn())
  {
    snprintf(pString,100, 
            "Enter value for mesh update basis (default = 8) : ");
    ddata = getDouble(pString);
  } 
  else ddata = 8.0;
  nomadp.set_MESH_UPDATE_BASIS ( ddata );

  if (psConfig_.InteractiveIsOn() && psConfig_.OptExpertModeIsOn())
  {
    snprintf(pString,100, 
            "Enter value for mesh coarsening exponent (default = 1) : ");
    idata = getInt(0, 10, pString);
  } 
  else idata = 1;
  nomadp.set_MESH_COARSENING_EXPONENT ( idata );
  nomadp.set_MAX_BB_EVAL (maxfun);
  for (ii = 0; ii < nInps; ii++)
  {
    ddata = 0.5 * (ubounds[ii] - lbounds[ii]);
    //**/ nomadp.set_INITIAL_MESH_SIZE ( 0.4 );
    nomadp.set_INITIAL_MESH_SIZE ( ii, ddata, false );
  }
  //**/ don't use this yet. Need to figure out first
  //nomadp.set_LH_SEARCH (3, 3);

  //**/ checking required
  //**/ nomadp.display(dout);
  nomadp.check();

  //**/ set up simulation model pointer
  PsuadeNomadEvaluator madEvaluator(nomadp);   
  Mads mads(nomadp, &madEvaluator);
  psNomadoData_ = odata;
  psNomadoData_->numFuncEvals_ = 0;
  VecPsNomadObjFcnStore_.setLength(maxfun);
  for (ii = 0; ii < maxfun; ii++) VecPsNomadObjFcnStore_[ii] = -1e35;

  //**/ ----------------------------------------------------------
  //**/ ------ run nomad and process outputs ------
  //**/ ----------------------------------------------------------
  psNomadStop_ = 0;
  mads.run();

  //**/ update optimal solution
  VecOptX_.setLength(nInps);
  for (ii = 0; ii < nInps; ii++) 
    VecOptX_[ii] = psNomadoData_->optimalX_[ii];
  optimalY_ = psNomadoData_->optimalY_;

  //**/ create matlab convergence file
  char name[1000];
  if (psConfig_.InteractiveIsOn() && psConfig_.OptExpertModeIsOn())
  {
    fp = fopen("matlabnomad.m", "w");
    if (fp != NULL)
    {
      fprintf(fp, "A = [\n");
      for (ii = 0; ii < maxfun; ii++) 
      {
        if (VecPsNomadObjFcnStore_[ii] <= -1e35) break;
        else fprintf(fp, "%e\n", VecPsNomadObjFcnStore_[ii]);
      }
      fprintf(fp, "];\n");
      fprintf(fp, "plot(A,'linewidth',2)\n");
      fwritePlotAxes(fp);
      strcpy(name, "Iteration count");
      fwritePlotXLabel(fp, name);
      strcpy(name, "Objective function value");
      fwritePlotYLabel(fp, name);
      strcpy(name, "Nomad Convergence Plot (Some values may be infeasible)");
      fwritePlotTitle(fp, name);
      fclose(fp);
      printf("matlabnomad.m is now available.\n");
    }
  }
    
  //**/ ----------------------------------------------------------
  //**/ ------ restore previous setting and clean up 
  //**/ ----------------------------------------------------------
  if ((odata->setOptDriver_ & 2) && psNomadCurrDriver_ >= 0)
  {
    printf("NOMAD INFO: reverting to original simulation driver.\n");
    odata->funcIO_->setDriver(psNomadCurrDriver_);
  }
#else
  printf("ERROR : NOMAD optimizer not installed.\n");
  exit(1);
#endif
}

// ************************************************************************
// assign operator
// ------------------------------------------------------------------------
NomadOptimizer& NomadOptimizer::operator=(const NomadOptimizer &)
{
  printf("NomadOptimizer operator= ERROR: operation not allowed.\n");
  exit(1);
  return (*this);
}

#ifdef HAVE_NOMAD
// ************************************************************************
// PsuadeNomadEvaluator constructor
// ------------------------------------------------------------------------
PsuadeNomadEvaluator::PsuadeNomadEvaluator(const Parameters &pset): 
                      Evaluator(pset)
{
}

// ************************************************************************
// PsuadeNomadEvaluator destructor
// ------------------------------------------------------------------------
PsuadeNomadEvaluator::~PsuadeNomadEvaluator()
{
}

// ************************************************************************
// PsuadeNomadEvaluator evaluation 
// ------------------------------------------------------------------------
bool PsuadeNomadEvaluator::eval_x(Eval_Point &X,const Double &h_max,
                                  bool &count_eval)
{
  int  ii, jj, funcID, nInps = X.get_n(), goodFlag;
  bool retFlag=true;
  FILE *fp=NULL;
  psVector vecXVals, vecYVals;

  //**/ If stop has been requested, just perform dummy evaluation
  vecXVals.setLength(nInps);
  vecYVals.setLength(psNomadnOutputs_);
  if (psNomadStop_ == 1)
  {
    vecYVals[0] = -1e50;
    X.set_bb_output(0, vecYVals[0]);
    vecYVals[0] = 0.0;
    for (ii = 1; ii < psNomadnOutputs_; ii++) 
      X.set_bb_output(ii, vecYVals[0]);
    retFlag = false;
    return retFlag;
  }

  //**/ convert nomad X format to double *
  for (ii = 0; ii < nInps; ii++) vecXVals[ii] = X[ii].value();
  funcID = psNomadoData_->numFuncEvals_++;

  //**/ evaluate the model, if needed
  int found = 0;
  for (ii = 0; ii < psNomadNSaved_; ii++)
  {
    for (jj = 0; jj < nInps; jj++)
      if (PABS(psNomadSaveX_[ii*nInps+jj]-vecXVals[jj])>1.0e-14) break;
    if (jj == nInps)
    {
      found = 1;
      printf("NOMAD: simulation results reuse.\n");
      break;
    }
  }
  if (found == 0)
  {
    if (psNomadoData_->optFunction_ != NULL)
      //**/ if users have loaded a simulation function, use it
      psNomadoData_->optFunction_(nInps, vecXVals.getDVector(), 
                           psNomadnOutputs_,vecYVals.getDVector());
    else if (psNomadoData_->funcIO_ != NULL)
      psNomadoData_->funcIO_->evaluate(funcID,nInps,vecXVals.getDVector(),
                                psNomadnOutputs_,vecYVals.getDVector(),0);
    else
    {
      printf("NomadOptimizer ERROR: no function evaluator.\n");
      exit(1);
    }
    if (psNomadSaveHistory_ == 1 && found == 0 &&
        (psNomadNSaved_+1)*nInps < psNomadMaxSaved_*10 &&
        (psNomadNSaved_+1)*psNomadnOutputs_ < psNomadMaxSaved_*10) 
    {
      for (jj = 0; jj < nInps; jj++)
        psNomadSaveX_[psNomadNSaved_*nInps+jj] = vecXVals[jj];
      for (jj = 0; jj < psNomadnOutputs_; jj++)
        psNomadSaveY_[psNomadNSaved_*psNomadnOutputs_+jj] = vecYVals[jj];
      psNomadNSaved_++;
      fp = fopen("psuade_nomad_history","w");
      if (fp != NULL)
      {
        for (ii = 0; ii < psNomadNSaved_; ii++)
        {
          fprintf(fp, "999 %d ", nInps);
          for (jj = 0; jj < nInps; jj++)
            fprintf(fp, "%24.16e ", psNomadSaveX_[ii*nInps+jj]);
          fprintf(fp, "%24.16e\n", psNomadSaveY_[ii]);
        }
        fclose(fp);
      }
    }
  }
  else
  {
    for (jj = 0; jj < psNomadnOutputs_; jj++)
      vecYVals[jj] = psNomadSaveY_[ii*psNomadnOutputs_+jj];
  }

  //**/ Eval_Point X can also store simulation outputs
  for (ii = 0; ii < psNomadnOutputs_; ii++) 
    X.set_bb_output(ii, vecYVals[ii]);

  //**/ If objective function is smaller and constraints are satisfied,
  //**/ store the results
  if (vecYVals[0] < psNomadoData_->optimalY_) 
  {
    goodFlag = 1;
    for (ii = 1; ii < psNomadnOutputs_; ii++) 
      if (vecYVals[ii] > 0) goodFlag = 0;
    if (goodFlag == 1)
    {
      for (ii = 0; ii < nInps; ii++) 
        psNomadoData_->optimalX_[ii] = vecXVals[ii];
      psNomadoData_->optimalY_ = vecYVals[0];
      if (psNomadPrintLevel_ > 0)
        printf("   NOMAD current best Y = %e (nfevals=%d)\n", 
               vecYVals[0], funcID+1);
    }
    else
    {
      if (psNomadPrintLevel_ > 1)
      {
        printf("   NOMAD: ignore infeasible solution (nfeval=%d)\n",
               funcID+1);
      }
    }
  }

  //**/ analyze convergence
  VecPsNomadObjFcnStore_[funcID] = vecYVals[0];
  int    converged;
  double diff;
  if (funcID >= 100000000)
  {
    converged = 1;
    for (ii = funcID; ii > funcID-30; ii--) 
    {
      diff = VecPsNomadObjFcnStore_[ii] - VecPsNomadObjFcnStore_[ii-1]; 
      if (diff < 0) diff = -diff;
      if (diff > psNomadTolerance_) 
      {
        converged = 0;
        break;
      }
    }
    if (converged == 1)
    {
      printf("NOMAD INFO: convergence detected at iteration %d ==> stop.\n",
             funcID);
      psNomadStop_ = 1;
    }
  }

  //**/ clean up
  fp = fopen("psuade_stop", "r");
  if (fp != NULL)
  {
    printf("**************** ==> \n");
    printf("INFO: PSUADE found psuade_stop (REQUEST TO TERMINATE).\n");
    printf("      Remove this file if this is not what you desire.\n");
    printf("<== **************** \n");
    fclose(fp);
    psNomadStop_ = 1;
  }
#ifdef MACOS
  printf("NOMAD WARNING: Crash may occur around here with MacOS.\n");
#endif
  return retFlag;
}
#endif


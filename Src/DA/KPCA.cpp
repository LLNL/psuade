// ************************************************************************
// Copyright (c) 2015   Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the PSUADE team.
// All rights reserved.
//
// Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
// disclaimer, contact information and the GNU Lesser General Public License.
//
// DASSI is free software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License (as published by the Free Software
// Foundation) version 2.1 dated February 1999.
//
// DASSI is distributed in the hope that it will be useful, but WITHOUT ANY
// WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or FITNESS
// FOR A PARTICULAR PURPOSE.  See the terms and conditions of the GNU General
// Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this program; if not, write to the Free Software Foundation,
// Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 USA
// ************************************************************************
// KPCA functions (for Xiao Chen's LDRD project)
// DATE   : 2015
// ************************************************************************
#include <assert.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "PsuadeUtil.h"
#include "KPCA.h"

// ************************************************************************
// External optimizer and fortran (to facilitate AD) functions
// ------------------------------------------------------------------------
extern "C" 
{
  void gensnapshot_(int *,int *,int*,int *, double *,double *,double *,
                    double *,double *,double *,double *,double *,double *,
                     int*,int*);
}

// ************************************************************************
// constructor
// ------------------------------------------------------------------------
KPCA::KPCA()
{
  kernelType_ = DA_GAUSSIAN;
  gaussScale_ = 11.0;
  trueScale_ = 1;
  rDim_ = 0;
  initialized_ = 0;
  convergenceTol_ = 1.0e-10;
}

// ************************************************************************
// destructor
// ------------------------------------------------------------------------
KPCA::~KPCA()
{
}

// ************************************************************************
// set kernel 
// ------------------------------------------------------------------------
int KPCA::setKernel(const int kernel)
{
  kernelType_ = kernel;
  if (kernel < 0 || kernel > 5) kernelType_ = getKernel();
  if (outputLevel_ > 0)
  {
    if      (kernel == 0) printf("KPCA: Gaussian  kernel selected.\n");
    else if (kernel == 1) printf("KPCA: Linear    kernel selected.\n");
    else if (kernel == 2) printf("KPCA: Quadratic kernel selected.\n");
    else if (kernel == 3) printf("KPCA: Cubic     kernel selected.\n");
    else if (kernel == 4) printf("KPCA: Quartic   kernel selected.\n");
    else if (kernel == 5) printf("KPCA: 5th-order kernel selected.\n");
  } 
  initialized_ = 0;
  return 0;
}

// ************************************************************************
// set reduced dimension
// ------------------------------------------------------------------------
int KPCA::setReducedDimension(const int dim)
{
  if (dim < 1)
  {
    printf("KPCA setReducedDimension ERROR: dim has to be > 0.\n");
    return -1;
  } 
  rDim_ = dim;
  initialized_ = 0;
  return 0;
}

// ************************************************************************
// set scale of Gaussian kernel
// ------------------------------------------------------------------------
int KPCA::setgaussScale(const float scale)
{
  gaussScale_ = scale;
  initialized_ = 0;
  return 0;
}

// ************************************************************************
// set print level 
// ------------------------------------------------------------------------
int KPCA::setOutputLevel(int level)
{
  if (level >= 0) outputLevel_ = level;
  else            outputLevel_ = 0;
  return 0;
}

// ************************************************************************
// set convergence tolerance 
// ------------------------------------------------------------------------
int KPCA::setConvergenceTol(double conv)
{
  if (conv <= 0)
  {
    printf("KPCA setConvergenceTol ERROR: tolerance <= 0.0\n");
    printf("                       Reset to 1.0e-4\n");
    convergenceTol_ = 1.0e-4;
  } 
  else convergenceTol_ = conv;
  return 0;
}

// ************************************************************************
// set snapshots 
// ------------------------------------------------------------------------
int KPCA::loadSnapshots(psMatrix sMat)
{
  snapMat_ = sMat;
  initialized_ = 0;
  return 0;
}

// ************************************************************************
// initialize
// ------------------------------------------------------------------------
int KPCA::initialize(psMatrix &featureMat)
{
  int      nrows, ii, jj, col;
  double   dsum, ddata, dmax;
  char     cinput[100];
  psMatrix tmpMat, Kmat;
  psVector tmpVec;
  FILE     *fp = NULL;

  //**/ error checking
  if (snapMat_.ncols() <= 0 || snapMat_.nrows() <= 0)
  {
    printf("KPCA ERROR: snapshots have not been loaded.\n");
    return -1;
  }
  if (rDim_ <= 0)
  {
    rDim_ = snapMat_.nrows();
    printf("KPCA initialize INFO: reduced dimension has not been set.\n");
    printf("==> There will be no dimension reduction (at %d).\n",rDim_);
  }

  //**/ generate kernel matrix: K = (K - One*K - K*One + One*K*One)/N
  if (outputLevel_ > 0)
  {
    printf("KPCA initialize begins\n");
    printf("KPCA::Generate kernel K = (K0-One*K0 - K0*One + One*K0*One)/N\n");
  }
  computeKernelMatrix();

  //**/ eigendecompsition of kernel matrix
  if (outputLevel_ > 0) printf("Kernel eigensolve: K/N = V D V^t\n");
  kernelMat_.eigenSolve(eigenMatrix_,eigenValues_,0);

  //**/ calculate energies in different reduced eigenspace
  //if (outputLevel_ > 3) 
  double percentage=0;
  {
    dsum = 0.0;
    for (ii = 0; ii < eigenValues_.length(); ii++)
      dsum += eigenValues_[ii];
    ddata = 0.0;
    for (ii = 0; ii < eigenValues_.length(); ii++)
    {
      ddata += eigenValues_[ii];
      if (outputLevel_ > 0 && (ii >= (eigenValues_.length()-rDim_)))
        printf("KPCA::Eigenvalue %5d = %12.4e (percentage = %12.4e\n",
               ii+1, eigenValues_[ii], ddata/dsum*100.0);
      if (ii == (eigenValues_.length()-rDim_))
        percentage = (1 - ddata/dsum) * 100.0;
    }
  }

  //**/ testing how good eigen-decomposition is
  //**/ K * V
  psMatrix newMat, newMat2;
  if (outputLevel_ >= 4)
  {
    kernelMat_.matmult(eigenMatrix_, newMat);
    newMat2 = eigenMatrix_;
    dsum = dmax = 0.0;
    for (ii = 0; ii < newMat2.ncols(); ii++)
    {
      for (jj = 0; jj < newMat2.nrows(); jj++)
      {
        dsum += pow(newMat2.getEntry(jj,ii)*eigenValues_[ii]-
                    newMat.getEntry(jj,ii),2);
        if (ddata > dmax) dmax = ddata;
      }
    }
    printf("KPCA::KV-VD rms and max norms = %e %e\n", dsum, dmax);
  }

#ifdef DA_DEBUG
  eigenMatrix_.print("Kernel eigenvectors");
#endif
  if (outputLevel_ > 0)
    printf("KPCA::Kernel reduction: select %d principal components\n",
           rDim_);

  //**/ revised dimension of reduced space, if needed
  nrows = kernelMat_.nrows();
  for (ii = nrows-2; ii >= 0; ii--)
    if (eigenValues_[ii]/eigenValues_[ii+1] < 1.0e-8) break;
  if (nrows - ii - 1 < rDim_)
  {
    rDim_ = nrows - ii - 1;
    printf("KPCA::INFO: number of principal components reduced to %d\n",
           rDim_);
  }
  tmpMat.setFormat(PS_MAT2D);
  tmpMat.setDim(nrows, rDim_);
  tmpVec.setLength(rDim_);
  for (ii = nrows-rDim_; ii < nrows; ii++)
  {
    col = ii - nrows + rDim_;
    for (jj = 0; jj < nrows; jj++)
      tmpMat.setEntry(jj, col, eigenMatrix_.getEntry(jj,ii));
    tmpVec[col] = eigenValues_[ii];
  }
  eigenValues_ = tmpVec;
  eigenMatrix_ = tmpMat;
  strcpy(cinput, "Dominant Eigenvalues");
  if (outputLevel_ > 0) eigenValues_.print(cinput);

  //**/ compute principal components = K/N * V
  kernelMat_.matmult(eigenMatrix_, featureMat);
  featureMat.transpose();
  initialized_ = 1;

  //**/ compute xi = D^{-1/2} featureVec?
  //**/ Yes, supposedly (beta' * Cov * beta = xi' * xi)

  if (outputLevel_ > 2)
  {
    printf("KPCA::Create histograms for random variables? (y or n) ");
    scanf("%s", cinput);
    if (cinput[0] == 'y')
    {
      fp = fopen("rvplot.m","w");
      if (fp != NULL)
      {
        fprintf(fp, "%%the matrix gives distribution of xi\n");
        fprintf(fp, "A = [\n");
        for (ii = 0; ii < featureMat.ncols(); ii++)
        {
          featureMat.getCol(ii, tmpVec);
          for (jj = 0; jj < tmpVec.length(); jj++)
            tmpVec[jj] = tmpVec[jj] / eigenValues_[jj] *
                         sqrt(snapMat_.ncols());
          for (jj = 0; jj < tmpVec.length(); jj++)
            fprintf(fp, "%24.16e ",tmpVec[jj]);
          fprintf(fp, "\n");
        }
      }
      fprintf(fp, "];\n");
      fprintf(fp, "clf\n");
      fprintf(fp, "for ii = 1 : %d\n", eigenValues_.length());
      fprintf(fp, "   figure\n");
      fprintf(fp, "   X = A(:,ii);\n");
      fprintf(fp, "   std(X)\n");
      fprintf(fp, "   hist(X);\n");
      fprintf(fp, "end\n");
      fclose(fp);
    }
  }

  double *ximax, *ximin, xival;
  ximax = new double[featureMat.ncols()];
  ximin = new double[featureMat.ncols()];
  for (ii = 0; ii < rDim_; ii++)
  {
    ximax[ii] = -1e35;
    ximin[ii] = 1e35;
  }
  if (outputLevel_ > 1)
  {
    for (ii = 0; ii < rDim_; ii++)
    {
      featureMat.getCol(ii, tmpVec);
      dsum = 0.0;
      for (jj = 0; jj < tmpVec.length(); jj++)
      {
        xival = tmpVec[jj] / eigenValues_[jj] * 
                sqrt(snapMat_.ncols());
        if (xival > ximax[ii]) ximax[ii] = xival;
        if (xival < ximin[ii]) ximin[ii] = xival;
        dsum += xival;
      }
      dsum /= (double) tmpVec.length();
      ddata = 0.0;
      for (jj = 0; jj < tmpVec.length(); jj++)
      {
        xival = tmpVec[jj] / eigenValues_[jj] * 
                sqrt(snapMat_.ncols());
        ddata += pow(xival-dsum, 2.0);
      }
      ddata = sqrt(ddata/(double) (tmpVec.length() - 1.0));
      printf("Statistics (mean,stdev) of PC %5d = %12.4e %12.4e\n",
             ii+1, dsum, ddata);
    }
    printf("\nKPCA::Ranges of RV values from snapshots:\n");
    for (ii = 0; ii < rDim_; ii++)
      printf("RV %6d range = %e %e\n",ii+1,ximin[ii],ximax[ii]);
  }
  delete [] ximax;
  delete [] ximin;

  if (outputLevel_ > 0) 
    printf("KPCA initialize ends: principal components are returned\n");
  return 0;
}

// ************************************************************************
// validate KPCA 
// ------------------------------------------------------------------------
int KPCA::validateKPCA(psMatrix &featureMat, int printLevel)
{
  int    ii, jj, nSnaps, snapDim;
  double ddata, rnorm;
  char   cString[1000];
  FILE   *fp=NULL;
  psVector solnVec, tmpVec;

  if (initialized_ == 0)
  {
    printf("ERROR: KPCA has not been initialized yet.\n");
    printf("       No test performed.\n");
  }
  nSnaps  = snapMat_.ncols();
  snapDim = snapMat_.nrows();
  rnorm   = 0.0;
  fp = fopen("ReconstructedSnapshots", "w");
  for (ii = 0; ii < nSnaps; ii++)
  {
    //if ((ii+1) % (nSnaps/10) == 0)
    printf("KPCA::Processing snapshot %d out of %d\n", ii+1, nSnaps);
    featureMat.getCol(ii, tmpVec);
    genSnapshotFromStdRV(tmpVec, solnVec);
    if (printLevel > 2) 
    {
      strcpy(cString, "Computed Solution");
      solnVec.print(cString);
    }
    snapMat_.getCol(ii, tmpVec);
    if (printLevel > 2) 
    {
      strcpy(cString, "Expected Solution");
      tmpVec.print(cString);
    }
    for (jj = 0; jj < snapDim; jj++)
      fprintf(fp, "%16.8e ", solnVec[jj]);
    fprintf(fp, "\n");
    ddata = 0.0;
    for (jj = 0; jj < snapDim; jj++)
      ddata += pow(tmpVec[jj]-solnVec[jj], 2.0);
    rnorm += sqrt(ddata);
    printf("KPCA::Snapshot %5d: error 2-norm = %e\n", ii+1, sqrt(ddata));

    featureMat.getCol(ii, tmpVec);
    computePreImage(tmpVec, solnVec);
    snapMat_.getCol(ii, tmpVec);
    ddata = 0.0;
    for (jj = 0; jj < snapDim; jj++)
      ddata += pow(tmpVec[jj]-solnVec[jj], 2.0);
    rnorm += sqrt(ddata);
    printf("KPCA::Snapshot %5d: error 2-norm = %e (method 2)\n", ii+1, 
           sqrt(ddata));
  }
  fclose(fp);
  printf("KPCA::Test for all snapshots: avg residual norm = %e\n",
         rnorm/nSnaps);
  printf("KPCA::reconstructed snapshots in file ReconstructedSnapshots\n");
  return 0; 
}

// ************************************************************************
// generate beta from standard random variable 
// ------------------------------------------------------------------------
int KPCA::genBetaFromStdRV(psVector &xivec, psVector &beta)
{
  //**/ beta = 1/sqrt(m) V * sqrt(D) * xi but scale is not needed
  for (int mm = 0; mm < xivec.length(); mm++)
    xivec[mm] = xivec[mm] * sqrt(eigenValues_[mm]) / 
                sqrt(snapMat_.ncols());
  eigenMatrix_.matvec(xivec, beta, 0);
  return 0;
}

// ************************************************************************
// get snapshot dimension
// ------------------------------------------------------------------------
int KPCA::getSnapshotDim()
{
  if (initialized_ == 0)
  {
    printf("KPCA getSnapshotDim ERROR: initialization not done yet.\n");
    exit(1);
  }
  return snapMat_.nrows();
}

// ************************************************************************
// get feature random variable values 
// ------------------------------------------------------------------------
int KPCA::getFeatureStdRVValues(psMatrix &ximat)
{
  int      ii, jj;
  psVector tmpVec;
  psMatrix featureMat;

  kernelMat_.matmult(eigenMatrix_, featureMat);
  featureMat.transpose();
  ximat.setFormat(PS_MAT2D);
  ximat.setDim(featureMat.ncols(), rDim_);
  for (ii = 0; ii < featureMat.ncols(); ii++)
  {
    featureMat.getCol(ii, tmpVec);
    for (jj = 0; jj < rDim_; jj++)
    {
      tmpVec[jj] = tmpVec[jj] / eigenValues_[jj] * 
                   sqrt(1.0*snapMat_.ncols());
      ximat.setEntry(ii, jj, tmpVec[jj]);
    }
  }
  return 0;
}

// ************************************************************************
// get number of snapshots 
// ------------------------------------------------------------------------
int KPCA::getNumSnapshots()
{
  if (initialized_ == 0)
  {
    printf("KPCA getNumSnapshots ERROR: initialization not done yet.\n");
    exit(1);
  }
  return snapMat_.ncols();
}

// ************************************************************************
// get reduced dimension
// ------------------------------------------------------------------------
int KPCA::getReducedDimension()
{
  return rDim_;
}

// ************************************************************************
// get snapshot 
// ------------------------------------------------------------------------
int KPCA::getSnapshot(int num, psVector &v1)
{
  if (num < 0 || num >= snapMat_.ncols())
  {
    printf("KPCA getSnapshot ERROR: invalid snapshot %d (%d)\n",
           num, snapMat_.ncols());
    exit(1);
  } 
  v1.setLength(snapMat_.nrows());
  for (int ii = 0; ii < snapMat_.nrows(); ii++)
    v1[ii] = snapMat_.getEntry(ii,num);
  return 0;
}

// ************************************************************************
// compute pre-image
// ------------------------------------------------------------------------
int KPCA::computePreImage(psVector &featureVec, psVector &vSoln)
{
  return computePreImage(featureVec, vSoln, 0);
}

// ************************************************************************
// compute pre-image
// ------------------------------------------------------------------------
int KPCA::computePreImage(psVector &featureVec, psVector &vSoln, int solver)
{
  int      iter=0, N, M, ii, jj, kk, npts, iprint=0, maxfun;
  double   err, *work, *xvalues, rhobeg, rhoend, rsum;
  psVector vdiff, beta, currSoln, prevSoln, tmpVec, xiVec;

  if (initialized_ == 0)
  {
    printf("KPCA computePreImage ERROR: initialization not done yet.\n");
    exit(1);
  }
  if (outputLevel_ > 0) printf("KPCA computePreImage begins\n");
  //featureVec.print("Source Feature");

  //**/ solver == 0 ==> fixed point iteration
  if (solver <= 0)
  {
    N = snapMat_.nrows();
    M = snapMat_.ncols();
    prevSoln.setLength(N);

    //**/ generate initial guess (average of all snapshots)
    for (ii = 0; ii < M; ii++)
    {
      getSnapshot(ii, tmpVec);
      for (kk = 0; kk < N; kk++)
        prevSoln[kk] = prevSoln[kk] + tmpVec[kk];
    }
    for (kk = 0; kk < N; kk++) prevSoln[kk] /= (double) M;

    //**/ compute beta from feature principal components (fpc)
    //**/ beta = V D^{-1} fpc - rsum + 1 / numSnaps
    tmpVec = featureVec;
    for (kk = 0; kk < tmpVec.length(); kk++) 
      tmpVec[kk] = tmpVec[kk] / eigenValues_[kk];
    eigenMatrix_.matvec(tmpVec, beta, 0);

    //**/ compute gamma 
    rsum = 0.0;
    for (kk = 0; kk < beta.length(); kk++) rsum += beta[kk];
    rsum /= (double) beta.length();
    for (kk = 0; kk < beta.length(); kk++) 
      beta[kk] = beta[kk] - rsum + 1.0 / (double) snapMat_.ncols();
     
    //**/ compute xi (for checking - xi not unique given a feature)
    //**/ eigenMatrix_.matvec(beta, xiVec, 1);
    //**/ for (kk = 0; kk < xiVec.length(); kk++) 
    //**/    xiVec[kk] = xiVec[kk] / sqrt(eigenValues_[kk]);
    //**/ xiVec.print("xi");

    //**/ iterate
    iter = 0;
    do
    {
      iter++;
      iterate(prevSoln, beta, currSoln);
      vdiff = prevSoln;
      vdiff.axpy(-1.0e0, currSoln);
      err = vdiff.norm(); 
      rsum = currSoln.norm(); 
      if (outputLevel_ > 0 || (iter > 1000 && (iter % 1000 == 0)))
      {
        printf("Iteration = %5d, error norm = %e\n", iter, err);
        if (iter > 1000 && (iter % 1000 == 0))
        {
          for (kk = 0; kk < 10; kk++) 
            printf("Prev, Curr Soln %2d = %e %e (%e)\n", kk+1, 
                   prevSoln[kk], currSoln[kk], 
                   prevSoln[kk]/currSoln[kk]);
        }
      }
      prevSoln = currSoln;
    }
    while (err/rsum >= convergenceTol_);
    vSoln = currSoln;
    //printf("Final Solution:\n");
    //vSoln.print("Final Solution");
  }
  else
  {
    printf("KPCA::ERROR: solver type %d not supported.\n", solver);
    exit(1);
  }
  if (outputLevel_ > 0) printf("KPCA computePreImage ends\n");
  return 0;
}

// ************************************************************************
// generate a snapshot from Xi
// ------------------------------------------------------------------------
int KPCA::genSnapshotFromStdRV(psVector &xiVec, psVector &vSoln)
{
  int    N, M, NW, ii, jj, kk, nn, iZero=0;
  double *work, ddata, minVal;
  psVector xiV2, solnBest, prevSoln, tmpVec;

  if (initialized_ == 0)
  {
    printf("KPCA genSnapshotFromStdRV ERROR: need initialize first.\n");
    exit(1);
  }
  if (outputLevel_ > 0) printf("KPCA genSnapshotFromStdRV begins\n");

  //**/ fixed point iteration
  N = snapMat_.nrows();
  M = snapMat_.ncols();
  prevSoln.setLength(N);
  NW = 3*(N+M);

  //**/ get ready to call gensnapshot function 
  vSoln = prevSoln;
  double *xvec = xiVec.getDVector();
  double *svec = vSoln.getDVector();
  double *ivec = prevSoln.getDVector();
  double *evs  = eigenValues_.getDVector();
  double *evecs = new double[M*rDim_];
  double *svecs = new double[N*M];
  double **emat = eigenMatrix_.getMatrix2D();
  for (ii = 0; ii < M; ii++)
    //for (jj = 0; jj < rDim_; jj++) evecs[jj*M+ii] = emat[jj][ii];
    for (jj = 0; jj < rDim_; jj++) evecs[jj*M+ii] = eigenMatrix_.getEntry(ii,jj);
  double **smat = snapMat_.getMatrix2D();
  for (ii = 0; ii < N; ii++)
    //for (jj = 0; jj < M; jj++) svecs[jj*N+ii] = smat[jj][ii];
    for (jj = 0; jj < M; jj++) svecs[jj*N+ii] = snapMat_.getEntry(ii,jj);
  work = new double[3*(N+M)];

  //**/ generate initial guess (average of all snapshots)
  minVal = 1e35;
  for (nn = 0; nn < 1; nn++)
  {
    //for (ii = 0; ii < M; ii++)
    //{
    //   getSnapshot(ii, tmpVec);
    //   for (kk = 0; kk < N; kk++)
    //      prevSoln[kk] = prevSoln[kk] + tmpVec[kk];
    //}
    //for (kk = 0; kk < N; kk++) prevSoln[kk] /= (double) M;
    for (kk = 0; kk < N; kk++) prevSoln[kk] = 10.0 * drand48() - 5.0;

    //computePreImage not as good in RMS error
    //computePreImage(xiVec, vSoln);
    gensnapshot_(&M,&rDim_,&N,&NW,xvec,evs,evecs,work,ivec,svecs,svec,
                 &trueScale_,&convergenceTol_,&iZero,&kernelType_); 

    //**/ check
    xiV2.setLength(xiVec.length());
    projectFeatureToStdRV(vSoln, xiV2);

    ddata = 0;
    for (ii = 0; ii < xiVec.length(); ii++)
      ddata += pow(xiVec[ii] - xiV2[ii], 2.0);
    ddata = sqrt(ddata/xiVec.length());
    printf("genSnapshotFromStdRV %d: Xi RMS ERROR = %e\n", nn+1, ddata);
    if (ddata < minVal)
    {
      minVal = ddata;
      solnBest = vSoln;
    }
  }
  vSoln = solnBest;

  //**/ clean up
  delete [] evecs;
  delete [] work;
  delete [] svecs;
  return 0;
}

// ************************************************************************
// get feature random variable values 
// ------------------------------------------------------------------------
int KPCA::projectFeatureToStdRV(psVector &featVec, psVector &xivec)
{
  int      ii, jj, N;
  double   rsum, ddata, matsum;
  psVector tmpVec, snapVec;
  psMatrix featureMat;

  //**/ inner product with the snapshots
  //**/ (phi_k - phi_mean)^t (phi - phi_mean) = phi_k^t phi -
  //**/  phi_mean^t phi - phi_k^t phi_mean + phi_mean^t phi_mean
  //**/ where phi_mean^t phi is in kernelMeans_,
  //**/       phi_k^t phi is to be computed below,

  //**/ compute phi_k^t phi and
  //**/         phi_k^t phi_mean = 1/N phi_kt phi 1 1^t
  //**/                          = average of tmpVec
  N = snapMat_.ncols();
  tmpVec.setLength(N);
  rsum = 0.0;
  for (ii = 0; ii < N; ii++) 
  {
    snapMat_.getCol(ii, snapVec);
    ddata = computeInnerProduct(snapVec, featVec);
    tmpVec[ii] = ddata;
    rsum += ddata;
  }
  rsum /= (double) N;

  //**/ compute phi_mean^t phi_mean
  matsum = 0.0;
  for (ii = 0; ii < N; ii++) matsum += kernelMeans_[ii];
  matsum /= (double) N;

  //**/ combine all (and make sure to scale by N)
  for (ii = 0; ii < N; ii++) 
  {
    ddata = tmpVec[ii] - kernelMeans_[ii] - rsum + matsum;
    tmpVec[ii] = ddata / (double) N;
  }

  //**/ project into the eigenspace
  xivec.setLength(rDim_);
  eigenMatrix_.matvec(tmpVec, xivec, 1);
  for (jj = 0; jj < rDim_; jj++)
    xivec[jj] = xivec[jj] / eigenValues_[jj] * sqrt(1.0*N);
  return 0;
}

// ************************************************************************
// iterate 
// ------------------------------------------------------------------------
int KPCA::iterate(psVector yk, psVector beta, psVector &ykp1)
{
  int      mm, ii, M, N;
  double   denom, prod;
  psVector yi, xx;

  M = featureMat_.ncols();
  N = yk.length();
  ykp1.setLength(N);
  if (kernelType_ == DA_GAUSSIAN)
  {
    xx.setLength(beta.length());
    for (mm = 0; mm < snapMat_.ncols(); mm++)
    {
      getSnapshot(mm, yi);
      prod = computeInnerProduct(yk, yi); 
      xx[mm] = prod * beta[mm];
      denom += xx[mm];
    }
    snapMat_.matvec(xx, ykp1, 0);
    for (mm = 0; mm < N; mm++) ykp1[mm] /= denom;
  }
  else
  {
    denom = 0.0;
    for (mm = 0; mm < snapMat_.ncols(); mm++)
    {
      getSnapshot(mm, yi);
      prod = computeInnerProductPrime(yk, yi); 
      for (ii = 0; ii < N; ii++) 
        ykp1[ii] += beta[mm] * prod * yi[ii];
      denom += beta[mm] * prod;
    }
    // scaling from the paper. Why this work? 
    denom = 1.0 / denom;
    //denom = 1.0 / computeInnerProductPrime(yk, yk); 
    for (ii = 0; ii < N; ii++) ykp1[ii] *= denom;
  }
  return 0;
}

// ************************************************************************
// inner product in the polynomial feature space
// ------------------------------------------------------------------------
double KPCA::computeInnerProduct(psVector &v1, psVector &v2)
{
  int      ii, N;
  double   prod, prod2;

  N  = v1.length();
  if (N != snapMat_.nrows())
  {
    printf("KPCA computeInnerProduct ERROR: invalid vector length.\n");
    exit(1);
  }
  if (N != v2.length())
  {
    printf("KPCA computeInnerProduct ERROR: invalid vec length.\n");
    exit(1);
  }
  if (kernelType_ == DA_GAUSSIAN)
  {
    prod = 0.0;
    for (ii = 0; ii < N; ii++) prod += pow(v1[ii]-v2[ii],2.0);
    prod /= (2.0 * trueScale_ * trueScale_);
    prod = exp(- prod);
  }
  else if (kernelType_ >= DA_LINEAR)
  {
    prod  = 1.0;
    prod2 = 0.0;
    for (ii = 0; ii < N; ii++) prod2 += (v1[ii] * v2[ii]);
    prod += prod2;
    if (kernelType_ >= DA_QUADRATIC) prod += prod2 * prod2;
    if (kernelType_ >= DA_CUBIC) prod += pow(prod2, 3.0);
    if (kernelType_ >= DA_P4) prod += pow(prod2, 4.0);
    if (kernelType_ >= DA_P5) prod += pow(prod2, 5.0);
  }
  return prod;
}

// ************************************************************************
// inner product derivative in the polynomial feature space
// ------------------------------------------------------------------------
double KPCA::computeInnerProductPrime(psVector &v1, psVector &v2)
{
  int    ii, N;
  double prod, prod2;

  N  = v1.length();
  if (N != snapMat_.nrows())
  {
    printf("KPCA computeInnerProductPrime ERROR: invalid vec length.\n");
    exit(1);
  }
  if (N != v2.length())
  {
    printf("KPCA computeInnerProductPrime ERROR: invalid vec length.\n");
    exit(1);
  }
  if (kernelType_ == DA_GAUSSIAN)
  {
    printf("computeInnerProductPrime ERROR: not available for Gaussian.\n");
    return 0.0;
  }
  else if (kernelType_ >= DA_LINEAR)
  {
    if (kernelType_ == DA_LINEAR) return 1.0;
    prod = 1.0;
    prod2 = 0.0;
    for (ii = 0; ii < N; ii++) prod2 += (v1[ii] * v2[ii]);
    prod += 2.0 * prod2;
    if (kernelType_ >= DA_CUBIC) prod += 3.0 * prod2 * prod2;
    if (kernelType_ >= DA_P4) prod += 4.0 * pow(prod2, 3.0);
    if (kernelType_ >= DA_P5) prod += 5.0 * pow(prod2, 4.0);
    return prod;
  }
  return 0.0;
}

// ************************************************************************
// get eigenvector matrix
// ------------------------------------------------------------------------
int KPCA::getEigenMatrix(psMatrix &mat)
{
  assert(eigenMatrix_.nrows() > 0 && eigenMatrix_.ncols() > 0);
  mat = eigenMatrix_;
  return 0;
}

// ************************************************************************
// get eigenvalues 
// ------------------------------------------------------------------------
int KPCA::getEigenValues(psVector &vec)
{
  assert(eigenValues_.length() > 0);
  vec = eigenValues_;
  return 0;
}

// ************************************************************************
// generate normal samples
// ------------------------------------------------------------------------
int KPCA::genNormalSample(psVector &samVec)
{
  int    cnt=0, leng;
  double low, range, iroot2, u1, u2, rr, pi=3.1415928, theta, z1, z2;

  leng = samVec.length();
  iroot2 = sqrt(0.5);
  low    = 0.5 * (1.0 + erf(-5.0*iroot2));
  range  = 0.5 * (1.0 + erf(5.0*iroot2)) - low;
  while (cnt < leng)
  {
    u1 = drand48() * range + low;
    u2 = drand48() * range + low;
    rr = sqrt(-2.0 * log(u1));
    theta = 2.0 * pi * u2;
    z1 = rr * cos(theta);
    z2 = rr * sin(theta);
    samVec[cnt] = z1;
    cnt++;
    if (cnt >= leng) break;
    samVec[cnt] = z2;
    cnt++;
  }
  return 0;
}

// ************************************************************************
// compute kernel matrix (K/N)
// ------------------------------------------------------------------------
int KPCA::computeKernelMatrix()
{
  int      ncols, nrows, cc, ii, jj, N;
  double   ddata, ddata2, *coldata;
  psVector snap1, snap2;
  psMatrix tmpMat;

  //**/ compute inner product
  ncols = snapMat_.ncols();
  nrows = snapMat_.nrows();
  kernelMat_.setFormat(PS_MAT2D);
  kernelMat_.setDim(ncols, ncols);
  //**/ compute distance^2
  trueScale_ = sqrt(0.5);
  for (cc = 0; cc < ncols; cc++) 
  {
    snapMat_.getCol(cc, snap1);
    for (ii = cc; ii < ncols; ii++) 
    {
      snapMat_.getCol(ii, snap2);
      if (kernelType_ == DA_GAUSSIAN) 
      {
        ddata = 0.0;
        for (jj = 0; jj < snap1.length(); jj++) 
          ddata += pow(snap1[jj]-snap2[jj],2.0);
        ddata = sqrt(ddata);
      }
      else ddata = computeInnerProduct(snap1, snap2);
      kernelMat_.setEntry(cc,ii,ddata);
      kernelMat_.setEntry(ii,cc,ddata);
    }
  }

  //**/ if the kernel is Gaussian, need some more postprocessing
  if (kernelType_ == DA_GAUSSIAN)
  {
    //**/ find scale
    trueScale_ = 0.0;
    for (cc = 0; cc < ncols; cc++) 
    {
      ddata = 1.0e35;
      for (ii = 0; ii < ncols; ii++) 
      {
        ddata2 = kernelMat_.getEntry(ii,cc);
        if (ii != cc && ddata2 < ddata) ddata = ddata2;
      }
      trueScale_ += ddata;
    }
    trueScale_ /= (double) ncols;
    //**/ scale = scale * gaussScale_ and generate exponentials
    trueScale_ *= gaussScale_;
    if (outputLevel_ > 0) 
      printf("KPCA::Gaussian scale = %e\n",trueScale_);
    for (cc = 0; cc < ncols; cc++) 
    {
      for (ii = cc; ii < ncols; ii++) 
      {
        ddata = kernelMat_.getEntry(cc,ii);
        ddata = exp(-ddata*ddata/(2*trueScale_*trueScale_));
        kernelMat_.setEntry(cc,ii,ddata);
        kernelMat_.setEntry(ii,cc,ddata);
      }
    }
  }

  //**/ normalize kernel matrix K = (K - One*K - K*One + One*K*One)/N
  //**/ kernel matrix is N x N where N is the number of snapshots.
  //**/ It is supposed (and forced) to be a symmetric matrix
  tmpMat = kernelMat_;
  N = kernelMat_.nrows();
  kernelMeans_.setLength(N);
  //**/ K = K - One*K
  for (ii = 0; ii < N; ii++) 
  {
    //**/ compute ii-th column average
    ddata = 0.0;
    for (jj = 0; jj < N; jj++) ddata += tmpMat.getEntry(jj,ii);
    ddata /= (double) N;
    kernelMeans_[ii] = ddata;
    //**/ each column subtract the same average
    for (jj = 0; jj < N; jj++) 
    {
      ddata2 = kernelMat_.getEntry(jj,ii) - ddata;
      kernelMat_.setEntry(jj,ii,ddata2);
    }
  }
  //**/ K = K - One*K - K*One 
  for (ii = 0; ii < N; ii++) 
  {
    //**/ compute ii-th row average
    ddata = 0.0;
    for (jj = 0; jj < N; jj++) ddata += tmpMat.getEntry(ii,jj);
    ddata /= (double) N;
    //**/ each row subtract the same average
    for (jj = 0; jj < N; jj++) 
    {
      ddata2 = kernelMat_.getEntry(ii,jj) - ddata;
      kernelMat_.setEntry(ii,jj,ddata2);
    }
  }
  //**/ normalize kernel matrix K = (K - One*K - K*One + One*K*One)/N
  //**/ but first compute the matrix average
  ddata = 0.0;
  for (ii = 0; ii < N; ii++) 
    for (jj = 0; jj < N; jj++) ddata += tmpMat.getEntry(ii,jj);
  ddata /= (double) (N * N);
  for (ii = 0; ii < N; ii++) 
  {
    for (jj = 0; jj < N; jj++) 
    {
      ddata2 = kernelMat_.getEntry(ii,jj);
      ddata2 = ddata2 + ddata;
      ddata2 /= (double) N;
      kernelMat_.setEntry(ii,jj,ddata2);
    }
  }
  // kernelMat_.print("Kernel matrix");
  return 0;
}

// ************************************************************************
// save KPCA information
// ------------------------------------------------------------------------
int KPCA::saveKPCA(char *filename)
{
  int    ii, jj;
  double ddata;
  FILE   *fp = fopen(filename, "w");
  if (fp == NULL)
  {
    printf("KPCA ERROR: cannot open file to save KPCA information\n");
    exit(1);
  }
  int N = snapMat_.nrows();
  int M = snapMat_.ncols();
  fprintf(fp, "#snapMap\n");
  fprintf(fp, "%d %d\n", N, M);
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < M; jj++)
    {
      ddata = snapMat_.getEntry(ii, jj);
      fprintf(fp, "%24.16e ",ddata);
    }
    fprintf(fp, "\n");
  }
  N = eigenMatrix_.nrows();
  M = eigenMatrix_.ncols();
  fprintf(fp, "#eigenmat\n");
  fprintf(fp, "%d %d\n", N, M);
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < M; jj++)
    {
      ddata = eigenMatrix_.getEntry(ii, jj);
      fprintf(fp, "%24.16e ",ddata);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "#eigenvalues\n");
  N = eigenValues_.length();
  fprintf(fp, "%d\n", N);
  for (ii = 0; ii < N; ii++)
    fprintf(fp, "%24.16e ",eigenValues_[ii]);
  fprintf(fp, "\n");

  N = kernelMat_.nrows();
  M = kernelMat_.ncols();
  fprintf(fp, "#kernelmat\n");
  fprintf(fp, "%d %d\n", N, M);
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < M; jj++)
    {
      ddata = kernelMat_.getEntry(ii, jj);
      fprintf(fp, "%24.16e ",ddata);
    }
    fprintf(fp, "\n");
  }
  fprintf(fp, "#kernelMeans\n");
  fprintf(fp, "%d\n", kernelMeans_.length());
  for (ii = 0; ii < kernelMeans_.length(); ii++)
    fprintf(fp, "%24.16e ",kernelMeans_[ii]);
  fprintf(fp, "\n");
  fprintf(fp, "rdim = %d\n",rDim_);
  fprintf(fp, "kernel = %d\n",kernelType_);
  fclose(fp);
  return 0;
}

// ************************************************************************
// read KPCA information
// ------------------------------------------------------------------------
int KPCA::retrieveKPCA(char *filename)
{
  int    ii, jj, M, N;
  double ddata;
  char   lineIn[1001], keyword[1000], equal[1000];
  FILE   *fp = fopen(filename, "r");

  rDim_ = 20;
  kernelType_ = 5;
  printf("KPCA: retrieving from %s\n", filename);
  if (fp == NULL)
  {
    printf("KPCA ERROR: cannot open file to retrieve KPCA information\n");
    exit(1);
  }

  //**/ first argument expected: snapMap
  fgets(lineIn, 1000, fp);
  sscanf(lineIn, "%s", keyword);
  if (strcmp(keyword, "#snapMap"))
  {
    printf("KPCA ERROR: KPCA file not in correct format\n");
    exit(1);
  }
  fscanf(fp, "%d %d", &N, &M);
  snapMat_.setFormat(PS_MAT2D);
  snapMat_.setDim(N, M);
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < M; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      snapMat_.setEntry(ii, jj, ddata);
    }
  }

  //**/ second argument expected: eigenmat
  fgets(lineIn, 1000, fp);
  sscanf(lineIn, "%s", keyword);
  if (strcmp(keyword, "#eigenmat"))
  {
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", keyword);
  }
  if (strcmp(keyword, "#eigenmat"))
  {
    printf("KPCA ERROR: KPCA file not in correct format (2)\n");
    exit(1);
  }
  fscanf(fp, "%d %d", &N, &M);
  eigenMatrix_.setFormat(PS_MAT2D);
  eigenMatrix_.setDim(N, M);
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < M; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      eigenMatrix_.setEntry(ii, jj, ddata);
    }
  }

  //**/ third argument expected: eigenvalues
  fgets(lineIn, 1000, fp);
  sscanf(lineIn, "%s", keyword);
  if (strcmp(keyword, "#eigenvalues"))
  {
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", keyword);
  }
  if (strcmp(keyword, "#eigenvalues"))
  {
    printf("KPCA ERROR: KPCA file not in correct format (3)\n");
    fclose(fp);
    exit(1);
  }
  fscanf(fp, "%d", &N);
  eigenValues_.setLength(N);
  for (ii = 0; ii < N; ii++)
  {
    fscanf(fp, "%lg", &ddata);
    eigenValues_[ii] = ddata;
  }

  //**/ fourth argument expected: kernelmat
  fgets(lineIn, 1000, fp);
  sscanf(lineIn, "%s", keyword);
  if (strcmp(keyword, "#kernelmat"))
  {
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", keyword);
  }
  if (strcmp(keyword, "#kernelmat"))
  {
    printf("KPCA ERROR: KPCA file not in correct format (4)\n");
    fclose(fp);
    exit(1);
  }
  fscanf(fp, "%d %d", &N, &M);
  kernelMat_.setFormat(PS_MAT2D);
  kernelMat_.setDim(N, M);
  for (ii = 0; ii < N; ii++)
  {
    for (jj = 0; jj < M; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      kernelMat_.setEntry(ii, jj, ddata);
    }
  }

  //**/ fifth argument expected: kernelMeans
  fgets(lineIn, 1000, fp);
  sscanf(lineIn, "%s", keyword);
  if (strcmp(keyword, "#kernelMeans"))
  {
    fgets(lineIn, 1000, fp);
    sscanf(lineIn, "%s", keyword);
  }
  if (strcmp(keyword, "#kernelMeans"))
  {
    printf("KPCA ERROR: KPCA file not in correct format (5)\n");
    fclose(fp);
    exit(1);
  }
  fscanf(fp, "%d", &N);
  kernelMeans_.setLength(N);
  for (ii = 0; ii < N; ii++)
  {
    fscanf(fp, "%lg", &ddata);
    kernelMeans_[ii] = ddata;
  }

  //**/ the rest
  while (feof(fp) == 0)
  {
    strcpy(lineIn, "none\0");
    strcpy(keyword, "none\0");
    fgets(lineIn,5000,fp);
    sscanf(lineIn, "%s", keyword);
    if (!strcmp(keyword, "rdim"))
    {
      sscanf(lineIn, "%s %s %d", keyword, equal, &rDim_);
      if (rDim_ <= 0)
        printf("KPCA ERROR: invalid reduced dimension\n");
    }
    if (!strcmp(keyword, "kernel"))
    {
      sscanf(lineIn, "%s %s %d", keyword, equal, &kernelType_);
      if (kernelType_ < 0 || kernelType_ > 5)
        printf("KPCA ERROR: invalid kernel (should be 0-5)\n");
    }
  }
  fclose(fp);
  printf("KPCA: retrieval completed\n");
  printf("  nSnaps = %d\n", N);
  printf("  kernel = %d\n", kernelType_);
  printf("  rdim   = %d\n", rDim_);
  initialized_ = 1;
  return 0;
}

// ************************************************************************
// create a KPCA model file
// ------------------------------------------------------------------------
int KPCA::genKPCAModelFile(psMatrix &matSnapshots, int kernel, int rdim,
                           char *modelFile)
{
  if (kernel < 0 || kernel > 5) 
  {
    printf("KPCA genKPCA ERROR: invalid kernel (should be 0-5)\n");
    exit(1);
  }
  printf("Creating KPCA Model file: \n");
  printf("Number of snapshots    = %d\n", matSnapshots.ncols());
  printf("Random field dimension = %d\n", matSnapshots.nrows());
  printf("KPCA kernel            = %d\n", kernel);
  printf("KPCA reduced dimension = %d\n", rdim);

  setOutputLevel(0);
  loadSnapshots(matSnapshots);
  setReducedDimension(rdim);
  setKernel(kernel);
  psMatrix featureMat;
  initialize(featureMat);
  saveKPCA(modelFile);
  return 0;
}

// ************************************************************************
// perform inversion 
// ------------------------------------------------------------------------
int KPCA::inverseKPCA(char *modelFile, psVector vecXi, psVector &vecSnap) 
{
  outputLevel_ = 0;
  //**/ retrieve KPCA information from file
  int status = retrieveKPCA(modelFile);
  //**/ error checking
  if (status != 0)
  {
    printf("KPCA inverseKPCA ERROR: cannot read model file %s\n",modelFile);
    exit(1);
  }
  printf("KPCA inverseKPCA INFO: feature file dimension = %d\n",rDim_);
  if (vecXi.length() != rDim_)
  {
    printf("KPCA inverseKPCA ERROR: feature dimension mismatch %d.\n",
           vecXi.length());
    exit(1);
  }
  //**/ perform inversion
  genSnapshotFromStdRV(vecXi, vecSnap);
  //**/ perform forward and compare
  psVector vecXiT;
  vecXiT.setLength(rDim_);
  projectFeatureToStdRV(vecSnap, vecXiT);
  double ddata = 0;
  for (int ii = 0; ii < rDim_; ii++) 
    ddata += pow(vecXi[ii] - vecXiT[ii],2.0);
  printf("Xi rms error = %e\n", sqrt(ddata/rDim_));
  return 0;
}

// ************************************************************************
// read snapshot file 
// ------------------------------------------------------------------------
int KPCA::readSnapshotsFile(char *snapshotName, psMatrix &Ysnapshots) 
{
  int    fieldDim, nSnaps;
  double ddata;
  FILE   *fp = fopen(snapshotName, "r");

  if (fp == NULL)
  {
    printf("KPCA ERROR: snapshot file %s not found.\n",snapshotName);
    return -1;
  }
  nSnaps = Ysnapshots.ncols();
  fieldDim = Ysnapshots.nrows();
  printf("KPCA Read Snapshots\n");
  printf("Number of snapshots = %d\n", nSnaps);
  printf("Snapshot dimension  = %d\n", fieldDim);
  for (int ii = 0; ii < nSnaps; ii++)
  {
    for (int jj = 0; jj < fieldDim; jj++)
    {
      fscanf(fp, "%lg", &ddata);
      Ysnapshots.setEntry(jj,ii,ddata);
    }
  }
  fclose(fp);
  return 0;
}

// ************************************************************************
// get kernel interactively
// ------------------------------------------------------------------------
int KPCA::getKernel()
{
  char pString[1001];
  printf("Select KPCA kernel. Available types are:\n");
  printf("   0: DA_GAUSSIAN \n");
  printf("   1: DA_LINEAR \n");
  printf("   2: DA_QUADRATIC \n");
  printf("   3: DA_CUBIC \n");
  printf("   4: DA_P4 \n");
  printf("   5: DA_P5 \n");
  snprintf(pString,1000,"Choose kernel (0 - 5) : ");
  int kernel = getInt(0, 5, pString);
  return kernel;
}


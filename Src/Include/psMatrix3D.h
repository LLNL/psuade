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
// psMatrix definition
// AUTHOR : CHARLES TONG
// DATE   : 2023
// ************************************************************************
#ifndef __MATRIX3DH__
#define __MATRIX3DH__

#include "psMatrix.h"

/**
 * @name psMatrix3D class
 *
 **/
/*@{*/

// ************************************************************************
// class definition
// ************************************************************************

class psMatrix3D
{
  int    nDim1_, nDim2_, nDim3_;
  double ***Mat3D_;

public:

  psMatrix3D();
  psMatrix3D(const psMatrix3D & ma);
  psMatrix3D & operator=(const psMatrix3D & ma);
  ~psMatrix3D();
  int    ndim1();
  int    ndim2();
  int    ndim3();
  int    load(psMatrix3D &);
  int    load(int, int, int, double ***);
  int    setDim(int, int, int);
  void   setEntry(const int, const int, const int, const double);
  double getEntry(const int, const int, const int);
  double ***getMatrix3D();
  double ***takeMatrix3D();
  void   collapse2D(int dim, psMatrix &);
  void   clean();
};

/*@}*/

#endif /* __MATRIX3DH__ */


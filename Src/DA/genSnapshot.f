C ***********************************************************************
C Copyright (c) 2015   Lawrence Livermore National Security, LLC.
C Produced at the Lawrence Livermore National Laboratory.
C Written by the PSUADE team.
C All rights reserved.
C
C Please see the COPYRIGHT_and_LICENSE file for the copyright notice,
C disclaimer, contact information and the GNU Lesser General Public
C License.
C
C DASSI is free software; you can redistribute it and/or modify it
C under the terms of the GNU General Public License (as published by 
C the Free Software Foundation) version 2.1 dated February 1999.
C
C DASSI is distributed in the hope that it will be useful, but WITHOUT
C ANY WARRANTY; without even the IMPLIED WARRANTY OF MERCHANTABILITY or
C FITNESS FOR A PARTICULAR PURPOSE.  See the terms and conditions of 
C the GNU General Public License for more details.
C 
C You should have received a copy of the GNU Lesser General Public
C License along with this program; if not, write to the Free Software
C Foundation, Inc., 59 Temple Place, Suite 330, Boston, MA 02111-1307 
C USA
C ***********************************************************************
C KPCA functions (supported by Xiao Chen's LDRD)
C DATE : Jan 2016
C ***********************************************************************
C     generate a snapshot given an independent random variable
C     m        - number of snapshots
C     p        - the reduced dimension
C     n        - dimension of the snapshots 
C     xi       - the set of independent random variable values
C     evalues  - p eigenvalues 
C     evectors - p eigenvectors (each of length m) 
C     work     - at least 3 * (n+m)
C     iguess   - initial guess
C     snaps    - the snapshots (m columns of n elements each)
C     soln     - solution (length n)
C     scal     - scaling factor for Gaussian kernel
C     tol      - convergence tolerance
C     plevel   - print level
C     kernel   - kernel 
C ***********************************************************************
      SUBROUTINE genSnapshot(m,p,n,nw,xi,evalues,evectors,work,iguess,
     C           snaps,soln,scal,tol,plevel,kernel)
      integer m, n, p, its, plevel, kernel, nw
      double precision xi(p), evalues(*), evectors(m,*), iguess(*)
      double precision work(nw), dsum, soln(n), vsum, tol, innerp, scal
      double precision dtemp, snaps(n,*), err1
C
C     compute beta (work(p:p+m)) from xi (beta = 1/sqrt(m) V sqrt(ev) xi)
C
      DO I = 1, p
C        work(I) = xi(I) * SQRT(evalues(I) / m) 
         work(I) = xi(I) / SQRT(1.0d0*m) 
      END DO
      DO I = 1, m
         work(p+I) = 0.0d0
         DO J = 1, p
            work(p+I) = work(p+I) + evectors(I,J) * work(J)
         END DO
      END DO
C
C     compute gamma (work(1:m)) from beta (work(p:p+m))
C
      dsum = 0.0d0
      DO I = 1, m
         dsum = dsum + work(p+I)
      END DO
      dsum = dsum / m
      DO I = 1, m
         work(I) = work(p+I) - dsum + 1.0 / m
      END DO
C
C     set yk = initial guess (work(m+1:m+n))
C
      DO I = 1, n
         work(m+I) = iguess(I)
      END DO
C
C     iterate
C
      its = 0
      err1 = 1.0d0
      vsum = 1.0d0

C     DO WHILE (err1/vsum .gt. tol)
      DO WHILE (err1 .gt. tol)
         its  = its + 1
C
C        compute ykp1
C        1. set ykp1 = 0 (work(m+n+1:m+2*n))
C
         DO J = 1, n
            work(n+m+J) = 0
         END DO
C
C        2.accumulate numerator (work(m+n+1:m+2n) and denominator (vsum)
C
         vsum = 0.0d0
         DO I = 1, m
C           compute inner product
            IF (kernel .eq. 0) THEN
               innerp = 0.0d0
               DO J = 1, n
                  dtemp = (work(m+J) - snaps(J,I))**2
                  innerp = innerp + dtemp
               END DO
               innerp = innerp / (2 * scal * scal)
               innerp = exp(- innerp)
               vsum = vsum + innerp * work(I)
            ELSEIF (kernel .ge. 1) THEN
               innerp = 1.0d0
               IF (kernel .ge. 2) THEN
                  dtemp = 0.0d0
                  DO J = 1, n
                     dtemp = dtemp + work(m+J) * snaps(J,I)
                  END DO
                  innerp = innerp + 2.0d0 * dtemp
                  IF (kernel .gt. 2) THEN
                     nn = kernel
                     DO WHILE (nn .gt. 2)
                       nn = nn - 1
                       innerp = innerp + (2.0 + kernel - nn) *
     C                          dtemp**(1.0+kernel-nn)
                     END DO
                  ENDIF
               ENDIF
C              Equation (29)
               vsum = vsum + innerp * work(I)
            ENDIF
            DO J = 1, n
               work(m+n+J)=work(m+n+J)+innerp*work(I)*snaps(J,I) 
            END DO
         END DO
C 
C        calculate denominator for polynomial kernel for Eqn (28)
C        It seems equation (29) works better
C
C        IF (kernel .ge. 1) THEN
C           vsum = 1.0d0
C           IF (kernel .ge. 2) THEN
C              dtemp = 0.0d0
C              DO I = 1, n
C                 dtemp = dtemp + work(m+I) * work(m+I)
C              END DO
C              vsum = vsum + 2.0d0 * dtemp
C              IF (kernel .gt. 2) THEN
C                 nn = kernel
C                 DO WHILE (nn .gt. 2)
C                    nn = nn - 1
C                    vsum = vsum + (2.0 + kernel - nn) *
C    C                      dtemp**(1.0+kernel-nn)
C                 END DO
C              ENDIF
C           ENDIF
C        ENDIF
C 
C        3. scale ykp1
C
         DO I = 1, n
            work(m+n+I) = work(m+n+I) / vsum
         END DO
C 
C        4. convergence check
C
         err1 = 0.0d0
         DO I = 1, n
            err1 = err1 + (work(m+I) - work(n+m+I))**2
C           if (I .le. 10) then
C              print *, work(m+I), work(n+m+I)
C           endif
         END DO
         err1 = SQRT(err1)
         IF ((plevel .gt. 1) .or. 
     C       ((its .gt. 1000) .and. (mod(its,1000) .eq. 0))) THEN
           PRINT *,"genSnapShot iteration = ",its,"  error norm = ",err1
         ENDIF
         IF ((plevel .ge. 1) .and. (ITS .EQ. 1)) THEN
            PRINT *,"genSnapShot initial error norm = ",err1
         ENDIF
C
C        5. yk = ykp1
C
         DO I = 1, n
            work(m+I) = work(n+m+I)
         END DO
         IF (ITS .EQ. 5000) THEN
            PRINT *,"INFO: genSnapShot not converged in 5k iterations"
            exit
         ENDIF
      END DO
C     IF ((plevel .ge. 1) .and. (ITS .EQ. 1)) THEN
      IF (plevel .ge. 1) THEN
         PRINT *,"genSnapShot total iterations = ",its,
     +           "  final error norm = ",err1
         PRINT *,"Final relative error = ",err1/vsum
      ENDIF
C
C     Copy result to solution vector 
      DO I = 1, n
         soln(I) = work(n+m+I)
      END DO
      return
      END


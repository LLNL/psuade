      SUBROUTINE NCALFUN (N,X,F)
      IMPLICIT REAL*8 (A-H,O-Z)
      DIMENSION X(*),Y(10,10)
      COMMON FMODE, OFMODE, KFMODE, NFMODE
      IF (NFMODE .EQ. 7777) THEN
C        Call NEWUOA optimizer 
         CALL NewuoaEvalFunc(N, X, F);
      ELSE IF (NFMODE .EQ. 8888) THEN
C        Call OUU 
         CALL OUUEvalFunc(N, X, F);
      ELSE IF (NFMODE .EQ. 6666) THEN
C        Call Kriging 
         CALL KRIBOBYQAEvalFunc(N, X, F);
      ELSE IF (NFMODE .EQ. 5555) THEN
C        Call GP3 
         CALL GP3NEWUOAEvalFunc(N, X, F);
      ELSE IF (NFMODE .EQ. 4444) THEN
C        Call Homogeneous Kriging
         CALL HKRINEWUOAEvalFunc(N, X, F);
      ELSE IF (NFMODE .EQ. 9999) THEN
C        Call PsuadeRegression
         CALL PSRNEWUOAEvalFunc(N, X, F);
      ENDIF
      RETURN
C     DO 10 J=1,N
C     Y(1,J)=1.0D0
C  10 Y(2,J)=2.0D0*X(J)-1.0D0
C     DO 20 I=2,N
C     DO 20 J=1,N
C  20 Y(I+1,J)=2.0D0*Y(2,J)*Y(I,J)-Y(I-1,J)
C     F=0.0D0
C     NP=N+1
C     IW=1
C     DO 40 I=1,NP
C     SUM=0.0D0
C     DO 30 J=1,N
C  30 SUM=SUM+Y(I,J)
C     SUM=SUM/DFLOAT(N)
C     IF (IW .GT. 0) SUM=SUM+1.0D0/DFLOAT(I*I-2*I)
C     IW=-IW
C  40 F=F+SUM*SUM
C     RETURN
      END

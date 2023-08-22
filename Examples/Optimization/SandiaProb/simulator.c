#include <math.h>
#include <stdio.h>
#include <stdlib.h>

main(int argc, char **argv)
{
   int    count, i;
   double X[2], Y, R, theta, pi=3.1459;
   FILE   *fIn  = fopen(argv[1], "r");
   FILE   *fOut = fopen(argv[2], "w");
   char   lineIn[100], stringPtr[100], equal[2];

   if (fIn == NULL || fOut == NULL)
   {
      printf("Simulator ERROR - cannot open in/out files.\n");
      exit(1);
   }
   fscanf(fIn, "%d", &count);
   for (i = 0; i < 2; i++) fscanf(fIn, "%lg", &X[i]);
   theta = atan2(X[1],X[0]);
   R = sqrt(X[0] * X[0] + X[1] * X[1]);
   Y = - (0.8 * R + 0.35 * sin(2.4*pi*R/sqrt(2.0))) * (1.5 * sin(1.3*theta));
   fprintf(fOut, "%e\n", Y);
   fclose(fIn);   
   fclose(fOut);   
}


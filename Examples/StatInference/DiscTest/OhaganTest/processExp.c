#include <math.h>
#include <stdio.h>
#include <stdlib.h>

double gaussrand();
double genNormal();      
#define PABS(x) ((x > 0) ? (x) : -(x))

int main(int argc, char **argv)
{
   int    ss, ii, nSams, nInps, nOuts;
   double X, Y;
   FILE   *fIn = fopen("exp.std", "r");
   FILE   *fOut;
                                                                                
   if (fIn == NULL)
   {
      printf("processExp ERROR - cannot open input file.\n");
      exit(1);
   }
   fOut = fopen("expdata", "w");
   if (fOut == NULL)
   {
      printf("processExp ERROR - cannot open output file.\n");
      exit(1);
   }
   fscanf(fIn, "%d %d %d", &nSams,&nInps,&nOuts);
   if (nSams <= 0 || nInps <= 0 || nOuts <= 0)
   {
      printf("processExp ERROR - wrong input file.\n");
      exit(1);
   }
   fprintf(fOut, "%d %d 1 1\n", nSams, nOuts);
   for (ss = 0; ss < nSams; ss++) 
   {
     fprintf(fOut, "%d ", ss+1);
     fscanf(fIn, "%lg", &X);
     fprintf(fOut, "%16.8e ", X);
     for (ii = 0; ii < nOuts; ii++) 
     {
       fscanf(fIn, "%lg", &Y);
       fprintf(fOut, "%16.8e %16.8e", Y, 0.01*Y);
     }
     fprintf(fOut, "\n");
   }
   fclose(fIn);
   fclose(fOut);
}


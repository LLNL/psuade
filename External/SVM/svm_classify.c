/***********************************************************************/
/*                                                                     */
/*   svm_classify.c                                                    */
/*                                                                     */
/*   Classification module of Support Vector Machine.                  */
/*                                                                     */
/*   Author: Thorsten Joachims                                         */
/*   Date: 02.07.02                                                    */
/*                                                                     */
/*   Copyright (c) 2002  Thorsten Joachims - All rights reserved       */
/*                                                                     */
/*   This software is available for non-commercial use only. It must   */
/*   not be modified and distributed without prior permission of the   */
/*   author. The author is not responsible for implications from the   */
/*   use of this software.                                             */
/*                                                                     */
/************************************************************************/

# include "svm_common.h"

char docfile[200];
char modelfile[200];
char predictionsfile[200];

void read_input_parameters2(int, char **, char *, char *, char *, long *, 
			   long *);
void print_help(void);


#ifdef ORIGINAL
int main (int argc, char* argv[])
#else
void SVMInterp(int nTests, int nInputs, double *testInputs,
               double *testOutput, double *stds)
#endif
{
  DOC *doc;   /* test example */
  WORD *words;
#ifdef ORIGINAL
  long max_docs,max_words_doc,lld;
#else
  long max_words_doc,lld;
#endif
  long totdoc=0,queryid,slackid;
#ifdef ORIGINAL
  long correct=0,incorrect=0,no_accuracy=0;
  long res_a=0,res_b=0,res_c=0,res_d=0,wnum,pred_format;
#else
  long wnum,pred_format;
#endif
  long j;
  double t1,runtime=0;
  double dist,doc_label,costfactor;
  char *line,*comment; 
#ifdef ORIGINAL
  FILE *predfl,*docfl;
#endif
  MODEL *model; 
  int  ii, jj;

#ifdef ORIGINAL
  read_input_parameters2(argc,argv,docfile,modelfile,predictionsfile,
			&verbosity,&pred_format);
  nol_ll(docfile,&max_docs,&max_words_doc,&lld); /* scan size of input file */
#else
  verbosity = 0;
  pred_format = 1;
  max_words_doc=nInputs;
  lld=100;
#endif
  max_words_doc+=2;
  lld+=2;

  line = (char *)my_malloc(sizeof(char)*lld);
  words = (WORD *)my_malloc(sizeof(WORD)*(max_words_doc));
  comment = (char *)malloc(sizeof(char)*10);

#ifdef ORIGINAL
  model=read_model(modelfile);
#else
  model=SVM_model;
  if (model == NULL) {
     printf("SVM interpolation error: no model.\n");
     exit(1);
  }
#endif

  if(model->kernel_parm.kernel_type == 0) { /* linear kernel */
    /* compute weight vector */
    add_weight_vector_to_linear_model(model);
  }
  
  if(verbosity>=2) {
    printf("Classifying test examples.."); fflush(stdout);
  }

#ifdef ORIGINAL
  if ((docfl = fopen (docfile, "r")) == NULL)
  { perror (docfile); exit (1); }
  if ((predfl = fopen (predictionsfile, "w")) == NULL)
  { perror (predictionsfile); exit (1); }
#endif

#ifdef ORIGINAL
  while((!feof(docfl)) && fgets(line,(int)lld,docfl)) {
#else
  for (ii = 0; ii < nTests; ii++) {
#endif
#ifdef ORIGINAL
    if(line[0] == '#') continue;  /* line contains comments */
    parse_document(line,words,&doc_label,&queryid,&slackid,&costfactor,&wnum,
		   max_words_doc,&comment);
#else
    doc_label = 0.0;
    queryid = 0;
    slackid = 0;
    costfactor = 1.0;
    wnum = nInputs + 1;
    comment[0] = '\0';
    words[nInputs].wnum = 0;
    for (jj=0; jj<nInputs;jj++){
       words[jj].weight = testInputs[ii*nInputs+jj];
       words[jj].wnum = jj+1;
    }
#endif
    totdoc++;
    if(model->kernel_parm.kernel_type == 0) {   /* linear kernel */
      for(j=0;(words[j]).wnum != 0;j++) {  /* Check if feature numbers   */
	if((words[j]).wnum>model->totwords) /* are not larger than in     */
	  (words[j]).wnum=0;               /* model. Remove feature if   */
      }                                        /* necessary.                 */
      doc = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));
      t1=get_runtime();
      dist=classify_example_linear(model,doc);
      runtime+=(get_runtime()-t1);
      free_example(doc,1);
    }
    else {                             /* non-linear kernel */
      doc = create_example(-1,0,0,0.0,create_svector(words,comment,1.0));
      t1=get_runtime();
      dist=classify_example(model,doc);
      runtime+=(get_runtime()-t1);
      free_example(doc,0);
    }
#ifdef ORIGINAL
    if(dist>0) {
      if(pred_format==0) { /* old weired output format */
	fprintf(predfl,"%.8g:+1 %.8g:-1\n",dist,-dist);
      }
      if(doc_label>0) correct++; else incorrect++;
      if(doc_label>0) res_a++; else res_b++;
    }
    else {
      if(pred_format==0) { /* old weired output format */
	fprintf(predfl,"%.8g:-1 %.8g:+1\n",-dist,dist);
      }
      if(doc_label<0) correct++; else incorrect++;
      if(doc_label>0) res_c++; else res_d++;
    }
    if(pred_format==1) { /* output the value of decision function */
      fprintf(predfl,"%.8g\n",dist);
    }
    if((int)(0.01+(doc_label*doc_label)) != 1) 
      { no_accuracy=1; } /* test data is not binary labeled */
    if(verbosity>=2) {
      if(totdoc % 100 == 0) {
	printf("%ld..",totdoc); fflush(stdout);
      }
    }
#else
    testOutput[ii] = dist;
    if (stds != NULL) stds[ii] = 0.0;
#endif
  }  
  free(line);
  free(words);
  free(comment);
  if (nTests <= 0)
  {
     free_model(SVM_model,1);
     SVM_model = NULL;
     for(ii=0;ii<SVM_totdoc;ii++)
        free_example(SVM_docs[ii],1);
     free(SVM_docs);
     SVM_docs = NULL;
  }

  if(verbosity>=2) {
    printf("done\n");

/*   Note by Gary Boone                     Date: 29 April 2000        */
/*      o Timing is inaccurate. The timer has 0.01 second resolution.  */
/*        Because classification of a single vector takes less than    */
/*        0.01 secs, the timer was underflowing.                       */
    printf("Runtime (without IO) in cpu-seconds: %.2f\n",
	   (float)(runtime/100.0));
    
  }
#ifdef ORIGINAL
  if((!no_accuracy) && (verbosity>=1)) {
    printf("Accuracy on test set: %.2f%% (%ld correct, %ld incorrect, %ld total)\n",(float)(correct)*100.0/totdoc,correct,incorrect,totdoc);
    printf("Precision/recall on test set: %.2f%%/%.2f%%\n",(float)(res_a)*100.0/(res_a+res_b),(float)(res_a)*100.0/(res_a+res_c));
  }
  return(0);
#endif
}

void read_input_parameters2(int argc, char **argv, char *docfile, 
			   char *modelfile, char *predictionsfile, 
			   long int *verbosity, long int *pred_format)
{
  long i;
  
  /* set default */
  strcpy (modelfile, "svm_model");
  strcpy (predictionsfile, "svm_predictions"); 
  (*verbosity)=2;
  (*pred_format)=1;

  for(i=1;(i<argc) && ((argv[i])[0] == '-');i++) {
    switch ((argv[i])[1]) 
      { 
      case 'h': print_help(); exit(0);
      case 'v': i++; (*verbosity)=atol(argv[i]); break;
      case 'f': i++; (*pred_format)=atol(argv[i]); break;
      default: printf("\nUnrecognized option %s!\n\n",argv[i]);
	       print_help();
	       exit(0);
      }
  }
  if((i+1)>=argc) {
    printf("\nNot enough input parameters!\n\n");
    print_help();
    exit(0);
  }
  strcpy (docfile, argv[i]);
  strcpy (modelfile, argv[i+1]);
  if((i+2)<argc) {
    strcpy (predictionsfile, argv[i+2]);
  }
  if(((*pred_format) != 0) && ((*pred_format) != 1)) {
    printf("\nOutput format can only take the values 0 or 1!\n\n");
    print_help();
    exit(0);
  }
}

#ifdef ORIGINAL
void print_help(void)
{
  printf("\nSVM-light %s: Support Vector Machine, classification module     %s\n",VERSION,VERSION_DATE);
  copyright_notice();
  printf("   usage: svm_classify [options] example_file model_file output_file\n\n");
  printf("options: -h         -> this help\n");
  printf("         -v [0..3]  -> verbosity level (default 2)\n");
  printf("         -f [0,1]   -> 0: old output format of V1.0\n");
  printf("                    -> 1: output the value of decision function (default)\n\n");
}
#endif




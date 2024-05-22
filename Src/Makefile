#########################################################
# Makefile for PSUADE
#########################################################

SHELL = /bin/sh

.SUFFIXES: .c .cpp .f .o .cxx 

SRCDIR = .

INCLUDES  = -I$(SRCDIR) -I$(SRCDIR)/Include
CFLAGS2   = $(INCLUDES) -D$(ARCH)
CXXFLAGS2 = $(INCLUDES) -D$(ARCH)
FFLAGS2   = $(INCLUDES) -D$(ARCH)

#########################################################
# General c, c++, and fortran suffix rules
#########################################################

.c.o:
	@echo "Making $@"
	$(CC) -c $(CFLAGS) $(CFLAGS2) $(DEFINES) -o $@ $<
.cpp.o:
	@echo "Making $@"
	$(CXX) -c $(CXXFLAGS) $(CXXFLAGS2) $(DEFINES) -o $@ $<
.cxx.o:
	@echo "Making $@"
	$(CXX) -c $(CXXFLAGS) $(CXXFLAGS2) $(DEFINES) -o $@ $<
.f.o:
	@echo "Making $@"
	$(F77) -c $(FFLAGS) $(FFLAGS2) -o $@ $<


#########################################################
# a list of C and Fortran files
########################################################

OBJ_MAIN = Main/Psuade.o
OBJ_BASE = Base/PsuadeBase.o Base/PsuadeInterpreter.o \
           Base/PsuadeGenDriver.o Base/PsuadeGradRSM.o \
	   Base/Globals.o
OBJ_SAMP = Samplings/Sampling.o \
           Samplings/MCSampling.o \
           Samplings/FactorialSampling.o \
           Samplings/LHSampling.o \
           Samplings/OASampling.o \
           Samplings/OALHSampling.o \
           Samplings/MOATSampling.o \
           Samplings/SobolSampling.o \
           Samplings/LPtauSampling.o \
           Samplings/MetisSampling.o \
           Samplings/FASTSampling.o \
           Samplings/SFASTSampling.o \
           Samplings/GMOATSampling.o \
           Samplings/BoxBehnkenSampling.o \
           Samplings/PlackettBurmanSampling.o \
           Samplings/FractFactSampling.o \
           Samplings/CentralCompositeSampling.o \
           Samplings/UserMetisSampling.o \
           Samplings/GMetisSampling.o \
           Samplings/SparseGridSampling.o \
           Samplings/DiscreteSampling.o \
           Samplings/LSASampling.o \
           Samplings/RFractFactSampling.o \
           Samplings/RSConstraints.o \
           Samplings/MOATConstraints.o \
           Samplings/GridNode.o
OBJ_PDF =  PDFLib/PDFBase.o \
           PDFLib/PDFNormal.o \
           PDFLib/PDFLogNormal.o \
           PDFLib/PDFTriangle.o \
           PDFLib/PDFBeta.o \
           PDFLib/PDFWeibull.o \
           PDFLib/PDFMVNormal.o \
           PDFLib/PDFMVLogNormal.o \
           PDFLib/PDFGamma.o \
           PDFLib/PDFExponential.o \
           PDFLib/PDFSpecial.o \
           PDFLib/PDFManager.o \
           PDFLib/pdfData.o
OBJ_UTIL = Util/PsuadeUtil.o Util/isaac_randport.o Util/cygwin_system.o \
           Util/PsuadeParser.o Util/Matrix.o Util/Vector.o Util/iVector.o \
           Util/gamma_function.o Util/incomplete_gamma_function.o \
           Util/entire_incomplete_gamma_function.o Util/ln_gamma_function.o \
           Util/incomplete_beta_function.o Util/beta_function.o \
           Util/factorial.o
OBJ_G2C  = Util/f2cStuff.o
OBJ_PSIO = DataIO/PsuadeData.o DataIO/FunctionInterface.o \
           DataIO/pData.o DataIO/PsuadeConfig.o
OBJ_PLTC = Plot/plotbogus.o
OBJ_PLTF = Plot/pgplotlink.o
OBJ_ANAL = Analysis/AnalysisManager.o \
           Analysis/Analyzer.o \
           Analysis/aData.o \
           Analysis/MomentAnalyzer.o \
           Analysis/MainEffectAnalyzer.o \
           Analysis/TwoParamAnalyzer.o \
           Analysis/AnovaAnalyzer.o \
           Analysis/GradStatAnalyzer.o \
           Analysis/CorrelationAnalyzer.o \
           Analysis/RSFuncApproxAnalyzer.o \
           Analysis/MOATAnalyzer.o \
           Analysis/SobolAnalyzer.o \
           Analysis/IntegrationAnalyzer.o \
           Analysis/FASTAnalyzer.o \
           Analysis/FFAnalyzer.o \
           Analysis/LSAnalyzer.o \
           Analysis/PCAnalyzer.o \
           Analysis/OneSigmaAnalyzer.o \
           Analysis/FORMAnalyzer.o \
           Analysis/RSMSobol1Analyzer.o \
           Analysis/RSMSobol2Analyzer.o \
           Analysis/RSMSobolTSIAnalyzer.o \
           Analysis/TSIAnalyzer.o \
           Analysis/BootstrapAnalyzer.o \
           Analysis/RSMSobolGAnalyzer.o \
           Analysis/BinomialAnalyzer.o \
           Analysis/OneSampleAnalyzer.o \
           Analysis/TwoSampleAnalyzer.o \
           Analysis/MCMCAnalyzer.o \
           Analysis/DeltaAnalyzer.o \
           Analysis/GowerAnalyzer.o \
           Analysis/EtaAnalyzer.o
OBJ_FACC = FuncApprox/MarsInterface.o
OBJ_COMM = Comm/BaseComm.o \
           Comm/CommManager.o \
           Comm/CommMPICH.o \
           Comm/JobCntl.o
OBJ_FACP = FuncApprox/Mars.o \
           FuncApprox/FuncApprox.o \
           FuncApprox/FuncApproxFilter.o \
           FuncApprox/Regression.o \
           FuncApprox/SelectiveRegression.o \
           FuncApprox/LegendreRegression.o \
           FuncApprox/GradLegendreRegression.o \
           FuncApprox/UserRegression.o \
           FuncApprox/SGRegression.o \
           FuncApprox/Ann.o \
           FuncApprox/GP1.o \
           FuncApprox/GP2.o \
           FuncApprox/TBGP.o \
           FuncApprox/SVM.o \
           FuncApprox/PWLinear.o \
           FuncApprox/MarsBagg.o \
           FuncApprox/Earth.o \
           FuncApprox/SumOfTrees.o \
           FuncApprox/Kriging.o \
           FuncApprox/NPLearning.o \
           FuncApprox/TreeNode.o \
           FuncApprox/AnnFunctions.o
OBJ_OPT  = Optimizers/TxMathOptimizer.o Optimizers/TxMathPSUADE.o \
           Optimizers/APPSPACKOptimizer.o Optimizers/MinpackOptimizer.o \
           Optimizers/CobylaOptimizer.o Optimizers/SMOptimizer.o \
           Optimizers/MMOptimizer.o Optimizers/Optimizer.o \
           Optimizers/BobyqaOptimizer.o Optimizers/SCEOptimizer.o \
           Optimizers/MultiObjectiveOptimizer.o Optimizers/oData.o
OBJ_PYTH = PythonWrapper/psuade_wrap.o

OBJ_ALL  = $(OBJ_BASE) $(OBJ_SAMP) $(OBJ_UTIL) $(OBJ_PSIO) $(OBJ_PLTC) \
           $(OBJ_PLTF) $(OBJ_ANAL) $(OBJ_FACC) $(OBJ_FACP) $(OBJ_OPT) \
           $(OBJ_MAIN) $(OBJ_PDF) $(OBJ_COMM) $(OBJ_PYTH) $(OBJ_G2C)

OBJ_NPLT = $(OBJ_BASE) $(OBJ_SAMP) $(OBJ_UTIL) $(OBJ_PSIO) $(OBJ_PLTC) \
           $(OBJ_ANAL) $(OBJ_FACC) $(OBJ_FACP) $(OBJ_OPT) $(OBJ_COMM) \
           $(OBJ_PDF) 

OBJ_PLOT = $(OBJ_BASE) $(OBJ_SAMP) $(OBJ_UTIL) $(OBJ_PSIO) $(OBJ_PLTF) \
           $(OBJ_ANAL) $(OBJ_FACC) $(OBJ_FACP) $(OBJ_OPT) $(OBJ_COMM) \
           $(OBJ_PDF)

ifeq ($(HAVE_PYTH),1)
  PYTH = $(OBJ_PYTH)
else
  PYTH =
endif

#########################################################
# main library
#########################################################

LIBPSUADE_A1 = libpsuadePlot.a
LIBPSUADE_A2 = libpsuadeNoPlot.a

libpsuadePlot : copy_include $(OBJ_PLOT) $(PYTH) $(OBJ_G2C)
	$(AR) ruv Lib/$(LIBPSUADE_A1) $(OBJ_PLOT)
	$(RANLIB) Lib/$(LIBPSUADE_A1)
	$(AR) ruv Lib/libg2cutil.a $(OBJ_G2C)
	$(RANLIB) Lib/libg2cutil.a

libpsuadeNoPlot : copy_include $(OBJ_NPLT) $(PYTH) $(OBJ_G2C)
	$(AR) ruv Lib/$(LIBPSUADE_A2) $(OBJ_NPLT)
	$(RANLIB) Lib/$(LIBPSUADE_A2)
	$(AR) ruv Lib/$(LIBPSUADE_A2) $(OBJ_NPLT)
	$(AR) ruv Lib/libg2cutil.a $(OBJ_G2C)
	$(RANLIB) Lib/libg2cutil.a

copy_include:
	rm -f Include/*.h
	cp -f */*.h Include

clean:
	@rm -f $(OBJ_ALL)
	@rm -f Lib/*.a
	@rm -f Lib/*.so
	@rm -f Include/*.h
	@rm -f Optimizers/psuade_install_dir
	(cd Main; make clean)

#########################################################
# individual rules
#########################################################

Main/Psuade.o : Main/Psuade.cpp Main/Psuade.h Base/PsuadeBase.h \
                Base/Globals.h Util/dtype.h Util/sysdef.h

Base/PsuadeBase.o : Base/PsuadeBase.cpp Base/PsuadeBase.h Base/Globals.h Util/dtype.h \
                Util/sysdef.h Util/Exceptions.h DataIO/PsuadeData.h \
                DataIO/pData.h DataIO/FunctionInterface.h \
                Samplings/Sampling.h PDFLib/pdfData.h PDFLib/PDFBase.h \
                PDFLib/PDFNormal.h PDFLib/PDFLogNormal.h \
                PDFLib/PDFTriangle.h PDFLib/PDFBeta.h PDFLib/PDFMVNormal.h

Base/PsuadeInterpreter.o : Base/PsuadeInterpreter.cpp Base/Globals.h Base/PsuadeBase.h \
                Util/dtype.h Util/sysdef.h DataIO/pData.h

Base/PsuadeGenDriver.o : Base/PsuadeGenDriver.cpp Base/Globals.h Base/PsuadeBase.h 

Base/PsuadeGradRSM.o : Base/PsuadeGradRSM.cpp Base/Globals.h Base/PsuadeBase.h

Samplings/Sampling.o : Samplings/Sampling.cpp Samplings/Sampling.h

Samplings/MCSampling.o : Samplings/MCSampling.cpp Samplings/MCSampling.h \
                         Samplings/Sampling.h \
                         Util/sysdef.h Util/PsuadeUtil.h

Samplings/FactorialSampling.o : Samplings/FactorialSampling.cpp Util/sysdef.h \
                                Samplings/Sampling.h Util/PsuadeUtil.h \
                                Samplings/FactorialSampling.h 

Samplings/LHSampling.o : Samplings/LHSampling.cpp Samplings/LHSampling.h \
                         Samplings/Sampling.h Util/sysdef.h \
                         Util/PsuadeUtil.h

Samplings/OASampling.o : Samplings/OASampling.cpp Samplings/OASampling.h \
                         Samplings/Sampling.h Util/sysdef.h \
                         Util/PsuadeUtil.h

Samplings/OALHSampling.o : Samplings/OALHSampling.cpp Samplings/OALHSampling.h \
                           Samplings/Sampling.h Util/sysdef.h \
                           Util/PsuadeUtil.h

Samplings/MOATSampling.o : Samplings/MOATSampling.cpp \
                           Samplings/MOATSampling.h \
                           Samplings/Sampling.h Util/sysdef.h \
                           Util/PsuadeUtil.h

Samplings/SobolSampling.o : Samplings/SobolSampling.cpp \
                            Samplings/SobolSampling.h \
                            Samplings/Sampling.h Util/sysdef.h \
                            Util/PsuadeUtil.h

Samplings/LPtauSampling.o : Samplings/LPtauSampling.cpp \
                            Samplings/LPtauSampling.h \
                            Samplings/Sampling.h Util/sysdef.h \
                            Util/PsuadeUtil.h

Samplings/MetisSampling.o : Samplings/MetisSampling.cpp \
                            Samplings/MetisSampling.h \
                            Samplings/Sampling.h Util/sysdef.h \
                            Util/PsuadeUtil.h

Samplings/FASTSampling.o : Samplings/FASTSampling.cpp \
                           Samplings/FASTSampling.h \
                           Samplings/Sampling.h Util/sysdef.h \
                           Util/PsuadeUtil.h

Samplings/SFASTSampling.o : Samplings/SFASTSampling.cpp \
                            Samplings/SFASTSampling.h \
                            Samplings/Sampling.h Util/sysdef.h \
                            Util/PsuadeUtil.h

Samplings/BoxBehnkenSampling.o : Samplings/BoxBehnkenSampling.cpp \
                                 Samplings/BoxBehnkenSampling.h \
                                 Samplings/Sampling.h Util/sysdef.h \
                                 Util/PsuadeUtil.h

Samplings/PlackettBurmanSampling.o : Samplings/PlackettBurmanSampling.cpp \
                                     Samplings/PlackettBurmanSampling.h \
                                     Samplings/Sampling.h Util/sysdef.h \
                                     Util/PsuadeUtil.h

Samplings/FractFactSampling.o : Samplings/FractFactSampling.cpp \
                                Util/sysdef.h Samplings/Sampling.h \
                                Util/PsuadeUtil.h \
                                Samplings/FractFactSampling.h 

Samplings/CentralCompositeSampling.o : Samplings/CentralCompositeSampling.cpp \
                                       Util/sysdef.h Samplings/Sampling.h \
                                       Util/PsuadeUtil.h \
                                       Samplings/CentralCompositeSampling.h 

Samplings/UserMetisSampling.o : Samplings/UserMetisSampling.cpp \
                                Samplings/UserMetisSampling.h \
                                Samplings/Sampling.h Util/sysdef.h \
                                Util/PsuadeUtil.h

Samplings/GMetisSampling.o : Samplings/GMetisSampling.cpp \
                             Samplings/GMetisSampling.h \
                             Samplings/Sampling.h Util/sysdef.h \
                             Util/PsuadeUtil.h

Samplings/GMOATSampling.o : Samplings/GMOATSampling.cpp \
                            Samplings/GMOATSampling.h \
                            Samplings/Sampling.h Util/sysdef.h \
                            Util/PsuadeUtil.h

Samplings/SparseGridSampling.o : Samplings/SparseGridSampling.cpp \
                                 Samplings/SparseGridSampling.h \
                                 Samplings/Sampling.h Util/sysdef.h \
                                 Util/PsuadeUtil.h

Samplings/DiscreteSampling.o : Samplings/DiscreteSampling.cpp \
                               Samplings/DiscreteSampling.h \
                               Samplings/Sampling.h Util/sysdef.h \
                               Util/PsuadeUtil.h

Samplings/LSASampling.o : Samplings/LSASampling.cpp \
                          Samplings/LSASampling.h \
                          Samplings/Sampling.h Util/sysdef.h \
                          Util/PsuadeUtil.h

Samplings/RSConstraints.o : Samplings/RSConstraints.cpp Util/PsuadeUtil.h \
                            Samplings/RSConstraints.h \
                            FuncApprox/FuncApproxFilter.h \
                            DataIO/PsuadeData.h

Samplings/MOATConstraints.o : Samplings/MOATConstraints.cpp Util/PsuadeUtil.h \
                              Samplings/MOATConstraints.h \
                              FuncApprox/FuncApproxFilter.h \
                              DataIO/PsuadeData.h

Samplings/GridNode.o : Samplings/GridNode.cpp Samplings/GridNode.h

PDFLib/PDFBase.o : PDFLib/PDFBase.cpp Util/PsuadeUtil.h PDFLib/PDFBase.h

PDFLib/PDFNormal.o : PDFLib/PDFNormal.cpp Util/PsuadeUtil.h \
                     PDFLib/PDFBase.h PDFLib/PDFNormal.h

PDFLib/PDFLogNormal.o : PDFLib/PDFLogNormal.cpp Util/PsuadeUtil.h \
                        PDFLib/PDFBase.h PDFLib/PDFLogNormal.h

PDFLib/PDFTriangle.o : PDFLib/PDFTriangle.cpp Util/PsuadeUtil.h \
                       PDFLib/PDFBase.h PDFLib/PDFTriangle.h

PDFLib/PDFBeta.o : PDFLib/PDFBeta.cpp Util/PsuadeUtil.h \
                   PDFLib/PDFBase.h PDFLib/PDFBeta.h

PDFLib/PDFWeibull.o : PDFLib/PDFWeibull.cpp Util/PsuadeUtil.h \
                      PDFLib/PDFBase.h PDFLib/PDFWeibull.h

PDFLib/PDFGamma.o : PDFLib/PDFGamma.cpp Util/PsuadeUtil.h \
                    PDFLib/PDFBase.h PDFLib/PDFGamma.h

PDFLib/PDFExponential.o : PDFLib/PDFExponential.cpp \
                          PDFLib/PDFBase.h PDFLib/PDFExponential.h

PDFLib/PDFSpecial.o : PDFLib/PDFSpecial.cpp \
                      PDFLib/PDFBase.h PDFLib/PDFSpecial.h

PDFLib/PDFMVNormal.o : PDFLib/PDFMVNormal.cpp Util/PsuadeUtil.h \
                       PDFLib/PDFMVNormal.h

PDFLib/PDFMVLogNormal.o : PDFLib/PDFMVLogNormal.cpp Util/PsuadeUtil.h \
                          PDFLib/PDFMVLogNormal.h

PDFLib/PDFManager.o : PDFLib/PDFManager.cpp Util/PsuadeUtil.h PDFLib/pdfData.h

PDFLib/pdfData.o : PDFLib/pdfData.cpp PDFLib/pdfData.h

Samplings/Bose/construct.o : Samplings/Bose/construct.c 
Samplings/Bose/akconst.o   : Samplings/Bose/akconst.c 
Samplings/Bose/primes.o    : Samplings/Bose/primes.c 
Samplings/Bose/gfields.o   : Samplings/Bose/gfields.c 
Samplings/Bose/galois.o    : Samplings/Bose/galois.c 
Samplings/Bose/memory.o    : Samplings/Bose/memory.c 
Samplings/Bose/boselink.o  : Samplings/Bose/boselink.c 
Samplings/Bose/oa.o        : Samplings/Bose/oa.c 

Util/PsuadeUtil.o : Util/PsuadeUtil.cpp Util/PsuadeUtil.h \
                    Util/isaac_standard.h Util/isaac_rand.h

Util/PsuadeParser.o : Util/PsuadeParser.cpp

Util/Matrix.o : Util/Matrix.cpp Util/Matrix.h

Util/Vector.o : Util/Vector.cpp Util/Vector.h

Util/iVector.o : Util/iVector.cpp Util/iVector.h

Util/isaac_randport.o : Util/isaac_randport.c Util/isaac_rand.h \
                        Util/isaac_standard.h

Util/gamma_function.o : Util/gamma_function.c 

Util/incomplete_gamma_function.o : Util/incomplete_gamma_function.c 

Util/entire_incomplete_gamma_function.o : Util/entire_incomplete_gamma_function.c 

Util/ln_gamma_function.o : Util/ln_gamma_function.c 

Util/beta_function.o : Util/beta_function.c 

Util/incomplete_beta_function.o : Util/incomplete_beta_function.c 

Util/factorial.o : Util/factorial.c 

Util/f2cStuff.o : Util/f2cStuff.c Util/f2c.h

Util/cygwin_system.o : Util/cygwin_system.c

DataIO/PsuadeData.o : DataIO/PsuadeData.cpp DataIO/PsuadeData.h \
                      Util/sysdef.h Util/dtype.h

DataIO/PsuadeConfig.o : DataIO/PsuadeConfig.cpp DataIO/PsuadeConfig.h

DataIO/pData.o : DataIO/pData.cpp DataIO/pData.h

DataIO/FunctionInterface.o : DataIO/FunctionInterface.cpp \
                             DataIO/FunctionInterface.h \
                             Util/sysdef.h Util/dtype.h

Plot/plotbogus.o : Plot/plotbogus.cpp

Plot/pgplotlink.o : Plot/pgplotlink.f

Analysis/AnalysisManager.o : Analysis/AnalysisManager.cpp \
                             Analysis/Analyzer.h \
                             Analysis/aData.h Util/PsuadeUtil.h

Analysis/Analyzer.o : Analysis/Analyzer.cpp Analysis/Analyzer.h \
                      Analysis/aData.h Util/PsuadeUtil.h

Analysis/aData.o : Analysis/aData.cpp Analysis/aData.h \
                   Analysis/aData.h Util/PsuadeUtil.h

Analysis/MomentAnalyzer.o : Analysis/MomentAnalyzer.cpp \
                            Analysis/MomentAnalyzer.h \
                            Analysis/aData.h Util/PsuadeUtil.h

Analysis/MainEffectAnalyzer.o : Analysis/MainEffectAnalyzer.cpp \
                                Analysis/MainEffectAnalyzer.h \
                                Analysis/aData.h Util/PsuadeUtil.h

Analysis/TwoParamAnalyzer.o : Analysis/TwoParamAnalyzer.cpp \
                              Analysis/TwoParamAnalyzer.h \
                              FuncApprox/FuncApprox.h FuncApprox/Mars.h \
                              Analysis/aData.h Util/PsuadeUtil.h

Analysis/AnovaAnalyzer.o : Analysis/AnovaAnalyzer.cpp \
                           Analysis/AnovaAnalyzer.h FuncApprox/FuncApprox.h \
                           FuncApprox/Mars.h Analysis/aData.h \
                           Util/PsuadeUtil.h

Analysis/GradStatAnalyzer.o : Analysis/GradStatAnalyzer.cpp \
                              Analysis/GradStatAnalyzer.h Util/PsuadeUtil.h \
                              Analysis/aData.h

Analysis/CorrelationAnalyzer.o : Analysis/CorrelationAnalyzer.cpp \
                                 Analysis/CorrelationAnalyzer.h \
                                 Analysis/aData.h

Analysis/RSFuncApproxAnalyzer.o : Analysis/RSFuncApproxAnalyzer.cpp \
                                  Analysis/RSFuncApproxAnalyzer.h \
                                  Analysis/aData.h

Analysis/MOATAnalyzer.o : Analysis/MOATAnalyzer.cpp \
                          Analysis/MOATAnalyzer.h Analysis/aData.h

Analysis/SobolAnalyzer.o : Analysis/SobolAnalyzer.cpp \
                           Analysis/SobolAnalyzer.h Analysis/aData.h

Analysis/IntegrationAnalyzer.o : Analysis/IntegrationAnalyzer.cpp \
                                 Analysis/IntegrationAnalyzer.h \
                                 Analysis/aData.h

Analysis/FASTAnalyzer.o : Analysis/FASTAnalyzer.cpp \
                          Analysis/FASTAnalyzer.h Analysis/aData.h

Analysis/FFAnalyzer.o : Analysis/FFAnalyzer.cpp \
                        Analysis/FFAnalyzer.h Analysis/aData.h

Analysis/LSAnalyzer.o : Analysis/LSAnalyzer.cpp \
                        Analysis/LSAnalyzer.h Analysis/aData.h

Analysis/PCAnalyzer.o : Analysis/PCAnalyzer.cpp \
                        Analysis/PCAnalyzer.h Analysis/aData.h

Analysis/OneSigmaAnalyzer.o : Analysis/OneSigmaAnalyzer.cpp \
                              Analysis/OneSigmaAnalyzer.h \
                              Util/PsuadeUtil.h Analysis/aData.h

Analysis/FORMAnalyzer.o : Analysis/FORMAnalyzer.cpp \
                          Analysis/FORMAnalyzer.h \
                          Util/PsuadeUtil.h Analysis/aData.h

Analysis/RSMSobol1Analyzer.o : Analysis/RSMSobol1Analyzer.cpp \
                               Analysis/RSMSobol1Analyzer.h Analysis/aData.h

Analysis/RSMSobol2Analyzer.o : Analysis/RSMSobol2Analyzer.cpp \
                               Analysis/RSMSobol2Analyzer.h Analysis/aData.h

Analysis/RSMSobolTSIAnalyzer.o : Analysis/RSMSobolTSIAnalyzer.cpp \
                                 Analysis/RSMSobolTSIAnalyzer.h \
                                 Analysis/aData.h

Analysis/TSIAnalyzer.o : Analysis/TSIAnalyzer.cpp Analysis/TSIAnalyzer.h \
                         Analysis/aData.h

Analysis/BootstrapAnalyzer.o : Analysis/BootstrapAnalyzer.cpp \
                               Analysis/BootstrapAnalyzer.h Analysis/aData.h

Analysis/RSMSobolGAnalyzer.o : Analysis/RSMSobolGAnalyzer.cpp \
                               Analysis/RSMSobolGAnalyzer.h Analysis/aData.h

Analysis/BinomialAnalyzer.o : Analysis/BinomialAnalyzer.cpp \
                              Analysis/BinomialAnalyzer.h Analysis/aData.h

Analysis/OneSampleAnalyzer.o : Analysis/OneSampleAnalyzer.cpp \
                               Analysis/OneSampleAnalyzer.h Analysis/aData.h

Analysis/TwoSampleAnalyzer.o : Analysis/TwoSampleAnalyzer.cpp \
                               Analysis/TwoSampleAnalyzer.h Analysis/aData.h

Analysis/MCMCAnalyzer.o : Analysis/MCMCAnalyzer.cpp \
                          Analysis/MCMCAnalyzer.h Analysis/aData.h

Analysis/DeltaAnalyzer.o : Analysis/DeltaAnalyzer.cpp \
                           Analysis/DeltaAnalyzer.h Analysis/aData.h

Analysis/GowerAnalyzer.o : Analysis/GowerAnalyzer.cpp \
                           Analysis/GowerAnalyzer.h Analysis/aData.h

Analysis/EtaAnalyzer.o : Analysis/EtaAnalyzer.cpp \
                         Analysis/EtaAnalyzer.h Analysis/aData.h

FuncApprox/FuncApprox.o : FuncApprox/FuncApprox.cpp FuncApprox/FuncApprox.h

FuncApprox/MarsInterface.o : FuncApprox/MarsInterface.c

FuncApprox/Mars.o : FuncApprox/Mars.cpp FuncApprox/Mars.h \
                    FuncApprox/FuncApprox.h

FuncApprox/Earth.o : FuncApprox/Earth.cpp FuncApprox/Earth.h \
                     FuncApprox/FuncApprox.h

FuncApprox/SumOfTrees.o : FuncApprox/SumOfTrees.cpp FuncApprox/SumOfTrees.h \
                          FuncApprox/FuncApprox.h

FuncApprox/TreeNode.o : FuncApprox/TreeNode.cpp FuncApprox/TreeNode.h

FuncApprox/MarsBagg.o : FuncApprox/MarsBagg.cpp FuncApprox/MarsBagg.h \
                        FuncApprox/Mars.h FuncApprox/FuncApprox.h

FuncApprox/Regression.o : FuncApprox/Regression.cpp FuncApprox/Regression.h \
                          FuncApprox/FuncApprox.h

FuncApprox/SelectiveRegression.o : FuncApprox/SelectiveRegression.cpp \
                                   FuncApprox/SelectiveRegression.h \
                                   FuncApprox/FuncApprox.h

FuncApprox/UserRegression.o : FuncApprox/UserRegression.cpp \
                              FuncApprox/UserRegression.h \
                              FuncApprox/FuncApprox.h

FuncApprox/LegendreRegression.o : FuncApprox/LegendreRegression.cpp \
                                  FuncApprox/LegendreRegression.h \
                                  FuncApprox/FuncApprox.h

FuncApprox/GradLegendreRegression.o : FuncApprox/GradLegendreRegression.cpp \
                                      FuncApprox/GradLegendreRegression.h \
                                      FuncApprox/FuncApprox.h

FuncApprox/SGRegression.o : FuncApprox/SGRegression.cpp \
                            FuncApprox/SGRegression.h \
                            FuncApprox/FuncApprox.h

FuncApprox/FuncApproxFilter.o : FuncApprox/FuncApproxFilter.cpp \
                                FuncApprox/FuncApproxFilter.h

FuncApprox/Ann.o : FuncApprox/Ann.cpp FuncApprox/Ann.h \
                   FuncApprox/FuncApprox.h FuncApprox/AnnFunctions.h

FuncApprox/AnnFunctions.o : FuncApprox/AnnFunctions.c \
                            FuncApprox/AnnFunctions.h \
                            FuncApprox/AnnFunctions.ph

FuncApprox/GP1.o : FuncApprox/GP1.cpp FuncApprox/GP1.h FuncApprox/FuncApprox.h

FuncApprox/GP2.o : FuncApprox/GP2.cpp FuncApprox/GP2.h FuncApprox/FuncApprox.h

FuncApprox/TBGP.o : FuncApprox/TBGP.cpp FuncApprox/TBGP.h \
                    FuncApprox/FuncApprox.h

FuncApprox/SVM.o : FuncApprox/SVM.cpp FuncApprox/SVM.h FuncApprox/FuncApprox.h

FuncApprox/PWLinear.o : FuncApprox/PWLinear.cpp FuncApprox/PWLinear.h \
                        FuncApprox/FuncApprox.h

FuncApprox/Kriging.o : FuncApprox/Kriging.cpp FuncApprox/Kriging.h FuncApprox/FuncApprox.h

FuncApprox/NPLearning.o : FuncApprox/NPLearning.cpp FuncApprox/NPLearning.h FuncApprox/FuncApprox.h

Comm/BaseComm.o : Comm/BaseComm.cpp Comm/BaseComm.h

Comm/CommManager.o : Comm/CommManager.cpp Comm/CommManager.h \
                     Comm/CommMPICH.cpp Comm/CommMPICH.h

Comm/CommMPICH.o : Comm/CommMPICH.cpp Comm/CommMPICH.h \
                   Comm/BaseComm.cpp Comm/BaseComm.h

Comm/JobCntl.o : Comm/JobCntl.cpp Comm/JobCntl.h

Optimizers/TxMathOptimizer.o : Optimizers/TxMathOptimizer.cpp \
                               Optimizers/TxMathOptimizer.h

Optimizers/APPSPACKOptimizer.o : Optimizers/APPSPACKOptimizer.cpp \
                                 Optimizers/APPSPACKOptimizer.h

Optimizers/MinpackOptimizer.o : Optimizers/MinpackOptimizer.cpp \
                                Optimizers/MinpackOptimizer.h

Optimizers/CobylaOptimizer.o : Optimizers/CobylaOptimizer.cpp \
                               Optimizers/CobylaOptimizer.h

Optimizers/SMOptimizer.o : Optimizers/SMOptimizer.cpp \
                           Optimizers/SMOptimizer.h

Optimizers/MMOptimizer.o : Optimizers/MMOptimizer.cpp \
                           Optimizers/MMOptimizer.h

Optimizers/BobyqaOptimizer.o : Optimizers/BobyqaOptimizer.cpp \
                               Optimizers/BobyqaOptimizer.h

Optimizers/SCEOptimizer.o : Optimizers/SCEOptimizer.cpp \
                            Optimizers/SCEOptimizer.h

Optimizers/MultiObjectiveOptimizer.o : Optimizers/MultiObjectiveOptimizer.cpp \
                            Optimizers/MultiObjectiveOptimizer.h

Optimizers/Optimizer.o : Optimizers/Optimizer.cpp Optimizers/Optimizer.h

Optimizers/TxMathPSUADE.o : Optimizers/TxMathPSUADE.cpp

Optimizers/oData.o : Optimizers/oData.cpp Optimizers/oData.h

PythonWrapper/psuade_wrap.o : PythonWrapper/psuade_wrap.cxx
	@echo "Making $@"
	$(CXX) -w -c $(CXXFLAGS) $(CXXFLAGS2) $(DEFINES) $(INCPYTH) -o $@ $<


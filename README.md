# PSUADE

PSUADE is an acronym for Problem Solving environment for Uncertainty Analysis 
and Design Exploration. It is a software toolkit to facilitate the UQ tasks 
described below. PSUADE has a rich set of tools for performing uncertainty 
analysis, global sensitivity analysis, design optimization, model calibration, 
and more. In particular, PSUADE supports a global sensitivity methodology for 
models with a large number of parameters and complex parameter correlations.

# History

It enjoyed its first public release in 2007 and it has been over 15 years.
In 2016, as part of the Carbon Capture Simulation Initiative (CCSI program)
software suite (it is the computational workhorse of the CCSI FOQUS
software), it has earned the RD100 award. PSUADE has been downloaded and
used across several continents for many diverse scientific applications.

## Quick Installation Guide

For problems with installation, contact:

Charles Tong at tong10@llnl.gov, or James Leek at leek2@llnl.gov

### Linux

A. To install the PSUADE package, you need to make sure you have cmake on your
   system. (cmake is open source software availible at www.cmake.org) cmake 
   should also have installed an easier interface program called ccmake, but 
   that isn't required.  If you do not have ccmake, you will need to turn on 
   some of the packages manually.

   You will also need gcc, g++, and gfortran of version 4.4 or greater.
   icc 10 or higher will also work.

   Follow the steps below:

   1. mkdir build
   2. cd build
   3. if you have a preferred Fortran compiler, set the environment
      variable FC to it (use setenv in c-shell or export in other shells)
   4. ccmake ..
      hit 'c'

      BUILD_SHARED, MARS, BOBYQA, and METIS should have been selected
      (for additional packages, contact us).

      If you would like to install psuade at a designated location accessible 
      to other users, set the installation directory.

      hit 'c'
      hit 'c' again until you are able to hit 'g'.
      hit 'g' to generate an exit

      If you do not have ccmake, :
      cmake ..
      and then open the CMakeCache.txt file and make sure the packages
      MARS (set to on from off), BOBYQA, and METIS are turned on.

   5. now do a "make" or "make install" if you desire to install it somewhere

B. After all compilation is done successfully, the executable "psuade"
   can be found in the bin directory and the libraries will be in
   the lib directory.

C. You can run a simple test by going to the
   <toplevel>/Examples/SimpleUQ/Bungee/Basic directory and issuing:
       cc -o simulator simulator.c -lm
       ../../../../build/bin/psuade psuade.in
   Afterward, you should see a file called 'psuadeData'.

D. You can also run the built in tests my running 'make test' from the
   build directory.  WARNINGS:  
   1. This will take a long time.  At least 20 minutes
   2. Some tests are expected to fail if you aren't running on LLNL LC
      cluster.  PSUADE is very sensitive to the processor it's running on
      and the numeric results will be off on different processors and
      environment variables.  As long as a few tests pass you're probably OK.

E. You can install PSUADE by running 'make install'

F. You can build a package for other people to install by running 'make package'

G. Now read the short manual in the Doc/Manual directory and follow the
   instructions to get a simple application running within minutes.

-----------------------------------------------------------------------------

### Windows (Installation has not been tested for years)

   Requires Cmake and mingw (preferably including gfortran).  If you want to 
   build an installable package you will need NSIS.

A. Make sure you have cmake >= 2.8 installed on your system.  Then start the
   cmake-gui.  
   Select you PSUADE source tree, and where you want it to build.
   Click configure.
   Select MingGW make files.
   Select BUILD_SHARED, BOBYQA, and METIS.
   Click Generate.

B. Open a command line window, either powershell of cmd.
   cd builddir
   c:\mingw\bin\mingw-make.exe
   It should build for a while.

C. You can also run the built in tests my running:
   c:\mingw\bin\mingw-make.exe test
   WARNING:
   1. This will take a long time.  At least 20 minutes
   2. Some tests are expected to fail on Windows. PSUADE is very sensitive
      to the processor it's running on and the numeric results will be off
      on different processors and enviromnet variables.  As long as a few
      tests pass you're probably OK.

E. You can install PSUADE by running c:\mingw\bin\mingw-make.exe install

F. You can build a package for other people to install by running
   c:\mingw\bin\mingw-make.exe package

G. Now read the short manual in the Doc/Manual directory and follow the
   instructions to get a simple application running within minutes.

-----------------------------------------------------------------------------

### MacOSX

   Requires cmake, and cc/gcc, c++/g++, gfortran >= 4.4.

A. Check to make sure you have cc, c++, gfortran, ccmake

B. Now let's run cmake.  Go to your psuade source.
   mkdir build
   cd build
   ccmake ..
   hit 'c' to get started
   Make sure BUILD_SHARED, MARS, BOBYQA, and METIS have already been selected.
   hit 'c'
   hit 't' to go to advanced options.  
   I can see in my case cmake has picked up the wrong cc:
   CMAKE_C_COMPILER                 /usr/bin/cc          
   I need to change it to:
   CMAKE_C_COMPILER                 /opt/local/bin//gcc
   If ccmake fails and the option 'g' is not available, remove the specifics
   in OSX_DEPLOYMENT_TARGET to see if this fixes the problem.

   Aside from that it seems OK, so I hit 'c' until I can hit 'g'
   hit 'g' to generate and exit.
   Note: If you are using compilers from XCode and compilation errors occur,
         try setting the compilers to the ones in /usr/bin.

C. Now build: run 'make'  It should build for a while.

D. You can run a simple test by going to the
   Examples/SimpleUQ/Bungee/Basic directory and issuing:
       cc -o simulator simulator.c -lm
       ../../../../build/bin/psuade psuade.in
   Afterward, you should see a file called 'psuadeData'.

E. You can install PSUADE by running 'make install'

F. You can build a package for other people to install by running 'make package'

g. Now read the short manual in the Doc/Manual directory and follow the
   instructions to get a simple application running within minutes.

*****************************************************************************

## Explanation of directories

Examples : test programs
Src      : source code
Doc      : documentations
Python   : Python script to run visualization
External : external packages

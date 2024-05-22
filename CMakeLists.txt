cmake_minimum_required (VERSION 2.8)
project (psuade_project)
include(ExternalProject)

# enable testing
enable_testing ()

set (PROJECT_NAME psuade)
# The version number.
set (psuade_VERSION_MAJOR 3)
set (psuade_VERSION_MINOR 0)
set (psuade_VERSION_PATCH 0)

set (PROJECT_INCLUDE_DIR "${PROJECT_BINARY_DIR}/include")
set (PROJECT_BIN_DIR "${PROJECT_BINARY_DIR}/bin}")
set (PROJECT_LIB_DIR "${PROJECT_BINARY_DIR}/lib}")

set (EXTRA_LD_FLAGS "")


# configure a header file to pass some of the CMake settings
# to the source code
configure_file (
  "${PROJECT_SOURCE_DIR}/Src/Main/PsuadeCmakeConfig.h.in"
  "${PROJECT_INCLUDE_DIR}/PsuadeCmakeConfig.h"
  )

#########################################################
# Setup SYSTEM type, useful for some #defines in PSUADE
#########################################################

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")
add_definitions(-DLINUX)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
add_definitions(-DWINDOWS)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "MacOSX")
add_definitions(-DMACOS)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
add_definitions(-DMACOS)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "cygwin")
add_definitions(-DCYGWIN)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "IRIX")
add_definitions(-DIRIX)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "HP-UX")
add_definitions(-DHPUX)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "AIX")
add_definitions(-DAIX)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "OFS/1")
add_definitions(-DALPHA)

elseif("${CMAKE_SYSTEM_NAME}" STREQUAL "SunOS")
add_definitions(-DSUNOS)

else("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")

add_definitions(-DUSER)
endif("${CMAKE_SYSTEM_NAME}" STREQUAL "Linux")



#########################################################
# Setup the RPATH Junk
#########################################################
# use, i.e. don't skip the full RPATH for the build tree
if(BUILD_SHARED)

SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
SET(CMAKE_SKIP_RPATH FALSE)
# when building, don't use the install RPATH already
# (but later on when installing)
SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 

SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")

# add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
# the RPATH to be used when installing, but only if it's not a system directory
LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
IF("${isSystemDir}" STREQUAL "-1")
   SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
ENDIF("${isSystemDir}" STREQUAL "-1")
IF("${CMAKE_SYSTEM_NAME}" STREQUAL "MacOSX")
   SET(CMAKE_MACOSX_RPATH 0)
ENDIF("${CMAKE_SYSTEM_NAME}" STREQUAL "MacOSX")
IF("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
   SET(CMAKE_MACOSX_RPATH 0)
ENDIF("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")

endif(BUILD_SHARED)
#########################################################
#END RPATH SETUP
#########################################################

#########################################################
# Special build types and extra flags
#########################################################
#SET(CMAKE_BUILD_TYPE nosse)
SET(CMAKE_CXX_FLAGS_NOSSE "-O0 -g -mno-sse")
SET(CMAKE_C_FLAGS_NOSSE "-O0 -g -mno-sse")
SET(CMAKE_Fortran_FLAGS_NOSSE "-O0 -g -mno-sse")

#SET(CMAKE_BUILD_TYPE debug)
SET(CMAKE_CXX_FLAGS_DEBUG "-O0 -g -mfpmath=sse -msse2")
SET(CMAKE_C_FLAGS_DEBUG "-O0 -g -mfpmath=sse -msse2")
SET(CMAKE_Fortran_FLAGS_DEBUG "-O0 -g -mfpmath=sse -msse2")

SET(CMAKE_CXX_FLAGS_OMPDEBUG " -O0 -g -fopenmp -mfpmath=sse -msse2")
SET(CMAKE_C_FLAGS_OMPDEBUG " -O0 -g -fopenmp -mfpmath=sse -msse2")
SET(CMAKE_Fortran_FLAGS_OMPDEBUG " -O0 -g -fopenmp -Wreturn-type")
SET(CMAKE_LD_FLAGS_OMPDEBUG "-fopenmp -mfpmath=sse -msse2")

SET(CMAKE_CXX_FLAGS_OMP " -O2 -fopenmp -mfpmath=sse -msse2")
SET(CMAKE_C_FLAGS_OMP " -O2 -fopenmp -mfpmath=sse -msse2")
SET(CMAKE_Fortran_FLAGS_OMP " -O2 -fopenmp -mfpmath=sse -msse2")
SET(CMAKE_LD_FLAGS_OMP "-fopenmp")

 
SET(CMAKE_CXX_FLAGS_RELEASE "-O2")
SET(CMAKE_C_FLAGS_RELEASE "-O2")
SET(CMAKE_Fortran_FLAGS_RELEASE "-O2")


# add the binary tree to the search path for include files
# so that we will find PsuadeCmakeConfig.h
include_directories("${PROJECT_INCLUDE_DIR}")
include_directories("Src")
include_directories("Src/Base")
include_directories("Src/Main")
include_directories("Src/Samplings")
include_directories("Src/PDFLib")
include_directories("Src/Util")
include_directories("Src/DataIO")
include_directories("Src/Analysis")
include_directories("Src/FuncApprox")
include_directories("Src/Comm")
include_directories("Src/Optimizers")
include_directories("Src/DA")

set( PDF_SRC 
  "Src/PDFLib/PDFBase.cpp"
  "Src/PDFLib/PDFNormal.cpp"
  "Src/PDFLib/PDFLogNormal.cpp"
  "Src/PDFLib/PDFTriangle.cpp"
  "Src/PDFLib/PDFBeta.cpp"
  "Src/PDFLib/PDFWeibull.cpp"
  "Src/PDFLib/PDFMVNormal.cpp"
  "Src/PDFLib/PDFMVLogNormal.cpp"
  "Src/PDFLib/PDFGamma.cpp"
  "Src/PDFLib/PDFInvGamma.cpp"
  "Src/PDFLib/PDFCauchy.cpp"
  "Src/PDFLib/PDFSample.cpp"
  "Src/PDFLib/PDFSampleHist.cpp"
  "Src/PDFLib/PDFUser.cpp"
  "Src/PDFLib/PDFHistogram.cpp"
  "Src/PDFLib/PDFExponential.cpp"
  "Src/PDFLib/PDFF.cpp"
  "Src/PDFLib/PDFUser.cpp"
  "Src/PDFLib/PDFManager.cpp"
  "Src/PDFLib/pdfData.cpp"
  "Src/PDFLib/KSDensity.cpp"
)

# add the executable
file(GLOB psuade_SRC
  "Src/Base/*.cpp"
  "Src/Samplings/*.cpp"
  "Src/Util/*.cpp"
  "Src/Util/*.c"
  "Src/DataIO/*.cpp"
  "Src/Analysis/*.cpp"
  "Src/FuncApprox/*.cpp"
  "Src/FuncApprox/*.c"
  "Src/Comm/*.cpp"
  "Src/Optimizers/*.cpp"
  "Src/DA/*.cpp"
  "Src/DA/genSnapshot.f"
)

file(GLOB psuade_HDRS
  "Src/Base/*.h"
  "Src/Samplings/*.h"
  "Src/Util/*.h"
  "Src/DataIO/*.h"
  "Src/Analysis/*.h"
  "Src/FuncApprox/*.h"
  "Src/Comm/*.h"
  "Src/Optimizers/*.h"
  "Src/DA/*.h"
)

###############################################
#SETUP THE MAIN PSUADE LIBRARY AND EXECUTABLES
###############################################
option (BUILD_SHARED "Build all the libraries as shared libs, and build libpsuade.so" ON)
if (BUILD_SHARED)
  set(LIBRARY_TYPE SHARED)
else (BUILD_SHARED)
  set (LIBRARY_TYPE STATIC)
  SET(CMAKE_EXE_LINKER_FLAGS -static)
endif(BUILD_SHARED)

#set(CMAKE_RUNTIME_OUTPUT_DIRECTORY )
set(CMAKE_ARCHIVE_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")
set(CMAKE_LIBRARY_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/lib")


add_library (psuade-lib ${LIBRARY_TYPE} ${psuade_SRC}  ${psuade_HDRS} ${PDF_SRC})
add_executable (psuade-bin Src/Main/Psuade.cpp)

set (PSUADE_NAME "psuade")

SET_TARGET_PROPERTIES(psuade-bin
  PROPERTIES 
  OUTPUT_NAME ${PSUADE_NAME}
  RUNTIME_OUTPUT_DIRECTORY "${CMAKE_BINARY_DIR}/bin")

if(BUILD_SHARED)

SET_TARGET_PROPERTIES(psuade-lib
  PROPERTIES
  OUTPUT_NAME ${PSUADE_NAME})

else(BUILD_SHARED)

SET_TARGET_PROPERTIES(psuade-lib
  PROPERTIES
  OUTPUT_NAME ${PSUADE_NAME}
  LINK_FLAGS "-static-libgcc -static-libstdc++ -static-libgfortran")

endif(BUILD_SHARED)


###############################################
# MPI?
###############################################
if(NOT MINGW)
option(PARALLEL_BUILD "Use MPI to build parallel PSUADE")
endif(NOT MINGW)

if (PARALLEL_BUILD)
add_definitions(-DHAVE_PARALLEL -DHAVE_MPICH)
include(FindMPI)
if(NOT MPI_FOUND)
  message(ERROR "MPI not found")
endif()

set (EXTRA_LIBS ${EXTRA_LIBS} "${MPI_LIBRARIES}")
set (LINK_FLAGS ${LINK_FLAGS} "${MPI_LINK_FLAGS}")
include_directories(${MPI_INCLUDE_PATH})

endif (PARALLEL_BUILD)


###############################################
#SETUP 3RD PARTY LIBRARIES
###############################################

# add Bose
if(EXISTS "${PROJECT_SOURCE_DIR}/External/Bose")
   include_directories ("${PROJECT_SOURCE_DIR}/External/Bose")
   add_subdirectory (External/Bose) 
   set (EXTRA_LIBS ${EXTRA_LIBS} bose)
endif() 

#APPSPACK IGNORED, IT IS OBSOLETE

#BOBYQA
if(EXISTS "${PROJECT_SOURCE_DIR}/External/BOBYQA")
  option (USE_BOBYQA
        "Build and use the BOBYQA library." ON)
endif() 
if (USE_BOBYQA)
  enable_language(Fortran)
  include_directories ("${PROJECT_SOURCE_DIR}/External/BOBYQA")
  add_subdirectory (External/BOBYQA) 
  set (EXTRA_LIBS ${EXTRA_LIBS} bobyqa)
  add_definitions(-DHAVE_BOBYQA)
endif (USE_BOBYQA)

#COBYLA
if(EXISTS "${PROJECT_SOURCE_DIR}/External/COBYLA")
  option (USE_COBYLA
        "Build and use the COBYLA library." ON)
endif() 
if (USE_COBYLA)
  include_directories ("${PROJECT_SOURCE_DIR}/External/COBYLA")
  add_subdirectory (External/COBYLA) 
  set (EXTRA_LIBS ${EXTRA_LIBS} cobyla2c)
  add_definitions(-DHAVE_COBYLA)
endif (USE_COBYLA)

#EARTH
if(EXISTS "${PROJECT_SOURCE_DIR}/External/EARTH")
  option (USE_EARTH
        "Build and use the EARTH library." OFF)
endif() 
if (USE_EARTH)
  include_directories ("${PROJECT_SOURCE_DIR}/External/EARTH")
  add_subdirectory (External/EARTH) 
  set (EXTRA_LIBS  earth ${EXTRA_LIBS})
  add_definitions(-DHAVE_EARTH)
endif (USE_EARTH)

#GPMC doesn't work on Windows
if(EXISTS "${PROJECT_SOURCE_DIR}/External/GPMC" AND NOT MINGW)
  option (USE_GPMC
        "Build and use the GPMC library." OFF)
endif() 
if (USE_GPMC)
  include_directories ("${PROJECT_SOURCE_DIR}/External/GPMC")
  add_subdirectory (External/GPMC) 
  set (EXTRA_LIBS ${EXTRA_LIBS} gpmc)
  add_definitions(-DHAVE_GPMC)
endif (USE_GPMC)

#LINCOA
if(EXISTS "${PROJECT_SOURCE_DIR}/External/LINCOA")
  option (USE_LINCOA
        "Build and use the LINCOA library." ON)
endif() 
if (USE_LINCOA)
  enable_language(Fortran)
  include_directories ("${PROJECT_SOURCE_DIR}/External/LINCOA")
  add_subdirectory (External/LINCOA) 
  set (EXTRA_LIBS ${EXTRA_LIBS} lincoa)
  add_definitions(-DHAVE_LINCOA)
endif (USE_LINCOA)

#NEWUOA
if(EXISTS "${PROJECT_SOURCE_DIR}/External/NEWUOA")
  option (USE_NEWUOA
        "Build and use the NEWUOA library." ON)
endif() 
if (USE_NEWUOA)
  enable_language(Fortran)
  include_directories ("${PROJECT_SOURCE_DIR}/External/NEWUOA")
  add_subdirectory (External/NEWUOA) 
  set (EXTRA_LIBS ${EXTRA_LIBS} newuoa)
  add_definitions(-DHAVE_NEWUOA)
endif (USE_NEWUOA)

#LBFGS
if(EXISTS "${PROJECT_SOURCE_DIR}/External/L-BFGS-B-C")
  option (USE_LBFGS
        "Build and use the LBFGS library." ON)
endif() 
if (USE_LBFGS)
  include_directories ("${PROJECT_SOURCE_DIR}/External/L-BFGS-B-C")
  add_subdirectory (External/L-BFGS-B-C) 
  set (EXTRA_LIBS ${EXTRA_LIBS} lbfgs)
  add_definitions(-DHAVE_LBFGS)
endif (USE_LBFGS)

#NOMAD
if(EXISTS "${PROJECT_SOURCE_DIR}/External/NOMAD")
  option (USE_NOMAD
        "Build and use the NOMAD library." ON)
endif() 
if (USE_NOMAD)
  include_directories ("${PROJECT_SOURCE_DIR}/External/NOMAD")
  add_subdirectory (External/NOMAD) 
  set (EXTRA_LIBS ${EXTRA_LIBS} nomad)
  add_definitions(-DHAVE_NOMAD)
endif (USE_NOMAD)

#optional libraries, see if dir exists, then see if user wants it.  If both yes, build it.
if(EXISTS "${PROJECT_SOURCE_DIR}/External/MARS")
  option (USE_MARS
        "Build and use the MARS library." ON)
else ()
  add_definitions(-DPSUADE_NO_MARS)
endif() 
enable_language(Fortran)
if (USE_MARS)
  include_directories ("${PROJECT_SOURCE_DIR}/External/MARS")
  add_subdirectory (External/MARS) 
  set (EXTRA_LIBS ${EXTRA_LIBS} mars)
  add_definitions(-DHAVE_MARS)

else (USE_MARS)
  add_definitions(-DPSUADE_NO_MARS)
endif (USE_MARS)

#METIS
if(EXISTS "${PROJECT_SOURCE_DIR}/External/METIS")
  option (USE_METIS
        "Build and use the METIS library." ON)
endif() 
if (USE_METIS)
  include_directories ("${PROJECT_SOURCE_DIR}/External/METIS/Lib")
  add_subdirectory (External/METIS) 
  set (EXTRA_LIBS ${EXTRA_LIBS} metis)
  add_definitions(-DHAVE_METIS)
endif (USE_METIS)

#MINPACK
if(EXISTS "${PROJECT_SOURCE_DIR}/External/MINPACK")
  option (USE_MINPACK
        "Build and use the MINPACK library." OFF)
endif() 
if (USE_MINPACK)
  enable_language(Fortran)
  include_directories ("${PROJECT_SOURCE_DIR}/External/MINPACK")
  add_subdirectory (External/MINPACK) 
  set (EXTRA_LIBS ${EXTRA_LIBS} minpack)
  add_definitions(-DHAVE_MINPACK)
endif (USE_MINPACK)

#SVM 
if(EXISTS "${PROJECT_SOURCE_DIR}/External/SVM")
  option (USE_SVM
        "Build and use the SVM library." OFF)
endif() 
if (USE_SVM)
  include_directories ("${PROJECT_SOURCE_DIR}/External/SVM")
  add_subdirectory (External/SVM) 
  set (EXTRA_LIBS ${EXTRA_LIBS} svmlight)
  add_definitions(-DHAVE_SVM)
endif (USE_SVM)

#TGP  (Apparently an R library)
if(EXISTS "${PROJECT_SOURCE_DIR}/External/TGP/")
  option (USE_TGP
        "Build and use the TGP library." OFF)
endif() 
if (USE_TGP)
  set(USE_R ON)
  include_directories ("${PROJECT_SOURCE_DIR}/External/TGP/src")
  include_directories ("${PROJECT_SOURCE_DIR}/External/R-2.4.0/nmath")
  add_subdirectory (External/TGP/src) 
  set (EXTRA_LIBS tgp ${EXTRA_LIBS})
  add_definitions(-DHAVE_TGP)
else(USE_TGP)
  set(USE_R OFF)
endif (USE_TGP)

#TPROS
if(EXISTS "${PROJECT_SOURCE_DIR}/External/TPROS")
  option (USE_TPROS
        "Build and use the TPROS library." OFF)
endif() 
if (USE_TPROS)
  include_directories ("${PROJECT_SOURCE_DIR}/External/TPROS")
  add_subdirectory (External/TPROS) 
  set (EXTRA_LIBS tpros ${EXTRA_LIBS})
  add_definitions(-DHAVE_TPROS)
endif (USE_TPROS)


#R-2.4.0
#if(EXISTS "${PROJECT_SOURCE_DIR}/External/R-2.4.0/nmath")
#  option (USE_NMATH
#        "Build and use the R-2.4.0 NMATH library." OFF)
#endif() 
if (USE_R)
  include_directories ("${PROJECT_SOURCE_DIR}/External/R-2.4.0/nmath")
  add_subdirectory (External/R-2.4.0/nmath) 
  set (EXTRA_LIBS ${EXTRA_LIBS} nmath)
  add_definitions(-DHAVE_R)
endif (USE_R)



###################################################
# Lapack and BLAS are weird special cases, because we may find
# them or build it ourselves, so it gets this special section
###################################################
#Lapack and BLAS are required, but we should only build them
#if the aren't already installed  (Or if we're doing a release. Can't expect users to have them.)
#(Ordering is a little silly to get the link order right)


find_package(LAPACK)
if(LAPACK_FOUND AND NOT CMAKE_BUILD_TYPE MATCHES Release AND NOT MINGW)
  set (EXTRA_LIBS ${EXTRA_LIBS} ${LAPACK_LIBRARIES}) 
else()
  message(STATUS "LaPack included with Psuade will be built instead. Nothing more is required.")
  enable_language(Fortran)
  include_directories ("${PROJECT_SOURCE_DIR}/External/Lapack")
  add_subdirectory (External/Lapack) 
  set (EXTRA_LIBS ${EXTRA_LIBS} lapack)
endif()

find_package(BLAS)
if(BLAS_FOUND AND NOT CMAKE_BUILD_TYPE MATCHES Release AND NOT MINGW) #Build blas for windows and release
  set (EXTRA_LIBS ${EXTRA_LIBS} "${BLAS_LIBRARIES}") 
else()
  message(STATUS "BLAS included with Psuade will be built instead. Nothing more is required.")
  enable_language(Fortran)
  include_directories ("${PROJECT_SOURCE_DIR}/External/BLAS")
  add_subdirectory (External/BLAS) 
  set (EXTRA_LIBS ${EXTRA_LIBS} blas)
endif()

###################################################
# Link psuade library and the binary to the library
###################################################

target_link_libraries (psuade-lib ${EXTRA_LIBS})
#if("${CMAKE_SYSTEM_NAME}" STREQUAL "MacOSX")
#  target_link_libraries (psuade-bin psuade-lib)  #I think OSX doesn't have "whole-archive?"  Linux and Windows need it
#else("${CMAKE_SYSTEM_NAME}" STREQUAL "MacOSX")
#  target_link_libraries (psuade-bin -Wl,-whole-archive psuade-lib  -Wl,-no-whole-archive)
#endif("${CMAKE_SYSTEM_NAME}" STREQUAL "MacOSX")
if("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  target_link_libraries (psuade-bin psuade-lib)  #I think OSX doesn't have "whole-archive?"  Linux and Windows need it
else("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")
  target_link_libraries (psuade-bin -Wl,-whole-archive psuade-lib  -Wl,-no-whole-archive)
endif("${CMAKE_SYSTEM_NAME}" STREQUAL "Darwin")


###################################################
# Main psuade files installation section
###################################################
# add the install targets
install (TARGETS psuade-bin psuade-lib 
	RUNTIME DESTINATION bin
	LIBRARY DESTINATION lib
	ARCHIVE DESTINATION lib)
install (FILES 
    "${PROJECT_BINARY_DIR}/include/PsuadeCmakeConfig.h"  
    ${psuade_HDRS}        
    DESTINATION include)

#######################################################
# With mingGw you have to include libgcc etc. in the package.  So do that here.
#######################################################
if( MINGW )
    get_filename_component( Mingw_Path ${CMAKE_CXX_COMPILER} PATH )
    set( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS ${Mingw_Path}/libgcc_s_dw2-1.dll ${Mingw_Path}/libstdc++-6.dll ${Mingw_Path}/libgfortran-3.dll ${Mingw_Path}/libquadmath-0.dll)
endif( MINGW )

include( InstallRequiredSystemLibraries )

if( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS )
    install( PROGRAMS ${CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS} DESTINATION bin COMPONENT System )
endif( CMAKE_INSTALL_SYSTEM_RUNTIME_LIBS )

###################################################
# Packaging
###################################################
include (InstallRequiredSystemLibraries)
set (CPACK_RESOURCE_FILE_LICENSE "${CMAKE_CURRENT_SOURCE_DIR}/LICENSE")
set (CPACK_PACKAGE_VERSION_MAJOR "${psuade_VERSION_MAJOR}")
set (CPACK_PACKAGE_VERSION_MINOR "${psuade_VERSION_MINOR}")
set (CPACK_PACKAGE_VERSION_PATCH "${psuade_VERSION_PATCH}")
set (CPACK_PACKAGE_CONTACT       "tong10@llnl.gov")

if("${CMAKE_SYSTEM_NAME}" STREQUAL "Windows")
  SET(CPACK_GENERATOR NSIS ZIP) #DEB and RPM are also options on appropriate systems
  SET(CPACK_NSIS_MODIFY_PATH ON)
  SET(CPACK_NSIS_CREATE_ICONS "CreateShortCut '\$SMPROGRAMS\\\\$STARTMENU_FOLDER\\\\${PSUADE_NAME}.lnk' '\$INSTDIR\\\\bin\\\\${PSUADE_NAME}.exe'") 

else()
  SET(CPACK_GENERATOR TGZ STGZ) #DEB and RPM are also options on appropriate sysftems
endif()

include (CPack)

#add_subdirectory (Examples/PDFTest) 
#add_subdirectory (Examples/ARSM) 
#MARS tests (mostly)
#add_subdirectory (scripts) 
#add_subdirectory (Examples/Bungee) 
#add_subdirectory (Examples/Morris20) 
#add_subdirectory (Examples/MCMCTest) 
#BOBYQA
#add_subdirectory (Examples/OptRosenbrock) 
#add_subdirectory (Examples/MOO) 

#C++ tests (using psuade as a library)
#add_subdirectory (Examples/C++_MOAT)

# enable dashboard scripting
include (CTest)

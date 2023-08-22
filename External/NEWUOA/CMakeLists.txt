file(GLOB newuoa_SRC
  "*.f"
)

#NEWUOA has a call to a function inside PSUADE, so it's undefined at library link time
#On some systems that means we can only build it statically
if(MINGW OR APPLE)
add_library(newuoa STATIC ${newuoa_SRC}) 
else(MINGW OR APPLE)
add_library(newuoa ${LIBRARY_TYPE} ${newuoa_SRC}) 
endif(MINGW OR APPLE)

install (TARGETS newuoa
  RUNTIME DESTINATION bin
  LIBRARY DESTINATION lib
  ARCHIVE DESTINATION lib)


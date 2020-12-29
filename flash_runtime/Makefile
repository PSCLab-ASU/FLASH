# ==== Begin prologue boilerplate.
.RECIPEPREFIX =
.SECONDEXPANSION:

NAME          := flash_runtime
BUILD_DIR     := ${GLOBAL_BUILD_DIR}
OUTPUTDIR     := ${GLOBAL_BUILD_DIR}/$(NAME)/
FLASH_VARIANT ?= ${NAME}
FLASH_BASE    ?= $(BUILD_DIR)/../
FLASH_RT      := $(FLASH_BASE)/flash_runtime/
UTILS_DIR     := $(FLASH_BASE)/utils

SHELL := /bin/bash
COMPILER=gcc

CXX.gcc := g++
CC.gcc  := g++
LD.gcc  := g++
AR.gcc  := ar

CXXFLAGS.gcc.debug := -O0 -g -fstack-protector-all
CXXFLAGS.gcc.release := -O3 -march=native -DNDEBUG
CXXFLAGS.gcc := -fPIC -fvisibility=default -std=c++2a ${CXXFLAGS.gcc.${BUILD}}

CXXFLAGS := ${CXXFLAGS.${COMPILER}}
CFLAGS   := ${CFLAGS.${COMPILER}}
CXX      := ${CXX.${COMPILER}}
CC       := ${CC.${COMPILER}}
LD       := ${LD.${COMPILER}}
AR       := ${AR.${COMPILER}}

LDFLAGS.common  := -std=c++2a -lstdc++ -lpthread -ldl
LDFLAGS.debug   := $(LDFLAGS.common)
LDFLAGS.release := $(LDFLAGS.common)
LDFLAGS         := ${LDFLAGS.${BUILD}}
LDLIBS          := 

all : create_build_dir backend

                   
backend : $(NAME)

#ifeq ($(FLASH_VARIANT),all)
#backend : $(NAME)
#else ifeq ($(FLASH_VARIANT),$(NAME))
#backend : $(NAME) 
#else
#backend :
#	@echo Bypassing making $(NAME) in $(CURDIR) 
#endif


COMPILE.CXX = ${CXX} -c $^ -o $@ -I$(dir $^) -I${UTILS_DIR} -I$(FLASH_BASE) -I$(FLASH_RT) ${CXXFLAGS} 

###################################################################################
#top level object generation
$(OUTPUTDIR)%.o : $(CURDIR)/%.cc
	@echo Compiling $(notdir $@)...
	$(COMPILE.CXX)

$(NAME)    :  $(patsubst %.cc, %.o, $(addprefix $(OUTPUTDIR), $(wildcard *.cc) ))

#####################################################################################

# Create the build directory and sub dirs on demand.
create_build_dir : ${OUTPUTDIR} 

${OUTPUTDIR} :
	@mkdir -p $@
        

clean:
	@rm -rf ${OUTPUTDIR} 

.PHONY : clean all 

# ==== End rest of boilerplate

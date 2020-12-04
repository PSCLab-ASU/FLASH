# ==== Begin prologue boilerplate.
.RECIPEPREFIX =
.SECONDEXPANSION:

BUILD         ?= debug
NAME          := utils
BUILD_DIR     := ${GLOBAL_BUILD_DIR}
OUTPUTDIR     := ${GLOBAL_BUILD_DIR}/$(NAME)/
FLASH_VARIANT ?= $(NAME)
FLASH_BASE    ?= $(BUILD_DIR)/../../

SHELL := /bin/bash
COMPILER=gcc

CXX.gcc := g++
CC.gcc  := g++
LD.gcc  := g++
AR.gcc  := ar

CXXFLAGS.gcc.debug := -O0 -g -fstack-protector-all
CXXFLAGS.gcc.release := -O3 -march=native -DNDEBUG
CXXFLAGS.gcc := -fPIC -std=c++2a ${CXXFLAGS.gcc.${BUILD}}

CXXFLAGS := ${CXXFLAGS.${COMPILER}}
CFLAGS   := ${CFLAGS.${COMPILER}}
CXX      := ${CXX.${COMPILER}}
CC       := ${CC.${COMPILER}}
LD       := ${LD.${COMPILER}}
AR       := ${AR.${COMPILER}}

LDFLAGS.common  := -lpthread -ldl
LDFLAGS.debug   := $(LDFLAGS.common)
LDFLAGS.release := $(LDFLAGS.common)
LDFLAGS         := ${LDFLAGS.${BUILD}}
LDLIBS          := 

all : create_build_dir backend write_linker_dep

ifeq ($(filter $(NAME),$(FLASH_VARIANT)),)
backend : $(NAME)
else ifeq ($(filter all,$(FLASH_VARIANT)),)
backend : $(NAME) 
else
backend :
	@echo Bypassing making $(NAME) in $(CURDIR) : FV=$(FLASH_VARIANT)
endif


COMPILE.CXX = ${CXX} -c $^ -o $@ -I$(dir $^) -I$(FLASH_BASE) ${CXXFLAGS} 

###################################################################################
#top level object generation
$(OUTPUTDIR)%.o : $(CURDIR)/%.cc
	@echo Compiling $(notdir $@)...
	@$(COMPILE.CXX)

$(NAME)    :  $(patsubst %.cc, %.o, $(addprefix $(OUTPUTDIR), $(wildcard *.cc) ))

#####################################################################################

# Create the build directory and sub dirs on demand.
create_build_dir : ${OUTPUTDIR} 

write_linker_dep :
	@echo $(LDLIBS) >> $(OUTPUTDIR)/LDINCS
	@echo $(LDFLAGS) >> $(OUTPUTDIR)/LDFLAGS 
	
${OUTPUTDIR} : 
	@mkdir -p $@
        

clean:
	@rm -rf ${OUTPUTDIR} 

.PHONY : clean all 

# ==== End rest of boilerplate

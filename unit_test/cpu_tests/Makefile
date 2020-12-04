# ==== Begin prologue boilerplate.
.RECIPEPREFIX =
.SECONDEXPANSION:

NAME          := cpu_runtime
UT_DIR        := unit_test/cpu_tests/
BUILD_DIR     := ${GLOBAL_BUILD_DIR}
OUTPUTDIR     := ${GLOBAL_BUILD_DIR}/$(NAME)/
OUTPUTDIR_UT  := $(OUTPUTDIR)/unit_test/
FLASH_VARIANT := all
FLASH_BASE    := $(BUILD_DIR)/../
FLASH_RT      := $(BUILD_DIR)/flash_runtime/

SHELL := /bin/bash
COMPILER=gcc

CXX.gcc := g++
CC.gcc  := g++
LD.gcc  := g++
AR.gcc  := ar

CXXFLAGS.gcc.debug := -O0 -g -fstack-protector-all
CXXFLAGS.gcc.release := -O3 -march=native -DNDEBUG
CXXFLAGS.gcc := -pthread -std=c++2a -fconcepts-ts -fPIC -Wno-return-type -Wno-return-local-addr ${CXXFLAGS.gcc.${BUILD}}

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
                   
ifeq ($(FLASH_VARIANT), all)
backend : $(NAME)
else ifeq ($(FLASH_VARIANT), $(NAME) )
backend : $(NAME) 
else
backend :
endif

ifeq ($(FLASH_VARIANT)_$(FLASH_OPT), all_add_unit_tests)
unit_tests : unit_test   
else ifeq ($(FLASH_VARIANT)_$(FLASH_OPT), $(NAME)_add_unit_tests)
unit_tests : unit_test   
else
unit_tests : 
endif

COMPILE.CXX = ${CXX} -c $^ -o $@ -I$(dir $^) -I$(FLASH_BASE) -I$(FLASH_RT) ${CXXFLAGS} 

###################################################################################
#top level object generation
$(OUTPUTDIR)%.o : $(CURDIR)/%.cc
	@echo Current direction is $(CURDIR)
	@echo Compiling $(notdir $@)...
	@$(COMPILE.CXX)
	#@echo @$(COMPILE.CXX)

$(NAME)    :  $(patsubst %.cc, %.o, $(addprefix $(OUTPUTDIR), $(wildcard *.cc) ))
unit_test  :  $(patsubst %.cc, %.o, $(addprefix $(OUTPUTDIR), $(wildcard unit_test/cpu_tests/* ) ))
#unit_test  : $$(subst .cc, .o, $$(addprefix $(OUTPUTDIR_UT), $$(wildcard unit_test/cpu_tests/* ) ))


#####################################################################################

# Create the build directory and sub dirs on demand.
create_build_dir : ${OUTPUTDIR} $(OUTPUTDIR_UT)

${OUTPUTDIR} $(OUTPUTDIR_UT): 
	@mkdir -p $@
        

clean:
	@rm -rf ${OUTPUTDIR} 
	@rm -rf $(OUTPUTDIR_UT)

.PHONY : clean all 

# ==== End rest of boilerplate

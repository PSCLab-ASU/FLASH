# ==== Begin prologue boilerplate.
.RECIPEPREFIX =
.SECONDEXPANSION:

export BUILD  ?= debug
BUILD_DIR     := ${CURDIR}/build/${BUILD}/
export GLOBAL_BUILD_DIR := $(BUILD_DIR)

FLASH_LIB     := $(BUILD_DIR)lib64/libflash_wrapper.so
FLASH_H       := $(BUILD_DIR)/include/flash.h
MAKEFILES_RT  := $(shell find ./*/ -type f -name "Makefile" -not -path "./*/*/*")
FLASH_VARIANT ?= cpu_runtime
FLASH_INC_H   := $(CURDIR)/flash_runtime
RT_LIBS       = $(shell cat $(BUILD_DIR)/*/LDINCS)
RT_SYMS       = $(shell cat $(BUILD_DIR)/*/LDFLAGS)
UTILS_DIR     := $(CURDIR)/utils

SHELL := /bin/bash
COMPILER=gcc

CXX.gcc := g++
CC.gcc  := g++
LD.gcc  := g++
AR.gcc  := ar

CXXFLAGS.gcc.debug := -O0 -g -fstack-protector-all
CXXFLAGS.gcc.release := -O3 -march=native -DNDEBUG
CXXFLAGS.gcc := -std=c++2a -shared -fPIC -fvisibility=hidden ${CXXFLAGS.gcc.${BUILD}}

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

all : create_build_dir build_flash_so move_flash_header
                   
###################################################################################
#top level object generation
build_flash_so : runtimes
	$(LD) $(shell find build/ -name \*.o ) -o $(FLASH_LIB) -I$(FLASH_INC_H) -I$(UTILS_DIR) $(CXXFLAGS) $(LDLIBS) $(RT_LIBS) $(RT_SYMS) $(LDFLAGS)

runtimes : $(addsuffix .mk, $(MAKEFILES_RT) )
	@echo "Completed Runtime builds"

$(addsuffix .mk, $(MAKEFILES_RT) ):
	@echo global_build_dir = $(GLOBAL_BUILD_DIR)
	@$(MAKE) -C $(dir $@) all FLASH_VARIANT=$(FLASH_VARIANT) FLASH_BASE=$(CURDIR)
#####################################################################################

#####################################################################################
#####################################################################################
#####################################################################################

move_flash_header : 
	@echo Copying flash.h...
	@cp $(CURDIR)/$(notdir $(FLASH_H) ) $(FLASH_H)

# Create the build directory and sub dirs on demand.
create_build_dir : $(dir ${FLASH_LIB} ) $(dir ${FLASH_H} )

$(dir ${FLASH_LIB} ) $(dir ${FLASH_H} ): 
	@mkdir -p $@

clean:
	@rm -rf ${build_dir}

.PHONY : clean all 

# ==== End rest of boilerplate

# ==== Begin prologue boilerplate.
.RECIPEPREFIX =
.SECONDEXPANSION:

export BUILD := debug
BUILD_DIR    := ${CURDIR}/${BUILD}/
FLASH_LIB    := $(BUILD_DIR)/lib/libflash_wrapper.so
NVCC_LIBS    := $()
NVCC_INC     := $()
OCL_LIBS     := $()
OCL_INC      := $()

SHELL := /bin/bash
COMPILER=gcc

CXX.gcc := gcc
CC.gcc  := gcc
LD.gcc  := gcc
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

all : create_build_dir build_flash_so
                   
ifeq ($(FLASH_VARIANT), gpu_only)
backends : gpu_runtime 
else ifeq ($(FLASH_VARIANT), fpga_only)
backends : fpga_runtime 
else ifeq ($(FLASH_VARIANT), cpu_only)
backends : cpu_runtime 
else ifeq ($(FLASH_VARIANT), all)
backends : backend_all 
else
backends : cpu_runtime
endif

###################################################################################
#top level object generation
$(BUILD_DIR)%.o : $(CURDIR)%.cc $(CURDIR)%.h
	@echo Compiling $(notdir $@)...
	@$(COMPILE.CXX)

flashrt        :  $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard flash_runtime/* ) ))
cpu_runtime    :  $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard cpu_runtime/*   ) ))
gpu_runtime    :  $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard gpu_runtime/*   ) ))
fpga_runtime   :  $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard fpga_runtime/*  ) ))
backend_all    :  cpu_runtime gpu_runtime fpga_runtime

unit_test_cpu  : $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard unit_test/cpu_tests/* ) ))
unit_test_gpu  : $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard unit_test/cpu_tests/* ) ))
unit_test_fpga : $$(subst .cc, .o, $$(addprefix $(BUILD_DIR), $$(wildcard unit_test/cpu_tests/* ) ))
unit_test_all  : unit_test_cpu unit_test_gpu unit_test_fpga


build_flash_so : flashrt backends
	$(LD) $(shell find build/ -name \*.o ) -o $(FLASH_LIB) $(LDLIBS) $(LDFLAGS) -shared -fvisibility=hidden


#####################################################################################

#####################################################################################
#####################################################################################
#####################################################################################

# Create the build directory and sub dirs on demand.
create_build_dir : ${build_dir} ${OUTPUT_DIRS}

${build_dir} ${OUTPUT_DIRS} : 
	@mkdir -p $@

clean:
	@rm -rf ${build_dir}

.PHONY : clean all 

# ==== End rest of boilerplate

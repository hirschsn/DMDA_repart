
all: libdmda_repart.a

PETSC_DIR = ~/opt/petsc

SRC = example.c dmda_repart.c
OBJ = $(SRC:.c=.o)
CLEANFILES = $(OBJ) libdmda_repart.a

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

libdmda_repart.a: dmda_repart.o
	$(AR) rc $@ $?
	$(RANLIB) $@

example: example.o libdmda_repart.a
	$(CLINKER) -o $@ $? $(PETSC_SNES_LIB) -L. -ldmda_repart

NP?=4
MPI?=mpich

run: example
	mpiexec.${MPI} -n ${NP} ./example

.PHONY: run

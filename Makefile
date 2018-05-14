
all: libdmda_repart.a

PETSC_DIR=~/opt/petsc

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

libdmda_repart.a: dmda_repart.o
	$(AR) rc $@ $?
	$(RANLIB) $@

example: example.o libdmda_repart.a
	$(CLINKER) -o example $? $(PETSC_SNES_LIB) -L. -ldmda_repart

NP?=4
MPI?=mpich

run: example
	mpiexec.${MPI} -n ${NP} ./example

.PHONY: example run

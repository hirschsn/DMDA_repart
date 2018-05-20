
all: libdmda_repart.a

PETSC_DIR = ~/opt/petsc

LIBSRC = dmda_repart.c dmda_repart_weights.c
LIBOBJ = $(LIBSRC:.c=.o)
EXSRC = examples/ex1.c examples/ex2.c examples/ex3.c
EXBIN = $(EXSRC:.c=)
SRC = $(LIBSRC) $(EXSRC)
OBJ = $(SRC:.c=.o)
CLEANFILES = $(OBJ) libdmda_repart.a $(EXBIN)

CFLAGS=-I.

include ${PETSC_DIR}/lib/petsc/conf/variables
include ${PETSC_DIR}/lib/petsc/conf/rules

libdmda_repart.a: $(LIBOBJ)
	$(AR) rc $@ $?
	$(RANLIB) $@

$(EXBIN): %: %.o libdmda_repart.a
	$(CLINKER) -o $@ $< $(PETSC_LIB) -L. -ldmda_repart

ex1: examples/ex1
ex2: examples/ex2
ex3: examples/ex3

NP?=4
MPI?=mpich

run1: ex1
	mpiexec.${MPI} -n ${NP} ./examples/ex1

run2: ex2
	mpiexec.${MPI} -n ${NP} ./examples/ex2

run3: ex3
	mpiexec.${MPI} -n ${NP} ./examples/ex3


.PHONY: run1 run2


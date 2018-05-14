
#ifndef DMDA_REPART_INCLUDED
#define DMDA_REPART_INCLUDED

#include <petscdmda.h>

/** Repartitioning of a Petsc DMDA.
 *
 * Creates a new DMDA and a new global vector based on the
 * old DMDA and the old global vector and replaces
 * the old ones with them.
 *
 * The ownership range of the new da is given by lx, ly and lz.
 *
 * Note: Reset TS, SNES, etc. afterwards!
 *
 * @param da Pointer to the DMDA to be replaced
 * @param X Pointer to the Vec to be replaced
 * @param lx Number of nodes per process in x direction
 * @param ly Number of nodes per process in y direction
 * @param lz Number of nodes per process in z direction
 */
PetscErrorCode DMDA_repart(DM* da, Vec *X, PetscInt lx[], PetscInt ly[], PetscInt lz[]);

#endif

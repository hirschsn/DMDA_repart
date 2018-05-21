
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
 * This function copies the TSIFunction, the ApplicationContext and the Field
 * Names from the old to the new, repartitioned da.
 *
 * Note: Reset TS, SNES, etc. afterwards!
 *
 * @param da Pointer to the DMDA to be replaced
 * @param X Pointer to the Vec to be replaced
 * @param lx Number of nodes per process in x direction
 * @param ly Number of nodes per process in y direction
 * @param lz Number of nodes per process in z direction
 * @param setFromOptions Set this to PETSC_TRUE if DMSetFromOptions should
 *                       be called on the newly created DMDA
 */
PetscErrorCode
DMDA_repart(DM* da, Vec *X, PetscInt lx[], PetscInt ly[], PetscInt lz[],
            PetscBool setFromOptions);


/** Determine new ownership ranges for repartitioning.
 *
 * @param da DMDA
 * @param W global Vec of weights for each cell
 * @param lx Number of cells per process in x-direction (out)
 * @param ly Number of cells per process in y-direction (out)
 * @param lz Number of cells per process in z-direction (out)
 */
PetscErrorCode
DMDA_repart_ownership_ranges(DM da, Vec W, 
                             PetscInt lx[], PetscInt ly[], PetscInt lz[]);

#endif

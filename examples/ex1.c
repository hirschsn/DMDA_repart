#include <petscdmda.h>

#include "dmda_repart.h"

typedef struct {
  PetscReal v;
  PetscReal w;
} Field;

// Creates a vector, prints its ownership, redistributes and prints the new
// ownership.
int main(int argc, char **argv)
{
  DM da;
  PetscErrorCode ierr;
  Vec X;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Repart test", "");
    CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = DMDACreate3d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,
                      /* Grid dimension */ 4, 4, 1,
                      /* Node grid */ 2, 2, 1, // 4 processes (currently)
                      /* dof */ 2, /* stencil width */ 1,
                      /* Nodes per cell */ (PetscInt[]){2, 2},
                                           (PetscInt[]){2, 2},
                                           (PetscInt[]){1},
                      &da); CHKERRQ(ierr);
  //ierr = DMSetFromOptions(da);CHKERRQ(ierr); 
  ierr = DMSetUp(da);CHKERRQ(ierr);
  
  ierr = DMDASetFieldName(da, 0, "test-values-1");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da, 1, "test-values-2");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da, &X); CHKERRQ(ierr);

  Field ***x;
  PetscReal v;
  PetscMPIInt myrank;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;

  ierr = DMDAVecGetArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);

  for (k = zs; k < zs+zm; k++) {
      for (j = ys; j < ys+ym; j++) {
          for (i = xs; i < xs+xm; i++) {
            v = i * 100 + j * 10 + k;
	    printf("[%i] Setting %i %i %i to %.0lf\n", myrank, k, j, i, v);
            x[k][j][i].v = v;
            x[k][j][i].w = v / 2;
          }
      }
  }

  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);

  ierr = DMDA_repart(&da, &X,
                     (PetscInt[]){1, 3},
                     (PetscInt[]){1, 3},
                     (PetscInt[]){1});

  // Unreliable hack to get DM swarm migration output not mixed
  // with the following output.
  fflush(stdout);
  MPI_Barrier(PETSC_COMM_WORLD);

  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  for (k = zs; k < zs+zm; k++) {
      for (j = ys; j < ys+ym; j++) {
          for (i = xs; i < xs+xm; i++) {
            printf("[%i] Have %i %i %i: %.0lf %.0lf\n", myrank, i, j, k, x[k][j][i].v, x[k][j][i].w);
          }
      }
  }

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
}

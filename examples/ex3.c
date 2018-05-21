#include <petscdmda.h>

#include "dmda_repart.h"

#define N_REPART_STEPS 10

typedef struct {
  PetscReal v;
} Field;

PetscInt get_value(PetscInt i, PetscInt j, PetscInt k, PetscInt grid[3])
{
  return i * grid[1] * grid[2] + j * grid[2] + k;
}

// Checks the contents of vector "X" against the values it should hold
PetscErrorCode check_vector(DM da, Vec X, PetscInt grid[3])
{
  PetscErrorCode ierr;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;
  PetscReal v;
  Field ***x;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da,X,&x);CHKERRQ(ierr);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        v = get_value(i, j, k, grid);
        if (x[k][j][i].v != v) {
          SETERRQ5(PETSC_COMM_WORLD, PETSC_ERR_PLIB, "Error: Have %i %i %i: %.0lf should be %.0lf\n", i, j, k, x[k][j][i].v, v);
        }
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


// Create a 3d DMDA and a vector, repart randomly and check vector contents
int main(int argc, char **argv)
{
  DM da;
  PetscErrorCode ierr;
  PetscMPIInt dims[3] = {0, 0, 0}, size, rank;
  PetscInt grid[3], d, *lx, *ly, *lz;
  Vec X, W;
  PetscReal v;
  Field ***x;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;
  PetscRandom r;
  double t1 = 0., t2 = 0.;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Repart test", "");
    CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Dims_create(size, 3, dims);

  for (d = 0; d < 3; ++d)
    grid[d] = dims[d] * 10;

  ierr = DMDACreate3d(PETSC_COMM_WORLD,
                      DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                      DMDA_STENCIL_STAR,
                      /* Grid dimension */ grid[0], grid[1], grid[2],
                      /* Node grid */ dims[0], dims[1], dims[2],
                      /* dof */ 1, /* stencil width */ 1,
                      /* Nodes per cell */ NULL,
                                           NULL,
                                           NULL,
                      &da); CHKERRQ(ierr);
  ierr = DMSetUp(da);CHKERRQ(ierr);

  ierr = DMDASetFieldName(da, 0, "test-values");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da, &X); CHKERRQ(ierr);

  ierr = DMDAVecGetArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        x[k][j][i].v = get_value(i, j, k, grid);
      }
    }
  }

  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);

  ierr = PetscMalloc3(dims[0], &lx,
                      dims[1], &ly,
                      dims[2], &lz); CHKERRQ(ierr);

  ierr = PetscRandomCreate(PETSC_COMM_WORLD, &r); CHKERRQ(ierr);
  ierr = PetscRandomSetType(r, PETSCRAND); CHKERRQ(ierr);


  for (i = 1; i <= N_REPART_STEPS; ++i) {
    ierr = PetscRandomSetInterval(r, 0.5, i * (rank + 1)); CHKERRQ(ierr);
    ierr = PetscRandomGetValueReal(r, &v);
    ierr = PetscRandomSetInterval(r, 0.5, v); CHKERRQ(ierr);

    ierr = DMCreateGlobalVector(da, &W); CHKERRQ(ierr);
    ierr = VecSetRandom(W, r);
    t1 -= MPI_Wtime();
    ierr = DMDA_repart_ownership_ranges(da, W, lx, ly, lz); CHKERRQ(ierr);
    t1 += MPI_Wtime();

    PetscPrintf(PETSC_COMM_WORLD, "\nRepart no %D\n", i);
    PetscPrintf(PETSC_COMM_WORLD, "================\n", i);
    PetscPrintf(PETSC_COMM_WORLD, "lx:");
    for (j = 0; j < dims[0]; ++j)
      PetscPrintf(PETSC_COMM_WORLD, " %D", lx[j]);
    PetscPrintf(PETSC_COMM_WORLD, "\nly:");
    for (j = 0; j < dims[1]; ++j)
      PetscPrintf(PETSC_COMM_WORLD, " %D", ly[j]);
    PetscPrintf(PETSC_COMM_WORLD, "\nlz:");
    for (j = 0; j < dims[2]; ++j)
      PetscPrintf(PETSC_COMM_WORLD, " %D", lz[j]);
    PetscPrintf(PETSC_COMM_WORLD, "\n");

    ierr = VecDestroy(&W);
    t2 -= MPI_Wtime();
    ierr = DMDA_repart(&da, &X, lx, ly, lz, PETSC_FALSE); CHKERRQ(ierr);
    t2 += MPI_Wtime();

    PetscPrintf(PETSC_COMM_WORLD, "Checking result...");
    ierr = check_vector(da, X, grid); CHKERRQ(ierr);
    PetscPrintf(PETSC_COMM_WORLD, " successful.\n");

  }

  t1 *= 1000.0 / N_REPART_STEPS;
  t2 *= 1000.0 / N_REPART_STEPS;
  PetscPrintf(PETSC_COMM_WORLD, "%D DMDA_repart_ownership_ranges took on average: %lf ms\n", N_REPART_STEPS, t1);
  PetscPrintf(PETSC_COMM_WORLD, "%D DMDA_repart                  took on average: %lf ms\n", N_REPART_STEPS, t2);

  ierr = PetscRandomDestroy(&r); CHKERRQ(ierr);
  ierr = PetscFree3(lx, ly, lz); CHKERRQ(ierr);

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
}


#include <petscdmda.h>

#include "dmda_repart.h"

typedef struct {
  PetscReal v;
  PetscReal w;
} Field;

static PetscErrorCode
fill_vec_3d(DM da, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;
  PetscReal v;
  Field ***x, ***y;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Y, &y); CHKERRQ(ierr);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        v = i * 100 + j * 10 + k;
        printf("[%i] Setting %i %i %i to %.0lf\n", rank, k, j, i, v);
        x[k][j][i].v = v;
        x[k][j][i].w = v / 2;
        y[k][j][i].v = v * 1000;
        y[k][j][i].w = v * 1000 / 2;
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Y, &y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
check_3d(DM da, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;
  Field ***x, ***y;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, X, &x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Y, &y);CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        printf("[%i] Have %i %i %i: %.0lf %.0lf %.0lf %.0lf\n", rank, i, j, k, x[k][j][i].v, x[k][j][i].w, y[k][j][i].v, y[k][j][i].w);
      }
    }
  }
  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Y, &y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
fill_vec_2d(DM da, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt xs, ys, xm, ym, i, j;
  PetscReal v;
  Field **x, **y;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Y, &y); CHKERRQ(ierr);

  for (j = ys; j < ys+ym; j++) {
    for (i = xs; i < xs+xm; i++) {
      v = i * 10 + j;
      printf("[%i] Setting %i %i to %.0lf\n", rank, j, i, v);
      x[j][i].v = v;
      x[j][i].w = v / 2;
      y[j][i].v = v * 1000;
      y[j][i].w = v * 1000 / 2;
    }
  }
  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Y, &y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
check_2d(DM da, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt xs, ys, xm, ym, i, j;
  Field **x, **y;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, X, &x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Y, &y);CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  for (j = ys; j < ys+ym; j++) {
    for (i = xs; i < xs+xm; i++) {
      printf("[%i] Have %i %i: %.0lf %.0lf %.0lf %.0lf\n", rank, i, j, x[j][i].v, x[j][i].w, y[j][i].v, y[j][i].w);
    }
  }
  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Y, &y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
fill_vec_1d(DM da, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt xs, xm, i;
  PetscReal v;
  Field *x, *y;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Y, &y); CHKERRQ(ierr);

  for (i = xs; i < xs+xm; i++) {
    v = i;
    printf("[%i] Setting %i to %.0lf\n", rank, i, v);
    x[i].v = v;
    x[i].w = v / 2;
    y[i].v = v * 1000;
    y[i].w = v * 1000 / 2;
  }
  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Y, &y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
check_1d(DM da, Vec X, Vec Y)
{
  PetscErrorCode ierr;
  PetscMPIInt rank;
  PetscInt xs, xm, i;
  Field *x, *y;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, X, &x);CHKERRQ(ierr);
  ierr = DMDAVecGetArray(da, Y, &y);CHKERRQ(ierr);
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  for (i = xs; i < xs+xm; i++) {
    printf("[%i] Have %i: %.0lf %.0lf %.0lf %.0lf\n", rank, i, x[i].v, x[i].w, y[i].v, y[i].w);
  }
  ierr = DMDAVecRestoreArray(da, X, &x); CHKERRQ(ierr);
  ierr = DMDAVecRestoreArray(da, Y, &y); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Creates a vector, prints its ownership, redistributes and prints the new
// ownership.
int main(int argc, char **argv)
{
  DM da;
  PetscErrorCode ierr;
  PetscInt dim = 3;
  Vec X, Y;

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Repart test", "");
    CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-d", &dim, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%iD test\n", dim); CHKERRQ(ierr);

  switch (dim) {
  case 3:
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
    break;
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        /* Grid dimension */ 4, 4,
                        /* Node grid */ 2, 2, // 4 processes (currently)
                        /* dof */ 2, /* stencil width */ 1,
                        /* Nodes per cell */ (PetscInt[]){2, 2},
                                             (PetscInt[]){2, 2},
                        &da); CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMDACreate1d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE,
                        /* Grid dimension */ 16,
                        /* dof */ 2, /* stencil width */ 1,
                        /* Nodes per cell */ (PetscInt[]){4, 4, 4, 4},
                        &da); CHKERRQ(ierr);
    break;
  }

  //ierr = DMSetFromOptions(da);CHKERRQ(ierr); 
  ierr = DMSetUp(da);CHKERRQ(ierr);
  
  ierr = DMDASetFieldName(da, 0, "test-values-1");CHKERRQ(ierr);
  ierr = DMDASetFieldName(da, 1, "test-values-2");CHKERRQ(ierr);

  ierr = DMCreateGlobalVector(da, &X); CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &Y); CHKERRQ(ierr);

  switch (dim) {
  case 3:
    ierr = fill_vec_3d(da, X, Y); CHKERRQ(ierr);
    ierr = DMDA_repartl(&da,
                        (PetscInt[]){1, 3},
                        (PetscInt[]){1, 3},
                        (PetscInt[]){1},
                        PETSC_FALSE,
                        &X, &Y, (Vec*) NULL); CHKERRQ(ierr);
    break;
  case 2:
    ierr = fill_vec_2d(da, X, Y); CHKERRQ(ierr);
    ierr = DMDA_repartl(&da,
                        (PetscInt[]){1, 3},
                        (PetscInt[]){1, 3},
                        NULL,
                        PETSC_FALSE,
                        &X, &Y, (Vec *) NULL); CHKERRQ(ierr);
    break;
  case 1:
    ierr = fill_vec_1d(da, X, Y); CHKERRQ(ierr);
    ierr = DMDA_repartl(&da,
                        (PetscInt[]){1, 1, 1, 13},
                        NULL,
                        NULL,
                        PETSC_FALSE,
                        &X, &Y, (Vec *) NULL); CHKERRQ(ierr);
    break;
  }

  // Unreliable hack to get DM swarm migration output not mixed
  // with the following output.
  fflush(stdout);
  MPI_Barrier(PETSC_COMM_WORLD);

  switch (dim) {
  case 3:
    ierr = check_3d(da, X, Y); CHKERRQ(ierr);
    break;
  case 2:
    ierr = check_2d(da, X, Y); CHKERRQ(ierr);
    break;
  case 1:
    ierr = check_1d(da, X, Y); CHKERRQ(ierr);
    break;
  }

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
}


#include <dmda_repart.h>

static PetscErrorCode
fill_vec_3d(DM da, Vec X)
{
  PetscErrorCode ierr;
  PetscMPIInt rank, size;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;
  PetscReal ****x;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        x[k][j][i][0] = (PetscReal) (size - rank);
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
fill_vec_2d(DM da, Vec X)
{
  PetscErrorCode ierr;
  PetscMPIInt rank, size;
  PetscInt xs, ys, xm, ym, i, j;
  PetscReal ***x;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);

  for (j = ys; j < ys+ym; j++) {
    for (i = xs; i < xs+xm; i++) {
      x[j][i][0] = (PetscReal) (size - rank);
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
fill_vec_1d(DM da, Vec X)
{
  PetscErrorCode ierr;
  PetscMPIInt rank, size;
  PetscInt xs, xm, i;
  PetscReal **x;

  PetscFunctionBegin;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);
  MPI_Comm_size(PETSC_COMM_WORLD, &size);

  ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);

  for (i = xs; i < xs+xm; i++) {
    x[i][0] = (PetscReal) (size - rank);
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static void
print_array(int *a, size_t size)
{
  size_t i;
  for (i = 0; i < size; ++i)
    printf(" %i", a[i]);
  printf("\n");
}

// Creates a vector of weights, determines new ownership ranges according to
// these weights and prints them.
int main(int argc, char **argv)
{
  DM da;
  PetscErrorCode ierr;
  Vec X;
  PetscInt dim = 3;
  int size, dims[3] = {0};

  ierr = PetscInitialize(&argc, &argv, NULL, NULL); CHKERRQ(ierr);

  ierr = PetscOptionsBegin(PETSC_COMM_WORLD, PETSC_NULL, "Repart test", "");
    CHKERRQ(ierr);
  ierr = PetscOptionsGetInt(NULL, NULL, "-d", &dim, NULL); CHKERRQ(ierr);
  ierr = PetscOptionsEnd();CHKERRQ(ierr);

  ierr = PetscPrintf(PETSC_COMM_WORLD, "%iD test\n", dim); CHKERRQ(ierr);

  MPI_Comm_size(PETSC_COMM_WORLD, &size);
  MPI_Dims_create(size, dim, dims);

  switch (dim) {
  case 1:
    ierr = DMDACreate1d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE,
                        /* Grid dimension */
                        10 * dims[0],
                        /* dof */ 1, /* stencil width */ 1,
                        /* Nodes per cell */ NULL,
                        &da); CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMDACreate2d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        /* Grid dimension */
                        10 * dims[0], 10 * dims[1],
                        /* Node grid */
                        dims[0], dims[1],
                        /* dof */ 1, /* stencil width */ 1,
                        /* Nodes per cell */ NULL, NULL,
                        &da); CHKERRQ(ierr);
    break;
  case 3:
    ierr = DMDACreate3d(PETSC_COMM_WORLD,
                        DM_BOUNDARY_NONE, DM_BOUNDARY_NONE, DM_BOUNDARY_NONE,
                        DMDA_STENCIL_STAR,
                        /* Grid dimension */
                        10 * dims[0], 10 * dims[1], 10 * dims[2],
                        /* Node grid */
                        dims[0], dims[1], dims[2],
                        /* dof */ 1, /* stencil width */ 1,
                        /* Nodes per cell */ NULL, NULL, NULL,
                        &da); CHKERRQ(ierr);
    break;
  }

  //ierr = DMSetFromOptions(da);CHKERRQ(ierr); 
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &X); CHKERRQ(ierr);

  switch (dim) {
  case 1:
    ierr = fill_vec_1d(da, X); CHKERRQ(ierr);
    break;
  case 2:
    ierr = fill_vec_2d(da, X); CHKERRQ(ierr);
    break;
  case 3:
    ierr = fill_vec_3d(da, X); CHKERRQ(ierr);
    break;
  }

  PetscInt *lx, *ly, *lz;
  ierr = PetscMalloc3(dims[0], &lx, dims[1], &ly, dims[2], &lz); CHKERRQ(ierr);
  ierr = DMDA_repart_ownership_ranges(da, X, lx, ly, lz); CHKERRQ(ierr);

  PetscMPIInt myrank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
  for (int i = 0; i < size; ++i) {
    MPI_Barrier(PETSC_COMM_WORLD);
    if (i != myrank)
      continue;

    printf("[%i] lx:", myrank);
    print_array(lx, dims[0]);
    if (dim == 1)
      continue;
    printf("[%i] ly:", myrank);
    print_array(ly, dims[1]);
    if (dim == 2)
      continue;
    printf("[%i] lz:", myrank);
    print_array(lz, dims[2]);
    fflush(stdout);
  }

  ierr = PetscFree3(lx, ly, lz);
  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
}


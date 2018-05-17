
#include <dmda_repart.h>

// Creates a vector of weights, determines new ownership ranges according to
// these weights and prints them.
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
                      /* Grid dimension */ 9, 9, 1,
                      /* Node grid */ 3, 3, 1, // 4 processes (currently)
                      /* dof */ 2, /* stencil width */ 1,
                      /* Nodes per cell */ (PetscInt[]){3, 3, 3},
                                           (PetscInt[]){3, 3, 3},
                                           (PetscInt[]){1},
                      &da); CHKERRQ(ierr);

  //ierr = DMSetFromOptions(da);CHKERRQ(ierr); 
  ierr = DMSetUp(da);CHKERRQ(ierr);
  ierr = DMCreateGlobalVector(da, &X); CHKERRQ(ierr);

  PetscMPIInt rank;
  PetscInt xs, ys, zs, xm, ym, zm, i, j, k;
  PetscReal ****x;
  MPI_Comm_rank(PETSC_COMM_WORLD, &rank);

  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);

  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        //x[k][j][i][0] = 1.0;
        //x[k][j][i][0] = (PetscReal) rank;
        x[k][j][i][0] = (PetscReal) (3.0 - rank);
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);


  PetscInt lx[3] = {-1, -1, -1}, ly[3] = {-1, -1, -1}, lz[1] = {-1};
  ierr = DMDA_repart_ownership_ranges(da, X, lx, ly, lz);

  PetscMPIInt myrank;
  MPI_Comm_rank(PETSC_COMM_WORLD, &myrank);
  for (int i = 0; i < 4; ++i) {
    MPI_Barrier(PETSC_COMM_WORLD);
    if (i != myrank)
      continue;
    printf("[%i] lx: %i %i %i\n", myrank, lx[0], lx[1], lx[2]);
    printf("[%i] ly: %i %i %i\n", myrank, ly[0], ly[1], ly[2]);
    printf("[%i] lz: %i\n", myrank, lz[0]);
    fflush(stdout);
  }

  ierr = VecDestroy(&X);CHKERRQ(ierr);
  ierr = DMDestroy(&da);CHKERRQ(ierr);
}


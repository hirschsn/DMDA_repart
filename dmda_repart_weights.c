
#include <petscsys.h>
#include <petscvec.h>
#include <petscdmda.h>
#include <mpi.h>

typedef struct {
  DM da;
  MPI_Comm comm;
  PetscInt xyzm[3]; // Local grid size {xm, ym, zm} in Petsc speak
  // "cs" is a set of three vectors in (one in x, y, z-direction)
  // with each element, e.g., cs[0][i] holding the sum
  // over the co-dimension (y-z-plane in this case) for x=i.
  // "cs[1]" und "cs[2]" are defined likewise in y- and z-direction,
  // respectively.
  PetscReal *cs[3];

  // n: Coordinates of this process on the process grid.
  // N: Dimension of the process grid, i.e. "(m, n, p)" in Petsc speak.
  PetscInt n[3], N[3];

  MPI_Datatype mpi_petsc_real, mpi_petsc_int;
} PState;

// Inverse of rank_of_coord
static void
proc_coord_of_rank(PetscMPIInt rank, PetscInt n[3], PetscInt N[3])
{
  n[0] = rank % N[0];
  rank /= N[0];
  n[1] = rank % N[1];
  n[2] = rank / N[1];
}

// Initializes a PState struct.
static PetscErrorCode
PStateCreate(PState *ps, DM da)
{
  PetscErrorCode ierr;
  PetscInt myrank;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, NULL, NULL, NULL,
                        &ps->xyzm[0], &ps->xyzm[1], &ps->xyzm[2]);
    CHKERRQ(ierr);
  ierr = PetscMalloc3(ps->xyzm[0], &ps->cs[0],
                      ps->xyzm[1], &ps->cs[1],
                      ps->xyzm[2], &ps->cs[2]); CHKERRQ(ierr);
  ps->da = da;

  ierr = PetscObjectGetComm((PetscObject) da, &ps->comm); CHKERRQ(ierr);

  MPI_Comm_rank(ps->comm, &myrank);
  ierr = DMDAGetInfo(da, NULL, NULL, NULL, NULL,
                     &ps->N[0], &ps->N[1], &ps->N[2],
                     NULL, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  proc_coord_of_rank(myrank, ps->n, ps->N);

  ierr = PetscDataTypeToMPIDataType(PETSC_REAL, &ps->mpi_petsc_real);
    CHKERRQ(ierr);
  ierr = PetscDataTypeToMPIDataType(PETSC_INT, &ps->mpi_petsc_int);
    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode
PStateDestroy(PState *ps)
{
  return PetscFree3(ps->cs[0], ps->cs[1], ps->cs[2]);
}

// Produces local 1d co-dim sums from "W" and stores the in "ps"
static PetscErrorCode
PStateLocalSum(PState *ps, Vec W)
{
  PetscErrorCode ierr;
  PetscInt i, j, k, xs, ys, zs, xm, ym, zm;
  PetscReal ****x, el;

  PetscFunctionBegin;

  for (j = 0; j < 3; ++j)
    for (i = 0; i < ps->xyzm[j]; ++i)
        ps->cs[j][i] = 0.0;

  ierr = DMDAGetCorners(ps->da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  for (k = 0; k < zm; k++) {
    for (j = 0; j < ym; j++) {
      for (i = 0; i < xm; i++) {
        el = x[zs+k][ys+j][xs+i][0];
        ps->cs[0][i] += el;
        ps->cs[1][j] += el;
        ps->cs[2][k] += el;
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(ps->da, W, &x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Globally sums the 1d co-dim sum vectors.
// Note: LocalSum *must* have been called before.
static PetscErrorCode
PStateGlobalSum(PState *ps)
{
  // A process with process grid indices (nx, ny, nz)
  // sends its cxs data to (nx,  *,  *)
  //       its cys data to ( *, ny,  *)
  //   and its czs data to ( *,  *, nz).
  // All processes reduce this data in place to their local sums, i.e.
  // all will have a global sum of that particular co-dimension of the
  // complete grid.
  PetscInt d;
  MPI_Comm c[3];
  MPI_Request req[3];

  PetscFunctionBegin;

  for (d = 0; d < 3; ++d) {
    // Key is not important since we do an Allreduce.
    MPI_Comm_split(ps->comm, ps->n[d], 0, &c[d]);
    MPI_Iallreduce(MPI_IN_PLACE, ps->cs[d], ps->xyzm[d], ps->mpi_petsc_real,
                   MPI_SUM, c[d], &req[d]);
  }

  MPI_Waitall(3, req, MPI_STATUSES_IGNORE);

  for (d = 0; d < 3; ++d)
    MPI_Comm_free(&c[d]);

  PetscFunctionReturn(0);
}

// Determines 1d ownership range "ld" from "cs".
// "Cs" is of length "len" and "ld" of length ps->N[x] where x is the
// direction of the field.
// "Comm" is a communicator of a 1d subset of processes along a particular
// dimension.
static PetscErrorCode
create_1d_subdomains(PState *ps, PetscReal *cs, PetscInt *ld, PetscInt len,
                     MPI_Comm comm, MPI_Request *req)
{
  PetscInt i, csize, proc;
  PetscReal lsum = 0., prefix = 0., oldprefix, target, gsum;
  MPI_Request r1, r2;

  PetscFunctionBegin;

  for (i = 0; i < len; ++i)
    lsum += cs[i];
  
  // Determine target and prefix load
  // Global sum is the same regardless of direction, so calculate it only once
  MPI_Iallreduce(&lsum, &gsum, 1, ps->mpi_petsc_real, MPI_SUM, comm, &r1);
  MPI_Iexscan(&lsum, &prefix, 1, ps->mpi_petsc_real, MPI_SUM, comm, &r2);
  MPI_Comm_size(comm, &csize);

  for (i = 0; i < csize; ++i)
    ld[i] = 0;

  MPI_Wait(&r1, MPI_STATUS_IGNORE);
  target = gsum / csize;

  MPI_Wait(&r2, MPI_STATUS_IGNORE);
  // Assign 1d ranges to processes and create the ownership range
  for (i = 0; i < len; ++i) {
    // "Heuristic 2" from:
    // MIGUET, Serge; PIERSON, Jean-Marc. Heuristics for 1d rectilinear
    // partitioning as a low cost and high quality answer to dynamic load
    // balancing. In: International Conference on High-Performance Computing
    // and Networking. Springer, Berlin, Heidelberg, 1997. S. 550-564
    oldprefix = prefix;
    prefix += cs[i];
    proc = (prefix + oldprefix) / (2 * target);
    if (proc >= csize)
      proc = csize - 1;
    ld[proc] += 1;
  }

  MPI_Iallreduce(MPI_IN_PLACE, ld, csize, ps->mpi_petsc_int, MPI_SUM, comm,
                 req);

  PetscFunctionReturn(0);
}

// Determines the ownership ranges "ls" = {lx, ly, lz}.
// Note: PStateGlobalSum must have been called before.
static PetscErrorCode
PStateReduce(PState *ps, PetscInt *ls[3])
{
  PetscErrorCode ierr;
  MPI_Comm c[3]; 
  MPI_Request req[3];
  PetscInt d;
  int color, ndir, nndir;

  PetscFunctionBegin;

  for (d = 0; d < 3; ++d) {
    ndir = (d + 1) % 3;
    nndir = (d + 2) % 3;
    // Direction perpendicular to the one of PStateGlobalSum.
    // I.e. the x-direction for a given ny, nz: (*, ny, nz).
    // Therefore, we create a unique integer for every
    // (ny, nz) pair.
    color = ps->n[ndir] * ps->N[nndir] + ps->n[nndir];
    MPI_Comm_split(ps->comm, color, 0, &c[d]);
    ierr = create_1d_subdomains(ps, ps->cs[d], ls[d], ps->xyzm[d], c[d],
                                &req[d]); CHKERRQ(ierr);
  }

  MPI_Waitall(3, req, MPI_STATUSES_IGNORE);

  for (d = 0; d < 3; ++d)
    MPI_Comm_free(&c[d]);

  PetscFunctionReturn(0);
}


PetscErrorCode
DMDA_repart_ownership_ranges(DM da, Vec W,
                             PetscInt lx[], PetscInt ly[], PetscInt lz[])
{
  PetscErrorCode ierr;
  PetscReal el;
  PetscInt dim;
  PState ps;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  if (dim != 3) {
    PetscObjectGetComm((PetscObject) da, &comm);
    SETERRQ(comm, PETSC_ERR_SUP, "DMDA_repart only implemented for 3d DMDAs.");
  }

  ierr = VecMin(W, NULL, &el);
  if (el < 0.0) {
    PetscObjectGetComm((PetscObject) da, &comm);
    SETERRQ(comm, PETSC_ERR_ARG_WRONG, "DMDA_repart_ownership_ranges: Weights must be positive!");
  }

  ierr = PStateCreate(&ps, da); CHKERRQ(ierr);
  ierr = PStateLocalSum(&ps, W); CHKERRQ(ierr);
  ierr = PStateGlobalSum(&ps); CHKERRQ(ierr);
  ierr = PStateReduce(&ps, (PetscInt *[]){lx, ly, lz}); CHKERRQ(ierr);

  PStateDestroy(&ps);
  PetscFunctionReturn(0);
}


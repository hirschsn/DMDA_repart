
#include <petscsys.h>
#include <petscvec.h>
#include <petscdmda.h>
#include <mpi.h>
#include <math.h>

typedef struct {
  DM da;
  MPI_Comm comm;
  PetscInt dim; // Number of dimensions
  PetscInt dof; // Number of dof in weight vector
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

  PetscInt grid_min; // Minimum number of grid points in any direction

  MPI_Datatype mpi_petsc_real, mpi_petsc_int;
} PState;

// Inverse of rank_of_coord
static void
PStateSetProcCoords(PState *ps, PetscMPIInt rank)
{
  ps->n[0] = rank % ps->N[0];

  if (ps->dim == 1)
    return;

  rank /= ps->N[0];
  ps->n[1] = rank % ps->N[1];

  if (ps->dim == 2)
    return;

  ps->n[2] = rank / ps->N[1];
}

// Initializes a PState struct.
static PetscErrorCode
PStateCreate(PState *ps, DM da, PetscInt grid_min)
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
  ps->grid_min = grid_min;

  ierr = PetscObjectGetComm((PetscObject) da, &ps->comm); CHKERRQ(ierr);

  MPI_Comm_rank(ps->comm, &myrank);
  ierr = DMDAGetInfo(da, &ps->dim, NULL, NULL, NULL,
                     &ps->N[0], &ps->N[1], &ps->N[2],
                     &ps->dof, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  PStateSetProcCoords(ps, myrank);

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
PStateLocalSum3D(PState *ps, Vec W)
{
  PetscErrorCode ierr;
  PetscInt i, j, k, xs, ys, zs, xm, ym, zm, d;
  PetscReal ****x, el;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(ps->da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  for (k = 0; k < zm; k++) {
    for (j = 0; j < ym; j++) {
      for (i = 0; i < xm; i++) {
        el = 0.0;
        for (d = 0; d < ps->dof; d++)
          el += x[zs+k][ys+j][xs+i][d];
        ps->cs[0][i] += el;
        ps->cs[1][j] += el;
        ps->cs[2][k] += el;
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
PStateLocalSum2D(PState *ps, Vec W)
{
  PetscErrorCode ierr;
  PetscInt i, j, xs, ys, xm, ym, d;
  PetscReal ***x, el;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(ps->da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  for (j = 0; j < ym; j++) {
    for (i = 0; i < xm; i++) {
      el = 0.0;
      for (d = 0; d < ps->dof; d++)
        el += x[ys+j][xs+i][d];
      ps->cs[0][i] += el;
      ps->cs[1][j] += el;
    }
  }
  ierr = DMDAVecRestoreArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
PStateLocalSum1D(PState *ps, Vec W)
{
  PetscErrorCode ierr;
  PetscInt i, xs, xm, d;
  PetscReal **x, el;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(ps->da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);
  ierr = DMDAVecGetArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  for (i = 0; i < xm; i++) {
    el = 0.0;
    for (d = 0; d < ps->dof; d++)
      el += x[xs+i][d];
    ps->cs[0][i] += el;
  }
  ierr = DMDAVecRestoreArrayDOF(ps->da, W, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
PStateLocalSum(PState *ps, Vec W)
{
  static PetscErrorCode (*sum_fn[])(PState *, Vec) = {
    [1] = PStateLocalSum1D,
    [2] = PStateLocalSum2D,
    [3] = PStateLocalSum3D
  };

  PetscErrorCode ierr;
  PetscInt i, j;

  PetscFunctionBegin;
  // Zero out all sum vectors
  for (j = 0; j < ps->dim; ++j)
    for (i = 0; i < ps->xyzm[j]; ++i)
        ps->cs[j][i] = 0.0;

  ierr = sum_fn[ps->dim](ps, W); CHKERRQ(ierr);
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

  for (d = 0; d < ps->dim; ++d) {
    // Key is not important since we do an Allreduce.
    // NB: For 1d this is N allreduces over 1 process each
    MPI_Comm_split(ps->comm, ps->n[d], 0, &c[d]);
    MPI_Iallreduce(MPI_IN_PLACE, ps->cs[d], ps->xyzm[d], ps->mpi_petsc_real,
                   MPI_SUM, c[d], &req[d]);
  }

  MPI_Waitall(ps->dim, req, MPI_STATUSES_IGNORE);

  for (d = 0; d < ps->dim; ++d)
    MPI_Comm_free(&c[d]);

  PetscFunctionReturn(0);
}

// Determines 1d ownership range "ld" from "cs".
// "Cs" is of length "len" and "ld" of length ps->N[x] where x is the
// direction of the field.
// "Comm" is a communicator of a 1d subset of processes along a particular
// dimension.
// This function does not respect ps->grid_min!
static PetscErrorCode
create_1d_subdomains_par(PState *ps, PetscReal *cs, PetscInt *ld, PetscInt len,
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

  if (!isnormal(target) || !isnormal(1.0 / target)) {
    SETERRQ(comm, PETSC_ERR_SUP,
            "Sum of weights is not a normal floating point value."
            " All weights zero?");
  }

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

// Determines 1d ownership range "ld" from "cs".
// "Cs" is of length "len" and "ld" of length ps->N[x] where x is the
// direction of the field.
// "Comm" is a communicator of a 1d subset of processes along a particular
// dimension.
static PetscErrorCode
create_1d_subdomains_seq(PState *ps, PetscReal *cs, PetscInt *ld, PetscInt len,
                         MPI_Comm comm, MPI_Request *req)
{
  struct {
    PetscInt proc;   // Current partition
    PetscInt ngridp; // Number of grid points currently assigned to "proc"
    PetscInt gloi;   // Global index in global cs field
    PetscReal w;     // Currently assigned weight to "proc"
  } state = {0, 0, 0, 0.0};

  PetscInt i, left_processes, left_ngridp, glolen;
  PetscReal lsum = 0., gsum, target;
  PetscMPIInt csize, rank;
  MPI_Request r1, r2;

  PetscFunctionBegin;

  for (i = 0; i < len; ++i)
    lsum += cs[i];
  
  // Determine target and prefix load
  // Global sum is the same regardless of direction, so calculate it only once
  MPI_Iallreduce(&lsum, &gsum, 1, ps->mpi_petsc_real, MPI_SUM, comm, &r1);
  MPI_Iallreduce(&len, &glolen, 1, ps->mpi_petsc_int, MPI_SUM, comm, &r2);
  MPI_Comm_size(comm, &csize);
  MPI_Comm_rank(comm, &rank);

  for (i = 0; i < csize; ++i)
    ld[i] = 0;

  MPI_Wait(&r1, MPI_STATUS_IGNORE);
  target = gsum / csize;

  if (!isnormal(target) || !isnormal(1.0 / target)) {
    SETERRQ(comm, PETSC_ERR_SUP,
            "Sum of weights is not a normal floating point value."
            " All weights zero?");
  }

  MPI_Wait(&r2, MPI_STATUS_IGNORE);

  // Sanity check of ps->grid_min parameter
  if (csize * ps->grid_min > glolen) {
    SETERRQ2(comm, PETSC_ERR_SUP,
            "Not enough grid cells for grid min width of %i and %i"
            " processes in some direction.", ps->grid_min, csize);
  }

  // Sequential loop over all processes
  if (rank > 0)
    MPI_Recv(&state, sizeof(state), MPI_BYTE, rank - 1, 0, comm, MPI_STATUS_IGNORE);

  for (i = 0; i < len; ++i) {
    left_ngridp = glolen - state.gloi;
    left_processes = csize - state.proc - 1;

    state.w += cs[i];
    // Also assign to the next process if not enough grid cells would be left
    // for the rest of the processes
    if ((state.w > target && state.ngridp >= ps->grid_min)
        || (left_ngridp <= left_processes * ps->grid_min)) {
      state.proc++;
      state.w = 0.0;
      state.ngridp = 0;
    }

    if (state.proc >= csize)
      state.proc = csize - 1;

    ld[state.proc] += 1;
    state.ngridp++;

    state.gloi++;
  }

  if (rank < csize - 1)
    MPI_Send(&state, sizeof(state), MPI_BYTE, rank + 1, 0, comm);
  // Loop end

  MPI_Iallreduce(MPI_IN_PLACE, ld, csize, ps->mpi_petsc_int, MPI_SUM, comm,
                 req);

  PetscFunctionReturn(0);
}

typedef PetscErrorCode (*MultiSectionFn1D)(PState *, PetscReal *,
        PetscInt *, PetscInt, MPI_Comm, MPI_Request *);

static PetscInt
PStateReduceGetColor(PState *ps, PetscInt d)
{
  // Direction perpendicular to the one of PStateGlobalSum.
  // I.e. the x-direction for a given ny, nz: (*, ny, nz).
  // Therefore, we create a unique integer for every
  // (ny, nz) pair.
  PetscInt ndir = (d + 1) % ps->dim;
  PetscInt nndir = (d + 2) % ps->dim;

  if (ps->dim == 1)
    return 0;
  else if (ps->dim == 2)
    return ps->n[ndir];
  else // 3
    return ps->n[ndir] * ps->N[nndir] + ps->n[nndir];
}

// Determines the ownership ranges "ls" = {lx, ly, lz}.
// Note: PStateGlobalSum must have been called before.
static PetscErrorCode
PStateReduce(PState *ps, PetscInt *ls[3], MultiSectionFn1D create_1d_subdomains)
{
  PetscErrorCode ierr;
  MPI_Comm c[3]; 
  MPI_Request req[3];
  PetscInt d;

  PetscFunctionBegin;

  for (d = 0; d < ps->dim; ++d) {
    // The "key" value of "ps->n[d]" is necessary for the ordering of the
    // Allgather* operations in create_1d_sudomains_seq.
    MPI_Comm_split(ps->comm, PStateReduceGetColor(ps, d), ps->n[d], &c[d]);
    ierr = create_1d_subdomains(ps, ps->cs[d], ls[d], ps->xyzm[d], c[d],
                                &req[d]); CHKERRQ(ierr);
  }

  MPI_Waitall(ps->dim, req, MPI_STATUSES_IGNORE);

  for (d = 0; d < ps->dim; ++d)
    MPI_Comm_free(&c[d]);

  PetscFunctionReturn(0);
}

static PetscErrorCode
PStateCheckIntegrity(PState *ps, PetscInt *ls[])
{
  PetscErrorCode ierr;
  PetscInt i, d, sum, grid_size[3];

  PetscFunctionBegin;
  ierr = DMDAGetInfo(ps->da, NULL, &grid_size[0], &grid_size[1], &grid_size[2],
                     NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL, NULL);
    CHKERRQ(ierr);

  for (d = 0; d < ps->dim; ++d) {
    sum = 0;
    for (i = 0; i < ps->N[d]; ++i) {
      if (ls[d][i] == 0) {
        SETERRQ(ps->comm, PETSC_ERR_SUP,
                "DMDA_repart_ownership_ranges:"
                " At least one ownership range is zero."
                " Maybe weights too skewed?");
      }
      sum += ls[d][i];
    }

    if (sum != grid_size[d]) {
      SETERRQ(ps->comm, PETSC_ERR_SUP,
              "DMDA_repart_ownership_ranges:"
              " Internal error: Ownership range does not sum to grid size."
              " File a bug report.");
    }
  }
  PetscFunctionReturn(0);
}

PetscErrorCode
SetORangesFromDMDA(DM da, PetscInt lx[], PetscInt ly[], PetscInt lz[])
{
  PetscErrorCode ierr;
  const PetscInt *slx, *sly, *slz;
  PetscInt dim, m, n, p;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, &dim, NULL, NULL, NULL, &m, &n, &p, NULL , NULL,
                     NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  ierr = DMDAGetOwnershipRanges(da, &slx, &sly, &slz); CHKERRQ(ierr);
  ierr = PetscMemcpy(lx, slx, m * sizeof(PetscInt)); CHKERRQ(ierr);
  if (dim >= 2) {
    ierr = PetscMemcpy(ly, sly, n * sizeof(PetscInt)); CHKERRQ(ierr);
  }
  if (dim >= 3) {
    ierr = PetscMemcpy(lz, slz, p * sizeof(PetscInt)); CHKERRQ(ierr);
  }

  PetscFunctionReturn(0);
}

PetscErrorCode
DMDA_repart_ownership_ranges(DM da, Vec W,
                             PetscInt lx[], PetscInt ly[], PetscInt lz[],
                             PetscInt grid_min)
{
  PetscErrorCode ierr;
  PetscReal el, sum;
  PState ps;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = VecMin(W, NULL, &el); CHKERRQ(ierr);
  if (el < 0.0) {
    PetscObjectGetComm((PetscObject) da, &comm);
    SETERRQ(comm, PETSC_ERR_ARG_WRONG,
            "DMDA_repart_ownership_ranges: Weights must be positive!");
  }

  ierr = VecSum(W, &sum); CHKERRQ(ierr);
  if (sum == 0.0 || !isnormal(sum) || !isnormal(1.0 / sum)) {
    PetscObjectGetComm((PetscObject) da, &comm);
    PetscPrintf(comm, "[DMDA_repart_ownership_ranges] WARNING:"
                      " Sum of weight vector is zero or not a normal floating-point value."
                      " Returning current ownership ranges of the DMDA.\n");
    ierr = SetORangesFromDMDA(da, lx, ly, lz); CHKERRQ(ierr);
    PetscFunctionReturn(0);
  }

  ierr = PStateCreate(&ps, da, grid_min); CHKERRQ(ierr);
  ierr = PStateLocalSum(&ps, W); CHKERRQ(ierr);
  ierr = PStateGlobalSum(&ps); CHKERRQ(ierr);
  // The "_par" version works in parallel but cannot respect "grid_min".
  if (grid_min > 0) {
    ierr = PStateReduce(&ps, (PetscInt *[]){lx, ly, lz},
                        create_1d_subdomains_seq); CHKERRQ(ierr);
  } else {
    ierr = PStateReduce(&ps, (PetscInt *[]){lx, ly, lz},
                        create_1d_subdomains_par); CHKERRQ(ierr);
  }
  ierr = PStateCheckIntegrity(&ps, (PetscInt *[]){lx, ly, lz}); CHKERRQ(ierr);

  PStateDestroy(&ps);
  PetscFunctionReturn(0);
}


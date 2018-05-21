
#include "dmda_repart.h"

#include <petscdmswarm.h>
#include <petscts.h>


/*******************************************/
/* Grid partitioning storage and functions */
/*******************************************/

typedef struct {
  PetscInt m, n, p;      // Number of processes in x, y, z direction
  PetscInt *lx, *ly, *lz; // Number of cells in direction x, y, z per process
  PetscInt *prefx, *prefy, *prefz; // Prefixes in x, y, z direction
} GridPartitioning;

// Initializes a GridPartitioning struct and allocates data.
static PetscErrorCode
GridPartitioningCreate(GridPartitioning *part,
                       PetscInt m, PetscInt n, PetscInt p)
{
  part->m = m; part->n = n; part->p = p;
  return PetscMalloc6(m, &part->lx,
                      m + 1, &part->prefx,
                      n, &part->ly,
                      n + 1, &part->prefy,
                      p, &part->lz,
                      p + 1, &part->prefz);
}

static PetscErrorCode GridPartitioningDestroy(GridPartitioning *p)
{
  return PetscFree6(p->lx, p->prefx, p->ly, p->prefy, p->lz, p->prefz);
}

// Prepares a GridPartitioning struct for rank search
static void GridPartitioningBuildPrefixes(GridPartitioning *p)
{
  PetscInt i;
  p->prefx[0] = p->prefy[0] = p->prefz[0] = 0;
  for (i = 1; i <= p->m; ++i)
    p->prefx[i] = p->prefx[i - 1] + p->lx[i - 1];
  for (i = 1; i <= p->n; ++i)
    p->prefy[i] = p->prefy[i - 1] + p->ly[i - 1];
  for (i = 1; i <= p->p; ++i)
    p->prefz[i] = p->prefz[i - 1] + p->lz[i - 1];
}

// Initializes a GridPartitioning struct from nodes pre grid dimension array.
// They have the same meaning as ownership ranges in Petsc.
static PetscErrorCode
GridPartitioningCreateFromLs(GridPartitioning *part,
                             PetscInt m, PetscInt n, PetscInt p,
                             PetscInt lx[], PetscInt ly[], PetscInt lz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  GridPartitioningCreate(part, m, n, p);
  ierr = PetscMemcpy(part->lx, lx, part->m * sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscMemcpy(part->ly, ly, part->n * sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscMemcpy(part->lz, lz, part->p * sizeof(PetscInt)); CHKERRQ(ierr);

  GridPartitioningBuildPrefixes(part);
  PetscFunctionReturn(0);
}

// Returns the owner rank of a cell with global index (x, y, z)
// under the partitioning "p".
static PetscInt
GridPartitioningGetOwnerRank(GridPartitioning *p,
                             PetscInt x, PetscInt y, PetscInt z)
{
  PetscInt i, mm = -1, nn = -1, pp = -1;

  // TODO: binary search
  for (i = 1; i <= p->m; ++i) {
    if (p->prefx[i] > x) {
      mm = i - 1;
      break;
    }
  }

  for (i = 1; i <= p->n; ++i) {
    if (p->prefy[i] > y) {
      nn = i - 1;
      break;
    }
  }

  for (i = 1; i <= p->p; ++i) {
    if (p->prefz[i] > z) {
      pp = i - 1;
      break;
    }
  }

  if (mm == -1 || nn == -1 || pp == -1) {
    PetscFPrintf(PETSC_COMM_WORLD, stderr,
                 "Error: Could not find owner rank of cell.\n");
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  // Ordering from Telescope:
  // https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/pc/impls/telescope/telescope_dmda.c.html
  return mm + nn * p->m + pp * p->m * p->n;
}


/******************/
/* Data migration */
/******************/

// Data migration state storage
typedef struct {
  DM ds;         // Swarm DM used for migration
  PetscInt dof;  // Number of degrees of freedom (block size of "field")
  PetscInt size; // Number of process local data points

  PetscReal *field; // Data from or for this process
                    // (depends if DoMigrate has been called)

  PetscInt *index; // Indices corresponding to the entries in "field"

  PetscInt *ranks;     // Target ranks (new owners)

  GridPartitioning *pnew; // GridPartitioning struct that defines the
                          // new partitioning.
} DataMigration;

// Initializes a DataMigration struct.
static PetscErrorCode
DataMigrationCreate(DataMigration *mig, MPI_Comm comm, PetscInt size,
                    GridPartitioning *pnew, PetscInt dof)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &mig->ds); CHKERRQ(ierr);
  ierr = DMSetType(mig->ds, DMSWARM); CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(mig->ds); CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(mig->ds, "index", 3, PETSC_INT);
    CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(mig->ds, "field", dof, PETSC_REAL);
    CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(mig->ds); CHKERRQ(ierr);

  mig->size = size;
  ierr = DMSwarmSetLocalSizes(mig->ds, mig->size, 1); CHKERRQ(ierr);

  ierr = DMSwarmGetField(mig->ds, "index", NULL, NULL,
                         (void**) &mig->index); CHKERRQ(ierr);
  ierr = DMSwarmGetField(mig->ds, "field", NULL, NULL,
                         (void**) &mig->field); CHKERRQ(ierr);
  ierr = DMSwarmGetField(mig->ds, DMSwarmField_rank, NULL,
                         NULL, (void**) &mig->ranks); CHKERRQ(ierr);

  mig->pnew = pnew;
  mig->dof = dof;

  PetscFunctionReturn(0);
}

static PetscErrorCode DataMigrationDestroy(DataMigration *mig)
{
  return DMDestroy(&mig->ds);
}

// Insert a "mig->dof" PetscReals starting at "data".
// These data are associated to grid cell (i, j, k).
// and are inserted into the DM swarm field
// at position insert_index.
static PetscErrorCode
DataMigrationInsert(DataMigration *mig,
                    PetscInt i, PetscInt j, PetscInt k, PetscReal *data,
                    PetscInt insert_index)
{
  PetscErrorCode ierr;
  PetscInt rank;

  PetscFunctionBegin;
  rank = GridPartitioningGetOwnerRank(mig->pnew, i, j, k);

  mig->ranks[insert_index] = rank;

  // Insert payload (i, j, k, data) at "insert_index"
  ierr = PetscMemcpy(&mig->field[mig->dof * insert_index], data,
                     mig->dof * sizeof(PetscReal)); CHKERRQ(ierr);

  mig->index[3 * insert_index + 0] = i;
  mig->index[3 * insert_index + 1] = j;
  mig->index[3 * insert_index + 2] = k;

  PetscFunctionReturn(0);
}

// Extract the extract_index'th payload.
// "data" will be a borrowed reference from "mig"
// holding "mig->dof" number of PetscReals.
// Do not free it.
static PetscErrorCode
DataMigrationExtract(DataMigration *mig, PetscInt extract_index,
                     PetscInt *i, PetscInt *j, PetscInt *k,
                     PetscReal **data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  if (extract_index > mig->size) {
    MPI_Comm comm;
    ierr = PetscObjectGetComm((PetscObject) mig->ds, &comm); CHKERRQ(ierr);
    SETERRQ2(comm, PETSC_ERR_ARG_OUTOFRANGE,
             "Requested payload %D but have only %D elements",
             extract_index, mig->size);
  }

  *i = mig->index[3 * extract_index + 0];
  *j = mig->index[3 * extract_index + 1];
  *k = mig->index[3 * extract_index + 2];

  *data = &mig->field[mig->dof * extract_index];

  PetscFunctionReturn(0);
}

static PetscInt DataMigrationGetSize(DataMigration *mig)
{
  return mig->size;
}

// Migrates all data to the correct owner.
static PetscErrorCode DataMigrationDoMigrate(DataMigration *mig)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  // Store the fields, migrate and get the field again for extraction.
  ierr = DMSwarmRestoreField(mig->ds, "index", NULL, NULL,
                             (void**) &mig->index); CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(mig->ds, "field", NULL, NULL,
                             (void**) &mig->field); CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(mig->ds, DMSwarmField_rank, NULL, NULL,
                             (void**) &mig->ranks); CHKERRQ(ierr);
  ierr = DMSwarmMigrate(mig->ds, PETSC_TRUE); CHKERRQ(ierr);
  ierr = DMSwarmGetField(mig->ds, "index", NULL, NULL,
                         (void**) &mig->index); CHKERRQ(ierr);
  ierr = DMSwarmGetField(mig->ds, "field", NULL, NULL,
                         (void**) &mig->field); CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(mig->ds, &mig->size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/***********/
/* Utility */
/***********/

// Copies all field names from src to dst, the application context
// and the IFunctional
static PetscErrorCode CopyDMInfo(DM dst, DM src)
{
  PetscErrorCode ierr;
  const char *name;
  PetscInt dof, i;
  void *ctx;
  TSIFunction f;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(src, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     &dof, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (i = 0; i < dof; ++i) {
    ierr = DMDAGetFieldName(src, i, &name); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dst, i, name); CHKERRQ(ierr);
  }

  ierr = DMGetApplicationContext(src, &ctx); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dst, ctx); CHKERRQ(ierr);

  ierr = DMTSGetIFunction(src, &f, &ctx); CHKERRQ(ierr);
  ierr = DMTSSetIFunction(dst, f, ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

typedef struct {
  PetscInt dim;     // dimension of the distributed array (1, 2, or 3)
  PetscInt M, N, P; // global dimension in each direction of the array
  PetscInt m, n, p; // corresponding number of procs in each dimension
  PetscInt dof;     // number of degrees of freedom per node 
  PetscInt s;       // stencil width
  DMBoundaryType bx, by, bz; // type of ghost nodes at boundary
  DMDAStencilType st; // stencil type
} DMDAInfo;

static PetscErrorCode DMDAGetInfoStruct(DM da, DMDAInfo *info)
{
  return DMDAGetInfo(da, &info->dim, &info->M, &info->N, &info->P,
                     &info->m, &info->n, &info->p, &info->dof, &info->s,
                     &info->bx, &info->by, &info->bz, &info->st);
}

/********************/
/* Public interface */
/********************/

// Migrates all data from Vec X to Xn.
// Where X is a global vector from da and Xn from rda.
static PetscErrorCode
DMDA_repart_migrate_data(DM da, DM rda, MPI_Comm comm, Vec X, Vec Xn,
                         GridPartitioning *pnew)
{
  PetscErrorCode ierr;
  DataMigration mig;
  PetscInt dof, size, ei, i, j, k, xs, ys, zs, xm, ym, zm;
  PetscReal *data;
  PetscReal ****x;
  PetscInt num;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &dof,
                     NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
 
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);
  ierr = DataMigrationCreate(&mig, comm, xm * ym * zm, pnew, dof); CHKERRQ(ierr);

  // Feed all local data into DataMigration
  ierr = DMDAVecGetArrayDOF(da, X, &x);CHKERRQ(ierr);
  num = 0;
  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        ierr = DataMigrationInsert(&mig, i, j, k, x[k][j][i], num++);
          CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);

  // Migrate the data to their respective new owners
  ierr = DataMigrationDoMigrate(&mig); CHKERRQ(ierr);

  // Extract data and fill "Xn"
  ierr = DMDAVecGetArrayDOF(rda, Xn, &x); CHKERRQ(ierr);
  size = DataMigrationGetSize(&mig);

  for (ei = 0; ei < size; ++ei) {
    ierr = DataMigrationExtract(&mig, ei, &i, &j, &k, &data);
      CHKERRQ(ierr);
    ierr = PetscMemcpy(x[k][j][i], data, dof * sizeof(PetscReal));
  }
  ierr = DMDAVecRestoreArrayDOF(rda, Xn, &x); CHKERRQ(ierr);

  ierr = DataMigrationDestroy(&mig); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode
DMDA_repart(DM* da, Vec *X, PetscInt lx[], PetscInt ly[], PetscInt lz[],
            PetscBool setFromOptions)
{
  PetscErrorCode ierr;
  DMDAInfo info;
  DM rda;
  GridPartitioning pnew;
  Vec Xn;
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) *da, &comm); CHKERRQ(ierr);

  // Get infos from da
  ierr = DMDAGetInfoStruct(*da, &info); CHKERRQ(ierr);

  if (info.dim != 3) {
    SETERRQ(comm, PETSC_ERR_SUP, "DMDA_repart only implemented for 3d DMDAs.");
  }

  ierr = GridPartitioningCreateFromLs(&pnew, info.m, info.n, info.p,
                                      lx, ly, lz); CHKERRQ(ierr);

  // Create new da
  ierr = DMDACreate3d(comm, info.bx, info.by, info.bz, info.st,
                     info.M, info.N, info.P, info.m, info.n, info.p,
                     info.dof, info.s, lx, ly, lz,
                     &rda); CHKERRQ(ierr);
  if (setFromOptions) {
    ierr = DMSetFromOptions(rda); CHKERRQ(ierr);
  }
  ierr = DMSetUp(rda); CHKERRQ(ierr);
  ierr = CopyDMInfo(rda, *da); CHKERRQ(ierr);

  // Migrate data
  ierr = DMCreateGlobalVector(rda, &Xn); CHKERRQ(ierr);
  ierr = DMDA_repart_migrate_data(*da, rda, comm, *X, Xn, &pnew); CHKERRQ(ierr);

  // Reset "da" and "X"
  ierr = DMDestroy(da); CHKERRQ(ierr);
  ierr = VecDestroy(X); CHKERRQ(ierr);
  *da = rda;
  *X = Xn;

  ierr = GridPartitioningDestroy(&pnew); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


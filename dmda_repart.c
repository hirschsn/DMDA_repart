
#include "dmda_repart.h"

#include <petscdmswarm.h>


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
  GridPartitioningCreate(part, m, n, p);
  ierr = PetscMemcpy(part->lx, lx, part->m * sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscMemcpy(part->ly, ly, part->n * sizeof(PetscInt)); CHKERRQ(ierr);
  ierr = PetscMemcpy(part->lz, lz, part->p * sizeof(PetscInt)); CHKERRQ(ierr);

  GridPartitioningBuildPrefixes(part);
  PetscFunctionReturn(0);
}

// Returns a owner rank of a cell (x, y, z) under the partitioning "p".
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

// Payload storing the data that is migrated.
typedef struct {
  PetscInt i, j, k;
  // Flexible array member
  char data[];
} Payload;

// Data migration state storage
typedef struct {
  DM ds;         // Swarm DM used for migration
  PetscInt size; // Number of process local data points

  // Use void* here, as Payload* leads to wrong
  // pointer arithmetic because of the flexible
  // array member
  void *payload;       // Data from or for this process
                       // (depends if DoMigrate has been called)
  PetscInt payload_bs;

  PetscInt *ranks;     // Target ranks (new owners)
  PetscInt rank_bs;

  GridPartitioning *pnew; // GridPartitioning struct that defines the
                          // new partitioning.
} DataMigration;

// Initializes a DataMigration struct.
static PetscErrorCode
DataMigrationCreate(DataMigration *mig, MPI_Comm comm, PetscInt size,
                    GridPartitioning *pnew, size_t payload_struct_size)
{
  PetscErrorCode ierr;
  ierr = DMCreate(comm, &mig->ds); CHKERRQ(ierr);
  ierr = DMSetType(mig->ds, DMSWARM); CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(mig->ds); CHKERRQ(ierr);
  ierr = DMSwarmRegisterUserStructField(mig->ds, "field", payload_struct_size);
    CHKERRQ(ierr);
  ierr = DMSwarmFinalizeFieldRegister(mig->ds); CHKERRQ(ierr);

  mig->size = size;
  ierr = DMSwarmSetLocalSizes(mig->ds, mig->size, 1); CHKERRQ(ierr);

  ierr = DMSwarmGetField(mig->ds, "field", &mig->payload_bs, NULL,
                         (void**) &mig->payload); CHKERRQ(ierr);
  ierr = DMSwarmGetField(mig->ds, DMSwarmField_rank, &mig->rank_bs,
                         NULL, (void**) &mig->ranks); CHKERRQ(ierr);

  mig->pnew = pnew;

  PetscFunctionReturn(0);
}

static PetscErrorCode DataMigrationDestroy(DataMigration *mig)
{
  return DMDestroy(&mig->ds);
}

// Insert a data point at grid cell (i, j, k).
// Data is of size payload_data_size and the associated payload
// struct of size payload_struct_size.
// The payload element is inserted into the DM swarm field
// at position insert_index.
static PetscErrorCode
DataMigrationInsert(DataMigration *mig,
                    PetscInt i, PetscInt j, PetscInt k, void *data,
                    size_t payload_struct_size, size_t payload_data_size,
                    PetscInt insert_index)
{
  PetscErrorCode ierr;
  PetscInt rank;

  rank = GridPartitioningGetOwnerRank(mig->pnew, i, j, k);

  mig->ranks[insert_index] = rank;

  // Insert payload (i, j, k, data) at "insert_index"
  void *addr = mig->payload + insert_index * payload_struct_size;
  *(PetscInt *) (addr + 0 * sizeof(PetscInt)) = i;
  *(PetscInt *) (addr + 1 * sizeof(PetscInt)) = j;
  *(PetscInt *) (addr + 2 * sizeof(PetscInt)) = k;

  addr += payload_struct_size - payload_data_size;
  ierr = PetscMemcpy(addr, data, payload_data_size); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

// Extract the i'th payload data struct.
static PetscErrorCode
DataMigrationExtract(DataMigration *mig, PetscInt i, Payload *p,
                     size_t payload_struct_size)
{
  PetscErrorCode ierr;
  if (i > mig->size) {
    PetscFPrintf(PETSC_COMM_WORLD, stderr,
                 "DataMigration Error: Requesting payload %i but have only %i\n",
                 i, mig->size);
    MPI_Abort(PETSC_COMM_WORLD, 1);
  }

  void *addr = mig->payload + i * payload_struct_size;
  ierr = PetscMemcpy(p, addr, payload_struct_size); CHKERRQ(ierr);

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
  // Store the fields, migrate and get the field again for extraction.
  ierr = DMSwarmRestoreField(mig->ds, "field", &mig->payload_bs, NULL,
                             (void**) &mig->payload); CHKERRQ(ierr);
  ierr = DMSwarmRestoreField(mig->ds, DMSwarmField_rank, &mig->rank_bs, NULL,
                             (void**) &mig->ranks); CHKERRQ(ierr);
  ierr = DMSwarmMigrate(mig->ds, PETSC_TRUE); CHKERRQ(ierr);
  ierr = DMSwarmGetField(mig->ds, "field", &mig->payload_bs, NULL,
                         (void**) &mig->payload); CHKERRQ(ierr);
  ierr = DMSwarmGetLocalSize(mig->ds, &mig->size); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


/***********/
/* Utility */
/***********/

// Copies all field names from src to dst.
static PetscErrorCode CopyFieldNames(DM dst, DM src)
{
  PetscErrorCode ierr;
  const char *name;
  PetscInt dof, i;

  ierr = DMDAGetInfo(src, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     &dof, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (i = 0; i < dof; ++i) {
    ierr = DMDAGetFieldName(src, i, &name); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dst, i, name); CHKERRQ(ierr);
  }

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
static PetscErrorCode DMDA_repart_migrate_data(DM da, DM rda, MPI_Comm comm, Vec X, Vec Xn, GridPartitioning *pnew, size_t payload_struct_size, size_t payload_data_size)
{
  PetscErrorCode ierr;
  DataMigration mig;
  PetscInt size, i, j, k, xs, ys, zs, xm, ym, zm;
  Payload *payload;
  PetscReal ***x;
  PetscInt num;

  // Memory for payload
  PetscMalloc(payload_struct_size, &payload);

  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);

  ierr = DataMigrationCreate(&mig, comm, xm * ym * zm, pnew, payload_struct_size); CHKERRQ(ierr);

  // Feed all local data into DataMigration
  ierr = DMDAVecGetArray(da, X, &x);CHKERRQ(ierr);
  num = 0;
  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        DataMigrationInsert(&mig, i, j, k, (void *) &x[k][j][i], payload_struct_size, payload_data_size, num++);
      }
    }
  }

  // Migrate the data to their respective new owners
  ierr = DataMigrationDoMigrate(&mig); CHKERRQ(ierr);

  // Change the DMDA of "X"
  ierr = VecSetDM(X, rda); CHKERRQ(ierr);

  // Extract data and fill "X"
  ierr = DMDAVecGetArray(rda, Xn, &x);CHKERRQ(ierr);
  size = DataMigrationGetSize(&mig);
  for (i = 0; i < size; ++i) {
    DataMigrationExtract(&mig, i, payload, payload_struct_size);
    ierr = PetscMemcpy(&x[payload->k][payload->j][payload->i], payload->data, payload_data_size);
  }
  ierr = DMDAVecRestoreArray(rda, Xn, &x);

  ierr = DataMigrationDestroy(&mig); CHKERRQ(ierr);
  ierr = PetscFree(payload); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}


PetscErrorCode DMDA_repart(DM* da, Vec *X, PetscInt lx[], PetscInt ly[], PetscInt lz[])
{
  PetscErrorCode ierr;
  DMDAInfo info;
  DM rda;
  GridPartitioning pnew;
  Vec Xn;
  MPI_Comm comm;
  size_t payload_data_size, payload_struct_size;

  ierr = PetscObjectGetComm((PetscObject) *da, &comm); CHKERRQ(ierr);

  // Get infos from da
  ierr = DMDAGetInfoStruct(*da, &info); CHKERRQ(ierr);
  payload_data_size = info.dof * sizeof(PetscReal);
  payload_struct_size = sizeof(Payload) + payload_data_size;

  if (info.dim != 3) {
    PetscFPrintf(comm, stderr, "DMDA_repart only implemented for 3d DMDAs.\n");
    PetscFunctionReturn(1);
  }

  ierr = GridPartitioningCreateFromLs(&pnew, info.m, info.n, info.p, lx, ly, lz);

  // Create new da
  ierr = DMDACreate3d(comm, info.bx, info.by, info.bz, info.st,
                     info.M, info.N, info.P, info.m, info.n, info.p,
                     info.dof, info.s, lx, ly, lz,
                     &rda); CHKERRQ(ierr);
  ierr = DMSetUp(rda); CHKERRQ(ierr);
  ierr = CopyFieldNames(rda, *da);

  // Migrate data
  ierr = DMCreateGlobalVector(rda, &Xn); CHKERRQ(ierr);
  ierr = DMDA_repart_migrate_data(*da, rda, comm, *X, Xn, &pnew, payload_struct_size, payload_data_size); CHKERRQ(ierr);

  // Reset "da" and "X"
  ierr = DMDestroy(da); CHKERRQ(ierr);
  ierr = VecDestroy(X); CHKERRQ(ierr);
  *da = rda;
  *X = Xn;

  ierr = GridPartitioningDestroy(&pnew); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}


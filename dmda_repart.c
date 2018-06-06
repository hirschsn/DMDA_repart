
#include "dmda_repart.h"

#include <petscdmswarm.h>
#include <petscts.h>
#include <stdarg.h>


/*******************************************/
/* Grid partitioning storage and functions */
/*******************************************/

typedef struct {
  PetscInt dim;       // Number of dimensions
  PetscInt N[3];      // Number of processes in x, y, z direction (Petsc: m, n, p)
  PetscInt *ls[3];    // Ownership range in direction x, y, z (non-owning pointers)
  PetscInt *prefs[3]; // Prefixes of ownership range in x, y, z direction
} GridPartitioning;

// Initializes a GridPartitioning struct and allocates data.
// m, n, p are the processes in x, y, z direction, respectively.
// Supply any value ("0") for p if dim < 3. Also for n if dim < 2.
static PetscErrorCode
GridPartitioningCreate(GridPartitioning *part, PetscInt dim,
                       PetscInt m, PetscInt n, PetscInt p)
{
  part->dim = dim;
  part->N[0] = m; part->N[1] = n; part->N[2] = p;
  return PetscMalloc3(m + 1, &part->prefs[0],
                      n + 1, &part->prefs[1],
                      p + 1, &part->prefs[2]);
}

static PetscErrorCode GridPartitioningDestroy(GridPartitioning *p)
{
  return PetscFree3(p->prefs[0], p->prefs[1], p->prefs[2]);
}

// Prepares a GridPartitioning struct for rank search
static void GridPartitioningBuildPrefixes(GridPartitioning *p)
{
  PetscInt i, d;

  for (d = 0; d < p->dim; ++d) {
    p->prefs[d][0] = 0;
    for (i = 1; i <= p->N[d]; ++i)
      p->prefs[d][i] = p->prefs[d][i - 1] + p->ls[d][i - 1];
  }
}

// Initializes a GridPartitioning struct from nodes pre grid dimension array.
// They have the same meaning as ownership ranges in Petsc.
// Supply any value ("NULL") for lz or ly if dim < 3 or < 2.
static PetscErrorCode
GridPartitioningCreateFromLs(GridPartitioning *part, PetscInt dim,
                             PetscInt m, PetscInt n, PetscInt p,
                             PetscInt lx[], PetscInt ly[], PetscInt lz[])
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = GridPartitioningCreate(part, dim, m, n, p); CHKERRQ(ierr);
  part->ls[0] = lx;
  part->ls[1] = ly;
  part->ls[2] = lz;

  GridPartitioningBuildPrefixes(part);
  PetscFunctionReturn(0);
}

// Returns the owner rank of a cell with global index "coord" (x, y, z)
// under the partitioning "p".
static PetscInt
GridPartitioningGetOwnerRank(GridPartitioning *p, PetscInt coord[3])
{
  PetscInt i, d, n[] = {-1, -1, -1}; // Process coord

  for (d = 0; d < p->dim; ++d) {
    // TODO: binary search
    for (i = 0; i <= p->N[d]; ++i) {
      if (p->prefs[d][i] > coord[d]) {
        n[d] = i - 1;
        break;
      }
    }

    if (n[d] == -1) {
      PetscFPrintf(PETSC_COMM_WORLD, stderr,
                   "Error: Could not find owner rank of cell.\n");
      MPI_Abort(PETSC_COMM_WORLD, 1);
    }
  }

  // Ordering from Telescope:
  // https://www.mcs.anl.gov/petsc/petsc-current/src/ksp/pc/impls/telescope/telescope_dmda.c.html
  if (p->dim == 1)
    return n[0];
  else if (p->dim == 2)
    return n[0] + n[1] * p->N[0];
  else
    return n[0] + n[1] * p->N[0] + n[2] * p->N[0] * p->N[1];
}


/******************/
/* Data migration */
/******************/

// Data migration state storage
typedef struct {
  DM ds;         // Swarm DM used for migration
  PetscInt nvec; // Number of vectors to migrate
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
                    GridPartitioning *pnew, PetscInt dof, PetscInt nvec)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = DMCreate(comm, &mig->ds); CHKERRQ(ierr);
  ierr = DMSetType(mig->ds, DMSWARM); CHKERRQ(ierr);

  ierr = DMSwarmInitializeFieldRegister(mig->ds); CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(mig->ds, "index", 3, PETSC_INT);
    CHKERRQ(ierr);
  ierr = DMSwarmRegisterPetscDatatypeField(mig->ds, "field", dof * nvec, PETSC_REAL);
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
  mig->nvec = nvec;

  PetscFunctionReturn(0);
}

static PetscErrorCode DataMigrationDestroy(DataMigration *mig)
{
  return DMDestroy(&mig->ds);
}


// Associates "insert_index" with the grid cell at coordinate
// "coord".
static PetscErrorCode 
DataMigrationInsertIndex(DataMigration *mig, PetscInt coord[3],
                         PetscInt insert_index)
{
  PetscInt d;

  PetscFunctionBegin;
  mig->ranks[insert_index] = GridPartitioningGetOwnerRank(mig->pnew, coord);
  // Insert coordinate -- 3d regardless of dimensionality.
  for (d = 0; d < 3; ++d)
    mig->index[3 * insert_index + d] = coord[d];
  PetscFunctionReturn(0);
}

// Insert a "mig->dof" PetscReals starting "data"
// These data are associated to the "insert_index"th grid cell.
static PetscErrorCode
DataMigrationInsertData(DataMigration *mig, PetscInt insert_index, PetscInt vno,
                        PetscReal *data)
{
  PetscErrorCode ierr;

  PetscFunctionBegin;
  ierr = PetscMemcpy(&mig->field[mig->dof * (mig->nvec * insert_index + vno)], data,
                     mig->dof * sizeof(PetscReal)); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Extract the extract_index'th payload.
// "data" will be a borrowed reference from "mig"
// holding "mig->dof" number of PetscReals.
// Do not free it.
// Supply NULL as "k" or "j" if dimensionality < 3 or < 2.
// If "k" or "j" are valid pointers in these cases, their contents will
// be set to 0.
static PetscErrorCode
DataMigrationExtract(DataMigration *mig, PetscInt extract_index,
                     PetscInt vno,
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
  if (j)
    *j = mig->index[3 * extract_index + 1];
  if (k)
    *k = mig->index[3 * extract_index + 2];

  *data = &mig->field[mig->dof * (mig->nvec * extract_index + vno)];

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

static PetscErrorCode
DataMigrationInsertVecIndex3D(DataMigration *mig, DM da)
{
  PetscErrorCode ierr;
  PetscInt i, j, k, xs, ys, zs, xm, ym, zm, num;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);

  num = 0;
  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
          ierr = DataMigrationInsertIndex(mig, (PetscInt[]) {i, j, k},
                                          num++); CHKERRQ(ierr);
      }
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationInsertVecData3D(DataMigration *mig, DM da, PetscInt vno, Vec X)
{
  PetscErrorCode ierr;
  PetscInt i, j, k, xs, ys, zs, xm, ym, zm, num;
  PetscReal ****x;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, &zs, &xm, &ym, &zm); CHKERRQ(ierr);

  // Feed all local data into DataMigration
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);
  num = 0;
  for (k = zs; k < zs+zm; k++) {
    for (j = ys; j < ys+ym; j++) {
      for (i = xs; i < xs+xm; i++) {
        ierr = DataMigrationInsertData(mig, num++, vno, x[k][j][i]);
          CHKERRQ(ierr);
      }
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);
  
  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationInsertVecIndex2D(DataMigration *mig, DM da)
{
  PetscErrorCode ierr;
  PetscInt i, j, xs, ys, xm, ym, num;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

  num = 0;
  for (j = ys; j < ys+ym; j++) {
    for (i = xs; i < xs+xm; i++) {
      ierr = DataMigrationInsertIndex(mig, (PetscInt[]) {i, j, 0}, num++);
        CHKERRQ(ierr);
    }
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationInsertVecData2D(DataMigration *mig, DM da, PetscInt vno, Vec X)
{
  PetscErrorCode ierr;
  PetscInt i, j, xs, ys, xm, ym, num;
  PetscReal ***x;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, &ys, NULL, &xm, &ym, NULL); CHKERRQ(ierr);

  // Feed all local data into DataMigration
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);
  num = 0;
  for (j = ys; j < ys+ym; j++) {
    for (i = xs; i < xs+xm; i++) {
      ierr = DataMigrationInsertData(mig, num++, vno, x[j][i]);
        CHKERRQ(ierr);
    }
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationInsertVecIndex1D(DataMigration *mig, DM da)
{
  PetscErrorCode ierr;
  PetscInt i, xs, xm, num;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);
  num = 0;
  for (i = xs; i < xs+xm; i++) {
    ierr = DataMigrationInsertIndex(mig, (PetscInt[]) {i, 0, 0}, num++);
      CHKERRQ(ierr);
  }
  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationInsertVecData1D(DataMigration *mig, DM da, PetscInt vno, Vec X)
{
  PetscErrorCode ierr;
  PetscInt i, xs, xm, num;
  PetscReal **x;

  PetscFunctionBegin;
  ierr = DMDAGetCorners(da, &xs, NULL, NULL, &xm, NULL, NULL); CHKERRQ(ierr);

  // Feed all local data into DataMigration
  ierr = DMDAVecGetArrayDOF(da, X, &x); CHKERRQ(ierr);
  num = 0;
  for (i = xs; i < xs+xm; i++) {
    ierr = DataMigrationInsertData(mig, num++, vno, x[i]);
      CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayDOF(da, X, &x); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationInsertVecIndex(DataMigration *mig, DM da)
{
  static PetscErrorCode (*insert_fn[])(DataMigration *, DM) = {
    [1] = DataMigrationInsertVecIndex1D,
    [2] = DataMigrationInsertVecIndex2D,
    [3] = DataMigrationInsertVecIndex3D,
  };

  return insert_fn[mig->pnew->dim](mig, da);
}

static PetscErrorCode
DataMigrationInsertVecData(DataMigration *mig, DM da, PetscInt vno, Vec X)
{
  static PetscErrorCode (*insert_fn[])(DataMigration *, DM, PetscInt, Vec) = {
    [1] = DataMigrationInsertVecData1D,
    [2] = DataMigrationInsertVecData2D,
    [3] = DataMigrationInsertVecData3D,
  };

  return insert_fn[mig->pnew->dim](mig, da, vno, X);
}

static PetscErrorCode
DataMigrationExtractVec3D(DataMigration *mig, DM rda, PetscInt vno, Vec Xn)
{
  PetscErrorCode ierr;
  PetscInt i, j, k, ei, size;
  PetscReal ****x;
  PetscReal *data;

  PetscFunctionBegin;
  ierr = DMDAVecGetArrayDOF(rda, Xn, &x); CHKERRQ(ierr);
  size = DataMigrationGetSize(mig);

  for (ei = 0; ei < size; ++ei) {
    ierr = DataMigrationExtract(mig, ei, vno, &i, &j, &k, &data);
      CHKERRQ(ierr);
    ierr = PetscMemcpy(x[k][j][i], data, mig->dof * sizeof(PetscReal));
      CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayDOF(rda, Xn, &x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationExtractVec2D(DataMigration *mig, DM rda, PetscInt vno, Vec Xn)
{
  PetscErrorCode ierr;
  PetscInt i, j, ei, size;
  PetscReal ***x;
  PetscReal *data;

  PetscFunctionBegin;
  ierr = DMDAVecGetArrayDOF(rda, Xn, &x); CHKERRQ(ierr);
  size = DataMigrationGetSize(mig);

  for (ei = 0; ei < size; ++ei) {
    ierr = DataMigrationExtract(mig, ei, vno, &i, &j, NULL, &data);
      CHKERRQ(ierr);
    ierr = PetscMemcpy(x[j][i], data, mig->dof * sizeof(PetscReal));
      CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayDOF(rda, Xn, &x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationExtractVec1D(DataMigration *mig, DM rda, PetscInt vno, Vec Xn)
{
  PetscErrorCode ierr;
  PetscInt i, ei, size;
  PetscReal **x;
  PetscReal *data;

  PetscFunctionBegin;
  ierr = DMDAVecGetArrayDOF(rda, Xn, &x); CHKERRQ(ierr);
  size = DataMigrationGetSize(mig);

  for (ei = 0; ei < size; ++ei) {
    ierr = DataMigrationExtract(mig, ei, vno, &i, NULL, NULL, &data);
      CHKERRQ(ierr);
    ierr = PetscMemcpy(x[i], data, mig->dof * sizeof(PetscReal));
      CHKERRQ(ierr);
  }
  ierr = DMDAVecRestoreArrayDOF(rda, Xn, &x); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode
DataMigrationExtractVec(DataMigration *mig, DM rda, PetscInt vno, Vec Xn)
{
  static PetscErrorCode (*extract_fn[])(DataMigration *, DM, PetscInt, Vec) = {
    [1] = DataMigrationExtractVec1D,
    [2] = DataMigrationExtractVec2D,
    [3] = DataMigrationExtractVec3D,
  };

  return extract_fn[mig->pnew->dim](mig, rda, vno, Xn);
}


/***********/
/* Utility */
/***********/

// Copies all field names from src to dst and the application context
static PetscErrorCode CopyDMInfo(DM dst, DM src)
{
  PetscErrorCode ierr;
  const char *name;
  PetscInt dof, i;
  void *ctx;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(src, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     &dof, NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);

  for (i = 0; i < dof; ++i) {
    ierr = DMDAGetFieldName(src, i, &name); CHKERRQ(ierr);
    ierr = DMDASetFieldName(dst, i, name); CHKERRQ(ierr);
  }

  ierr = DMGetApplicationContext(src, &ctx); CHKERRQ(ierr);
  ierr = DMSetApplicationContext(dst, ctx); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

static PetscErrorCode
get_local_data_size(DM da, PetscInt *size)
{
  PetscErrorCode ierr;
  PetscInt xm, ym, zm, dim;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, &dim, NULL, NULL, NULL, NULL, NULL, NULL, NULL,
                     NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = DMDAGetCorners(da, NULL, NULL, NULL, &xm, &ym, &zm); CHKERRQ(ierr);

  switch (dim) {
  case 1:
    *size = xm;
    break;
  case 2:
    *size = xm * ym;
    break;
  case 3:
    *size = xm * ym * zm;
    break;
  }

  PetscFunctionReturn(0);
}

/********************/
/* Public interface */
/********************/

// Migrates all data from every Vec in X to the corresponding in Xn.
// Both their lengths is nvec.
// Where X is an array of global vectors from da and Xn from rda.
static PetscErrorCode
DMDA_repart_migrate_data(DM da, DM rda, MPI_Comm comm, PetscInt nvec,
                         Vec *X[], Vec Xn[], GridPartitioning *pnew)
{
  PetscErrorCode ierr;
  DataMigration mig;
  PetscInt dof, lsize, vno;

  PetscFunctionBegin;
  ierr = DMDAGetInfo(da, NULL, NULL, NULL, NULL, NULL, NULL, NULL, &dof,
                     NULL, NULL, NULL, NULL, NULL); CHKERRQ(ierr);
  ierr = get_local_data_size(da, &lsize);
  ierr = DataMigrationCreate(&mig, comm, lsize, pnew, dof, nvec);
    CHKERRQ(ierr);
  ierr = DataMigrationInsertVecIndex(&mig, da); CHKERRQ(ierr);
  for (vno = 0; vno < nvec; ++vno) {
    ierr = DataMigrationInsertVecData(&mig, da, vno, *X[vno]); CHKERRQ(ierr);
  }

  // Migrate the data to their respective new owners
  ierr = DataMigrationDoMigrate(&mig); CHKERRQ(ierr);

  // Extract data and fill "Xn"
  for (vno = 0; vno < nvec; ++vno) {
    ierr = DataMigrationExtractVec(&mig, rda, vno, Xn[vno]); CHKERRQ(ierr);
  }

  ierr = DataMigrationDestroy(&mig); CHKERRQ(ierr);
  PetscFunctionReturn(0);
}

// Actually, Vec X[] would be sufficient to be able to modify the component
// Vecs of the array. However, the interface of DMDA_repartl cannot satisfy
// this as it has multiple pointers each to one Vec only. Therefore the
// additional indirection.
static PetscErrorCode
DMDA_repartn(DM* da, PetscInt lx[], PetscInt ly[], PetscInt lz[],
             PetscBool setFromOptions, PetscInt nvec, Vec *X[])
{
  PetscErrorCode ierr;
  PetscInt dim, M, N, P, m, n, p, dof, s, vno;
  DMDAStencilType st;
  DMBoundaryType bx, by, bz;
  DM rda;
  GridPartitioning pnew;
  Vec Xn[nvec];
  MPI_Comm comm;

  PetscFunctionBegin;
  ierr = PetscObjectGetComm((PetscObject) *da, &comm); CHKERRQ(ierr);

  // Get infos from da
  ierr = DMDAGetInfo(*da, &dim, &M, &N, &P, &m, &n, &p, &dof, &s,
                     &bx, &by, &bz, &st); CHKERRQ(ierr);

  ierr = GridPartitioningCreateFromLs(&pnew, dim, m, n, p, lx, ly, lz);
    CHKERRQ(ierr);

  // Create new da
  switch (dim) {
  case 3:
    ierr = DMDACreate3d(comm, bx, by, bz, st, M, N, P, m, n, p, dof, s,
                        lx, ly, lz, &rda); CHKERRQ(ierr);
    break;
  case 2:
    ierr = DMDACreate2d(comm, bx, by, st, M, N, m, n, dof, s, lx, ly, &rda);
      CHKERRQ(ierr);
    break;
  case 1:
    ierr = DMDACreate1d(comm, bx, M, dof, s, lx, &rda);
      CHKERRQ(ierr);
    break;
  }

  if (setFromOptions) {
    ierr = DMSetFromOptions(rda); CHKERRQ(ierr);
  }
  ierr = DMSetUp(rda); CHKERRQ(ierr);
  ierr = CopyDMInfo(rda, *da); CHKERRQ(ierr);

  // Migrate data
  for (vno = 0; vno < nvec; vno++) {
    ierr = DMCreateGlobalVector(rda, &Xn[vno]); CHKERRQ(ierr);
  }
  ierr = DMDA_repart_migrate_data(*da, rda, comm, nvec, X, Xn, &pnew); CHKERRQ(ierr);

  // Reset "da" and "X"
  ierr = DMDestroy(da); CHKERRQ(ierr);
  for (vno = 0; vno < nvec; vno++) {
    ierr = VecDestroy(X[vno]); CHKERRQ(ierr);
    *X[vno] = Xn[vno];
  }
  *da = rda;

  ierr = GridPartitioningDestroy(&pnew); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode
DMDA_repartv(DM* da, PetscInt lx[], PetscInt ly[], PetscInt lz[],
             PetscBool setFromOptions, PetscInt nvec, Vec X[])
{
  PetscErrorCode ierr;
  Vec *XX[nvec];
  PetscInt i;

  PetscFunctionBegin;
  // Copy the data into array of pointers to vec
  for (i = 0; i < nvec; ++i)
    XX[i] = &X[i];

  ierr = DMDA_repartn(da, lx, ly, lz, setFromOptions, nvec, XX);
    CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode
DMDA_repartl(DM* da,
             PetscInt lx[], PetscInt ly[], PetscInt lz[],
             PetscBool setFromOptions, Vec *X, ...)
{
  PetscErrorCode ierr;
  va_list ap1, ap2;
  size_t nelem, i;
  Vec **arr;

  PetscFunctionBegin;
  // Copy the elements of the va_list into an array
  va_start(ap1, X);
  va_copy(ap2, ap1);
  for (nelem = 1; va_arg(ap1, Vec*); nelem++);
  va_end(ap1);

  ierr = PetscMalloc1(nelem, &arr); CHKERRQ(ierr);
  arr[0] = X;
  for (i = 1; i < nelem; ++i)
    arr[i] = va_arg(ap2, Vec*);
  va_end(ap2);

  ierr = DMDA_repartn(da, lx, ly, lz, setFromOptions, nelem, arr); CHKERRQ(ierr);
  ierr = PetscFree(arr); CHKERRQ(ierr);

  PetscFunctionReturn(0);
}

PetscErrorCode
DMDA_repart(DM* da, Vec *X,
            PetscInt lx[], PetscInt ly[], PetscInt lz[],
            PetscBool setFromOptions)
{
  return DMDA_repartn(da, lx, ly, lz, setFromOptions, 1, &X);
}


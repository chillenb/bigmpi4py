from mpi4py import MPI

from .utils import BIGMPI_MAXSIZE

Datatype = MPI.Datatype
COMBINER_STRUCT = MPI.COMBINER_STRUCT
COMBINER_VECTOR = MPI.COMBINER_VECTOR


def BigMPI_Type_contiguous(oldtype, count, offset=0):
    c = count / BIGMPI_MAXSIZE
    r = count % BIGMPI_MAXSIZE

    if c >= BIGMPI_MAXSIZE or r >= BIGMPI_MAXSIZE:
        msg = f"count={count} is too large for BigMPI"
        raise ValueError(msg)

    chunks = oldtype.Create_vector(c, BIGMPI_MAXSIZE, BIGMPI_MAXSIZE)
    remainder = oldtype.Create_contiguous(r)
    _, extent = oldtype.Get_extent()

    remdisp = c * BIGMPI_MAXSIZE * extent
    newtype = Datatype.Create_struct(
        (1, 1), (offset, offset + remdisp), (chunks, remainder)
    )
    chunks.Free()
    remainder.Free()
    return newtype


def BigMPI_Decode_contiguous_x(intype, count):
    _, _, _, combiner = intype.Get_envelope()
    if combiner not in (COMBINER_STRUCT, COMBINER_VECTOR):
        msg = "not a BigMPI type"
        raise ValueError(msg)
    cbs, _, vbasetype = intype.Get_contents()
    count = cbs[0] * cbs[1]
    basetype = vbasetype[0]
    assert cbs[1] == cbs[2]  # blocklength = stride
    return basetype, count

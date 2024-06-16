from mpi4py.util.dtlib import from_numpy_dtype

BIGMPI_MAXSIZE = 2**30


def unpack_bufspec(bufspec):
    if hasattr(bufspec, "dtype"):
        return bufspec, bufspec.size, from_numpy_dtype(bufspec.dtype)
    if len(bufspec) == 2:
        return bufspec[0], bufspec[1], from_numpy_dtype(bufspec[0].dtype)
    assert len(bufspec) == 3
    return bufspec[0], bufspec[1], bufspec[2]

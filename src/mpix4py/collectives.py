from .type_contiguous_x import BigMPI_Type_contiguous
from .utils import BIGMPI_MAXSIZE, unpack_bufspec


def MPIX_Bcast_x(comm, buf, root=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        comm.Bcast([bufr, count, datatype], root=root)
    else:
        newtype = BigMPI_Type_contiguous(datatype, count)
        comm.Bcast([bufr, 1, newtype], root=root)
        newtype.Free()


def MPIX_Gather_x(comm, sendbuf, recvbuf, root=0):
    sbufr, scount, sdatatype = unpack_bufspec(sendbuf)
    rbufr, rcount, rdatatype = unpack_bufspec(recvbuf)

    if scount <= BIGMPI_MAXSIZE and rcount <= BIGMPI_MAXSIZE:
        comm.Gather([sbufr, scount, sdatatype], [rbufr, rcount, rdatatype], root=root)
    else:
        newtyper = BigMPI_Type_contiguous(rdatatype, rcount)
        newtypes = BigMPI_Type_contiguous(sdatatype, scount)
        comm.Gather([sbufr, 1, newtypes], [rbufr, 1, newtyper], root=root)
        newtypes.Free()
        newtyper.Free()


def MPIX_Scatter_x(comm, sendbuf, recvbuf, root=0):
    sbufr, scount, sdatatype = unpack_bufspec(sendbuf)
    rbufr, rcount, rdatatype = unpack_bufspec(recvbuf)

    if scount <= BIGMPI_MAXSIZE and rcount <= BIGMPI_MAXSIZE:
        comm.Scatter([sbufr, scount, sdatatype], [rbufr, rcount, rdatatype], root=root)
    else:
        newtyper = BigMPI_Type_contiguous(rdatatype, rcount)
        newtypes = BigMPI_Type_contiguous(sdatatype, scount)
        comm.Scatter([sbufr, 1, newtypes], [rbufr, 1, newtyper], root=root)
        newtypes.Free()
        newtyper.Free()


def MPIX_Allgather_x(comm, sendbuf, recvbuf):
    sbufr, scount, sdatatype = unpack_bufspec(sendbuf)
    rbufr, rcount, rdatatype = unpack_bufspec(recvbuf)
    if scount <= BIGMPI_MAXSIZE and rcount <= BIGMPI_MAXSIZE:
        comm.Allgather([sbufr, scount, sdatatype], [rbufr, rcount, rdatatype])
    else:
        newtypes = BigMPI_Type_contiguous(sdatatype, scount)
        newtyper = BigMPI_Type_contiguous(rdatatype, rcount)
        comm.Allgather([sbufr, 1, newtypes], [rbufr, 1, newtyper])
        newtypes.Free()
        newtyper.Free()


def MPIX_Alltoall_x(comm, sendbuf, recvbuf):
    sbufr, scount, sdatatype = unpack_bufspec(sendbuf)
    rbufr, rcount, rdatatype = unpack_bufspec(recvbuf)
    if scount <= BIGMPI_MAXSIZE and rcount <= BIGMPI_MAXSIZE:
        comm.Alltoall([sbufr, scount, sdatatype], [rbufr, rcount, rdatatype])
    else:
        newtypes = BigMPI_Type_contiguous(sdatatype, scount)
        newtyper = BigMPI_Type_contiguous(rdatatype, rcount)
        comm.Alltoall([sbufr, 1, newtypes], [rbufr, 1, newtyper])
        newtypes.Free()
        newtyper.Free()

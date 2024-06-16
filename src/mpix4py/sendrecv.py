from mpi4py import MPI

from .type_contiguous_x import BigMPI_Type_contiguous
from .utils import BIGMPI_MAXSIZE, unpack_bufspec


def MPIX_Send_x(comm, buf, dest, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        comm.Send([bufr, count, datatype], dest=dest, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    comm.Send([bufr, 1, newtype], dest=dest, tag=tag)
    newtype.Free()


def MPIX_Recv_x(comm, buf, source=MPI.ANY_SOURCE, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        comm.Recv([bufr, count, datatype], source=source, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    comm.Recv([bufr, 1, newtype], source=source, tag=tag)
    newtype.Free()


def MPIX_Isend_x(comm, buf, dest, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        return comm.Isend([bufr, count, datatype], dest=dest, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    req = comm.Isend([bufr, 1, newtype], dest=dest, tag=tag)
    newtype.Free()
    return req


def MPIX_Irecv_x(comm, buf, source=MPI.ANY_SOURCE, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        return comm.Irecv([bufr, count, datatype], source=source, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    req = comm.Irecv([bufr, 1, newtype], source=source, tag=tag)
    newtype.Free()
    return req


def MPIX_Sendrecv_x(
    comm, sendbuf, dest, sendtag, recvbuf, source=MPI.ANY_SOURCE, recvtag=MPI.ANY_TAG
):
    sbufr, scount, sdatatype = unpack_bufspec(sendbuf)
    rbufr, rcount, rdatatype = unpack_bufspec(recvbuf)

    if scount <= BIGMPI_MAXSIZE and rcount <= BIGMPI_MAXSIZE:
        comm.Sendrecv(
            [sbufr, scount, sdatatype],
            dest=dest,
            sendtag=sendtag,
            recvbuf=[rbufr, rcount, rdatatype],
            source=source,
            recvtag=recvtag,
        )
    elif scount <= BIGMPI_MAXSIZE < rcount:
        newtype = BigMPI_Type_contiguous(rdatatype, rcount)
        comm.Sendrecv(
            [sbufr, scount, sdatatype],
            dest=dest,
            sendtag=sendtag,
            recvbuf=[rbufr, 1, newtype],
            source=source,
            recvtag=recvtag,
        )
        newtype.Free()
    elif rcount <= BIGMPI_MAXSIZE < scount:
        newtype = BigMPI_Type_contiguous(sdatatype, scount)
        comm.Sendrecv(
            [sbufr, 1, newtype],
            dest=dest,
            sendtag=sendtag,
            recvbuf=[rbufr, rcount, rdatatype],
            source=source,
            recvtag=recvtag,
        )
        newtype.Free()
    else:
        newstype = BigMPI_Type_contiguous(sdatatype, scount)
        newrtype = BigMPI_Type_contiguous(rdatatype, rcount)
        comm.Sendrecv(
            [sbufr, 1, newstype],
            dest=dest,
            sendtag=sendtag,
            recvbuf=[rbufr, 1, newrtype],
            source=source,
            recvtag=recvtag,
        )
        newstype.Free()
        newrtype.Free()


def MPIX_Sendrecv_replace_x(
    comm, buf, dest, sendtag, source=MPI.ANY_SOURCE, recvtag=MPI.ANY_TAG, status=None
):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        comm.Sendrecv_replace(
            [bufr, count, datatype],
            dest=dest,
            sendtag=sendtag,
            source=source,
            recvtag=recvtag,
            status=status,
        )
    else:
        newtype = BigMPI_Type_contiguous(datatype, count)
        comm.Sendrecv_replace(
            [bufr, 1, newtype],
            dest=dest,
            sendtag=sendtag,
            source=source,
            recvtag=recvtag,
            status=status,
        )
        newtype.Free()


def MPIX_Ssend_x(comm, buf, dest, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        comm.Ssend([bufr, count, datatype], dest=dest, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    comm.Ssend([bufr, 1, newtype], dest=dest, tag=tag)
    newtype.Free()


def MPIX_Rsend_x(comm, buf, dest, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        comm.Rsend([bufr, count, datatype], dest=dest, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    comm.Rsend([bufr, 1, newtype], dest=dest, tag=tag)
    newtype.Free()


def MPIX_Issend_x(comm, buf, dest, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        return comm.Issend([bufr, count, datatype], dest=dest, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    req = comm.Issend([bufr, 1, newtype], dest=dest, tag=tag)
    newtype.Free()
    return req


def MPIX_Irsend_x(comm, buf, dest, tag=0):
    bufr, count, datatype = unpack_bufspec(buf)
    if count <= BIGMPI_MAXSIZE:
        return comm.Irsend([bufr, count, datatype], dest=dest, tag=tag)
    newtype = BigMPI_Type_contiguous(datatype, count)
    req = comm.Irsend([bufr, 1, newtype], dest=dest, tag=tag)
    newtype.Free()
    return req

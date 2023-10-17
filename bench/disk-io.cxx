#include <cassert>
#include <string>
#include <CLI11.hpp>
#include <mpi.h>

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  int np, rank;

  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &np);

  double to_GB(8.0 / 1024 / 1024 / 1024);
  int64_t no(10), nv(100), nk(1);

  /************************
   * Purpose of this test:
   * Profile the disk-IO bandwidth for realistic atrip settings.
   * We focus on the Tabij object sliced with respect the first
   * virtual orbital index (data in the array is provided such
   * that this is slowest index). Virtual orbitals are distributed
   * within a round robin fashion.
   ***********************/
  CLI::App app{"IO bench mainly for k-point version of atrip"};
  app.add_option("--no", no, "Occupied orbitals");
  app.add_option("--nv", nv, "Virtual orbitals");
  app.add_option("--nk", nk, "k-points");
  CLI11_PARSE(app, argc, argv);

  int64_t total_elements(nk * nk * nk * no * no * nv * nv);
  if (!rank) printf("%lf\n", total_elements * to_GB);
  // we distribute all data from file into nk*nv buffers distribtued
  // in a round robin fashion among all ranks
  int64_t buffer_size(nk * nk * no * no * nv);
  int64_t buffer_per_rank(nk * nv / np);
  if ((nk * nv) % np) buffer_per_rank++;
  int64_t elements_per_rank(buffer_size * buffer_per_rank);
  // we slighly increase the total_elements to simplify stuff
  total_elements = elements_per_rank * np;

  if (!rank) {
    printf("Working with the following parameters:\n");
    printf("no: %ld\n", no);
    printf("nv: %ld\n", nv);
    printf("nk: %ld\n", nk);
    printf("Total    buffer size %lf GB\n", total_elements * to_GB);
    printf("Per rank buffer size %lf GB\n", elements_per_rank * to_GB);
    printf("np: %d\n", np);
    printf("Total number of buffers: %ld (%ld * %ld)\n", nk * nv, nv, nk);
    printf("Buffer per rank: %ld\n", buffer_per_rank);
  }

  if (sizeof(double) * elements_per_rank >= 2147483648) {
    if (!rank) printf("Buffer too large for given ranks\n");
    return 1;
  }
  double *wdata = (double *)malloc(elements_per_rank * sizeof(double));
  double *rdata = (double *)malloc(elements_per_rank * sizeof(double));

  for (decltype(elements_per_rank) i(0); i < elements_per_rank; i++)
    wdata[i] = 3.14 / (i + 2.1);

  MPI_Offset file_offset;
  MPI_Status stat;
  MPI_Datatype mpiType = MPI_DOUBLE; // MPI_COMPLEX

  MPI_File file;
  MPI_File_open(MPI_COMM_WORLD,
                "f.tmp",
                MPI_MODE_RDWR | MPI_MODE_CREATE,
                MPI_INFO_NULL,
                &file);

  /*****************
   * WRITE TO DISK *
   *****************/
  MPI_Barrier(MPI_COMM_WORLD);
  double start_write = MPI_Wtime();
  file_offset = elements_per_rank * sizeof(double) * rank;

  MPI_File_write_at(file,
                    file_offset,
                    wdata,
                    elements_per_rank,
                    mpiType,
                    &stat);
  MPI_Barrier(MPI_COMM_WORLD);
  double write_time = MPI_Wtime() - start_write;

  if (!rank)
    printf("Write file: %lf s, bandwidth: %lf GB/s\n",
           write_time,
           total_elements * to_GB / write_time);

  /**************************
   * READ NAIVELY FROM DISK *
   *************************/

  MPI_Barrier(MPI_COMM_WORLD);
  double start_naive_read = MPI_Wtime();

  MPI_File_read_at(file, file_offset, rdata, elements_per_rank, mpiType, &stat);

  MPI_Barrier(MPI_COMM_WORLD);
  double naive_read_time = MPI_Wtime() - start_naive_read;

  if (!rank)
    printf(
        "Read file naively: %lf s, bandwidth: %lf GB/s, 1 read %ld elements\n",
        naive_read_time,
        total_elements * to_GB / naive_read_time,
        elements_per_rank);

  /***************************************
   * READ IN ROUND ROBIN STYLE FROM DISK *
   **************************************/

  // if nk == 1:
  // we have nv slices distribued in a round robin fashion
  // ie.: nv slices each nv*no*no of size. Because nv is the
  // slowest index we can read each slice with a single call
  // of MPI_File_read_at (with the corresponding offset)
  //
  // if nk  > 1:
  // the considered tensor Tabij has nk*nk*nk * nv*nv*no*no elements
  // typically the k-point indices nk are the slowest indices
  // in atrip we want have nv*nk slices with each of size nk*nk*nv*no*no
  // the problem is that now the data is not consecutive and we need
  // nk*nk calls of MPI_File_read_at with different offsets. Each call
  // reads only nv*no*no elements
  double total_round_robin_read_time(0.0);
  for (decltype(buffer_per_rank) i(0); i < buffer_per_rank; i++) {
    for (size_t k(0); k < nk * nk; k++) {
      MPI_Barrier(MPI_COMM_WORLD);
      double start_round_robin_read = MPI_Wtime();
      // file offset is a bit cumbersome
      // the number of slices is buffer_per_rank * np
      // which is larger equal nk*nv
      // first we have to get the corresponding nk and nv index:
      int64_t nk_nv_id = i * np + rank;
      int64_t _nk = nk_nv_id / nv;
      int64_t _nv = nk_nv_id % nv;
      file_offset = _nk * nk * nk * nv * nv * no * no;
      file_offset += k * nv * nv * no * no;
      file_offset += _nv * nv * no * no;
      file_offset *= sizeof(double);
      // the following holds only for single k-point
      // file_offset = (i*np + rank) * buffer_size * sizeof(double);
      // the array offset is the sum of
      // 1.) the buffer-offset (total number of buffers: nk*nv)
      // 2.) the nk*nk blocks of size no*no*nv
      size_t array_offset = i * buffer_size + k * no * no * nv;
      MPI_File_read_at(file,
                       file_offset,
                       rdata + array_offset,
                       no * no * nv,
                       mpiType,
                       &stat);

      MPI_Barrier(MPI_COMM_WORLD);
      double round_robin_read_time = MPI_Wtime() - start_round_robin_read;
      total_round_robin_read_time += round_robin_read_time;
      // if (!rank) printf("Time to read the file in round robin style: %lf,
      // overall bandwidth: %lf\n", round_robin_read_time, buffer_size*np*to_GB
      // / round_robin_read_time);
    }
  }
  if (!rank)
    printf(
        "Read file round robin style: %lf s, bandwidth: %lf GB/s, %ld reads "
        "%ld elements\n",
        total_round_robin_read_time,
        total_elements * to_GB / total_round_robin_read_time,
        nk * nk,
        no * no * nv);

  MPI_File_close(&file);
  remove("f.tmp");

  MPI_Finalize();
  return 0;
}

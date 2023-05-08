#include "atrip/Atrip.hpp"
#include <iostream>
#include <vector>
#include <functional>

#include <CLI11.hpp>

#include <bench/utils.hpp>

#include <atrip.hpp>
#include <atrip/Debug.hpp>
#include <atrip/Utils.hpp>

#define _print_size(what, size)                                                \
  do {                                                                         \
    if (rank == 0) {                                                           \
      std::cout << #what << " => " << (double)size * elem_to_gb << "GB"        \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

template <typename F>
class InputTensors {
public:
  using T = CTF::Tensor<F>;
  using Map = std::map<std::string, T *>;

  Map map;

  T *read_or_fill(std::string const &name,
                  int order,
                  int *lens,
                  int *syms,
                  CTF::World world,
                  std::string const &path,
                  F const a,
                  F const b) {
    int rank;
    MPI_Comm_rank(world.comm, &rank);
    map[name] = new T(order, lens, syms, world);
    auto tsr = map[name];
    if (path.size()) {
      tsr->read_dense_from_file(path.c_str());
    } else {
      if (!rank)
        std::cout << "WARNING: file not found! Random initialization\n";
      tsr->fill_random(a, b);
    }
    return tsr;
  }
};

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  CTF::World world(argc, argv);
  int rank, nranks;
  MPI_Comm_rank(world.comm, &rank);
  MPI_Comm_size(world.comm, &nranks);

  size_t checkpoint_it, max_iterations;
  int no(10), nv(100), itMod(-1), percentageMod(10);
  float checkpoint_percentage;
  bool nochrono(false), barrier(false), rankRoundRobin(false), keepVppph(false),
      noCheckpoint = false, blocking = false, complex = false, cT = false;
  std::string tuplesDistributionString = "naive",
              checkpoint_path = "checkpoint.yaml";

  CLI::App app{"Main bench for atrip"};

  defoption(app, "--no", no, "Occupied orbitals")->required();
  defoption(app, "--nv", nv, "Virtual orbitals")->required();
  defoption(app, "--mod", itMod, "Iteration modifier");
  defoption(app,
            "--max-iterations",
            max_iterations,
            "Maximum number of iterations to run");
  defoption(app, "--dist", tuplesDistributionString, "Which distribution")
      ->required();
  defflag(app, "--complex", complex, "Use the complex version of atrip bench");
  defflag(app, "--keep-vppph", keepVppph, "Do not delete the tensor Vppph");
  defflag(app, "--nochrono", nochrono, "Do not print chrono");
  defflag(app, "--rank-round-robin", rankRoundRobin, "Do rank round robin");
  defflag(app, "--barrier", barrier, "Use the first barrier");

  defflag(app, "--blocking", blocking, "Perform blocking communication");
  defoption(app, "-%", percentageMod, "Percentage to be printed");

  // checkpointing
  defflag(app, "--nocheckpoint", noCheckpoint, "Do not use checkpoint");
  defoption(app, "--checkpoint-path", checkpoint_path, "Path for checkpoint");
  defoption(app,
            "--checkpoint-it",
            checkpoint_it,
            "Checkpoint at every iteration");
  defoption(app,
            "--checkpoint-%",
            checkpoint_percentage,
            "Percentage for checkpoints");
  // completeTriples
  defflag(app, "--cT", cT, "Perform (cT) calculation");

  // Optional tensor files
  std::string ei_path, ea_path, Tph_path, Tpphh_path, Vpphh_path, Vhhhp_path,
      Vppph_path, Jppph_path, Jhphh_path;
  defoption(app, "--ei", ei_path, "Path for ei");
  defoption(app, "--ea", ea_path, "Path for ea");
  defoption(app, "--Tpphh", Tpphh_path, "Path for Tpphh");
  defoption(app, "--Tph", Tph_path, "Path for Tph");
  defoption(app, "--Vpphh", Vpphh_path, "Path for Vpphh");
  defoption(app, "--Vhhhp", Vhhhp_path, "Path for Vhhhp");
  defoption(app, "--Vppph", Vppph_path, "Path for Vppph");

  defoption(app, "--Jppph", Jppph_path, "Path for Jppph");
  defoption(app, "--Jhphh", Jhphh_path, "Path for Jhphh");

#if defined(HAVE_CUDA)
  size_t ooo_threads = 0, ooo_blocks = 0;
  defoption(
      app,
      "--ooo-blocks",
      ooo_blocks,
      "CUDA: Number of blocks per block for kernels going through ooo tensors");
  defoption(app,
            "--ooo-threads",
            ooo_threads,
            "CUDA: Number of threads per block for kernels going through "
            "ooo tensors");
#endif

  CLI11_PARSE(app, argc, argv);

  constexpr double elem_to_gb = 8.0 / 1024.0 / 1024.0 / 1024.0;

  // USER PRINTING TEST BEGIN
  const double doublesFlops = no * no * no // common parts of the matrices
                            * (no + nv)    // particles and holes
                            * 2.0          // flops has to be times 2
                            * 6.0          // how many dgemms are there
                            / 1.0e9;       // calculate it in gflops
  double lastElapsedTime = 0;
  bool firstHeaderPrinted = false;
  atrip::registerIterationDescriptor(
      [doublesFlops, &firstHeaderPrinted, rank, &lastElapsedTime](
          atrip::IterationDescription const &d) {
        const char *fmt_nums = "%-13.0f%-10.0f%-13.3f";
        char out[256];
        if (!firstHeaderPrinted) {
          const char *fmt_header = "%-13s%-10s%-13s";
          sprintf(out, fmt_header, "Progress(%)", "time(s)", "GFLOP/s");
          firstHeaderPrinted = true;
          if (rank == 0) std::cout << out << "\n";
        }
        sprintf(out,
                fmt_nums,
                double(d.currentIteration) / double(d.totalIterations) * 100,
                (d.currentElapsedTime - lastElapsedTime),
                d.currentIteration * doublesFlops / d.currentElapsedTime);
        lastElapsedTime = d.currentElapsedTime;
        if (rank == 0) std::cout << out << "\n";
      });

  // USER PRINTING TEST END

  size_t const

      f = complex ? sizeof(double) : sizeof(atrip::Complex),

      n_tuples =

          nv * (nv + 1) * (nv + 2) / 6 - nv, // All tuples

      atrip_memory =

          3 * sizeof(size_t) * n_tuples // tuples_memory

          //
          // one dimensional slices (all ranks)
          //

          + f * nranks * 6 * nv * no * no // taphh
          + f * nranks * 6 * no * no * no // hhha

          //
          // two dimensional slices (all ranks)
          //

          + f * nranks * 12 * nv * no // abph
          + f * nranks * 6 * no * no  // abhh
          + f * nranks * 6 * no * no  // tabhh

          //
          // distributed sources (all ranks)
          //

          + f * nv * nv * no * no // tpphh
          + f * no * no * no * nv // vhhhp
          + f * nv * nv * nv * no // vppph
          + f * nv * nv * no * no // vpphh
          + f * nv * nv * no * no // tpphh2

          //
          // tensors in every rank
          //

          + f * nranks * no * no * no // tijk
          + f * nranks * no * no * no // zijk
          + f * nranks * (no + nv)    // epsp
          + f * nranks * no * nv      // tai
      ;                               // end

  if (rank == 0) {
    std::cout << "Tentative MEMORY USAGE (GB): "
              << double(atrip_memory) / 1024.0 / 1024.0 / 1024.0 << "\n";
  }

  std::vector<int> symmetries(4, NS), vo({nv, no}), vvoo({nv, nv, no, no}),
      ooov({no, no, no, nv}), vvvo({nv, nv, nv, no}), ovoo({no, nv, no, no});

  _print_size(Vabci, no * nv * nv * nv);
  _print_size(Vabij, no * no * nv * nv);
  _print_size(Vijka, no * no * no * nv);

  if (!rank)
    for (auto const &fn : input_printer)
      // print input parameters
      fn();

  atrip::Atrip::init(MPI_COMM_WORLD);

#define RUN_ATRIP(FIELD)                                                       \
  do {                                                                         \
                                                                               \
    atrip::Atrip::Input<FIELD>::TuplesDistribution tuplesDistribution;         \
    {                                                                          \
      using atrip::Atrip;                                                      \
      if (tuplesDistributionString == "naive") {                               \
        tuplesDistribution = Atrip::Input<FIELD>::TuplesDistribution::NAIVE;   \
      } else if (tuplesDistributionString == "group") {                        \
        tuplesDistribution =                                                   \
            Atrip::Input<FIELD>::TuplesDistribution::GROUP_AND_SORT;           \
      } else {                                                                 \
        std::cout << "--dist should be either naive or group\n";               \
        std::exit(1);                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    InputTensors<FIELD> tensors;                                               \
    auto Jhphh = tensors.read_or_fill("Jhphh",                                 \
                                      4,                                       \
                                      ovoo.data(),                             \
                                      symmetries.data(),                       \
                                      world,                                   \
                                      Jhphh_path,                              \
                                      0,                                       \
                                      1);                                      \
    auto Jhhhp = tensors.read_or_fill("Jhhhp",                                 \
                                      4,                                       \
                                      ooov.data(),                             \
                                      symmetries.data(),                       \
                                      world,                                   \
                                      "",                                      \
                                      0,                                       \
                                      1);                                      \
    (*Jhhhp)["ijka"] = (*Jhphh)["kaij"];                                       \
    const auto in =                                                            \
        atrip::Atrip::Input<FIELD>()                                           \
            .with_epsilon_i(tensors.read_or_fill("ei",                         \
                                                 1,                            \
                                                 ooov.data(),                  \
                                                 symmetries.data(),            \
                                                 world,                        \
                                                 ei_path,                      \
                                                 -40.0,                        \
                                                 -2))                          \
            .with_epsilon_a(tensors.read_or_fill("ea",                         \
                                                 1,                            \
                                                 vo.data(),                    \
                                                 symmetries.data(),            \
                                                 world,                        \
                                                 ea_path,                      \
                                                 2,                            \
                                                 50))                          \
            .with_Tai(tensors.read_or_fill("Tph",                              \
                                           2,                                  \
                                           vo.data(),                          \
                                           symmetries.data(),                  \
                                           world,                              \
                                           Tph_path,                           \
                                           0,                                  \
                                           1))                                 \
            .with_Tabij(tensors.read_or_fill("Tpphh",                          \
                                             4,                                \
                                             vvoo.data(),                      \
                                             symmetries.data(),                \
                                             world,                            \
                                             Tpphh_path,                       \
                                             0,                                \
                                             1))                               \
            .with_Vabij(tensors.read_or_fill("Vpphh",                          \
                                             4,                                \
                                             vvoo.data(),                      \
                                             symmetries.data(),                \
                                             world,                            \
                                             Vpphh_path,                       \
                                             0,                                \
                                             1))                               \
            .with_Vijka(tensors.read_or_fill("Vhhhp",                          \
                                             4,                                \
                                             ooov.data(),                      \
                                             symmetries.data(),                \
                                             world,                            \
                                             Vhhhp_path,                       \
                                             0,                                \
                                             1))                               \
            .with_Vabci(tensors.read_or_fill("Vppph",                          \
                                             4,                                \
                                             vvvo.data(),                      \
                                             symmetries.data(),                \
                                             world,                            \
                                             Vppph_path,                       \
                                             0,                                \
                                             1))                               \
                                                                               \
            .with_Jabci(tensors.read_or_fill("Jppph",                          \
                                             4,                                \
                                             vvvo.data(),                      \
                                             symmetries.data(),                \
                                             world,                            \
                                             Jppph_path,                       \
                                             0,                                \
                                             1))                               \
            .with_Jijka(Jhhhp)                                                 \
            .with_deleteVppph(!keepVppph)                                      \
            .with_barrier(barrier)                                             \
            .with_blocking(blocking)                                           \
            .with_chrono(!nochrono)                                            \
            .with_rankRoundRobin(rankRoundRobin)                               \
            .with_iterationMod(itMod)                                          \
            .with_percentageMod(percentageMod)                                 \
            .with_tuplesDistribution(tuplesDistribution)                       \
            .with_maxIterations(max_iterations)                                \
                                                                               \
            .with_checkpointAtEveryIteration(checkpoint_it)                    \
            .with_checkpointAtPercentage(checkpoint_percentage)                \
            .with_checkpointPath(checkpoint_path)                              \
            .with_readCheckpointIfExists(!noCheckpoint);                       \
                                                                               \
    try {                                                                      \
      auto out = atrip::Atrip::run<FIELD>(in);                                 \
      if (!atrip::Atrip::rank) {                                               \
        std::cout << "Used " << #FIELD << " version of atrip" << std::endl;    \
        std::cout << "Energy: " << out.energy << std::endl;                    \
      }                                                                        \
    } catch (const char *msg) {                                                \
      if (!atrip::Atrip::rank)                                                 \
        std::cout << "Atrip throwed with msg:\n\t\t " << msg << "\n";          \
    }                                                                          \
  } while (0)

  if (complex) RUN_ATRIP(atrip::Complex);
  else RUN_ATRIP(double);

  // if (!in.deleteVppph) delete tensors->[]Vppph;

  MPI_Finalize();
  return 0;
}

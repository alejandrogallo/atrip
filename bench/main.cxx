#include <iostream>
#include <vector>
#include <functional>

#include <CLI11.hpp>

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

// Printer for options
std::vector<std::function<void(void)>> input_printer;
#define _register_printer(flag, variable)                                      \
  input_printer.push_back([&variable]() {                                      \
    std::cout << "Input " << flag << " " << variable << std::endl;             \
  })

#define defoption(app, flag, variable, description)                            \
  _register_printer(flag, variable);                                           \
  app.add_option(flag, variable, description)

#define defflag(app, flag, variable, description)                              \
  _register_printer(flag, variable);                                           \
  app.add_flag(flag, variable, description)

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
      noCheckpoint = false, blocking = false;
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
  defflag(app, "--keep-vppph", keepVppph, "Do not delete the tensor Vppph");
  defflag(app, "--nochrono", nochrono, "Do not print chrono");
  defflag(app, "--rank-round-robin", rankRoundRobin, "Do rank round robin");
  defflag(app, "--barrier", barrier, "Use the first barrier");

  app.add_flag("--blocking", blocking, "Perform blocking communication");
  app.add_option("-%", percentageMod, "Percentage to be printed");

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

  // Optional tensor files
  std::string ei_path, ea_path, Tph_path, Tpphh_path, Vpphh_path, Vhhhp_path,
      Vppph_path;
  defoption(app, "--ei", ei_path, "Path for ei");
  defoption(app, "--ea", ea_path, "Path for ea");
  defoption(app, "--Tpphh", Tpphh_path, "Path for Tpphh");
  defoption(app, "--Tph", Tph_path, "Path for Tph");
  defoption(app, "--Vpphh", Vpphh_path, "Path for Vpphh");
  defoption(app, "--Vhhhp", Vhhhp_path, "Path for Vhhhp");
  defoption(app, "--Vppph", Vppph_path, "Path for Vppph");

#if defined(HAVE_CUDA)
  size_t ooo_threads = 0, ooo_blocks = 0;
  app.add_option(
      "--ooo-blocks",
      ooo_blocks,
      "CUDA: Number of blocks per block for kernels going through ooo tensors");
  app.add_option("--ooo-threads",
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

  atrip::Atrip::Input<double>::TuplesDistribution tuplesDistribution;
  {
    using atrip::Atrip;
    if (tuplesDistributionString == "naive") {
      tuplesDistribution = Atrip::Input<double>::TuplesDistribution::NAIVE;
    } else if (tuplesDistributionString == "group") {
      tuplesDistribution =
          Atrip::Input<double>::TuplesDistribution::GROUP_AND_SORT;
    } else {
      std::cout << "--dist should be either naive or group\n";
      std::exit(1);
    }
  }

  size_t const

      f = sizeof(double),

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
      ooov({no, no, no, nv}), vvvo({nv, nv, nv, no});

  CTF::Tensor<double>                                  // All input tensors
      ei(1, ooov.data(), symmetries.data(), world),    // Holes
      ea(1, vo.data(), symmetries.data(), world),      // Particles
      Tph(2, vo.data(), symmetries.data(), world),     // Singles
      Tpphh(4, vvoo.data(), symmetries.data(), world), // Doubles
      Vpphh(4, vvoo.data(), symmetries.data(), world), // Vabij
      Vhhhp(4, ooov.data(), symmetries.data(), world); // Vijka

  // initialize deletable tensors in heap
  auto Vppph =
      new CTF::Tensor<double>(4, vvvo.data(), symmetries.data(), world);

  _print_size(Vabci, no * nv * nv * nv);
  _print_size(Vabij, no * no * nv * nv);
  _print_size(Vijka, no * no * no * nv);

#define _read_or_fill(tsr, a, b)                                               \
  do {                                                                         \
    if (tsr##_path.size()) {                                                   \
      tsr.read_dense_from_file(tsr##_path.c_str());                            \
    } else {                                                                   \
      tsr.fill_random(a, b);                                                   \
    }                                                                          \
  } while (0)

  _read_or_fill(ei, -40.0, -2);
  _read_or_fill(ea, 2, 50);
  _read_or_fill(Tpphh, 0, 1);
  _read_or_fill(Tph, 0, 1);
  _read_or_fill(Vpphh, 0, 1);
  _read_or_fill(Vhhhp, 0, 1);
  if (Vppph_path.size()) {
    Vppph->read_dense_from_file(Vppph_path.c_str());
  } else {
    Vppph->fill_random(0, 1);
  }

  atrip::Atrip::init(MPI_COMM_WORLD);
  const auto

      in = atrip::Atrip::Input<double>()
               // Tensors
               .with_epsilon_i(&ei)
               .with_epsilon_a(&ea)
               .with_Tai(&Tph)
               .with_Tabij(&Tpphh)
               .with_Vabij(&Vpphh)
               .with_Vijka(&Vhhhp)
               .with_Vabci(Vppph)
               // some options
               .with_deleteVppph(!keepVppph)
               .with_barrier(barrier)
               .with_blocking(blocking)
               .with_chrono(!nochrono)
               .with_rankRoundRobin(rankRoundRobin)
               .with_iterationMod(itMod)
               .with_percentageMod(percentageMod)
               .with_tuplesDistribution(tuplesDistribution)
               .with_maxIterations(max_iterations)
               // checkpoint options
               .with_checkpointAtEveryIteration(checkpoint_it)
               .with_checkpointAtPercentage(checkpoint_percentage)
               .with_checkpointPath(checkpoint_path)
               .with_readCheckpointIfExists(!noCheckpoint)
#if defined(HAVE_CUDA)
               .with_oooThreads(ooo_threads)
               .with_oooBlocks(ooo_blocks)
#endif
      ;

  if (!rank)
    for (auto const &fn : input_printer)
      // print input parameters
      fn();

  try {
    auto out = atrip::Atrip::run(in);
    if (atrip::Atrip::rank == 0)
      std::cout << "Energy: " << out.energy << std::endl;
  } catch (const char *msg) {
    if (atrip::Atrip::rank == 0)
      std::cout << "Atrip throwed with msg:\n\t\t " << msg << "\n";
  }

  if (!in.deleteVppph) delete Vppph;

  MPI_Finalize();
  return 0;
}

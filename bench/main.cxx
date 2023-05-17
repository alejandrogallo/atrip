#include "atrip/Atrip.hpp"
#include "atrip/Complex.hpp"
#include "mpi.h"
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
F _conj(const F &i) {
  return i;
}

template <>
atrip::Complex _conj(const atrip::Complex &i) {
  return std::conj(i);
}

template <typename F>
double complex_norm2(CTF::Tensor<F> *i, std::string const idx) {
  CTF::Tensor<F> o(*i);
  const auto add_to_conj =
      CTF::Transform<F, F>([](F const &d, F &f) { f -= _conj<F>(d); });

  add_to_conj((*i)[idx.c_str()], o[idx.c_str()]);
  return o.norm2();
}

template <typename F>
CTF::Tensor<F> *read_or_fill(std::string const &name,
                             int order,
                             int *lens,
                             int *syms,
                             CTF::World &world,
                             std::string const &path,
                             F const a,
                             F const b) {

  const auto file_exists = [](std::string const &filename) {
    ifstream file(filename.c_str());
    return file.good();
  };
  int rank;
  MPI_Comm_rank(world.comm, &rank);
  auto tsr = new CTF::Tensor<F>(order, lens, syms, world);
  if (!rank)
    std::cout << _FORMAT("made tsr %s<%p>",
                         name.c_str(),
                         static_cast<void *>(tsr))
              << std::endl;
  if (path.size() && file_exists(path)) {
    tsr->read_dense_from_file(path.c_str());
  } else {
    if (path.size() && !rank) {
      std::cout << "WARNING: file " << path << " provided but not found!\n";
    }
    if (!rank)
      std::cout << "Random initialization for tensor " << name << std::endl;
    tsr->fill_random(a, b);
  }
  return tsr;
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  CTF::World world(argc, argv);
  int rank, nranks;
  MPI_Comm_rank(world.comm, &rank);
  MPI_Comm_size(world.comm, &nranks);

  size_t checkpoint_it, max_iterations;
  int no(10), nv(100), it_mod(-1), percentage_mod(10);
  float checkpoint_percentage;
  bool nochrono(false), barrier(false), rank_round_robin(false),
      keep_Vppph(false), no_checkpoint = false, blocking = false,
                         complex = false, cT = false;
  std::string tuples_distribution_string = "naive",
              checkpoint_path = "checkpoint.yaml";

  CLI::App app{"Main bench for atrip"};

  defoption(app, "--no", no, "Occupied orbitals")->required();
  defoption(app, "--nv", nv, "Virtual orbitals")->required();
  defoption(app, "--mod", it_mod, "Iteration modifier");
  defoption(app,
            "--max-iterations",
            max_iterations,
            "Maximum number of iterations to run");
  defoption(app, "--dist", tuples_distribution_string, "Which distribution")
      ->required();
  defflag(app, "--complex", complex, "Use the complex version of atrip bench");
  defflag(app, "--keep-vppph", keep_Vppph, "Do not delete the tensor Vppph");
  defflag(app, "--nochrono", nochrono, "Do not print chrono");
  defflag(app, "--rank-round-robin", rank_round_robin, "Do rank round robin");
  defflag(app, "--barrier", barrier, "Use the first barrier");

  defflag(app, "--blocking", blocking, "Perform blocking communication");
  defoption(app, "-%", percentage_mod, "Percentage to be printed");

  // checkpointing
  defflag(app, "--nocheckpoint", no_checkpoint, "Do not use checkpoint");
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
  const double doubles_flops = no * no * no // common parts of the matrices
                             * (no + nv)    // particles and holes
                             * (complex ? 4.0 : 1.0)
                             * 2.0    // flops has to be times 2
                             * 6.0    // how many dgemms are there
                             / 1.0e9; // calculate it in gflops
  double last_elapsed_time = 0;
  bool first_header_printed = false;
  atrip::register_iteration_descriptor(
      [doubles_flops, &first_header_printed, rank, &last_elapsed_time](
          atrip::IterationDescription const &d) {
        const char *fmt_nums = "%-13.0f%-10.0f%-13.3f";
        char out[256];
        if (!first_header_printed) {
          const char *fmt_header = "%-13s%-10s%-13s";
          sprintf(out, fmt_header, "Progress(%)", "time(s)", "GFLOP/s");
          first_header_printed = true;
          if (rank == 0) std::cout << out << "\n";
        }
        sprintf(out,
                fmt_nums,
                double(d.current_iteration) / double(d.total_iterations) * 100,
                (d.current_elapsed_time - last_elapsed_time),
                d.current_iteration * doubles_flops / d.current_elapsed_time);
        last_elapsed_time = d.current_elapsed_time;
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

  void *epsi, *epsa;
  CTF::Tensor<double> *real_epsi = read_or_fill<double>("real-ei",
                                                        1,
                                                        ovoo.data(),
                                                        symmetries.data(),
                                                        world,
                                                        ei_path,
                                                        -40.0,
                                                        -2),
                      *real_epsa = read_or_fill<double>("real-ea",
                                                        1,
                                                        vo.data(),
                                                        symmetries.data(),
                                                        world,
                                                        ea_path,
                                                        -40.0,
                                                        -2);

  if (complex) {
    using F = atrip::Complex;
    const auto to_complex = CTF::Transform<double, atrip::Complex>(
        [](double d, atrip::Complex &f) { f = d; });
    epsi = new CTF::Tensor<F>(1, ovoo.data(), symmetries.data(), world);
    epsa = new CTF::Tensor<F>(1, vo.data(), symmetries.data(), world);
    to_complex((*real_epsi)["i"], (*(CTF::Tensor<F> *)epsi)["i"]);
    to_complex((*real_epsa)["a"], (*(CTF::Tensor<F> *)epsa)["a"]);
  } else {
    epsi = static_cast<void *>(real_epsi);
    epsa = static_cast<void *>(real_epsa);
  }

  if (!rank) {
    std::cout << "np " << nranks << std::endl;
    std::cout << "np " << world.np << std::endl;
    for (auto const &fn : input_printer)
      // print input parameters
      fn();
  }

  atrip::Atrip::init(world.comm);

#define RUN_ATRIP(FIELD)                                                       \
  do {                                                                         \
                                                                               \
    atrip::Atrip::Input<FIELD>::TuplesDistribution tuples_distribution;        \
    {                                                                          \
      using atrip::Atrip;                                                      \
      if (tuples_distribution_string == "naive") {                             \
        tuples_distribution = Atrip::Input<FIELD>::TuplesDistribution::NAIVE;  \
      } else if (tuples_distribution_string == "group") {                      \
        tuples_distribution =                                                  \
            Atrip::Input<FIELD>::TuplesDistribution::GROUP_AND_SORT;           \
      } else {                                                                 \
        std::cout << "--dist should be either naive or group\n";               \
        std::exit(1);                                                          \
      }                                                                        \
    }                                                                          \
                                                                               \
    CTF::Tensor<FIELD> *Jppph = nullptr, *Jhphh = nullptr, *Jhhhp = nullptr;   \
    if (cT || (Jppph_path.size() && Jhphh_path.size())) {                      \
      if (!rank) std::cout << "doing cT" << std::endl;                         \
      /**/                                                                     \
      /**/                                                                     \
      /**/                                                                     \
      Jhphh =                                                                  \
          new CTF::Tensor<FIELD>(4, ovoo.data(), symmetries.data(), world);    \
      Jhphh->read_dense_from_file(Jhphh_path.c_str());                         \
      /*Jhphh = read_or_fill<FIELD>("Jhphh",                                   \
                                   4,                                          \
                                   ovoo.data(),                                \
                                   symmetries.data(),                          \
                                   world,                                      \
                                   Jhphh_path,                                 \
                                   0,                                          \
                                   1);*/                                       \
      MPI_Barrier(world.comm);                                                 \
      if (!rank)                                                               \
        std::cout << _FORMAT("init Jhphh done <%p>",                           \
                             static_cast<void *>(Jhphh))                       \
                  << std::endl;                                                \
      /**/                                                                     \
      /**/                                                                     \
      /**/                                                                     \
      Jhhhp =                                                                  \
          new CTF::Tensor<FIELD>(4, ooov.data(), symmetries.data(), world);    \
      MPI_Barrier(world.comm);                                                 \
      if (!rank)                                                               \
        std::cout << _FORMAT("made Jhhhp done <%p>",                           \
                             static_cast<void *>(Jhhhp))                       \
                  << std::endl;                                                \
      MPI_Barrier(world.comm);                                                 \
      /**/                                                                     \
      /**/                                                                     \
      /**/                                                                     \
      Jppph =                                                                  \
          new CTF::Tensor<FIELD>(4, vvvo.data(), symmetries.data(), world);    \
      if (!rank)                                                               \
        std::cout << _FORMAT("made Jppph done <%p>",                           \
                             static_cast<void *>(Jppph))                       \
                  << std::endl;                                                \
      Jppph->read_dense_from_file(Jppph_path.c_str());                         \
      MPI_Barrier(world.comm);                                                 \
      if (!rank)                                                               \
        std::cout << _FORMAT("read Jppph done <%p>",                           \
                             static_cast<void *>(Jppph))                       \
                  << std::endl;                                                \
      MPI_Barrier(world.comm);                                                 \
      if (!rank) std::cout << "Setting Jhhhp from Jhphh" << std::endl;         \
      MPI_Barrier(world.comm);                                                 \
      /* (*Jhhhp)["ijka"] = (*Jhphh)["kaij"];*/                                \
      const auto conjugate = CTF::Transform<FIELD, FIELD>(                     \
          [](FIELD d, FIELD &f) { f = atrip::maybe_conjugate(d); });           \
      conjugate((*Jhphh)["kaij"], (*Jhhhp)["ijka"]);                           \
      MPI_Barrier(world.comm);                                                 \
      if (!rank) std::cout << "done" << std::endl;                             \
    }                                                                          \
                                                                               \
    const auto in = atrip::Atrip::Input<FIELD>()                               \
                        .with_epsilon_i((CTF::Tensor<FIELD> *)epsi)            \
                        .with_epsilon_a((CTF::Tensor<FIELD> *)epsa)            \
                        .with_Tai(read_or_fill<FIELD>("Tph",                   \
                                                      2,                       \
                                                      vo.data(),               \
                                                      symmetries.data(),       \
                                                      world,                   \
                                                      Tph_path,                \
                                                      0,                       \
                                                      1))                      \
                        .with_Tabij(read_or_fill<FIELD>("Tpphh",               \
                                                        4,                     \
                                                        vvoo.data(),           \
                                                        symmetries.data(),     \
                                                        world,                 \
                                                        Tpphh_path,            \
                                                        0,                     \
                                                        1))                    \
                        .with_Vabij(read_or_fill<FIELD>("Vpphh",               \
                                                        4,                     \
                                                        vvoo.data(),           \
                                                        symmetries.data(),     \
                                                        world,                 \
                                                        Vpphh_path,            \
                                                        0,                     \
                                                        1))                    \
                        .with_Vijka(read_or_fill<FIELD>("Vhhhp",               \
                                                        4,                     \
                                                        ooov.data(),           \
                                                        symmetries.data(),     \
                                                        world,                 \
                                                        Vhhhp_path,            \
                                                        0,                     \
                                                        1))                    \
                        .with_Vabci(read_or_fill<FIELD>("Vppph",               \
                                                        4,                     \
                                                        vvvo.data(),           \
                                                        symmetries.data(),     \
                                                        world,                 \
                                                        Vppph_path,            \
                                                        0,                     \
                                                        1))                    \
                                                                               \
                        .with_Jabci(Jppph)                                     \
                        .with_Jijka(Jhhhp)                                     \
                        .with_delete_Vppph(!keep_Vppph)                        \
                        .with_barrier(barrier)                                 \
                        .with_blocking(blocking)                               \
                        .with_chrono(!nochrono)                                \
                        .with_rank_round_robin(rank_round_robin)               \
                        .with_iteration_mod(it_mod)                            \
                        .with_percentage_mod(percentage_mod)                   \
                        .with_tuples_distribution(tuples_distribution)         \
                        .with_max_iterations(max_iterations)                   \
                                                                               \
                        .with_checkpoint_at_every_iteration(checkpoint_it)     \
                        .with_checkpoint_at_percentage(checkpoint_percentage)  \
                        .with_checkpoint_path(checkpoint_path)                 \
                        .with_read_checkpoint_if_exists(!no_checkpoint);       \
                                                                               \
    try {                                                                      \
      auto out = atrip::Atrip::run<FIELD>(in);                                 \
      if (!atrip::Atrip::rank) {                                               \
        std::cout << "Used " << #FIELD << " version of atrip" << std::endl;    \
        std::cout << "Energy: " << out.energy << std::endl;                    \
        std::cout << "Energy (cT): " << out.ct_energy << std::endl;            \
      }                                                                        \
    } catch (const char *msg) {                                                \
      if (!atrip::Atrip::rank)                                                 \
        std::cout << "Atrip throwed with msg:\n\t\t " << msg << "\n";          \
    }                                                                          \
  } while (0)

  if (complex) RUN_ATRIP(atrip::Complex);
  else RUN_ATRIP(double);

  // if (!in.delete_Vppph) delete tensors->[]Vppph;

  MPI_Finalize();
  return 0;
}

#include "atrip/Atrip.hpp"
#include "atrip/Complex.hpp"
#include "mpi.h"
#include <iostream>
#include <vector>
#include <functional>

#include "CLI11.hpp"
#include "utils.hpp"

#include <atrip.hpp>
#include <atrip/Debug.hpp>
#include <atrip/Utils.hpp>

constexpr double elem_to_gb = 8.0 / 1024.0 / 1024.0 / 1024.0;

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

  if (!rank)
    std::cout << _FORMAT("made tsr %s<%p>",
                         name.c_str(),
                         static_cast<void *>(tsr))
              << std::endl;

  return tsr;
}

struct Cli {
  size_t checkpoint_it = 0, max_iterations = 0;
  int no = 10, nv = 100, it_mod = -1, percentage_mod = 10;
  float checkpoint_percentage;
  bool nochrono = false, barrier = false, rank_round_robin = false,
       keep_Vppph = false, no_checkpoint = false, blocking = false,
       complex = false, cT = false;
  std::string tuples_distribution_string = "naive",
              checkpoint_path = "checkpoint.yaml";

  // Optional tensor files
  std::string ei_path, ea_path, Tph_path, Tpphh_path, Vpphh_path, Vhhhp_path,
      Vppph_path, Jppph_path, Jhphh_path;
};

template <typename F>
atrip::Atrip::Output run_atrip(CTF::World world, Cli const &cli) {
  int rank, nranks;
  MPI_Comm_rank(world.comm, &rank);
  MPI_Comm_size(world.comm, &nranks);

  std::vector<int> symmetries(4, NS), vo({cli.nv, cli.no}),
      vvoo({cli.nv, cli.nv, cli.no, cli.no}),
      ooov({cli.no, cli.no, cli.no, cli.nv}),
      vvvo({cli.nv, cli.nv, cli.nv, cli.no}),
      ovoo({cli.no, cli.nv, cli.no, cli.no});

  _print_size(Vabci, cli.no * cli.nv * cli.nv * cli.nv);
  _print_size(Vabij, cli.no * cli.no * cli.nv * cli.nv);
  _print_size(Vijka, cli.no * cli.no * cli.no * cli.nv);

  void *epsi, *epsa;
  CTF::Tensor<double> *real_epsi = read_or_fill<double>("real-ei",
                                                        1,
                                                        ovoo.data(),
                                                        symmetries.data(),
                                                        world,
                                                        cli.ei_path,
                                                        -40.0,
                                                        -2),
                      *real_epsa = read_or_fill<double>("real-ea",
                                                        1,
                                                        vo.data(),
                                                        symmetries.data(),
                                                        world,
                                                        cli.ea_path,
                                                        -40.0,
                                                        -2);

  if (cli.complex) {
    using C = atrip::Complex;
    const auto to_complex = CTF::Transform<double, atrip::Complex>(
        [](double d, atrip::Complex &f) { f = d; });
    epsi = new CTF::Tensor<C>(1, ovoo.data(), symmetries.data(), world);
    epsa = new CTF::Tensor<C>(1, vo.data(), symmetries.data(), world);
    to_complex((*real_epsi)["i"], (*(CTF::Tensor<C> *)epsi)["i"]);
    to_complex((*real_epsa)["a"], (*(CTF::Tensor<C> *)epsa)["a"]);
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

  typename atrip::Atrip::Input<F>::TuplesDistribution tuples_distribution;
  {
    using atrip::Atrip;
    if (cli.tuples_distribution_string == "naive") {
      tuples_distribution = Atrip::Input<F>::TuplesDistribution::NAIVE;
    } else if (cli.tuples_distribution_string == "group") {
      tuples_distribution = Atrip::Input<F>::TuplesDistribution::GROUP_AND_SORT;
    } else {
      std::cout << "--dist should be either naive or group\n";
      std::exit(1);
    }
  }

  CTF::Tensor<F> *Jppph = nullptr, *Jhphh = nullptr, *Jhhhp = nullptr;
  if (cli.cT || (cli.Jppph_path.size() && cli.Jhphh_path.size())) {
    if (!rank) std::cout << "doing cT" << std::endl;
    /**/
    /**/
    /**/
    Jhphh = new CTF::Tensor<F>(4, ovoo.data(), symmetries.data(), world);
    Jhphh->read_dense_from_file(cli.Jhphh_path.c_str());
    /*Jhphh = read_or_fill<F>("Jhphh",
      4,
      ovoo.data(),
      symmetries.data(),
      world,
      cli.Jhphh_path,
      0,
      1);*/
    MPI_Barrier(world.comm);
    if (!rank)
      std::cout << _FORMAT("init Jhphh done <%p>", static_cast<void *>(Jhphh))
                << std::endl;
    /**/
    /**/
    /**/
    Jhhhp = new CTF::Tensor<F>(4, ooov.data(), symmetries.data(), world);
    MPI_Barrier(world.comm);
    if (!rank)
      std::cout << _FORMAT("made Jhhhp done <%p>", static_cast<void *>(Jhhhp))
                << std::endl;
    MPI_Barrier(world.comm);
    /**/
    /**/
    /**/
    Jppph = new CTF::Tensor<F>(4, vvvo.data(), symmetries.data(), world);
    if (!rank)
      std::cout << _FORMAT("made Jppph done <%p>", static_cast<void *>(Jppph))
                << std::endl;
    Jppph->read_dense_from_file(cli.Jppph_path.c_str());
    MPI_Barrier(world.comm);
    if (!rank)
      std::cout << _FORMAT("read Jppph done <%p>", static_cast<void *>(Jppph))
                << std::endl;
    MPI_Barrier(world.comm);
    if (!rank) std::cout << "Setting Jhhhp from Jhphh" << std::endl;
    MPI_Barrier(world.comm);
    /* (*Jhhhp)["ijka"] = (*Jhphh)["kaij"];*/
    const auto conjugate =
        CTF::Transform<F, F>([](F d, F &f) { f = atrip::maybe_conjugate(d); });
    conjugate((*Jhphh)["kaij"], (*Jhhhp)["ijka"]);
    MPI_Barrier(world.comm);
    if (!rank) std::cout << "done" << std::endl;
  }

  const auto in = atrip::Atrip::Input<F>()
                      .with_epsilon_i((CTF::Tensor<F> *)epsi)
                      .with_epsilon_a((CTF::Tensor<F> *)epsa)
                      .with_Tai(read_or_fill<F>("Tph",
                                                2,
                                                vo.data(),
                                                symmetries.data(),
                                                world,
                                                cli.Tph_path,
                                                0,
                                                1))
                      .with_Tabij(read_or_fill<F>("Tpphh",
                                                  4,
                                                  vvoo.data(),
                                                  symmetries.data(),
                                                  world,
                                                  cli.Tpphh_path,
                                                  0,
                                                  1))
                      .with_Vabij(read_or_fill<F>("Vpphh",
                                                  4,
                                                  vvoo.data(),
                                                  symmetries.data(),
                                                  world,
                                                  cli.Vpphh_path,
                                                  0,
                                                  1))
                      .with_Vijka(read_or_fill<F>("Vhhhp",
                                                  4,
                                                  ooov.data(),
                                                  symmetries.data(),
                                                  world,
                                                  cli.Vhhhp_path,
                                                  0,
                                                  1))
                      .with_Vabci(read_or_fill<F>("Vppph",
                                                  4,
                                                  vvvo.data(),
                                                  symmetries.data(),
                                                  world,
                                                  cli.Vppph_path,
                                                  0,
                                                  1))

                      .with_Jabci(Jppph)
                      .with_Jijka(Jhhhp)
                      .with_delete_Vppph(!cli.keep_Vppph)
                      .with_barrier(cli.barrier)
                      .with_blocking(cli.blocking)
                      .with_chrono(!cli.nochrono)
                      .with_rank_round_robin(cli.rank_round_robin)
                      .with_iteration_mod(cli.it_mod)
                      .with_percentage_mod(cli.percentage_mod)
                      .with_tuples_distribution(tuples_distribution)
                      .with_max_iterations(cli.max_iterations)

                      .with_checkpoint_at_every_iteration(cli.checkpoint_it)
                      .with_checkpoint_at_percentage(cli.checkpoint_percentage)
                      .with_checkpoint_path(cli.checkpoint_path)
                      .with_read_checkpoint_if_exists(!cli.no_checkpoint);

  return atrip::Atrip::run<F>(in);
}

int main(int argc, char **argv) {
  MPI_Init(&argc, &argv);
  CTF::World world(argc, argv);
  int rank, nranks;
  MPI_Comm_rank(world.comm, &rank);
  MPI_Comm_size(world.comm, &nranks);

  Cli cli;

  CLI::App app{"Main bench for atrip"};

  defoption(app, "--no", cli.no, "Occupied orbitals")->required();
  defoption(app, "--nv", cli.nv, "Virtual orbitals")->required();
  defoption(app, "--mod", cli.it_mod, "Iteration modifier");
  defoption(app,
            "--max-iterations",
            cli.max_iterations,
            "Maximum number of iterations to run");
  defoption(app, "--dist", cli.tuples_distribution_string, "Which distribution")
      ->required();
  defflag(app,
          "--complex",
          cli.complex,
          "Use the complex version of atrip bench");
  defflag(app,
          "--keep-vppph",
          cli.keep_Vppph,
          "Do not delete the tensor Vppph");
  defflag(app, "--nochrono", cli.nochrono, "Do not print chrono");
  defflag(app,
          "--rank-round-robin",
          cli.rank_round_robin,
          "Do rank round robin");
  defflag(app, "--barrier", cli.barrier, "Use the first barrier");

  defflag(app, "--blocking", cli.blocking, "Perform blocking communication");
  defoption(app, "-%", cli.percentage_mod, "Percentage to be printed");

  // checkpointing
  defflag(app, "--nocheckpoint", cli.no_checkpoint, "Do not use checkpoint");
  defoption(app,
            "--checkpoint-path",
            cli.checkpoint_path,
            "Path for checkpoint");
  defoption(app,
            "--checkpoint-it",
            cli.checkpoint_it,
            "Checkpoint at every iteration");
  defoption(app,
            "--checkpoint-%",
            cli.checkpoint_percentage,
            "Percentage for checkpoints");
  // completeTriples
  defflag(app, "--cT", cli.cT, "Perform (cT) calculation");

  defoption(app, "--ei", cli.ei_path, "Path for ei");
  defoption(app, "--ea", cli.ea_path, "Path for ea");
  defoption(app, "--Tpphh", cli.Tpphh_path, "Path for Tpphh");
  defoption(app, "--Tph", cli.Tph_path, "Path for Tph");
  defoption(app, "--Vpphh", cli.Vpphh_path, "Path for Vpphh");
  defoption(app, "--Vhhhp", cli.Vhhhp_path, "Path for Vhhhp");
  defoption(app, "--Vppph", cli.Vppph_path, "Path for Vppph");

  defoption(app, "--Jppph", cli.Jppph_path, "Path for Jppph");
  defoption(app, "--Jhphh", cli.Jhphh_path, "Path for Jhphh");

  CLI11_PARSE(app, argc, argv);

  // USER PRINTING TEST BEGIN
  const double doubles_flops =
      cli.no * cli.no * cli.no          // common parts of the matrices
      * (cli.no + cli.nv)               // particles and holes
      * (cli.complex ? 4.0 : 1.0) * 2.0 // flops has to be times 2
      * 6.0                             // how many dgemms are there
      / 1.0e9;                          // calculate it in gflops
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

      f = cli.complex ? sizeof(double) : sizeof(atrip::Complex),

      n_tuples =

          cli.nv * (cli.nv + 1) * (cli.nv + 2) / 6 - cli.nv, // All tuples

      atrip_memory =

          3 * sizeof(size_t) * n_tuples // tuples_memory

          //
          // one dimensional slices (all ranks)
          //

          + f * nranks * 6 * cli.nv * cli.no * cli.no // taphh
          + f * nranks * 6 * cli.no * cli.no * cli.no // hhha

          //
          // two dimensional slices (all ranks)
          //

          + f * nranks * 12 * cli.nv * cli.no // abph
          + f * nranks * 6 * cli.no * cli.no  // abhh
          + f * nranks * 6 * cli.no * cli.no  // tabhh

          //
          // distributed sources (all ranks)
          //

          + f * cli.nv * cli.nv * cli.no * cli.no // tpphh
          + f * cli.no * cli.no * cli.no * cli.nv // vhhhp
          + f * cli.nv * cli.nv * cli.nv * cli.no // vppph
          + f * cli.nv * cli.nv * cli.no * cli.no // vpphh
          + f * cli.nv * cli.nv * cli.no * cli.no // tpphh2

          //
          // tensors in every rank
          //

          + f * nranks * cli.no * cli.no * cli.no // tijk
          + f * nranks * cli.no * cli.no * cli.no // zijk
          + f * nranks * (cli.no + cli.nv)        // epsp
          + f * nranks * cli.no * cli.nv          // tai
      ;                                           // end

  if (rank == 0) {
    std::cout << "Tentative MEMORY USAGE (GB): "
              << double(atrip_memory) / 1024.0 / 1024.0 / 1024.0 << "\n";
  }

  try {
    atrip::Atrip::Output out;
    if (cli.complex) out = run_atrip<atrip::Complex>(world, cli);
    else out = run_atrip<double>(world, cli);
    if (!atrip::Atrip::rank) {
      std::cout << "Used " << (cli.complex ? "complex" : "real")
                << " version of atrip" << std::endl;
      std::cout << "Energy: " << out.energy << std::endl;
      std::cout << "Energy (cT): " << out.ct_energy << std::endl;
    }
  } catch (const char *msg) {
    if (!atrip::Atrip::rank)
      std::cout << "Atrip throwed with msg:\n\t\t " << msg << "\n";
  }

  // if (!in.delete_Vppph) delete tensors->[]Vppph;

  MPI_Finalize();
  return 0;
}

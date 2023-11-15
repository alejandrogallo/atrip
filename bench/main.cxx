#include <iostream>
#include <vector>
#include <functional>

#include <mpi.h>

#include <CLI11.hpp>

#include <bench/utils.hpp>

#include <atrip/Atrip.hpp>
#include <atrip/Complex.hpp>
#include <atrip.hpp>
#include <atrip/Debug.hpp>
#include <atrip/Utils.hpp>
#include <atrip/Operations.hpp>

#define _print_size(what, size)                                                \
  do {                                                                         \
    if (rank == 0) {                                                           \
      std::cout << #what << " => " << (double)size * elem_to_gb << "GB"        \
                << std::endl;                                                  \
    }                                                                          \
  } while (0)

#define _flip(one, two)                                                        \
  do {                                                                         \
    auto tmp = one;                                                            \
    one = two;                                                                 \
    two = tmp;                                                                 \
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

struct Settings {
  size_t checkpoint_it, max_iterations;
  int no, nv, it_mod, percentage_mod;
  float checkpoint_percentage;
  bool nochrono, barrier, rank_round_robin, keep_Vppph, no_checkpoint, blocking,
      complex, cT, ijkabc;
  std::string tuples_distribution_string, checkpoint_path;
  // paths
  std::string ei_path, ea_path, Tph_path, Tpphh_path, Vpphh_path, Vhhhp_path,
      Vppph_path, Jppph_path, Jhhhp_path;
};

template <typename FIELD>
void run(int argc, char **argv, Settings &s) {

  MPI_Init(&argc, &argv);
  CTF::World world(argc, argv);
  int rank, nranks;
  MPI_Comm_rank(world.comm, &rank);
  MPI_Comm_size(world.comm, &nranks);

  constexpr double elem_to_gb = 8.0 / 1024.0 / 1024.0 / 1024.0;
  if (s.ijkabc) { _flip(s.no, s.nv); }

  // USER PRINTING TEST BEGIN
  const double doubles_flops =
      s.no * s.no * s.no              // common parts of the matrices
      * (s.no + s.nv)                 // particles and holes
      * (s.complex ? 4.0 : 1.0) * 2.0 // flops has to be times 2
      * 6.0                           // how many dgemms are there
      / 1.0e9;                        // calculate it in gflops
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

      f = s.complex ? sizeof(double) : sizeof(atrip::Complex),

      n_tuples =

          s.nv * (s.nv + 1) * (s.nv + 2) / 6 - s.nv, // All tuples

      atrip_memory =

          3 * sizeof(size_t) * n_tuples // tuples_memory

          //
          // one dimensional slices (all ranks)
          //

          + f * nranks * 6 * s.nv * s.no * s.no // taphh
          + f * nranks * 6 * s.no * s.no * s.no // hhha

          //
          // two dimensional slices (all ranks)
          //

          + f * nranks * 12 * s.nv * s.no // abph
          + f * nranks * 6 * s.no * s.no  // abhh
          + f * nranks * 6 * s.no * s.no  // tabhh

          //
          // distributed sources (all ranks)
          //

          + f * s.nv * s.nv * s.no * s.no // tpphh
          + f * s.no * s.no * s.no * s.nv // vhhhp
          + f * s.nv * s.nv * s.nv * s.no // vppph
          + f * s.nv * s.nv * s.no * s.no // vpphh
          + f * s.nv * s.nv * s.no * s.no // tpphh2

          //
          // tensors in every rank
          //

          + f * nranks * s.no * s.no * s.no // tijk
          + f * nranks * s.no * s.no * s.no // zijk
          + f * nranks * (s.no + s.nv)      // epsp
          + f * nranks * s.no * s.nv        // tai
      ;                                     // end

  if (rank == 0) {
    std::cout << "Tentative MEMORY USAGE (GB): "
              << double(atrip_memory) / 1024.0 / 1024.0 / 1024.0 << "\n";
  }

  std::vector<int> symmetries(4, NS), vo({s.nv, s.no}),
      vvoo({s.nv, s.nv, s.no, s.no}), ooov({s.no, s.no, s.no, s.nv}),
      vvvo({s.nv, s.nv, s.nv, s.no}), ovoo({s.no, s.nv, s.no, s.no});

  _print_size(Vabci, s.no * s.nv * s.nv * s.nv);
  _print_size(Vabij, s.no * s.no * s.nv * s.nv);
  _print_size(Vijka, s.no * s.no * s.no * s.nv);

  void *epsi, *epsa;
  CTF::Tensor<double> *real_epsi = read_or_fill<double>("real-ei",
                                                        1,
                                                        ovoo.data(),
                                                        symmetries.data(),
                                                        world,
                                                        s.ei_path,
                                                        -40.0,
                                                        -2),
                      *real_epsa = read_or_fill<double>("real-ea",
                                                        1,
                                                        vo.data(),
                                                        symmetries.data(),
                                                        world,
                                                        s.ea_path,
                                                        -40.0,
                                                        -2);

  if (s.complex) {
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

  // For the printing we work with the 'correct' definition of no && nv
  if (s.ijkabc) { _flip(s.no, s.nv); }
  if (!rank) {
    std::cout << "np " << nranks << std::endl;
    std::cout << "np " << world.np << std::endl;
    for (auto const &fn : input_printer)
      // print input parameters
      fn();
    if (s.ijkabc)
      std::cout << "ijkabc used, we flip No && Nv internally" << std::endl;
  }
  if (s.ijkabc) { _flip(s.no, s.nv); }

  atrip::Atrip::init(world.comm);

  typename atrip::Atrip::Input<FIELD>::TuplesDistribution tuples_distribution;
  {
    using atrip::Atrip;
    if (s.tuples_distribution_string == "naive") {
      tuples_distribution = Atrip::Input<FIELD>::TuplesDistribution::NAIVE;
    } else if (s.tuples_distribution_string == "group") {
      tuples_distribution =
          Atrip::Input<FIELD>::TuplesDistribution::GROUP_AND_SORT;
    } else {
      std::cout << "dist should be either naive or group\n";
      std::exit(1);
    }
  }

  /* We use the s.notation p = v and q = o for the initial load of T1&T2 */
  /* If we use the p <-> h algorithm we will flip the T-amplitudes     */
  std::vector<int> pq({s.nv, s.no}), ppqq({s.nv, s.nv, s.no, s.no});
  if (s.ijkabc) {
    pq = {s.no, s.nv};
    ppqq = {s.no, s.no, s.nv, s.nv};
  }

  CTF::Tensor<FIELD> *Tph = nullptr, *Tpphh = nullptr, *Vpphh = nullptr;
  CTF::Tensor<FIELD> *iTph = read_or_fill<FIELD>("tph",
                                                 2,
                                                 pq.data(),
                                                 symmetries.data(),
                                                 world,
                                                 s.Tph_path,
                                                 0.0,
                                                 1.0),
                     *iTpphh = read_or_fill<FIELD>("tpphh",
                                                   4,
                                                   ppqq.data(),
                                                   symmetries.data(),
                                                   world,
                                                   s.Tpphh_path,
                                                   0.0,
                                                   1.0),
                     *iVpphh = read_or_fill<FIELD>("Vpphh",
                                                   4,
                                                   ppqq.data(),
                                                   symmetries.data(),
                                                   world,
                                                   s.Vpphh_path,
                                                   0,
                                                   1);
  /*if (P<->H) Switch Tph */
  if (s.ijkabc) {
    Tph = new CTF::Tensor<FIELD>(2, vo.data(), symmetries.data(), world);
    Tpphh = new CTF::Tensor<FIELD>(4, vvoo.data(), symmetries.data(), world);
    Vpphh = new CTF::Tensor<FIELD>(4, vvoo.data(), symmetries.data(), world);

    (*Tph)["ia"] = (*iTph)["ai"];
    (*Tpphh)["ijab"] = (*iTpphh)["abij"];
    (*Vpphh)["ijab"] = (*iVpphh)["abij"];
  } else {
    Tph = iTph;
    Tpphh = iTpphh;
    Vpphh = iVpphh;
  }

  CTF::Tensor<FIELD> *Jppph = nullptr, *Jhhhp = nullptr;
  if (s.cT || (s.Jppph_path.size() && s.Jhhhp_path.size())) {
    if (!rank) std::cout << "doing cT" << std::endl;
    /**/
    /**/
    /**/
    Jhhhp = new CTF::Tensor<FIELD>(4, ooov.data(), symmetries.data(), world);
    MPI_Barrier(world.comm);
    if (!rank)
      std::cout << _FORMAT("made Jhhhp done <%p>", static_cast<void *>(Jhhhp))
                << std::endl;
    const auto conjugate = CTF::Transform<FIELD, FIELD>([](FIELD d, FIELD &f) {
      f = atrip::acc::maybe_conjugate_scalar<FIELD>(d);
    });
    Jhhhp->read_dense_from_file(s.Jhhhp_path.c_str());
    MPI_Barrier(world.comm);
    /**/
    /**/
    /**/
    Jppph = new CTF::Tensor<FIELD>(4, vvvo.data(), symmetries.data(), world);
    if (!rank)
      std::cout << _FORMAT("made Jppph done <%p>", static_cast<void *>(Jppph))
                << std::endl;
    Jppph->read_dense_from_file(s.Jppph_path.c_str());
    MPI_Barrier(world.comm);
    if (!rank)
      std::cout << _FORMAT("read Jppph done <%p>", static_cast<void *>(Jppph))
                << std::endl;
    MPI_Barrier(world.comm);
  }

  const auto in = atrip::Atrip::Input<FIELD>()
                      .with_epsilon_i((CTF::Tensor<FIELD> *)epsi)
                      .with_epsilon_a((CTF::Tensor<FIELD> *)epsa)
                      .with_Tai(Tph)
                      .with_Tabij(Tpphh)
                      .with_Vabij(Vpphh)
                      .with_Vijka(read_or_fill<FIELD>("Vhhhp",
                                                      4,
                                                      ooov.data(),
                                                      symmetries.data(),
                                                      world,
                                                      s.Vhhhp_path,
                                                      0,
                                                      1))
                      .with_Vabci(read_or_fill<FIELD>("Vppph",
                                                      4,
                                                      vvvo.data(),
                                                      symmetries.data(),
                                                      world,
                                                      s.Vppph_path,
                                                      0,
                                                      1))

                      .with_Jabci(Jppph)
                      .with_Jijka(Jhhhp)
                      .with_delete_Vppph(!s.keep_Vppph)
                      .with_barrier(s.barrier)
                      .with_blocking(s.blocking)
                      .with_chrono(!s.nochrono)
                      .with_rank_round_robin(s.rank_round_robin)
                      .with_iteration_mod(s.it_mod)
                      .with_percentage_mod(s.percentage_mod)
                      .with_tuples_distribution(tuples_distribution)
                      .with_max_iterations(s.max_iterations)

                      .with_checkpoint_at_every_iteration(s.checkpoint_it)
                      .with_checkpoint_at_percentage(s.checkpoint_percentage)
                      .with_checkpoint_path(s.checkpoint_path)
                      .with_read_checkpoint_if_exists(!s.no_checkpoint)

                      .with_ijkabc(s.ijkabc);
  try {
    auto out = atrip::Atrip::run<FIELD>(in);
    if (!atrip::Atrip::rank) {
      std::cout << "Energy: " << out.energy << std::endl;
      std::cout << "Energy (cT): " << out.ct_energy << std::endl;
    }
  } catch (const char *msg) {
    if (!atrip::Atrip::rank)
      std::cout << "Atrip throwed with msg:\n\t\t " << msg << "\n";
  } catch (const std::string &msg) {
    if (!atrip::Atrip::rank)
      std::cout << "Atrip throwed with msg:\n\t\t " << msg << "\n";
  }

  MPI_Finalize();
}

int main(int argc, char **argv) {

  Settings s; // CLI settings go here

  CLI::App app{"Main bench for atrip"}; // CLI11 Application object

  //
  // REQUIRED
  //
  defoption(app, "--no", s.no, "Number of occupied orbitals")
      ->default_val(10)
      ->check(CLI::PositiveNumber)
      ->required();
  defoption(app, "--nv", s.nv, "Number of Virtual orbitals")
      ->default_val(100)
      ->check(CLI::PositiveNumber)
      ->required();
  defoption(app, "--dist", s.tuples_distribution_string, "Tuples distribution")
      ->default_val("group")
      ->check(CLI::IsMember({"group", "naive"}))
      ->required();

  //
  // OPTIONAL
  //
  defflag(app, "--ijkabc", s.ijkabc, "Use the ijkabc-algorithm")
      ->default_val(false);
  defoption(app, "--mod", s.it_mod, "Iteration modifier")->default_val(-1);
  defoption(app,
            "--max-iterations",
            s.max_iterations,
            "Maximum number of iterations to run")
      ->default_val(0);
  defflag(app, "--complex", s.complex, "Use the complex version of atrip bench")
      ->default_val(false);
  defflag(app, "--keep-vppph", s.keep_Vppph, "Do not delete the tensor Vppph")
      ->default_val(false);
  defflag(app, "--nochrono", s.nochrono, "Do not print chrono")
      ->default_val(false);
  defflag(app, "--rank-round-robin", s.rank_round_robin, "Do rank round robin")
      ->default_val(false);
  defflag(app, "--barrier", s.barrier, "Use the first barrier")
      ->default_val(false);

  defflag(app, "--blocking", s.blocking, "Perform blocking communication")
      ->default_val(false);
  defoption(app, "-%", s.percentage_mod, "Percentage to be printed")
      ->default_val(10);

  //
  // checkpointing
  //
  defflag(app, "--nocheckpoint", s.no_checkpoint, "Do not use checkpoint")
      ->default_val(false);
  defoption(app, "--checkpoint-path", s.checkpoint_path, "Path for checkpoint")
      ->default_val("checkpoint.yaml");
  defoption(app,
            "--checkpoint-it",
            s.checkpoint_it,
            "Checkpoint at every iteration")
      ->default_val(0);
  defoption(app,
            "--checkpoint-%",
            s.checkpoint_percentage,
            "Percentage for checkpoints")
      ->default_val(0.0);

  //
  // Optional tensor files
  //
  defoption(app, "--ei", s.ei_path, "Path for HF energies ε_i")
      ->check(CLI::ExistingFile);
  defoption(app, "--ea", s.ea_path, "Path for HF energies ε_a")
      ->check(CLI::ExistingFile);
  defoption(app, "--Tpphh", s.Tpphh_path, "Path for Tpphh (Tabij)")
      ->check(CLI::ExistingFile);
  defoption(app, "--Tph", s.Tph_path, "Path for Tph (Tai)")
      ->check(CLI::ExistingFile);
  defoption(app, "--Vpphh", s.Vpphh_path, "Path for Vpphh (Vabij)")
      ->check(CLI::ExistingFile);
  defoption(app, "--Vhhhp", s.Vhhhp_path, "Path for Vhhhp (Vijka)")
      ->check(CLI::ExistingFile);
  defoption(app, "--Vppph", s.Vppph_path, "Path for Vppph (Vabci)")
      ->check(CLI::ExistingFile);

  //
  // completeTriples
  //
  defflag(app, "--cT", s.cT, "Perform (cT) calculation")->default_val(false);
  defoption(app, "--Jppph", s.Jppph_path, "Path for Jppph intermediates")
      ->check(CLI::ExistingFile);
  defoption(app, "--Jhhhp", s.Jhhhp_path, "Path for Jhhhp intermediates")
      ->check(CLI::ExistingFile);

  CLI11_PARSE(app, argc, argv);

  s.complex ? run<atrip::Complex>(argc, argv, s) : run<double>(argc, argv, s);

  return 0;
}

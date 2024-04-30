#include <iostream>
#include <vector>
#include <functional>

#include <mpi.h>

#include <CLI11.hpp>

#include <bench/utils.hpp>

#include <atrip/Atrip.hpp>
#include <atrip/CTFReader.hpp>
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
std::vector<F> *
get_epsilon(std::string const &path, size_t const len, MPI_Comm comm) {
  auto result = new std::vector<F>(len, 0.5);
  if (path.size()) *result = atrip::read_all<F>({len}, path, comm);
  return result;
}

template std::vector<double> *
get_epsilon<double>(std::string const &path, size_t const len, MPI_Comm comm);

template <>
std::vector<atrip::Complex> *
get_epsilon<atrip::Complex>(std::string const &path,
                            size_t const len,
                            MPI_Comm comm) {
  std::vector<double> *real = get_epsilon<double>(path, len, comm);
  auto result = new std::vector<atrip::Complex>(len, 0.5);
  for (size_t i = 0; i < len; i++) {
    (*result)[i] = atrip::Complex((*real)[i]);
  }
  delete real;
  return result;
}

#if defined(HAVE_CTF)
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
#endif /*   defined(HAVE_CTF) */

struct Settings {
  size_t checkpoint_it, max_iterations;
  int no, nv, it_mod, percentage_mod;
  float checkpoint_percentage;
  bool nochrono, barrier, rank_round_robin, keep_Vppph, no_checkpoint, blocking,
    complex, single, cT, ijkabc;
#if defined(HAVE_CTF)
  bool ei_ctf, ea_ctf, Tph_ctf, Tpphh_ctf, Vpphh_ctf, Vhhhp_ctf, Vppph_ctf,
      Jppph_ctf, Jhhhp_ctf;
#endif /*   defined(HAVE_CTF) */
  std::string tuples_distribution_string, checkpoint_path;
  // paths
  std::string ei_path, ea_path, Tph_path, Tpphh_path, Vpphh_path, Vhhhp_path,
      Vppph_path, Jppph_path, Jhhhp_path;
};

template <typename FIELD>
void run(int argc, char **argv, Settings const &s) {

  MPI_Init(&argc, &argv);
#if defined(HAVE_CTF)
  CTF::World world(argc, argv);
  MPI_Comm comm = world.comm;
#else
  MPI_Comm comm = MPI_COMM_WORLD;
#endif /* defined(HAVE_CTF) */
  int rank, nranks;
  MPI_Comm_rank(comm, &rank);
  MPI_Comm_size(comm, &nranks);
  int no = s.no, nv = s.nv;

  auto in = atrip::Atrip::Input<FIELD>()
                .with_delete_Vppph(!s.keep_Vppph)
                .with_barrier(s.barrier)
                .with_blocking(s.blocking)
                .with_chrono(!s.nochrono)
                .with_rank_round_robin(s.rank_round_robin)
                .with_iteration_mod(s.it_mod)
                .with_percentage_mod(s.percentage_mod)
                .with_max_iterations(s.max_iterations)
                .with_checkpoint_at_every_iteration(s.checkpoint_it)
                .with_checkpoint_at_percentage(s.checkpoint_percentage)
                .with_checkpoint_path(s.checkpoint_path)
                .with_read_checkpoint_if_exists(!s.no_checkpoint)
                .with_ijkabc(s.ijkabc);

  constexpr double elem_to_gb = 8.0 / 1024.0 / 1024.0 / 1024.0;
  if (s.ijkabc) { _flip(no, nv); }

  // USER PRINTING TEST BEGIN
  const double doubles_flops = no * no * no // common parts of the matrices
                             * (no + nv)    // particles and holes
                             * (s.complex ? 4.0 : 1.0)
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

      f = s.complex ? sizeof(double) : sizeof(atrip::Complex),

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

#if defined(HAVE_CTF)
  std::vector<int> symmetries(4, NS);
#endif /* defined(HAVE_CTF) */

  std::vector<int>

      vo({nv, no}), vvoo({nv, nv, no, no}), ooov({no, no, no, nv}),
      vvvo({nv, nv, nv, no}), ovoo({no, nv, no, no});

  _print_size(Vabci, no * nv * nv * nv);
  _print_size(Vabij, no * no * nv * nv);
  _print_size(Vijka, no * no * no * nv);

  std::vector<FIELD> *epsi = get_epsilon<FIELD>(s.ei_path, no, comm),
                     *epsa = get_epsilon<FIELD>(s.ea_path, nv, comm);
  in.with_epsilon_i(epsi).with_epsilon_a(epsa);

  MPI_Barrier(comm);

  // For the printing we work with the 'correct' definition of no && nv
  if (s.ijkabc) { _flip(no, nv); }
  if (!rank) {
    std::cout << "np " << nranks << std::endl;
    for (auto const &fn : input_printer)
      // print input parameters
      fn();
    if (s.ijkabc)
      std::cout << "ijkabc used, we flip No && Nv internally" << std::endl;
  }
  if (s.ijkabc) { _flip(no, nv); }

  atrip::Atrip::init(comm);

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
  in.with_tuples_distribution(tuples_distribution);

  /* We use the s.notation p = v and q = o for the initial load of T1&T2 */
  /* If we use the p <-> h algorithm we will flip the T-amplitudes     */
  std::vector<int> pq({nv, no}), ppqq({nv, nv, no, no});
  if (s.ijkabc) {
    pq = {no, nv};
    ppqq = {no, no, nv, nv};
  }

  std::vector<FIELD> *Tph = new std::vector<FIELD>(no * nv, 0.1);

#if defined(HAVE_CTF)
  CTF::Tensor<FIELD> *Tpphh = nullptr, *Vpphh = nullptr;
  CTF::Tensor<FIELD> /**iTph = read_or_fill<FIELD>("tph",
                                                 2,
                                                 pq.data(),
                                                 symmetries.data(),
                                                 world,
                                                 s.Tph_path,
                                                 0.0,
                                                 1.0),*/
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
    Tpphh = new CTF::Tensor<FIELD>(4, vvoo.data(), symmetries.data(), world);
    Vpphh = new CTF::Tensor<FIELD>(4, vvoo.data(), symmetries.data(), world);

    (*Tpphh)["ijab"] = (*iTpphh)["abij"];
    (*Vpphh)["ijab"] = (*iVpphh)["abij"];
  } else {
    Tpphh = iTpphh;
    Vpphh = iVpphh;
  }
#endif /*   defined(HAVE_CTF) */

#if defined(HAVE_CTF)
  if (s.Vpphh_ctf) {
    in.with_Vpphh(Vpphh);
  } else {
    in.with_Vpphh_path(s.Vpphh_path);
  }

  if (s.Tpphh_ctf) {
    in.with_Tpphh(Tpphh);
  } else {
    in.with_Tpphh_path(s.Tpphh_path);
  }
#endif /* defined(HAVE_CTF) */

#if defined(HAVE_CTF)
  CTF::Tensor<FIELD> *Jppph = nullptr, *Jhhhp = nullptr;
  if (s.cT || (s.Jppph_path.size() && s.Jhhhp_path.size())) {
    if (!rank) std::cout << "doing cT" << std::endl;
    // TODO: do it with read_or_fill
    if (s.Jhhhp_ctf) {
      /**/
      /**/
      /**/
      Jhhhp = read_or_fill<FIELD>("Jhhhp",
                                  4,
                                  ooov.data(),
                                  symmetries.data(),
                                  world,
                                  s.Jhhhp_path,
                                  0,
                                  1);
      in.with_Jhhhp(Jhhhp);
    } else {
      in.with_Jhhhp_path(s.Jhhhp_path);
    }
    /**/
    /**/
    /**/
    if (s.Jppph_ctf) {
      Jppph = read_or_fill<FIELD>("Jppph",
                                  4,
                                  vvvo.data(),
                                  symmetries.data(),
                                  world,
                                  s.Jppph_path,
                                  0,
                                  1);
      in.with_Jppph(Jppph);
    } else {
      in.with_Jppph_path(s.Jppph_path);
    }
  }

  if (s.Vhhhp_ctf) {
    in.with_Vhhhp(read_or_fill<FIELD>("Vhhhp",
                                      4,
                                      ooov.data(),
                                      symmetries.data(),
                                      world,
                                      s.Vhhhp_path,
                                      0,
                                      1));
  } else {
    in.with_Vhhhp_path(s.Vhhhp_path);
  }

  if (s.Vppph_ctf) {
    in.with_Vppph(read_or_fill<FIELD>("Vppph",
                                      4,
                                      vvvo.data(),
                                      symmetries.data(),
                                      world,
                                      s.Vppph_path,
                                      0,
                                      1));
  } else {
    in.with_Vppph_path(s.Vppph_path);
  }
#endif

  if (s.Tph_path.size()) {
    if (!rank) std::cout << "Reading Tai\n";
    *Tph = atrip::read_all<FIELD>({nv, no}, s.Tph_path, comm);
  }
  in.with_Tph(Tph);

#if !defined(HAVE_CTF)
  in.with_Vpphh_path(s.Vpphh_path);
  in.with_Tpphh_path(s.Tpphh_path);

  in.with_Jhhhp_path(s.Jhhhp_path);
  in.with_Jppph_path(s.Jppph_path);
  in.with_Vhhhp_path(s.Vhhhp_path);
  in.with_Vppph_path(s.Vppph_path);
#endif /* defined(HAVE_CTF) */

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
  defflag(app, "--single", s.single, "Use single precision algorithm")
      ->default_val(false);
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

  // completeTriples
  //
  defflag(app, "--cT", s.cT, "Perform (cT) calculation")->default_val(false);
  defoption(app, "--Jppph", s.Jppph_path, "Path for Jppph intermediates")
      ->check(CLI::ExistingFile);
  defoption(app, "--Jhhhp", s.Jhhhp_path, "Path for Jhhhp intermediates")
      ->check(CLI::ExistingFile);

#if defined(HAVE_CTF)
  // Use reader from ctf or not
  defflag(app, "--ei-from-ctf", s.ei_ctf, "Read using CTF the HF energies ε_i")
      ->default_val(true);
  defflag(app, "--ea-from-ctf", s.ea_ctf, "Read using CTF the HF energies ε_a")
      ->default_val(true);
  defflag(app,
          "--Tpphh-from-ctf",
          s.Tpphh_ctf,
          "Read using CTF the Tpphh (Tabij)")
      ->default_val(true);
  defflag(app, "--Tph-from-ctf", s.Tph_ctf, "Read using CTF the Tph (Tai)")
      ->default_val(true);
  defflag(app,
          "--Vpphh-from-ctf",
          s.Vpphh_ctf,
          "Read using CTF the Vpphh (Vabij)")
      ->default_val(true);
  defflag(app,
          "--Vhhhp-from-ctf",
          s.Vhhhp_ctf,
          "Read using CTF the Vhhhp (Vijka)")
      ->default_val(true);
  defflag(app,
          "--Vppph-from-ctf",
          s.Vppph_ctf,
          "Read using CTF the Vppph (Vabci)")
      ->default_val(true);
  defflag(app,
          "--Jppph-from-ctf",
          s.Jppph_ctf,
          "Read using CTF for Jppph intermediates")
      ->default_val(true);
  defflag(app,
          "--Jhhhp-from-ctf",
          s.Jhhhp_ctf,
          "Read using CTF for Jhhhp intermediates")
      ->default_val(true);
#endif /* defined(HAVE_CTF) */

  CLI11_PARSE(app, argc, argv);

  if (s.complex) {
    if (s.single) {
      // run<std::complex<float>>(argc, argv, s);
      throw "Not implemented";
    } else {
      run<atrip::Complex>(argc, argv, s);
    }
  } else {
    if (s.single) {
      run<float>(argc, argv, s);
    } else {
      run<double>(argc, argv, s);
    }
  }

  return 0;
}

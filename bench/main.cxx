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
#include <atrip/RiskReader.hpp>

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
std::vector<F> generate_random_vector(size_t N, F min = 0.0, F max = 1.0) {
  std::vector<F> result(N);
  std::random_device rd;                           // Non-deterministic random seed
  std::mt19937 gen(rd());                          // Mersenne Twister engine
  std::uniform_real_distribution<F> dist(min, max); // Uniform distribution in [min, max)

  for (size_t i = 0; i < N; ++i) result[i] = dist(gen);

  return result;
}


template <typename F>
std::vector<F> *
get_epsilon(std::string const &path, size_t const len, MPI_Comm comm, bool sign) {
  std::vector<F> *result = nullptr;
  F max = (sign) ? F(1) : F(-1);
  if (path.size()) {
    result = new std::vector<F>(atrip::read_all<F>({len}, path, comm));
  } else {
    result = new std::vector<F>(generate_random_vector<F>(len, F(0), max));
  }
  return result;
}

template std::vector<double> *
get_epsilon<double>(std::string const &path, size_t const len, MPI_Comm comm, bool sign);

template <>
std::vector<atrip::Complex> *
get_epsilon<atrip::Complex>(std::string const &path,
                            size_t const len,
                            MPI_Comm comm,
                            bool sign) {
  std::vector<double> *real = get_epsilon<double>(path, len, comm, sign);
  auto result = new std::vector<atrip::Complex>(len);
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

  auto tsr = new CTF::Tensor<F>(order, lens, syms, world, name.c_str());
  if (path.size()) {
    if (!atrip::Atrip::rank)
      std::cout << "Read tensor data from file " << path << std::endl;
    tsr->read_dense_from_file(path.c_str());
  } else {
    if (!atrip::Atrip::rank)
      std::cout << "Random initialization for tensor " << name << std::endl;
    tsr->fill_random(a, b);
  }

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
  bool Tph_ctf, Tpphh_ctf, Vpphh_ctf, Vhhhp_ctf, Vppph_ctf,
      Jppph_ctf, Jhhhp_ctf, use_ctf;
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
  if (no == 0) no = 10;
  if (nv == 0) nv = 100;

  const auto file_exists = [](std::string const &filename) {
    ifstream file(filename.c_str());
    return file.good();
  };

  if (s.ei_path.size() && s.ea_path.size()) {
    if (!rank) std::cout << "EigenEnergies provided - system dimensions"
                         << "will be inferred from the file size." << std::endl;
    ifstream ifile(s.ei_path, ios::in|ios::binary);
    ifstream afile(s.ea_path, ios::in|ios::binary);
    ifile.seekg(0, std::ios::end);
    size_t lengthI = ifile.tellg();
    afile.seekg(0, std::ios::end);
    size_t lengthA = afile.tellg();
    no = lengthI / sizeof(double);
    nv = lengthA / sizeof(double);
  }

  auto check_file = [rank,no,nv](std::string const &filename, std::vector<int> dims) {
    size_t els(1);
    for (size_t i: dims) els *= i;
    els *= sizeof(FIELD);
    ifstream file(filename.c_str());
    file.seekg(0, std::ios::end);
    size_t length = file.tellg();
    if (length == els) return;
    if (!rank) std::cout << "File size of " << filename
                         << " is not consistent with the dimensions:\n"
                         << "no: " << no << " , nv: " << nv << " !!" << std::endl;
    MPI_Finalize();
    return;
  };


  if (file_exists(s.Tpphh_path)) check_file(s.Tpphh_path, {nv, nv, no, no});
  if (file_exists(s.Tph_path))   check_file(s.Tph_path,   {nv, no});
  if (file_exists(s.Vpphh_path)) check_file(s.Vpphh_path, {nv, nv, no, no});
  if (file_exists(s.Vppph_path)) check_file(s.Vppph_path, {nv, nv, nv, no});
  if (file_exists(s.Vhhhp_path)) check_file(s.Vhhhp_path, {no, no, no, nv});
  if (file_exists(s.Jppph_path)) check_file(s.Jppph_path, {nv, nv, nv, no});
  if (file_exists(s.Jhhhp_path)) check_file(s.Jhhhp_path, {no, no, no, nv});

  if (s.use_ctf == false) {
    bool allfiles(file_exists(s.ei_path)    &&
                  file_exists(s.ea_path)    &&
                  file_exists(s.Tpphh_path) &&
                  file_exists(s.Tph_path)   &&
                  file_exists(s.Vpphh_path) &&
                  file_exists(s.Vhhhp_path) &&
                  file_exists(s.Vppph_path)
                 );
    if (allfiles == false) {
      if (!rank) std::cout << "If working without CTF all files have to be present!" << std::endl;
      MPI_Finalize();
      return;
    }
  }


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


  std::vector<int>

      vo({nv, no}), vvoo({nv, nv, no, no}), ooov({no, no, no, nv}),
      vvvo({nv, nv, nv, no}), ovoo({no, nv, no, no});

  _print_size(Vabci, no * nv * nv * nv);
  _print_size(Vabij, no * no * nv * nv);
  _print_size(Vijka, no * no * no * nv);

  //this is a hack because there is an issue
  MPI_Comm_rank(MPI_COMM_WORLD, (int *)&atrip::Atrip::rank);

  std::vector<FIELD> *epsi = get_epsilon<FIELD>(s.ei_path, no, comm, false),
                     *epsa = get_epsilon<FIELD>(s.ea_path, nv, comm, true);
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

  // split the communicator and let only every n-th rank be part of the game

  int omp_threads = 4; // for example
  int color = (rank % omp_threads == 0) ? 1 : MPI_UNDEFINED;


  //MPI_Comm atrip_comm;
  //MPI_Comm_split(comm, color, rank, &atrip_comm);

  //if (color == 1) {
    //atrip::Atrip::init(atrip_comm);
    atrip::Atrip::init(s.rank_round_robin, comm);
  //}
  //
  //


  size_t const

      f = s.complex ? sizeof(double) : sizeof(atrip::Complex),

      n_tuples =

          nv * (nv + 1) * (nv + 2) / 6 - nv, // All tuples

      atrip_memory =

          3 * sizeof(size_t) * n_tuples // tuples_memory

          //
          // one dimensional slices (all ranks)
          //

          + f * atrip::Atrip::np * 6 * nv * no * no // taphh
          + f * atrip::Atrip::np * 6 * no * no * no // hhha

          //
          // two dimensional slices (all ranks)
          //

          + f * atrip::Atrip::np * 12 * nv * no // abph
          + f * atrip::Atrip::np * 6 * no * no  // abhh
          + f * atrip::Atrip::np * 6 * no * no  // tabhh

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

          + f * atrip::Atrip::np * no * no * no // tijk
          + f * atrip::Atrip::np * no * no * no // zijk
          + f * atrip::Atrip::np * (no + nv)    // epsp
          + f * atrip::Atrip::np * no * nv      // tai
      ;                               // end

  if (rank == 0) {
    std::cout << "Tentative MEMORY USAGE (GB): "
              << double(atrip_memory) / 1024.0 / 1024.0 / 1024.0 << "\n";
  }

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

/*
  //TODO
  //if (P<->H) Switch Tph //
  if (s.ijkabc) {
    Tpphh = new CTF::Tensor<FIELD>(4, vvoo.data(), symmetries.data(), world);
    Vpphh = new CTF::Tensor<FIELD>(4, vvoo.data(), symmetries.data(), world);

    (*Tpphh)["ijab"] = (*iTpphh)["abij"];
    (*Vpphh)["ijab"] = (*iVpphh)["abij"];
  } else {
    Tpphh = iTpphh;
    Vpphh = iVpphh;
  o}
*/

  std::vector<FIELD> *Tph = new std::vector<FIELD>(no * nv, 0.1);
  if (s.Tph_path.size()) {
    *Tph = atrip::read_all<FIELD>({nv, no}, s.Tph_path, comm);
  }
  in.with_Tph(Tph);

  void *Tph_   = nullptr;
  void *Tpphh_ = nullptr;
  void *Vppph_ = nullptr;
  void *Vpphh_ = nullptr;
  void *Vhhhp_ = nullptr;
  void *Jhhhp_ = nullptr;
  void *Jppph_ = nullptr;
#if defined(HAVE_CTF)
  int syms[] = {NS, NS, NS, NS};
  if (s.use_ctf) {
    Tph_   = read_or_fill<FIELD>("Tph", 2, vo.data(), syms, world, s.Tph_path, 0, 1);
    Tpphh_ = read_or_fill<FIELD>("Tpphh", 4, vvoo.data(), syms, world, s.Tpphh_path, 0, 1);
    Vppph_ = read_or_fill<FIELD>("Vppph", 4, vvvo.data(), syms, world, s.Vppph_path, 0, 1);
    Vpphh_ = read_or_fill<FIELD>("Vpphh", 4, vvoo.data(), syms, world, s.Vpphh_path, 0, 1);
    Vhhhp_ = read_or_fill<FIELD>("Vhhhp", 4, ooov.data(), syms, world, s.Vhhhp_path, 0, 1);
    if (s.cT) {
      Jppph_ = read_or_fill<FIELD>("Jppph", 4, vvvo.data(), syms, world, s.Jppph_path, 0, 1);
      Jhhhp_ = read_or_fill<FIELD>("Jhhhp", 4, ooov.data(), syms, world, s.Jhhhp_path, 0, 1);
    }
  }
#endif

  auto sVabph = atrip::newReader<FIELD>(Vppph_,
                                        {nv, nv, nv, no},
                                        {1, 1, 0, 0},
                                        no,
                                        nv,
                                        s.Vppph_path,
                                        true);

  auto sVabhh = atrip::newReader<FIELD>(Vpphh_,
                                        {nv, nv, no, no},
                                        {1, 1, 0, 0},
                                        no,
                                        nv,
                                        s.Vpphh_path,
                                        true);

  auto sTabhh = atrip::newReader<FIELD>(Tpphh_,
                                        {nv, nv, no, no},
                                        {1, 1, 0, 0},
                                        no,
                                        nv,
                                        s.Tpphh_path,
                                        true);

  auto sTaphh = atrip::newReader<FIELD>(Tpphh_,
                                        {nv, nv, no, no},
                                        {1, 0, 0, 0},
                                        no,
                                        nv,
                                        s.Tpphh_path,
                                        true);

  auto sVhhha = atrip::newReader<FIELD>(Vhhhp_,
                                        {no, no, no, nv},
                                        {0, 0, 0, 1},
                                        no,
                                        nv,
                                        s.Vhhhp_path,
                                        false);


  in.with_sVabph(&sVabph);
  in.with_sVabhh(&sVabhh);
  in.with_sTabhh(&sTabhh);
  in.with_sTaphh(&sTaphh);
  in.with_sVhhha(&sVhhha);

  //this will not work as sJabph will go out of scope...i will have to move it
  if (s.cT) {
    auto* sJabph = new atrip::Sources<FIELD>(
      atrip::newReader<FIELD>(Jppph_,
                              {nv, nv, nv, no},
                              {1, 1, 0, 0},
                              no,
                              nv,
                              s.Jppph_path,
                              true)
    );

    auto* sJhhha = new atrip::Sources<FIELD>(
      atrip::newReader<FIELD>(Jhhhp_,
                              {no, no, no, nv},
                              {0, 0, 0, 1},
                              no,
                              nv,
                              s.Jhhhp_path,
                              false)
    );


    in.with_sJabph(sJabph);
    in.with_sJhhha(sJhhha);
  }

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
      ->default_val(0)
      ->check(CLI::PositiveNumber);
//      ->required();
  defoption(app, "--nv", s.nv, "Number of Virtual orbitals")
      ->default_val(0)
      ->check(CLI::PositiveNumber);
//      ->required();
  defoption(app, "--dist", s.tuples_distribution_string, "Tuples distribution")
      ->default_val("group")
      ->check(CLI::IsMember({"group", "naive"}));
//      ->required();

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
  defflag(app,
          "--use_ctf",
          s.use_ctf,
          "Read tensors using CTF")
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

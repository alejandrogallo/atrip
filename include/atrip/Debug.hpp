// [[file:../../atrip.org::*Debug][Debug:1]]
#pragma once
#define ATRIP_BENCHMARK
//#define ATRIP_DONT_SLICE
//#define ATRIP_WORKLOAD_DUMP
#define ATRIP_USE_DGEMM
//#define ATRIP_PRINT_TUPLES

#ifndef ATRIP_DEBUG
#define ATRIP_DEBUG 1
#endif

#ifndef LOG
#define LOG(level, name) if (Atrip::rank == 0) std::cout << name << ": "
#endif

#if ATRIP_DEBUG == 4
#  pragma message("WARNING: You have OCD debugging ABC triples "\
                  "expect GB of output and consult your therapist")
#  include <dbg.h>
#  define HAVE_OCD
#  define OCD_Barrier(com) MPI_Barrier(com)
#  define WITH_OCD
#  define WITH_ROOT if (atrip::Atrip::rank == 0)
#  define WITH_SPECIAL(r) if (atrip::Atrip::rank == r)
#  define WITH_RANK std::cout << atrip::Atrip::rank << ": "
#  define WITH_CRAZY_DEBUG
#  define WITH_DBG
#  define DBG(...) dbg(__VA_ARGS__)
#elif ATRIP_DEBUG == 3
#  pragma message("WARNING: You have crazy debugging ABC triples,"\
                  " expect GB of output")
#  include <dbg.h>
#  define OCD_Barrier(com)
#  define WITH_OCD if (false)
#  define WITH_ROOT if (atrip::Atrip::rank == 0)
#  define WITH_SPECIAL(r) if (atrip::Atrip::rank == r)
#  define WITH_RANK std::cout << atrip::Atrip::rank << ": "
#  define WITH_CRAZY_DEBUG
#  define WITH_DBG
#  define DBG(...) dbg(__VA_ARGS__)
#elif ATRIP_DEBUG == 2
#  pragma message("WARNING: You have some debugging info for ABC triples")
#  include <dbg.h>
#  define OCD_Barrier(com)
#  define WITH_OCD if (false)
#  define WITH_ROOT if (atrip::Atrip::rank == 0)
#  define WITH_SPECIAL(r) if (atrip::Atrip::rank == r)
#  define WITH_RANK std::cout << atrip::Atrip::rank << ": "
#  define WITH_CRAZY_DEBUG if (false)
#  define WITH_DBG
#  define DBG(...) dbg(__VA_ARGS__)
#else
#  define OCD_Barrier(com)
#  define WITH_OCD if (false)
#  define WITH_ROOT if (false)
#  define WITH_SPECIAL(r) if (false)
#  define WITH_RANK if (false) std::cout << atrip::Atrip::rank << ": "
#  define WITH_DBG if (false)
#  define WITH_CRAZY_DEBUG if (false)
#  define DBG(...)
#endif
// Debug:1 ends here

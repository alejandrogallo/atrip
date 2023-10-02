dnl  This is an autoconf script.
dnl  To rebuild the `configure' script from this, execute the command
dnl 	autoconf
dnl  in the directory containing this script.
dnl

AC_INIT([atrip],[0.0.0],[https://gihtub.com/alejandrogallo/atrip])
AC_CONFIG_SRCDIR([configure.ac])
AC_PREREQ(2.69)
AC_CONFIG_MACRO_DIRS([./etc/m4])
AC_CONFIG_AUX_DIR([./etc/autotools])
AM_INIT_AUTOMAKE([foreign -Wall])


dnl -----------------------------------------------------------------------
dnl Configure switches

AC_ARG_ENABLE(shared,
              AS_HELP_STRING([--enable-shared],
                             [enable loading compiled
                                     files (default=YES)]),
              [], [enable_shared=yes])


AC_ARG_ENABLE([docs],
              [AS_HELP_STRING([--enable-docs],
                              [Enable building docs])],
              [build_docs=YES], [build_docs=NO])

dnl LIBCTTF library options

AC_ARG_WITH(ctf,
            [AS_HELP_STRING([--with-ctf],
                           [prefix for CTF includes and libraries] )],
            [LIBCTF_PATH="`readlink -f $withval`";
             LIBCTF_CPATH="`readlink -f $withval`/include";
             LIBCTF_LD_LIBRARY_PATH="`readlink -f $withval`/lib";
             BUILD_CTF=NO],
            [BUILD_CTF=YES
             LIBCTF_PATH="$ac_pwd"/extern/ctf
             LIBCTF_CPATH="${LIBCTF_PATH}/include"
             LIBCTF_LD_LIBRARY_PATH="${LIBCTF_PATH}/lib"
             ])

AC_ARG_WITH([clang-check],
            [AS_HELP_STRING([--with-clang-check],
                            [Check with clang-check])],
            [clang_check=YES;
             AC_CHECK_PROG([CLANG_CHECK], [clang-check])],
            [clang_check=NO])
AM_CONDITIONAL([WITH_CLANG_CHECK], [test x${clang_check} = xYES])


dnl CUDA ----

AC_ARG_ENABLE([cuda],
              [AS_HELP_STRING([--enable-cuda],
                              [Build with cuda])],
                              [WITH_CUDA=yes],
                              [WITH_CUDA=no])
AC_ARG_VAR([NVCC], [Path to the nvidia cuda compiler.])
AC_ARG_VAR([CUDA_LDFLAGS], [LDFLAGS to find libraries -lcuda, -lcudart, -lcublas.])
AC_ARG_VAR([CUDA_CXXFLAGS], [CXXFLAGS to find the CUDA headers])

dnl HIP ----
AC_ARG_ENABLE([hip],
              [AS_HELP_STRING([--enable-hip],
                              [Build with hip])],
                              [WITH_HIP=yes],
                              [WITH_HIP=no])
AC_ARG_VAR([HIPCC], [Path to the nvidia hip compiler.])
AC_ARG_VAR([HIP_LDFLAGS], [LDFLAGS for the HIPCC compiler])
AC_ARG_VAR([HIP_CXXFLAGS], [CXXFLAGS to find the HIP headers])

dnl profile ----

AC_ARG_ENABLE([profiler],
              [AS_HELP_STRING([--enable-profiler],
                              [Enable profiler tools for HIP or CUDA])],
                              [WITH_PROFILER=yes],
                              [WITH_PROFILER=no])
AC_ARG_VAR([PROFILER_PATH], [Path to the profiler include and lib directories])

dnl -----------------------------------------------------------------------
dnl ATRIP CPP DEFINES
dnl -----------------------------------------------------------------------

AC_ARG_WITH([atrip-debug],
            [AS_HELP_STRING([--with-atrip-debug],
                            [Debug level for atrip, possible values:
                                   1, 2, 3, 4])],
            [AC_DEFINE([ATRIP_DEBUG],[atrip-debug],[Atrip debug level])],
            [AC_DEFINE([ATRIP_DEBUG],[1],[Atrip debug level])])


AC_ARG_ENABLE([atrip_dgemm],
              [AS_HELP_STRING([--disable-dgemm],
                              [Disable using dgemm for the doubles equations])],
              [],
              [AC_DEFINE([ATRIP_USE_DGEMM],
                         1,
                         [Use dgemm for the doubles equations])])

ATRIP_DEF([slice], [disable],
          [ATRIP_DONT_SLICE],
          [Disable the step of slicing tensors for CTF, this is useful
          for example for benchmarking or testing.])

ATRIP_DEF([only-dgemm], [enable],
          [ATRIP_ONLY_DGEMM],
          [Run only the parts of atrip that involve dgemm calls, this
          is useful for benchmarking and testing the code, it is
          intended for developers of Atrip.])

ATRIP_DEF([naive-slow], [enable],
          [ATRIP_NAIVE_SLOW],
          [Run slow but correct code for the mapping of (iteration,
          rank) to tuple of the naive tuple distribution.])

ATRIP_DEF([sources-in-gpu], [enable],
          [ATRIP_SOURCES_IN_GPU],
          [When using CUDA, activate storing all sources (slices of
          the input tensors) in the GPU. This means that a lot of GPUs
          will be needed.])

ATRIP_DEF([cuda-aware-mpi], [enable],
          [ATRIP_CUDA_AWARE_MPI],
          [When using MPI, assume support for CUDA aware mpi by the
          given MPI implementation.])

ATRIP_DEF([mpi-staging-buffers], [enable],
          [ATRIP_MPI_STAGING_BUFFERS],
          [Enable mpi staging buffers int the GPU code. Every source
           will first be copied to this buffer and then subsequently
           sent.
           You should probably enable ATRIP_ALLOCATE_ADDITIONAL_FREE_POINTERS
           since the additional buffers are free pointers and they might
           run out, so dynamic allocation of free buffers is necessary.])

ATRIP_DEF([allocate-additional-free-pointers], [enable],
          [ATRIP_ALLOCATE_ADDITIONAL_FREE_POINTERS],
          [Enable the allocation of additional free buffers at runtime
           if the free buffer pool runs empty.])

ATRIP_DEF([print-tuples], [enable],
          [ATRIP_PRINT_TUPLES],
          [Print the tuple energies for each (abc) tuple at then end of the run.])


dnl -----------------------------------------------------------------------

ATRIP_LDFLAGS="-L$PWD/src -latrip"
AC_SUBST(ATRIP_LDFLAGS)

dnl -----------------------------------------------------------------------
dnl CTF EXPORT ------------------------------------------------------------
dnl -----------------------------------------------------------------------
AM_CONDITIONAL([WITH_BUILD_CTF], [test x${BUILD_CTF} = xYES])
CTF_COMMIT="53ae5daad851bf3b198ebe1fa761c13b12291116"
AC_SUBST([CTF_COMMIT])
AC_SUBST(LIBCTF_PATH)
AC_SUBST(LIBCTF_CPATH)
AC_SUBST(LIBCTF_LD_LIBRARY_PATH)

dnl -----------------------------------------------------------------------
AC_MSG_NOTICE([

              CHECKING YOUR OVERALL SYSTEM ...

               ])
AC_CANONICAL_BUILD
AC_CANONICAL_HOST

dnl Guess compilers
AC_LANG([C++])
AC_MSG_NOTICE([
               CHECKING COMPILERS...
               ])
AC_PROG_CXX
AX_MPI([], [AC_MSG_FAILURE([No MPI C++ compiler found.])])
AX_COMPILER_VENDOR
CXX="${MPICXX}"
ATRIP_MPI=$(${MPICXX} -show | awk '{$1=""; print $NL}' | sed "s/^ //")
AC_SUBST([ATRIP_MPI])

AM_PROG_AR
AC_PROG_CPP           dnl sets variable CPP
AC_PROG_RANLIB        dnl sets variable RANLIB
AC_PROG_INSTALL       dnl sets variables INSTALL, INSTALL_DATA, INSTALL_PROGRAM
AC_PROG_LN_S          dnl sets variable LN_S
AC_PROG_MAKE_SET      dnl set $(MAKE) if needed
AC_PROG_RANLIB        dnl set variable RANLIB
AC_CHECK_TOOL([AR],[ar]) dnl set variable AR to appropriate `ar' program

dnl -----------------------------------------------------------------------
AC_OPENMP
AX_APPEND_COMPILE_FLAGS([-pedantic -Wall -Wpedantic  -Wextra])
AX_APPEND_COMPILE_FLAGS([-pthread])
CXXFLAGS="${CXXFLAGS} ${OPENMP_CXXFLAGS}"
LDFLAGS="${LDFLAGS} ${OPENMP_CXXFLAGS}"

dnl -----------------------------------------------------------------------
AX_BLAS([LIBS="${LIBS} ${BLAS_LIBS}"],
        [AC_MSG_FAILURE([Could not find BLAS library.])])
AX_LAPACK([LIBS="${LIBS} ${LAPACK_LIBS}"],
          [AC_MSG_FAILURE([Could not find LAPACK library.])])
SISI_SCALAPACK([LIBS="${LIBS} ${SCALAPACK_LIBS}"],
               [AC_MSG_FAILURE([could not find a working scalapack installation.])])

dnl -----------------------------------------------------------------------
dnl CXX checks
dnl
AX_CXX_STATIC_CAST
AX_CXX_TEMPLATES
AX_CXX_TYPENAME
AX_CXX_COMPILE_STDCXX([11], [noext], [mandatory])
AC_TYPE_SIZE_T

dnl -----------------------------------------------------------------------
dnl CHECK CTF
if test xYES = x${BUILD_CTF}; then
AC_MSG_WARN([

             You will have to do make ctf before building the project.

    ])
else
  CPPFLAGS="$CPPFLAGS -I${LIBCTF_CPATH}"
  LDFLAGS="$LDFLAGS -L${LIBCTF_LD_LIBRARY_PATH} -lctf"
  AC_CHECK_HEADER([ctf.hpp],
                  [],
                  [AC_MSG_ERROR([no ctf.hpp])])
fi


dnl CUDA NVIDIA -----------------------------------------------------------

AM_CONDITIONAL([WITH_CUDA], [test x${WITH_CUDA} = xyes])
if test x${WITH_CUDA} = xyes; then
  AC_MSG_RESULT([
                     CUDA SUPPORT IS ENABLED (NVIDIA)
                     --------------------------------
  ])
  AC_CHECK_PROGS([NVCC], [nvcc])
  AC_SUBST([CUDA_LDFLAGS])
  AC_DEFINE([HAVE_CUDA],1,[AMD CUDA support flag])
  # TODO: make sure to find cuda and cudart
  # AC_SEARCH_LIBS([cudaMalloc], [cuda cudart], [FOUND_CUDA=yes])
  ATRIP_CUBLAS
fi

dnl HIP AMD -----------------------------------------------------------

AM_CONDITIONAL([WITH_HIP], [test x${WITH_HIP} = xyes])
if test x${WITH_HIP} = xyes; then
  AC_MSG_RESULT([
                     HIP SUPPORT IS ENABLED AMD
                     --------------------------
  ])
  AC_SUBST([HIPBLAS])
  AC_CHECK_PROGS([HIPCC], [hipcc])
  AC_SUBST([HIP_LDFLAGS])
  CPPFLAGS="$CPPFLAGS -I$HIPBLAS/include"
  LDFLAGS="$LDFLAGS -lamdhip64 -lhipblas"
  AC_DEFINE([HAVE_HIP],1,[Wether we are using HIP])
  AC_DEFINE([HOP_TARGET_HIP],1,[Wether we are using HIP])
fi

dnl PROFILER -----------------------------------------------------------

if test x${WITH_PROFILER} = xyes; then
  AC_DEFINE([WITH_PROFILER],1,[Use profiler for HIP or CUDA])
  AC_MSG_RESULT([
                     Checking profilers
                     ------------------
  ])
  if test x${WITH_HIP} = xyes; then
  LDFLAGS="$LDFLAGS -L${PROFILER_PATH}/lib"
  AC_CHECK_HEADER([omnitrace/user.h],
		  [],
		  [AC_MSG_ERROR([No omnitrace/user.h header library found, set the PROFILER_PATH variable maybe])])
  AC_CHECK_LIB([omnitrace-user],
               [omnitrace_user_push_region],
               [],
               [AC_MSG_ERROR([No omnitrace-user library found, set the PROFILER_PATH variable maybe])])
  fi
fi

dnl HIP AMD -----------------------------------------------------------

if test x${WITH_CUDA} = xyes -o x${WITH_HIP} = xyes; then
  WITH_ACC=yes
else
  WITH_ACC=no
fi
AM_CONDITIONAL([WITH_ACC], [test x${WITH_ACC} = xyes])
if test x${WITH_ACC} = xyes; then
  AC_DEFINE([HAVE_ACC],
            1,
            [Code with accelerator support])
fi

dnl -----------------------------------------------------------------------
dnl Docs
if test x${build_docs} = xYES; then
  AC_CHECK_PROGS([DOXYGEN], [doxygen])
  AC_CHECK_PROGS([DOT], [dot])
  AX_PROG_EMACS
  AX_ELISP_CHECK_FEATURE([org])
  AX_ELISP_CHECK_FBOUNDP([package-initialize], [package])
  AX_ELISP_CHECK_FBOUNDP([package-install], [package])
  AX_ELISP_CHECK_FBOUNDP([string-join], [subr-x])
  AX_ELISP_CHECK_BOUNDP([org-babel-post-tangle-hook], [ob])
fi


dnl -----------------------------------------------------------------------
AC_CONFIG_FILES([Makefile
                 src/Makefile
                 .dir-locals.el:tools/dir-locals.el.in
                 bench/Makefile
                 extern/Makefile
                 extern/ctf/Makefile
                 test/Makefile
                 integration-tests/Makefile
                 integration-tests/run-h2o.sh
                 docs/Makefile
                 docs/conf.py:docs/conf.py
                 docs/Doxyfile:docs/Doxyfile
               ])
AC_CONFIG_HEADERS([config.h])


AC_MSG_RESULT([
                       AUTOMATIC CONFIGURATION
                       -----------------------
                       Vendor   = $ax_cv_cxx_compiler_vendor
                       CXX      = $CXX
                       MPICXX   = $MPICXX
                       CXXFLAGS = $CXXFLAGS
                       CPPFLAGS = $CPPFLAGS
                       LDFLAGS  = $LDFLAGS
                       ATRIP_LDFLAGS  = $ATRIP_LDFLAGS
                       BLAS = ${BLAS_LIBS}
                       LIBS = ${LIBS}
])

AC_OUTPUT

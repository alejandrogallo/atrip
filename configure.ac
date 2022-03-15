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


dnl LIBGC library options
AC_ARG_WITH(libctf-prefix,
            AS_HELP_STRING([--with-libctf-prefix=path],
                           [prefix for CTF includes and libraries] ),
            [LIBCTF_PATH="`readlink -f $withval`";
             LIBCTF_CPATH="`readlink -f $withval`/include";
             LIBCTF_LD_LIBRARY_PATH="`readlink -f $withval`/lib";
             BUILD_CTF=NO],
            [BUILD_CTF=YES])

AC_ARG_WITH([clang-check],
            [AS_HELP_STRING([--with-clang-check],
                            [Check with clang-check])],
            [clang_check=YES;
             AC_CHECK_PROG([CLANG_CHECK], [clang-check])],
            [clang_check=NO])
AM_CONDITIONAL([WITH_CLANG_CHECK], [test x${clang_check} = xYES])

AC_ARG_WITH(dgemm,
            AS_HELP_STRING([--without-dgemm], [Disable dgemm]),
            [with_dgemm=NO],
            [with_dgemm=YES])



dnl -----------------------------------------------------------------------

ATRIP_LDFLAGS="-L$PWD/src -latrip"
AC_SUBST(ATRIP_LDFLAGS)

dnl -----------------------------------------------------------------------
AM_CONDITIONAL([WITH_BUILD_CTF], [test x${BUILD_CTF} = xYES])
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
AX_BLAS([], [AC_MSG_FAILURE([Could not find BLAS library.])])
AX_LAPACK([], [AC_MSG_FAILURE([Could not find LAPACK library.])])

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
  AC_MSG_WARN([Sorry, building CTF not supported yet provide a build path
                      with --with-libctf-prefix=path])
else
  CPPFLAGS="$CPPFLAGS -I${LIBCTF_CPATH}"
  LDFLAGS="$LDFLAGS -L${LIBCTF_LD_LIBRARY_PATH} -lctf"
  AC_CHECK_HEADER([ctf.hpp],
                  [],
                  [AC_MSG_ERROR([no ctf.hpp])])
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
                 bench/Makefile
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

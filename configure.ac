dnl  This is an autoconf script.
dnl  To rebuild the `configure' script from this, execute the command
dnl 	autoconf
dnl  in the directory containing this script.
dnl

AC_INIT([atrip],[0.0.0],[])
AC_CONFIG_SRCDIR([.])
AC_PREREQ(2.69)
AC_LANG([C++])
AC_CONFIG_MACRO_DIR([./etc/m4])
AC_CONFIG_AUX_DIR([./etc/autotools])
AM_INIT_AUTOMAKE([foreign -Wall])


dnl -----------------------------------------------------------------------
dnl Configure switches

AC_ARG_ENABLE(shared,
              AS_HELP_STRING([--enable-shared],
                             [enable loading compiled
                                     files (default=YES)]),
              [], [enable_shared=yes])

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
dnl Installation directories
dnl AC_ARG_VAR([ecldir], [the directory where *.fas files are installed])

ATRIP_LDFLAGS="-L$PWD/src -latrip"
AC_SUBST(ATRIP_LDFLAGS)

AM_CONDITIONAL([WITH_BUILD_CTF], [test x${BUILD_CTF} = xYES])
AC_SUBST(LIBCTF_PATH)
AC_SUBST(LIBCTF_CPATH)
AC_SUBST(LIBCTF_LD_LIBRARY_PATH)

dnl -----------------------------------------------------------------------
dnl Guess operating system of host.
AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AC_PROG_CXX           dnl sets variable CXX
AM_PROG_AR
AC_PROG_CPP           dnl sets variable CPP
AC_PROG_RANLIB        dnl sets variable RANLIB
AC_PROG_INSTALL       dnl sets variables INSTALL, INSTALL_DATA, INSTALL_PROGRAM
AC_PROG_LN_S          dnl sets variable LN_S
AC_PROG_MAKE_SET      dnl set $(MAKE) if needed
AC_PROG_RANLIB        dnl set variable RANLIB
AC_CHECK_TOOL([AR],[ar]) dnl set variable AR to appropriate `ar' program
AC_OPENMP

AC_TYPE_SIZE_T
AC_FUNC_MALLOC


AX_BLAS([], [AC_MSG_FAILURE([Could not find BLAS library.])])
AX_LAPACK([], [AC_MSG_FAILURE([Could not find LAPACK library.])])

AX_CXX_COMPILE_STDCXX([11])
AX_CXX_STATIC_CAST
AX_CXX_TEMPLATES
AX_CXX_TYPENAME
AX_MPI([],
       [AC_MSG_ERROR([You don not have an MPI Implementation!])])
AX_PROG_CXX_MPI
CXX=${MPICXX}

dnl ---------- CHECK CTF
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


AC_CONFIG_FILES([Makefile
                 src/Makefile
                 bench/Makefile
                 ])
AC_CONFIG_HEADERS([config.h])

AC_MSG_RESULT([
                       AUTOMATIC CONFIGURATION
                       -----------------------
                       CXX      = $CXX
                       CXXFLAGS = $CXXFLAGS
                       LDFLAGS  = $LDFLAGS
                       ATRIP_LDFLAGS  = $ATRIP_LDFLAGS
])

AC_OUTPUT

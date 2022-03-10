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
AM_INIT_AUTOMAKE([foreign])


dnl -----------------------------------------------------------------------
dnl Configure switches

AC_ARG_ENABLE(shared,
  AS_HELP_STRING( [--enable-shared],
                  [enable loading compiled files (default=YES)]),
  [], [enable_shared=yes])

dnl LIBGC library options
AC_ARG_WITH(libctf-prefix,
  AS_HELP_STRING( [--with-libctf-prefix=path],
                  [prefix for system LIBGC includes and libraries] ),
  [LIBCTF_CPATH="$withval/include";
   LIBCTF_LD_LIBRARY_PATH="$withval/lib";
   BUILD_CTF=NO],
  [BUILD_CTF=YES])

AC_ARG_WITH(dgemm,
  AS_HELP_STRING( [--without-dgemm],
                  [Disable dgemm]),
  [with_dgemm=NO], [with_dgemm=YES])



dnl -----------------------------------------------------------------------
dnl Installation directories
dnl AC_ARG_VAR([ecldir], [the directory where *.fas files are installed])

ATRIP_LDFLAGS="-L$PWD/src -latrip"
AC_SUBST(ATRIP_LDFLAGS)

AC_SUBST(LIBCTF_CPATH)
AC_SUBST(LIBCTF_LD_LIBRARY_PATH)

dnl -----------------------------------------------------------------------
dnl Guess operating system of host.
AC_CANONICAL_BUILD
AC_CANONICAL_HOST

AC_PROG_CXX           dnl sets variable CXX
AC_PROG_CPP           dnl sets variable CPP
AC_PROG_RANLIB        dnl sets variable RANLIB
AC_PROG_INSTALL       dnl sets variables INSTALL, INSTALL_DATA, INSTALL_PROGRAM
AC_PROG_LN_S          dnl sets variable LN_S
AC_PROG_MAKE_SET      dnl set $(MAKE) if needed
AC_PROG_RANLIB        dnl set variable RANLIB
AC_CHECK_TOOL([AR],[ar]) dnl set variable AR to appropriate `ar' program

AC_TYPE_SIZE_T
AC_FUNC_MALLOC

AX_CXX_COMPILE_STDCXX([11])

AX_MPI([],
       [AC_MSG_ERROR([You don not have an MPI Implementation!])])
CXX=${MPICXX}

CXXFLAGS="$CXXFLAGS -I$PWD/include"
CXXFLAGS="$CXXFLAGS -I${LIBCTF_CPATH}"
LDFLAGS="$LDFLAGS -L${LIBCTF_LD_LIBRARY_PATH} -lctf"

AC_CONFIG_FILES([
  src/Makefile
  bench/Makefile
  etc/config/auto.mk
 ])

AC_CONFIG_HEADERS([config.h])

AC_MSG_RESULT([
                       AUTOMATIC CONFIGURATION
                       -----------------------
                       CXX      = $CXX
                       CXXFLAGS = $CXXFLAGS
                       DEFAULT_INCLUDES = $DEFAULT_INCLUDES
                       LDFLAGS  = $LDFLAGS
                       ATRIP_LDFLAGS  = $ATRIP_LDFLAGS
                       bindir   = $bindir
                       libdir   = $libdir
])

AC_OUTPUT

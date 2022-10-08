AC_DEFUN([ATRIP_DEF],
         [AC_ARG_ENABLE([$1],
                        [AS_HELP_STRING([--$2-$1],
                                        [$4])],
                        [AC_DEFINE([$3],
                                   1,
                                   [$4])])])


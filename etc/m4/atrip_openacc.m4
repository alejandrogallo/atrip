#   SYNOPSIS
#
#   ATRIP_OPENACC([ACTION-SUCCESS], [ACTION-FAILURE])
#
#   DESCRIPTION
#
#   Check whether the given the -fopenacc flag works with the current language's compiler
#   or gives an error.
#
#   ACTION-SUCCESS/ACTION-FAILURE are shell commands to execute on
#   success/failure.
#
#   LICENSE
#
#   Copyright (c) 2023 Alejandro Gallo <aamsgallo@gmail.com>
#
#   Copying and distribution of this file, with or without modification, are
#   permitted in any medium without royalty provided the copyright notice
#   and this notice are preserved.  This file is offered as-is, without any
#   warranty.

AC_DEFUN([ATRIP_OPENACC],
[
AC_MSG_CHECKING([that the compiler works with the -fopenacc])
AC_COMPILE_IFELSE([AC_LANG_SOURCE([_ATRIP_OPENACC_SOURCE])],
                  [
                   $1
                   AC_MSG_RESULT([yes])
                   ],
                  [
                   $2
                   AC_MSG_ERROR([no])
                   ])
])dnl DEFUN

m4_define([_ATRIP_OPENACC_SOURCE], [[
#include <stdio.h>
#include <stdlib.h>
#include <openacc.h>

#define SIZE 10

int main(int argc, char **argv) {
  float matrix[SIZE * SIZE];
  float result[SIZE * SIZE];

  // Initialize the matrix with random values
  for (int i = 0; i < SIZE * SIZE; i++) {
    matrix[i] = rand() / (float)RAND_MAX;
  }

#pragma acc data                                            \
        copy(matrix[0:SIZE * SIZE])           \
        copyout(result[0:SIZE * SIZE])
  {
    // Calculate the matrix multiplication
#pragma acc parallel loop collapse(2)
    for (int i = 0; i < SIZE; i++) {
      for (int j = 0; j < SIZE; j++) {
        float sum = 0.0f;
        for (int k = 0; k < SIZE; k++) {
          sum += matrix[i * SIZE + k] * matrix[j * SIZE + k];
        }
        result[i * SIZE + j] = sum;
      }
    }
  }
  return 0;
}
]])

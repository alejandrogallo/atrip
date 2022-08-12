// Copyright 2022 Alejandro Gallo
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

// [[file:~/cuda/atrip/atrip.org::*Prolog][Prolog:2]]
#include<atrip/Equations.hpp>
#include<atrip/CUDA.hpp>

#if defined(HAVE_CUDA)
#include <cuda.h>
#endif

namespace atrip {
// Prolog:2 ends here

#ifdef HAVE_CUDA
namespace cuda {

  // cuda kernels

  template <typename F>
  __global__
  void zeroing(F* a, size_t n) {
    F zero = {0};
    for (size_t i = 0; i < n; i++) {
      a[i] = zero;
    }
  }

  ////
  template <typename F>
  __device__
  F maybeConjugateScalar(const F a);

  template <>
  __device__
  double maybeConjugateScalar(const double a) { return a; }

  template <>
  __device__
  cuDoubleComplex
  maybeConjugateScalar(const cuDoubleComplex a) {
    return {a.x, -a.y};
  }

  template <typename F>
  __global__
  void maybeConjugate(F* to, F* from, size_t n) {
    for (size_t i = 0; i < n; ++i) {
      to[i] = maybeConjugateScalar<F>(from[i]);
    }
  }
  

  template <typename F>
  __global__
  void reorder(F* to, F* from, size_t size, size_t I, size_t J, size_t K) {
    size_t idx = 0;
    const size_t IDX = I + J*size + K*size*size;
    for (size_t k = 0; k < size; k++)
    for (size_t j = 0; j < size; j++)
    for (size_t i = 0; i < size; i++, idx++)
      to[idx] += from[IDX];
  }

  // I mean, really CUDA... really!?
  template <typename F>
  __device__
  F multiply(const F &a, const F &b);
  template <>
  __device__
  double multiply(const double &a, const double &b) { return a * b; }

  template <>
  __device__
  cuDoubleComplex multiply(const cuDoubleComplex &a, const cuDoubleComplex &b) {
   return
     {a.x * b.x - a.y * b.y,
      a.x * b.y + a.y * b.x};
  }

  template <typename F>
  __device__
  void sum_in_place(F* to, const F* b);

  template <>
  __device__
  void sum_in_place(double* to, const double *b) { *to += *b; }

  template <>
  __device__
  void sum_in_place(cuDoubleComplex* to, const cuDoubleComplex* b) {
   to->x += b->x;
   to->y += b->y;
  }

  __device__
  cuDoubleComplex& operator+=(cuDoubleComplex& lz, cuDoubleComplex const& rz) {
    lz.x += rz.x;
    lz.y += rz.y;
    return lz;
  }



};
#endif

// [[file:~/cuda/atrip/atrip.org::*Energy][Energy:2]]
template <typename F>
double getEnergyDistinct
  ( F const epsabc
  , size_t const No
  , F* const epsi
  , F* const Tijk
  , F* const Zijk
  ) {
  constexpr size_t blockSize=16;
  F energy(0.);
  for (size_t kk=0; kk<No; kk+=blockSize){
    const size_t kend( std::min(No, kk+blockSize) );
    for (size_t jj(kk); jj<No; jj+=blockSize){
      const size_t jend( std::min( No, jj+blockSize) );
      for (size_t ii(jj); ii<No; ii+=blockSize){
        const size_t iend( std::min( No, ii+blockSize) );
        for (size_t k(kk); k < kend; k++){
          const F ek(epsi[k]);
          const size_t jstart = jj > k ? jj : k;
          for (size_t j(jstart); j < jend; j++){
            F const ej(epsi[j]);
            F const facjk = j == k ? F(0.5) : F(1.0);
            size_t istart = ii > j ? ii : j;
            for (size_t i(istart); i < iend; i++){
              const F
                  ei(epsi[i])
                , facij = i == j ? F(0.5) : F(1.0)
                , denominator(epsabc - ei - ej - ek)
                , U(Zijk[i + No*j + No*No*k])
                , V(Zijk[i + No*k + No*No*j])
                , W(Zijk[j + No*i + No*No*k])
                , X(Zijk[j + No*k + No*No*i])
                , Y(Zijk[k + No*i + No*No*j])
                , Z(Zijk[k + No*j + No*No*i])
                , A(maybeConjugate<F>(Tijk[i + No*j + No*No*k]))
                , B(maybeConjugate<F>(Tijk[i + No*k + No*No*j]))
                , C(maybeConjugate<F>(Tijk[j + No*i + No*No*k]))
                , D(maybeConjugate<F>(Tijk[j + No*k + No*No*i]))
                , E(maybeConjugate<F>(Tijk[k + No*i + No*No*j]))
                , _F(maybeConjugate<F>(Tijk[k + No*j + No*No*i]))
                , value
                  = 3.0 * ( A * U
                            + B * V
                            + C * W
                            + D * X
                            + E * Y
                            + _F * Z )
                 + ( ( U + X + Y )
                   - 2.0 * ( V + W + Z )
                   ) * ( A + D + E )
                 + ( ( V + W + Z )
                   - 2.0 * ( U + X + Y )
                   ) * ( B + C + _F )
                ;
              energy += 2.0 * value / denominator * facjk * facij;
            } // i
          } // j
        } // k
      } // ii
    } // jj
  } // kk
  return std::real(energy);
}


template <typename F>
double getEnergySame
  ( F const epsabc
  , size_t const No
  , F* const epsi
  , F* const Tijk
  , F* const Zijk
  ) {
  constexpr size_t blockSize = 16;
  F energy = F(0.);
  for (size_t kk=0; kk<No; kk+=blockSize){
    const size_t kend( std::min( kk+blockSize, No) );
    for (size_t jj(kk); jj<No; jj+=blockSize){
      const size_t jend( std::min( jj+blockSize, No) );
      for (size_t ii(jj); ii<No; ii+=blockSize){
        const size_t iend( std::min( ii+blockSize, No) );
        for (size_t k(kk); k < kend; k++){
          const F ek(epsi[k]);
          const size_t jstart = jj > k ? jj : k;
          for(size_t j(jstart); j < jend; j++){
            const F facjk( j == k ? F(0.5) : F(1.0));
            const F ej(epsi[j]);
            const size_t istart = ii > j ? ii : j;
            for(size_t i(istart); i < iend; i++){
              const F
                ei(epsi[i])
              , facij ( i==j ? F(0.5) : F(1.0))
              , denominator(epsabc - ei - ej - ek)
              , U(Zijk[i + No*j + No*No*k])
              , V(Zijk[j + No*k + No*No*i])
              , W(Zijk[k + No*i + No*No*j])
              , A(maybeConjugate<F>(Tijk[i + No*j + No*No*k]))
              , B(maybeConjugate<F>(Tijk[j + No*k + No*No*i]))
              , C(maybeConjugate<F>(Tijk[k + No*i + No*No*j]))
              , value
                = F(3.0) * ( A * U
                           + B * V
                           + C * W
                           )
                - ( A + B + C ) * ( U + V + W )
              ;
              energy += F(2.0) * value / denominator * facjk * facij;
            } // i
          } // j
        } // k
      } // ii
    } // jj
  } // kk
  return std::real(energy);
}
// Energy:2 ends here

// [[file:~/cuda/atrip/atrip.org::*Energy][Energy:3]]
// instantiate double
template
double getEnergyDistinct
  ( double const epsabc
  , size_t const No
  , double* const epsi
  , double* const Tijk
  , double* const Zijk
  );

template
double getEnergySame
  ( double const epsabc
  , size_t const No
  , double* const epsi
  , double* const Tijk
  , double* const Zijk
  );

// instantiate Complex
template
double getEnergyDistinct
  ( Complex const epsabc
  , size_t const No
  , Complex* const epsi
  , Complex* const Tijk
  , Complex* const Zijk
  );

template
double getEnergySame
  ( Complex const epsabc
  , size_t const No
  , Complex* const epsi
  , Complex* const Tijk
  , Complex* const Zijk
  );
// Energy:3 ends here

// [[file:~/cuda/atrip/atrip.org::*Singles%20contribution][Singles contribution:2]]
template <typename F>
#ifdef HAVE_CUDA
__global__
#endif
  void singlesContribution
    ( size_t No
    , size_t Nv
    , size_t a
    , size_t b
    , size_t c
    , DataFieldType<F>* const Tph
    , DataFieldType<F>* const VABij
    , DataFieldType<F>* const VACij
    , DataFieldType<F>* const VBCij
    , DataFieldType<F>* Zijk
    ) {
    const size_t NoNo = No*No;
    // TODO: change order of for loops
    for (size_t k = 0; k < No; k++)
    for (size_t i = 0; i < No; i++)
    for (size_t j = 0; j < No; j++) {
      const size_t ijk = i + j*No + k*No*No;

#ifdef HAVE_CUDA
#  define GO(__TPH, __VABIJ)                                          \
    {                                                                 \
      const DataFieldType<F> product                                  \
            = cuda::multiply<DataFieldType<F>>((__TPH), (__VABIJ));   \
      cuda::sum_in_place<DataFieldType<F>>(&Zijk[ijk], &product);     \
    }
#else
#  define GO(__TPH, __VABIJ) Zijk[ijk] += (__TPH) * (__VABIJ);
#endif
      GO(Tph[ a + i * Nv ], VBCij[ j + k * No ])
      GO(Tph[ b + j * Nv ], VACij[ i + k * No ])
      GO(Tph[ c + k * Nv ], VABij[ i + j * No ])
#undef GO
    } // for loop j
  }


// instantiate
  template
#ifdef HAVE_CUDA
  __global__
#endif
  void singlesContribution<double>( size_t No
                                  , size_t Nv
                                  , size_t a
                                  , size_t b
                                  , size_t c
                                  , double* const Tph
                                  , double* const VABij
                                  , double* const VACij
                                  , double* const VBCij
                                  , double* Zijk
                                  );

  template
#ifdef HAVE_CUDA
  __global__
#endif
  void singlesContribution<Complex>( size_t No
                                   , size_t Nv
                                   , size_t a
                                   , size_t b
                                   , size_t c
                                   , DataFieldType<Complex>* const Tph
                                   , DataFieldType<Complex>* const VABij
                                   , DataFieldType<Complex>* const VACij
                                   , DataFieldType<Complex>* const VBCij
                                   , DataFieldType<Complex>* Zijk
                                   );
// Singles contribution:2 ends here

// [[file:~/cuda/atrip/atrip.org::*Doubles%20contribution][Doubles contribution:2]]
  template <typename F>
  void doublesContribution
    ( const ABCTuple &abc
    , size_t const No
    , size_t const Nv
    // -- VABCI
    , DataPtr<F> const VABph
    , DataPtr<F> const VACph
    , DataPtr<F> const VBCph
    , DataPtr<F> const VBAph
    , DataPtr<F> const VCAph
    , DataPtr<F> const VCBph
    // -- VHHHA
    , DataPtr<F> const VhhhA
    , DataPtr<F> const VhhhB
    , DataPtr<F> const VhhhC
    // -- TA
    , DataPtr<F> const TAphh
    , DataPtr<F> const TBphh
    , DataPtr<F> const TCphh
    // -- TABIJ
    , DataPtr<F> const TABhh
    , DataPtr<F> const TAChh
    , DataPtr<F> const TBChh
    // -- TIJK
    // , DataPtr<F> Tijk_
    , DataFieldType<F>* Tijk_
    ) {

    const size_t a = abc[0], b = abc[1], c = abc[2]
              , NoNo = No*No, NoNv = No*Nv
              ;

    typename DataField<F>::type* Tijk = (typename DataField<F>::type*) Tijk_;
    LOG(0, "AtripCUDA") <<  "in doubles " << "\n";

#if defined(ATRIP_USE_DGEMM)
#define _IJK_(i, j, k) i + j*No + k*NoNo
#if defined(HAVE_CUDA)
// TODO
#define REORDER(__II, __JJ, __KK)
#define __TO_DEVICEPTR(_v) (_v)
#define DGEMM_PARTICLES(__A, __B)                   \
  atrip::xgemm<F>("T",                              \
                  "N",                              \
                  (int const*)&NoNo,                \
                  (int const*)&No,                  \
                  (int const*)&Nv,                  \
                  &one,                             \
                  (DataFieldType<F>*)__A,           \
                  (int const*)&Nv,                  \
                  (DataFieldType<F>*)__B,           \
                  (int const*)&Nv,                  \
                  &zero,                            \
                  _t_buffer_p,                      \
                  (int const*)&NoNo);
#define DGEMM_HOLES(__A, __B, __TRANSB)             \
  atrip::xgemm<F>("N",                              \
                  __TRANSB,                         \
                  (int const*)&NoNo,                \
                  (int const*)&No,                  \
                  (int const*)&No,                  \
                  &m_one,                           \
                  __TO_DEVICEPTR(__A),              \
                  (int const*)&NoNo,                \
                  (DataFieldType<F>*)__B,           \
                  (int const*)&No,                  \
                  &zero,                            \
                  _t_buffer_p,                      \
                  (int const*)&NoNo                 \
                  );
#define MAYBE_CONJ(_conj, _buffer)                                      \
  cuda::maybeConjugate<<<1,1>>>((DataFieldType<F>*)_conj, (DataFieldType<F>*)_buffer, NoNoNo);
#else
#define REORDER(__II, __JJ, __KK)                                   \
    WITH_CHRONO("doubles:reorder",                                  \
    for (size_t k = 0; k < No; k++)                                 \
    for (size_t j = 0; j < No; j++)                                 \
    for (size_t i = 0; i < No; i++) {                               \
      Tijk[_IJK_(i, j, k)] += _t_buffer_p[_IJK_(__II, __JJ, __KK)];   \
    }                                                               \
    )
#define __TO_DEVICEPTR(_v) (_v)
#define DGEMM_PARTICLES(__A, __B)               \
  atrip::xgemm<F>("T",                          \
                  "N",                          \
                  (int const*)&NoNo,            \
                  (int const*)&No,              \
                  (int const*)&Nv,              \
                  &one,                         \
                  __A,                          \
                  (int const*)&Nv,              \
                  __B,                          \
                  (int const*)&Nv,              \
                  &zero,                        \
                  _t_buffer_p,                  \
                  (int const*)&NoNo             \
                  );
#define DGEMM_HOLES(__A, __B, __TRANSB)         \
  atrip::xgemm<F>("N",                          \
                  __TRANSB,                     \
                  (int const*)&NoNo,            \
                  (int const*)&No,              \
                  (int const*)&No,              \
                  &m_one,                       \
                  __A,                          \
                  (int const*)&NoNo,            \
                  __B,                          \
                  (int const*)&No,              \
                  &zero,                        \
                  _t_buffer_p,                  \
                  (int const*)&NoNo             \
                  );
#define MAYBE_CONJ(_conj, _buffer)                \
  for (size_t __i = 0; __i < NoNoNo; ++__i)       \
    _conj[__i] = maybeConjugate<F>(_buffer[__i]);
#endif

    F one{1.0}, m_one{-1.0}, zero{0.0};
    DataFieldType<F> zero_h{0.0};
    const size_t NoNoNo = No*NoNo;
#ifdef HAVE_CUDA
    DataFieldType<F>* _t_buffer;
    DataFieldType<F>* _vhhh;
    LOG(0, "AtripCUDA") <<  "getting memory" << "\n";
    cuMemAlloc((CUdeviceptr*)&_t_buffer, NoNoNo * sizeof(DataFieldType<F>));
    cuMemAlloc((CUdeviceptr*)&_vhhh, NoNoNo * sizeof(DataFieldType<F>));
    LOG(0, "AtripCUDA") <<  "cuda::zeroing " << "\n";
    cuda::zeroing<<<1,1>>>((DataFieldType<F>*)_t_buffer, NoNoNo);
    cuda::zeroing<<<1,1>>>((DataFieldType<F>*)_vhhh, NoNoNo);
#else
    F* _t_buffer = (F*)malloc(NoNoNo * sizeof(F));
    F* _vhhh = (F*)malloc(NoNoNo * sizeof(F));
    for (size_t i=0; i < NoNoNo; i++) {
      _t_buffer[i] = zero_h;
      _vhhh[i] = zero_h;
    }
#endif
    //_t_buffer.reserve(NoNoNo);
    DataFieldType<F>* _t_buffer_p = __TO_DEVICEPTR(_t_buffer);

#ifdef HAVE_CUDA
    LOG(0, "AtripCUDA") <<  "cuda::zeroing Tijk" << "\n";
    cuda::zeroing<<<1,1>>>((DataFieldType<F>*)Tijk, NoNoNo);
#else
    WITH_CHRONO("double:reorder",
      for (size_t k = 0; k < NoNoNo; k++) {
        Tijk[k] = DataFieldType<F>{0.0};
       })
#endif

    LOG(0, "AtripCUDA") <<  "doing holes" << "\n";
    // TOMERGE: replace chronos
    WITH_CHRONO("doubles:holes",
      { // Holes part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


        // VhhhC[i + k*No + L*NoNo] * TABhh[L + j*No]; H1
        LOG(0, "AtripCUDA") <<  "conj 1" << "\n";
        MAYBE_CONJ(_vhhh, VhhhC)
        LOG(0, "AtripCUDA") <<  "done" << "\n";
        WITH_CHRONO("doubles:holes:1",
          LOG(0, "AtripCUDA") <<  "dgemm 1" << "\n";
          DGEMM_HOLES(_vhhh, TABhh, "N")
          LOG(0, "AtripCUDA") <<  "reorder 1" << "\n";
          REORDER(i, k, j)
        )
        // VhhhC[j + k*No + L*NoNo] * TABhh[i + L*No]; H0
        WITH_CHRONO("doubles:holes:2",
          LOG(0, "AtripCUDA") <<  "dgemm 2" << "\n";
          DGEMM_HOLES(_vhhh, TABhh, "T")
          REORDER(j, k, i)
        )

        // VhhhB[i + j*No + L*NoNo] * TAChh[L + k*No]; H5
        LOG(0, "AtripCUDA") <<  "conj 2" << "\n";
        MAYBE_CONJ(_vhhh, VhhhB)
        LOG(0, "AtripCUDA") <<  "done" << "\n";
        WITH_CHRONO("doubles:holes:3",
          DGEMM_HOLES(_vhhh, TAChh, "N")
          REORDER(i, j, k)
        )
        // VhhhB[k + j*No + L*NoNo] * TAChh[i + L*No]; H3
        WITH_CHRONO("doubles:holes:4",
          DGEMM_HOLES(_vhhh, TAChh, "T")
          REORDER(k, j, i)
        )

        // VhhhA[j + i*No + L*NoNo] * TBChh[L + k*No]; H1
        LOG(0, "AtripCUDA") <<  "conj 3" << "\n";
        MAYBE_CONJ(_vhhh, VhhhA)
        WITH_CHRONO("doubles:holes:5",
          DGEMM_HOLES(_vhhh, TBChh, "N")
          REORDER(j, i, k)
        )
        // VhhhA[k + i*No + L*NoNo] * TBChh[j + L*No]; H4
        WITH_CHRONO("doubles:holes:6",
          DGEMM_HOLES(_vhhh, TBChh, "T")
          REORDER(k, i, j)
        )

      }
    )
  #undef MAYBE_CONJ

    LOG(0, "AtripCUDA") <<  "doing particles" << "\n";
    WITH_CHRONO("doubles:particles",
      { // Particle part %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
        // TAphh[E + i*Nv + j*NoNv] * VBCph[E + k*Nv]; P0
        WITH_CHRONO("doubles:particles:1",
          DGEMM_PARTICLES(TAphh, VBCph)
          REORDER(i, j, k)
        )
        // TAphh[E + i*Nv + k*NoNv] * VCBph[E + j*Nv]; P3
        WITH_CHRONO("doubles:particles:2",
          DGEMM_PARTICLES(TAphh, VCBph)
          REORDER(i, k, j)
        )
        // TCphh[E + k*Nv + i*NoNv] * VABph[E + j*Nv]; P5
        WITH_CHRONO("doubles:particles:3",
          DGEMM_PARTICLES(TCphh, VABph)
          REORDER(k, i, j)
        )
        // TCphh[E + k*Nv + j*NoNv] * VBAph[E + i*Nv]; P2
        WITH_CHRONO("doubles:particles:4",
          DGEMM_PARTICLES(TCphh, VBAph)
          REORDER(k, j, i)
        )
        // TBphh[E + j*Nv + i*NoNv] * VACph[E + k*Nv]; P1
        WITH_CHRONO("doubles:particles:5",
          DGEMM_PARTICLES(TBphh, VACph)
          REORDER(j, i, k)
        )
        // TBphh[E + j*Nv + k*NoNv] * VCAph[E + i*Nv]; P4
        WITH_CHRONO("doubles:particles:6",
          DGEMM_PARTICLES(TBphh, VCAph)
          REORDER(j, k, i)
        )
      }
    )
    LOG(0, "AtripCUDA") <<  "particles done" << "\n";

  { // free resources
#ifdef HAVE_CUDA
    LOG(0, "AtripCUDA") <<  "free mem" << "\n";
    cuMemFree((CUdeviceptr)_vhhh);
    cuMemFree((CUdeviceptr)_t_buffer);
    LOG(0, "AtripCUDA") <<  "free mem done" << "\n";
#else
    free(_vhhh);
    free(_t_buffer);
#endif
  }

  #undef REORDER
  #undef DGEMM_HOLES
  #undef DGEMM_PARTICLES
  #undef _IJK_
  #else
    for (size_t k = 0; k < No; k++)
    for (size_t j = 0; j < No; j++)
    for (size_t i = 0; i < No; i++){
      const size_t ijk = i + j*No + k*NoNo
                ,  jk = j + k*No
                ;
      Tijk[ijk] = 0.0; // :important
      // HOLE DIAGRAMS: TABHH and VHHHA
      for (size_t L = 0; L < No; L++){
        // t[abLj] * V[Lcik]        H1
        // t[baLi] * V[Lcjk]        H0      TODO: conjugate T for complex
        Tijk[ijk] -= TABhh[L + j*No] * VhhhC[i + k*No + L*NoNo];
        Tijk[ijk] -= TABhh[i + L*No] * VhhhC[j + k*No + L*NoNo];

        // t[acLk] * V[Lbij]        H5
        // t[caLi] * V[Lbkj]        H3
        Tijk[ijk] -= TAChh[L + k*No] * VhhhB[i + j*No + L*NoNo];
        Tijk[ijk] -= TAChh[i + L*No] * VhhhB[k + j*No + L*NoNo];

        // t[bcLk] * V[Laji]        H2
        // t[cbLj] * V[Laki]        H4
        Tijk[ijk] -= TBChh[L + k*No] * VhhhA[j + i*No + L*NoNo];
        Tijk[ijk] -= TBChh[j + L*No] * VhhhA[k + i*No + L*NoNo];
      }
      // PARTILCE DIAGRAMS: TAPHH and VABPH
      for (size_t E = 0; E < Nv; E++) {
        // t[aEij] * V[bcEk]        P0
        // t[aEik] * V[cbEj]        P3 // TODO: CHECK THIS ONE, I DONT KNOW
        Tijk[ijk] += TAphh[E + i*Nv + j*NoNv] * VBCph[E + k*Nv];
        Tijk[ijk] += TAphh[E + i*Nv + k*NoNv] * VCBph[E + j*Nv];

        // t[cEki] * V[abEj]        P5
        // t[cEkj] * V[baEi]        P2
        Tijk[ijk] += TCphh[E + k*Nv + i*NoNv] * VABph[E + j*Nv];
        Tijk[ijk] += TCphh[E + k*Nv + j*NoNv] * VBAph[E + i*Nv];

        // t[bEji] * V[acEk]        P1
        // t[bEjk] * V[caEi]        P4
        Tijk[ijk] += TBphh[E + j*Nv + i*NoNv] * VACph[E + k*Nv];
        Tijk[ijk] += TBphh[E + j*Nv + k*NoNv] * VCAph[E + i*Nv];
      }

    }
#endif
  }



  // instantiate templates
  template
  void doublesContribution<double>
    ( const ABCTuple &abc
    , size_t const No
    , size_t const Nv
    // -- VABCI
    , DataPtr<double> const VABph
    , DataPtr<double> const VACph
    , DataPtr<double> const VBCph
    , DataPtr<double> const VBAph
    , DataPtr<double> const VCAph
    , DataPtr<double> const VCBph
    // -- VHHHA
    , DataPtr<double> const VhhhA
    , DataPtr<double> const VhhhB
    , DataPtr<double> const VhhhC
    // -- TA
    , DataPtr<double> const TAphh
    , DataPtr<double> const TBphh
    , DataPtr<double> const TCphh
    // -- TABIJ
    , DataPtr<double> const TABhh
    , DataPtr<double> const TAChh
    , DataPtr<double> const TBChh
    // -- TIJK
    , DataFieldType<double>* Tijk
    );

  template
  void doublesContribution<Complex>
    ( const ABCTuple &abc
    , size_t const No
    , size_t const Nv
    // -- VABCI
    , DataPtr<Complex> const VABph
    , DataPtr<Complex> const VACph
    , DataPtr<Complex> const VBCph
    , DataPtr<Complex> const VBAph
    , DataPtr<Complex> const VCAph
    , DataPtr<Complex> const VCBph
    // -- VHHHA
    , DataPtr<Complex> const VhhhA
    , DataPtr<Complex> const VhhhB
    , DataPtr<Complex> const VhhhC
    // -- TA
    , DataPtr<Complex> const TAphh
    , DataPtr<Complex> const TBphh
    , DataPtr<Complex> const TCphh
    // -- TABIJ
    , DataPtr<Complex> const TABhh
    , DataPtr<Complex> const TAChh
    , DataPtr<Complex> const TBChh
    // -- TIJK
    , DataFieldType<Complex>* Tijk
    );
// Doubles contribution:2 ends here

// [[file:~/cuda/atrip/atrip.org::*Epilog][Epilog:2]]
}
// Epilog:2 ends here

// [[file:../../atrip.org::*Equations][Equations:1]]
#pragma once

#include<atrip/Slice.hpp>
#include<atrip/Blas.hpp>

namespace atrip {

  double getEnergyDistinct
    ( const double epsabc
    , std::vector<double> const& epsi
    , std::vector<double> const& Tijk_
    , std::vector<double> const& Zijk_
    ) {
    constexpr size_t blockSize=16;
    double energy(0.);
    const size_t No = epsi.size();
    for (size_t kk=0; kk<No; kk+=blockSize){
      const size_t kend( std::min(No, kk+blockSize) );
      for (size_t jj(kk); jj<No; jj+=blockSize){
        const size_t jend( std::min( No, jj+blockSize) );
        for (size_t ii(jj); ii<No; ii+=blockSize){
          const size_t iend( std::min( No, ii+blockSize) );
          for (size_t k(kk); k < kend; k++){
            const double ek(epsi[k]);
            const size_t jstart = jj > k ? jj : k;
            for (size_t j(jstart); j < jend; j++){
              const double ej(epsi[j]);
              double facjk( j == k ? 0.5 : 1.0);
              size_t istart = ii > j ? ii : j;
              for (size_t i(istart); i < iend; i++){
                const double ei(epsi[i]);
                double facij ( i==j ? 0.5 : 1.0);
                double denominator(epsabc - ei - ej - ek);
                double U(Zijk_[i + No*j + No*No*k]);
                double V(Zijk_[i + No*k + No*No*j]);
                double W(Zijk_[j + No*i + No*No*k]);
                double X(Zijk_[j + No*k + No*No*i]);
                double Y(Zijk_[k + No*i + No*No*j]);
                double Z(Zijk_[k + No*j + No*No*i]);

                double A(Tijk_[i + No*j + No*No*k]);
                double B(Tijk_[i + No*k + No*No*j]);
                double C(Tijk_[j + No*i + No*No*k]);
                double D(Tijk_[j + No*k + No*No*i]);
                double E(Tijk_[k + No*i + No*No*j]);
                double F(Tijk_[k + No*j + No*No*i]);
                double value(3.0*(A*U+B*V+C*W+D*X+E*Y+F*Z)
                            +((U+X+Y)-2.0*(V+W+Z))*(A+D+E)
                            +((V+W+Z)-2.0*(U+X+Y))*(B+C+F));
                energy += 2.0*value / denominator * facjk * facij;
              } // i
            } // j
          } // k
        } // ii
      } // jj
    } // kk
    return energy;
  }


  double getEnergySame
    ( const double epsabc
    , std::vector<double> const& epsi
    , std::vector<double> const& Tijk_
    , std::vector<double> const& Zijk_
    ) {
    constexpr size_t blockSize = 16;
    const size_t No = epsi.size();
    double energy(0.);
    for (size_t kk=0; kk<No; kk+=blockSize){
      const size_t kend( std::min( kk+blockSize, No) );
      for (size_t jj(kk); jj<No; jj+=blockSize){
        const size_t jend( std::min( jj+blockSize, No) );
        for (size_t ii(jj); ii<No; ii+=blockSize){
          const size_t iend( std::min( ii+blockSize, No) );
          for (size_t k(kk); k < kend; k++){
            const double ek(epsi[k]);
            const size_t jstart = jj > k ? jj : k;
            for(size_t j(jstart); j < jend; j++){
              const double facjk( j == k ? 0.5 : 1.0);
              const double ej(epsi[j]);
              const size_t istart = ii > j ? ii : j;
              for(size_t i(istart); i < iend; i++){
                double ei(epsi[i]);
                double facij ( i==j ? 0.5 : 1.0);
                double denominator(epsabc - ei - ej - ek);
                double U(Zijk_[i + No*j + No*No*k]);
                double V(Zijk_[j + No*k + No*No*i]);
                double W(Zijk_[k + No*i + No*No*j]);
                double A(Tijk_[i + No*j + No*No*k]);
                double B(Tijk_[j + No*k + No*No*i]);
                double C(Tijk_[k + No*i + No*No*j]);
                double value(3.0*( A*U + B*V + C*W) - (A+B+C)*(U+V+W));
                energy += 2.0*value / denominator * facjk * facij;
              } // i
            } // j
          } // k
        } // ii
      } // jj
    } // kk
    return energy;
  }

  void singlesContribution
    ( size_t No
    , size_t Nv
    , const ABCTuple &abc
    , double const* Tph
    , double const* VABij
    , double const* VACij
    , double const* VBCij
    , double *Zijk
    ) {
    const size_t a(abc[0]), b(abc[1]), c(abc[2]);
    for (size_t k=0; k < No; k++)
    for (size_t i=0; i < No; i++)
    for (size_t j=0; j < No; j++) {
      const size_t ijk = i + j*No + k*No*No
                ,  jk = j + No * k
                ;
      Zijk[ijk] += Tph[ a + i * Nv ] * VBCij[ j + k * No ];
      Zijk[ijk] += Tph[ b + j * Nv ] * VACij[ i + k * No ];
      Zijk[ijk] += Tph[ c + k * Nv ] * VABij[ i + j * No ];
    }
  }

  void doublesContribution
    ( const ABCTuple &abc
    , size_t const No
    , size_t const Nv
    // -- VABCI
    , double const* VABph
    , double const* VACph
    , double const* VBCph
    , double const* VBAph
    , double const* VCAph
    , double const* VCBph
    // -- VHHHA
    , double const* VhhhA
    , double const* VhhhB
    , double const* VhhhC
    // -- TA
    , double const* TAphh
    , double const* TBphh
    , double const* TCphh
    // -- TABIJ
    , double const* TABhh
    , double const* TAChh
    , double const* TBChh
    // -- TIJK
    , double *Tijk
    ) {

    const size_t a = abc[0], b = abc[1], c = abc[2]
              , NoNo = No*No, NoNv = No*Nv
              ;

#if defined(ATRIP_USE_DGEMM)
#define _IJK_(i, j, k) i + j*No + k*NoNo
#define REORDER(__II, __JJ, __KK)                           \
  WITH_CHRONO("double:reorder",                             \
              for (size_t k = 0; k < No; k++)               \
                for (size_t j = 0; j < No; j++)             \
                  for (size_t i = 0; i < No; i++) {         \
                    Tijk[_IJK_(i, j, k)]                    \
                    += _t_buffer[_IJK_(__II, __JJ, __KK)];  \
                  }                                         \
              )
#define DGEMM_PARTICLES(__A, __B)               \
  atrip::dgemm_("T",                            \
                "N",                            \
                (int const*)&NoNo,              \
                (int const*)&No,                \
                (int const*)&Nv,                \
                &one,                           \
                __A,                            \
                (int const*)&Nv,                \
                __B,                            \
                (int const*)&Nv,                \
                &zero,                          \
                _t_buffer.data(),               \
                (int const*)&NoNo);
#define DGEMM_HOLES(__A, __B, __TRANSB)         \
  atrip::dgemm_("N",                            \
                __TRANSB,                       \
                (int const*)&NoNo,              \
                (int const*)&No,                \
                (int const*)&No,                \
                &m_one,                         \
                __A,                            \
                (int const*)&NoNo,              \
                __B,                            \
                (int const*)&No,                \
                &zero,                          \
                _t_buffer.data(),               \
                (int const*)&NoNo);

    using F = double;
    const size_t NoNoNo = No*NoNo;
    std::vector<double> _t_buffer;
    _t_buffer.reserve(NoNoNo);
    F one{1.0}, m_one{-1.0}, zero{0.0};

    WITH_CHRONO("double:reorder",
      for (size_t k = 0; k < NoNoNo; k++) {
         Tijk[k] = 0.0;
       })

    WITH_CHRONO("doubles:holes",
                { // Holes part ================================================
                  // VhhhC[i + k*No + L*NoNo] * TABhh[L + j*No]; H1
                  WITH_CHRONO("doubles:holes:1",
                              DGEMM_HOLES(VhhhC, TABhh, "N")
                              REORDER(i, k, j)
                              )
                  // VhhhC[j + k*No + L*NoNo] * TABhh[i + L*No]; H0
                  WITH_CHRONO("doubles:holes:2",
                              DGEMM_HOLES(VhhhC, TABhh, "T")
                              REORDER(j, k, i)
                              )
                  // VhhhB[i + j*No + L*NoNo] * TAChh[L + k*No]; H5
                  WITH_CHRONO("doubles:holes:3",
                              DGEMM_HOLES(VhhhB, TAChh, "N")
                              REORDER(i, j, k)
                              )
                  // VhhhB[k + j*No + L*NoNo] * TAChh[i + L*No]; H3
                  WITH_CHRONO("doubles:holes:4",
                              DGEMM_HOLES(VhhhB, TAChh, "T")
                              REORDER(k, j, i)
                              )
                  // VhhhA[j + i*No + L*NoNo] * TBChh[L + k*No]; H1
                  WITH_CHRONO("doubles:holes:5",
                              DGEMM_HOLES(VhhhA, TBChh, "N")
                              REORDER(j, i, k)
                              )
                  // VhhhA[k + i*No + L*NoNo] * TBChh[j + L*No]; H4
                  WITH_CHRONO("doubles:holes:6",
                              DGEMM_HOLES(VhhhA, TBChh, "T")
                              REORDER(k, i, j)
                              )
                }
                )

      WITH_CHRONO("doubles:particles",
                  { // Particle part ===========================================
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

}
// Equations:1 ends here

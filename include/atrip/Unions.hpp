// [[file:~/atrip/atrip.org::*Unions][Unions:1]]
#pragma once
#include <atrip/SliceUnion.hpp>

namespace atrip {

  void sliceIntoVector
    ( std::vector<double> &v
    , CTF::Tensor<double> &toSlice
    , std::vector<int64_t> const low
    , std::vector<int64_t> const up
    , CTF::Tensor<double> const& origin
    , std::vector<int64_t> const originLow
    , std::vector<int64_t> const originUp
    ) {
    // Thank you CTF for forcing me to do this
    struct { std::vector<int> up, low; }
        toSlice_ = { {up.begin(), up.end()}
                   , {low.begin(), low.end()} }
      , origin_ = { {originUp.begin(), originUp.end()}
                  , {originLow.begin(), originLow.end()} }
      ;

    WITH_OCD
    WITH_RANK << "slicing into " << pretty_print(toSlice_.up)
                          << "," << pretty_print(toSlice_.low)
              << " from " << pretty_print(origin_.up)
                   << "," << pretty_print(origin_.low)
              << "\n";

#ifndef ATRIP_DONT_SLICE
    toSlice.slice( toSlice_.low.data()
                 , toSlice_.up.data()
                 , 0.0
                 , origin
                 , origin_.low.data()
                 , origin_.up.data()
                 , 1.0);
    memcpy(v.data(), toSlice.data, sizeof(double) * v.size());
#endif

  }


  struct TAPHH : public SliceUnion {
    TAPHH( Tensor const& sourceTensor
         , size_t No
         , size_t Nv
         , size_t np
         , MPI_Comm child_world
         , MPI_Comm global_world
         ) : SliceUnion( sourceTensor
                       , {Slice::A, Slice::B, Slice::C}
                       , {Nv, No, No} // size of the slices
                       , {Nv}
                       , np
                       , child_world
                       , global_world
                       , Slice::TA
                       , 6) {
           init(sourceTensor);
         }

    void sliceIntoBuffer(size_t it, Tensor &to, Tensor const& from) override
    {
      const int Nv = sliceLength[0]
              , No = sliceLength[1]
              , a = rankMap.find({static_cast<size_t>(Atrip::rank), it});
              ;


      sliceIntoVector( sources[it]
                     , to,   {0, 0, 0},    {Nv, No, No}
                     , from, {a, 0, 0, 0}, {a+1, Nv, No, No}
                     );

    }

  };


  struct HHHA : public SliceUnion {
    HHHA( Tensor const& sourceTensor
        , size_t No
        , size_t Nv
        , size_t np
        , MPI_Comm child_world
        , MPI_Comm global_world
        ) : SliceUnion( sourceTensor
                      , {Slice::A, Slice::B, Slice::C}
                      , {No, No, No} // size of the slices
                      , {Nv}         // size of the parametrization
                      , np
                      , child_world
                      , global_world
                      , Slice::VIJKA
                      , 6) {
           init(sourceTensor);
         }

    void sliceIntoBuffer(size_t it, Tensor &to, Tensor const& from) override
    {

      const int No = sliceLength[0]
              , a = rankMap.find({static_cast<size_t>(Atrip::rank), it})
              ;

      sliceIntoVector( sources[it]
                     , to,   {0, 0, 0},    {No, No, No}
                     , from, {0, 0, 0, a}, {No, No, No, a+1}
                     );

    }
  };

  struct ABPH : public SliceUnion {
    ABPH( Tensor const& sourceTensor
        , size_t No
        , size_t Nv
        , size_t np
        , MPI_Comm child_world
        , MPI_Comm global_world
        ) : SliceUnion( sourceTensor
                      , { Slice::AB, Slice::BC, Slice::AC
                        , Slice::BA, Slice::CB, Slice::CA
                        }
                      , {Nv, No} // size of the slices
                      , {Nv, Nv} // size of the parametrization
                      , np
                      , child_world
                      , global_world
                      , Slice::VABCI
                      , 2*6) {
           init(sourceTensor);
         }

    void sliceIntoBuffer(size_t it, Tensor &to, Tensor const& from) override {

      const int Nv = sliceLength[0]
              , No = sliceLength[1]
              , el = rankMap.find({static_cast<size_t>(Atrip::rank), it})
              , a = el % Nv
              , b = el / Nv
              ;


      sliceIntoVector( sources[it]
                     , to,   {0, 0},       {Nv, No}
                     , from, {a, b, 0, 0}, {a+1, b+1, Nv, No}
                     );

    }

  };

  struct ABHH : public SliceUnion {
    ABHH( Tensor const& sourceTensor
        , size_t No
        , size_t Nv
        , size_t np
        , MPI_Comm child_world
        , MPI_Comm global_world
        ) : SliceUnion( sourceTensor
                      , {Slice::AB, Slice::BC, Slice::AC}
                      , {No, No} // size of the slices
                      , {Nv, Nv} // size of the parametrization
                      , np
                      , child_world
                      , global_world
                      , Slice::VABIJ
                      , 6) {
           init(sourceTensor);
         }

    void sliceIntoBuffer(size_t it, Tensor &to, Tensor const& from) override {

      const int Nv = from.lens[0]
              , No = sliceLength[1]
              , el = rankMap.find({static_cast<size_t>(Atrip::rank), it})
              , a = el % Nv
              , b = el / Nv
              ;

      sliceIntoVector( sources[it]
                     , to,   {0, 0},       {No, No}
                     , from, {a, b, 0, 0}, {a+1, b+1, No, No}
                     );


    }

  };


  struct TABHH : public SliceUnion {
    TABHH( Tensor const& sourceTensor
         , size_t No
         , size_t Nv
         , size_t np
         , MPI_Comm child_world
         , MPI_Comm global_world
         ) : SliceUnion( sourceTensor
                       , {Slice::AB, Slice::BC, Slice::AC}
                       , {No, No} // size of the slices
                       , {Nv, Nv} // size of the parametrization
                       , np
                       , child_world
                       , global_world
                       , Slice::TABIJ
                       , 6) {
           init(sourceTensor);
         }

    void sliceIntoBuffer(size_t it, Tensor &to, Tensor const& from) override {
      // TODO: maybe generalize this with ABHH

      const int Nv = from.lens[0]
              , No = sliceLength[1]
              , el = rankMap.find({static_cast<size_t>(Atrip::rank), it})
              , a = el % Nv
              , b = el / Nv
              ;

      sliceIntoVector( sources[it]
                     , to,   {0, 0},       {No, No}
                     , from, {a, b, 0, 0}, {a+1, b+1, No, No}
                     );


    }

  };

}
// Unions:1 ends here

// -*- C++ -*-
// ex1_histogram.cc
// an exercise for the sandia 2014 clinic team.
// here we do a histogram calculation over unsigned ints

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <array>
#include <string>
#include <chrono>
#include <algorithm>

// header file for openmp
#include <omp.h>

// header files for tbb
#include <tbb/blocked_range.h>
#include <tbb/parallel_reduce.h>
#include <tbb/parallel_for.h>
#include <tbb/task_scheduler_init.h>
#include <tbb/atomic.h>
// header files for cuda implementation
#include "ex1_histogram_cuda.cuh"

// header files for kokkos
#include <Kokkos_Core.hpp>

using std::string;
using std::vector;
using std::array;
using std::chrono::high_resolution_clock;
using std::chrono::duration;
using std::chrono::duration_cast;
using tbb::atomic;

class TbbOutputter {
public:

  vector<unsigned int> * input_;

  vector<atomic<unsigned long>> * result_;

  TbbOutputter(vector<unsigned int> * input, vector<atomic<unsigned long>> * result):
    input_(input), result_(result){
    
  }

  TbbOutputter(const TbbOutputter & other,
               tbb::split)
               : input_(other.input_), result_(other.result_){
    //printf("split copy constructor called\n");
  }

  void operator()(const tbb::blocked_range<size_t> & range) {
    //printf("TbbOutputter asked to process range from %7zu to %7zu\n",
           //range.begin(), range.end());

    unsigned long bucketSize = input_->size()/result_->size();
    for(unsigned long i=range.begin(); i!= range.end(); ++i ) {
      unsigned int value= (*input_)[i];
      result_->at(value/bucketSize).fetch_and_increment();
    }
  }

  void join(const TbbOutputter & other) {

  }

private:
  TbbOutputter();

};

struct KokkosFunctor {

  const unsigned int _bucketSize;

  KokkosFunctor(const double bucketSize) : _bucketSize(bucketSize) {
  }

  KOKKOS_INLINE_FUNCTION
  void operator()(const unsigned int elementIndex) const {
  }

private:
  KokkosFunctor();

};

int main(int argc, char* argv[]) {

  // a couple of inputs.  change the numberOfIntervals to control the amount
  //  of work done
  const unsigned int numberOfElements = 1e2;
  // The number of buckets in our histogram
  const unsigned int numberOfBuckets = 1e2;

  // these are c++ timers...for timing
  high_resolution_clock::time_point tic;
  high_resolution_clock::time_point toc;

  printf("Creating the input vector \n");
  vector<unsigned int> input(numberOfElements);
  for(unsigned int i = 0; i < numberOfElements; ++i) {
    input[i] = i;
  }
  //std::random_shuffle(input.begin(), input.end());

  // ===============================================================
  // ********************** < do slow serial> **********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  vector<unsigned int> slowSerialHistogram(numberOfBuckets);
  tic = high_resolution_clock::now();
  const unsigned int bucketSize = input.size()/numberOfBuckets;
  for (unsigned int index = 0; index < numberOfElements; ++index) {
    const unsigned int value = input[index];
    const unsigned int bucketNumber = value / bucketSize;
    slowSerialHistogram[bucketNumber] += 1;
  }
  toc = high_resolution_clock::now();
  const double slowSerialElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();
  for(unsigned int bucketIndex = 0; bucketIndex < numberOfBuckets; ++bucketIndex){
    if(slowSerialHistogram[bucketIndex]!= bucketSize)
      fprintf(stderr, "wrong");
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do slow serial> **********************
  // ===============================================================

  // ===============================================================
  // ********************** < do fast serial> **********************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  vector<unsigned int> fastSerialHistogram(numberOfBuckets, 0);
  tic = high_resolution_clock::now();

  // TODO: can you make the serial one go faster? i can get about a
  //  15-20% speedup, but that's about it.  not very interesting
  // TODO: do this better

  for (unsigned int index = 0; index < numberOfElements; ++index) {
    const unsigned int value = input[index];
    const unsigned int bucketNumber = value / bucketSize;
    fastSerialHistogram[bucketNumber] += 1;
  }
  toc = high_resolution_clock::now();
  const double fastSerialElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  for (unsigned int bucketIndex = 0;
       bucketIndex < numberOfBuckets; ++bucketIndex) {
    if (fastSerialHistogram[bucketIndex] != bucketSize) {
      fprintf(stderr, "bucket %u has the wrong value: %u instead of %u\n",
              bucketIndex, fastSerialHistogram[bucketIndex], bucketSize);
      exit(1);
    }
  }

  // output speedup
  printf("fast: time %8.2e speedup %8.2e\n",
         fastSerialElapsedTime,
         slowSerialElapsedTime / fastSerialElapsedTime);

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do fast serial> **********************
  // ===============================================================

  // we will repeat the computation for each of the numbers of threads
  vector<unsigned int> numberOfThreadsArray;
  numberOfThreadsArray.push_back(1);
  numberOfThreadsArray.push_back(2);
  numberOfThreadsArray.push_back(4);
  numberOfThreadsArray.push_back(8);
  numberOfThreadsArray.push_back(16);
  numberOfThreadsArray.push_back(24);

  const size_t grainSize =
    std::max(unsigned(1e4), numberOfElements / 48);

  // ===============================================================
  // ********************** < do tbb> ******************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with tbb\n");
  // for each number of threads
  for (const unsigned int numberOfThreads :
         numberOfThreadsArray) {

    // initialize tbb's threading system for this number of threads
    tbb::task_scheduler_init init(numberOfThreads);

    vector<atomic<unsigned long>> results(numberOfBuckets,0);
    TbbOutputter tbbOutputter(&input, &results);

    // start timing
    tic = high_resolution_clock::now();
    // dispatch threads
    parallel_reduce(tbb::blocked_range<size_t>(0, numberOfElements,
                                               grainSize),
                    tbbOutputter);
    // stop timing
    toc = high_resolution_clock::now();
    const double threadedElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    for (unsigned int bucketIndex = 0;
         bucketIndex < numberOfBuckets; ++bucketIndex) {
      if (results[bucketIndex] != bucketSize) {
        fprintf(stderr, "bucket %u has the wrong value: %u instead of %u\n",
                bucketIndex, unsigned(results[bucketIndex]),
                bucketSize);
        exit(1);
      }
    }

    // output speedup
    printf("%3u : time %8.2e speedup %8.2e (%%%5.1f of ideal)\n",
           numberOfThreads,
           threadedElapsedTime,
           fastSerialElapsedTime / threadedElapsedTime,
           100. * fastSerialElapsedTime / threadedElapsedTime / numberOfThreads);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do tbb> ******************************
  // ===============================================================


  // ===============================================================
  // ********************** < do openmp> ***************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with openmp\n");
  // for each number of threads
  for (const unsigned int numberOfThreads :
         numberOfThreadsArray) {

    // set the number of threads for openmp
    omp_set_num_threads(numberOfThreads);

    vector<unsigned int> ompHistogram(numberOfBuckets, 0);

    // start timing
    tic = high_resolution_clock::now();

    // TODO: do openmp

    // stop timing
    toc = high_resolution_clock::now();
    const double threadedElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    for (unsigned int bucketIndex = 0;
         bucketIndex < numberOfBuckets; ++bucketIndex) {
      if (ompHistogram[bucketIndex] != bucketSize) {
        fprintf(stderr, "bucket %u has the wrong value: %u instead of %u\n",
                bucketIndex, ompHistogram[bucketIndex], bucketSize);
        exit(1);
      }
    }

    // output speedup
    printf("%3u : time %8.2e speedup %8.2e (%%%5.1f of ideal)\n",
           numberOfThreads,
           threadedElapsedTime,
           fastSerialElapsedTime / threadedElapsedTime,
           100. * fastSerialElapsedTime / threadedElapsedTime / numberOfThreads);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do openmp> ***************************
  // ===============================================================

  // ===============================================================
  // ********************** < do cuda> *****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with cuda\n");
  // we will repeat the computation for each of the numbers of threads
  vector<unsigned int> threadsPerBlockArray;
  threadsPerBlockArray.push_back(32);
  threadsPerBlockArray.push_back(64);
  threadsPerBlockArray.push_back(128);
  threadsPerBlockArray.push_back(256);
  threadsPerBlockArray.push_back(512);

  printf("performing calculations with cuda\n");
  // for each number of threads per block
  for (const unsigned int numberOfThreadsPerBlock :
         threadsPerBlockArray) {

    vector<unsigned int> cudaHistogram(numberOfBuckets, 0);

    // start timing
    tic = high_resolution_clock::now();

    // TODO: do cuda stuff

    // do scalar integration with cuda for this number of threads per block
    cudaDoHistogramPopulation(numberOfThreadsPerBlock,
                              &cudaHistogram[0]);

    // stop timing
    toc = high_resolution_clock::now();
    const double cudaElapsedTime =
      duration_cast<duration<double> >(toc - tic).count();

    // check the answer
    for (unsigned int bucketIndex = 0;
         bucketIndex < numberOfBuckets; ++bucketIndex) {
      if (cudaHistogram[bucketIndex] != bucketSize) {
        fprintf(stderr, "bucket %u has the wrong value: %u instead of %u\n",
                bucketIndex, cudaHistogram[bucketIndex], bucketSize);
        exit(1);
      }
    }

    // output speedup
    printf("%3u : time %8.2e speedup %8.2e\n",
           numberOfThreadsPerBlock,
           cudaElapsedTime,
           fastSerialElapsedTime / cudaElapsedTime);
  }

  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do cuda> *****************************
  // ===============================================================

  // ===============================================================
  // ********************** < do kokkos> *****************************
  // vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

  printf("performing calculations with kokkos running on %s\n",
         typeid(Kokkos::DefaultExecutionSpace).name());

  Kokkos::initialize();

  // start timing
  tic = high_resolution_clock::now();

  // TODO: do kokkos stuff

  // stop timing
  toc = high_resolution_clock::now();
  const double kokkosElapsedTime =
    duration_cast<duration<double> >(toc - tic).count();

  // check the answer
  vector<unsigned int> kokkosHistogram(numberOfBuckets, 0);
  for (unsigned int bucketIndex = 0;
       bucketIndex < numberOfBuckets; ++bucketIndex) {
    if (kokkosHistogram[bucketIndex] != bucketSize) {
      fprintf(stderr, "bucket %u has the wrong value: %u instead of %u\n",
              bucketIndex, kokkosHistogram[bucketIndex], bucketSize);
      exit(1);
    }
  }

  // output speedup
  printf("kokkos : time %8.2e speedup %8.2e\n",
         kokkosElapsedTime,
         fastSerialElapsedTime / kokkosElapsedTime);


  Kokkos::finalize();
  // ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  // ********************** </do kokkos> ***************************
  // ===============================================================

  return 0;
}

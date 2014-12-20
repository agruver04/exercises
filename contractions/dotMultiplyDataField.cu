// -*- C++ -*-
// matrixMultiplication.cc
// a huge comparison of doing naive and tiled matrix multiplication using many
//  different methods and technologies

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <string>
#include <algorithm>

// yucky, but for asking the system how many cores we have
#include <unistd.h>

// header file for openmp
#include <omp.h>

// header files for kokkos
#include <Kokkos_Core.hpp>
#include "Teuchos_Array.hpp"
#include "Intrepid_ArrayTools.hpp"
#include "Intrepid_FieldContainer.hpp"
#include "Intrepid_RealSpaceTools.hpp"
#include "Teuchos_oblackholestream.hpp"
#include "Teuchos_RCP.hpp"
#include "Teuchos_ScalarTraits.hpp"
#include "Teuchos_GlobalMPISession.hpp"

#include <cuda_runtime.h>

using std::string;
using std::vector;
using Intrepid::FieldContainer;

typedef Intrepid::RealSpaceTools<double> rst;

#define BLOCK_SIZE 64;

//Pre-C++11 timing (thanks jeff)
double getElapsedTime(const timespec start, const timespec end) {
	timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return double(temp.tv_sec) + double(temp.tv_nsec) / 1e9;
}


__global__
void
cudaDocontractFieldFieldScalar_kernelColMajor(double * d_left, double * d_right,
		double * d_out,
		int numCells,
		int numPoints) {

	int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(myID < numCells) {
		double temp = 0;
		for (int qp = 0; qp < numPoints; qp++) {
			temp += d_left[myID + qp*numCells] * d_right[myID + qp*numCells];
		}
		d_out[myID]=temp;
	}
}

__global__
void
cudaDocontractFieldFieldScalar_kernel(const double * const __restrict__ d_left, const double * const __restrict__ d_right,
double * d_out,
int numCells,
int numLeftFields,
int numRightFields,
int numPoints) {

	int myID = (blockIdx.x * blockDim.x) + threadIdx.x;

	if(myID < (numCells * numLeftFields * numRightFields)) {
		int myMatrix = myID / (numLeftFields * numRightFields);
		int matrixIndex = myID % (numLeftFields * numRightFields);

		int matrixRow = matrixIndex / numRightFields;
		int matrixCol = matrixIndex % numRightFields;

		double temp = 0;
		for (int qp = 0; qp < numPoints; qp++) {
			temp += d_left[myMatrix*numPoints*numLeftFields + numPoints*matrixRow + qp] *
							d_right[myMatrix*numPoints*numRightFields + qp*numRightFields + matrixCol];
		}
		d_out[myID]= temp;
	}
}

void
cudaDocontractFieldFieldScalar(double * h_out,
		double * h_inLeft,
		double * h_inRight,
		int numCells,
		int numLeftFields,
		int numRightFields,
		int numPoints,
		timespec * tic,
		timespec * toc) {

	double * d_right;
	double * d_left;
	double * d_out;

	cudaMalloc(&d_right, sizeof(double) * numCells  * numPoints * numRightFields);

	cudaMalloc(&d_left, sizeof(double) * numCells * numPoints * numLeftFields);

	cudaMalloc(&d_out, sizeof(double) * numCells * numRightFields * numLeftFields);

	cudaMemset(d_out, 0, sizeof(double) * numCells * numRightFields * numLeftFields);

	cudaMemcpy(d_right, h_inRight,
			sizeof(double) * numCells * numPoints * numRightFields, cudaMemcpyHostToDevice);

	cudaMemcpy(d_left, h_inLeft,
			sizeof(double) * numCells * numPoints * numLeftFields, cudaMemcpyHostToDevice);


	dim3 blockSize(1024);
	dim3 gridSize((numCells * numLeftFields * numRightFields / 1024) + 1);
	
	clock_gettime(CLOCK_MONOTONIC, tic);
	cudaDocontractFieldFieldScalar_kernel<<<gridSize, blockSize>>>(d_left,
			d_right, d_out, numCells, numLeftFields, numRightFields, numPoints);
	
	cudaDeviceSynchronize();
	clock_gettime(CLOCK_MONOTONIC, toc);
	cudaMemcpy(h_out, d_out, sizeof(double) * numCells * numLeftFields * numRightFields, cudaMemcpyDeviceToHost);

}
/*
__global__
void
cudaDocontractFieldFieldScalar_kernelUnrolled(double * d_left, double * d_right,
double * d_out,
int numCells,
int numPoints) {
	int myID = (blockIdx.x * blockDim.x) + threadIdx.x + (blockIdx.y * numPoints);
	if(myID < numCells * numPoints) {
		temp += d_left[myID*numPoints + qp] * d_right[myID*numPoints + qp];
		d_out[myID]= temp;
	}
}
*/
/*
void
cudaDocontractFieldFieldScalarUnrolled(double * h_out,
double * h_inLeft,
double * h_inRight,
int numCells,
int numPoints,
bool colMajor) {
	double * d_right;
	double * d_left;
	double * d_out;
	cudaMalloc(&d_right, sizeof(double) * numCells  * numPoints);
	cudaMalloc(&d_left, sizeof(double) * numCells * numPoints);
	cudaMalloc(&d_out, sizeof(double) * numCells);
	cudaMemset(d_out, 0, sizeof(double) * numCells);
	cudaMemcpy(d_right, h_inRight,
	sizeof(double) * numCells * numPoints, cudaMemcpyHostToDevice);
	cudaMemcpy(d_left, h_inLeft,
	sizeof(double) * numCells * numPoints, cudaMemcpyHostToDevice);
	dim3 blockSize(64);
	dim3 gridSize((numPoints / 64) + 1, (numCells / 64) + 1);
	cudaDocontractFieldFieldScalar_kernelUnrolled<<<gridSize, blockSize>>>(d_left,
	d_right, d_out, numCells,numPoints);
	cudaMemcpy(h_out, d_out, sizeof(double) * numCells, cudaMemcpyDeviceToHost);
}
*/
template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct contractFieldFieldScalarFunctor {
	typedef DeviceType device_type;
	LeftViewType _leftFields;
	RightViewType _rightFields;
	OutputViewType _outputFields;
	int _numPoints;
	int _numLeftFields;
	int _numRightFields;

	contractFieldFieldScalarFunctor(LeftViewType leftFields,
			RightViewType rightFields,
			OutputViewType outputFields,
			int numLeftFields,
			int numRightFields,
			int numPoints) :
		_leftFields(leftFields),
		_rightFields(rightFields),
		_outputFields(outputFields),
		_numPoints(numPoints),
		_numLeftFields(numLeftFields),
		_numRightFields(numRightFields)
	{
		// Nothing to do
	}

	KOKKOS_INLINE_FUNCTION
		void operator()(const unsigned int elementIndex) const {
			for (int lbf = 0; lbf < _numLeftFields; lbf++) {
				for (int rbf = 0; rbf < _numRightFields; rbf++) {
					double tmpVal = 0;
					for (int qp = 0; qp < _numPoints; qp++) {
						tmpVal += _leftFields(elementIndex, lbf, qp)*_rightFields(elementIndex, rbf, qp);
					} // P-loop
					_outputFields(elementIndex, lbf, rbf) = tmpVal;
				} // R-loop
			} // L-loop
		}
};




template<class DeviceType, class LeftViewType, class RightViewType, class OutputViewType>
struct contractFieldFieldScalarFunctor1D {
	typedef DeviceType device_type;
	LeftViewType _leftFields;
	RightViewType _rightFields;
	OutputViewType _outputFields;
	int _numLeftFields;
	int _numRightFields;
	int _numPoints;
	int _dim1Tensor;
	int _dim2Tensor;
	int _numCells;

	contractFieldFieldScalarFunctor1D(LeftViewType leftFields,
			RightViewType rightFields,
			OutputViewType outputFields,
			int numLeftFields,
			int numRightFields,
			int numPoints,
			int dim1Tensor,
			int dim2Tensor,
			int numCells) :
		_leftFields(leftFields),
		_rightFields(rightFields),
		_outputFields(outputFields),
		_numLeftFields(numLeftFields),
		_numRightFields(numRightFields),
		_numPoints(numPoints),
		_dim1Tensor(dim1Tensor),
		_dim2Tensor(dim2Tensor),
		_numCells(numCells)
	{
		// Nothing to do
	}

	KOKKOS_INLINE_FUNCTION
		void operator()(const unsigned int elementIndex) const {

			for (int lbf = 0; lbf < _numLeftFields; lbf++) {
				for (int rbf = 0; rbf < _numRightFields; rbf++) {
					double tmpVal = 0;
					for (int qp = 0; qp < _numPoints; qp++) {
						for (int iTens1 = 0; iTens1 < _dim1Tensor; iTens1++) {
							for (int iTens2 = 0; iTens2 < _dim2Tensor; iTens2++) {
								tmpVal +=
									_leftFields(lbf*_numPoints*_dim1Tensor*_dim2Tensor*_numCells +
											qp*_dim1Tensor*_dim2Tensor*_numCells +
											iTens1*_dim2Tensor*_numCells + iTens2*_numCells + elementIndex)
									*_rightFields(rbf*_numPoints*_dim1Tensor*_dim2Tensor*_numCells +
											qp*_dim1Tensor*_dim2Tensor*_numCells +
											iTens1*_dim2Tensor*_numCells + iTens2*_numCells + elementIndex);
							} // D2-loop
						} // D1-loop
					} // P-loop
					_outputFields(lbf*_numRightFields*_numCells +
							rbf*_numCells + elementIndex) = tmpVal;
				} // R-loop
			} // L-loop
		}
};





// Serial contractFieldFieldScalar.  Contracts FieldContainers of doubles.
void contractFieldFieldScalarSerial(FieldContainer<double> &  outputFields,
		const FieldContainer<double> &              leftFields,
		const FieldContainer<double> &              rightFields,
		double *                                    time = 0) {

	// TODO(ellen): Might later want to template this so that both the container
	//              and the scalars inside the container are template arguments,
	//              so we can hand it kokkos views or custom structs.
	int numCells        = leftFields.dimension(0);
	int numLeftFields   = leftFields.dimension(1);
	int numRightFields  = rightFields.dimension(1);
	int numPoints       = leftFields.dimension(2);

	for (int cl = 0; cl < numCells; cl++) {
		for (int lbf = 0; lbf < numLeftFields; lbf++) {
			for (int rbf = 0; rbf < numRightFields; rbf++) {
				double tmpVal = 0;
				for (int qp = 0; qp < numPoints; qp++) {
					tmpVal += leftFields(cl, lbf, qp)*rightFields(cl, rbf, qp);
				} // P-loop
				outputFields(cl, lbf, rbf) = tmpVal;
			} // R-loop
		} // L-loop
	} // C-loop
}


/*
 * Kokkos Cuda contractFieldFieldScalar.
 *
 * Contracts two Kokkos Cuda host views (two double *** tensors -> one double
 * *** tensor). Since
 *
 * Note that all input and output is in Kokkos host views --- the user is
 * responsible for getting the data in and out of them.
 */
template <class DeviceType, class input_view_t, class output_view_t, class input_host_t, class output_host_t>
void contractFieldFieldScalarKokkos(output_host_t &   outHost,
		const input_host_t &                      leftHost,
		const input_host_t &                      rightHost,
		output_view_t &                           outDevice,
		input_view_t &                            leftDevice,
		input_view_t &                            rightDevice,
		double *                                  time = 0) {

	// get sizes
	int numCells        = leftHost.dimension(0);
	int numLeftFields   = leftHost.dimension(1);
	int numRightFields  = rightHost.dimension(1);
	int numPoints       = leftHost.dimension(2);

	// Deep copy Kokkos host views into device views
	Kokkos::deep_copy(leftDevice, leftHost);
	Kokkos::deep_copy(rightDevice, rightHost);
	Kokkos::deep_copy(outDevice, outHost);

	timespec tic;
	if(time != 0)
		clock_gettime(CLOCK_MONOTONIC, &tic);

	contractFieldFieldScalarFunctor<DeviceType, input_view_t, input_view_t, output_view_t>
		kokkosFunctor(leftDevice, rightDevice, outDevice, numLeftFields,
		numRightFields, numPoints);

	Kokkos::parallel_for(numCells, kokkosFunctor);

	Kokkos::fence();

	timespec toc;
	if(time !=0){
		clock_gettime(CLOCK_MONOTONIC, &toc);
		*time += getElapsedTime(tic, toc);
	}

	Kokkos::deep_copy(outHost, outDevice);
}


template <class DeviceType, class input_view_t, class output_view_t, class input_host_t, class output_host_t>
void contractFieldFieldScalarKokkos1D(output_host_t &   outHost,
		const input_host_t &                      leftHost,
		const input_host_t &                      rightHost,
		output_view_t &                           outDevice,
		input_view_t &                            leftDevice,
		input_view_t &                            rightDevice,
		int   numCells,
		int numLeftFields,
		int numRightFields,
		int numPoints,
		int dim1Tensor,
		int dim2Tensor,
		double *                                  time = 0
		) {
	/*
	// get sizes
	int numCells        = leftHost.dimension(0);
	int numLeftFields   = leftHost.dimension(1);
	int numRightFields  = rightHost.dimension(1);
	int numPoints       = leftHost.dimension(2);
	int dim1Tensor      = leftHost.dimension(3);
	int dim2Tensor      = leftHost.dimension(4);
	 */


	// Deep copy Kokkos host views into device views
	Kokkos::deep_copy(leftDevice, leftHost);
	Kokkos::deep_copy(rightDevice, rightHost);
	Kokkos::deep_copy(outDevice, outHost);

	timespec tic;
	if(time != 0)
		clock_gettime(CLOCK_MONOTONIC, &tic);

	contractFieldFieldScalarFunctor1D<DeviceType, input_view_t, input_view_t, output_view_t>
		kokkosFunctor(leftDevice, rightDevice, outDevice, numLeftFields,
				numRightFields, numPoints, dim1Tensor, dim2Tensor, numCells);

	Kokkos::parallel_for(numCells, kokkosFunctor);

	Kokkos::fence();

	timespec toc;
	if(time !=0){
		clock_gettime(CLOCK_MONOTONIC, &toc);
		*time += getElapsedTime(tic, toc);
	}

	Kokkos::deep_copy(outHost, outDevice);
}



int main(int argc, char* argv[]) {
	int c=10000, p=10, l=10, r=10;

	FieldContainer<double> in_c_l_p(c, l, p);
	FieldContainer<double> in_c_r_p(c, r, p);
	FieldContainer<double> out1_c_l_r(c, l, r);
	FieldContainer<double> out2_c_l_r(c, l, r);
	double zero = Intrepid::INTREPID_TOL*100000.0;

	// fill with random numbers
	for (int i=0; i<in_c_l_p.size(); i++) {
		in_c_l_p[i] = Teuchos::ScalarTraits<double>::random();
	}
	for (int i=0; i<in_c_r_p.size(); i++) {
		in_c_r_p[i] = Teuchos::ScalarTraits<double>::random();
	}
	std::cout << "Created vectors" << std::endl;

	// ===============================================================
	// ********************** < Kokkos setup> ************************
	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

	// Doing all of this here might throw off the timing -- we're not counting the
	// cost of the copy into Kokkos or the deep copy from Kokkos host to Kokkos
	// device.

	Kokkos::initialize();

	// Kokkos Cuda views
	typedef Kokkos::View<double ***, Kokkos::Cuda> cuda_input_view_t;
	typedef Kokkos::View<double ***, Kokkos::Cuda> cuda_output_view_t;
	typedef typename cuda_input_view_t::HostMirror cuda_input_host_t;
	typedef typename cuda_output_view_t::HostMirror cuda_output_host_t;

	// Kokkos OpenMP views
	typedef Kokkos::View<double ***, Kokkos::OpenMP> omp_input_view_t;
	typedef Kokkos::View<double ***, Kokkos::OpenMP> omp_output_view_t;
	typedef typename omp_input_view_t::HostMirror omp_input_host_t;
	typedef typename omp_output_view_t::HostMirror omp_output_host_t;


	//Cuda arrays

	double * cudaRight = new double[c * r * p];
	double * cudaLeft = new double[c * l * p];

	double * cudaOut = new double[c * l * r];


	// Make equivalent Kokkos views

	cuda_input_view_t cuda_kokkosLeft("left_input", c,l, p);
	cuda_input_view_t cuda_kokkosRight("right_input", c, r, p);
	cuda_output_view_t cuda_kokkosOut("output", c, l, r );

	omp_input_view_t omp_kokkosLeft("left_input", c, l, p);
	omp_input_view_t omp_kokkosRight("right_input",  c,r, p);
	omp_output_view_t omp_kokkosOut("output", c,l ,r);

	// And their host mirrors

	cuda_input_host_t cuda_hostLeft = Kokkos::create_mirror_view(cuda_kokkosLeft);
	cuda_input_host_t cuda_hostRight = Kokkos::create_mirror_view(cuda_kokkosRight);
	cuda_output_host_t cuda_hostOut = Kokkos::create_mirror_view(cuda_kokkosOut);

	omp_input_host_t omp_hostLeft = Kokkos::create_mirror_view(omp_kokkosLeft);
	omp_input_host_t omp_hostRight = Kokkos::create_mirror_view(omp_kokkosRight);
	omp_output_host_t omp_hostOut = Kokkos::create_mirror_view(omp_kokkosOut);

	// Copy into Kokkos host views and cuda
	// Need to change this so that its 1-D and cl has stride 1
	for (int cl = 0; cl < c; ++cl) {
		for (int qp = 0; qp < p; ++qp) {
			for(int rbf = 0; rbf < r; ++rbf) {
				cuda_hostRight(cl,rbf, qp) = in_c_r_p(cl,rbf,qp);
				omp_hostRight(cl,rbf,qp) = in_c_r_p(cl,rbf,qp);

				cudaRight[cl * p * r + r * qp + rbf] = in_c_r_p(cl,rbf,qp);
			}
			for(int lbf = 0; lbf < l; ++lbf) {
				cuda_hostLeft(cl, lbf, qp) = in_c_l_p(cl,lbf, qp);
				omp_hostLeft(cl,lbf, qp) = in_c_l_p(cl,lbf,qp);

				cudaLeft[cl * p * l + p * lbf + qp] = in_c_l_p(cl,lbf,qp);
			}
			//cudaRightColMajor[cl + c*qp] = in_r_c_p(cl,qp);
			//cudaLeftColMajor[cl + c*qp] = in_l_c_p(cl,qp);

			//cudaRightRowMajor[cl * p + qp] = in_r_c_p(cl,qp);
			//cudaLeftRowMajor[cl * p + qp] = in_l_c_p(cl,qp);
		}
	}



	// ===============================================================
	// ********************** </Kokkos setup> ************************
	// vvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvvv

	std::cout << "trying serial" << std::endl;

	//Warmup
	contractFieldFieldScalarSerial(out2_c_l_r, in_c_l_p, in_c_r_p);

	timespec tic;
	clock_gettime(CLOCK_MONOTONIC, &tic);

	//repeat the calculation 5 times so we can average out some randomness
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarSerial(out2_c_l_r, in_c_l_p, in_c_r_p);
	}

	timespec toc;
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_serial = getElapsedTime(tic, toc);

	printf("trying kokkos openmp\n");
/*
	//Warmpup
	contractFieldFieldScalarKokkos<Kokkos::OpenMP, omp_input_view_t,
		omp_output_view_t, omp_input_host_t, omp_output_host_t>
			(omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
			 omp_kokkosLeft, omp_kokkosRight);
	clock_gettime(CLOCK_MONOTONIC, &tic);
	//repeat the calculation 5 times so we can average out some randomness
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarKokkos<Kokkos::OpenMP, omp_input_view_t,
			omp_output_view_t, omp_input_host_t, omp_output_host_t>
				(omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
				 omp_kokkosLeft, omp_kokkosRight);
	}
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_kokkos_omp = getElapsedTime(tic, toc);
	// Copy out from kokkos output view (NOT timing this)
	for (int cl = 0; cl < c; ++cl) {
		for(int lbf = 0; lbf < l; ++lbf) {
			for(int rbf = 0; rbf < r; ++rbf) {
				out1_c_l_r(cl,lbf,rbf) = omp_hostOut(cl,lbf,rbf);
			}
		}
	}
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check COMP_CPP vs. COMP_KOKKOS; "
			<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
	std::cout << "kokkos omp speedup of " << elapsedTime_serial/elapsedTime_kokkos_omp << std::endl;
*/
/*
	printf("trying kokkos cuda\n");
 
	//Warmpup
	contractFieldFieldScalarKokkos<Kokkos::Cuda, cuda_input_view_t,
		cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>
			(cuda_hostOut, cuda_hostLeft, cuda_hostRight, cuda_kokkosOut,
			 cuda_kokkosLeft, cuda_kokkosRight);
	clock_gettime(CLOCK_MONOTONIC, &tic);
	//repeat the calculation 5 times so we can average out some randomness
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarKokkos<Kokkos::Cuda, cuda_input_view_t,
			cuda_output_view_t, cuda_input_host_t, cuda_output_host_t>
				(cuda_hostOut, cuda_hostLeft, cuda_hostRight, cuda_kokkosOut,
				 cuda_kokkosLeft, cuda_kokkosRight);
	}
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_kokkos_cuda = getElapsedTime(tic, toc);
	// Copy out from kokkos output view (NOT timing this)
	for (int cl = 0; cl < c; ++cl) {
		for(int lbf = 0; lbf < l; ++lbf) {
			for(int rbf = 0; rbf < r; ++rbf) {
				out1_c_l_r(cl,lbf,rbf) = omp_hostOut(cl,lbf,rbf);
			}
		}
	}
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (0): check COMP_CPP vs. COMP_KOKKOS; "
			<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
	std::cout << "kokkos cuda speedup of " << elapsedTime_serial/elapsedTime_kokkos_cuda << std::endl;
*/
	Kokkos::finalize();

	std::cout << "trying cuda major" << std::endl;
	//Now try the cuda version, start with warmup
	cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, &tic, &toc);
	double elapsedTime_cuda = 0;
	
	for(int i = 0; i < 5; ++i){
		cudaDocontractFieldFieldScalar(cudaOut,cudaLeft,cudaRight, c, l, r, p, &tic, &toc);
		elapsedTime_cuda += getElapsedTime(tic,toc);
	}

	for (int cl = 0; cl < c; ++cl) {
		for(int lbf = 0; lbf < l; ++lbf) {
			for(int rbf = 0; rbf < r; ++rbf) {
				out1_c_l_r(cl,lbf,rbf) = cudaOut[cl * l * r + lbf * r + rbf];
			}
		}
	}

	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda; "
		<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}

	std::cout << "cuda speedup of " << elapsedTime_serial/elapsedTime_cuda << std::endl;

	/*
	std::cout << "trying cuda row major" << std::endl;
	//Now try the cuda version, start with warmup
	cudaDocontractFieldFieldScalar(cudaOut,cudaLeftRowMajor,cudaRightRowMajor, c, p, 0);
	clock_gettime(CLOCK_MONOTONIC, &tic);
	for(int i = 0; i < 5; ++i){
		cudaDocontractFieldFieldScalar(cudaOut,cudaLeftRowMajor,cudaRightRowMajor, c, p, 0);
	}
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_cudaRow = getElapsedTime(tic, toc);
	for (int cl = 0; cl < c; ++cl) {
			out1_c_l_r(cl) = cudaOut[cl];
	}
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda; "
		<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
	std::cout << "cuda row major speedup of " << elapsedTime_serial/elapsedTime_cudaRow << std::endl;
	std::cout << "trying cuda thread-per-index" << std::endl;
	//Now try the cuda version, start with warmup
	cudaDocontractFieldFieldScalarUnrolled(cudaOut,cudaLeftRowMajor,cudaRightRowMajor, c, p);
	clock_gettime(CLOCK_MONOTONIC, &tic);
	for(int i = 0; i < 5; ++i){
		cudaDocontractFieldFieldScalarUnrolled(cudaOut,cudaLeftRowMajor,cudaRightRowMajor, c, p);
	}
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_cudaUnrolled = getElapsedTime(tic, toc);
	for (int cl = 0; cl < c; ++cl) {
		out1_c_l_r(cl) = cudaOut[cl];
	}
	rst::subtract(&out1_c_l_r[0], &out2_c_l_r[0], out2_c_l_r.size());
	if (rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) > zero) {
		std::cout << "\n\nINCORRECT contractFieldFieldTensor (1): check cuda unrolled; "
		<< " diff-1norm = " << rst::vectorNorm(&out1_c_l_r[0], out1_c_l_r.size(), Intrepid::NORM_ONE) << "\n\n";
	}
	std::cout << "cuda unrolled speedup of " << elapsedTime_serial/elapsedTime_cudaUnrolled << std::endl;
	*/

#if 0
	//Warmpup
	contractFieldFieldScalarKokkos<Kokkos::OpenMP, omp_input_view_t, omp_output_view_t, omp_input_host_t, omp_output_host_t>
		(omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut,
		 omp_kokkosLeft,omp_kokkosRight); clock_gettime(CLOCK_MONOTONIC, &tic);
	//repeat the calculation 5 times so we can average out some randomness
	for(int i = 0; i < 5; ++i){
		contractFieldFieldScalarKokkos<Kokkos::OpenMP, omp_input_view_t, omp_output_view_t, omp_input_host_t, omp_output_host_t>
			(omp_hostOut, omp_hostLeft, omp_hostRight, omp_kokkosOut, omp_kokkosLeft,
			 omp_kokkosRight);
	}
	clock_gettime(CLOCK_MONOTONIC, &toc);
	const double elapsedTime_kokkos = getElapsedTime(tic, toc);
	// Copy out from kokkos output view (NOT timing this)
	for (int cl = 0; cl < c; ++cl) {
		for (int lbf = 0; lbf < l; ++lbf) {
			for (int rbf = 0; rbf < r; ++rbf) {
				out1_c_l_r(cl, lbf, rbf) = omp_hostOut(cl, lbf, rbf);
			}
		}
	}
#endif

	return 0;
}
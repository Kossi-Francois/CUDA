#include <iostream>
#include <math.h>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <opencv2/core/cuda.hpp>

using namespace cv;
using namespace std;

const string PROJECT_PATH = "/home/franfran/cuda/learning/cuda-course3-lastproject/";
int ROWS, COLS;


/***
 * CUDA kernel to apply convolution on image/vector
 * @param input: input image
 * @param ouput: output image
 * @param img_width: image width
 * @param img_height: image height
 * @param kernelConv: convolution kernel
 * @param kernel_width: kernel width
 * @param kernel_height: kernel height
 */
__global__ void image_convolution_kernel(float *input, float *ouput, const int img_width, const int img_height,
                                        float *kernelConv, const int kernel_width, const int kernel_height )
{

  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if( (x >= img_width) || (y >= img_height) ) return; // thread bound check

  float sum = 0;
  for ( int j = 0; j < kernel_height; j++ ){
    for ( int i = 0; i < kernel_width; i++ ){

      int dX = x + i - kernel_width / 2;
      int dY = y + j - kernel_height / 2;

      dX = max(0, dX);
      dX = min(img_width - 1, dX);

      dY = max(0, dY);
      dY = min(img_height - 1, dY);


      const int idMat = j * kernel_width + i;
      const int idPixel = dY * img_width + dX;
      sum += (float)input[idPixel] * kernelConv[idMat];
    }
  }

  ouput[y * img_width + x] = sum;
}


/***
 * CUDA kernel to compute L2 norm of two images/vectors
 * @param input_A: input image A
 * @param input_B: input image B
 * @param ouput: output image
 * @param img_width: image width
 * @param img_height: image height
 */
__global__ void compute_l2norm(float *input_A, float *input_B, float *ouput, const int img_width, const int img_height )
{
  int x = threadIdx.x + blockIdx.x * blockDim.x;
  int y = threadIdx.y + blockIdx.y * blockDim.y;
  if( (x >= img_width) || (y >= img_height) ) return; // thread bound check

  const int idx = y * img_width + x;
  ouput[idx] = sqrt ( input_A[idx] * input_A[idx] + input_B[idx] * input_B[idx]);
}


/***
 * Save image to disk
 * @param ouptImg: image to save
 * @param fileName: output file name
 */
__host__ void saveImg(float * ouptImg, string fileName = "output.jpg"){
  Mat img(ROWS, COLS, CV_8UC1, Scalar(0));
  for(int r = 0; r < ROWS; ++r){
    for(int c = 0; c < COLS; ++c)
    {
      img.at<uchar>(r, c) = (uchar) max(0.0f, min(ouptImg[r*ROWS + c], 255.0f));
    }
  }
  imwrite(PROJECT_PATH + fileName, img);
  cout << "Image saved:: " << fileName << endl;
}

/***
 * Build convolution kernel
 * @param kernel: kernel values
 * @param kernel_width: kernel width
 * @param kernel_height: kernel height
 * @param factor: factor to multiply kernel values
 * @return tuple with kernel, kernel_width, kernel_height
 */
tuple<float*, int, int> build_convKernel(float *kernel, int kernel_width=3, int kernel_height=3, float factor = 1.0){
  float *conv_kernel; cudaMallocManaged(&conv_kernel, kernel_height*kernel_width*sizeof(float));
  for (int i = 0; i < kernel_height*kernel_width; i++)
  { conv_kernel[i] = kernel[i] * factor;}

  return make_tuple(conv_kernel, kernel_width, kernel_height);
}


/***
 * Get Gaussian convolution kernel
 * @return tuple with kernel, kernel_width, kernel_height
 */
__host__ tuple<float*, int, int> get_gaussian_convKernel(){
  int kernel_height = 3;
  int kernel_width = 3;
  float kernel[] ={ 1,2,1,
                    2,4,2,
                    1,2,1
                  };
  float factor = 1.0/16.0;
  return build_convKernel(kernel, kernel_width, kernel_height, factor);
}


/*** 
 * Get Canny convolution kernel
 * @param isX: true for kx, false for ky
 * @return tuple with kernel, kernel_width, kernel_height
 */
__host__ tuple<float*, int, int> get_canny_convKer(bool isX = true){
  int kernel_height = 3;
  int kernel_width = 3;
  float factor = 1.0;

  float kernel_kx[] ={  -1,  0,  1,
                        -2,  0,  2,
                        -1,  0,  1
  };  

  float kernel_ky[] ={  1,  2,  1,
                        0,  0,  0,
                       -1, -2, -1
  };

  return build_convKernel(isX ? kernel_kx: kernel_ky, kernel_width, kernel_height, factor);
}


/***
 * Process image
 * Apply Gaussian filter, Canny filter kx, Canny filter ky and compute L2 norm of the two canny filters
 * @param filename: image file name
 */
__host__ void processImg(  std::string filename ){

  // Read image
  Mat image = imread( filename, IMREAD_GRAYSCALE );
  ROWS = image.rows; COLS = image.cols;
  int N = image.rows*image.cols;
  cout<< "image read:: "<< filename  << "|| Rows: " << ROWS << " Columns: " << COLS << "\n";

  // Allocate Unified Memory – accessible from CPU or GPU -- for image
  float *img_input; cudaMallocManaged(&img_input, N*sizeof(float));
  float *img_ouput; cudaMallocManaged(&img_ouput, N*sizeof(float));
  for(int r = 0; r < ROWS; ++r){
    for(int c = 0; c < COLS; ++c)
    {
      img_input[r*ROWS + c] = (float)image.at<uchar>(r, c);
      img_ouput[r*ROWS + c] = 0;
    }
  }

  // define GPU block and grid size
  dim3 blocksize(32, 32);
  dim3 gridsize(  (image.cols+blocksize.x-1)/blocksize.x,
                  (image.rows+blocksize.y-1)/blocksize.y
  );


  //Apply convolution // Gaussian filter ==> smoothing
  auto [conv_kernel_gauss, kernel_width_gauss, kernel_height_gauss] = get_gaussian_convKernel();
  image_convolution_kernel<<<gridsize,blocksize>>>(
    img_input, img_ouput, image.cols, image.rows,
    conv_kernel_gauss, kernel_width_gauss, kernel_height_gauss
  );
  cudaDeviceSynchronize();
  saveImg(img_ouput, "output_gauss.jpg");

  //Apply convolution // Canny filter kx ==> vertical edge detection
  auto [conv_kernel_cannyKx, kernel_width_cannyKx, kernel_height_cannyKx] = get_canny_convKer(true);
  // swap(img_input, img_ouput);
  float *img_ouput_cannyKx; cudaMallocManaged(&img_ouput_cannyKx, N*sizeof(float));

  image_convolution_kernel<<<gridsize,blocksize>>>(
    img_ouput, img_ouput_cannyKx, image.cols, image.rows,
    conv_kernel_cannyKx, kernel_width_cannyKx, kernel_height_cannyKx
  );
  cudaDeviceSynchronize();
  saveImg(img_ouput_cannyKx, "output_cannyKx.jpg");


  //Apply convolution // Canny filter ky ==> horizontal edge detection
  auto [conv_kernel_cannyKy, kernel_width_cannyKy, kernel_height_cannyKy] = get_canny_convKer(false); 
  swap(img_input, img_ouput);
  image_convolution_kernel<<<gridsize,blocksize>>>(
    img_input, img_ouput, image.cols, image.rows,
    conv_kernel_cannyKy, kernel_width_cannyKy, kernel_height_cannyKy
  );
  cudaDeviceSynchronize();
  saveImg(img_ouput, "output_cannyKy.jpg");


  //Compute L2 norm
  swap(img_input, img_ouput);
  compute_l2norm<<<gridsize,blocksize>>>(img_ouput_cannyKx, img_input, img_ouput, image.cols, image.rows);
  cudaDeviceSynchronize();
  saveImg(img_ouput, "output_canny.jpg");


  //implement step 4 and 5 :: non-maximum suppression and hysteresis thresholding



  // Free memory
  cudaFree(img_input);
  cudaFree(img_ouput);
  cudaFree(img_ouput_cannyKx);
  cudaFree(conv_kernel_gauss);
  cudaFree(conv_kernel_cannyKx);
  cudaFree(conv_kernel_cannyKy);
}



int main(void)
{
  std::string filename; std::cin >> filename;
  std::string imgPath = samples::findFile( PROJECT_PATH + "lena.jpg");

  processImg( imgPath);

  // Wait for GPU to finish before accessing on host
  cudaDeviceSynchronize();
  cudaError_t error = cudaGetLastError();
  if(error != cudaSuccess)
  {
    //print the CUDA error message and exit
    cout<< "CUDA kernel error::" << cudaGetErrorString(error) << std::endl;
    exit(-1);
  }

  return 0;
}
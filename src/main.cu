#include <cstdlib>
#include <string>
#include <iostream>
#include <stdio.h>
#include <chrono>
// =========================
// CUDA imports 
// =========================
#include <cuda_runtime.h>
#include "lbm/LBMSolver.h" 
#include "CudaTimer.h"

int main(int argc, char* argv[])
{
  // read parameter file and initialize parameter
  // parse parameters from input file
  std::string input_file = argc>1 ? std::string(argv[1]) : "flowAroundCylinder.ini";

  ConfigMap configMap(input_file);

  // create a timer====================
  CudaTimer copyTimer;
  //===================================
  // start time measurement============
  copyTimer.start();
  //===================================

  // test: create a LBMParams object
  LBMParams params = LBMParams();
  params.setup(configMap);

  // print parameters on screen
  params.print();


      // Début du chronomètre
  auto start = std::chrono::high_resolution_clock::now(); // l'heure du début de calcul 

  LBMSolver* solver = new LBMSolver(params);

  solver->run();

  auto end = std::chrono::high_resolution_clock::now(); // L'heure de la fin d'exécution


    // Calcul de la durée écoulée en secondes et en nanosecondes
  auto duration_seconds = std::chrono::duration_cast<std::chrono::seconds>(end - start);
  auto duration_nanoseconds = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start);

  auto duration_millisecond = std::chrono::duration_cast<std::chrono::milliseconds>(end - start);

  std::cout << "Temps d'execution : ============================================================\n" 
            << "secondes     : "<<duration_seconds.count() << "    |    " 
            << "milliseconds : " <<duration_millisecond.count() << "    |    " 
            << "nanosecondes : " <<duration_nanoseconds.count() << std::endl;
  std::cout << "================================================================================\n";


  copyTimer.stop();

  // print bandwidth:=====================
  // {
  //   long  numBytes = 7257600000; // factor 2 because 1 read + 1 write
  //   printf("bandwidth is %f GBytes (%f)/s\n",
  //    1e-9*numBytes/copyTimer.elapsed_in_second(),
  //    copyTimer.elapsed_in_second() );

  // }
  // print peak bandwidth=================
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, 0);
    printf("  Peak Memory Bandwidth (GB/s): %f\n",
     2.0*deviceProp.memoryClockRate*(deviceProp.memoryBusWidth/8)/1.0e6);
  }
//========================================

  return EXIT_SUCCESS;
}

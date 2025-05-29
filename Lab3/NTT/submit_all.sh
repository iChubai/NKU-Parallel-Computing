#!/bin/bash
# g++ -std=c++17 -O2 -o cache_radix4_openmp openmp/src/main_cache_radix-4_openmp.cc -fopenmp
# mv cache_radix4_openmp main
# bash test.sh  3 1 4
# rm main

# g++ -std=c++17 -O2 -o crt_openmp openmp/src/main_CRT_openmp.cc -fopenmp
# mv crt_openmp main
# bash test.sh  3 1 4
# rm main

# g++ -std=c++17 -O2 -o montgomery_bigp_openmp openmp/src/main_openmp_Montgomery_BigP.cc -fopenmp
# mv montgomery_bigp_openmp main
# bash test.sh  3 1 4
# rm main

# g++ -std=c++17 -O2 -o montgomery_openmp openmp/src/main_openmp_Montgomery.cc -fopenmp
# mv montgomery_openmp main
# bash test.sh  3 1 4
# rm main

# g++ -std=c++17 -O2 -o openmp_v1 openmp/src/main_openmp_v1.cc -fopenmp
# mv openmp_v1 main
# bash test.sh  3 1 4
# rm main

# g++ -std=c++17 -O2 -o openmp_v2 openmp/src/main_openmp_v2.cc -fopenmp
# mv openmp_v2 main
# bash test.sh  3 1 4
# rm main

# g++ -std=c++17 -O2 -o split_radix_final openmp/src/split_radix.cc -fopenmp
# mv split_radix_final main
# bash test.sh  3 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/main_pthread_DIF_DIT.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/main_pthread.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/main_pthread_v1.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/main_pthread_v2.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/main_pthread_v3.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/main_radix-4_pthread.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/simple_three_mod_crt.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

# g++ -O2 -march=native -pthread pthread/src/crt_ptread.cc -o ntt_pth
# mv ntt_pth main
# bash test.sh  2 1 4
# rm main

g++ -O2 -march=native -pthread pthread/src/split_radix_pthread_v1.cc -o ntt_pth
mv ntt_pth main
bash test.sh  2 1 4 
rm main

g++ -O2 -march=native -pthread pthread/src/split_radix_pthread_v2.cc -o ntt_pth
mv ntt_pth main
bash test.sh  2 1 4
rm main

g++ -O2 -march=native -pthread pthread/src/split_radix_pthread_v3.cc -o ntt_pth
mv ntt_pth main
bash test.sh  2 1 4
rm main

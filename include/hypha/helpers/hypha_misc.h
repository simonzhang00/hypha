/*  Copyright 2018, 2019 The Ohio State University
    Contributed by: Simon Zhang, Mengbai Xiao, Chengxin Guo, Liang Geng, Hao Wang, Xiaodong Zhang
    HYPHA makes calls to the PHAT library: (Ulrich Bauer, Michael Kerber, Jan Reininghaus, Hubert Wagner)

    This file is part of HYPHA.

    HYPHA is free software: you can redistribute it and/or modify
    it under the terms of the GNU Lesser General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    HYPHA is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU Lesser General Public License for more details.

    You should have received a copy of the GNU Lesser General Public License
    along with HYPHA.  If not, see <http://www.gnu.org/licenses/>. */
#pragma once

// STL includes
#include <iostream>
#include <fstream>
#include <string>
#include <vector>
#include <set>
#include <list>
#include <map>
#include <algorithm>
#include <queue>
#include <cassert>
#include <sstream>
#include <algorithm>
#include <iomanip>
#include <cmath>
#include <cstdlib>
#include <iterator>

// VS2008 and below unfortunately do not support stdint.h
#if defined(_MSC_VER)&& _MSC_VER < 1600
typedef __int8 int8_t;
    typedef unsigned __int8 uint8_t;
    typedef __int16 int16_t;
    typedef unsigned __int16 uint16_t;
    typedef __int32 int32_t;
    typedef unsigned __int32 uint32_t;
    typedef __int64 int64_t;
    typedef unsigned __int64 uint64_t;
#else
#include <stdint.h>
#endif

namespace hypha {
    //typdef int index;
    typedef int64_t index;//CHANGE THIS TO typedef int index IFF phat/helpers/misc.h has typdef int index
    typedef int8_t dimension;
    typedef std::vector <index> column;

    enum index_type { COMPRESSIBLE, INCOMPRESSIBLE, UNKNOWN};

    //#define MAXINT 1024*1024*512//avoid using this if you can. Assume nr_columns<=MAXINTEGER=2^31-1
    #define GPU_CHUNK_SIZE (128*1024*1024)
    #define MAX_GPU_CHUNK_NR 1024

    //translation functions to convert an offset to double pointer indices for the CSC data matrix
    #define BLOCK_ID(off) ((off) / GPU_CHUNK_SIZE)
    #define BLOCK_OFF(off) ((off) % GPU_CHUNK_SIZE)
    #define BLOCK_ST(off) ((BLOCK_ID(off)) * GPU_CHUNK_SIZE)
    #define NEXT_BLOCK_ST(off) ((BLOCK_ID(off) + 1) * GPU_CHUNK_SIZE)

    #define CUDACHECK(cmd) do {\
    cudaError_t e = cmd;\
    if( e != cudaSuccess ) {\
        printf("Failed: Cuda error %s:%d '%s'\n",\
        __FILE__,__LINE__,cudaGetErrorString(e));\
    exit(EXIT_FAILURE);\
    }\
    } while(0)
    #if defined(_MSC_VER)
        #include <BaseTsd.h>
        typedef SSIZE_T ssize_t;
    #endif
}

#include <phat/helpers/thread_local_storage.h>


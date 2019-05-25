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

#include <phat/helpers/misc.h>
#include <hypha/algorithms/hypha_accelerator.h>
#include <hypha/helpers/hypha_misc.h>
#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <hypha/helpers/stopwatch.h>
#include <thrust/scan.h>
#include <thrust/execution_policy.h>
#include <thrust/system/cuda/execution_policy.h>
#include <cub/cub.cuh>
#include <cub/grid/grid_barrier.cuh>
#include <thread>
#include <mutex>
#include <algorithm>

namespace hypha{
    template< class Representation = phat::bit_tree_pivot_column >
    class hypha_boundary_matrix : public phat::boundary_matrix<Representation>
    {
        // variables for GPU and I/O with GPU and PHAT's boundary_matrix
    protected:

        volatile bool flag = false;
        std::mutex mtx;
        int grid_size;
        cub::GridBarrierLifetime global_barrier;

        // Matrix buffer (mb) in CSC format for CPU side
        /* CPU Storage */
        index nr_columns;
        index *h_mb;//CPU side matrix buffer
        index *h_mb_col_offs;//offsets/indices array for CSC matrix format
        index *h_mb_col_nrs;//array of column lengths per column index

        //matrix in CSC format for GPU side
        /* GPU Storage: d_ prefix means pointer to GPU address*/
        index *d_lookup = NULL;//pivot lookup table d_lookup[i]=j means there is a pivot (i,j) if j>=0; d_lookup[i]=-1 is default value; d_lookup[i]=-2 means there cannot be a pivot in row i
        index *d_leftmostnz = NULL;//d_leftmostnz[r]= c means the leftmost nonzero in row r is at column c
        index *d_unstables = NULL;//the sorted set of unstable column indices
        index **h_gpuaddresses_matrix = NULL;//intermediate CPU side double pointer data structure for storing GPU addresses
        index *d_col_nrs = NULL;//array of column lengths per column index
        index *d_col_offs = NULL;//d_col_offs is the offsets array for CSC matrix on GPU
        index *d_us_num = NULL;//number of unstable columns
        index *d_stable_flags = NULL;//d_stable_flags[i]= 1 if column i is stable, 0 otherwise.
        index **d_matrix = NULL;//d_matrix is a double pointer for storing the CSC matrix indices. see hypha_misc.h for how to convert an offset index to a double pointer index.
        //d_matrix is a double pointer in order to alleviate excess memory allocation time for GPU device memory.

        //types ssize_t and int64_t should be equivalent
        ssize_t mb_tail;//keeps track of number of nonzeros stored in h_mb
        const size_t mb_incr = 100 * 1024 * 1024;//realloc memory increment size

        bool compression= true;//toggle compression on CPU side here

        //interface of hypha_boundary_matrix:
    public:

        //use this meta data from GPU after GPU_scan
        index *lookup, *unstables;//lookup table and unstables are public
        index* unstable_nr;//number of unstables is also public

        bool with_compression(){
            return compression;
        }

        void set_compression(bool comp){
            compression= comp;
        }

        //run this after GPU is loaded with the boundary matrix
        void GPU_scan(index number_of_columns)
        {
            cudaDeviceProp deviceProp;
            cudaGetDeviceProperties(&deviceProp, 0);

            cudaOccupancyMaxActiveBlocksPerMultiprocessor( &grid_size, scan_kernel, 256, 0);
            grid_size  *= deviceProp.multiProcessorCount;
            global_barrier.Setup(grid_size);


            cudaHostGetDevicePointer(&d_us_num,unstable_nr,0);

            hypha::scan_kernel<<<grid_size,256>>>(d_matrix,number_of_columns,d_leftmostnz,d_lookup,global_barrier,d_col_nrs,d_col_offs,d_stable_flags,d_us_num,d_unstables);//,1, d_maxleftfixed);//, d_negative);

            cudaDeviceSynchronize();
            printf("number of unstable columns: %d out of %d\n", *unstable_nr, number_of_columns);

            //don't forget to sort unstable columns
            thrust::sort(thrust::device, d_unstables, d_unstables + (*unstable_nr));


            cudaMemcpy(unstables, d_unstables, sizeof(index)*(*unstable_nr), cudaMemcpyDeviceToHost);//unstable columns sorted.

            cudaMemcpy(lookup, d_lookup, sizeof(index)*number_of_columns, cudaMemcpyDeviceToHost);//fully reduced column corresponding to a given row

        }

        hypha_boundary_matrix() : h_mb(nullptr), h_mb_col_offs(nullptr), h_mb_col_nrs(nullptr), mb_tail(0), lookup(nullptr), unstables(nullptr), unstable_nr(nullptr)
        {}
        //I/O helper methods
    protected:

        void check_gpu_mem(){
            //IT IS RECOMMENDED THAT A GPU WITH >=8GB DEVICE MEMORY IS USED, HOWEVER SMALLER DATASETS WILL RUN ON WEAKER GPU's e.g. on GTX 780 with ~3 GB global memory
            cudaDeviceProp devProp;
            cudaGetDeviceProperties(&devProp, 0);
            int64_t total_globalmem= devProp.totalGlobalMem;

            //if GPU DRAM size is not big enough, you will get errors when running on large datasets
            //beware: we need atleast nnz*8 total global mem just for the boundary matrix
            printf("GPU has %lf gigabytes of global memory\n", total_globalmem/1000.0/1000.0/1000.0);
        }
        //call this from a second thread; used by hybrid
        void alloc_gpu_mem(index number_of_columns)
        {
            CUDACHECK(cudaMalloc((void **)&d_lookup, sizeof(index)*number_of_columns));
            CUDACHECK(cudaMalloc((void **)&d_leftmostnz, sizeof(index)*number_of_columns));
            CUDACHECK(cudaMalloc((void **)&d_unstables, sizeof(index)*number_of_columns));

            h_gpuaddresses_matrix = (index **)malloc(sizeof(index*) * MAX_GPU_CHUNK_NR);
            assert(h_gpuaddresses_matrix != NULL);

            memset(h_gpuaddresses_matrix, NULL, MAX_GPU_CHUNK_NR);

            CUDACHECK(cudaMalloc((void **)&d_stable_flags, sizeof(index)*number_of_columns));
            CUDACHECK(cudaMalloc((void **)&d_col_nrs, sizeof(index)*number_of_columns));
            CUDACHECK(cudaMalloc((void **)&d_col_offs, sizeof(index)*number_of_columns));
            printf("num columns: %d\n", number_of_columns);

            //set lookup tables to -1's (use memset instead?)
            CUDACHECK(cudaMemcpy(d_lookup,lookup,sizeof(index)*number_of_columns,cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(d_leftmostnz,lookup,sizeof(index)*number_of_columns,cudaMemcpyHostToDevice));
            CUDACHECK(cudaMemcpy(d_unstables,unstables,sizeof(index)*number_of_columns,cudaMemcpyHostToDevice));

            CUDACHECK(cudaMemset(d_stable_flags,0,sizeof(index)*number_of_columns));

            CUDACHECK(cudaHostAlloc((void **)&unstable_nr, sizeof(index), cudaHostAllocPortable | cudaHostAllocMapped));

            index cur_col = 0;
            index cp_col_st = cur_col;
            size_t read_indices = 0;

            while(cur_col < number_of_columns)
            {
                while(h_mb_col_nrs[cur_col]==-1)//h_mb_col_nrs[cur_col] is changed from -1 LAST in _set_mb_col()
                {
                    if(flag){
                        printf("Load %d Columns\n",number_of_columns);
                        flag = false;
                    }
                }

                read_indices += h_mb_col_nrs[cur_col];
                if(cur_col != number_of_columns - 1 && read_indices <  GPU_CHUNK_SIZE) {
                    cur_col++;
                    continue;
                }

                index head = h_mb_col_offs[cp_col_st];
                index head_bid = BLOCK_ID(head);
                index head_boff = BLOCK_OFF(head);
                index tail = h_mb_col_offs[cur_col] + h_mb_col_nrs[cur_col];
                index tail_bid = BLOCK_ID(tail);
                for(index bid = head_bid; bid <= tail_bid; bid++){
                    if(h_gpuaddresses_matrix[bid] == nullptr){
                        CUDACHECK( cudaMalloc(h_gpuaddresses_matrix + bid, sizeof(index) * GPU_CHUNK_SIZE) );
                    }
                }

                CUDACHECK( cudaMemcpy(d_col_offs + cp_col_st, h_mb_col_offs + cp_col_st, sizeof(index) * (cur_col + 1 - cp_col_st), cudaMemcpyHostToDevice) ); // col_offs
                CUDACHECK( cudaMemcpy(d_col_nrs + cp_col_st, h_mb_col_nrs + cp_col_st, sizeof(index) * (cur_col + 1 - cp_col_st), cudaMemcpyHostToDevice) ); // col_nrs

                mtx.lock();
                if(head_bid != tail_bid){//the data is split between atleast two gpu_matrix blocks (1st dim)
                    CUDACHECK( cudaMemcpy(h_gpuaddresses_matrix[head_bid] + head_boff, h_mb + head, sizeof(index) * (NEXT_BLOCK_ST(head) - head), cudaMemcpyHostToDevice) );
                    for(index bid = BLOCK_ID(head) + 1; bid < BLOCK_ID(tail); bid++){
                        CUDACHECK( cudaMemcpy(h_gpuaddresses_matrix[bid], h_mb + bid * GPU_CHUNK_SIZE , sizeof(index) * GPU_CHUNK_SIZE, cudaMemcpyHostToDevice) );
                    }
                    CUDACHECK( cudaMemcpy(h_gpuaddresses_matrix[tail_bid], h_mb + BLOCK_ST(tail), sizeof(index) * (tail - BLOCK_ST(tail)), cudaMemcpyHostToDevice) );
                }else{
                    CUDACHECK( cudaMemcpy(h_gpuaddresses_matrix[head_bid] + head_boff, h_mb + head, sizeof(index) * (tail - head), cudaMemcpyHostToDevice) );
                }
                mtx.unlock();
                cur_col++;
                cp_col_st = cur_col;
                read_indices = 0;
            }

            *unstable_nr = 0;
            CUDACHECK(cudaMalloc((void **)&d_matrix,sizeof(index*)*MAX_GPU_CHUNK_NR));
            CUDACHECK(cudaMemcpy(d_matrix,h_gpuaddresses_matrix,sizeof(index*)*MAX_GPU_CHUNK_NR,cudaMemcpyHostToDevice));
        }

        void _set_arrays_for_gpu (index number_of_columns){
            nr_columns= number_of_columns;
            h_mb_col_offs = (index *) malloc ( number_of_columns * sizeof(index) );
            h_mb_col_nrs = (index *) malloc ( number_of_columns * sizeof(index) );
            lookup = (index *) malloc ( number_of_columns * sizeof(index) );
            unstables = (index *) malloc ( number_of_columns * sizeof(index) );

            assert ( h_mb_col_offs != nullptr && h_mb_col_nrs != nullptr && lookup != nullptr && unstables != nullptr );

            std::fill(lookup, lookup + number_of_columns, -1);
            std::fill(unstables, unstables + number_of_columns, -1);
            std::fill(h_mb_col_nrs, h_mb_col_nrs + number_of_columns, -1);
        }


        void _set_mb_col (index col_id, std::vector<index> &col){
            size_t _free = mb_tail % mb_incr == 0 ? 0 : (mb_incr - mb_tail % mb_incr);
            if( _free < col.size() ){
                size_t _incr = _free;
                do{
                    _incr += mb_incr;
                }while(_incr < col.size());
                mtx.lock();
                h_mb = (index *) realloc( h_mb, (mb_tail + _incr) * sizeof(index) );
                assert(h_mb != nullptr);
                mtx.unlock();
            }
            h_mb_col_offs[col_id] = mb_tail;

            for(auto &e : col)
                h_mb[mb_tail++] = e;

            h_mb_col_nrs[col_id] = col.size();//this is the last operation done to change -1 to col.size()>=0
        }

        //I/O functions for hypha_boundary_matrix
    public:
        // Loads boundary_matrix from given file
        // Format: nr_columns % dim1 % N1 % row1 row2 % ...% rowN1 % dim2 % N2 % ...
        bool load_binary_hybrid( std::string filename ) {

            check_gpu_mem();

            std::ifstream input_stream( filename.c_str( ), std::ios_base::binary | std::ios_base::in );
            if( input_stream.fail( ) )
                return false;

            //TODO Note that if you read a file that isn't actu5ally a data file, you may get
            //a number of columns that is bigger than the available memory, which leads to crashes
            //with deeply confusing error messages. Consider ways to prevent this.
            //Magic number in the file header? Check for more columns than bytes in the file?
            int64_t nr_columns;
            input_stream.read( (char*)&nr_columns, sizeof( int64_t ) );
            this->set_num_cols( (index)nr_columns );
            std::unique_ptr<std::thread> t1;

            _set_arrays_for_gpu( nr_columns );

            t1 = std::unique_ptr<std::thread>(new std::thread(&hypha_boundary_matrix::alloc_gpu_mem, this, nr_columns));

            column temp_col;
            for( index cur_col = 0; cur_col < nr_columns; cur_col++ ) {
                int64_t cur_dim;
                input_stream.read( (char*)&cur_dim, sizeof( int64_t ) );
                this->set_dim( cur_col, (dimension)cur_dim );
                int64_t nr_rows;
                input_stream.read( (char*)&nr_rows, sizeof( int64_t ) );
                temp_col.resize( ( std::size_t )nr_rows );
                for( index idx = 0; idx < nr_rows; idx++ ) {
                    int64_t cur_row;
                    input_stream.read( (char*)&cur_row, sizeof( int64_t ) );
                    temp_col[ idx ] = (index)cur_row;
                }

                this->set_col( cur_col, temp_col );

                _set_mb_col( cur_col, temp_col );
            }
            flag = true;
            input_stream.close( );

            t1->join();

            printf("--GPU LOADED FOR HYBRID COMPUTATION--\n");

            return true;
        }
        // Loads the boundary_matrix from given file in ascii format
        // Format: each line represents a column, first number is dimension, other numbers are the content of the column.
        // Ignores empty lines and lines starting with a '#'.
        bool load_ascii_hybrid( std::string filename ) {

            check_gpu_mem();

            // first count number of columns:
            std::string cur_line;
            std::ifstream dummy( filename .c_str() );
            if( dummy.fail() )
                return false;
            index number_of_columns = 0;
            while( getline( dummy, cur_line ) ) {
                cur_line.erase(cur_line.find_last_not_of(" \t\n\r\f\v") + 1);
                if( cur_line != "" && cur_line[ 0 ] != '#' )
                    number_of_columns++;

            }
            this->set_num_cols( number_of_columns );
            std::unique_ptr<std::thread> t1;

            _set_arrays_for_gpu( number_of_columns );
            t1 = std::unique_ptr<std::thread>(new std::thread(&hypha_boundary_matrix::alloc_gpu_mem, this, number_of_columns));


            dummy.close();

            std::ifstream input_stream( filename.c_str() );
            if( input_stream.fail() )
                return false;

            column temp_col;
            index cur_col = -1;

            while( getline( input_stream, cur_line ) ) {
                cur_line.erase(cur_line.find_last_not_of(" \t\n\r\f\v") + 1);
                if( cur_line != "" && cur_line[ 0 ] != '#' ) {
                    cur_col++;
                    std::stringstream ss( cur_line );

                    int64_t temp_dim;
                    ss >> temp_dim;
                    this->set_dim( cur_col, (dimension) temp_dim );

                    int64_t temp_index;
                    temp_col.clear();
                    while( ss.good() ) {
                        ss >> temp_index;
                        temp_col.push_back( (index)temp_index );
                    }
                    std::sort( temp_col.begin(), temp_col.end() );
                    this->set_col( cur_col, temp_col );

                    _set_mb_col(cur_col, temp_col);

                }
            }

            flag = true;
            input_stream.close();

            t1->join();
            printf("--GPU LOADED FOR HYBRID COMPUTATION--\n");

            return true;
        }

        //frees up all pointers from hypha_boundary_matrix
        void free_memory(){
            for(index i=0; i<MAX_GPU_CHUNK_NR; i++){
                cudaFree(h_gpuaddresses_matrix[i]);
            }
            free(h_gpuaddresses_matrix);
            free(h_mb);
            free(h_mb_col_nrs);
            free(h_mb_col_offs);

            cudaFree(d_matrix);
            cudaFree(d_col_offs);
            cudaFree(d_col_nrs);
            cudaFree(d_lookup);
            cudaFree(d_leftmostnz);
            cudaFree(d_stable_flags);
            cudaFree(d_unstables);
            cudaFree(d_us_num);

        }

    };
}

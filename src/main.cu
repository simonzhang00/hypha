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

#include <phat/compute_persistence_pairs.h>

#include <phat/representations/vector_vector.h>
#include <phat/representations/vector_heap.h>
#include <phat/representations/vector_set.h>
#include <phat/representations/vector_list.h>
#include <phat/representations/sparse_pivot_column.h>
#include <phat/representations/heap_pivot_column.h>
#include <phat/representations/full_pivot_column.h>
#include <phat/representations/bit_tree_pivot_column.h>

#include <hypha/helpers/stopwatch.h>

#include <phat/algorithms/twist_reduction.h>
#include <hypha/algorithms/hypha_twist_reduction.h>

#include <phat/algorithms/standard_reduction.h>
#include <phat/algorithms/row_reduction.h>
#include <phat/algorithms/chunk_reduction.h>
#include <phat/algorithms/spectral_sequence_reduction.h>

#include <phat/helpers/dualize.h>
#include <hypha/hypha_boundary_matrix.h>
#include <hypha/algorithms/hypha_accelerator.h>
#include <hypha/hypha_compute_persistence_pairs.h>
#include <hypha/algorithms/hypha_twist_reduction.h>
#include <hypha/algorithms/hypha_spectral_sequence_reduction.h>

enum Representation_type { VECTOR_VECTOR, VECTOR_HEAP, VECTOR_SET, VECTOR_LIST,
            SPARSE_PIVOT_COLUMN, FULL_PIVOT_COLUMN, BIT_TREE_PIVOT_COLUMN, HEAP_PIVOT_COLUMN };
enum Algorithm_type  {STANDARD,
    TWIST, ROW, CHUNK, CHUNK_SEQUENTIAL, SPECTRAL_SEQUENCE, HYPHA_SPECTRAL_SEQUENCE, HYPHA_TWIST };



void print_help() {
    std::cerr << "Usage: " << "hypha " << "[options] input_filename output_filename" << std::endl;
    std::cerr << std::endl;

    std::cerr << "Options:" << std::endl;
    std::cerr << std::endl;

    std::cerr << "--ascii         --  use ascii file format" << std::endl;
    std::cerr << "--binary        --  use binary file format (default)" << std::endl;
    std::cerr << "--help          --  prints this screen" << std::endl;
    std::cerr << "--verbose       --  verbose is always on" << std::endl;
    std::cerr << std::endl;

    std::cerr << "--dual_input    --  convert pairs to persistent homology pairs from reduction of a dualized input boundary matrix" << std::endl;
    std::cerr << "--nocompression --  removes compression (may be beneficial with dualized input)" <<std::endl;
    std::cerr << std::endl;

    std::cerr << "--vector_vector, --full_pivot_column, --sparse_pivot_column, --heap_pivot_column, --bit_tree_pivot_column"<<std::endl;
    std::cerr << "--  selects a representation data structure for boundary matrices (default is '--bit_tree_pivot_column')" << std::endl;
    std::cerr << std::endl;

    std::cerr << "--twist, --spectral_sequence" <<std::endl;
    std::cerr << "--  selects a final stage matrix reduction algorithm (default is '--twist')" << std::endl;
}

void print_help_and_exit() {
    print_help();
    exit( EXIT_FAILURE );
}

void parse_command_line( int argc, char** argv, bool& use_binary, Representation_type& representation, Algorithm_type& algorithm,
                         std::string& input_filename, std::string& output_filename, bool& verbose, bool& dualize, bool& ga, bool& dual_input, bool& compression){

    if( argc < 3 ) print_help_and_exit();

    input_filename = argv[ argc - 2 ];
    output_filename = argv[ argc - 1 ];

    for( int idx = 1; idx < argc - 2; idx++ ) {
        const std::string option = argv[ idx ];


        if( option == "--ascii" ) use_binary = false;
        else if( option == "--binary" ) use_binary = true;
        else if( option == "--vector_vector" ) representation = VECTOR_VECTOR;
        else if( option == "--full_pivot_column" )  representation = FULL_PIVOT_COLUMN;
        else if( option == "--bit_tree_pivot_column" )  representation = BIT_TREE_PIVOT_COLUMN;
        else if( option == "--sparse_pivot_column" ) representation = SPARSE_PIVOT_COLUMN;
        else if( option == "--heap_pivot_column" ) representation = HEAP_PIVOT_COLUMN;
        else if( option == "--twist" ) algorithm = TWIST;
        else if( option == "--spectral_sequence" ) algorithm = SPECTRAL_SEQUENCE;
        else if( option == "--verbose" ) verbose = true;//verbose is always on
        else if( option == "--dual_input" ) dual_input= true;
        else if( option == "--nocompression" ) compression= false;
        else if( option == "--help" ) print_help_and_exit();
        else print_help_and_exit();
    }

    if(ga==true) {
        if(representation==VECTOR_HEAP || representation==VECTOR_LIST || representation==VECTOR_SET || dualize==true
           || algorithm==STANDARD || algorithm==CHUNK || algorithm==CHUNK_SEQUENTIAL || algorithm==ROW ){
            print_help_and_exit();
        }

        cudaDeviceProp deviceProp;
        CUDACHECK( cudaGetDeviceProperties(&deviceProp, 0) );
        std::cout << "GPU detected: " << deviceProp.name << std::endl;
        std::cout << "CUDA Compute Capability: " << deviceProp.major << "." << deviceProp.minor << std::endl;

        //update algorithm type if gpu_assist is toggled
        if(algorithm==TWIST){
            algorithm= HYPHA_TWIST;
        }else if (algorithm==SPECTRAL_SEQUENCE){
            algorithm= HYPHA_SPECTRAL_SEQUENCE;
        }
    }else{
        print_help_and_exit();
    }
}

#define LOG(msg) if( verbose ) std::cout << msg << std::endl;

template<typename Representation, typename Algorithm>
void compute_pairing( std::string input_filename, std::string output_filename, bool use_binary, bool verbose, bool dualize, bool ga, bool dual_input, bool compression){

    if(ga) {
        hypha::hypha_boundary_matrix<Representation> matrix_hypha;

        matrix_hypha.set_compression(compression);
        bool read_successful;
        double read_timer = omp_get_wtime();

        if( use_binary ) {
            LOG( "Reading input file " << input_filename << " in binary mode" )
            read_successful = matrix_hypha.load_binary_hybrid( input_filename );
        } else {
            LOG( "Reading input file " << input_filename << " in ascii mode" )
            read_successful = matrix_hypha.load_ascii_hybrid( input_filename );
        }

        double read_time = omp_get_wtime() - read_timer;
        double read_time_rounded = read_time;
        LOG( "Reading input file took " << std::setiosflags( std::ios::fixed ) << std::setiosflags( std::ios::showpoint ) << std::setprecision( 5 ) << read_time_rounded <<"s" )

        if( !read_successful ) {
            std::cerr << "Error opening file " << input_filename << std::endl;
            print_help_and_exit();
        }

        hypha::index num_cols = matrix_hypha.get_num_cols();

        double pairs_timer = omp_get_wtime();
        phat::persistence_pairs pairs;
        LOG( "Computing persistence pairs ..." )
        if(std::is_same<Algorithm, hypha::hypha_twist_reduction>::value){
            hypha::hypha_twist_reduction hypha_twist;
            hypha::hypha_compute_persistence_pairs<hypha::hypha_twist_reduction>(hypha_twist, pairs, matrix_hypha );
        }else if(std::is_same<Algorithm, hypha::hypha_spectral_sequence_reduction>::value){
            hypha::hypha_spectral_sequence_reduction hypha_spectral_sequence;
            hypha::hypha_compute_persistence_pairs<hypha::hypha_spectral_sequence_reduction>(hypha_spectral_sequence, pairs, matrix_hypha );
        }

        if(dual_input) {
            LOG("Dualizing pairs ...")
            dualize_persistence_pairs( pairs, num_cols );
        }
        double pairs_time = omp_get_wtime() - pairs_timer;
        double pairs_time_rounded= pairs_time;

        LOG( "Computing persistence pairs took " << std::setiosflags( std::ios::fixed ) << std::setiosflags( std::ios::showpoint ) << std::setprecision( 5 ) << pairs_time_rounded <<"s" )

        double write_timer = omp_get_wtime();
        if( use_binary ) {
            LOG( "Writing output file " << output_filename << " in binary mode ..." )
            pairs.save_binary( output_filename );
        } else {
            LOG( "Writing output file " << output_filename << " in ascii mode ..." )
            pairs.save_ascii( output_filename );
        }
        double write_time = omp_get_wtime() - write_timer;
        double write_time_rounded = write_time;
        LOG( "Writing output file took " << std::setiosflags( std::ios::fixed ) << std::setiosflags( std::ios::showpoint ) << std::setprecision( 5 ) << write_time_rounded <<"s" )

        matrix_hypha.free_memory();
    }else {
        print_help_and_exit();
    }


}

#define COMPUTE_PAIRING(Representation) \
    switch( algorithm ) { \
    case STANDARD: compute_pairing< phat::Representation, phat::standard_reduction> ( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
    case TWIST: compute_pairing< phat::Representation, phat::twist_reduction> ( input_filename, output_filename, use_binary, verbose, dualize , ga, dual_input, compression); break; \
    case ROW: compute_pairing< phat::Representation, phat::row_reduction >( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
    case SPECTRAL_SEQUENCE: compute_pairing< phat::Representation, phat::spectral_sequence_reduction >( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
    case CHUNK: compute_pairing< phat::Representation, phat::chunk_reduction >( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
    case HYPHA_TWIST: compute_pairing< phat::Representation, hypha::hypha_twist_reduction> ( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
    case HYPHA_SPECTRAL_SEQUENCE: compute_pairing<phat::Representation, hypha::hypha_spectral_sequence_reduction>  ( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
    case CHUNK_SEQUENTIAL: int num_threads = omp_get_max_threads(); \
                           omp_set_num_threads( num_threads ); \
                           compute_pairing< phat::Representation, phat::chunk_reduction >( input_filename, output_filename, use_binary, verbose, dualize, ga, dual_input, compression); break; \
                           omp_set_num_threads( num_threads ); \
                           break; \
    }



int main( int argc, char** argv )
{

    bool use_binary = true; // interpret input as binary or ascii file
    Representation_type representation = BIT_TREE_PIVOT_COLUMN; // representation class
    Algorithm_type algorithm = TWIST; // reduction algorithm
    std::string input_filename; // name of file that contains the boundary matrix
    std::string output_filename; // name of file that will contain the persistence pairs
    bool verbose = true; // print timings / info (always on in hypha)
    bool dualize = false; // toggle for dualization approach
    bool ga = true; // toggle for gpu assist approach or phat alone
    bool compression= true; //toggle compression (no compression on dualized matrices may give better performance)
    bool dual_input= false;

    parse_command_line( argc, argv, use_binary, representation, algorithm, input_filename, output_filename, verbose, dualize, ga, dual_input, compression);//,go, gob);

    switch( representation ) {
        case VECTOR_VECTOR: COMPUTE_PAIRING(vector_vector) break;
        case FULL_PIVOT_COLUMN: COMPUTE_PAIRING(full_pivot_column) break;
        case BIT_TREE_PIVOT_COLUMN: COMPUTE_PAIRING(bit_tree_pivot_column) break;
        case SPARSE_PIVOT_COLUMN: COMPUTE_PAIRING(sparse_pivot_column) break;
        case HEAP_PIVOT_COLUMN: COMPUTE_PAIRING(heap_pivot_column) break;
    }

}

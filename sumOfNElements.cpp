#include <iostream>
#include <CL/sycl.hpp>
#include <array>
#include <ctime>
#include <chrono>

using namespace sycl;
using namespace std;
using namespace std::chrono;

/**
 * 
 * Objective - design and implementation of sum of N elements using GPU parallel programming using oneAPI DPC++ Platform
 * Similar to parallel sum
 * Each operation is dependent in this case because the addition of the next sum id dependent on the complementing of the previous sum
 * to be precise this calculation is done with a binary tree approach => this means partial sums of two consecutive numbers
 *
 * Tree Reduction Approach
 * Applying reduction to adjacent numbers which produces multiple levels => tree like
 * 
 * Code inspired by: 
 * https://www.intel.com/content/www/us/en/developer/articles/technical/reduction-operations-in-data-parallel-cpp
 * https://www.codingame.com/playgrounds/48226/introduction-to-sycl/using-local-memory-barriers-and-writing-the-result
 */

int main()
{

    // Create a Queue
    queue Q;

    // prompt users for the sum of N elements to calculate

    const int N = 8;

    // create an array for the number of elements that needs calculation
    std::array<int, N> arr;
    
    // populate the array with values from 0 to N
    for (int i = 0; i < arr.size(); i++)
    {
        // populate the array with the number of N numbers to calculate
        if (N % 2 == 0){
        arr[i] = i+1;
        }

        else if (N % 2 == 1) {
            arr[i] = i;
        }
    }

    // display the data being calculated to the console
    std::cout << "The array: [";
    for (auto& i : arr)
    {
        if (N % 2 == 0) { std::cout << i << " "; }
        else if (N % 2 == 1) { std::cout << i +1 << " "; }
    }
    std::cout << "]" << std::endl;

    // create a buffer for the array items and it also takes a range in memory for the array size specified
    buffer<int, 1> input_buf(arr.data(), range<1>(arr.size()));

    // create a size for working groups
    size_t work_group_size = 512;
    //This is the maximum number of elements that a work group reduces. Each work item reduces two elements, so it's twice the work group. 
    size_t double_work_group_size = work_group_size * 2;
   

    auto iterationCount = arr.size();


        //Finding the number of work groups for this iteration of reduction: the remaining length left divided by the max number of elements
        auto work_group_count = (iterationCount + double_work_group_size - 1) / double_work_group_size;
        auto start = high_resolution_clock::now();
        Q.submit([&](auto& h)
            {

                // make an accessor to store the value of the input buffer and set the type of operation the buffer can perform
                accessor<int, 1, access::mode::read_write, access::target::local> local_mem(range<1>(work_group_size), h);
                // create a global mem to store the result which have the access read_write so its content can be read from also mutated
                auto global_mem = input_buf.get_access<access::mode::read_write>(h);
                
                //nd range represents number of work-items per dimension, number of work-items in a work-group
                h.parallel_for(nd_range<1>(work_group_size * work_group_count, work_group_size), [=](nd_item<1> item) {
                    // now extract the local, global and group id for each iteration from the item
                    // this method get_local.global_linear_id returns the respective id mapped to the dimension => global / local
                    int local_id = item.get_local_linear_id();
                    int global_id = item.get_global_linear_id();
                    

                    //loading the data from global to local for better performance
                    local_mem[local_id] = 0;
                    //
                    if ((2 * global_id) < iterationCount) {
                        //each work item gets two elements loaded into each other
                        local_mem[local_id] = global_mem[2 * global_id] + global_mem[2 * global_id + 1];
                    }
                    // using work group barriers here to sync work items by forcing the items to reach here and wait source https://docs.oneapi.io/versions/latest/dpcpp/iface/nd_item.html
                    item.barrier(access::fence_space::local_space);
                    //reduce two elements into one work item, repeat for entire work group
                    for (int i = 1; i < work_group_size; i *= 2) {
                        auto index = 2 * i * local_id;
                        if (index < work_group_size) {
                            local_mem[index] = local_mem[index] + local_mem[index + i];
                        }
                        //sync work-items
                        item.barrier(access::fence_space::local_space);
                    }

                    //write the final result back to global memory
                    if (local_id == 0) {
                        global_mem[item.get_group_linear_id()] = local_mem[0];
                    }

                    }); });
        auto end = high_resolution_clock::now();
        // this is to make the queue wait for the completion of the current execution then proceed to the next iteration as the next execution is dependent on the result of the current iteration.
        Q.wait_and_throw();

        //the iteration count needs to be set again since a number is produced and so 1 iteration step can be reduced
        iterationCount = work_group_count;
    

    // print the result byr reading the resulting value from the buffer
    auto result = input_buf.get_access<access::mode::read>();

    std::cout << "The Sum: " << result[0] << std::endl;
    
    std::cout << "Time it took: " << duration_cast<microseconds>(end - start).count() << " microseconds" << std::endl;
}

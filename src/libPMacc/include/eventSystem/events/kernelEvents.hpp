/**
 * Copyright 2013-2016 Felix Schmitt, Rene Widera, Benjamin Worpitz
 *
 * This file is part of libPMacc.
 *
 * libPMacc is free software: you can redistribute it and/or modify
 * it under the terms of either the GNU General Public License or
 * the GNU Lesser General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * libPMacc is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License and the GNU Lesser General Public License
 * for more details.
 *
 * You should have received a copy of the GNU General Public License
 * and the GNU Lesser General Public License along with libPMacc.
 * If not, see <http://www.gnu.org/licenses/>.
 */

#pragma once

#include "dimensions/DataSpace.hpp"
#include "eventSystem/EventSystem.hpp"
#include "types.h"

/* No namespace in this file since we only declare macro defines */

/*if this flag is defined all kernel calls would be checked and synchronize
 * this flag must set by the compiler or inside of the Makefile
 */
#if (PMACC_SYNC_KERNEL  == 1)
    #define CUDA_CHECK_KERNEL_MSG(COMMAND, MSG)                                \
        try                                                                    \
        {                                                                      \
            COMMAND;                                                           \
        }                                                                      \
        catch(...)                                                             \
        {                                                                      \
            std::cerr<<MSG<<std::endl;                                         \
        }
#else
    /*no synchronize and check of kernel calls*/
    #define CUDA_CHECK_KERNEL_MSG(COMMAND, MSG)
#endif

/** Call activate kernel from taskKernel.
 *  If PMACC_SYNC_KERNEL is 1 cudaDeviceSynchronize() is called before
 *  and after activation.
 */
#define PMACC_ACTIVATE_KERNEL                                                           \
        CUDA_CHECK_KERNEL_MSG(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getAccDevice()),"Crash after kernel launch");     \
        taskKernel->activateChecks();                                                   \
        CUDA_CHECK_KERNEL_MSG(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getAccDevice()),"Crash after kernel activation");

/**
 * Appends kernel arguments to generated code and activates kernel task.
 *
 * @param ... parameters to pass to kernel
 */
#define PMACC_CUDAPARAMS(...)                                                  \
        auto const workDiv =                                                   \
            ::alpaka::workdiv::WorkDivMembers<                                 \
                KernelDim,                                                     \
                ::PMacc::alpaka::IdxSize                                       \
            >(                                                                 \
                gridExtent,                                                    \
                blockExtent,                                                   \
                ::PMacc::math::Vector<                                         \
                    ::PMacc::alpaka::IdxSize,                                  \
                    KernelDim::value                                           \
                >::create(1u)                                                  \
            );                                                                 \
        auto const exec(                                                       \
            ::alpaka::exec::create<                                            \
                ::PMacc::alpaka::Acc<                                    \
                    KernelDim                                                  \
                >                                                              \
            >(                                                                 \
                workDiv,                                                       \
                theOneAndOnlyKernel,                                           \
                __VA_ARGS__                                                    \
            )                                                                  \
        );                                                                     \
        ::alpaka::stream::enqueue(taskKernel->getEventStream()->getCudaStream(), exec); \
        PMACC_ACTIVATE_KERNEL                                                  \
    }   /*this is used if call is EventTask.waitforfinished();*/

/**
 * Configures block and grid sizes and shared memory for the kernel.
 *
 * @param grid sizes of grid on gpu
 * @param block sizes of block on gpu
 * @param ... amount of shared memory for the kernel (optional)
 */
#define PMACC_CUDAKERNELCONFIG(grid,block)                                     \
    const auto&& gridExtent(grid);                                             \
    const auto&& blockExtent(block);                                           \
    PMACC_CUDAPARAMS

/**
 * Calls a CUDA kernel and creates an EventTask which represents the kernel.
 *
 * @param ... name of the CUDA kernel (can also used with templates etc. myKernel<1>)
 */
#define __cudaKernel(...) {                                                    \
    using KernelType = __VA_ARGS__;                                            \
    const KernelType theOneAndOnlyKernel;                                      \
    using KernelDim = ::PMacc::alpaka::Dim<KernelType::dim>;                   \
    CUDA_CHECK_KERNEL_MSG(::alpaka::wait::wait(::PMacc::Environment<>::get().DeviceManager().getAccDevice()),"Crash before kernel call"); \
    PMacc::TaskKernel *taskKernel = ::PMacc::Environment<>::get().Factory().createTaskKernel(#__VA_ARGS__);     \
    PMACC_CUDAKERNELCONFIG

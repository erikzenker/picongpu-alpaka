/**
 * Copyright 2013-2016 Rene Widera, Benjamin Worpitz, Alexander Grund
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

//#include <cuSTL/container/HostBuffer.hpp>
#include "memory/buffers/Buffer.hpp"
#include "dimensions/DataSpace.hpp"

namespace PMacc
{

    class EventTask;

    template <class TYPE, unsigned DIM>
    class Buffer;

    template <class TYPE, unsigned DIM>
    class DeviceBuffer;

    /**
     * Interface for a DIM-dimensional Buffer of type TYPE on the host
     *
     * @tparam TYPE datatype for buffer data
     * @tparam DIM dimension of the buffer
     */
    template <class TYPE, unsigned DIM>
    class HostBuffer : public Buffer<TYPE, DIM>
    {
    public:
        using DataView = ::alpaka::mem::view::ViewPlainPtr<
            alpaka::HostDev,
            TYPE,
            alpaka::Dim<DIM>,
            alpaka::MemSize
         >;

        using Data1DBuf = ::alpaka::mem::buf::Buf<
            alpaka::HostDev,
            TYPE,
            alpaka::Dim<DIM1>,
            alpaka::MemSize
        >;

        /**
         * Returns a view to the internal alpaka buffer.
         *
         * @return view to internal alpaka buffer
         *
         * @{
         */
        virtual
        DataView const &
        getMemBufView() const = 0;

        virtual
        DataView &
        getMemBufView() = 0;
        ///@}

        virtual
        Data1DBuf const &
        getMemBufView1D() const = 0;

        virtual
        Data1DBuf &
        getMemBufView1D() = 0;

        /**
         * Copies the data from the given DeviceBuffer to this HostBuffer.
         *
         * @param other DeviceBuffer to copy data from
         */
        virtual void copyFrom(DeviceBuffer<TYPE, DIM>& other) = 0;

        /**
         * Returns the current size pointer.
         *
         * @return pointer to current size
         */
        virtual size_t* getCurrentSizePointer()
        {
            __startOperation(ITask::TASK_HOST);
            return this->current_size;
        }

        /**
         * Destructor.
         */
        virtual ~HostBuffer()
        {
        };
/*
        HINLINE
        container::HostBuffer<TYPE, DIM>
        cartBuffer()
        {
            container::HostBuffer<TYPE, DIM> result;
            auto & memBufView = this->getMemBufView();
            result.dataPointer = ::alpaka::mem::view::getPtrNative(memBufView);
            result._size = math::Size_t<DIM>(this->getDataSpace());
            if(DIM >= 2)
                result.pitch[0] = result._size.x() * sizeof(TYPE);
            if(DIM >= 3)
                result.pitch[1] = result.pitch[0] * result._size.y();
            result.refCount = new int;
            *result.refCount = 2;
            return result;
        }
*/
    protected:

        /** Constructor.
         *
         * @param size extent for each dimension (in elements)
         *             if the buffer is a view to an existing buffer the size
         *             can be less than `physicalMemorySize`
         * @param physicalMemorySize size of the physical memory (in elements)
         */
        HostBuffer(DataSpace<DIM> size, DataSpace<DIM> physicalMemorySize) :
        Buffer<TYPE, DIM>(size, physicalMemorySize)
        {

        }
    };

} //namespace PMacc

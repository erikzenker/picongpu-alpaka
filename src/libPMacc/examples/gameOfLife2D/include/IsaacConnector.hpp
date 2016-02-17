/**
 * Copyright 2016 Erik Zenker
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

#include "memory/buffers/GridBuffer.hpp"
#include "memory/boxes/DataBox.hpp"
#include "memory/boxes/PitchedBox.hpp"

#include <isaac.hpp>
#include <alpaka/alpaka.hpp>

#include <string>
#include <vector>

#define DIM 2
#define TYPE uint8_t

ISAAC_NO_HOST_DEVICE_WARNING
template<
    typename T_Dim>
class GolSubDomain
{
public:
    using Buffer_t = PMacc::DataBox<PMacc::PitchedBox<TYPE, DIM> >;
    
    static const std::string name;
    static const size_t feature_dim = 1;
    static const bool has_guard = true;
    static const bool persistent = false;

    Buffer_t buffer;
    alpaka::Vec<T_Dim, size_t> extent;
    
    ISAAC_NO_HOST_DEVICE_WARNING
    GolSubDomain (Buffer_t buffer,
                  alpaka::Vec<T_Dim, size_t> extent) :
        buffer(buffer),
        extent(extent)
    { }

    ISAAC_HOST_INLINE void update(bool enabled, void* pointer) {
        buffer.fixedPointer = reinterpret_cast<Buffer_t*>(pointer)->fixedPointer;
    }
		
    ISAAC_NO_HOST_DEVICE_WARNING		
    ISAAC_HOST_DEVICE_INLINE isaac::isaac_float_dim<1> operator[] (const isaac::isaac_int3& nIndex) const
    {
        isaac::isaac_float value = static_cast<isaac::isaac_float>(buffer[nIndex.x][nIndex.y]);
        isaac::isaac_float_dim<1> result;
        result.value.x = value;
        return result;
    }
};

template < typename TDim>
const std::string GolSubDomain< TDim >::name = "GolSubDomain";

template<
    typename T_Host,
    typename T_Acc,
    typename T_Stream,
    typename T_Dim_Alpaka,
    typename T_Dim_Sim
    >
struct IsaacConnector {

    using Buffer_t = PMacc::DataBox<PMacc::PitchedBox<TYPE, DIM> >;    
    using DevHost = alpaka::dev::Dev<T_Host>;
    using DevAcc  = alpaka::dev::Dev<T_Acc>;
    
    using GolSubDomain_t = GolSubDomain<T_Dim_Sim>;
    using SourceList = boost::fusion::list<GolSubDomain_t>;
    using IsaacVisualization_t = isaac::IsaacVisualization <
        T_Host, //Alpaka specific Host Dev Type
        T_Acc, //Alpaka specific Accelerator Dev Type
        T_Stream, //Alpaka specific Stream Type
        T_Dim_Alpaka, //Alpaka specific Acceleration Dimension Type
        T_Dim_Sim, //Dimension of the Simulation. In this case: 3D
        SourceList, //The boost::fusion list of Source Types
        alpaka::Vec<T_Dim_Sim, size_t>, //Type of the 3D vectors used later
        1024, //Size of the transfer functions
        std::vector<float> //user defined type of scaling
	>;

    DevHost devHost;
    DevAcc devAcc;
    
    std::string name;
    GolSubDomain_t golSubDomain;    
    SourceList sources;
    T_Stream stream;
    //alpaka::stream::StreamCpuSync stream;
    IsaacVisualization_t visualization;
    
    IsaacConnector(Buffer_t buffer,
                   //T_Host devHost,
                   //T_Acc devAcc,
                   std::string server,
                   int port,
                   isaac::isaac_size2 framebuffer_size,
                   alpaka::Vec<T_Dim_Sim, size_t> global_size,
                   alpaka::Vec<T_Dim_Sim, size_t> local_size,
                   alpaka::Vec<T_Dim_Sim, size_t> position,
                   std::vector<float> scaling) :
        devHost(alpaka::dev::DevMan<T_Host>::getDevByIdx(0)),
        devAcc(alpaka::dev::DevMan<T_Acc>::getDevByIdx(0)),        
        name("PMacc Game of Life"),
        golSubDomain(buffer, local_size),
        sources(golSubDomain),
        stream(devAcc),        
        visualization( devHost, //Alpaka specific host dev instance
                       devAcc, //Alpaka specific accelerator dev instance
                       stream, //Alpaka specific stream instance
                       name.c_str(), //Name of the visualization shown to the client
                       0, //Master rank, which will opens the connection to the server
                       server.c_str(), //Address of the server
                       port, //Inner port of the server
                       framebuffer_size, //Size of the rendered image
                       global_size, //Size of the whole volumen including all nodes
                       local_size, //Local size of the subvolume
                       position, //Position of the subvolume in the globale volume
                       sources, //instances of the sources to render
                       scaling)
                                                                {
        
    }

    void init(){
        visualization.init();
    }

    void draw(Buffer_t buffer){
        bool pause = false;
        json_decref( visualization.doVisualization(isaac::META_MASTER, &buffer,!pause));
    }

};

cmake_minimum_required(VERSION 2.4.6)
include($ENV{ROS_ROOT}/core/rosbuild/rosbuild.cmake)

# Set the build type.  Options are:
#  Coverage       : w/ debug symbols, w/o optimization, w/ code-coverage
#  Debug          : w/ debug symbols, w/o optimization
#  Release        : w/o debug symbols, w/ optimization
#  RelWithDebInfo : w/ debug symbols, w/ optimization
#  MinSizeRel     : w/o debug symbols, w/ optimization, stripped binaries
#set(ROS_BUILD_TYPE RelWithDebInfo)

rosbuild_init()

#set the default path for built executables to the "bin" directory
set(EXECUTABLE_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/bin)
#set the default path for built libraries to the "lib" directory
set(LIBRARY_OUTPUT_PATH ${PROJECT_SOURCE_DIR}/lib)

#uncomment if you have defined messages
#rosbuild_genmsg()
#uncomment if you have defined services
#rosbuild_gensrv()

#common commands for building c++ executables and libraries
#rosbuild_add_library(${PROJECT_NAME} src/example.cpp)
#target_link_libraries(${PROJECT_NAME} another_library)
#rosbuild_add_boost_directories()
#rosbuild_link_boost(${PROJECT_NAME} thread)
#rosbuild_add_executable(example examples/example.cpp)
#target_link_libraries(example ${PROJECT_NAME})
rosbuild_add_executable( ardrone_image_node src/ardrone_image_node.cpp)

# Add FPL library
MACRO( find_package_and_add ) #_packageToFind _packageToAdd _packageDef )
    set( _packageToFind ${ARGV0} )
    set( _packageToAdd ${ARGV1} )


    if( NOT _packageToFind OR NOT _packageToAdd ) 
        message( FATAL_ERROR "USAGE: find_package_and_add( packageToFind packageToAdd [DefinitionToAdd] )" )
    endif()

    find_package( ${_packageToFind} REQUIRED )

    set( get_inc "${_packageToFind}_INCLUDE_DIR" )
    set( get_lib "${_packageToFind}_LIBRARIES" )
    set( get_lib2 "${_packageToFind}_LIBRARY" )
    set( get_def "${_packageToFind}_DEFINITIONS" )
    
    set( add_inc "${_packageToAdd}_INCLUDE_DIR" )
    set( add_lib "${_packageToAdd}_LIBRARIES" )
    set( add_def "${_packageToAdd}_DEFINITIONS" )

    set( ${add_inc} ${${add_inc}} ${${get_inc}} )
    set( ${add_lib} ${${add_lib}} ${${get_lib}} )
    set( ${add_lib} ${${add_lib}} ${${get_lib2}} )
    set( ${add_def} ${${add_def}} ${${get_def}} )

    # Convention: include and libraries names are in uppercase
    string( TOUPPER ${_packageToFind} _packageToFind )

    set( get_inc "${_packageToFind}_INCLUDE_DIR" )
    set( get_lib "${_packageToFind}_LIBRARIES" )
    set( get_lib2 "${_packageToFind}_LIBRARY" )
    set( get_def "${_packageToFind}_DEFINITIONS" )
    
    set( add_inc "${_packageToAdd}_INCLUDE_DIR" )
    set( add_lib "${_packageToAdd}_LIBRARIES" )
    set( add_def "${_packageToAdd}_DEFINITIONS" )

    set( ${add_inc} ${${add_inc}} ${${get_inc}} )
    set( ${add_lib} ${${add_lib}} ${${get_lib}} )
    set( ${add_lib} ${${add_lib}} ${${get_lib2}} )
    set( ${add_def} ${${add_def}} ${${get_def}} )
    
    if( ${ARGV3} )
        set( _packageDef ${ARGV4} )
        #message( STATUS "_packageDef: " ${_packageDef} )
        set( ${add_def} ${${add_def}} ${_packageDef} )
    endif()
ENDMACRO( find_package_and_add )

SET( ARDRONE_IMAGE "" )

find_package_and_add( CEIGEN ARDRONE_IMAGE )
find_package_and_add( CGEOM ARDRONE_IMAGE )
find_package_and_add( CCAMERAMODEL ARDRONE_IMAGE )
find_package_and_add( CCAMERASENSOR ARDRONE_IMAGE )
find_package_and_add( CTRACK ARDRONE_IMAGE )

find_package_and_add( Eigen ARDRONE_IMAGE )
find_package_and_add( Fiducials ARDRONE_IMAGE )
find_package_and_add( COPENCV ARDRONE_IMAGE )
find_package_and_add( CVARS ARDRONE_IMAGE )

# Need C++0x for fiducials
set( CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS}  -std=c++0x " )

include_directories( ${ARDRONE_IMAGE_INCLUDE_DIR} )
target_link_libraries( ardrone_image_node ${ARDRONE_IMAGE_LIBRARIES} )
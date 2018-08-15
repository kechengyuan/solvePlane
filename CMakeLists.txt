CMAKE_MINIMUM_REQUIRED(VERSION 2.6)
PROJECT(solvePlane)

find_package(PCL REQUIRED)
include_directories(${PCL_INCLUDE_DIRS})
add_definitions(${PCL_DEFINITIONS})

find_package(OpenCV REQUIRED)
find_package(PCL REQUIRED)
include_directories( ${OpenCV_INCLUDE_DIRS} )
include_directories( ${PCL_INCLUDE_DIRS} )


add_executable(solvePlane ${PROJECT_SOURCE_DIR}/src/main.cpp )

target_link_libraries(solvePlane ${PCL_LIBRARIES} ${OpenCV_LIBS})

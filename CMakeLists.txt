cmake_minimum_required(VERSION 4.0.1)

project(SDL)

set(SRCS main.cpp upng.c)
set(SDL_ROOT "C:/my_opt/SDL2-devel-2.0.12-mingw/SDL2-2.0.12/x86_64-w64-mingw32")

# Find SDL3 import library (libSDL2.dll.a)
find_library(SDL_LIB
    NAMES SDL2
    PATHS "${SDL_ROOT}/lib"
)

add_executable(main ${SRCS})

if(SDL_LIB)
    message(STATUS "Found SDL_LIB: ${SDL_LIB}")
    target_include_directories(main
        PRIVATE "${SDL_ROOT}/include"
    )
    target_link_libraries(main ${SDL_LIB})
else()
    message(FATAL_ERROR "SDL_LIB is required")
endif()

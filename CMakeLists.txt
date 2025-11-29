cmake_minimum_required(VERSION 3.20)

set(TARGET_NAME myapp)
project(${TARGET_NAME})

find_package(depthai REQUIRED)
message(STATUS "Found depthai: ${depthai_DIR}")

find_package(OpenCV REQUIRED)
find_package(OpenGL REQUIRED)
find_package(GLUT REQUIRED)
find_library(GLU_LIBRARY NAMES GLU GLU32 libGLU)

add_executable("${TARGET_NAME}"
    src/main.cpp
)

target_link_libraries(${TARGET_NAME}
    PUBLIC
        depthai::core
        ${OpenCV_LIBS}
)

if(TARGET OpenGL::GL)
    target_link_libraries(${TARGET_NAME} PUBLIC OpenGL::GL)
elseif(DEFINED OPENGL_gl_LIBRARY)
    target_link_libraries(${TARGET_NAME} PUBLIC ${OPENGL_gl_LIBRARY})
elseif(DEFINED OPENGL_LIBRARIES)
    target_link_libraries(${TARGET_NAME} PUBLIC ${OPENGL_LIBRARIES})
endif()

if(DEFINED GLU_LIBRARY)
    target_link_libraries(${TARGET_NAME} PUBLIC ${GLU_LIBRARY})
endif()

if(TARGET GLUT::GLUT)
    target_link_libraries(${TARGET_NAME} PUBLIC GLUT::GLUT)
elseif(DEFINED GLUT_LIBRARY)
    target_link_libraries(${TARGET_NAME} PUBLIC ${GLUT_LIBRARY})
elseif(DEFINED GLUT_LIBRARIES)
    target_link_libraries(${TARGET_NAME} PUBLIC ${GLUT_LIBRARIES})
endif()
    if(DEFINED GLUT_INCLUDE_DIRS)
        target_include_directories(${TARGET_NAME} PUBLIC ${GLUT_INCLUDE_DIRS})
    endif()

if(NOT MSVC)
    target_compile_options(${TARGET_NAME} PRIVATE $<$<COMPILE_LANGUAGE:CXX>:-Werror=return-type>)
endif()

set_property(TARGET ${TARGET_NAME} PROPERTY CXX_STANDARD 17)



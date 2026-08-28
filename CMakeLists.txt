cmake_minimum_required(VERSION 3.15)

set(PLUGIN_NAME basic_locomotion_controller)
project(${PLUGIN_NAME})
find_package(Python3 REQUIRED COMPONENTS Interpreter)

execute_process(
    COMMAND
        ${Python3_EXECUTABLE} -c "import sysconfig; schemes = sysconfig.get_scheme_names(); scheme = 'deb_system' if 'deb_system' in schemes else sysconfig.get_default_scheme(); print(sysconfig.get_path('purelib', scheme=scheme))"
    OUTPUT_VARIABLE
        BASIC_LOCOMOTION_DLS_PYTHON_ROOT
    OUTPUT_STRIP_TRAILING_WHITESPACE
)

file(READ version.txt PROJECT_VERSION)
string(REPLACE "." ";" VERSION_ARRAY ${PROJECT_VERSION})
list(POP_FRONT VERSION_ARRAY
    PROJECT_VERSION_MAJOR
    PROJECT_VERSION_MINOR
    PROJECT_VERSION_PATCH
)
string(REPLACE "\n" "" PROJECT_VERSION_PATCH ${PROJECT_VERSION_PATCH})
string(JOIN "." PROJECT_VERSION
    ${PROJECT_VERSION_MAJOR}
    ${PROJECT_VERSION_MINOR}
    ${PROJECT_VERSION_PATCH}
)
unset(VERSION_ARRAY)

set(BASIC_LOCOMOTION_RUNTIME_ROOT /usr/share/${PLUGIN_NAME})
set(BASIC_LOCOMOTION_TASKS_ROOT
    ${BASIC_LOCOMOTION_RUNTIME_ROOT}/source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks
)
set(BASIC_LOCOMOTION_SHARED_PYTHON
    /usr/lib/dls2/python/bin/python3
)

configure_file(
    cmake/basic_locomotion_controller.in
    ${CMAKE_CURRENT_BINARY_DIR}/${PLUGIN_NAME}
    @ONLY
)

install(
    FILES       plugin/config/scheduler.yaml
    DESTINATION /usr/include/dls2/schedulers
    COMPONENT   ${PROJECT_NAME}
    RENAME      ${PLUGIN_NAME}.yaml
)

install(
    PROGRAMS    ${CMAKE_CURRENT_BINARY_DIR}/${PLUGIN_NAME}
    DESTINATION /usr/bin
    COMPONENT   ${PROJECT_NAME}
)

install(
    PROGRAMS    deploy/installation/install_runtime_deps.sh
    DESTINATION ${BASIC_LOCOMOTION_RUNTIME_ROOT}/deploy/installation
    COMPONENT   ${PROJECT_NAME}
)

install(
    FILES
        deploy/config.py
        deploy/console.py
        deploy/controller_core.py
        deploy/locomotion_policy_wrapper.py
        deploy/run_controller_dls2.py
    DESTINATION ${BASIC_LOCOMOTION_RUNTIME_ROOT}/deploy
    COMPONENT   ${PROJECT_NAME}
)

install(
    FILES       deploy/installation/runtime_requirements.txt
    DESTINATION ${BASIC_LOCOMOTION_RUNTIME_ROOT}/deploy/installation
    COMPONENT   ${PROJECT_NAME}
)

install(
    FILES
        source/basic_locomotion_isaaclab/basic_locomotion_isaaclab/tasks/supervised_learning_networks.py
    DESTINATION ${BASIC_LOCOMOTION_TASKS_ROOT}
    COMPONENT   ${PROJECT_NAME}
)

install(
    DIRECTORY   tested_policies/
    DESTINATION ${BASIC_LOCOMOTION_RUNTIME_ROOT}/tested_policies
    COMPONENT   ${PROJECT_NAME}
    FILES_MATCHING
    PATTERN "__pycache__" EXCLUDE
    PATTERN "*/git" EXCLUDE
    PATTERN "*/git/*" EXCLUDE
    PATTERN "*/exported/*.onnx"
    PATTERN "*/exported/*.pth"
    PATTERN "*/params/*.yaml"
)

set(CPACK_PACKAGE_VERSION_MAJOR             ${PROJECT_VERSION_MAJOR})
set(CPACK_PACKAGE_VERSION_MINOR             ${PROJECT_VERSION_MINOR})
set(CPACK_PACKAGE_VERSION_PATCH             ${PROJECT_VERSION_PATCH})
set(CPACK_DEBIAN_PACKAGE_VERSION            ${PROJECT_VERSION})
set(CPACK_PACKAGE_VENDOR                    "dlslab")
set(CPACK_PACKAGE_CONTACT                   "dlslab")
set(CPACK_DEBIAN_PACKAGE_MAINTAINER         "dlslab")
set(CPACK_VERBATIM_VARIABLES                ON)
set(CPACK_GENERATOR                         "DEB")
set(CPACK_DEB_COMPONENT_INSTALL             ON)
set(CPACK_DEBIAN_PACKAGE_SHLIBDEPS          ON)

string(TOUPPER ${PROJECT_NAME} COMPONENT_NAME)

set(CPACK_DEBIAN_${COMPONENT_NAME}_PACKAGE_NAME        "dls-${PROJECT_NAME}" CACHE INTERNAL "")
set(CPACK_DEBIAN_${COMPONENT_NAME}_FILE_NAME           "dls-${PROJECT_NAME}-${PROJECT_VERSION}.deb" CACHE INTERNAL "")
set(CPACK_DEBIAN_${COMPONENT_NAME}_PACKAGE_DEPENDS
    "dls2-runtime" CACHE INTERNAL "")

get_cmake_property(CPACK_COMPONENTS_ALL COMPONENTS)
list(REMOVE_ITEM CPACK_COMPONENTS_ALL "Unspecified")

include(CPack)

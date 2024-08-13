cmake_minimum_required(VERSION 3.25.0 FATAL_ERROR) # Need cmake 3.25 for finding volk in vulkan package
project(graveler_vk VERSION 0.1.0 LANGUAGES C)
add_executable(graveler_vk source/graveler_vk.h source/main.c)
install(TARGETS graveler_vk)

# Find the vulkan sdk and the glslangValidator
find_package(Vulkan QUIET REQUIRED volk glslangValidator)
if(NOT DEFINED Vulkan_volk_LIBRARY)
	message(FATAL_ERROR "You need volk installed to build, update vulkan sdk")
else()
	message(STATUS "Found volk \"${Vulkan_volk_LIBRARY}\"")
endif()
if(NOT DEFINED Vulkan_GLSLANG_VALIDATOR_EXECUTABLE)
	message(FATAL_ERROR "Could not find glslangValidator")
else()
	message(STATUS "Found glslangValidator \"${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE}\"")
endif()
target_link_libraries(graveler_vk PUBLIC ${Vulkan_volk_LIBRARY} Vulkan::Headers)

# find python for dumping the shader as source
find_package (Python QUIET REQUIRED COMPONENTS Interpreter)
message(STATUS "Found python \"${Python_EXECUTABLE}\"")

# Add the shaders
function(add_comp_shader input_glsl)
	if(NOT EXISTS ${input_glsl})
		message(FATAL_ERROR "Cannot find ${input_glsl}")
	endif()
	target_sources(graveler_vk PRIVATE ${input_glsl})

	get_filename_component(glsl_name ${input_glsl} NAME_WE)
	message(STATUS "Adding shader ${glsl_name}")

	set(output_spirv_name ${CMAKE_CURRENT_BINARY_DIR}/${glsl_name}.spv)
	set(output_binary_name ${CMAKE_CURRENT_BINARY_DIR}/${glsl_name}.spv.c)
	set(command_args "-V" "--target-env" "vulkan1.0" "-S" "comp" ${input_glsl} "-o" ${output_spirv_name})
	add_custom_command( 
		OUTPUT ${output_spirv_name}
		DEPENDS ${input_glsl}
		COMMAND ${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} ${command_args}
		COMMENT "${Vulkan_GLSLANG_VALIDATOR_EXECUTABLE} ${command_args}"
		VERBATIM)
	target_sources(graveler_vk PRIVATE ${output_spirv_name})

	add_custom_command(
		OUTPUT ${output_binary_name}
		DEPENDS ${output_spirv_name} ${${CMAKE_CURRENT_LIST_DIR}/dump_spirv.py}
		COMMENT "Dumping spirv to ${output_binary_name}"
		COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_LIST_DIR}/dump_spirv.py --input=\"${output_spirv_name}\" --output=\"${output_binary_name}\" --var_name=spirv_${glsl_name})
	target_sources(graveler_vk PRIVATE ${output_binary_name})
endfunction()

add_comp_shader(${CMAKE_CURRENT_LIST_DIR}/source/random_roll.glsl)

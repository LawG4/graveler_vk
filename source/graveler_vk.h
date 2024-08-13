#ifndef __GRAVELER_HEADER_H__
#define __GRAVELER_HEADER_H__

#ifndef _WIN32
#error Currently only looking at win32 support
#endif

#include <Volk/volk.h>
#include <stdio.h>
#include <stdlib.h>
#include <stdbool.h>

#define MALLOC_CHECK(VAR_NAME) if(VAR_NAME == NULL) {printf("FATAL: Memory allocation for " #VAR_NAME " failed"); exit(-1);}

#define VK_CHECK(VK_CALL) if(VK_CALL != VK_SUCCESS){printf("FATAL: Vulkan call failed " #VK_CALL ". this is fatal"); exit(-1);}

// Command line args which the user can use to configure the program running 
typedef struct CmdArgs {
	uint32_t run_multiplication;
	bool try_enable_validation;
	bool write_per_workgroup_results;
}CmdArgs;
CmdArgs parse_command_line_args(int argc, char* argv[]);

typedef struct InstanceNMessenger {
	VkInstance instance;
	VkDebugUtilsMessengerEXT messenger;
}InstanceNMessenger;

// Initialized Volk dynamically and creates a vulkan instance, OR it exits the program
InstanceNMessenger create_instance(bool try_enable_validation);

// Creates a debug callback, or returns a null handle
VkDebugUtilsMessengerEXT create_debug_messenger(VkInstance instance);

// Selects the physical device to use. Exits on no vulkan physical devices 
VkPhysicalDevice select_vk_physical_device(VkInstance instance);

// How do we plan to dispatch the compute shaders 
typedef struct ComputeDispatchDimentions {
	uint32_t invocations_per_workgroup_x;
	uint32_t workgroups_per_dispatch_x;
	uint32_t dispatches_x;
}ComputeDispatchDimentions;
ComputeDispatchDimentions select_dispatch_dimentions_from_limits(VkPhysicalDeviceLimits limits);

// A group of info which we need to keep for submitting to the compute queue
typedef struct DeviceNQueue
{
	VkDevice device;
	struct VolkDeviceTable pfn;
	uint32_t family_index;
	VkQueue compute_queue;
}DeviceNQueue;

// Creates a device, along with the selected queue to send work to. OR it exits the program
DeviceNQueue create_device(VkInstance instance, VkPhysicalDevice physical);

typedef struct ComputePipeNShader {
	VkShaderModule shader;
	VkPipelineLayout pipe_layout;
	VkDescriptorSetLayout desc_layout;
	VkDescriptorPool desc_pool;
	VkDescriptorSet desc_set;
	VkPipeline pipeline;
}ComputePipeNShader;
ComputePipeNShader create_dice_roll_shader(DeviceNQueue* dnq);

typedef struct ComputeResultBuffers {
	VkBuffer buffer;
	VkDeviceMemory memory;
	VkDeviceSize size;
}ComputeResultBuffers;
ComputeResultBuffers create_result_buffers(DeviceNQueue* dnq, VkPhysicalDevice physical, ComputeDispatchDimentions dispatch);

void associate_buffers_with_pipeline(DeviceNQueue* dnq, ComputePipeNShader* compute, ComputeResultBuffers* results);

// We need to get a command pool and command buffer to allocate from, we're only do one shot
typedef struct CommandPoolNBuffer {
	VkCommandPool pool;
	VkCommandBuffer buffer;
}CommandPoolNBuffer;

// Fetches the command buffer and pool, or it exits the program 
CommandPoolNBuffer create_command_buffer(DeviceNQueue* dnq);

// Sync objects which are required for running the program 
typedef struct SyncObjects {
	VkFence fence;
}SyncObjects;
SyncObjects create_sync_object(DeviceNQueue* dnq);


#endif // !__GRAVELER_HEADER_H__

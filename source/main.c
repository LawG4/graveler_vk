#include "graveler_vk.h"
#include <string.h>
#include <Windows.h>

#define num_dice_rolls 1000000000

int main(int argc, char* argv[]) {

	// Start application, get cmd arguments and seed random numbers on CPU
	CmdArgs args = parse_command_line_args(argc, argv);
	uint64_t start_time = GetTickCount64();
	srand(start_time & 0xffffffff);

	// Create an instance  and maybe a debug callback too
	InstanceNMessenger inst = create_instance(args.try_enable_validation);

	// Allow the user to select the physical device, or automatically select when only one exists
	VkPhysicalDevice physical_device = select_vk_physical_device(inst.instance);
	VkPhysicalDeviceProperties physical_props = {0};
	vkGetPhysicalDeviceProperties(physical_device, &physical_props);
	printf("Success: Physical device \"%s\" was selected\n", physical_props.deviceName);
	
	// Select how large we need to make the compute shader dispatches 
	ComputeDispatchDimentions compute_dims = select_dispatch_dimentions_from_limits(physical_props.limits);

	// Create a device to send work over to 
	DeviceNQueue dnq = create_device(inst.instance, physical_device);
	printf("Success: Logical device with compute work created\n");

	// Create a compute pipeline and the outlines required along with buffers it uses
	ComputePipeNShader compute = create_dice_roll_shader(&dnq);
	ComputeResultBuffers result_buffers = create_result_buffers(&dnq, physical_device, compute_dims);
	associate_buffers_with_pipeline(&dnq, &compute, &result_buffers);
	printf("Success: Compute Pipelines and buffers created\n");

	// Create a buffer to record our work into 
	CommandPoolNBuffer cmd = create_command_buffer(&dnq);
	SyncObjects sync = create_sync_object(&dnq);
	printf("Success: Command buffer and sync objects created\n");

	// This How many times are we doing billion runs, and what was the highest encountered so far
	uint32_t run_count = compute_dims.dispatches_x * args.run_multiplication;
	uint32_t highest_roll = 0;
	
	// File handle for writing the per workgroup results 
	FILE* fp = NULL;

	// Iterate through the number dispatches that we need to do the total number of runs 
	for (uint32_t d = 0; d < run_count; d++)
	{
		uint64_t curr_time = GetTickCount64();
		curr_time ^= ((uint64_t)rand()) << 32; // mix top 32 bits of time for more randomness
		printf("\tRunning GPU dispatch %d/%d : ", d+1, run_count);

		// Open a file handle only when the user has requested we record results 
		if (args.write_per_workgroup_results) {
			char str[250] = { 0 };
			sprintf(str, "batch_%d.csv", d);
			fp = fopen(str, "w");
		}

		// Record the command buffer work. We don't need to wait for it to be returned yet 
		VK_CHECK(dnq.pfn.vkResetCommandPool(dnq.device, cmd.pool, VK_COMMAND_POOL_RESET_RELEASE_RESOURCES_BIT));
		VkCommandBufferBeginInfo begin = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_BEGIN_INFO };
		VK_CHECK(dnq.pfn.vkBeginCommandBuffer(cmd.buffer, &begin));

		dnq.pfn.vkCmdBindPipeline(cmd.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipeline);
		dnq.pfn.vkCmdBindDescriptorSets(cmd.buffer, VK_PIPELINE_BIND_POINT_COMPUTE, compute.pipe_layout, 0, 1, &compute.desc_set, 0, NULL);

		// We use the uint64 current time to seed the random timer on the gpu, so each dispatch has a new seed value
		dnq.pfn.vkCmdPushConstants(cmd.buffer, compute.pipe_layout, VK_SHADER_STAGE_COMPUTE_BIT, 0, sizeof(uint64_t), &curr_time);

		dnq.pfn.vkCmdDispatch(cmd.buffer, compute_dims.workgroups_per_dispatch_x, 1, 1);

		// Commands recorded, end the command buffer
		VK_CHECK(dnq.pfn.vkEndCommandBuffer(cmd.buffer));

		// Submit the work 
		VkSubmitInfo submit = { .sType = VK_STRUCTURE_TYPE_SUBMIT_INFO, .commandBufferCount = 1, .pCommandBuffers = &cmd.buffer, };
		VK_CHECK(dnq.pfn.vkQueueSubmit(dnq.compute_queue, 1, &submit, sync.fence));

		// Wait for the queue to finish 
		VK_CHECK(dnq.pfn.vkWaitForFences(dnq.device, 1, &sync.fence, VK_TRUE, UINT64_MAX));
		VK_CHECK(dnq.pfn.vkResetFences(dnq.device, 1, &sync.fence));
		printf("Done!\n");

		// Get the buffer back and then find the highest number in that buffer
		uint32_t* result_buffer = NULL;
		uint32_t local_highest_roll = 0;
		printf("\tSearching for highest roll in this batch on CPU : ");
		VK_CHECK(dnq.pfn.vkMapMemory(dnq.device, result_buffers.memory, 0, result_buffers.size, 0, &result_buffer));
		for (size_t i = 0; i < compute_dims.workgroups_per_dispatch_x; i++)
		{
			uint32_t val = result_buffer[i];
			if(fp) fprintf(fp, "%d,\n", val);
			if (val > local_highest_roll) local_highest_roll = val;
		}
		dnq.pfn.vkUnmapMemory(dnq.device, result_buffers.memory);
		if (fp) { fclose(fp); fp = NULL; }; // Close handle

		// Report info back to user 
		if (local_highest_roll > highest_roll) highest_roll = local_highest_roll;
		printf("Highest roll in this batch was %d\n", local_highest_roll);
	}

	// End time
	uint64_t end_time = GetTickCount64();
	printf("Success: Performed all dice runs\n\n");

	printf("Performed 1 dice run per invocation\n");
	printf("Performed %d invocations per workgroup\n", compute_dims.invocations_per_workgroup_x);
	printf("Performed %d workgroups per dispatch\n", compute_dims.workgroups_per_dispatch_x);
	printf("Performed %d dispatches\n", compute_dims.dispatches_x);
	printf("Total dice runs = 1 x %d x %d x %d = %zu\n", compute_dims.invocations_per_workgroup_x, compute_dims.workgroups_per_dispatch_x, compute_dims.dispatches_x,
		(uint64_t)compute_dims.invocations_per_workgroup_x * (uint64_t)compute_dims.workgroups_per_dispatch_x * (uint64_t)compute_dims.dispatches_x);
	printf("Highest roll found in total was %d\n", highest_roll);
	printf("Took %zu ms to complete\n\n", end_time - start_time);

	// Shutdown vulkan!!! 
	dnq.pfn.vkDeviceWaitIdle(dnq.device);
	dnq.pfn.vkDestroyCommandPool(dnq.device, cmd.pool, NULL);
	dnq.pfn.vkDestroyFence(dnq.device, sync.fence, NULL);
	dnq.pfn.vkDestroyBuffer(dnq.device, result_buffers.buffer, NULL);
	dnq.pfn.vkFreeMemory(dnq.device, result_buffers.memory, NULL);
	dnq.pfn.vkDestroyPipeline(dnq.device, compute.pipeline, NULL);
	dnq.pfn.vkDestroyPipelineLayout(dnq.device, compute.pipe_layout, NULL);
	dnq.pfn.vkDestroyShaderModule(dnq.device, compute.shader, NULL);
	dnq.pfn.vkDestroyDescriptorSetLayout(dnq.device, compute.desc_layout, NULL);
	dnq.pfn.vkDestroyDescriptorPool(dnq.device, compute.desc_pool, NULL);
	dnq.pfn.vkDestroyDevice(dnq.device, NULL);
	if (inst.messenger != VK_NULL_HANDLE) vkDestroyDebugUtilsMessengerEXT(inst.instance, inst.messenger, NULL);
	vkDestroyInstance(inst.instance, NULL);
}


static const char* const s_help_str = "Graveler random number generator\n"
"\t--help/-h : print this help message\n"
"\t-r [val] : run multiplier, how many times do you want to repeat a billion runs\n"
"\t-v : try enable vulkan api validation\n"
"\t-w : write highest number of 1s rolled per workgroup\n\n";

CmdArgs parse_command_line_args(int argc, char* argv[]) {

	// Default values
	CmdArgs out = { .run_multiplication = 1, .try_enable_validation = false, .write_per_workgroup_results = false };

	// Iterate through all options 
	for (size_t i = 1; i < argc; i++)
	{
		if (argv[i] == NULL) continue;

		// Help requested ? 
		if (strcmp(argv[i], "-h") == 0 || strcmp(argv[i], "--help") == 0) {
			printf("%s\n", s_help_str);
			exit(0);
		}

		// Validation?
		if (strcmp(argv[i], "-v") == 0) {
			out.try_enable_validation = true;
			continue;
		}

		// Writing results? 
		if (strcmp(argv[i], "-w") == 0) {
			out.write_per_workgroup_results = true;
		}

		// Run multiplier?
		if (strcmp(argv[i], "-r") == 0) {
			if (i >= argc - 1) {
				printf("Failed parsing cmd args : nothing found after -r\n%s\n", s_help_str);
				exit(-1);
			}

			// Convert it
			out.run_multiplication = strtol(argv[i + 1], NULL, 10);
			if (out.run_multiplication == 0) {
				printf("Failed parsing cmd args : -r = 0 or not a number\n%s\n", s_help_str);
				exit(-1);
			}
			i++; // Additional i movement
		}
	}

	return out;

}

InstanceNMessenger create_instance(bool try_enable_validation) {

	// output value 
	InstanceNMessenger out = { .instance = VK_NULL_HANDLE, .messenger = VK_NULL_HANDLE };
	if (volkInitialize() != VK_SUCCESS) {
		printf("Failed to initialize Volk. Your machine might not be Vulkan compatible\n");
		exit(-1);
	}
	printf("Success: Volk initialized\n");

	// Default value for the instance create info 
	VkApplicationInfo app_info = { .sType = VK_STRUCTURE_TYPE_APPLICATION_INFO, .apiVersion = VK_MAKE_API_VERSION(0,1, 0, 0), .pApplicationName = "graveler_vk" };
	VkInstanceCreateInfo instance_info = { .sType = VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, .pApplicationInfo = &app_info, };
	
	// Has user asked for validation layers to be enabled
	bool validation_enabled = false;
	const char* const validation_layers_name = "VK_LAYER_KHRONOS_validation";
	const char* const validation_ext_name = VK_EXT_DEBUG_UTILS_EXTENSION_NAME;
	if (try_enable_validation) {
		bool found_layer = false, found_ext = false;
		uint32_t count = 0;
		
		// Check the layers
		VK_CHECK(vkEnumerateInstanceLayerProperties(&count, NULL));
		VkLayerProperties* layers = malloc(count * sizeof(VkLayerProperties));
		MALLOC_CHECK(layers);
		VK_CHECK(vkEnumerateInstanceLayerProperties(&count, layers));
		for (uint32_t i = 0; i < count; i++)
		{
			if (strcmp(layers[i].layerName, validation_layers_name) == 0) {
				found_layer = true;
				break;
			}
		}
		free(layers);

		// Check the extensions
		VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL, &count, NULL));
		VkExtensionProperties* ext = malloc(count * sizeof(VkExtensionProperties));
		MALLOC_CHECK(ext);
		VK_CHECK(vkEnumerateInstanceExtensionProperties(NULL, &count, ext));
		for (uint32_t i = 0; i < count; i++)
		{
			if (strcmp(ext[i].extensionName, validation_ext_name) == 0) {
				found_ext = true;
				break;
			}
		}
		free(ext);

		if (found_layer && found_ext) {
			validation_enabled = true;
			instance_info.ppEnabledLayerNames = &validation_layers_name;
			instance_info.enabledLayerCount = 1;
			instance_info.ppEnabledExtensionNames = &validation_ext_name;
			instance_info.enabledExtensionCount = 1;
		}
	}

	if (vkCreateInstance(&instance_info, NULL, &out.instance) != VK_SUCCESS) {
		printf("Failed to initialize vulkan instance");
		exit(-1);
	}
	printf("Success: Vulkan instance created\n");
	volkLoadInstanceOnly(out.instance);

	// Did we enable validation let's find out by making a messenger
	if (validation_enabled) out.messenger = create_debug_messenger(out.instance);
	return out;
}

VkPhysicalDevice select_vk_physical_device(VkInstance instance) {
	
	VkPhysicalDevice selected_device = VK_NULL_HANDLE;
	uint32_t count = 0;
	VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, NULL));
	switch (count)
	{
	case 0: 
		printf("You do not have any compatible vulkan physical devices\n");
		exit(-1);
		break;
	case 1: 
		printf("\tOne physical device found, selecting default one automatically\n");
		VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, &selected_device));
		break;
	default: {
		VkPhysicalDevice* physical_devices = malloc(count * sizeof(VkPhysicalDevice));
		MALLOC_CHECK(physical_devices);
		VK_CHECK(vkEnumeratePhysicalDevices(instance, &count, physical_devices));

		printf("\tFound %d physical devices, please select :\n", count);
		for (uint32_t i = 0; i < count; i++)
		{
			VkPhysicalDeviceProperties props;
			vkGetPhysicalDeviceProperties(physical_devices[i], &props);
			printf("\t\t%d: %s\n", i, props.deviceName);
		}

		int32_t selected_index = -1;
		while (selected_index < 0 && selected_index >= count) {
			char input_buffer[4] = { 0 };
			printf("\tSelect device index : ");
			fgets(input_buffer, 4, stdin);

			long index = strtol(input_buffer, NULL, 10);
			if (index >= 0 && index < count) {
				selected_index = index;
				break;
			}
			printf("\tInvalid user input which was \"%d\"\n", index);
		}
		printf("\tSelected device %d\n\n", selected_index);
		selected_device = physical_devices[selected_index];
		free(physical_devices);
		physical_devices = NULL;
	}
	}
	

	return selected_device;	
}

ComputeDispatchDimentions select_dispatch_dimentions_from_limits(VkPhysicalDeviceLimits limits) {

	// Deciding the dimensions of a compute dispatch is a very big factor for the performance of a shader run.
	// You need to be optimizing occupancy, cache coherency, and minimizing dispatches. I have personally found
	// that my device can fit everything in the X dimensions within just one dispatch. 
	//
	// That is the best layout for this problem, additional customization is possible, but harder to configure and 
	// outside the scope of this project 

	// Under my constraints we will only be dispatching in x dimension, my device actually has max invocations and size[x]
	// as equal, this is probably as the expect dispatches in flat lines or squares or cubes with a fixed capacity
	uint64_t invocations_per_workgroup = limits.maxComputeWorkGroupInvocations;
	if (invocations_per_workgroup > limits.maxComputeWorkGroupSize[0]) {
		printf("Warning: Workgroups could be more efficient in higher dimension dispatch\n");
		invocations_per_workgroup = limits.maxComputeWorkGroupSize[0];
	}

	// In my example we're doing a single dice roll per invocation as it fits in a single dispatch
	uint64_t required_workgroup_count = (num_dice_rolls + (invocations_per_workgroup - 1)) / invocations_per_workgroup;
	// In the case that your device doesn't fit in one dispatch, I'll have a backup which performs multiple dispatches
	// this is SOOOO much slower than just doing multiple rolls per invocation. You could make this customizable per 
	// device with specialization constants but leave as exercise to reader
	uint64_t workgroups_per_dispatch = required_workgroup_count;
	uint64_t required_dispatch_count = 1;
	if (required_workgroup_count > limits.maxComputeWorkGroupCount[0]) {
		printf("Warning: Using multiple dispatches, this is a limitation of how I've divided workload as one dispatch is enough on my device\n");
		workgroups_per_dispatch = limits.maxComputeWorkGroupCount[0];
		required_dispatch_count = (required_workgroup_count + (workgroups_per_dispatch - 1)) / workgroups_per_dispatch;
	}
	
	// Pack to return to the user 
	ComputeDispatchDimentions dispatch = { 
		.invocations_per_workgroup_x = invocations_per_workgroup,
		.workgroups_per_dispatch_x = workgroups_per_dispatch,
		.dispatches_x = required_dispatch_count 
	};
	return dispatch;
}

DeviceNQueue create_device(VkInstance instance, VkPhysicalDevice physical) {

	DeviceNQueue out = { 0 };
	 
	// Get the queue families and what they support
	uint32_t count = 0;
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, NULL);
	VkQueueFamilyProperties* props = malloc(count * sizeof(VkQueueFamilyProperties));
	MALLOC_CHECK(props);
	vkGetPhysicalDeviceQueueFamilyProperties(physical, &count, props);

	// Check we have required features
	VkPhysicalDeviceFeatures features = { 0 };
	vkGetPhysicalDeviceFeatures(physical, &features);
	if (features.shaderInt64 != VK_TRUE) {
		printf("Vulkan device doesn't support uint64 in shader");
		exit(-1);
	}

	// First one to support compute, yoink!
	bool found = false;
	for (size_t i = 0; i < count; i++)
	{
		if (props[i].queueFlags & VK_QUEUE_COMPUTE_BIT) {
			out.family_index = i;
			found = true;
			break;
		}
	}
	if (!found) {
		printf("Failed to find a valid compute queue\n");
		exit(-1);
	}

	// Get the queue create info from this 
	float queue_priority = 1.0f;
	VkDeviceQueueCreateInfo queue = { .sType = VK_STRUCTURE_TYPE_DEVICE_QUEUE_CREATE_INFO, .pQueuePriorities = &queue_priority, .queueCount = 1, .queueFamilyIndex = out.family_index };

	// Get the device from it!
	VkPhysicalDeviceFeatures enabled_features = { 0 };
	enabled_features.shaderInt64 = VK_TRUE;
	VkDeviceCreateInfo dev = { .sType = VK_STRUCTURE_TYPE_DEVICE_CREATE_INFO, .pQueueCreateInfos = &queue, .queueCreateInfoCount =1, .pEnabledFeatures = &enabled_features};

	VK_CHECK(vkCreateDevice(physical, &dev, NULL, &out.device));
	volkLoadDeviceTable(&out.pfn, out.device);

	// Get the compute queue from the device 
	out.pfn.vkGetDeviceQueue(out.device, out.family_index, 0, &out.compute_queue);
	return out;
}

extern const uint8_t spirv_random_roll_data[];
extern const uint32_t spirv_random_roll_size;
ComputePipeNShader create_dice_roll_shader(DeviceNQueue* dnq) {
	ComputePipeNShader out = { 0 };

	// Create the shader module
	// We store the data inside the binary as a series of bytes, Vulkan wants it in uint32 for some reason, but it's not a good idea
	// to store them as uint32 specifically due to endianness of the target compute might invert expected byte order 
	const uint32_t* shader_data = (uint32_t*)(&spirv_random_roll_data[0]);
	VkShaderModuleCreateInfo shader = { .sType = VK_STRUCTURE_TYPE_SHADER_MODULE_CREATE_INFO, .pCode = shader_data, .codeSize = spirv_random_roll_size };
	VK_CHECK(dnq->pfn.vkCreateShaderModule(dnq->device, &shader, NULL, &out.shader));

	// Layout has: --------------------------------------------------------
	// Push constant 0 - uint64_t
	// Buffer slot 0 - uint32_t roll results 
	VkPipelineLayoutCreateInfo layout = { .sType = VK_STRUCTURE_TYPE_PIPELINE_LAYOUT_CREATE_INFO, };

	// Push constants
	VkPushConstantRange push_constant = { .offset = 0, .size = sizeof(uint64_t), .stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	layout.pPushConstantRanges = &push_constant;
	layout.pushConstantRangeCount = 1;

	// Descriptor set bindings 
	VkDescriptorSetLayoutBinding binding = { .binding = 0, .descriptorCount = 1, .descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER,
		.stageFlags = VK_SHADER_STAGE_COMPUTE_BIT };
	VkDescriptorSetLayoutCreateInfo  descriptor_layout = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_LAYOUT_CREATE_INFO,
		.pBindings = &binding, .bindingCount = 1 };
	VK_CHECK(dnq->pfn.vkCreateDescriptorSetLayout(dnq->device, &descriptor_layout, NULL, &out.desc_layout));
	layout.pSetLayouts = &out.desc_layout;
	layout.setLayoutCount = 1;

	// Pipeline creation ---------------------------------------------------
	VK_CHECK(dnq->pfn.vkCreatePipelineLayout(dnq->device, &layout, NULL, &out.pipe_layout));

	VkPipelineShaderStageCreateInfo stage_info = { .sType = VK_STRUCTURE_TYPE_PIPELINE_SHADER_STAGE_CREATE_INFO,
		.module = out.shader, .pName = "main", .stage = VK_SHADER_STAGE_COMPUTE_BIT, };

	VkComputePipelineCreateInfo compute = { .sType = VK_STRUCTURE_TYPE_COMPUTE_PIPELINE_CREATE_INFO,
		.stage = stage_info, .layout = out.pipe_layout, };
	VK_CHECK(dnq->pfn.vkCreateComputePipelines(dnq->device, VK_NULL_HANDLE, 1, &compute, NULL, &out.pipeline));

	// Descriptor pool to match the descriptor layout
	VkDescriptorPoolSize pool_size = { .descriptorCount = 1, .type = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER };
	VkDescriptorPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_POOL_CREATE_INFO,
		.pPoolSizes = &pool_size, .poolSizeCount = 1, .maxSets = 1, };
	VK_CHECK(dnq->pfn.vkCreateDescriptorPool(dnq->device, &pool, NULL, &out.desc_pool));

	VkDescriptorSetAllocateInfo set = { .sType = VK_STRUCTURE_TYPE_DESCRIPTOR_SET_ALLOCATE_INFO,
		.descriptorPool = out.desc_pool, .descriptorSetCount = 1, .pSetLayouts = &out.desc_layout };
	VK_CHECK(dnq->pfn.vkAllocateDescriptorSets(dnq->device, &set, &out.desc_set));

	return out;
}

ComputeResultBuffers create_result_buffers(DeviceNQueue* dnq, VkPhysicalDevice physical, ComputeDispatchDimentions dispatch) {

	// We have one uint32 for each workgroup 
	ComputeResultBuffers out = { 0 };
	uint32_t size = sizeof(uint32_t) * dispatch.workgroups_per_dispatch_x;
	uint32_t required_memory_properties = VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT;

	VkBufferCreateInfo buffer = { .sType = VK_STRUCTURE_TYPE_BUFFER_CREATE_INFO, .size = size,
		.pQueueFamilyIndices = &dnq->family_index, .queueFamilyIndexCount = 1, .sharingMode = VK_SHARING_MODE_EXCLUSIVE,
		.usage = VK_BUFFER_USAGE_STORAGE_BUFFER_BIT };
	VK_CHECK(dnq->pfn.vkCreateBuffer(dnq->device, &buffer, NULL, &out.buffer));

	// Get the memory requirements of this buffer, and what types of memory the device supports 
	VkPhysicalDeviceMemoryProperties mem_props;
	VkMemoryRequirements req;
	vkGetPhysicalDeviceMemoryProperties(physical, &mem_props);
	dnq->pfn.vkGetBufferMemoryRequirements(dnq->device, out.buffer, &req);

	// Find which index into the supported memory groups 
	bool found = false;
	uint32_t memory_index = 0;
	for (uint32_t i = 0; i < mem_props.memoryTypeCount; i++)
	{
		// Is the memory type (i) suitable for memory which would match the requirements of the buffer
		if (req.memoryTypeBits & (1 << i)) {
			if ((mem_props.memoryTypes[i].propertyFlags & required_memory_properties) == required_memory_properties) {
				memory_index = i;
				found = true;
			}
		}
	}
	if (!found) {
		printf("Fatal, couldn't find suitable memory requirements\n");
		exit(-1);
	}

	// Allocate the device memory 
	VkMemoryAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_MEMORY_ALLOCATE_INFO, .allocationSize = req.size,
		.memoryTypeIndex = memory_index };
	VK_CHECK(dnq->pfn.vkAllocateMemory(dnq->device, &alloc, NULL, &out.memory));

	// bind the buffer and the memory together
	VK_CHECK(dnq->pfn.vkBindBufferMemory(dnq->device, out.buffer, out.memory, 0));
	out.size = alloc.allocationSize;
	return out;
}

void associate_buffers_with_pipeline(DeviceNQueue* dnq, ComputePipeNShader* compute, ComputeResultBuffers* results) {

	VkDescriptorBufferInfo info = { .buffer = results->buffer, .offset = 0, .range = VK_WHOLE_SIZE };
	VkWriteDescriptorSet write_set = { .sType = VK_STRUCTURE_TYPE_WRITE_DESCRIPTOR_SET,
		.dstSet = compute->desc_set, .dstBinding = 0, .dstArrayElement = 0,
		.descriptorType = VK_DESCRIPTOR_TYPE_STORAGE_BUFFER , .descriptorCount = 1,
		.pBufferInfo = &info };
	dnq->pfn.vkUpdateDescriptorSets(dnq->device, 1, &write_set, 0, NULL);

}

CommandPoolNBuffer create_command_buffer(DeviceNQueue* dnq) {

	CommandPoolNBuffer out = { 0 };
	VkCommandPoolCreateInfo pool = { .sType = VK_STRUCTURE_TYPE_COMMAND_POOL_CREATE_INFO, .queueFamilyIndex = dnq->family_index, };
	VK_CHECK(dnq->pfn.vkCreateCommandPool(dnq->device, &pool, NULL, &out.pool));

	VkCommandBufferAllocateInfo alloc = { .sType = VK_STRUCTURE_TYPE_COMMAND_BUFFER_ALLOCATE_INFO, .commandPool = out.pool, .commandBufferCount = 1, .level = VK_COMMAND_BUFFER_LEVEL_PRIMARY };
	VK_CHECK(dnq->pfn.vkAllocateCommandBuffers(dnq->device, &alloc, &out.buffer));

	return out;
}

SyncObjects create_sync_object(DeviceNQueue* dnq) {
	SyncObjects out = { 0 };
	VkFenceCreateInfo fence = { .sType = VK_STRUCTURE_TYPE_FENCE_CREATE_INFO, .flags = 0 };
	VK_CHECK(dnq->pfn.vkCreateFence(dnq->device, &fence, NULL, &out.fence));
	return out;
}


VKAPI_ATTR VkBool32 VKAPI_CALL debug_callback(
	VkDebugUtilsMessageSeverityFlagBitsEXT messageSeverity,
	VkDebugUtilsMessageTypeFlagsEXT messageType,
	const VkDebugUtilsMessengerCallbackDataEXT* pCallbackData,
	void* pUserData) {

	printf("Validation layer: \n%s\n\n",pCallbackData->pMessage );

	return VK_FALSE;
}

VkDebugUtilsMessengerEXT create_debug_messenger(VkInstance instance){

	VkDebugUtilsMessengerEXT out = VK_NULL_HANDLE;
	if (vkCreateDebugUtilsMessengerEXT == NULL) return VK_NULL_HANDLE;

	VkDebugUtilsMessengerCreateInfoEXT messenger_info = { .sType = VK_STRUCTURE_TYPE_DEBUG_UTILS_MESSENGER_CREATE_INFO_EXT, .pfnUserCallback = debug_callback, .messageType = VK_DEBUG_UTILS_MESSAGE_TYPE_VALIDATION_BIT_EXT, .messageSeverity = VK_DEBUG_UTILS_MESSAGE_SEVERITY_ERROR_BIT_EXT | VK_DEBUG_UTILS_MESSAGE_SEVERITY_WARNING_BIT_EXT };
	if (vkCreateDebugUtilsMessengerEXT(instance, &messenger_info, NULL, &out) == VK_SUCCESS) {
		return out;
	}
	else {
		return VK_NULL_HANDLE;
	}
}
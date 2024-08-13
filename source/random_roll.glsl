/**
 * Source code which runs on the gpu for generating a series of random numbers. There is one problem, 
 * the random numbers are pseudo-random which means they are entirely depend on their input. This uses 
 * an XOR shift for randomness, meaning that the next random number is ENTIRELY dependent on the 
 * previous one. 
 * 
 * I could implement a better system, such as Mersenne twister, but that would be out of scope.
 * 
 * To ensure we get good randomness, each thread needs a unique seed. We start with a single variable
 * from the CPU for all threads (This changes per dispatch) The global id of this thread is also used
 * as a secondary key, they are then mixed together to make a (probably)unique seed per thread
 * 
 * 
 * We're also only interested in the dice run which got the highest result inside this workgroup, we
 * could report each dice run but then the output buffer will be much bigger. So we have a variable 
 * shared between all of the threads in the workgroup which tracks the largest seen in the work group
 * 
 * We can then nominate a single thread to upload that maximum number :)
 */
#version 430
#extension GL_ARB_gpu_shader_int64 : require

// Push constant, data directly in the command buffer which seeds the random offset
layout( push_constant ) uniform constants {
	uint64_t pipe_seed;
}push_constants;

// Bound buffer to slot 0 which is a writeable ssbo
layout(std430, binding = 0) buffer RollResultSSBO {
	uint roll_results_out[];
};

// Shared memory to track the highest score in the workgroup 
shared uint wg_highest_dice_run;

// Function which mixes the bits from an input in the hope of producing a a well mixed number
// i.e we want close numbers to be far away from each other
uint64_t hash_bit_mix(uint64_t key);

// Slightly different aim from the bit mix, we want a sequence xn = f(xn-1) which produces uniformally
// distributed psudorandom values
uint64_t next_rand(uint64_t past);

void main() {
	// One invocation in the workgroup should set the shared memory variables and then all 
	// invocations need to sync their shared memory
	if(gl_LocalInvocationID.x == 0) {
		wg_highest_dice_run = 0;
	}
	memoryBarrierShared();

	// We take an initial seed for our random number to be the combination of the current time 
	// from the push constant. We add in our global invocation id to make sure each thread has 
	// a unique starting seed. Then we hash it to introduce entropy and spread the seed out more
	uint64_t seed = hash_bit_mix(push_constants.pipe_seed) ^ hash_bit_mix(uint64_t(gl_GlobalInvocationID.x));

	// Get the first random number in the sequence
	uint64_t rand = next_rand(seed);
	uint number_of_1s = 0;

	// Perform a singular dice run, which will end when we get 177 1s or we have 231 rolls
	for(uint i = 0; i < 231; ++i) {

		// The prng should evenly distribute across entire uint64_t range, so it should have
		// a roughly uniform distribute, if it falls in the bottom quarter of uint64_t we 
		// say that's the same as rolling a 1.
		rand = next_rand(rand); // next random number 
		if(rand <= 0x3FFFFFFFFFFFFFFFl) {
			number_of_1s+= 1;

			// Only need to check when incremented
			if(number_of_1s >= 177) {
				break; // exit loop if we hit 177
			}
		}
	}

	// That is the end of this dice run in this invocation. Now within this workgroup
	// who has the largest result?
	atomicMax(wg_highest_dice_run, number_of_1s);
	
	// Make sure to wait for the atomic max to resolve and then write to the buffer from 
	// a single elective thread 
	memoryBarrierShared();
	if(gl_LocalInvocationID.x == 0) {
		roll_results_out[gl_WorkGroupID.x] = wg_highest_dice_run;
	}
	return;
}

uint64_t hash_bit_mix(uint64_t key) {
	// This does Austin Appleby's MurmurHash3 algorithm
	key ^= (key >> 33);
  	key *= 0xff51afd7ed558ccdL;
  	key ^= (key >> 33);
  	key *= 0xc4ceb9fe1a85ec53L;
  	key ^= (key >> 33);

  	return key;
}

uint64_t next_rand(uint64_t past) {
	// Implementation of Marsaglia’s xorshift. Secondary source : 
	// https://towardsdatascience.com/how-to-generate-a-vector-of-random-numbers-on-a-gpu-a37230f887a6
	past ^= (past << 13);
    past ^= (past >> 17);    
    past ^= (past << 5);    
    return past;
}
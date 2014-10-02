#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

/* GPU configuration	*/
typedef struct gpu_config_t_st
{
	int gpu_enabled;
	int platform_id;
	int device_id;
	
	const char* _cl_kernel_string;
	size_t _cl_kernel_size;
	cl_context _cl_context;
	cl_program _cl_program;
	cl_kernel _cl_kernel_rk4;
	cl_command_queue _cl_queue;
		
}gpu_config_t;


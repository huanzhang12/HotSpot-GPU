#ifndef __GPU_H
#define __GPU_H

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
	
	const unsigned char* _cl_kernel_string;
	size_t _cl_kernel_size;
	cl_context _cl_context;
	cl_program _cl_program;
	cl_kernel _cl_kernel_rk4;
	cl_command_queue _cl_queue;
		
}gpu_config_t;

/* Forward declaration */
typedef struct grid_model_t_st grid_model_t;
typedef struct grid_model_vector_t_st grid_model_vector_t;

gpu_config_t default_gpu_config(void);
void gpu_config_from_strs(gpu_config_t *config, str_pair *table, int size);
int gpu_config_to_strs(gpu_config_t *config, str_pair *table, int max_entries);
double gpu_rk4(void *model, double *y, void *p, int n, double *h, double *yout);

void gpu_init(gpu_config_t *config);
void gpu_destroy(gpu_config_t *config);

double rk4_gpu(void *model, double *y, void *p, int n, double *h, double *yout, gpu_config_t *config);
void rk4_core_gpu(void *model, double *y, double *k1, void *p, int n, double h, double *yout);
void slope_fn_grid_gpu(grid_model_t *model, double *v, grid_model_vector_t *p, double *dv);
void slope_fn_pack_gpu(grid_model_t *model, double *v, grid_model_vector_t *p, double *dv);

#endif

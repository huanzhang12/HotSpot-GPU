#ifndef __GPU_H
#define __GPU_H

#ifdef __APPLE__
    #include "OpenCL/opencl.h"
#else
    #include "CL/cl.h"
#endif

#include "gpu_rk4.h"

enum kernel_fn_grid_args 	{ GRID_CONST_MODEL, GRID_CONST_LAYER, GRID_IO_V, GRID_OUT_DV, GRID_NL, GRID_NR, GRID_NC, GRID_LOCALMEM, GRID_IN_CUBOID, GRID_H, GRID_IN_K, GRID_IN_Y };
enum kernel_average_args 	{ AVG_IN_Y, AVG_IN_K1, AVG_IN_K2, AVG_IN_K3, AVG_IN_K4, AVG_H, AVG_OUT_YOUT, AVG_N, AVG_IN_YTEMP, AVG_LOCALMEM };
enum kernel_max_reduce_args	{ MAX_IO_Y, MAX_N, MAX_LOCALMEM };

/* GPU configuration	*/
typedef struct gpu_config_t_st
{
	int gpu_enabled;
	int platform_id;
	int device_id;
	size_t local_work_size[2], global_work_size[2];
	
	const unsigned char* _cl_kernel_string;
	size_t _cl_kernel_size;
	cl_context _cl_context;
	cl_program _cl_program;
	cl_kernel _cl_kernel_rk4;
	cl_kernel _cl_kernel_average;
	cl_kernel _cl_kernel_average_with_maxdiff;
	cl_kernel _cl_kernel_max_reduce;
	cl_command_queue _cl_queue;

	/* memories */
	cl_mem c_model;
	cl_mem c_layer;
	cl_mem h_v;
	cl_mem h_y;
	cl_mem h_result;
	cl_mem h_cuboid;
	int h_cuboid_reference;
	void* pinned_h_v;
	void* pinned_h_y;
	void* pinned_h_cuboid;
	void* pinned_h_result;
	cl_mem d_v;
	cl_mem d_y, d_ytemp;
	cl_mem d_dv;
	cl_mem d_k1, d_k2, d_k3, d_k4;
	cl_mem d_t1;
	cl_mem d_p_cuboid;
	cl_mem d_c_model;
	cl_mem d_c_layer;

	/* sizes */
	size_t element_size;
	size_t extra_size;
	size_t cuboid_size;
	size_t vector_size;

	gpu_grid_model_t model;
	gpu_layer_t* layer;
	size_t layer_size;
		
}gpu_config_t;

/* Forward declaration */
typedef struct grid_model_t_st grid_model_t;
typedef struct grid_model_vector_t_st grid_model_vector_t;

gpu_config_t default_gpu_config(void);
void gpu_config_from_strs(gpu_config_t *config, str_pair *table, int size);
int gpu_config_to_strs(gpu_config_t *config, str_pair *table, int max_entries);
double gpu_rk4(void *model, double *y, void *p, int n, double *h, double *yout);

void gpu_init(gpu_config_t *config, grid_model_t *model);
void gpu_destroy(gpu_config_t *config);
void gpu_create_buffers(gpu_config_t *config, grid_model_t *model);
void gpu_delete_buffers(gpu_config_t *config);

void* gpu_allocate_cuboid_static(size_t size);
void gpu_free_cuboid_static(void* cuboid);

double rk4_gpu(gpu_config_t *config, void *model, double *y, void *p, int n, double *h, double *yout);
void rk4_core_gpu_kernel(gpu_config_t *config, void *model, cl_mem *d_y, cl_mem *d_k1, void *p, int n, double h, cl_mem *d_yout, cl_mem *d_ytemp, int do_maxdiff);
void rk4_core_gpu(gpu_config_t *config, void *model, double *y, double *k1, void *p, int n, double h, double *yout);
void slope_fn_grid_gpu_kernel(gpu_config_t *config, grid_model_t *model, cl_mem *d_v, grid_model_vector_t *p, cl_mem *d_dv);
void slope_fn_grid_gpu(gpu_config_t *config, grid_model_t *model, double *v, grid_model_vector_t *p, double *dv);
void slope_fn_pack_gpu_kernel(gpu_config_t *config, grid_model_t *model, double *v, grid_model_vector_t *p, double *dv);
void slope_fn_pack_gpu(gpu_config_t *config, grid_model_t *model, double *v, grid_model_vector_t *p, double *dv);

#endif

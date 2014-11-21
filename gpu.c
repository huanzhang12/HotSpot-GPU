#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <fcntl.h>
#include <time.h>

#include "temperature_grid.h"
#include "temperature.h"
#include "gpu.h"

unsigned char rk4_cl[] = {
#include "rk4_kernel_str.c"
};

static gpu_config_t* current_gpu_config = NULL;

/* default GPU configuration parameters	*/
gpu_config_t default_gpu_config(void)
{
	gpu_config_t config;
	
	config.gpu_enabled = 0;
	config.platform_id = 0;
	config.device_id = 0;
	config.local_work_size[0] = 16;
	config.local_work_size[1] = 16;
	config.global_work_size[0] = 0; // to be determined by the grid size
	config.global_work_size[1] = 0;
	
	config.h_cuboid_reference = -1;
	config.pinned_h_cuboid = NULL;

	/* rk4_cl is a string generated by rk4.cl */
	config._cl_kernel_string = rk4_cl;
	config._cl_kernel_size = sizeof(rk4_cl);
	config._cl_kernel_rk4 = NULL;
	config._cl_context = NULL;
	config._cl_program = NULL;
	config._cl_queue = NULL;

	config.last_h_y = NULL;
	config.last_io_buf = 0; // input y, output ytemp
	config.unified_memory_optimization = 0; // no zero-copy optimization

	return config;
}

/* 
 * parse a table of name-value string pairs and add the configuration
 * parameters to 'config'
 */
void gpu_config_from_strs(gpu_config_t *config, str_pair *table, int size)
{
	int idx;
	if ((idx = get_str_index(table, size, "gpu_enable")) >= 0) {
		if(sscanf(table[idx].value, "%d", &config->gpu_enabled) != 1)
			fatal("invalid format for configuration parameter gpu_enable\n");
	}
	if ((idx = get_str_index(table, size, "gpu_platform")) >= 0) {
		if(sscanf(table[idx].value, "%d", &config->platform_id) != 1)
			fatal("invalid format for configuration parameter gpu_platform\n");
	}
	if ((idx = get_str_index(table, size, "gpu_device")) >= 0) {
		if(sscanf(table[idx].value, "%d", &config->device_id) != 1)
			fatal("invalid format for configuration parameter gpu_device\n");
	}
}

/* 
 * convert config into a table of name-value pairs. returns the no.
 * of parameters converted
 */
int gpu_config_to_strs(gpu_config_t *config, str_pair *table, int max_entries)
{
	if (max_entries < 3)
		fatal("not enough entries in table for gpu_config_to_strs()\n");

	sprintf(table[0].name, "gpu_enable");
	sprintf(table[1].name, "gpu_platform");
	sprintf(table[2].name, "gpu_device");

	sprintf(table[0].value, "%d", config->gpu_enabled);
	sprintf(table[1].value, "%d", config->platform_id);
	sprintf(table[2].value, "%d", config->device_id);

	return 3;
}

void gpu_copy_constants(gpu_config_t *config, grid_model_t *model)
{
	int i, max_layers;
	config->model.n_layers = model->n_layers;
	config->model.rows = model->rows;
	config->model.cols = model->cols;
	config->model.width = model->width;
	config->model.height = model->height;
	config->model.total_n_blocks = model->total_n_blocks;
	config->model.r_ready = model->r_ready;
	config->model.c_ready = model->c_ready;
	config->model.has_lcf = model->has_lcf;
	config->model.base_n_units = model->base_n_units;

	// memcpy(&config->model.pack, &model->pack, sizeof(gpu_package_RC_t));
	/* this will convert double to float, if necessary */
	real* dst = (real*)&config->model.pack;
	double* src = (double*)&model->pack;
	for (i = 0; i < sizeof(config->model.pack) / sizeof(real); ++i) {
		*dst++ = *src++;
	}

	// memcpy(&config->model.config, &model->config, sizeof(gpu_thermal_config_t));
	config->model.config.ambient = model->config.ambient;
	config->model.config.model_secondary = model->config.model_secondary;
	config->model.config.r_convec_sec = model->config.r_convec_sec;
	config->model.config.s_pcb = model->config.s_pcb;
	max_layers = model->n_layers;
	config->layer = (gpu_layer_t*) calloc(max_layers, sizeof(gpu_layer_t));
	config->layer_size = max_layers * sizeof(gpu_layer_t);
	for(i = 0; i < max_layers; ++i)
	{
		config->layer[i].no = model->layers[i].no;
		config->layer[i].has_lateral = model->layers[i].has_lateral;
		config->layer[i].has_power = model->layers[i].has_power;
		config->layer[i].k = model->layers[i].k;
		config->layer[i].k1 = model->layers[i].k1;
		config->layer[i].thickness = model->layers[i].thickness;
		config->layer[i].sp = model->layers[i].sp;
		config->layer[i].sp1 = model->layers[i].sp1;
		config->layer[i].rx = model->layers[i].rx;
		config->layer[i].ry = model->layers[i].ry;
		config->layer[i].rz = model->layers[i].rz;
		config->layer[i].rx1 = model->layers[i].rx1;
		config->layer[i].ry1 = model->layers[i].ry1;
		config->layer[i].rz1 = model->layers[i].rz1;
		config->layer[i].c = model->layers[i].c;
		config->layer[i].c1 = model->layers[i].c1;
	}
}

void gpu_check_error(int err, char* msg)
{
	if (err < 0)
	{
		printf("%s\n. (Error no.: %d)\n",msg, err);
		fatal("OpenCL Runtime Error.");
	}
}

void gpu_create_buffers(gpu_config_t *config, grid_model_t *model)
{
	int err;
	int i, j;
	int nl = model->n_layers;
	int nr = model->rows;
	int nc = model->cols;

	size_t element_size = sizeof(real);
	/* we initialize memory to NaN. If the kernel has unexpected accesses, NaN will propagate. */
	real pattern = NAN;
	config->element_size = element_size;
	config->extra_size = ((model->config.model_secondary)? EXTRA + EXTRA_SEC : EXTRA) * element_size;
	config->cuboid_size = nr * nc * nl * element_size;
	config->vector_size = config->extra_size + config->cuboid_size;


	/* prepare device memory */
	config->d_dv = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_dv, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_dv failed.");
	config->d_p_cuboid = clCreateBuffer(config->_cl_context, CL_MEM_READ_ONLY, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_p_cuboid, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_p_cuboid failed.");
	config->d_y = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_y, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_y failed.");
	config->d_ytemp = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_ytemp, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_ytemp failed.");
	config->d_k1 = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_k1, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_k1 failed.");
	config->d_k2 = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_k2, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_k2 failed.");
	config->d_k3 = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_k3, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_k3 failed.");
	config->d_k4 = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_k4, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_k4 failed.");
	config->d_t1 = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE, config->vector_size, NULL, &err);
	err |= clEnqueueFillBuffer(config->_cl_queue, config->d_t1, &pattern, element_size, 0, config->vector_size, 0, NULL, NULL);
	gpu_check_error(err, "clCreateBuffer() for d_t1 failed.");

	/* prepare constant memory */
	config->d_c_model = clCreateBuffer(config->_cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, sizeof(config->model), &config->model, &err);
	gpu_check_error(err, "clCreateBuffer() for d_c_model failed.");
	config->d_c_layer = clCreateBuffer(config->_cl_context, CL_MEM_READ_ONLY | CL_MEM_COPY_HOST_PTR, config->layer_size, config->layer, &err);
	gpu_check_error(err, "clCreateBuffer() for d_c_layer failed.");

	/* prepare host memory, this is just for reading the max value from maxdiff kernel */
	config->h_result = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, element_size, NULL, &err);
	gpu_check_error(err, "clCreateBuffer() for h_result failed.");
	config->pinned_h_result = clEnqueueMapBuffer(config->_cl_queue, config->h_result, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, element_size, 0, NULL, NULL, &err);
	gpu_check_error(err, "clEnqueueMapBuffer() for pinned_h_result failed.");
	/* host memory for cuboid */
	config->h_cuboid = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, config->vector_size, NULL, &err);
	gpu_check_error(err, "clCreateBuffer() for h_cuboid failed.");
	config->pinned_h_cuboid = clEnqueueMapBuffer(config->_cl_queue, config->h_cuboid, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, config->vector_size, 0, NULL, NULL, &err);
	gpu_check_error(err, "clEnqueueMapBuffer() for h_cuboid failed.");
	config->h_cuboid_reference = 0;
	/* host memory for last_trans (the input y[]) */
	config->h_y = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, config->vector_size, NULL, &err);
	gpu_check_error(err, "clCreateBuffer() for h_y failed.");
	config->pinned_h_y = clEnqueueMapBuffer(config->_cl_queue, config->h_y, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, config->vector_size, 0, NULL, NULL, &err);
	gpu_check_error(err, "clEnqueueMapBuffer() for h_y failed.");
	/* Use the pinned buffer in model */
	free(model->last_trans->cuboid[0][0]);
	model->last_trans->cuboid[0][0] = config->pinned_h_y;
	/* save this cuboid pointer so that we can swap it later */
	config->cuboid_y = model->last_trans->cuboid;
	/* fix pointers in cuboid array */
	/* remaining pointers of the 2-d pointer array	*/
	for (i = 0; i < nl; i++)
		for (j = 0; j < nr; j++)
			/* to reach the jth row in the ith layer,
			 * one has to cross i layers i.e., i*(nr*nc)
			 * values first and then j rows i.e., j*nc
			 * values next
			 */
			model->last_trans->cuboid[i][j] =  model->last_trans->cuboid[0][0] + (nl * nr) * i + nr * j;
	/* host memory for ytemp (the output yout[]), only used if GPU/CPU has unified memory */
	if (config->unified_memory_optimization) {
		double ***m;
		config->h_ytemp = clCreateBuffer(config->_cl_context, CL_MEM_READ_WRITE | CL_MEM_ALLOC_HOST_PTR, config->vector_size, NULL, &err);
		gpu_check_error(err, "clCreateBuffer() for h_ytemp failed.");
		config->pinned_h_ytemp = clEnqueueMapBuffer(config->_cl_queue, config->h_ytemp, CL_TRUE, CL_MAP_READ | CL_MAP_WRITE, 0, config->vector_size, 0, NULL, NULL, &err);
		gpu_check_error(err, "clEnqueueMapBuffer() for h_ytemp failed.");
		/* create another cuboid */
		m = (double ***) calloc (nl, sizeof(double **));
		m[0] = (double **) calloc (nl * nr, sizeof(double *));
		/* remaining pointers of the 1-d pointer array	*/
		for (i = 1; i < nl; i++)
	    		m[i] =  m[0] + nr * i;
		/* remaining pointers of the 2-d pointer array	*/
		m[0][0] = config->pinned_h_ytemp;
		for (i = 0; i < nl; i++)
			for (j = 0; j < nr; j++)
				m[i][j] =  m[0][0] + (nl * nr) * i + nr * j;
		/* save this cuboid pointer so that we can swap it later */
		config->cuboid_ytemp = m;
		// printf("Mapped buffer addresses: config->pinned_h_y = %p, config->pinned_h_ytemp = %p\n", config->pinned_h_y, config->pinned_h_ytemp);
	}
}

struct timespec gpu_perf_timediff(struct timespec start, struct timespec end)
{
	struct timespec temp;
	if ((end.tv_nsec-start.tv_nsec)<0) {
		temp.tv_sec = end.tv_sec-start.tv_sec-1;
		temp.tv_nsec = 1000000000+end.tv_nsec-start.tv_nsec;
	} else {
		temp.tv_sec = end.tv_sec-start.tv_sec;
		temp.tv_nsec = end.tv_nsec-start.tv_nsec;
	}
	return temp;
}

void* gpu_allocate_cuboid_static(size_t size)
{
	/* dcuboid_tail will always request double size even if single precison is enabled. */
	/* However, only half of the array is used if we do GPU computation. pinned_h_cuboid will only have half of the requested size */
	/* We must make sure that current_gpu_config and current_gpu_config->gpu_enabled is set correctly if we are not doing GPU */
	size_t adj_vector_size = (current_gpu_config != NULL && current_gpu_config->gpu_enabled) ? current_gpu_config->vector_size / current_gpu_config->element_size * sizeof(double) : 0;
	if (current_gpu_config == NULL || !current_gpu_config->gpu_enabled || size != adj_vector_size) {
	// if(1) {
		void* p = malloc(size);
		// printf("malloc %ld bytes %p\n", size, p);
		return p;
	}
	if (current_gpu_config->h_cuboid_reference > 0) {
		fatal("cuboid memory has not been freed before allocation!");
	}
	if (current_gpu_config->h_cuboid_reference < 0) {
		fatal("cuboid memory has not been initialized.");
	}
	// printf("allocating pinned buffer at %p\n", current_gpu_config->pinned_h_cuboid);
	current_gpu_config->h_cuboid_reference++;
	return current_gpu_config->pinned_h_cuboid;
}

void gpu_free_cuboid_static(void* cuboid)
{
	if (current_gpu_config != NULL && (cuboid == current_gpu_config->pinned_h_y || cuboid == current_gpu_config->pinned_h_ytemp))
	{
		// the pinned buffer (current_gpu_config->pinned_h_y and current_gpu_config->pinned_h_ytemp) has been freed by gpu_destroy(). Don't double free here!
		return;
	}
	if (current_gpu_config == NULL || !current_gpu_config->gpu_enabled || cuboid != current_gpu_config->pinned_h_cuboid) {
	// if (1) {
		// printf("gpu_free_cuboid_static() freeing regular pointer %p\n", cuboid);
		free(cuboid);
		return;
	}
	// printf("gpu_free_cuboid_static() freeing pinned buffer at %p\n", cuboid);
	if (current_gpu_config->h_cuboid_reference > 0) {
		current_gpu_config->h_cuboid_reference--;
	}
}

void gpu_delete_buffers(gpu_config_t *config)
{
	clReleaseMemObject(config->d_dv);
	clReleaseMemObject(config->d_y);
	clReleaseMemObject(config->d_ytemp);
	clReleaseMemObject(config->d_k1);
	clReleaseMemObject(config->d_k2);
	clReleaseMemObject(config->d_k3);
	clReleaseMemObject(config->d_k4);
	clReleaseMemObject(config->d_t1);
	clReleaseMemObject(config->d_p_cuboid);
	clReleaseMemObject(config->d_c_model);
	clReleaseMemObject(config->d_c_layer);
	clEnqueueUnmapMemObject(config->_cl_queue, config->h_result, config->pinned_h_result, 0, NULL, NULL);
	clEnqueueUnmapMemObject(config->_cl_queue, config->h_cuboid, config->pinned_h_cuboid, 0, NULL, NULL);
	clEnqueueUnmapMemObject(config->_cl_queue, config->h_y, config->pinned_h_y, 0, NULL, NULL);
	clReleaseMemObject(config->h_result);
	clReleaseMemObject(config->h_cuboid);
	clReleaseMemObject(config->h_y);
	if (config->unified_memory_optimization) {
		clEnqueueUnmapMemObject(config->_cl_queue, config->h_ytemp, config->pinned_h_ytemp, 0, NULL, NULL);
		clReleaseMemObject(config->h_ytemp);
		/* only delete one cuboid here - the other one will be freed in free_grid_model_vector() */
		/* pinned buffer has been freed above, and gpu_free_cuboid_static() will take care of them */
		if (config->last_io_buf) {
			// printf("freeing cuboid_y at %p, %p\n", config->cuboid_y[0], config->cuboid_y);
			free(config->cuboid_y[0]);
			free(config->cuboid_y);
		}
		else {
			// printf("freeing cuboid_ytemp at %p, %p\n", config->cuboid_ytemp[0], config->cuboid_ytemp);
			free(config->cuboid_ytemp[0]);
			free(config->cuboid_ytemp);
		}
	}
}

void gpu_print_array(gpu_config_t *config, cl_mem d_mem, size_t offset, size_t size, char* msg)
{
#ifdef GPU_DEBUG_PRINT
	int i;
	real * buf = (real*)malloc(size * sizeof(real));
	if (msg == NULL) {
		msg = "";
	}
	clEnqueueReadBuffer(config->_cl_queue, d_mem, CL_TRUE, offset * sizeof(real), size * sizeof(real), buf, 0, NULL, NULL);
	for (i = 0; i < size; ++i) {
		printf("%s%.*g\n", msg, 21, buf[i]);
	}
#endif
}

void gpu_init(gpu_config_t *config, grid_model_t *model)
{
	cl_platform_id* platforms;
	cl_device_id* devices;
	cl_uint value_size;
	size_t info_size;
	char* device_name;
	char* platform_name;
	char compiler_options[512] = {0};
	int err;
	current_gpu_config = config;
	if (!config->gpu_enabled) {
#if ENABLE_TIMER > 0
		clock_gettime(CLOCK_MONOTONIC, &config->time_start);
#endif
		return;
	}
	
	gpu_copy_constants(config, model);

	printf("Initializing GPU device...\n");	

	// open platform	
	err = clGetPlatformIDs(0, NULL, &value_size);
	gpu_check_error(err, "Couldn't open OpenCL platform");

	printf("%d OpenCL platforms detected.\n", value_size);
	if (config->platform_id < 0)
		config->platform_id = 0;
	if (config->platform_id >= value_size) {
		printf("gpu_platform should be in range 0 - %d. Use %d instead\n", value_size - 1, value_size - 1);
		config->platform_id = value_size - 1;
	}
	platforms = (cl_platform_id*) malloc(sizeof(cl_platform_id) * value_size);
	clGetPlatformIDs(value_size, platforms, NULL);
	
	// detect platform info and enable uniform memory optimization if possible
	err = clGetPlatformInfo(platforms[config->platform_id], CL_PLATFORM_VENDOR, 0, NULL, &info_size);
	gpu_check_error(err, "Couldn't get OpenCL platform information");
	platform_name = (char*) malloc(sizeof(char) * info_size);
	clGetPlatformInfo(platforms[config->platform_id], CL_PLATFORM_VENDOR, info_size, platform_name, NULL);
	printf("OpenCL Platform Vendor: %s\n", platform_name);
	if (!strcmp(platform_name, "Intel")) {
		config->unified_memory_optimization = 1;
		printf("Unified memory optimization has been enabled on this platform.\n");
	}
	free(platform_name);
	
	// open device
	err = clGetDeviceIDs(platforms[config->platform_id], CL_DEVICE_TYPE_GPU, 0, NULL, &value_size);
	gpu_check_error(err, "Couldn't access any GPU devices");
	printf("%d OpenCL devices detected.\n", value_size);
	if (config->device_id < 0)
		config->device_id = 0;
	if (config->device_id >= value_size) {
		printf("gpu_device should be in range 0 - %d. Use %d instead\n", value_size - 1, value_size - 1);
		config->device_id = value_size - 1;
	}
	devices = (cl_device_id*) malloc(sizeof(cl_device_id) * value_size);
	clGetDeviceIDs(platforms[config->platform_id], CL_DEVICE_TYPE_GPU, value_size, devices, NULL);
	free(platforms);

	// print device name
	clGetDeviceInfo(devices[config->device_id], CL_DEVICE_NAME, 0, NULL, &info_size);
	device_name = (char*) malloc(info_size);
	clGetDeviceInfo(devices[config->device_id], CL_DEVICE_NAME, info_size, device_name, NULL);
	printf("Selected OpenCL Platform %d Device %d: %s\n", config->platform_id, config->device_id, device_name);
	free(device_name);

	// Create OpenCL context
	config->_cl_context = clCreateContext(NULL, 1, &devices[config->device_id], NULL, NULL, &err);
	gpu_check_error(err, "Couldn't create an OpenCL context");

	// Create a command queue
	config->_cl_queue = clCreateCommandQueue(config->_cl_context, devices[config->device_id], 0, &err);
	gpu_check_error(err, "Couldn't create an OpenCL command queue");

	// Create memory objects
	gpu_create_buffers(config, model);

	// Setup kernel dimensions
	if (model->rows % config->local_work_size[1] || model->cols % config->local_work_size[0])
	{
		fatal("Invalid number of rows or columns");
	}
	config->global_work_size[1] = model->rows;
	config->global_work_size[0] = model->cols;


	// Create OpenCL program
	config->_cl_program = clCreateProgramWithSource(config->_cl_context, 1, (const char**)&config->_cl_kernel_string, &config->_cl_kernel_size, &err);
	gpu_check_error(err, "Couldn't create the OpenCL program");
	int vector_size = config->vector_size / config->element_size;
	sprintf(compiler_options, "-cl-denorms-are-zero -cl-strict-aliasing -cl-fast-relaxed-math " \
			"-DENABLE_SECONDARY_MODEL=%d -DNUMBER_OF_LAYERS=%d -DLOCAL_SIZE_1=(size_t)%d -DLOCAL_SIZE_0=(size_t)%d -DNUMBER_OF_ROWS=(size_t)%d -DNUMBER_OF_COLS=(size_t)%d -DLOCAL_SIZE_1D=(size_t)%d " \
			"-Dreal=%s",
			model->config.model_secondary, model->n_layers, (int)config->local_work_size[1], (int)config->local_work_size[0], model->rows, model->cols, (int)(config->local_work_size[0] * config->local_work_size[1]),
			config->element_size == sizeof(double) ? "double" : "float -cl-single-precision-constant");
	// Build OpenCL program
	printf("compiling kernel with options: %s\n", compiler_options);
	err = clBuildProgram(config->_cl_program, 0, NULL, compiler_options, NULL, NULL);
	if(err < 0) {
		char* build_log;
		// Build failure, print log 
		clGetProgramBuildInfo(config->_cl_program, devices[config->device_id], CL_PROGRAM_BUILD_LOG, 0, NULL, &info_size);
		build_log = (char*) malloc(info_size + 1);
		build_log[info_size] = '\0';
		clGetProgramBuildInfo(config->_cl_program, devices[config->device_id], CL_PROGRAM_BUILD_LOG, info_size + 1, build_log, NULL);
		printf("%s\n", build_log);
		free(build_log);
		fatal("Unabled to build the OpenCL program");
	}
	
	// Create kernel entry point
	config->_cl_kernel_rk4 = clCreateKernel(config->_cl_program, "slope_fn_grid_gpu", &err);
	gpu_check_error(err, "Couldn't create OpenCL kernel slope_fn_grid_gpu()");
	config->_cl_kernel_average = clCreateKernel(config->_cl_program, "rk4_average", &err);
	gpu_check_error(err, "Couldn't create OpenCL kernel rk4_average()");
	config->_cl_kernel_average_with_maxdiff = clCreateKernel(config->_cl_program, "rk4_average_with_maxdiff", &err);
	gpu_check_error(err, "Couldn't create OpenCL kernel rk4_average_with_maxdiff()");
	config->_cl_kernel_max_reduce = clCreateKernel(config->_cl_program, "max_reduce", &err);
	gpu_check_error(err, "Couldn't create OpenCL kernel max_reduce()");


	/* Create kernel arguments */
	err  = clSetKernelArg(config->_cl_kernel_rk4, GRID_CONST_MODEL, sizeof(cl_mem), &config->d_c_model);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_CONST_LAYER, sizeof(cl_mem), &config->d_c_layer);
	// err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IO_V, sizeof(cl_mem), &config->d_v);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_OUT_DV, sizeof(cl_mem), &config->d_dv);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_NL, sizeof(model->n_layers), &model->n_layers);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_NR, sizeof(model->rows), &model->rows);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_NC, sizeof(model->cols), &model->cols);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_LOCALMEM, 4 * (config->local_work_size[0] + 2) * (config->local_work_size[1] + 2) * config->element_size + config->extra_size, NULL); // shared memory for 4 layers
	if (config->unified_memory_optimization) {
		// use the zero-copy buffer allocated by CL_MEM_ALLOC_HOST_PTR
		err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_CUBOID, sizeof(cl_mem), &config->h_cuboid);
	}
	else {
		err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_CUBOID, sizeof(cl_mem), &config->d_p_cuboid);
	}
	// the 9th argument is h
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_K, sizeof(cl_mem), &config->d_k1);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_Y, sizeof(cl_mem), &config->d_y);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument for slope_fn_grid_gpu()");

	err  = clSetKernelArg(config->_cl_kernel_average, AVG_IN_Y, sizeof(cl_mem), &config->d_y);
	err |= clSetKernelArg(config->_cl_kernel_average, AVG_IN_K1, sizeof(cl_mem), &config->d_k1);
	err |= clSetKernelArg(config->_cl_kernel_average, AVG_IN_K2, sizeof(cl_mem), &config->d_k2);
	err |= clSetKernelArg(config->_cl_kernel_average, AVG_IN_K3, sizeof(cl_mem), &config->d_k3);
	err |= clSetKernelArg(config->_cl_kernel_average, AVG_IN_K4, sizeof(cl_mem), &config->d_k4);
	// the 5th argument is h
	err |= clSetKernelArg(config->_cl_kernel_average, AVG_OUT_YOUT, sizeof(cl_mem), &config->d_dv);
	err |= clSetKernelArg(config->_cl_kernel_average, AVG_N, sizeof(vector_size), &vector_size);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument for rk4_average()");
	err  = clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_Y, sizeof(cl_mem), &config->d_y);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_K1, sizeof(cl_mem), &config->d_k1);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_K2, sizeof(cl_mem), &config->d_k2);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_K3, sizeof(cl_mem), &config->d_k3);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_K4, sizeof(cl_mem), &config->d_k4);
	// the 5th argument is h
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_OUT_YOUT, sizeof(cl_mem), &config->d_dv);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_N, sizeof(vector_size), &vector_size);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_YTEMP, sizeof(cl_mem), &config->d_ytemp);
	err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_LOCALMEM, (config->local_work_size[0]) * (config->local_work_size[1]) * config->element_size, NULL);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument for rk4_average()");
	int remained_items = (config->global_work_size[0] * config->global_work_size[1]) / (config->local_work_size[0] * config->local_work_size[1]);
	err  = clSetKernelArg(config->_cl_kernel_max_reduce, MAX_IO_Y, sizeof(cl_mem), &config->d_dv);
	err |= clSetKernelArg(config->_cl_kernel_max_reduce, MAX_N, sizeof(remained_items), &remained_items);
	err |= clSetKernelArg(config->_cl_kernel_max_reduce, MAX_LOCALMEM, (config->local_work_size[0]) * (config->local_work_size[1]) * config->element_size, NULL);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument for max_reduce()");
	
	printf("OpenCL initialized.\n");
#if CPU_C_STATE_CONTROL > 0
	config->cpu_dma = open("/dev/cpu_dma_latency", O_WRONLY);
	if (config->cpu_dma > 0) {
		int32_t target = 0;
		write(config->cpu_dma, &target, sizeof(target));
		printf("CPU C-states disabled.\n");
	}
	else {
		printf("Can't disable C-states. Do you have write permission to /dev/cpu_dma_latency?\n");
	}
#endif
	free(devices);
#if ENABLE_TIMER > 0
	clock_gettime(CLOCK_MONOTONIC, &config->time_start);
#endif
}

void gpu_destroy(gpu_config_t *config)
{
#if ENABLE_TIMER > 0
	clock_gettime(CLOCK_MONOTONIC, &config->time_end);
	struct timespec time_diff = gpu_perf_timediff(config->time_start, config->time_end);
#endif
	if (!config->gpu_enabled) {
#if ENABLE_TIMER > 0
		printf("CPU Time: %ld.%09lds\n", time_diff.tv_sec, time_diff.tv_nsec);
#endif
		return;
	}
#if ENABLE_TIMER > 0
	printf("GPU Time: %ld.%09lds\n", time_diff.tv_sec, time_diff.tv_nsec);
#endif
	free(config->layer);
	gpu_delete_buffers(config);

	clFinish(config->_cl_queue);
	clReleaseKernel(config->_cl_kernel_rk4);
	clReleaseKernel(config->_cl_kernel_average);
	clReleaseKernel(config->_cl_kernel_average_with_maxdiff);
	clReleaseKernel(config->_cl_kernel_max_reduce);
	
	clReleaseCommandQueue(config->_cl_queue);
	clReleaseProgram(config->_cl_program);
	clReleaseContext(config->_cl_context);
	config->gpu_enabled = 0; // prevents pinned memory allocation for steady state calculation
	puts("OpenCL environment has been cleaned up.\n");
#if CPU_C_STATE_CONTROL > 0
	if (config->cpu_dma > 0) {
		close(config->cpu_dma);
	}
#endif
}

/* 
 * 4th order Runge Kutta solver	with adaptive step sizing.
 * It integrates and solves the ODE dy + cy = p between
 * t and t+h. It returns the correct step size to be used 
 * next time. slope function f is the call back used to 
 * evaluate the derivative at each point
 */
#define RK4_SAFETY		0.95
#define RK4_MAXUP		5.0
#define RK4_MAXDOWN		10.0
#define RK4_PRECISION	0.01
double rk4_gpu(gpu_config_t *config, void *model, double *y, void *p, int n, double *h, grid_model_vector_t *last_trans)
{
	int i;
	real max, new_h = (*h);
	cl_mem *d_input, *d_output;

	size_t buffer_size = config->vector_size;
	// printf("buffer_size = %zu, n = %d\n", buffer_size, n);
	// printf("y = %p, yout = %p, p->cuboid = %p\n", y, last_trans->cuboid[0][0], ((grid_model_vector_t*)p)->cuboid[0][0]);
	if (config->unified_memory_optimization) {
		/* use zero-copy buffer config->h_cuboid as cuboid input */
		/* switching between config->h_y and config->h_ytemp */
		/* config->last_io_buf is initialized to 0. And initially we are using config->h_y as input */
		config->last_io_buf = !config->last_io_buf;
		if (config->last_io_buf) {
			d_input = &config->h_y;
			d_output = &config->h_ytemp;
		}
		else {
			d_input = &config->h_ytemp;
			d_output = &config->h_y;
		}
	}
	else {
		/* copy p->cuboid to device */
		/* if cuboid is single precision, it will be initialized as single precision from the caller */
		clEnqueueWriteBuffer(config->_cl_queue, config->d_p_cuboid, CL_FALSE, 0, buffer_size, ((grid_model_vector_t*)p)->cuboid[0][0], 0, NULL, NULL);
		/* switching between config->d_y and config->d_ytemp */
		config->last_io_buf = !config->last_io_buf;
		/* copy y to device, only if its pointer location changes */
		if ( y != config->last_h_y)
		{
			clEnqueueWriteBuffer(config->_cl_queue, config->d_y, CL_FALSE, 0, buffer_size, y, 0, NULL, NULL);
			config->last_h_y = y;
			/* reseting to the initial state */
			config->last_io_buf = 0;
		}

		if (config->last_io_buf) {
			d_input = &config->d_ytemp;
			d_output = &config->d_y;
		}
		else {
			d_input = &config->d_y;
			d_output = &config->d_ytemp;
		}
	}

	/* evaluate the slope k1 at the beginning */
	DEBUG_Flush(config->_cl_queue);
	// clEnqueueReadBuffer(config->_cl_queue, config->d_y, CL_TRUE, 0, buffer_size, input, 0, NULL, NULL);
	slope_fn_grid_gpu_kernel(config, model, d_input, p, &config->d_k1);
	/* try until accuracy is achieved	*/
	do {
		(*h) = new_h;
		DEBUG_Flush(config->_cl_queue);
		/* try RK4 once with normal step size	*/
		rk4_core_gpu_kernel(config, model, d_input, &config->d_k1, p, n, (*h), d_output, NULL, 0);
		DEBUG_Flush(config->_cl_queue);
		/* repeat it with two half-steps	*/
		rk4_core_gpu_kernel(config, model, d_input, &config->d_k1, p, n, (*h)/2.0, &config->d_t1, NULL, 0);
		DEBUG_Flush(config->_cl_queue);
		/* y after 1st half-step is in t1. re-evaluate k1 for this	*/
		slope_fn_grid_gpu_kernel(config, model, &config->d_t1, p, &config->d_k1);
		DEBUG_Flush(config->_cl_queue);
		/* get output of the second half-step in t2	*/	
		rk4_core_gpu_kernel(config, model, &config->d_t1, &config->d_k1, p, n, (*h)/2.0, &config->d_dv, d_output, 1);
		DEBUG_Flush(config->_cl_queue);
		/* find the max diff between these two results:
		 * use t1 to store the diff
		 */
		clEnqueueReadBuffer(config->_cl_queue, config->d_dv, CL_TRUE, 0, config->element_size, config->pinned_h_result, 0, NULL, NULL);
		max = ((real*)config->pinned_h_result)[0]; // TODO: support single precision

		/* 
		 * compute the correct step size: see equation 
		 * 16.2.10  in chapter 16 of "Numerical Recipes
		 * in C"
		 */
		/* accuracy OK. increase step size	*/
		if (max <= RK4_PRECISION) {
			new_h = RK4_SAFETY * (*h) * pow(fabs(RK4_PRECISION/max), 0.2);
			if (new_h > RK4_MAXUP * (*h))
				new_h = RK4_MAXUP * (*h);
		/* inaccuracy error. decrease step size	and compute again */
		} else {
			new_h = RK4_SAFETY * (*h) * pow(fabs(RK4_PRECISION/max), 0.25);
			if (new_h < (*h) / RK4_MAXDOWN)
				new_h = (*h) / RK4_MAXDOWN;
		}

	} while (new_h < (*h));

	/* commit ytemp to yout	*/
	if (config->unified_memory_optimization) {
		if (config->last_io_buf) {
			last_trans->cuboid = config->cuboid_ytemp; // cuboid config->h_ytemp
			// printf("last_trans->cuboid = cuboid_ytemp = %p\n", config->cuboid_ytemp);
		}
		else {
			last_trans->cuboid = config->cuboid_y; // cuboid config->h_y
			// printf("last_trans->cuboid = cuboid_y = %p\n", config->cuboid_y);
		}
	}
	else {
		clEnqueueReadBuffer(config->_cl_queue, *d_output, CL_TRUE, 0, buffer_size, last_trans->cuboid[0][0], 0, NULL, NULL);
	}

	/* return the step-size	*/
	return new_h;
}

/* core of the 4th order Runge-Kutta method, where the Euler step
 * (y(n+1) = y(n) + h * k1 where k1 = dydx(n)) is provided as an input.
 * to evaluate dydx at different points, a call back function f (slope
 * function) is also passed as a parameter. Given values for y, and k1, 
 * this function advances the solution over an interval h, and returns
 * the solution in yout. For details, see the discussion in "Numerical 
 * Recipes in C", Chapter 16, from 
 * http://www.nrbook.com/a/bookcpdf/c16-1.pdf
 */
void rk4_core_gpu_kernel(gpu_config_t *config, void *model, cl_mem *d_y, cl_mem *d_k1, void *p, int n, real h, cl_mem *d_yout, cl_mem *d_ytemp, int do_maxdiff)
{
	int i;
	real h_real;
	int err;
	/* k2 is the slope at the trial midpoint (t) found using
	 * slope k1 (which is at the starting point).
	 */
	/* t = y + h/2 * k1 (t = y; t += h/2 * k1) */

	/* k2 = slope at t */
	h_real = h / 2.0;
	/* inputs */
	err = clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_Y, sizeof(cl_mem), d_y);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_H, sizeof(h_real), &h_real);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_K, sizeof(cl_mem), d_k1);
	/* output to d_k2 */
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_OUT_DV, sizeof(cl_mem), &config->d_k2);
	/* IO buffer */
	err = clSetKernelArg(config->_cl_kernel_rk4, GRID_IO_V, sizeof(cl_mem), d_yout);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_core_gpu_kernel()");
	err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_rk4, 2, NULL, config->global_work_size, config->local_work_size, 0, NULL, NULL);
	gpu_check_error(err, "Cannot launch kernel rk4_core_gpu_kernel()!");
	/* k3 is the slope at the trial midpoint (t) found using
	 * slope k2 found above.
	 */
	/* t =  y + h/2 * k2 (t = y; t += h/2 * k2) */

	/* k3 = slope at t */
	/* h is not changed, d_k2 is the output of the previous kernel */
	err = clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_K, sizeof(cl_mem), &config->d_k2);
	/* output to d_k3 */
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_OUT_DV, sizeof(cl_mem), &config->d_k3);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_core_gpu_kernel()");
	err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_rk4, 2, NULL, config->global_work_size, config->local_work_size, 0, NULL, NULL);
	gpu_check_error(err, "Cannot launch kernel rk4_core_gpu_kernel()!");
	/* k4 is the slope at trial endpoint (t) found using
	 * slope k3 found above.
	 */
	/* t =  y + h * k3 (t = y; t += h * k3) */

	/* k4 = slope at t */
	/* inputs */
	h_real = h;
	err = clSetKernelArg(config->_cl_kernel_rk4, GRID_H, sizeof(h_real), &h_real);
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IN_K, sizeof(cl_mem), &config->d_k3);
	/* output to d_k4 */
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_OUT_DV, sizeof(cl_mem), &config->d_k4);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_core_gpu_kernel()");
	err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_rk4, 2, NULL, config->global_work_size, config->local_work_size, 0, NULL, NULL);
	gpu_check_error(err, "Cannot launch kernel rk4_core_gpu_kernel()!");
	/* yout = y + h*(k1/6 + k2/3 + k3/3 + k4/6)	*/
	if (do_maxdiff) {
		err = clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_Y, sizeof(cl_mem), d_y);
		err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_K1, sizeof(cl_mem), d_k1);
		err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_H, sizeof(h_real), &h_real);
		err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_OUT_YOUT, sizeof(cl_mem), d_yout);
		err |= clSetKernelArg(config->_cl_kernel_average_with_maxdiff, AVG_IN_YTEMP, sizeof(cl_mem), d_ytemp);
		gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_average_with_maxdiff()");
		/* first phase reduction, results to be saved to yout */
		size_t avg_kernel_global_size = config->global_work_size[0] * config->global_work_size[1];
		size_t avg_kernel_local_size = config->local_work_size[0] * config->local_work_size[1];
		err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_average_with_maxdiff, 1, NULL, &avg_kernel_global_size, &avg_kernel_local_size, 0, NULL, NULL);
		gpu_check_error(err, "Cannot launch kernel rk4_average_with_maxdiff()!");
		/* second phase reduction, with 1 block only, read inputs from yout */
		err = clSetKernelArg(config->_cl_kernel_max_reduce, MAX_IO_Y, sizeof(cl_mem), d_yout);
		gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_average_with_maxdiff()");
		err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_max_reduce, 1, NULL, &avg_kernel_local_size, &avg_kernel_local_size, 0, NULL, NULL);
		gpu_check_error(err, "Cannot launch kernel max_reduce()!");
	}
	else {
		err = clSetKernelArg(config->_cl_kernel_average, AVG_IN_Y, sizeof(cl_mem), d_y);
		err |= clSetKernelArg(config->_cl_kernel_average, AVG_IN_K1, sizeof(cl_mem), d_k1);
		err |= clSetKernelArg(config->_cl_kernel_average, AVG_H, sizeof(h_real), &h_real);
		err |= clSetKernelArg(config->_cl_kernel_average, AVG_OUT_YOUT, sizeof(cl_mem), d_yout);
		gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_average()");
		size_t avg_kernel_global_size = config->global_work_size[0] * config->global_work_size[1];
		size_t avg_kernel_local_size = config->local_work_size[0] * config->local_work_size[1];
		err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_average, 1, NULL, &avg_kernel_global_size, &avg_kernel_local_size, 0, NULL, NULL);
		gpu_check_error(err, "Cannot launch kernel rk4_average()!");
	}
}

void slope_fn_grid_gpu_kernel(gpu_config_t *config, grid_model_t *model, cl_mem *d_v, grid_model_vector_t *p, cl_mem *d_dv)
{
	int err;

	/* disable endpoint calculation */
	real h = 0.0;
	err = clSetKernelArg(config->_cl_kernel_rk4, GRID_H, sizeof(h), &h);
	/* input from d_v */
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_IO_V, sizeof(cl_mem), d_v);
	/* output to d_dv */
	err |= clSetKernelArg(config->_cl_kernel_rk4, GRID_OUT_DV, sizeof(cl_mem), d_dv);
	gpu_check_error(err, "Couldn't setup a OpenCL kernel argument in rk4_core_gpu_kernel()");
	err = clEnqueueNDRangeKernel(config->_cl_queue, config->_cl_kernel_rk4, 2, NULL, config->global_work_size, config->local_work_size, 0, NULL, NULL);
	gpu_check_error(err, "Cannot launch kernel slope_fn_grid_gpu()!");
}

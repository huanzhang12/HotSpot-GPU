#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include "temperature_grid.h"
#include "temperature.h"
#include "gpu.h"

#include "rk4_kernel_str.c"

/* default thermal configuration parameters	*/
gpu_config_t default_gpu_config(void)
{
	gpu_config_t config;
	
	config.gpu_enabled = 0;
	config.platform_id = 1;
	config.device_id = 1;
	
	/* rk4_cl is a string generated by rk4.cl */
	config._cl_kernel_string = rk4_cl;
	config._cl_kernel_size = rk4_cl_len;
	config._cl_kernel_rk4 = NULL;
	config._cl_context = NULL;
	config._cl_program = NULL;
	config._cl_queue = NULL;

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

void gpu_init(gpu_config_t *config)
{
	cl_platform_id platform;
	cl_device_id device;
	size_t value_size;
	char* device_name;
	int err;
	
	if (!config->gpu_enabled) {
		return;
	}
	
	printf("Initializing GPU device...\n");	

	// open platform	
	err = clGetPlatformIDs(config->platform_id, &platform, NULL);
	if(err < 0) {
		fatal("Couldn't open OpenCL platform");
	} 
	
	// open device
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, config->device_id, &device, NULL);
	if(err == CL_DEVICE_NOT_FOUND) {
		err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_CPU, config->device_id, &device, NULL);
		warning("No GPU devices found. Using CPU instead.");
	}
	if(err < 0) {
		fatal("Couldn't access any OpenCL devices"); 
	}

	// print device name
	clGetDeviceInfo(device, CL_DEVICE_NAME, 0, NULL, &value_size);
	device_name = (char*) malloc(value_size);
	clGetDeviceInfo(device, CL_DEVICE_NAME, value_size, device_name, NULL);
	printf("Selected OpenCL Platform %d Device %d: %s\n", config->platform_id, config->device_id, device_name);
	free(device_name);

	// Create OpenCL context
	config->_cl_context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	if(err < 0) {
		fatal("Couldn't create an OpenCL context");  
	}

	// Create OpenCL program
	config->_cl_program = clCreateProgramWithSource(config->_cl_context, 1, (const char**)&config->_cl_kernel_string, &config->_cl_kernel_size, &err);
	if(err < 0) {
		fatal("Couldn't create the OpenCL program");
	}
	
	// Build OpenCL program
	err = clBuildProgram(config->_cl_program, 0, NULL, NULL, NULL, NULL);
	if(err < 0) {
		char* build_log;
		// Build failure, print log 
		clGetProgramBuildInfo(config->_cl_program, device, CL_PROGRAM_BUILD_LOG, 0, NULL, &value_size);
		build_log = (char*) malloc(value_size + 1);
		build_log[value_size] = '\0';
		clGetProgramBuildInfo(config->_cl_program, device, CL_PROGRAM_BUILD_LOG, value_size + 1, build_log, NULL);
		printf("%s\n", build_log);
		free(build_log);
		fatal("Unabled to build the OpenCL program");
	}
	
	// Create a command queue
	config->_cl_queue = clCreateCommandQueue(config->_cl_context, device, 0, &err);
	if(err < 0) {
		fatal("Couldn't create an OpenCL command queue");
	};
	
	printf("OpenCL initialized.\n");
	
}

void gpu_destroy(gpu_config_t *config)
{
		
	if (!config->gpu_enabled) { 
		return;
	}
	
	// TODO
	// clReleaseKernel(config->_cl_kernel);
	// clReleaseMemObject(sum_buffer);
	// clReleaseMemObject(input_buffer);

	clReleaseCommandQueue(config->_cl_queue);
	clReleaseProgram(config->_cl_program);
	clReleaseContext(config->_cl_context);
	puts("OpenCL environment has been cleaned up.\n");
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
double rk4_gpu(void *model, double *y, void *p, int n, double *h, double *yout, gpu_config_t *config)
{
	int i;
	double *k1, *t1, *t2, *ytemp, max, new_h = (*h);

	k1 = dvector(n);
	t1 = dvector(n);
	t2 = dvector(n);
	ytemp = dvector(n); 

	/* evaluate the slope k1 at the beginning */
	slope_fn_grid_gpu(model, y, p, k1);

	/* try until accuracy is achieved	*/
	do {
		(*h) = new_h;

		/* try RK4 once with normal step size	*/
		rk4_core_gpu(model, y, k1, p, n, (*h), ytemp);

		/* repeat it with two half-steps	*/
		rk4_core_gpu(model, y, k1, p, n, (*h)/2.0, t1);

		/* y after 1st half-step is in t1. re-evaluate k1 for this	*/
		slope_fn_grid_gpu(model, t1, p, k1);

		/* get output of the second half-step in t2	*/	
		rk4_core_gpu(model, t1, k1, p, n, (*h)/2.0, t2);

		/* find the max diff between these two results:
		 * use t1 to store the diff
		 */
		for(i=0; i < n; i++)
			t1[i] = fabs(ytemp[i] - t2[i]);
		max = t1[0];
		for(i=1; i < n; i++)
			if (max < t1[i])
				max = t1[i];

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
	copy_dvector(yout, ytemp, n);

	/* clean up */
	free_dvector(k1);
	free_dvector(t1);
	free_dvector(t2);
	free_dvector(ytemp);

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
void rk4_core_gpu(void *model, double *y, double *k1, void *p, int n, double h, double *yout)
{
	int i;
	double *t, *k2, *k3, *k4;
	k2 = dvector(n);
	k3 = dvector(n);
	k4 = dvector(n);
	t = dvector(n);

	/* k2 is the slope at the trial midpoint (t) found using 
	 * slope k1 (which is at the starting point).
	 */
	/* t = y + h/2 * k1 (t = y; t += h/2 * k1) */
	for(i=0; i < n; i++)
		t[i] = y[i] + h/2.0 * k1[i];
	
	/* k2 = slope at t */
	slope_fn_grid_gpu(model, t, p, k2); 

	/* k3 is the slope at the trial midpoint (t) found using
	 * slope k2 found above.
	 */
	/* t =  y + h/2 * k2 (t = y; t += h/2 * k2) */
	for(i=0; i < n; i++)
		t[i] = y[i] + h/2.0 * k2[i];
	/* k3 = slope at t */
	slope_fn_grid_gpu(model, t, p, k3);

	/* k4 is the slope at trial endpoint (t) found using
	 * slope k3 found above.
	 */
	/* t =  y + h * k3 (t = y; t += h * k3) */
	for(i=0; i < n; i++)
		t[i] = y[i] + h * k3[i];

	/* k4 = slope at t */
	slope_fn_grid_gpu(model, t, p, k4);

	/* yout = y + h*(k1/6 + k2/3 + k3/3 + k4/6)	*/
	for (i =0; i < n; i++) 
		yout[i] = y[i] + h * (k1[i] + 2*k2[i] + 2*k3[i] + k4[i])/6.0;

	free_dvector(k2);
	free_dvector(k3);
	free_dvector(k4);
	free_dvector(t);
}

/* function to access a 1-d array as a 3-d matrix	*/
#define A3D(array,n,i,j,nl,nr,nc)		(array[(n)*(nr)*(nc) + (i)*(nc) + (j)])
/* macros for calculating currents(power values)	*/
/* current(power) from the next cell north. zero if on northern boundary	*/
# define NP(l,v,n,i,j,nl,nr,nc)		((i > 0) ? ((A3D(v,n,i-1,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].ry) : 0.0)
/* current(power) from the next cell south. zero if on southern boundary	*/
# define SP(l,v,n,i,j,nl,nr,nc)		((i < nr-1) ? ((A3D(v,n,i+1,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].ry) : 0.0)
/* current(power) from the next cell east. zero if on eastern boundary	*/
# define EP(l,v,n,i,j,nl,nr,nc)		((j < nc-1) ? ((A3D(v,n,i,j+1,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rx) : 0.0)
/* current(power) from the next cell west. zero if on western boundary	*/
# define WP(l,v,n,i,j,nl,nr,nc)		((j > 0) ? ((A3D(v,n,i,j-1,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rx) : 0.0)
/* current(power) from the next cell below. zero if on bottom face		*/
# define BP(l,v,n,i,j,nl,nr,nc)		((n < nl-1) ? ((A3D(v,n+1,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz) : 0.0)
/* current(power) from the next cell above. zero if on top face			*/
# define AP(l,v,n,i,j,nl,nr,nc)		((n > 0) ? ((A3D(v,n-1,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n-1].rz) : 0.0)

/* compute the slope vector for the grid cells. the transient
 * equation is CdV + sum{(T - Ti)/Ri} = P 
 * so, slope = dV = [P + sum{(Ti-T)/Ri}]/C
 */
void slope_fn_grid_gpu(grid_model_t *model, double *v, grid_model_vector_t *p, double *dv)
{
	int n, i, j;
	/* sum of the currents(power values)	*/
	double psum;
	
	/* shortcuts for cell width(cw) and cell height(ch)	*/
	double cw = model->width / model->cols;
	double ch = model->height / model->rows;

	/* shortcuts	*/
	thermal_config_t *c = &model->config;
	layer_t *l = model->layers;
	int nl = model->n_layers;
	int nr = model->rows;
	int nc = model->cols;
	int spidx, hsidx, metalidx, c4idx, subidx, solderidx, pcbidx;
	int model_secondary = model->config.model_secondary;
	
	/* pointer to the starting address of the extra nodes	*/
	double *x = v + nl*nr*nc;
	
	if (!model->config.model_secondary) {
		spidx = nl - DEFAULT_PACK_LAYERS + LAYER_SP;
		hsidx = nl - DEFAULT_PACK_LAYERS + LAYER_SINK;
	} else {
		spidx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS + LAYER_SP;
		hsidx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS + LAYER_SINK;
		metalidx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS - SEC_CHIP_LAYERS + LAYER_METAL;
		c4idx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS - SEC_CHIP_LAYERS + LAYER_C4;
		subidx = nl - SEC_PACK_LAYERS + LAYER_SUB;
		solderidx = nl - SEC_PACK_LAYERS + LAYER_SOLDER;
		pcbidx = nl - SEC_PACK_LAYERS + LAYER_PCB;		
	}
	
	/* for each grid cell	*/
	for(n=0; n < nl; n++)
		for(i=0; i < nr; i++)
			for(j=0; j < nc; j++) {
				if (n==LAYER_SI && model_secondary) { //top silicon layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,metalidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[metalidx].rz) +
					   ((A3D(v,n+1,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==spidx && model_secondary) { //spreader layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,metalidx-1,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[metalidx-1].rz) +
					   ((A3D(v,hsidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==metalidx && model_secondary) { //metal layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,c4idx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[c4idx].rz) +
					   ((A3D(v,LAYER_SI,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==metalidx-1 && model_secondary) { // TIM layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,metalidx-2,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[metalidx-2].rz) +
					   ((A3D(v,spidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==c4idx && model_secondary) { //C4 layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,subidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[subidx].rz) +
					   ((A3D(v,metalidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==subidx && model_secondary) { //Substrate layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,solderidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[solderidx].rz) +
					   ((A3D(v,c4idx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==pcbidx && model_secondary) { //PCB layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,solderidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz);
				} else if (n==hsidx && model_secondary) { // heatsink layer
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   ((A3D(v,spidx,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[spidx].rz);
				} else {
					/* sum the currents(power values) to cells north, south, 
				 	* east, west, above and below
				 	*/
					psum = NP(l,v,n,i,j,nl,nr,nc) + SP(l,v,n,i,j,nl,nr,nc) + 
					   EP(l,v,n,i,j,nl,nr,nc) + WP(l,v,n,i,j,nl,nr,nc) + 
					   AP(l,v,n,i,j,nl,nr,nc) + BP(l,v,n,i,j,nl,nr,nc);
				}

				/* spreader core is connected to its periphery	*/
				if (n == spidx) {
					/* northern boundary - edge cell has half the ry	*/
					if (i == 0)
						psum += (x[SP_N] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_sp1_y); 
					/* southern boundary - edge cell has half the ry	*/
					if (i == nr-1)
						psum += (x[SP_S] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_sp1_y); 
					/* eastern boundary	 - edge cell has half the rx	*/
					if (j == nc-1)
						psum += (x[SP_E] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_sp1_x); 
					/* western boundary	 - edge cell has half the rx	*/
					if (j == 0)
						psum += (x[SP_W] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_sp1_x); 
				/* heatsink core is connected to its inner periphery and ambient	*/
				} else if (n == hsidx) {
					/* all nodes are connected to the ambient	*/
					psum += (c->ambient - A3D(v,n,i,j,nl,nr,nc))/l[n].rz;
					/* northern boundary - edge cell has half the ry	*/
					if (i == 0)
						psum += (x[SINK_C_N] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_hs1_y); 
					/* southern boundary - edge cell has half the ry	*/
					if (i == nr-1)
						psum += (x[SINK_C_S] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_hs1_y); 
					/* eastern boundary	 - edge cell has half the rx	*/
					if (j == nc-1)
						psum += (x[SINK_C_E] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_hs1_x); 
					/* western boundary	 - edge cell has half the rx	*/
					if (j == 0)
						psum += (x[SINK_C_W] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_hs1_x); 
				}	else if (n == pcbidx && model->config.model_secondary) {
					/* all nodes are connected to the ambient	*/
					psum += (c->ambient - A3D(v,n,i,j,nl,nr,nc))/(model->config.r_convec_sec * 
								   (model->config.s_pcb * model->config.s_pcb) / (cw * ch));
					/* northern boundary - edge cell has half the ry	*/
					if (i == 0)
						psum += (x[PCB_C_N] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_pcb1_y); 
					/* southern boundary - edge cell has half the ry	*/
					if (i == nr-1)
						psum += (x[PCB_C_S] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_pcb1_y); 
					/* eastern boundary	 - edge cell has half the rx	*/
					if (j == nc-1)
						psum += (x[PCB_C_E] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_pcb1_x); 
					/* western boundary	 - edge cell has half the rx	*/
					if (j == 0)
						psum += (x[PCB_C_W] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_pcb1_x); 
				}	else if (n == subidx && model->config.model_secondary) {
					/* northern boundary - edge cell has half the ry	*/
					if (i == 0)
						psum += (x[SUB_N] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_sub1_y); 
					/* southern boundary - edge cell has half the ry	*/
					if (i == nr-1)
						psum += (x[SUB_S] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_sub1_y); 
					/* eastern boundary	 - edge cell has half the rx	*/
					if (j == nc-1)
						psum += (x[SUB_E] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_sub1_x); 
					/* western boundary	 - edge cell has half the rx	*/
					if (j == 0)
						psum += (x[SUB_W] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_sub1_x); 
				}	else if (n == solderidx && model->config.model_secondary) {
					/* northern boundary - edge cell has half the ry	*/
					if (i == 0)
						psum += (x[SOLDER_N] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_solder1_y); 
					/* southern boundary - edge cell has half the ry	*/
					if (i == nr-1)
						psum += (x[SOLDER_S] - A3D(v,n,i,j,nl,nr,nc))/(l[n].ry/2.0 + nc*model->pack.r_solder1_y); 
					/* eastern boundary	 - edge cell has half the rx	*/
					if (j == nc-1)
						psum += (x[SOLDER_E] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_solder1_x); 
					/* western boundary	 - edge cell has half the rx	*/
					if (j == 0)
						psum += (x[SOLDER_W] - A3D(v,n,i,j,nl,nr,nc))/(l[n].rx/2.0 + nr*model->pack.r_solder1_x); 
				}

				/* update the current cell's temperature	*/	   
				A3D(dv,n,i,j,nl,nr,nc) = (p->cuboid[n][i][j] + psum) / l[n].c;
			}
	slope_fn_pack_gpu(model, v, p, dv);
}

/* compute the slope vector for the package nodes	*/
void slope_fn_pack_gpu(grid_model_t *model, double *v, grid_model_vector_t *p, double *dv)
{
	int i, j;
	/* sum of the currents(power values)	*/
	double psum;
	
	/* shortcuts	*/
	package_RC_t *pk = &model->pack;
	thermal_config_t *c = &model->config;
	layer_t *l = model->layers;
	int nl = model->n_layers;
	int nr = model->rows;
	int nc = model->cols;
	int spidx, hsidx, metalidx, c4idx, subidx, solderidx, pcbidx;
	
	/* pointer to the starting address of the extra nodes	*/
	double *x = v + nl*nr*nc;

	
	if (!model->config.model_secondary) {
		spidx = nl - DEFAULT_PACK_LAYERS + LAYER_SP;
		hsidx = nl - DEFAULT_PACK_LAYERS + LAYER_SINK;
	} else {
		spidx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS + LAYER_SP;
		hsidx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS + LAYER_SINK;
		metalidx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS - SEC_CHIP_LAYERS + LAYER_METAL;
		c4idx = nl - DEFAULT_PACK_LAYERS - SEC_PACK_LAYERS - SEC_CHIP_LAYERS + LAYER_C4;
		subidx = nl - SEC_PACK_LAYERS + LAYER_SUB;
		solderidx = nl - SEC_PACK_LAYERS + LAYER_SOLDER;
		pcbidx = nl - SEC_PACK_LAYERS + LAYER_PCB;		
	}
	

	/* sink outer north/south	*/
	psum = (c->ambient - x[SINK_N])/(pk->r_hs_per + pk->r_amb_per) + 
		   (x[SINK_C_N] - x[SINK_N])/(pk->r_hs2_y + pk->r_hs);
	dv[nl*nr*nc + SINK_N] = psum / (pk->c_hs_per + pk->c_amb_per);
	psum = (c->ambient - x[SINK_S])/(pk->r_hs_per + pk->r_amb_per) + 
		   (x[SINK_C_S] - x[SINK_S])/(pk->r_hs2_y + pk->r_hs);
	dv[nl*nr*nc + SINK_S] = psum / (pk->c_hs_per + pk->c_amb_per);

	/* sink outer west/east	*/
	psum = (c->ambient - x[SINK_W])/(pk->r_hs_per + pk->r_amb_per) + 
		   (x[SINK_C_W] - x[SINK_W])/(pk->r_hs2_x + pk->r_hs);
	dv[nl*nr*nc + SINK_W] = psum / (pk->c_hs_per + pk->c_amb_per);
	psum = (c->ambient - x[SINK_E])/(pk->r_hs_per + pk->r_amb_per) + 
		   (x[SINK_C_E] - x[SINK_E])/(pk->r_hs2_x + pk->r_hs);
	dv[nl*nr*nc + SINK_E] = psum / (pk->c_hs_per + pk->c_amb_per);

	/* sink inner north/south	*/
	/* partition r_hs1_y among all the nc grid cells. edge cell has half the ry	*/
	psum = 0.0;
	for(j=0; j < nc; j++)
		psum += (A3D(v,hsidx,0,j,nl,nr,nc) - x[SINK_C_N]);
	psum /= (l[hsidx].ry / 2.0 + nc * pk->r_hs1_y);
	psum += (c->ambient - x[SINK_C_N])/(pk->r_hs_c_per_y + pk->r_amb_c_per_y) + 
			(x[SP_N] - x[SINK_C_N])/pk->r_sp_per_y +
			(x[SINK_N] - x[SINK_C_N])/(pk->r_hs2_y + pk->r_hs);
	dv[nl*nr*nc + SINK_C_N] = psum / (pk->c_hs_c_per_y + pk->c_amb_c_per_y);

	psum = 0.0;
	for(j=0; j < nc; j++)
		psum += (A3D(v,hsidx,nr-1,j,nl,nr,nc) - x[SINK_C_S]);
	psum /= (l[hsidx].ry / 2.0 + nc * pk->r_hs1_y);
	psum += (c->ambient - x[SINK_C_S])/(pk->r_hs_c_per_y + pk->r_amb_c_per_y) + 
			(x[SP_S] - x[SINK_C_S])/pk->r_sp_per_y +
			(x[SINK_S] - x[SINK_C_S])/(pk->r_hs2_y + pk->r_hs);
	dv[nl*nr*nc + SINK_C_S] = psum / (pk->c_hs_c_per_y + pk->c_amb_c_per_y);

	/* sink inner west/east	*/
	/* partition r_hs1_x among all the nr grid cells. edge cell has half the rx	*/
	psum = 0.0;
	for(i=0; i < nr; i++)
		psum += (A3D(v,hsidx,i,0,nl,nr,nc) - x[SINK_C_W]);
	psum /= (l[hsidx].rx / 2.0 + nr * pk->r_hs1_x);
	psum += (c->ambient - x[SINK_C_W])/(pk->r_hs_c_per_x + pk->r_amb_c_per_x) + 
			(x[SP_W] - x[SINK_C_W])/pk->r_sp_per_x +
			(x[SINK_W] - x[SINK_C_W])/(pk->r_hs2_x + pk->r_hs);
	dv[nl*nr*nc + SINK_C_W] = psum / (pk->c_hs_c_per_x + pk->c_amb_c_per_x);

	psum = 0.0;
	for(i=0; i < nr; i++)
		psum += (A3D(v,hsidx,i,nc-1,nl,nr,nc) - x[SINK_C_E]);
	psum /= (l[hsidx].rx / 2.0 + nr * pk->r_hs1_x);
	psum += (c->ambient - x[SINK_C_E])/(pk->r_hs_c_per_x + pk->r_amb_c_per_x) + 
			(x[SP_E] - x[SINK_C_E])/pk->r_sp_per_x +
			(x[SINK_E] - x[SINK_C_E])/(pk->r_hs2_x + pk->r_hs);
	dv[nl*nr*nc + SINK_C_E] = psum / (pk->c_hs_c_per_x + pk->c_amb_c_per_x);

	/* spreader north/south	*/
	/* partition r_sp1_y among all the nc grid cells. edge cell has half the ry	*/
	psum = 0.0;
	for(j=0; j < nc; j++)
		psum += (A3D(v,spidx,0,j,nl,nr,nc) - x[SP_N]);
	psum /= (l[spidx].ry / 2.0 + nc * pk->r_sp1_y);
	psum += (x[SINK_C_N] - x[SP_N])/pk->r_sp_per_y;
	dv[nl*nr*nc + SP_N] = psum / pk->c_sp_per_y;

	psum = 0.0;
	for(j=0; j < nc; j++)
		psum += (A3D(v,spidx,nr-1,j,nl,nr,nc) - x[SP_S]);
	psum /= (l[spidx].ry / 2.0 + nc * pk->r_sp1_y);
	psum += (x[SINK_C_S] - x[SP_S])/pk->r_sp_per_y;
	dv[nl*nr*nc + SP_S] = psum / pk->c_sp_per_y;

	/* spreader west/east	*/
	/* partition r_sp1_x among all the nr grid cells. edge cell has half the rx	*/
	psum = 0.0;
	for(i=0; i < nr; i++)
		psum += (A3D(v,spidx,i,0,nl,nr,nc) - x[SP_W]);
	psum /= (l[spidx].rx / 2.0 + nr * pk->r_sp1_x);
	psum += (x[SINK_C_W] - x[SP_W])/pk->r_sp_per_x;
	dv[nl*nr*nc + SP_W] = psum / pk->c_sp_per_x;

	psum = 0.0;
	for(i=0; i < nr; i++)
		psum += (A3D(v,spidx,i,nc-1,nl,nr,nc) - x[SP_E]);
	psum /= (l[spidx].rx / 2.0 + nr * pk->r_sp1_x);
	psum += (x[SINK_C_E] - x[SP_E])/pk->r_sp_per_x;
	dv[nl*nr*nc + SP_E] = psum / pk->c_sp_per_x;
	
	if (model->config.model_secondary) {
		/* PCB outer north/south	*/
		psum = (c->ambient - x[PCB_N])/(pk->r_amb_sec_per) + 
			   (x[PCB_C_N] - x[PCB_N])/(pk->r_pcb2_y + pk->r_pcb);
		dv[nl*nr*nc + PCB_N] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
		psum = (c->ambient - x[PCB_S])/(pk->r_amb_sec_per) + 
			   (x[PCB_C_S] - x[PCB_S])/(pk->r_pcb2_y + pk->r_pcb);
		dv[nl*nr*nc + PCB_S] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
  	
		/* PCB outer west/east	*/
		psum = (c->ambient - x[PCB_W])/(pk->r_amb_sec_per) + 
			   (x[PCB_C_W] - x[PCB_W])/(pk->r_pcb2_x + pk->r_pcb);
		dv[nl*nr*nc + PCB_W] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
		psum = (c->ambient - x[PCB_E])/(pk->r_amb_sec_per) + 
			   (x[PCB_C_E] - x[PCB_E])/(pk->r_pcb2_x + pk->r_pcb);
		dv[nl*nr*nc + PCB_E] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
  	
		/* PCB inner north/south	*/
		/* partition r_pcb1_y among all the nc grid cells. edge cell has half the ry	*/
		psum = 0.0;
		for(j=0; j < nc; j++)
			psum += (A3D(v,pcbidx,0,j,nl,nr,nc) - x[PCB_C_N]);
		psum /= (l[pcbidx].ry / 2.0 + nc * pk->r_pcb1_y);
		psum += (c->ambient - x[PCB_C_N])/(pk->r_amb_sec_c_per_y) + 
				(x[SOLDER_N] - x[PCB_C_N])/pk->r_pcb_c_per_y +
				(x[PCB_N] - x[PCB_C_N])/(pk->r_pcb2_y + pk->r_pcb);
		dv[nl*nr*nc + PCB_C_N] = psum / (pk->c_pcb_c_per_y + pk->c_amb_sec_c_per_y);
  	
		psum = 0.0;
		for(j=0; j < nc; j++)
			psum += (A3D(v,pcbidx,nr-1,j,nl,nr,nc) - x[PCB_C_S]);
		psum /= (l[pcbidx].ry / 2.0 + nc * pk->r_pcb1_y);
		psum += (c->ambient - x[PCB_C_S])/(pk->r_amb_sec_c_per_y) + 
				(x[SOLDER_S] - x[PCB_C_S])/pk->r_pcb_c_per_y +
				(x[PCB_S] - x[PCB_C_S])/(pk->r_pcb2_y + pk->r_pcb);
		dv[nl*nr*nc + PCB_C_S] = psum / (pk->c_pcb_c_per_y + pk->c_amb_sec_c_per_y);
  	
  	/* PCB inner west/east	*/
		/* partition r_pcb1_x among all the nr grid cells. edge cell has half the rx	*/
		psum = 0.0;
		for(i=0; i < nr; i++)
			psum += (A3D(v,pcbidx,i,0,nl,nr,nc) - x[PCB_C_W]);
		psum /= (l[pcbidx].rx / 2.0 + nr * pk->r_pcb1_x);
		psum += (c->ambient - x[PCB_C_W])/(pk->r_amb_sec_c_per_x) + 
				(x[SOLDER_W] - x[PCB_C_W])/pk->r_pcb_c_per_x +
				(x[PCB_W] - x[PCB_C_W])/(pk->r_pcb2_x + pk->r_pcb);
		dv[nl*nr*nc + PCB_C_W] = psum / (pk->c_pcb_c_per_x + pk->c_amb_sec_c_per_x);
  	
		psum = 0.0;
		for(i=0; i < nr; i++)
			psum += (A3D(v,pcbidx,i,nc-1,nl,nr,nc) - x[PCB_C_E]);
		psum /= (l[pcbidx].rx / 2.0 + nr * pk->r_pcb1_x);
		psum += (c->ambient - x[PCB_C_E])/(pk->r_amb_sec_c_per_x) + 
				(x[SOLDER_E] - x[PCB_C_E])/pk->r_pcb_c_per_x +
				(x[PCB_E] - x[PCB_C_E])/(pk->r_pcb2_x + pk->r_pcb);
		dv[nl*nr*nc + PCB_C_E] = psum / (pk->c_pcb_c_per_x + pk->c_amb_sec_c_per_x);
  	
		/* solder ball north/south	*/
		/* partition r_solder1_y among all the nc grid cells. edge cell has half the ry	*/
		psum = 0.0;
		for(j=0; j < nc; j++)
			psum += (A3D(v,solderidx,0,j,nl,nr,nc) - x[SOLDER_N]);
		psum /= (l[solderidx].ry / 2.0 + nc * pk->r_solder1_y);
		psum += (x[PCB_C_N] - x[SOLDER_N])/pk->r_pcb_c_per_y;
		dv[nl*nr*nc + SOLDER_N] = psum / pk->c_solder_per_y;
  	
		psum = 0.0;
		for(j=0; j < nc; j++)
			psum += (A3D(v,solderidx,nr-1,j,nl,nr,nc) - x[SOLDER_S]);
		psum /= (l[solderidx].ry / 2.0 + nc * pk->r_solder1_y);
		psum += (x[PCB_C_S] - x[SOLDER_S])/pk->r_pcb_c_per_y;
		dv[nl*nr*nc + SOLDER_S] = psum / pk->c_solder_per_y;
  	
		/* solder ball west/east	*/
		/* partition r_solder1_x among all the nr grid cells. edge cell has half the rx	*/
		psum = 0.0;
		for(i=0; i < nr; i++)
			psum += (A3D(v,solderidx,i,0,nl,nr,nc) - x[SOLDER_W]);
		psum /= (l[solderidx].rx / 2.0 + nr * pk->r_solder1_x);
		psum += (x[PCB_C_W] - x[SOLDER_W])/pk->r_pcb_c_per_x;
		dv[nl*nr*nc + SOLDER_W] = psum / pk->c_solder_per_x;
  	
		psum = 0.0;
		for(i=0; i < nr; i++)
			psum += (A3D(v,solderidx,i,nc-1,nl,nr,nc) - x[SOLDER_E]);
		psum /= (l[solderidx].rx / 2.0 + nr * pk->r_solder1_x);
		psum += (x[PCB_C_E] - x[SOLDER_E])/pk->r_pcb_c_per_x;
		dv[nl*nr*nc + SOLDER_E] = psum / pk->c_solder_per_x;
		
		/* package substrate north/south	*/
		/* partition r_sub1_y among all the nc grid cells. edge cell has half the ry	*/
		psum = 0.0;
		for(j=0; j < nc; j++)
			psum += (A3D(v,subidx,0,j,nl,nr,nc) - x[SUB_N]);
		psum /= (l[subidx].ry / 2.0 + nc * pk->r_sub1_y);
		psum += (x[SOLDER_N] - x[SUB_N])/pk->r_solder_per_y;
		dv[nl*nr*nc + SUB_N] = psum / pk->c_sub_per_y;
  	
		psum = 0.0;
		for(j=0; j < nc; j++)
			psum += (A3D(v,subidx,nr-1,j,nl,nr,nc) - x[SUB_S]);
		psum /= (l[subidx].ry / 2.0 + nc * pk->r_sub1_y);
		psum += (x[SOLDER_S] - x[SUB_S])/pk->r_solder_per_y;
		dv[nl*nr*nc + SUB_S] = psum / pk->c_sub_per_y;
  	
		/* sub ball west/east	*/
		/* partition r_sub1_x among all the nr grid cells. edge cell has half the rx	*/
		psum = 0.0;
		for(i=0; i < nr; i++)
			psum += (A3D(v,subidx,i,0,nl,nr,nc) - x[SUB_W]);
		psum /= (l[subidx].rx / 2.0 + nr * pk->r_sub1_x);
		psum += (x[SOLDER_W] - x[SUB_W])/pk->r_solder_per_x;
		dv[nl*nr*nc + SUB_W] = psum / pk->c_sub_per_x;
  	
		psum = 0.0;
		for(i=0; i < nr; i++)
			psum += (A3D(v,subidx,i,nc-1,nl,nr,nc) - x[SUB_E]);
		psum /= (l[subidx].rx / 2.0 + nr * pk->r_sub1_x);
		psum += (x[SOLDER_E] - x[SUB_E])/pk->r_solder_per_x;
		dv[nl*nr*nc + SUB_E] = psum / pk->c_sub_per_x;
	}
}


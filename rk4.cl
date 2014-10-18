#include "gpu_rk4.h"
#pragma OPENCL EXTENSION cl_amd_fp64 : enable
/*
#ifdef cl_khr_fp64
    #pragma OPENCL EXTENSION cl_khr_fp64 : enable
#elif defined(cl_amd_fp64)
    #pragma OPENCL EXTENSION cl_amd_fp64 : enable
#else
    #error "Double precision floating point not supported by OpenCL implementation."
#endif
*/

__kernel void rk4(__global float4* data, 
      __local float* local_result, __global float* group_result) {

   float sum;
   float4 input1, input2, sum_vector;
   uint global_addr, local_addr;
   
   global_addr = get_global_id(0) * 2;
   input1 = data[global_addr];
   input2 = data[global_addr+1];
   sum_vector = input1 + input2;

   local_addr = get_local_id(0);
   local_result[local_addr] = sum_vector.s0 + sum_vector.s1 + 
                              sum_vector.s2 + sum_vector.s3; 
   barrier(CLK_LOCAL_MEM_FENCE);

   if(get_local_id(0) == 0) {
      sum = 0.0f;
      for(int i=0; i<get_local_size(0); i++) {
         sum += local_result[i];
      }
      group_result[get_group_id(0)] = sum;
   }
}

/* function to access a 1-d array as a 3-d matrix	*/
// #define A3D(array,n,i,j,nl,nr,nc)		(array[(n)*(nr)*(nc) + (i)*(nc) + (j)])
#define A3D(array,n,i,j,nl,nr,nc)		(array[(nr)*mul24((int)(n), (int)(nc)) + mad24((int)(i), (int)(nc), (int)(j))])

__kernel void rk4_average(__global double *y, __global double *k1, __global double *k2, __global double *k3, __global double *k4, double h, __global double *yout, unsigned int n) {
	int id = get_global_id(0);
	int stride = get_global_size(0);
	for(int i = id; i < n; i += stride) {
		yout[i] = mad(h, (mad(2.0, k2[i], k1[i]) + mad(2.0, k3[i], k4[i])) / 6.0, y[i]);

	}
}

__kernel void rk4_average_with_maxdiff(__global double *y, __global double *k1, __global double *k2, __global double *k3, __global double *k4, double h, __global double *yout, unsigned int n, __global double *ytemp,  __local double *local_result) {
	int stride = get_global_size(0);
	int local_size = get_local_size(0);
	int local_id = get_local_id(0);
	int id = get_global_id(0);
	double private_max = 0.0;
	for(int i = id; i < n; i += stride) {
		double yout_value = mad(h, (mad(2.0, k2[i], k1[i]) + mad(2.0, k3[i], k4[i])) / 6.0, y[i]);
		private_max = max(private_max, fabs(ytemp[i] - yout_value));
	}
	local_result[local_id] = private_max;
	barrier(CLK_LOCAL_MEM_FENCE);
	// TODO: unroll this loop
	for (int i = local_size >> 1; i > 0; i >>= 1) {
		if (local_id < i) {
			local_result[local_id] = max(local_result[local_id], local_result[local_id + i]);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	yout[get_group_id(0)] = local_result[0];
}

__kernel void max_reduce(__global double *y, unsigned int n, __local double *local_result) {
	int stride = get_global_size(0);
	int local_size = get_local_size(0);
	int local_id = get_local_id(0);
	int id = get_global_id(0);
	double private_max = 0.0;
	for(int i = id; i < n; i += stride) {
		private_max = max(private_max, y[i]);
	}
	local_result[local_id] = private_max;
	barrier(CLK_LOCAL_MEM_FENCE);
	// TODO: unroll this loop
	for (int i = local_size>>1; i > 0; i >>= 1) {
		if (local_id < i) {
			local_result[local_id] = max(local_result[local_id], local_result[local_id + i]);
		}
	}
	barrier(CLK_LOCAL_MEM_FENCE);
	y[get_group_id(0)] = local_result[0];
}

//finds sum of a row minus subVal
void sum_row(__global double *v, int nl, int nr, int nc, __local double *local_result, int n, int i, double sub_val) {
	uint local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	uint threads_per_group = mul24(get_local_size(0), get_local_size(1));
	double private_sum = 0.0;
	for(int j = local_id; j < nc; j += threads_per_group)
		private_sum += (A3D(v,n,i,j,nl,nr,nc) - sub_val);
	local_result[local_id] = private_sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
		if(local_id < stride) {
			local_result[local_id] += local_result[local_id + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

void sum_row_with_endpoint(int nl, int nr, int nc, __local double *local_result, int n, int i, double sub_val, double h, __global double *k, __global double *y) {
	uint local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	uint threads_per_group = mul24(get_local_size(0), get_local_size(1));
	double private_sum = 0.0;
	for(int j = local_id; j < nc; j += threads_per_group)
		private_sum += (mad(h, A3D(k,n,i,j,nl,nr,nc), A3D(y,n,i,j,nl,nr,nc)) - sub_val);
	local_result[local_id] = private_sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
		if(local_id < stride) {
			local_result[local_id] += local_result[local_id + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

//finds sum of a column minus subVal
void sum_col(__global double *v, int nl, int nr, int nc, __local double *local_result, int n, int j, double sub_val) {
	uint local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	uint threads_per_group = mul24(get_local_size(0), get_local_size(1));
	double private_sum = 0.0;
	for(int i = local_id; i < nr; i += threads_per_group)
		private_sum += (A3D(v,n,i,j,nl,nr,nc) - sub_val);
	local_result[local_id] = private_sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
		if(local_id < stride) {
			local_result[local_id] += local_result[local_id + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

void sum_col_with_endpoint(int nl, int nr, int nc, __local double *local_result, int n, int j, double sub_val, double h, __global double *k, __global double *y) {
	uint local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	uint threads_per_group = mul24(get_local_size(0), get_local_size(1));
	double private_sum = 0.0;
	for(int i = local_id; i < nr; i += threads_per_group)
		private_sum += (mad(h, A3D(k,n,i,j,nl,nr,nc), A3D(y,n,i,j,nl,nr,nc)) - sub_val);
	local_result[local_id] = private_sum;
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
		if(local_id < stride) {
			local_result[local_id] += local_result[local_id + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void slope_fn_pack_gpu(__constant gpu_grid_model_t *model __attribute__((max_constant_size(sizeof(gpu_grid_model_t)))), __constant gpu_layer_t *l __attribute__((max_constant_size(MAX_LAYER_SUPPORT*sizeof(gpu_layer_t)))), __global double *v, __global double *dv, unsigned int nl, unsigned int nr, unsigned int nc, __local double *local_result, __local double *x, double h, __global double *k, __global double *y)
{

	/* sum of the currents(power values)	*/
	double psum;
	
	/* shortcuts	*/
	__constant gpu_package_RC_t *pk = &model->pack;
	double ambient = model->config.ambient;
	/* l is not used */
	// layer_t *l = model->layers;
	bool model_secondary = model->config.model_secondary;
	/* Now nl, nr, nc come from parameters */
	/*
	int nl = model->n_layers;
	int nr = model->rows;
	int nc = model->cols;
	*/
	unsigned int nl_nr_nc_product = mul24(nl, nr) * nc;
	int spidx, hsidx, metalidx, c4idx, subidx, solderidx, pcbidx;
	
	/* pointer to the starting address of the extra nodes (now passed as an argument)	*/
	// __global double *x = v + nl_nr_nc_product;

	unsigned int block_id = mad24(get_group_id(1), get_num_groups(0), get_group_id(0));
	unsigned int thread_id = mad24(get_global_id(1), get_global_size(0), get_global_id(0));
	unsigned int local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	unsigned int num_blocks_mask = (0x1u << (31 - clz(mul24(get_num_groups(0), get_num_groups(1))))) - 0x1u;
	unsigned int group_job_id = 1;

	/* Do we need to calculate endpoint instead of reading it directly? */
	bool do_endpoint = (h != 0.0);

	
	if (!model_secondary) {
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
	
	/* First block: sink outer computation */
	if (thread_id == 0)
	{
		/* sink outer north/south	*/
		psum = (ambient - x[SINK_N])/(pk->r_hs_per + pk->r_amb_per) + 
			   (x[SINK_C_N] - x[SINK_N])/(pk->r_hs2_y + pk->r_hs);
		dv[nl_nr_nc_product + SINK_N] = psum / (pk->c_hs_per + pk->c_amb_per);
	
		psum = (ambient - x[SINK_S])/(pk->r_hs_per + pk->r_amb_per) + 
			   (x[SINK_C_S] - x[SINK_S])/(pk->r_hs2_y + pk->r_hs);
		dv[nl_nr_nc_product + SINK_S] = psum / (pk->c_hs_per + pk->c_amb_per);
	
		/* sink outer west/east	*/
		psum = (ambient - x[SINK_W])/(pk->r_hs_per + pk->r_amb_per) + 
			   (x[SINK_C_W] - x[SINK_W])/(pk->r_hs2_x + pk->r_hs);
		dv[nl_nr_nc_product + SINK_W] = psum / (pk->c_hs_per + pk->c_amb_per);
	
		psum = (ambient - x[SINK_E])/(pk->r_hs_per + pk->r_amb_per) + 
			   (x[SINK_C_E] - x[SINK_E])/(pk->r_hs2_x + pk->r_hs);
		dv[nl_nr_nc_product + SINK_E] = psum / (pk->c_hs_per + pk->c_amb_per);
	}
	
	/* sink inner north/south	*/
	/* partition r_hs1_y among all the nc grid cells. edge cell has half the ry	*/
	if (block_id == group_job_id)
	{
		if (do_endpoint)		
			sum_row_with_endpoint(nl, nr, nc, local_result, hsidx, 0, x[SINK_C_N], h, k, y);
		else
			sum_row(v, nl, nr, nc, local_result, hsidx, 0, x[SINK_C_N]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[hsidx].ry / 2.0 + nc * pk->r_hs1_y);
			psum += (ambient - x[SINK_C_N])/(pk->r_hs_c_per_y + pk->r_amb_c_per_y) + 
					(x[SP_N] - x[SINK_C_N])/pk->r_sp_per_y +
					(x[SINK_N] - x[SINK_C_N])/(pk->r_hs2_y + pk->r_hs);
			dv[nl_nr_nc_product + SINK_C_N] = psum / (pk->c_hs_c_per_y + pk->c_amb_c_per_y);
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;
	
	if (block_id == group_job_id)
	{
		if (do_endpoint)	
			sum_row_with_endpoint(nl, nr, nc, local_result, hsidx, nr-1, x[SINK_C_S], h, k, y);
		else
			sum_row(v, nl, nr, nc, local_result, hsidx, nr-1, x[SINK_C_S]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[hsidx].ry / 2.0 + nc * pk->r_hs1_y);
			psum += (ambient - x[SINK_C_S])/(pk->r_hs_c_per_y + pk->r_amb_c_per_y) + 
					(x[SP_S] - x[SINK_C_S])/pk->r_sp_per_y +
					(x[SINK_S] - x[SINK_C_S])/(pk->r_hs2_y + pk->r_hs);
			dv[nl_nr_nc_product + SINK_C_S] = psum / (pk->c_hs_c_per_y + pk->c_amb_c_per_y);
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;

	/* sink inner west/east	*/
	/* partition r_hs1_x among all the nr grid cells. edge cell has half the rx	*/
	if (block_id == group_job_id)
	{
		if (do_endpoint)
			sum_col_with_endpoint(nl, nr, nc, local_result, hsidx, 0, x[SINK_C_W], h, k, y);
		else
			sum_col(v, nl, nr, nc, local_result, hsidx, 0, x[SINK_C_W]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[hsidx].rx / 2.0 + nr * pk->r_hs1_x);
			psum += (ambient - x[SINK_C_W])/(pk->r_hs_c_per_x + pk->r_amb_c_per_x) + 
					(x[SP_W] - x[SINK_C_W])/pk->r_sp_per_x +
					(x[SINK_W] - x[SINK_C_W])/(pk->r_hs2_x + pk->r_hs);
			dv[nl_nr_nc_product + SINK_C_W] = psum / (pk->c_hs_c_per_x + pk->c_amb_c_per_x);
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;

	if (block_id == group_job_id)
	{
		if (do_endpoint)
			sum_col_with_endpoint(nl, nr, nc, local_result, hsidx, nc-1, x[SINK_C_E], h, k, y);
		else
			sum_col(v, nl, nr, nc, local_result, hsidx, nc-1, x[SINK_C_E]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[hsidx].rx / 2.0 + nr * pk->r_hs1_x);
			psum += (ambient - x[SINK_C_E])/(pk->r_hs_c_per_x + pk->r_amb_c_per_x) + 
					(x[SP_E] - x[SINK_C_E])/pk->r_sp_per_x +
					(x[SINK_E] - x[SINK_C_E])/(pk->r_hs2_x + pk->r_hs);
			dv[nl_nr_nc_product + SINK_C_E] = psum / (pk->c_hs_c_per_x + pk->c_amb_c_per_x);
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;

	/* spreader north/south	*/
	/* partition r_sp1_y among all the nc grid cells. edge cell has half the ry	*/
	if (block_id == group_job_id)
	{
		if (do_endpoint)
			sum_row_with_endpoint(nl, nr, nc, local_result, spidx, 0, x[SP_N], h, k, y);
		else		
			sum_row(v, nl, nr, nc, local_result, spidx, 0, x[SP_N]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[spidx].ry / 2.0 + nc * pk->r_sp1_y);
			psum += (x[SINK_C_N] - x[SP_N])/pk->r_sp_per_y;
			dv[nl_nr_nc_product + SP_N] = psum / pk->c_sp_per_y;
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;

	if (block_id == group_job_id)
	{
		if (do_endpoint)
			sum_row_with_endpoint(nl, nr, nc, local_result, spidx, nr-1, x[SP_S], h, k, y);
		else
			sum_row(v, nl, nr, nc, local_result, spidx, nr-1, x[SP_S]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[spidx].ry / 2.0 + nc * pk->r_sp1_y);
			psum += (x[SINK_C_S] - x[SP_S])/pk->r_sp_per_y;
			dv[nl_nr_nc_product + SP_S] = psum / pk->c_sp_per_y;
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;

	/* spreader west/east	*/
	/* partition r_sp1_x among all the nr grid cells. edge cell has half the rx	*/
	if (block_id == group_job_id)
	{
		if (do_endpoint)
			sum_col_with_endpoint(nl, nr, nc, local_result, spidx, 0, x[SP_W], h, k, y);
		else
			sum_col(v, nl, nr, nc, local_result, spidx, 0, x[SP_W]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[spidx].rx / 2.0 + nr * pk->r_sp1_x);
			psum += (x[SINK_C_W] - x[SP_W])/pk->r_sp_per_x;
			dv[nl_nr_nc_product + SP_W] = psum / pk->c_sp_per_x;
		}
	}
	group_job_id = (group_job_id + 1) & num_blocks_mask;

	if (block_id == group_job_id)
	{
		if (do_endpoint)
			sum_col_with_endpoint(nl, nr, nc, local_result, spidx, nc-1, x[SP_E], h, k, y);
		else
			sum_col(v, nl, nr, nc, local_result, spidx, nc-1, x[SP_E]);
		if (local_id == 0)
		{
			psum = local_result[0];
			psum /= (l[spidx].rx / 2.0 + nr * pk->r_sp1_x);
			psum += (x[SINK_C_E] - x[SP_E])/pk->r_sp_per_x;
			dv[nl_nr_nc_product + SP_E] = psum / pk->c_sp_per_x;
		}
	}
	
	if (model_secondary) {
		group_job_id = (group_job_id + 1) & num_blocks_mask;
		
		/* PCB outer north/south	*/
		bool is_local_id_zero = (block_id == group_job_id) && (local_id == 0);
		if (is_local_id_zero)
		{
			psum = (ambient - x[PCB_N])/(pk->r_amb_sec_per) + 
				   (x[PCB_C_N] - x[PCB_N])/(pk->r_pcb2_y + pk->r_pcb);
			dv[nl_nr_nc_product + PCB_N] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
			psum = (ambient - x[PCB_S])/(pk->r_amb_sec_per) + 
				   (x[PCB_C_S] - x[PCB_S])/(pk->r_pcb2_y + pk->r_pcb);
			dv[nl_nr_nc_product + PCB_S] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
  	
			/* PCB outer west/east	*/
			psum = (ambient - x[PCB_W])/(pk->r_amb_sec_per) + 
				   (x[PCB_C_W] - x[PCB_W])/(pk->r_pcb2_x + pk->r_pcb);
			dv[nl_nr_nc_product + PCB_W] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
			psum = (ambient - x[PCB_E])/(pk->r_amb_sec_per) + 
				   (x[PCB_C_E] - x[PCB_E])/(pk->r_pcb2_x + pk->r_pcb);
			dv[nl_nr_nc_product + PCB_E] = psum / (pk->c_pcb_per + pk->c_amb_sec_per);
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  	
		/* PCB inner north/south	*/
		/* partition r_pcb1_y among all the nc grid cells. edge cell has half the ry	*/
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_row_with_endpoint(nl, nr, nc, local_result, pcbidx, 0, x[PCB_C_N], h, k, y);
			else
				sum_row(v, nl, nr, nc, local_result, pcbidx, 0, x[PCB_C_N]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[pcbidx].ry / 2.0 + nc * pk->r_pcb1_y);
				psum += (ambient - x[PCB_C_N])/(pk->r_amb_sec_c_per_y) + 
						(x[SOLDER_N] - x[PCB_C_N])/pk->r_pcb_c_per_y +
						(x[PCB_N] - x[PCB_C_N])/(pk->r_pcb2_y + pk->r_pcb);
				dv[nl_nr_nc_product + PCB_C_N] = psum / (pk->c_pcb_c_per_y + pk->c_amb_sec_c_per_y);
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  		
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_row_with_endpoint(nl, nr, nc, local_result, pcbidx, nr-1, x[PCB_C_S], h, k, y);
			else
				sum_row(v, nl, nr, nc, local_result, pcbidx, nr-1, x[PCB_C_S]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[pcbidx].ry / 2.0 + nc * pk->r_pcb1_y);
				psum += (ambient - x[PCB_C_S])/(pk->r_amb_sec_c_per_y) + 
						(x[SOLDER_S] - x[PCB_C_S])/pk->r_pcb_c_per_y +
						(x[PCB_S] - x[PCB_C_S])/(pk->r_pcb2_y + pk->r_pcb);
				dv[nl_nr_nc_product + PCB_C_S] = psum / (pk->c_pcb_c_per_y + pk->c_amb_sec_c_per_y);
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  	
  		/* PCB inner west/east	*/
		/* partition r_pcb1_x among all the nr grid cells. edge cell has half the rx	*/
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_col_with_endpoint(nl, nr, nc, local_result, pcbidx, 0, x[PCB_C_W], h, k, y);
			else
				sum_col(v, nl, nr, nc, local_result, pcbidx, 0, x[PCB_C_W]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[pcbidx].rx / 2.0 + nr * pk->r_pcb1_x);
				psum += (ambient - x[PCB_C_W])/(pk->r_amb_sec_c_per_x) + 
						(x[SOLDER_W] - x[PCB_C_W])/pk->r_pcb_c_per_x +
						(x[PCB_W] - x[PCB_C_W])/(pk->r_pcb2_x + pk->r_pcb);
				dv[nl_nr_nc_product + PCB_C_W] = psum / (pk->c_pcb_c_per_x + pk->c_amb_sec_c_per_x);
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  		
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_col_with_endpoint(nl, nr, nc, local_result, pcbidx, nc-1, x[PCB_C_E], h, k, y);
			else
				sum_col(v, nl, nr, nc, local_result, pcbidx, nc-1, x[PCB_C_E]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[pcbidx].rx / 2.0 + nr * pk->r_pcb1_x);
				psum += (ambient - x[PCB_C_E])/(pk->r_amb_sec_c_per_x) + 
						(x[SOLDER_E] - x[PCB_C_E])/pk->r_pcb_c_per_x +
						(x[PCB_E] - x[PCB_C_E])/(pk->r_pcb2_x + pk->r_pcb);
				dv[nl_nr_nc_product + PCB_C_E] = psum / (pk->c_pcb_c_per_x + pk->c_amb_sec_c_per_x);
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  	
		/* solder ball north/south	*/
		/* partition r_solder1_y among all the nc grid cells. edge cell has half the ry	*/
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_row_with_endpoint(nl, nr, nc, local_result, solderidx, 0, x[SOLDER_N], h, k, y);
			else
				sum_row(v, nl, nr, nc, local_result, solderidx, 0, x[SOLDER_N]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[solderidx].ry / 2.0 + nc * pk->r_solder1_y);
				psum += (x[PCB_C_N] - x[SOLDER_N])/pk->r_pcb_c_per_y;
				dv[nl_nr_nc_product + SOLDER_N] = psum / pk->c_solder_per_y;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  		
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_row_with_endpoint(nl, nr, nc, local_result, solderidx, nr-1, x[SOLDER_S], h, k, y);
			else
				sum_row(v, nl, nr, nc, local_result, solderidx, nr-1, x[SOLDER_S]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[solderidx].ry / 2.0 + nc * pk->r_solder1_y);
				psum += (x[PCB_C_S] - x[SOLDER_S])/pk->r_pcb_c_per_y;
				dv[nl_nr_nc_product + SOLDER_S] = psum / pk->c_solder_per_y;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  	
		/* solder ball west/east	*/
		/* partition r_solder1_x among all the nr grid cells. edge cell has half the rx	*/
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_col_with_endpoint(nl, nr, nc, local_result, solderidx, 0, x[SOLDER_W], h, k, y);
			else
				sum_col(v, nl, nr, nc, local_result, solderidx, 0, x[SOLDER_W]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[solderidx].rx / 2.0 + nr * pk->r_solder1_x);
				psum += (x[PCB_C_W] - x[SOLDER_W])/pk->r_pcb_c_per_x;
				dv[nl_nr_nc_product + SOLDER_W] = psum / pk->c_solder_per_x;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  		
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_col_with_endpoint(nl, nr, nc, local_result, solderidx, nc-1, x[SOLDER_E], h, k, y);
			else
				sum_col(v, nl, nr, nc, local_result, solderidx, nc-1, x[SOLDER_E]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[solderidx].rx / 2.0 + nr * pk->r_solder1_x);
				psum += (x[PCB_C_E] - x[SOLDER_E])/pk->r_pcb_c_per_x;
				dv[nl_nr_nc_product + SOLDER_E] = psum / pk->c_solder_per_x;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
		
		/* package substrate north/south	*/
		/* partition r_sub1_y among all the nc grid cells. edge cell has half the ry	*/
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_row_with_endpoint(nl, nr, nc, local_result, subidx, 0, x[SUB_N], h, k, y);
			else
				sum_row(v, nl, nr, nc, local_result, subidx, 0, x[SUB_N]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[subidx].ry / 2.0 + nc * pk->r_sub1_y);
				psum += (x[SOLDER_N] - x[SUB_N])/pk->r_solder_per_y;
				dv[nl_nr_nc_product + SUB_N] = psum / pk->c_sub_per_y;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  		
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_row_with_endpoint(nl, nr, nc, local_result, subidx, nr-1, x[SOLDER_S], h, k, y);
			else
				sum_row(v, nl, nr, nc, local_result, subidx, nr-1, x[SOLDER_S]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[subidx].ry / 2.0 + nc * pk->r_sub1_y);
				psum += (x[SOLDER_S] - x[SUB_S])/pk->r_solder_per_y;
				dv[nl_nr_nc_product + SUB_S] = psum / pk->c_sub_per_y;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  	
		/* sub ball west/east	*/
		/* partition r_sub1_x among all the nr grid cells. edge cell has half the rx	*/
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_col_with_endpoint(nl, nr, nc, local_result, subidx, 0, x[SUB_W], h, k, y);
			else
				sum_col(v, nl, nr, nc, local_result, subidx, 0, x[SUB_W]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[subidx].rx / 2.0 + nr * pk->r_sub1_x);
				psum += (x[SOLDER_W] - x[SUB_W])/pk->r_solder_per_x;
				dv[nl_nr_nc_product + SUB_W] = psum / pk->c_sub_per_x;
			}
		}
		group_job_id = (group_job_id + 1) & num_blocks_mask;
  		
		if (block_id == group_job_id)
		{
			if (do_endpoint)
				sum_col_with_endpoint(nl, nr, nc, local_result, subidx, nc-1, x[SUB_E], h, k, y);
			else
				sum_col(v, nl, nr, nc, local_result, subidx, nc-1, x[SUB_E]);
			if (local_id == 0)
			{
				psum = local_result[0];
				psum /= (l[subidx].rx / 2.0 + nr * pk->r_sub1_x);
				psum += (x[SOLDER_E] - x[SUB_E])/pk->r_solder_per_x;
				dv[nl_nr_nc_product + SUB_E] = psum / pk->c_sub_per_x;
			}
		}
	}
}

/* macros for calculating currents(power values)	*/
/* current(power) from the next cell north. zero if on northern boundary	*/
# define NP(l,v,n,i,j,nl,nr,nc)		((i > 0) ? ((A3D(v,n,i-1,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].ry) : 0.0)
/* current(power) from the next cell south. zero if on southern boundary	*/
# define SP(l,v,n,i,j,nl,nr,nc)		((i < nr-1) ? ((A3D(v,n,i+1,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].ry) : 0.0)
/* current(power) from the next cell east. zero if on eastern boundary	*/
# define EP(l,v,n,i,j,nl,nr,nc)		((j < nc-1) ? ((A3D(v,n,i,j+1,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rx) : 0.0)
/* current(power) from the next cell west. zero if on western boundary	*/
# define WP(l,v,n,i,j,nl,nr,nc)		((j > 0) ? ((A3D(v,n,i,j-1,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rx) : 0.0)
/* current(power) from the next cell below. zero if on bottom face (with LAYERS_MASKulo)		*/
# define BP(l,v,n,i,j,nl,nr,nc)		((n < nl-1) ? ((A3D(v,n+1,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n].rz) : 0.0)
/* current(power) from the next cell above. zero if on top face	(with LAYERS_MASKulo)		*/
# define AP(l,v,n,i,j,nl,nr,nc)		((n > 0) ? ((A3D(v,n-1,i,j,nl,nr,nc)-A3D(v,n,i,j,nl,nr,nc))/l[n-1].rz) : 0.0)

/* Macros for reading from shared memory */
#define LAYERS_IN_LOCAL_MEM 4
#define LAYERS_MASK ((LAYERS_IN_LOCAL_MEM) - 1)
/* current(power) from the next cell north. zero if on northern boundary	*/
# define NP_s(l,v,n,i,j,nl,nr,nc)		((A3D(v,(n)&(LAYERS_MASK),i-1,j,nl,nr,nc)-A3D(v,(n)&(LAYERS_MASK),i,j,nl,nr,nc))/l[n].ry)
/* current(power) from the next cell south. zero if on southern boundary	*/
# define SP_s(l,v,n,i,j,nl,nr,nc)		((A3D(v,(n)&(LAYERS_MASK),i+1,j,nl,nr,nc)-A3D(v,(n)&(LAYERS_MASK),i,j,nl,nr,nc))/l[n].ry)
/* current(power) from the next cell east. zero if on eastern boundary	*/
# define EP_s(l,v,n,i,j,nl,nr,nc)		((A3D(v,(n)&(LAYERS_MASK),i,j+1,nl,nr,nc)-A3D(v,(n)&(LAYERS_MASK),i,j,nl,nr,nc))/l[n].rx)
/* current(power) from the next cell west. zero if on western boundary	*/
# define WP_s(l,v,n,i,j,nl,nr,nc)		((A3D(v,(n)&(LAYERS_MASK),i,j-1,nl,nr,nc)-A3D(v,(n)&(LAYERS_MASK),i,j,nl,nr,nc))/l[n].rx)
/* current(power) from the next cell below. zero if on bottom face (with LAYERS_MASKulo)		*/
# define BP_s(l,v,n,i,j,nl,nr,nc)		((n < nl-1) ? ((A3D(v,(n+1)&(LAYERS_MASK),i,j,nl,nr,nc)-A3D(v,(n)&(LAYERS_MASK),i,j,nl,nr,nc))/l[n].rz) : 0.0)
/* current(power) from the next cell above. zero if on top face	(with LAYERS_MASKulo)		*/
# define AP_s(l,v,n,i,j,nl,nr,nc)		((n > 0) ? ((A3D(v,(n-1)&(LAYERS_MASK),i,j,nl,nr,nc)-A3D(v,(n)&(LAYERS_MASK),i,j,nl,nr,nc))/l[n-1].rz) : 0.0)

void load_v_to_shared(__global double *v, __local double * v_cached_layer, int n, unsigned int nl, unsigned int nr, unsigned int nc)
{
	int i = get_global_id(1); // row (row-major)
	int j = get_global_id(0); // column
	int index = mad24(get_local_id(1) + 1, get_local_size(0) + 2, get_local_id(0) + 1);
	double v_value = A3D(v,n,i,j,nl,nr,nc);
	v_cached_layer[index] = v_value; // local location: row = get_local_id(1)+1, column = get_local_id(0)+1
	/* use the first and last worker rows to retrive the first and last extra rows */
	if (get_local_id(1) == 0) {
		index = get_local_id(0) + 1;
		v_cached_layer[index] = (i > 0) ? A3D(v,n,i-1,j,nl,nr,nc) : v_value;
	}
	if (get_local_id(1) == get_local_size(1) - 1) {
		index = mad24(get_local_size(1) + 1, get_local_size(0) + 2, get_local_id(0) + 1);
		v_cached_layer[index] = (i < nr-1) ? A3D(v,n,i+1,j,nl,nr,nc) : v_value;
	}
	/* use the first and last worker columns to retrive the first and last extra columns */
	if (get_local_id(0) == 0) {
		index = mul24(get_local_id(1) + 1, get_local_size(0) + 2);
		v_cached_layer[index] = (j > 0) ? A3D(v,n,i,j-1,nl,nr,nc) : v_value;
	}
	if (get_local_id(0) == get_local_size(0) - 1) {
		index = mad24(get_local_id(1) + 1, get_local_size(0) + 2, get_local_size(0) + 1);
		v_cached_layer[index] = (j < nc-1) ? A3D(v,n,i,j+1,nl,nr,nc) : v_value;
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

/* load v = y + k * h into local memory */
void load_v_to_shared_with_endpoint(__global double *y, __global double *k, double h, __local double * v_cached_layer, int n, unsigned int nl, unsigned int nr, unsigned int nc, __global double *v, bool save_to_local, bool save_to_global)
{
	int i = get_global_id(1); // row (row-major)
	int j = get_global_id(0); // column
	double v_value = mad(h, A3D(k,n,i,j,nl,nr,nc), A3D(y,n,i,j,nl,nr,nc));
	if (save_to_local) {
		int index = mad24(get_local_id(1) + 1, get_local_size(0) + 2, get_local_id(0) + 1);
		v_cached_layer[index] = v_value; // local location: row = get_local_id(1)+1, column = get_local_id(0)+1
		/* use the first and last worker rows to retrive the first and last extra rows */
		if (get_local_id(1) == 0) {
			index = get_local_id(0) + 1;
			v_cached_layer[index] = (i > 0) ? mad(h, A3D(k,n,i-1,j,nl,nr,nc), A3D(y,n,i-1,j,nl,nr,nc)) : v_value;
		}
		if (get_local_id(1) == get_local_size(1) - 1) {
			index = mad24(get_local_size(1) + 1, get_local_size(0) + 2, get_local_id(0) + 1);
			v_cached_layer[index] = (i < nr-1) ? mad(h, A3D(k,n,i+1,j,nl,nr,nc), A3D(y,n,i+1,j,nl,nr,nc)) : v_value;
		}
		/* use the first and last worker columns to retrive the first and last extra columns */
		if (get_local_id(0) == 0) {
			index = mul24(get_local_id(1) + 1, get_local_size(0) + 2);
			v_cached_layer[index] = (j > 0) ? mad(h, A3D(k,n,i,j-1,nl,nr,nc), A3D(y,n,i,j-1,nl,nr,nc)) : v_value;
		}
		if (get_local_id(0) == get_local_size(0) - 1) {
			index = mad24(get_local_id(1) + 1, get_local_size(0) + 2, get_local_size(0) + 1);
			v_cached_layer[index] = (j < nc-1) ? mad(h, A3D(k,n,i,j+1,nl,nr,nc), A3D(y,n,i,j+1,nl,nr,nc)) : v_value;
		}
	}
	if (save_to_global) {
		A3D(v,n,i,j,nl,nr,nc) = v_value;
	}
	if (save_to_local) {
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}


void load_extra_to_shared_with_endpoint(__global double *y, __global double *k, double h, __local double * extra_cached, int n)
{
	int id = mad24(get_local_id(1), get_local_size(0), get_local_id(0)); // row (row-major)
	int stride = mul24(get_local_size(1), get_local_size(0));
	int i;
	for (i = id; i < n; i += stride) {
		extra_cached[i] = mad(h, k[i], y[i]);
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

void load_extra_to_shared(__global double *x, __local double * extra_cached, int n)
{
	int id = mad24(get_local_id(1), get_local_size(0), get_local_id(0)); // row (row-major)
	int stride = mul24(get_local_size(1), get_local_size(0));
	int i;
	for (i = id; i < n; i += stride) {
		extra_cached[i] = x[i];
	}
	barrier(CLK_LOCAL_MEM_FENCE);
}

/*  Test correctness of local memory caching */
__kernel void slope_fn_grid_gpu_test(__constant gpu_grid_model_t *model __attribute__((max_constant_size(sizeof(gpu_grid_model_t)))), __constant gpu_layer_t *l __attribute__((max_constant_size(MAX_LAYER_SUPPORT*sizeof(gpu_layer_t)))), __global double *v, __global double *dv, unsigned int nl, unsigned int nr, unsigned int nc, __local double *local_result, __global double *p_cuboid)
{
	int n;
	int i = get_global_id(1); // row (row-major)
	int j = get_global_id(0); // column
	
	/* sum of the currents(power values)	*/
	double psum;
	
	/* shortcuts for cell width(cw) and cell height(ch)	*/
	double cw = model->width / model->cols;
	double ch = model->height / model->rows;

	/* shortcuts	*/
	int spidx, hsidx, metalidx, c4idx, subidx, solderidx, pcbidx;
	bool model_secondary = model->config.model_secondary;
	double ambient = model->config.ambient;
	double s_pcb = model->config.s_pcb;
	/* pointer to the starting address of the extra nodes	*/
	// __global double *x = v + mul24(nl, nr) * nc;

	/* local memory cached v[] (4 layers maximum) */
	__local double * v_cached[4];
	n = mul24((get_local_size(0) + 2), (get_local_size(1) + 2));
	v_cached[0] = local_result;
	v_cached[1] = local_result + n;
	v_cached[2] = v_cached[1] + n;
	v_cached[3] = v_cached[2] + n;
	/* load the first 4 layers */
	for(n=0; n < min(nl, 0x4u); n++) {
		load_v_to_shared(v, v_cached[n], n, nl, nr, nc);
	}
	uint next_layer = n - 1;
	/* for local memory access */
	int i_s = get_local_id(1) + 1;
	int j_s = get_local_id(0) + 1;
	int nr_s = get_local_size(1) + 2;
	int nc_s = get_local_size(0) + 2;

	
	for(n=0; n < nl; n++) {
		bool load_next_layer = (n == next_layer) && (next_layer + 1 < nl);
		double psum = 0.0;
		int n_s = n & 0x3;
		if (load_next_layer) {
			++next_layer;
			load_v_to_shared(v, v_cached[next_layer & 0x3], next_layer, nl, nr, nc);
		}
		/* we expect psum  = 0 */
		psum = 	(NP(l,v,n,i,j,nl,nr,nc) - NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s)) + 
			(SP(l,v,n,i,j,nl,nr,nc) - SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s)) + 
			(WP(l,v,n,i,j,nl,nr,nc) - WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s)) + 
			(EP(l,v,n,i,j,nl,nr,nc) - EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s)) + 
			(AP(l,v,n,i,j,nl,nr,nc) - AP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s)) + 
			(BP(l,v,n,i,j,nl,nr,nc) - BP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s));
		// A3D(dv,n,i,j,nl,nr,nc) = A3D(v,n,i,j,nl,nr,nc) + psum;
		A3D(dv,n,i,j,nl,nr,nc) = A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s) + psum;
	}
}


/* compute the slope vector for the grid cells. the transient
 * equation is CdV + sum{(T - Ti)/Ri} = P 
 * so, slope = dV = [P + sum{(Ti-T)/Ri}]/C
 */
__kernel void slope_fn_grid_gpu(__constant gpu_grid_model_t *model __attribute__((max_constant_size(sizeof(gpu_grid_model_t)))), __constant gpu_layer_t *l __attribute__((max_constant_size(MAX_LAYER_SUPPORT*sizeof(gpu_layer_t)))), __global double *v, __global double *dv, unsigned int nl, unsigned int nr, unsigned int nc, __local double *local_result, __global double *p_cuboid, double h, __global double *k, __global double *y)
{
	int n;
	int i = get_global_id(1); // row (row-major)
	int j = get_global_id(0); // column
	
	/* sum of the currents(power values)	*/
	double psum;
	
	/* shortcuts for cell width(cw) and cell height(ch)	*/
	double cw = model->width / model->cols;
	double ch = model->height / model->rows;

	/* shortcuts	*/
	int spidx, hsidx, metalidx, c4idx, subidx, solderidx, pcbidx;
	bool model_secondary = model->config.model_secondary;
	double ambient = model->config.ambient;
	double s_pcb = model->config.s_pcb;
	/* pointer to the starting address of the extra nodes	*/
	// __global double *x = v + mul24(nl, nr) * nc;
	
	if (!model_secondary) {
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

	/* Do we need to calculate endpoint instead of reading it directly? */
	bool do_endpoint = (h != 0.0);

	/* local memory cached v[] (4 layers maximum) */
	__local double * v_cached[4];
	n = mul24((get_local_size(0) + 2), (get_local_size(1) + 2));
	v_cached[0] = local_result;
	v_cached[1] = local_result + n;
	v_cached[2] = v_cached[1] + n;
	v_cached[3] = v_cached[2] + n;
	/* local memory cached extra nodes */
	__local double * x = v_cached[3] + n;
	/* load the first 4 layers */
	for(n=0; n < min(nl, 0x4u); n++) {
		if (do_endpoint) {
			// we only need these layers to be saved to global memory when secondary model is enabled, because the cache is not large enough to hold all layers
			bool save_to_global = model_secondary && ((n == metalidx-1) || (n == subidx) || (n == c4idx));
			load_v_to_shared_with_endpoint(y, k, h, v_cached[n], n, nl, nr, nc, v, 1, save_to_global); // v = y + k * h
		}
		else {
			load_v_to_shared(v, v_cached[n], n, nl, nr, nc);
		}
	}
	bool preload_spidx = do_endpoint && model_secondary;
	if (preload_spidx) {
		// compute layer spidx and save it to global memory. This layer will be used by secondary model before it is loaded into local (and global) memory.
		load_v_to_shared_with_endpoint(y, k, h, v_cached[0], spidx, nl, nr, nc, v, 0, 1); // v = y + k * h
	}
	uint next_layer = n - 1;
	/* for local memory access */
	int i_s = get_local_id(1) + 1;
	int j_s = get_local_id(0) + 1;
	int nr_s = get_local_size(1) + 2;
	int nc_s = get_local_size(0) + 2;
	/* load extra nodes to local memory */
	if (do_endpoint) {
		load_extra_to_shared_with_endpoint(y + mul24(nl, nr) * nc, k + mul24(nl, nr) * nc, h, x, model_secondary ? (EXTRA + EXTRA_SEC) : (EXTRA));
	}
	else {
		load_extra_to_shared(v + mul24(nl, nr) * nc, x, model_secondary ? (EXTRA + EXTRA_SEC) : (EXTRA));
	}
	
	/* for each grid cell	*/
	for(n=0; n < nl; n++) {
		/* load the next layer to local memory (layer n is in buffer right now, but n+1 isn't) */
		bool load_next_layer = (n == next_layer) && (next_layer + 1 < nl);
		int n_s = n & 3;
		if (load_next_layer) {
			++next_layer;
			if (do_endpoint) {
				bool save_to_global = model_secondary && ((next_layer == metalidx-1) || (next_layer == subidx) || (next_layer == c4idx));
				load_v_to_shared_with_endpoint(y, k, h, v_cached[next_layer & 0x3], next_layer, nl, nr, nc, v, 1, save_to_global); // v + k * h
			}
			else {
				load_v_to_shared(v, v_cached[next_layer & 0x3], next_layer, nl, nr, nc);
			}
		}
		if (model_secondary) {
			/*** 
			Some of these layers require other layers that are not in cache (i.e., not the 4 adjacent layers).
			Some of them have been cached before, so simply save them to global memory when load_v_to_shared_with_endpoint() loads them.
			However, layer spidx (layer 4) must be load earily since TIM layer (layer 1) will need it before layer 4 has been cached.
			Because only the data and the current location (i_s, j_s) are used by these layers saved to global memory, there are no global synchronization issues.
			***/
			if (n==LAYER_SI) { //top silicon layer (layer 0, requires layer 2, 1)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v_cached[0],((n_s+2) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[metalidx].rz) + // metalidx
				   ((A3D(v_cached[0],((n_s+1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // n+1
			} else if (n==spidx) { //spreader layer (4, requires layer 1, 5)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v,metalidx-1,i,j,nl,nr,nc)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[metalidx-1].rz) + // must load from global memory (OK)
				   ((A3D(v_cached[0],((n_s+1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // hsidx
			} else if (n==metalidx) { //metal layer (2, requires layer 3, 0)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v_cached[0],((n_s+1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[c4idx].rz) + // c4idx
				   ((A3D(v_cached[0],((n_s-2) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // LAYER_SI
			} else if (n==metalidx-1) { // TIM layer (1, requires layer 0, 4)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v_cached[0],((n_s-1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[metalidx-2].rz) + // metalidx-2
				   ((A3D(v,spidx,i,j,nl,nr,nc)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // must load from global memory (need it early)
			} else if (n==c4idx) { //C4 layer (3, requires layer 6, 2)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v,subidx,i,j,nl,nr,nc)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[subidx].rz) + // must load from global memory (OK)
				   ((A3D(v_cached[0],((n_s-1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // metalidx
			} else if (n==subidx) { //Substrate layer (6, requires layer 7, 3)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v_cached[0],((n_s+1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[solderidx].rz) + // solderidx
				   ((A3D(v,c4idx,i,j,nl,nr,nc)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // must load from global memory (OK)
			} else if (n==pcbidx) { //PCB layer (8, requires layer 7)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v_cached[0],((n_s-1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz); // solderidx
			} else if (n==hsidx) { // heatsink layer (5, requires layer 4)
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   ((A3D(v_cached[0],((n_s-1) & 3),i_s,j_s,nl_s,nr_s,nc_s)-A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[spidx].rz); // spidx
			} else {
				/* sum the currents(power values) to cells north, south, 
			 	* east, west, above and below
			 	*/	
				psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   AP(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + BP(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s);
			}
		}
		else {
			psum = NP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + SP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   EP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + WP_s(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + 
				   AP(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s) + BP(l,v_cached[0],n,i_s,j_s,nl,nr_s,nc_s);
		}

		/* spreader core is connected to its periphery	*/
		if (n == spidx) {
			/* northern boundary - edge cell has half the ry	*/
			if (i == 0)
				psum += (x[SP_N] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_sp1_y); 
			/* southern boundary - edge cell has half the ry	*/
			if (i == nr-1)
				psum += (x[SP_S] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_sp1_y); 
			/* eastern boundary	 - edge cell has half the rx	*/
			if (j == nc-1)
				psum += (x[SP_E] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_sp1_x); 
			/* western boundary	 - edge cell has half the rx	*/
			if (j == 0)
				psum += (x[SP_W] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_sp1_x); 
		/* heatsink core is connected to its inner periphery and ambient	*/
		} else if (n == hsidx) {
			/* all nodes are connected to the ambient	*/
			psum += (ambient - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/l[n].rz;
			/* northern boundary - edge cell has half the ry	*/
			if (i == 0)
				psum += (x[SINK_C_N] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_hs1_y); 
			/* southern boundary - edge cell has half the ry	*/
			if (i == nr-1)
				psum += (x[SINK_C_S] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_hs1_y); 
			/* eastern boundary	 - edge cell has half the rx	*/
			if (j == nc-1)
				psum += (x[SINK_C_E] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_hs1_x); 
			/* western boundary	 - edge cell has half the rx	*/
			if (j == 0)
				psum += (x[SINK_C_W] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_hs1_x); 
		}	else if (n == pcbidx && model_secondary) {
			/* all nodes are connected to the ambient	*/
			psum += (ambient - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(model->config.r_convec_sec * 
						   (s_pcb * s_pcb) / (cw * ch));
			/* northern boundary - edge cell has half the ry	*/
			if (i == 0)
				psum += (x[PCB_C_N] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_pcb1_y); 
			/* southern boundary - edge cell has half the ry	*/
			if (i == nr-1)
				psum += (x[PCB_C_S] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_pcb1_y); 
			/* eastern boundary	 - edge cell has half the rx	*/
			if (j == nc-1)
				psum += (x[PCB_C_E] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_pcb1_x); 
			/* western boundary	 - edge cell has half the rx	*/
			if (j == 0)
				psum += (x[PCB_C_W] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_pcb1_x); 
		}	else if (n == subidx && model_secondary) {
			/* northern boundary - edge cell has half the ry	*/
			if (i == 0)
				psum += (x[SUB_N] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_sub1_y); 
			/* southern boundary - edge cell has half the ry	*/
			if (i == nr-1)
				psum += (x[SUB_S] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_sub1_y); 
			/* eastern boundary	 - edge cell has half the rx	*/
			if (j == nc-1)
				psum += (x[SUB_E] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_sub1_x); 
			/* western boundary	 - edge cell has half the rx	*/
			if (j == 0)
				psum += (x[SUB_W] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_sub1_x); 
		}	else if (n == solderidx && model_secondary) {
			/* northern boundary - edge cell has half the ry	*/
			if (i == 0)
				psum += (x[SOLDER_N] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_solder1_y); 
			/* southern boundary - edge cell has half the ry	*/
			if (i == nr-1)
				psum += (x[SOLDER_S] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].ry/2.0 + nc*model->pack.r_solder1_y); 
			/* eastern boundary	 - edge cell has half the rx	*/
			if (j == nc-1)
				psum += (x[SOLDER_E] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_solder1_x); 
			/* western boundary	 - edge cell has half the rx	*/
			if (j == 0)
				psum += (x[SOLDER_W] - A3D(v_cached[0],n_s,i_s,j_s,nl,nr_s,nc_s))/(l[n].rx/2.0 + nr*model->pack.r_solder1_x); 
		}

		/* update the current cell's temperature	*/	   
		A3D(dv,n,i,j,nl,nr,nc) = (A3D(p_cuboid,n,i,j,nl,nr,nc) + psum) / l[n].c;
	}
	slope_fn_pack_gpu(model, l, v, dv, nl, nr, nc, local_result, x, h, k ,y);
}




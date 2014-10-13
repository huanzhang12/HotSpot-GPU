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
#define A3D(array,n,i,j,nl,nr,nc)		(array[nr*mul24(n, nc) + mad24(i, nc, j)])

//finds sum of a row minus subVal
void sum_row(__global double *v, int nl, int nr, int nc, __local double *local_result, int n, int i, double sub_val) {
	uint local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	uint threads_per_group = mul24(get_local_size(0), get_local_size(1));
	local_result[local_id] = 0.0;
	for(int j = local_id; j < nc; j += threads_per_group)
		local_result[local_id] += (A3D(v,n,i,j,nl,nr,nc) - sub_val);
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
	local_result[local_id] = 0.0;
	for(int i = local_id; i < nr; i += threads_per_group)
		local_result[local_id] += (A3D(v,n,i,j,nl,nr,nc) - sub_val);
	barrier(CLK_LOCAL_MEM_FENCE);
	for(int stride = threads_per_group >> 1; stride > 0; stride >>= 1) {
		if(local_id < stride) {
			local_result[local_id] += local_result[local_id + stride];
		}
		barrier(CLK_LOCAL_MEM_FENCE);
	}
}

__kernel void slope_fn_pack_gpu(__constant gpu_grid_model_t *model, __constant gpu_layer_t *l, __global double *v, __global double *dv, unsigned int nl, unsigned int nr, unsigned int nc, __local double *local_result)
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
	
	/* pointer to the starting address of the extra nodes	*/
	__global double *x = v + nl_nr_nc_product;

	unsigned int block_id = mad24(get_group_id(1), get_num_groups(0), get_group_id(0));
	unsigned int thread_id = mad24(get_global_id(1), get_global_size(0), get_global_id(0));
	unsigned int local_id = mad24(get_local_id(1), get_local_size(0), get_local_id(0));
	unsigned int num_blocks_mask = (0x1u << (31 - clz(mul24(get_num_groups(0), get_num_groups(1))))) - 0x1u;
	unsigned int group_job_id = 1;

	
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




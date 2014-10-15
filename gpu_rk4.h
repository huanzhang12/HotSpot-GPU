#ifndef __GPU_RK4_H
#define __GPU_RK4_H

/* number of extra nodes due to the model:
 * 4 spreader nodes, 4 heat sink nodes under
 * the spreader (center), 4 peripheral heat
 * sink nodes (north, south, east and west)
 * and a separate node for the ambient
 */
#define EXTRA		12
/* spreader nodes	*/
#define	SP_W		0
#define	SP_E		1
#define	SP_N		2
#define	SP_S		3
/* central sink nodes (directly under the spreader) */
#define SINK_C_W	4
#define SINK_C_E	5
#define SINK_C_N	6
#define SINK_C_S	7
/* peripheral sink nodes	*/
#define	SINK_W		8
#define	SINK_E		9
#define	SINK_N		10
#define	SINK_S		11

/* secondary extra nodes */
#define EXTRA_SEC		16
/* package substrate nodes	*/
#define	SUB_W		12
#define	SUB_E		13
#define	SUB_N		14
#define	SUB_S		15
/* solder ball nodes	*/
#define	SOLDER_W		16
#define	SOLDER_E		17
#define	SOLDER_N		18
#define	SOLDER_S		19
/* central PCB nodes (directly under the solder balls) */
#define PCB_C_W	20
#define PCB_C_E	21
#define PCB_C_N	22
#define PCB_C_S	23
/* peripheral PCB nodes	*/
#define	PCB_W		24
#define	PCB_E		25
#define	PCB_N		26
#define	PCB_S		27

/* default no. of chip layers (excluding spreader
 * and sink). used when LCF file is not specified
 */
#define DEFAULT_CHIP_LAYERS	2
#define LAYER_SI			0
#define LAYER_INT			1

/* layers of secondary path with same area as die */
#define SEC_CHIP_LAYERS	2
#define LAYER_METAL 0
#define LAYER_C4	1

/* default no. of package layers	*/
#define DEFAULT_PACK_LAYERS	2
#define LAYER_SP			0
#define LAYER_SINK			1

/* additional package layers from secondary path */
#define SEC_PACK_LAYERS	3
#define LAYER_SUB	0
#define LAYER_SOLDER	1
#define LAYER_PCB	2

/* package parameters	*/
typedef struct gpu_package_RC_t_st
{
	/* lateral resistances	*/
	/* peripheral spreader nodes */
	double r_sp1_x;
	double r_sp1_y;
	/* sink's inner periphery */
	double r_hs1_x;
	double r_hs1_y;
	double r_hs2_x;
	double r_hs2_y;
	/* sink's outer periphery */
	double r_hs;
	
	/* vertical resistances */
	/* peripheral spreader nodes */
	double r_sp_per_x;
	double r_sp_per_y;
	/* sink's inner periphery	*/
	double r_hs_c_per_x;
	double r_hs_c_per_y;
	/* sink's outer periphery	*/
	double r_hs_per;

	/* vertical capacitances	*/
	/* peripheral spreader nodes */
	double c_sp_per_x;
	double c_sp_per_y;
	/* sink's inner periphery	*/
	double c_hs_c_per_x;
	double c_hs_c_per_y;
	/* sink's outer periphery	*/
	double c_hs_per;

	/* vertical R's and C's to ambient	*/
	/* sink's inner periphery	*/
	double r_amb_c_per_x;
	double c_amb_c_per_x;
	double r_amb_c_per_y;
	double c_amb_c_per_y;
	/* sink's outer periphery	*/
	double r_amb_per;
	double c_amb_per;
	
	/* secondary path R's and C's */
	
	/* lateral resistances	*/
	/* peripheral package substrate nodes */
	double r_sub1_x;
	double r_sub1_y;
	/* peripheral solder ball nodes */
	double r_solder1_x;
	double r_solder1_y;
	/* PCB's inner periphery */
	double r_pcb1_x;
	double r_pcb1_y;
	double r_pcb2_x;
	double r_pcb2_y;
	/* PCB's outer periphery */
	double r_pcb;
	
	/* vertical resistances */
	/* peripheral package substrate nodes */
	double r_sub_per_x;
	double r_sub_per_y;
	/* peripheral solder ball nodes */
	double r_solder_per_x;
	double r_solder_per_y;
	/* PCB's inner periphery	*/
	double r_pcb_c_per_x;
	double r_pcb_c_per_y;
	/* PCB's outer periphery	*/
	double r_pcb_per;

	/* vertical capacitances	*/
	/* peripheral package substrate nodes */
	double c_sub_per_x;
	double c_sub_per_y;
	/* peripheral solder ballnodes */
	double c_solder_per_x;
	double c_solder_per_y;
	/* PCB's inner periphery	*/
	double c_pcb_c_per_x;
	double c_pcb_c_per_y;
	/* PCB's outer periphery	*/
	double c_pcb_per;

	/* vertical R's and C's to ambient at PCB	*/
	/* PCB's inner periphery	*/
	double r_amb_sec_c_per_x;
	double c_amb_sec_c_per_x;
	double r_amb_sec_c_per_y;
	double c_amb_sec_c_per_y;
	/* PCB's outer periphery	*/
	double r_amb_sec_per;
	double c_amb_sec_per;
	
}gpu_package_RC_t;

/* one layer of the grid model. a 3-D chip is a stacked
 * set of layers
 */
typedef struct gpu_layer_t_st
{
	/* floorplan */
	// flp_t *flp;

	/* configuration parameters	*/
	int no;				/* serial number	*/
	int has_lateral;	/* model lateral spreading of heat?	*/
	int has_power;		/* dissipates power?	*/
	double k;			/* 1/resistivity	*/
	double k1;	/* thermal conductivity of the other material in some layers, such as C4/underfill */
	double thickness;
	double sp;			/* specific heat capacity	*/
	double sp1; /* specific heat of the other material in some layers, such as C4/underfill */

	/* extracted information	*/
	double rx, ry, rz;	/* x, y and z resistors	*/
	double rx1, ry1, rz1; /* resistors of the other material in some layers, e.g. c4/underfill*/
	double c, c1;			/* capacitance	*/

	/* block-grid map - 2-d array of block lists	*/
	// blist_t ***b2gmap;
	/* grid-block map - a 1-d array of grid lists	*/
	// glist_t *g2bmap;
}gpu_layer_t;

/* thermal model configuration for GPU (pointers and unused variables removed) */
typedef struct gpu_thermal_config_t_st
{
	/* chip specs	*/
	double t_chip;	/* chip thickness in meters	*/
	double k_chip;	/* chip thermal conductivity */
	double p_chip;	/* chip specific heat */
	double thermal_threshold;	/* temperature threshold for DTM (Kelvin)*/

	/* heat sink specs	*/
	double c_convec;	/* convection capacitance in J/K */
	double r_convec;	/* convection resistance in K/W	*/
	double s_sink;	/* heatsink side in meters	*/
	double t_sink;	/* heatsink thickness in meters	*/
	double k_sink;	/* heatsink thermal conductivity */
	double p_sink;	/* heatsink specific heat */

	/* heat spreader specs	*/
	double s_spreader;	/* spreader side in meters	*/
	double t_spreader;	/* spreader thickness in meters	*/
	double k_spreader;	/* spreader thermal conductivity */
	double p_spreader;	/* spreader specific heat */

	/* interface material specs	*/
	double t_interface;	/* interface material thickness in meters	*/
	double k_interface;	/* interface material thermal conductivity */
	double p_interface; /* interface material specific heat */
	
	/* secondary path specs */
	int model_secondary;
	double r_convec_sec;
	double c_convec_sec;
	int n_metal;
	double t_metal;
	double t_c4;
	double s_c4;
	int n_c4;
	double s_sub;
	double t_sub;
	double s_solder;
	double t_solder;
	double s_pcb;
	double t_pcb;

	/* others	*/
	double ambient;			/* ambient temperature in kelvin	*/
}gpu_thermal_config_t;

/* grid thermal model for GPU (pointers and unused variables removed) */
typedef struct gpu_grid_model_t_st
{
	/* configuration	*/
	gpu_thermal_config_t config;

	/* layer information	*/
	// layer_t *layers;
	int n_layers;

	/* grid resolution	*/
	int rows;
	int cols;
	/* dimensions	*/
	double width;
	double height;

	/* package parameters	*/
	gpu_package_RC_t pack;

	/* sum total of the functional blocks of all floorplans	*/
	int total_n_blocks;
	/* grid-to-block mapping mode	*/
	int map_mode;

	/* flags	*/
	int r_ready;	/* are the R's initialized?	*/
	int c_ready;	/* are the C's initialized?	*/
	int has_lcf;	/* LCF file specified?		*/

	/* internal state - most recently computed 
	 * steady state temperatures
	 */
	// grid_model_vector_t *last_steady;
	/* internal state - most recently computed 
	 * transient temperatures
	 */
	/* grid cell temperatures	*/
	// grid_model_vector_t *last_trans;
	/* block temperatures	*/
	// double *last_temp;

	/* to allow for resizing	*/
	int base_n_units;
}gpu_grid_model_t;

#endif


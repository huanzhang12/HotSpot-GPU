#ifndef __LEAKAGE_H_
#define __LEAKAGE_H_

typedef struct {
	double temperature;
	double coefficient;
} leakage_coeff;

leakage_coeff* load_leakage_coeff(FILE* fp, double* base_temp);
void adjust_coeffs(leakage_coeff* p_coeff, double base_temp, double scaling_coeff);
void free_leakage_coeff(leakage_coeff* p_coeff);
void dump_leakage_coeff(leakage_coeff* p_coeff);
double compute_leakage_power(leakage_coeff* p_coeff, double h, double w, double temperature);

#endif


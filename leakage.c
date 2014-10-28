#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <assert.h>

#include "util.h"

#include "leakage.h"

/* Load leakage power scaling coefficients from a file */
/* File format: */
/* base temperature (K) */
/* temperature (increasing order), power scaling coeff (per unit area) */
/* return a leakage_coeff array on heap, and the base temperature */
leakage_coeff* load_leakage_coeff(FILE* fp, double* base_temp)
{
	leakage_coeff* p_coeff;
	int first_line = TRUE;
	char line[LINE_SIZE], *src;
	int size = 1;
	double scaling_coeff;
	assert(fp != NULL);
	assert(base_temp != NULL);
	
	p_coeff = malloc(sizeof(leakage_coeff));
	
	/* skip empty lines and comments */
	while(!feof(fp)) {
		double param1, param2 = -1.0;
		fgets(line, LINE_SIZE, fp);
		/* new line not read yet */	
		if(line[strlen(line)-1] != '\n')
			fatal("leakage power coefficients file line too long\n");
		/* skip spaces, tabs, returns */
		for(src = line; src[0] && strchr(" \r\t\n", src[0]); ++src);
		/* skip empty lines and comments */
		if(src && src[0] && src[0] != '#')
		{
			printf("%s", src);
			if(!sscanf(src, "%lf,%lf", &param1, &param2))
				fatal("invalid format of temperature/coefficients\n");
			if(first_line) {
				/* first line: base temperature and scaling coefficient */
				*base_temp = param1;
				printf("Base = %f\n", *base_temp);
				scaling_coeff = param2;
				printf("Scaling factor = %f\n", scaling_coeff);
				first_line = FALSE;
			}
			else {
				/* check if temperatures are in increasing order */
				if(param1 < param2)
					fatal("temperatures must be in increasing order\n");
				if(!sscanf(src, "%lf,%lf", &param1, &param2))
					fatal("invalid format of temperature/coefficients\n");
				/* temperature, coefficient pairs */
				p_coeff[size - 1].temperature = param1;
				p_coeff[size - 1].coefficient = param2;
				size++;
				p_coeff = realloc(p_coeff, size*sizeof(leakage_coeff));
				/* save last temperature for comparison in next iteration */
				param2 = param1;
			}
		}
	}
	
	/* finally, add a -1.0 temperature as the ending */
	p_coeff[size - 1].temperature = -1.0;
	p_coeff[size - 1].coefficient = -1.0;
	
	/* normalize the coefficients to the base temperature */
	adjust_coeffs(p_coeff, *base_temp, scaling_coeff);
	
	dump_leakage_coeff(p_coeff);
	printf("Leakage coefficient loaded.\n");
	return p_coeff;
}

void adjust_coeffs(leakage_coeff* p_coeff, double base_temp, double scaling_coeff)
{
	double base_coeff; 
	
	/* assume unit area */
	/* but in fact, the area are not considered now, because we only need a relative value for each block */
	base_coeff = compute_leakage_power(p_coeff, 1, 1, base_temp);
	while(p_coeff->temperature >= 0.0f) {
		p_coeff->coefficient = p_coeff->coefficient/base_coeff * scaling_coeff;
		++p_coeff;
	}
}

void free_leakage_coeff(leakage_coeff* p_coeff)
{
	if(p_coeff != NULL)
		free(p_coeff);
}

void dump_leakage_coeff(leakage_coeff* p_coeff)
{
	while(p_coeff->temperature >= 0.0f) {
		printf("Temperature = %f, Coefficient = %f\n", p_coeff->temperature, p_coeff->coefficient);
		++p_coeff;
	}
}

/* compute the leakage power coefficient given a temperature */
double compute_leakage_power(leakage_coeff* p_coeff, double h, double w, double temperature)
{
	if(p_coeff == NULL)
		return 0.0;
	while(p_coeff->temperature >= 0.0f) {
		double prev_temp = p_coeff->temperature;
		double next_temp = (p_coeff+1)->temperature;
		if(prev_temp <= temperature && next_temp >= temperature) {
			/* linear interpolation */
			double t = (temperature - prev_temp)/(next_temp - prev_temp);
			// return w * h * (p_coeff->coefficient + ((p_coeff+1)->coefficient - p_coeff->coefficient) * t);
			// we don't consider w, h here because it is a relative value
			return (p_coeff->coefficient + ((p_coeff+1)->coefficient - p_coeff->coefficient) * t);
		}
		++p_coeff;
	}
	/* we shouldn't be here */
	dump_leakage_coeff(p_coeff);
	printf("Calculating temperature = %f\n", temperature);
	fatal("temperature out of range\n");
}



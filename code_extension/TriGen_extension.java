

/**
 * Simply paste these methods into the fitness function class (e.g. https://github.com/davgutavi/trlab-trigen/blob/master/src/main/java/fitnessfunctions/Lsl.java )
 * and adapt the source code respectively to pass tricluster, and class_vector
 */


import jdistlib.Binomial;
import org.apache.commons.math3.distribution.NormalDistribution;

import java.math.BigDecimal;
import java.util.*;



public double discriminative_power(AlgorithmIndividual individual, double desired_lift, Integer[] outcome_vector){
	Collection<Integer> genes = individual.getGenes();
	Collection<Integer> samples = individual.getSamples();
	Collection<Integer> timepoints = individual.getTimes();

	Integer[] class_vector = outcome_vector.clone();

	List<Integer> pattern_values = new ArrayList<>();
	for(Integer gene : genes){
		pattern_values.add(class_vector[gene.intValue()]);
	}

	double p_x = (double)genes.size() / (double)class_vector.length;
	List<Integer> pattern_classes = new ArrayList<Integer>(new HashSet<Integer>(pattern_values));
	List<Double> lifts = new ArrayList<>();

	for (Integer unique : pattern_classes) {
		double p_y = (double) Collections.frequency(Arrays.asList(class_vector), unique) / (double) class_vector.length;
		lifts.add(((double) Collections.frequency(pattern_values, unique) / (double) class_vector.length) / (p_y * p_x));
	}
	double max = Double.MIN_VALUE;
	int maxPos = -1;
	for (int i = 0; i < lifts.size(); i++) {
		double value = lifts.get(i);
		if (value > max) {
			max = value;
			maxPos = i;
		}
	}

	double lift = lifts.get(maxPos);
	Integer pattern_class = pattern_classes.get(maxPos);
	double p_y = (double) Collections.frequency(Arrays.asList(class_vector), pattern_class) / class_vector.length;
	double omega = Math.max(p_x+p_y-1, 1.0/class_vector.length)/(p_x*p_y);
	double v = 1.0 / Math.max(p_x, p_y);
	double std_lift;
	if((lift - omega) == 0.0) {
		std_lift = 0;
	}else if((v-omega) <= 0) {
		std_lift = 1;
	}else {
		std_lift = (lift - omega) / (v - omega);
	}
	if(lift < desired_lift)
		return 1;
	return (0.5*(1-std_lift)) + (0.5*((lift > desired_lift ? desired_lift/lift : 1.0)));

}

public double statistical_significance(AlgorithmIndividual individual){
	AlgorithmDataset dataset = (AlgorithmConfiguration.getInstance()).getData();

	int size_genes = dataset.getGenSize();
	int size_sample = dataset.getSampleSize();
	int size_times = dataset.getTimeSize();

	Collection<Integer> genes = individual.getGenes();
	Collection<Integer> samples = individual.getSamples();
	Collection<Integer> timepoints = individual.getTimes();


	DescriptiveStatistics stats = new DescriptiveStatistics();

	double[] samples_average = new double[size_sample];
	double[] samples_std = new double[size_sample];
	for(Integer sample : samples){
		for(int timepoint_i = 0; timepoint_i < size_times; timepoint_i++){
			for(int gene_i = 0; gene_i < size_genes; gene_i++){
				stats.addValue(dataset.getValue(gene_i, sample.intValue(), timepoint_i));
			}
		}
		samples_average[sample] = stats.getMean();
		samples_std[sample] = stats.getStandardDeviation();
		stats.clear();
	}

	double p = 1;
	DescriptiveStatistics secondary_stats = new DescriptiveStatistics();
	for(Integer time : timepoints){
		for(Integer sample : samples){
			for(Integer gene: genes){
				stats.addValue(dataset.getValue(gene, sample, time));
			}
			double mean = stats.getMean();
			double std = stats.getStandardDeviation();
			NormalDistribution x = new NormalDistribution(samples_average[sample], samples_std[sample]);
			double p1 = x.cumulativeProbability(mean-std);
			double p2 = x.cumulativeProbability(mean+std);
			secondary_stats.addValue(p2-p1);
			stats.clear();
		}
		p = p * secondary_stats.getMean();
		secondary_stats.clear();
	}
	BigDecimal pr = new BigDecimal(p).multiply(new BigDecimal(size_times - timepoints.size() + 1));//(cnk(size_times, timepoints.size())));

	p = pr.doubleValue();
	if (p > 1)
		p = 1;

	return Binomial.cumulative(genes.size(), size_genes, p,false,false);
}
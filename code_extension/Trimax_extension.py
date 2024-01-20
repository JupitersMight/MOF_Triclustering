
'''
    Methods used to expand the δ-Trimax Algorithm
    For additive approach use the method "modify_delta_additive(delta)", and for multiplicative approach use "modify_delta_multiplicative(delta)"
'''


def modify_delta_additive(delta):
    disc_powers = np.arange(0, 1, 0.01)
    stat_sigs = np.concatenate((np.linspace(pow(10,-50), 0.05, 5000), np.linspace(0.05, 0.9, 5000)), axis=None)
    hist_vals_add = []

    for disc_power in disc_powers:
        for stat_sig in stat_sigs:
            rescaled_stat_sig = (abs(math.log(stat_sig,10)))
            nr_parameters = 3
            w1 = (1/nr_parameters)
            w2 = (1/nr_parameters)
            w3 = (1/nr_parameters)
            add_val = (delta**w1) + ((delta * disc_power)**w2)+((delta * (1/rescaled_stat_sig))**w3)
            hist_vals_add.append(add_val)

    limit = sorted(hist_vals_add)[int(len(hist_vals_add)*0.05)]
    return limit


def modify_delta_multiplicative(delta):
    disc_powers = np.arange(0, 1, 0.01)
    stat_sigs = np.concatenate((np.linspace(pow(10,-50), 0.05, 5000), np.linspace(0.05, 0.9, 5000)), axis=None)
    hist_vals_mul = []

    for disc_power in disc_powers:
        for stat_sig in stat_sigs:
            rescaled_stat_sig = (abs(math.log(stat_sig,10)))
            nr_parameters = 3
            w1 = (1/nr_parameters)
            w2 = (1/nr_parameters)
            w3 = (1/nr_parameters)
            mul_val = (delta**w1) * ((disc_power**w2) * ((1/rescaled_stat_sig)**w3))
            hist_vals_mul.append(mul_val)

    limit = sorted(hist_vals_mul)[int(len(hist_vals_mul)*0.05)]
    return limit 

'''
    The "class_vector" parameter is a pandas.Series type that represents the outcome for each individual
    The "pattern_indexes" is a simple list containing the index of each individual belonging to the tricluster
'''
def lift(class_vector, pattern_indexes):
    pattern_values = class_vector.iloc[pattern_indexes]
    p_x = len(pattern_indexes)/len(class_vector)
    pattern_classes = list(pattern_values.unique())

    lifts = []
    for val_i in range(0, len(pattern_classes)):
        val = pattern_classes[val_i]
        p_y = len(class_vector[class_vector == val])/len(class_vector)
        lifts.append((len(pattern_values[pattern_values == val])/len(class_vector))/(p_x*p_y))

    class_with_most_lift = pattern_classes[lifts.index(max(lifts))]

    p_y = len(class_vector[class_vector == class_with_most_lift])/len(class_vector)
    lift = (len(pattern_values[pattern_values == class_with_most_lift])/len(class_vector))/(p_x*p_y)
    return lift

'''
    The "D" parameter is a numpy.array with "ndmin=3"
'''
def initialize_var_distribution(D):
    variables_dist = []
    combined_df = pd.concat(D)
    for i in range(len(combined_df.columns)):
        dist = scipy.stats.norm
        aux = combined_df[combined_df.columns[i]]
        aux = list(filter(lambda x: is_number(x), aux.tolist()))
        param = dist.fit(aux)
        variables_dist.append({
            "dist": scipy.stats.norm,
            "param": param
        })
    return variables_dist

'''
    The "tricluster" parameter describes a tricluster in the form of a disctionary:
    {
        "rows": [...]
        "columns": [...]
        "times": [...]
    }
    where "[...]" is a simple list containing the index of each individual ("rows"), variable ("columns"), and context("times").
'''
def statistical_significance(D, tricluster, variables_dist, cols, i_d_d=False):
    
    Z = len(D)
    Y = len(cols)

    filtered_dataframes = [D[i] for i in tricluster["times"]]

    p = Decimal(1.0)
    for z in range(len(filtered_dataframes)):
        sample_ps = []
        for col in tricluster["columns"]:
            pattern_observed_samples = list(filtered_dataframes[z][cols[col]].iloc[tricluster["rows"]])
            pattern_val = np.nanmean(pattern_observed_samples) #Counter(pattern_observed_samples).most_common(1)[0][0]
            pattern_std = np.nanstd(pattern_observed_samples)
            if not is_number(pattern_val):
                continue
            param = variables_dist[col]["param"]
            cdfs = variables_dist[col]["dist"].cdf([pattern_val-pattern_std, pattern_val+pattern_std], *param[:-2], loc=param[-2], scale=param[-1])
            p_y = Decimal(cdfs[1]-cdfs[0])
            sample_ps.append(p_y)
        if len(sample_ps) == 0:
            break
        p = p * mean(sample_ps)

    p = p * (len(D) - len(tricluster["times"]) + 1)
    if p > 1:
        p = Decimal(1.0)

    pvalue = Decimal(TriSig.binom(len(filtered_dataframes[0][filtered_dataframes[0].columns[0]]), p, len(tricluster["rows"])))

    if i_d_d:
        pvalue = pvalue * Decimal(comb(Y, len(tricluster["columns"])))
    if pvalue > 1:
        pvalue = 1.0

    return float(pvalue)

'''
    The "class_vector" parameter is a pandas.Series type that represents the outcome for each individual
    The "pattern_indexes" is a simple list containing the index of each individual belonging to the tricluster
'''
def discriminative_power(class_vector, pattern_indexes, desired_lift=1.2):
    pattern_values = class_vector.iloc[pattern_indexes]
    p_x = len(pattern_indexes)/len(class_vector)
    pattern_classes = list(pattern_values.unique())

    lifts = []
    for val_i in range(0, len(pattern_classes)):
        val = pattern_classes[val_i]
        p_y = len(class_vector[class_vector == val])/len(class_vector)
        lifts.append((len(pattern_values[pattern_values == val])/len(class_vector))/(p_x*p_y))

    class_with_most_lift = pattern_classes[lifts.index(max(lifts))]

    p_y = len(class_vector[class_vector == class_with_most_lift])/len(class_vector)
    omega = max(p_x+p_y-1, 1/len(class_vector))/(p_x*p_y)
    lift = (len(pattern_values[pattern_values == class_with_most_lift])/len(class_vector))/(p_x*p_y)
    v = 1 / (max(p_x,p_y))
    if (lift - omega) == 0:
        std_lift = 0
    elif (v-omega) <= 0:
        std_lift = 1
    else:
        std_lift = (lift-omega)/(v-omega)
    if lift < desired_lift:
        return 1
    disc_power = (0.5*(1-std_lift)) + (0.5*((desired_lift/lift if lift > desired_lift else 1)))
    return disc_power



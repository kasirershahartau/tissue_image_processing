import numpy as np
import pandas as pd
import scipy as sp
import matplotlib
from matplotlib import pyplot as plt
import statsmodels.api as sm
import statsmodels.formula.api as smf
from statsmodels.genmod.bayes_mixed_glm import PoissonBayesMixedGLM
from scipy.stats import norm
from scipy.interpolate import interp1d

ANDERSON_THRESHOLD = 0.05
DAGOSTINO_THRESHOLD = 0.05
SHAPIRO_THRESHOLD = 0.05
KOLMOGOROV_THRESHOLD = 0.05
F_THRESHOLD = 0.05

class TwoSampleCompare:
    def __init__(self, sample1, sample2, sample1_label='sample1', sample2_label='sample2', continues=True):
        self.sample1 = sample1
        self.sample2 = sample2
        self.sample1_label = sample1_label
        self.sample2_label = sample2_label
        self.continues = continues

    @staticmethod
    def anderson_test(sample):
        res = sp.stats.anderson(sample, dist='norm')
        pval = res.significance_level[np.argmin(np.abs(res.statistic - res.critical_values))]/100
        return pval

    @staticmethod
    def dagostino_test(sample):
        if sample.size < 8:
            return 0
        res = sp.stats.normaltest(sample, nan_policy='omit')
        return res.pvalue

    @staticmethod
    def shapiro_test(sample):
        res = sp.stats.shapiro(sample)
        return res.pvalue

    @staticmethod
    def kolmogorov_test(sample):
        res = sp.stats.ks_1samp(sample, sp.stats.norm.cdf)
        return res.pvalue

    def check_for_normality(self):
        anderson1 = self.anderson_test(self.sample1)
        # print("%s anderson test p-val: %.3f" % (self.sample1_label, anderson1))
        anderson2 = self.anderson_test(self.sample2)
        # print("%s anderson test p-val: %.3f" % (self.sample2_label, anderson2))
        anderson_passed = anderson1 > ANDERSON_THRESHOLD and anderson2 > ANDERSON_THRESHOLD
        dagostino1 = self.dagostino_test(self.sample1)
        # print("%s  dagostino test p-val: %.3f" % (self.sample1_label, dagostino1))
        dagostino2 = self.dagostino_test(self.sample2)
        # print("%s  dagostino test p-val: %.3f" % (self.sample2_label, dagostino2))
        dagostino_passed = dagostino1 > DAGOSTINO_THRESHOLD and dagostino2 > DAGOSTINO_THRESHOLD
        if self.sample1.size > 2 and self.sample2.size > 2:
            shapiro1 = self.shapiro_test(self.sample1)
            # print("%s  shapiro test p-val: %.3f" % (self.sample1_label, shapiro1))
            shapiro2 = self.shapiro_test(self.sample2)
            # print("%s  shapiro test p-val: %.3f" % (self.sample2_label, shapiro2))
            shapiro_passed = shapiro1 > SHAPIRO_THRESHOLD and shapiro2 > SHAPIRO_THRESHOLD
        else:
            shapiro_passed = False
        kolmogorov1 = self.kolmogorov_test(self.sample1)
        # print("%s  kolmogorov test p-val: %.3f" % (self.sample1_label, kolmogorov1))
        kolmogorov2 = self.kolmogorov_test(self.sample2)
        # print("%s  kolmogorov test p-val: %.3f" % (self.sample2_label, kolmogorov2))
        kolmogorov_passed = kolmogorov1 > KOLMOGOROV_THRESHOLD and kolmogorov2 > KOLMOGOROV_THRESHOLD
        return (anderson_passed or dagostino_passed) or (shapiro_passed or kolmogorov_passed)

    @staticmethod
    def f_test(x, y):
        x = np.array(x)
        y = np.array(y)
        x_var = np.var(x, ddof=1)
        y_var = np.var(y, ddof=1)
        f = max(x_var, y_var) / min(x_var, y_var)  # calculate F test statistic
        dfn = x.size - 1 if x_var > y_var else y.size  # define degrees of freedom numerator
        dfd = y.size - 1 if x_var > y_var else x.size  # define degrees of freedom denominator
        p = 1 - sp.stats.f.cdf(f, dfn, dfd)  # find p-value of F test statistic
        return f, p

    def compare_variances(self):
        stat, pval = self.f_test(self.sample1, self.sample2)
        # print("F-statistics for variances: %.3f, p-value: %.3f" % (stat, pval))
        return pval > F_THRESHOLD

    def compare_samples(self):
        if not self.check_for_normality() or not self.continues:
            statistics, pvalue = sp.stats.mannwhitneyu(self.sample1, self.sample2, nan_policy='omit')
            print("Using Mann-Whitney test: u - %.3f, p-value - %.3f" % (statistics, pvalue))
        else:
            equal_variances = self.compare_variances()
            # student's t-test if variances are equal or Welch's t-test otherwise
            test = "Student's-t" if equal_variances else "Welch's-t"
            statistics, pvalue = sp.stats.ttest_ind(self.sample1, self.sample2, equal_var=equal_variances, nan_policy='omit')
            print("Using %s test: statistics - %.3f, p-value - %.3f" % (test, statistics, pvalue))
        return pvalue

class HierarchicalTwoSamplesCompare:
    """ Comparing two samples with hierarchical structure (e.g. different cells in the same tissue for a few biological
     repeats"""

    def __init__(self, data1, data2, continues=True):
        """
        data1, data2 - DataCollector instances where each file is a biological repeat
        """
        self.data = self.rearrange_data_into_table(data1, data2)
        self.continues = continues

    @staticmethod
    def rearrange_data_into_table(data1, data2):
        df = []
        for data_index, data in enumerate([data1, data2]):
            for group in range(data.get_number_of_groups()):
                for measurement in data.get_partial_sample(group):
                    df.append({
                        'measurement': measurement,
                        'fixed_effect_label': data_index,
                        'biological_repeat': f"R{data.get_biological_repeat(group)}"
                    })
        df = pd.DataFrame(df)
        df['fixed_effect_label'] = df['fixed_effect_label'].astype('category')
        df['biological_repeat'] = df['biological_repeat'].astype('category')
        return df

    def fit_poissonian_GLMM(self):
        # Define model matrices
        endog = self.data['measurement']
        exog = sm.add_constant(self.data['fixed_effect_label'])
        exog_re = pd.get_dummies(self.data['biological_repeat'], drop_first=False)  # random effect
        ident = list(range(exog_re.shape[1]))

        # Fit Poisson GLMM
        model = PoissonBayesMixedGLM(endog, exog, exog_re, ident)
        result = model.fit_vb()
        return result

    def fit_LMM(self):
        model = smf.mixedlm("measurement ~ fixed_effect_label", self.data, groups=self.data["biological_repeat"])
        result = model.fit()
        return result

    @staticmethod
    def calc_pval_from_mean_and_sd(mean, sd):
        z_val = mean/sd
        p_val = 2 * (1 - norm.cdf(np.abs(z_val)))
        return p_val

    def compare_samples(self):
        try:
            if self.continues:
                res = self.fit_LMM()
                p_val = res.pvalues["fixed_effect_label[T.1]"]
            else:
                res = self.fit_poissonian_GLMM()
                fixed_effect_mean = res.fe_mean[1]
                fixed_effect_sd = res.fe_sd[1]
                p_val = self.calc_pval_from_mean_and_sd(fixed_effect_mean, fixed_effect_sd)
            print(res.summary())
        except np.linalg.LinAlgError:
            print("Didn't converge")
            return 1
        return p_val




def barplot_annotate_brackets(num1, num2, data, center, height, yerr=None, dh=.05, barh=.05, fs=None, maxasterix=None, ax=None):
    """
    Annotate barplot with p-values.

    :param num1: number of left bar to put bracket over
    :param num2: number of right bar to put bracket over
    :param data: string to write or number for generating asterixes
    :param center: centers of all bars (like plt.bar() input)
    :param height: heights of all bars (like plt.bar() input)
    :param yerr: yerrs of all bars (like plt.bar() input)
    :param dh: height offset over bar / bar + yerr in axes coordinates (0 to 1)
    :param barh: bar height in axes coordinates (0 to 1)
    :param fs: font size
    :param maxasterix: maximum number of asterixes to write (for very small p-values)
    """

    if type(data) is str:
        text = data
    else:
        text = str(data)
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        # text = ''
        # p = .05
        #
        # while data < p:
        #     text += '*'
        #     p /= 10.
        #
        #     if maxasterix and len(text) == maxasterix:
        #         break
        #
        # if len(text) == 0:
        #     text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr.size:
        ly += yerr[num1]
        ry += yerr[num2]

    ax_y0, ax_y1 = plt.gca().get_ylim()
    dh *= (ax_y1 - ax_y0)
    barh *= (ax_y1 - ax_y0)

    y = max(ly, ry) + dh

    barx = [lx, lx, rx, rx]
    bary = [y, y+barh, y+barh, y]
    mid = ((lx+rx)/2, y+barh)

    ax.plot(barx, bary, c='black')

    kwargs = dict(ha='center', va='bottom')
    if fs is not None:
        kwargs['fontsize'] = fs

    ax.text(*mid, text, **kwargs)
    return (y+barh)/(ax_y1 - ax_y0)

def compare_and_plot_samples(samples_list, pairs_to_compare, continues=True, plot_style="violin", color='white',
                             edge_color='grey', fig=None, ax=None, show_statistics=False, show_N=False,
                             hirarchical=False, scatter=False):

    # Statistical analysis for each pair
    pvalues = np.zeros((len(pairs_to_compare),))
    for index, pair in enumerate(pairs_to_compare):
        sample1_index, sample2_index = pair
        if hasattr(continues, "__len__"):
            continues_sample = continues[sample1_index] and continues[sample2_index]
        else:
            continues_sample = continues
        if hirarchical:
            analyzer = HierarchicalTwoSamplesCompare(samples_list[sample1_index], samples_list[sample2_index],
                                                     continues=continues_sample)
        else:
            analyzer = TwoSampleCompare(samples_list[sample1_index].get_sample(), samples_list[sample2_index].get_sample(),
                                        samples_list[sample1_index].name, samples_list[sample2_index].name,
                                        continues=continues_sample)
        pvalues[index] = analyzer.compare_samples()

    averages = np.array([sample.get_average() for sample in samples_list])
    standard_errors = np.array([sample.get_se() for sample in samples_list])
    sample_sizes = np.array([sample.get_number_of_data_points() for sample in samples_list])
    labels = [sample.name for sample in samples_list]

    # Creating box plot
    w = 0.6  # bar width
    x = np.arange(len(samples_list))  # x-coordinates of your bars
    if fig is None:
        fig, ax = plt.subplots()
    errors = False
    if not isinstance(color, list):
        color = [color for i in range(len(samples_list))]
    if not isinstance(edge_color, list):
        edge_color = [edge_color for i in range(len(samples_list))]
    if plot_style == "bar":
        ax.bar(x,
               height=averages,
               capsize=12,  # error bar cap width in points
               width=w,  # bar width
               tick_label=labels,
               color=color,  # face color transparent
               edgecolor=edge_color,
               )
        for pos, y, err, color in zip(x,averages, standard_errors, edge_color):
            ax.errorbar(pos, y, err, lw=2, capsize=15, capthick=2, color=color)
    elif plot_style == "box":
        parts = ax.boxplot([s.get_sample() for s in samples_list],
                   vert=True,
                   showmeans=True,
                   labels=labels,
                   positions=x,
                   showcaps=False,
                   patch_artist=True,
                   showfliers=True,
                   )
        for pc, c, ec in zip(parts['boxes'], color, edge_color):
            pc.set_facecolor(c)
            pc.set_edgecolor(ec)
        errors = False

    elif plot_style == "violin":
        parts = ax.violinplot([s.get_sample() for s in samples_list],
                      vert=True,
                      showmeans=False,
                      showextrema=False,
                      positions=x,
                      )
        ax.set_xticks(x)
        ax.set_xticklabels(labels)
        errors = True
        for pc, c, ec in zip(parts['bodies'], color, edge_color):
            pc.set_facecolor(c)
            pc.set_edgecolor(ec)
    elif plot_style == "histogram":
        y_lim = 0
        # for scatter if needed
        y_centers = []
        x_jitter_width = []
        for i, dataset in enumerate([s.get_sample() for s in samples_list]):
            hist, bin_edges = np.histogram(dataset, bins=(np.arange(np.max(dataset) + 2) - 0.5))
            norm_hist = hist/(1.5*np.max(hist))
            bottom = bin_edges[:-1] + 0.5
            heights = np.ones((bin_edges.size - 1))
            lefts = i - 0.5*norm_hist
            y_lim = max(y_lim, np.max(dataset))
            ax.barh(bottom, norm_hist, height=heights, left=lefts, color=color[i], edgecolor=edge_color[i], linewidth=1, alpha=0.6)
            if show_statistics:
                for j, val in enumerate(norm_hist):
                    ax.text(i, bin_edges[j] + 0.5,
                            "%f.1"%(100*(val/np.sum(norm_hist))),
                            horizontalalignment='center'
                            )
            y_centers.append(bottom)
            x_jitter_width.append(norm_hist)
        errors = True

        ax.set_xlim([-0.4, len(samples_list) - 0.6])
        ax.set_ylim([-1, y_lim+0.6])

    if show_N:
        for index in range(len(samples_list)):
            ax.text(x[index],
                    (averages[index] + standard_errors[index])*1.1,
                    "N = %d" % sample_sizes[index],
                    horizontalalignment='center'
                    )
    if scatter:
        # Adding scattered points on top of the bars
        if plot_style == "violin":

            violin_bodies = parts['bodies']
            paths = [violin_body.get_paths()[0] for violin_body in violin_bodies]
            vertices = [path.vertices for path in paths]

            for i in range(len(x)):
                v = vertices[i]
                left = v[v[:, 0] < x[i]]  # x < center
                right = v[v[:, 0] > x[i]]  # x > center
                increasing_left = np.hstack((1, np.diff(left[:, 1]))) > 1e-7
                f_left = interp1d(left[increasing_left, 1], left[increasing_left, 0], bounds_error=False, fill_value="extrapolate")
                decreasing_right = np.hstack((1, np.diff(right[:, 1]))) < -1e-7
                f_right = interp1d(np.flip(right[decreasing_right, 1]), np.flip(right[decreasing_right, 0]), bounds_error=False, fill_value="extrapolate")
                for group_idx in range(samples_list[i].get_number_of_groups()):
                    group = samples_list[i].get_partial_sample(group_idx)
                    # distribute scatter randomly across the local width of the violin plot
                    x_positions = []
                    if len(group) > 100:
                        group = np.random.choice(group, 100)
                    for y in group:
                        x_min = f_left(y)
                        x_max = f_right(y)
                        jittered_x = np.random.uniform(x_min, x_max)
                        x_positions.append(jittered_x)
                    # ax.scatter(x_positions, group, color=matplotlib.color_sequences["tab10"][group_idx], marker=".", s=10)
                    ax.scatter(x_positions, group, color="gray", marker=".", s=10)
        elif plot_style == "histogram":
            for i in range(len(x)):
                for group_idx in range(samples_list[i].get_number_of_groups()):
                    group = samples_list[i].get_partial_sample(group_idx)
                    # distribute scatter randomly across the local width of the violin plot
                    x_positions = []
                    y_positions = []
                    if len(group) > 100:
                        group = np.random.choice(group, 100)
                    for y in group:
                        w = x_jitter_width[i][np.where(y_centers[i]==y)[0]]
                        jittered_x = np.random.uniform(x[i] - w/2, x[i] + w/2)
                        jittered_y = np.random.uniform(y + 0.3, y - 0.3)
                        x_positions.append(jittered_x)
                        y_positions.append(jittered_y)
                    # ax.scatter(x_positions, y_positions, color=matplotlib.color_sequences["tab10"][group_idx], marker=".", s=10)
                    ax.scatter(x_positions, y_positions, color="gray", marker=".", s=10)
        else:
            y_jitter = 0
            for i in range(len(x)):
                for group_idx in range(samples_list[i].get_number_of_groups()):
                    group = samples_list[i].get_partial_sample(group_idx)
                    # distribute scatter randomly across whole width of bar
                    # ax.scatter(x[i] + np.random.random(group.size) * w - w / 2, group + y_jitter, color=matplotlib.color_sequences["tab10"][group_idx], marker=".", s=10)
                    ax.scatter(x[i] + np.random.random(group.size) * w - w / 2, group + y_jitter,
                               color="gray", marker=".", s=10)

    if errors:
        for i in range(len(x)):
            ax.errorbar(x[i],
                    averages[i],
                    yerr=standard_errors[i],
                    color="black",
                    marker="*",
                    markersize=10,
                    capsize=10,
                    elinewidth=2
                    )
    # Setting plot y-limits
    data_max = np.max(np.array([sample.get_max() for sample in samples_list]))
    data_min = np.min(np.array([sample.get_min() for sample in samples_list]))
    max_error = np.max(np.array([averages[i] + standard_errors[i] for i in range(len(samples_list))]))
    min_error = np.min(np.array([averages[i] - standard_errors[i] for i in range(len(samples_list))]))
    low_lim = min(min_error, data_min)
    high_lim = max(max_error, data_max)
    addition = (high_lim - low_lim) * 0.1
    # ax.set_ylim([low_lim - addition, high_lim + addition])

    # Adding p-value brackets for each compared pair
    if show_statistics:
        heights = np.zeros((len(samples_list),))
        for index, pair in enumerate(pairs_to_compare):
            sample1_index, sample2_index = pair
            dh = max(np.max(heights[sample1_index:sample2_index])/2 +0.1, 0.3)
            heights[sample1_index:sample2_index] = barplot_annotate_brackets(sample1_index, sample2_index, pvalues[index], x,
                                      averages, yerr=standard_errors, fs=10, maxasterix=4, dh=0.1, ax=ax)

    res = {"averages": averages, "standard errors":standard_errors, "sample sizes": sample_sizes,
           "pvalues": {pairs_to_compare[i]: pvalues[i] for i in range(len(pairs_to_compare))}}
    return fig, ax, res

if __name__ == "__main__":
    # Testing methods
    rng = np.random.default_rng()
    samples_list = [sp.stats.norm.rvs(loc=i, scale=(i+1), size=100, random_state=rng) for i in range(5)]
    pairs = [(0,3), (1,4)]
    labels = ['sample%d' % (i+1) for i in range(5)]
    res = compare_and_plot_samples(samples_list, labels, pairs)
    plt.show()
    print(res)




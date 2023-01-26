import numpy as np
import scipy as sp
from matplotlib import pyplot as plt

ANDERSON_THRESHOLD = 0.05
DAGOSTINO_THRESHOLD = 0.05
SHAPIRO_THRESHOLD = 0.05
KOLMOGOROV_THRESHOLD = 0.05
F_THRESHOLD = 0.05

class TwoSampleCompare:
    def __init__(self, sample1, sample2, sample1_label='sample1', sample2_label='sample2', continouse=True):
        self.sample1 = sample1
        self.sample2 = sample2
        self.sample1_label = sample1_label
        self.sample2_label = sample2_label
        self.continues = continouse

    @staticmethod
    def anderson_test(sample):
        res = sp.stats.anderson(sample, dist='norm')
        pval = res.significance_level[np.argmin(np.abs(res.statistic - res.critical_values))]/100
        return pval

    @staticmethod
    def dagostino_test(sample):
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
        print("%s anderson test p-val: %.3f" % (self.sample1_label, anderson1))
        anderson2 = self.anderson_test(self.sample2)
        print("%s anderson test p-val: %.3f" % (self.sample2_label, anderson2))
        anderson_passed = anderson1 > ANDERSON_THRESHOLD and anderson2 > ANDERSON_THRESHOLD
        dagostino1 = self.dagostino_test(self.sample1)
        print("%s  dagostino test p-val: %.3f" % (self.sample1_label, dagostino1))
        dagostino2 = self.dagostino_test(self.sample2)
        print("%s  dagostino test p-val: %.3f" % (self.sample2_label, dagostino2))
        dagostino_passed = dagostino1 > DAGOSTINO_THRESHOLD and dagostino2 > DAGOSTINO_THRESHOLD
        shapiro1 = self.shapiro_test(self.sample1)
        print("%s  shapiro test p-val: %.3f" % (self.sample1_label, shapiro1))
        shapiro2 = self.shapiro_test(self.sample2)
        print("%s  shapiro test p-val: %.3f" % (self.sample2_label, shapiro2))
        shapiro_passed = shapiro1 > SHAPIRO_THRESHOLD and shapiro2 > SHAPIRO_THRESHOLD
        kolmogorov1 = self.kolmogorov_test(self.sample1)
        print("%s  kolmogorov test p-val: %.3f" % (self.sample1_label, kolmogorov1))
        kolmogorov2 = self.kolmogorov_test(self.sample2)
        print("%s  kolmogorov test p-val: %.3f" % (self.sample2_label, kolmogorov2))
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
        print("F-statistics for variances: %.3f, p-value: %.3f" % (stat, pval))
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
        # * is p < 0.05
        # ** is p < 0.005
        # *** is p < 0.0005
        # etc.
        text = ''
        p = .05

        while data < p:
            text += '*'
            p /= 10.

            if maxasterix and len(text) == maxasterix:
                break

        if len(text) == 0:
            text = 'n. s.'

    lx, ly = center[num1], height[num1]
    rx, ry = center[num2], height[num2]

    if yerr:
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


def compare_and_plot_samples(samples_list, labels, pairs_to_compare, continues=True, plot_style="violin", color='white',
                             edge_color='grey', fig=None, ax=None):

    # Statistical analysis for each sample
    averages = [np.average(sample) for sample in samples_list]
    sample_sizes = [sample.size for sample in samples_list]
    standard_errors = [np.std(sample)/np.sqrt(sample.size) for sample in samples_list]

    # Statistical analysis for each pair
    pvalues = np.zeros((len(pairs_to_compare),))
    for index, pair in enumerate(pairs_to_compare):
        sample1_index, sample2_index = pair
        if hasattr(continues, "__len__"):
            continues_sample = continues[sample1_index] and continues[sample2_index]
        else:
            continues_sample = continues
        analyzer = TwoSampleCompare(samples_list[sample1_index], samples_list[sample2_index],
                                    labels[sample1_index], labels[sample2_index], continouse=continues_sample)
        pvalues[index] = analyzer.compare_samples()

    # Creating bar plot
    w = 0.6  # bar width
    x = np.arange(len(samples_list))  # x-coordinates of your bars
    if fig is None:
        fig, ax = plt.subplots()
    scatter = False
    errors = False
    if not isinstance(color, list):
        color = [color for i in range(len(samples_list))]
    if not isinstance(edge_color, list):
        edge_color = [edge_color for i in range(len(samples_list))]
    if plot_style == "bar":
        ax.bar(x,
               height=averages,
               yerr=standard_errors,  # error bars
               capsize=12,  # error bar cap width in points
               width=w,  # bar width
               tick_label=labels,
               color=color,  # face color transparent
               edgecolor=edge_color,
               )
        scatter = False
    elif plot_style == "box":
        ax.boxplot(samples_list,
                   vert=True,
                   showmeans=False,
                   labels=labels,
                   positions=x,
                   showcaps=False
                   )
        errors = True
    elif plot_style == "violin":
        parts = ax.violinplot(samples_list,
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
    for index in range(len(samples_list)):
        ax.text(x[index],
                (averages[index] + standard_errors[index])*1.1,
                "N = %d" % sample_sizes[index],
                horizontalalignment='center'
                )
    if scatter:
        # Adding scattered points on top of the bars
        for i in range(len(x)):
            # distribute scatter randomly across whole width of bar
            ax.scatter(x[i] + np.random.random(sample_sizes[i]) * w - w / 2, samples_list[i], color=(0., 0.5, 0.5, 0.5))
    if errors:
        ax.errorbar(x,
                    averages,
                    yerr=standard_errors,
                    fmt="ob",
                    capsize=20
                    )
    # Setting plot y-limits
    data_max = np.max(np.array([np.max(sample) for sample in samples_list]))
    data_min = np.min(np.array([np.min(sample) for sample in samples_list]))
    max_error = np.max(np.array([averages[i] + standard_errors[i] for i in range(len(samples_list))]))
    min_error = np.min(np.array([averages[i] - standard_errors[i] for i in range(len(samples_list))]))
    low_lim = min(min_error, data_min)
    high_lim = max(max_error, data_max)
    # addition = (high_lim - low_lim) * 0.1
    # ax.set_ylim([low_lim - addition, high_lim + addition])

    # Adding p-value brackets for each compared pair
    heights = np.zeros((len(samples_list),))
    for index, pair in enumerate(pairs_to_compare):
        sample1_index, sample2_index = pair
        dh = max(np.max(heights[sample1_index:sample2_index])/4 + 0.2, 0.2)
        heights[sample1_index:sample2_index] = barplot_annotate_brackets(sample1_index, sample2_index, pvalues[index], x,
                                  averages, yerr=standard_errors, fs=40, maxasterix=4, dh=dh, ax=ax)

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




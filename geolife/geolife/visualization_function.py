# visualization_function.py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

#-----------------------VISUALIZATION------------------------------
def plot_binary_target(y, x_label, y_label):
# plot distribution and scatterplot of the target
    fig, ax = plt.subplots(1,2, figsize=(15,5))
    ax[0].set_ylabel(y_label +' Density')
    ax[0].set_xticks([0.25,0.75])
    ax[0].set_xticklabels(['No ' + y_label, y_label])
    values, bins, bars = ax[0].hist(y, bins=2, rwidth=0.8, edgecolor='black')
    # compute percentage for both categories
    percentages = y.value_counts().sort_index()/y.count()*100
    ax[0].bar_label(bars, labels=[f'{percentage:.1f}%' for percentage in percentages], fontsize=10, color='navy')
    ax[0].set_title("Distribution")

    ax[1].set_ylabel(y_label)
    ax[1].set_xlabel(x_label)
    ax[1].set_yticks([0,1])
    ax[1].set_yticklabels(['No ' + y_label, y_label])
    ax[1].scatter(y.index,y)
    ax[1].set_title("Scatterplot")

    plt.show()


def plot_yearly_monthly(train, y, name):
    # plot yearly and monthly averages of the target or a features

    # temporary frame to contain year, month and ignition
    proportion = y.to_frame(name='target')
    proportion['year'] = train['year']
    proportion['month'] = train['month']

    # compute proportions of ignitions for each month and years
    proportion_by_month = proportion.groupby(["year","month"])['target'].mean()
    proportion_by_year = proportion.groupby(['year'])['target'].mean()
    del(proportion)

    # Linear fit of ignition proportion over the years
    coeffs = np.polyfit(proportion_by_year.index.values, proportion_by_year.values, deg=1)
    trend_line = np.poly1d(coeffs)

    fig, ax = plt.subplots(1,2, figsize=(15,5))
    # Year trend plot
    ax[0].plot(proportion_by_year.index.values, proportion_by_year, 'o-')
    ax[0].plot(proportion_by_year.index.values, trend_line(proportion_by_year.index.values), 'r--', label='Linear Trend')
    ax[0].set_title('Proportion of ' + name +' by Year')
    ax[0].set_xlabel('Year')
    ax[0].set_ylabel('Proportion of ' + name)
    ax[0].legend()
    ax[0].grid(True)
    # Month seasonality plot
    proportion_by_month.unstack(level=0).plot(ax=ax[1], marker='o')
    ax[1].set_title('Proportion of ' + name +' by Month')
    ax[1].set_xlabel('Month')
    ax[1].set_ylabel('Proportion of ' + name)
    ax[1].set_xticks(np.arange(1,13))
    ax[1].grid(True)
    ax[1].legend(title='Year', bbox_to_anchor=(1.05, 1))

    plt.show()

def distribution_stats_plot(X):
    # check distributions of features & basic stats
    columns_numerical = X.select_dtypes(include=("float64")).columns
    n_cols = 3
    n_rows = int(np.ceil(len(columns_numerical) / n_cols))

    # histogram for all features
    fig, axes = plt.subplots(nrows=n_rows, ncols=n_cols, figsize=(11, n_rows * 3+1))
    axes = axes.flatten()

    for i, column in enumerate(columns_numerical):
        ax = axes[i]
        col_data = X[column]
        ax.hist(col_data, bins=20, edgecolor='black')
        ax.set_title(column)
        ax.set_ylabel('Frequency')

    # add a basic stats summary from describe to the plots
        stats = col_data.describe().to_string()
        ax.text(0.95, 0.95, stats, transform=ax.transAxes, fontsize=8,
                verticalalignment='top', horizontalalignment='right')

    plt.tight_layout()
    plt.show()

def correlation_matrix(X):
    # check correlations
    cmap = sns.diverging_palette(250, 10, as_cmap=True)  
    columns_numerical = X.select_dtypes(include=("float64")).columns
    corr_matrix = X[columns_numerical].corr()
    plt.figure(figsize=(12,11))
    sns.heatmap(corr_matrix, cmap=cmap, annot=True)
    plt.show()

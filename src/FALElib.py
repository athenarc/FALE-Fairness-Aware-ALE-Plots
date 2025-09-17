import pandas as pd
import numpy as np
from collections import OrderedDict

"Classical multidimensional scaling used by categorical_order() to reduce the distance matrix to a one-dimensional distance measure."

def cmds(mat, k=1):
        """Classical multidimensional scaling. Please refer to:
        https://en.wikipedia.org/wiki/Multidimensional_scaling#Classical_multidimensional_scaling
        """
        n = mat.shape[0]
        mat_square = np.square(mat)
        mat_center = np.eye(n) - np.ones((n, n)) / n
        m = -0.5 * mat_center.dot(mat_square).dot(mat_center)
        eigen_values, eigen_vectors = np.linalg.eigh(m)
        idx = np.argsort(eigen_values)[::-1]
        eigen_values, eigen_vectors = eigen_values[idx], eigen_vectors[:, idx]
        eigen_sqrt_diag = np.diag(np.sqrt(eigen_values[0:k]))
        return eigen_vectors[:, 0:k].dot(eigen_sqrt_diag)


def categorical_order(data, feature_columns, categorical_features, column, num_bins=100):

    """
    Categorical Order Function
    The accumulated local effects method needs – by definition – the feature values to have an order, because the method accumulates effects in a certain direction.
    `categorical_order()` function creates an order for the categorical features.
    """

    from statsmodels.distributions.empirical_distribution import ECDF

    df = pd.DataFrame(data, columns=range(len(feature_columns)))
    cate_features = set(categorical_features)
    features = sorted(df[column].unique())

    scores = pd.DataFrame(0, index=features, columns=features)
    for i in range(len(feature_columns)):
        if i == column:
            continue
        s = pd.DataFrame(0, index=features, columns=features)
        if i in cate_features:
            counts = pd.crosstab(data[:,column], data[:,i])
            fractions = counts.div(np.sum(counts, axis=1), axis=0)
            for j in features:
                diff = abs(fractions - fractions.loc[j]).sum(axis=1) / 2
                s.loc[j, :] = diff
                s.loc[:, j] = diff
        else:
            seq = np.arange(0, 1, 1 / num_bins)
            q = df[i].quantile(seq).to_list()
            cdf = df.groupby(column)[i].agg(ECDF)
            q_cdf = cdf.apply(lambda x: x(q))
            for j in features:
                diff = q_cdf.apply(lambda x: max(abs(x - q_cdf[j])))
                s.loc[j, :] = diff
                s.loc[:, j] = diff
        scores += s

    z = cmds(scores, 1).flatten()
    sorted_indices = z.argsort()
    return [features[i] for i in sorted_indices]


def fale_categorical(data, features, column,feature_name,feature_columns,categorical_names,predict_fn,test_labels,prot_attr,metric):

    """
    Compute the fairness accumulated local effect of a categorical (or str) feature.
    
    The function computes the difference in fairness between the different groups.
    This function relies on an ordering of the unique values/groups of the feature, 
    if the feature is not an ordered categorical, then an ordering is computed,
    which orders the groups by their similarity based on the distribution of the 
    other features in each group.
    """

    X = data[:,column]
    feature_indices = {f: i for i, f in enumerate(features)}
    unique, counts = np.unique(X, return_counts=True)
    count_df = pd.DataFrame(counts, columns=["size"], index=unique).loc[features]
    fractions = count_df / (count_df.sum() + 1e-6)


    fair_df = pd.DataFrame(data, columns = feature_columns)
    fair_df['labels'] = test_labels.values
    metrics = []
    for i in fair_df[feature_name].unique():
        x = fair_df[fair_df[feature_name] == i]
        pred = predict_fn.predict(x.loc[:,x.columns!='labels'])
        x = x.set_index(prot_attr)
        metrics.append(metric(x.labels, pred, prot_attr=prot_attr))
        #if metric == 'average_odds_difference':
        #    metrics.append(average_odds_difference(x.labels, pred, prot_attr=prott_attr))
        #elif metric == 'statistical_parity_difference':
        #    metrics.append(statistical_parity_difference(x.labels, pred, prot_attr=prott_attr))
        #elif metric == 'equal_opportunity_difference':
        #    metrics.append(equal_opportunity_difference(x.labels, pred, prot_attr=prott_attr))
    fair = pd.DataFrame(metrics,index=list(fair_df[feature_name].unique()))
    if fair.isna().sum()[0]>=1:
        df = pd.DataFrame(np.zeros(shape=(len(features),1)),index=features)
        return pd.DataFrame(
            np.zeros(shape=(len(features),1)),
            columns = df.columns,
            index=[categorical_names[column][int(i)] for i in df.index.values])
    else:

        z = data.copy()
        z[:,column] = [features[min(feature_indices[f] + 1, len(features) - 1)] for f in data[:,column]]
        ya_indices = (X != features[-1])
        right = pd.DataFrame(z[ya_indices],columns=feature_columns)
        right['labels'] = test_labels.values[ya_indices]

        metrics = []
        for i in right[feature_name].unique():
            x = right[right[feature_name] == i]    
            pred = predict_fn.predict(x.loc[:,x.columns!='labels'])
            x = x.set_index(prot_attr)
            metrics.append(metric(x.labels, pred, prot_attr=prot_attr))
            #if metric == 'average_odds_difference':
            #    metrics.append(average_odds_difference(x.labels, pred, prot_attr=prott_attr))
            #elif metric == 'statistical_parity_difference':
            #    metrics.append(statistical_parity_difference(x.labels, pred, prot_attr=prott_attr))
            #elif metric == 'equal_opportunity_difference':
            #    metrics.append(equal_opportunity_difference(x.labels, pred, prot_attr=prott_attr))
        pred_ya = pd.DataFrame(metrics,index=right[feature_name].unique())
        index = []
        for i in pred_ya.index:
            i = features[feature_indices[i]-1]
            index.append(float(i))
        pred_ya = pd.DataFrame(metrics,index=index)
        pred_ya = pred_ya.sort_index()
        fair = fair.sort_index()
        df_a = pred_ya - fair
        index = []
        for i in pred_ya.index:
            i = features[feature_indices[i]+1]
            index.append(float(i))
        df_a = df_a.dropna()
        df_a = pd.DataFrame(data=df_a[0].values,index=index)
        

        z[:,column] = [features[max(feature_indices[f] - 1, 0)]
                        for f in data[:,column]]
        yb_indices = (X != features[0])

        left = pd.DataFrame(z[yb_indices],columns=feature_columns)
        left['labels'] = test_labels.values[yb_indices]
        metrics = []
        for i in left[feature_name].unique():
            x = left[left[feature_name] == i]
            pred = predict_fn.predict(x.loc[:,x.columns!='labels'])
            x = x.set_index(prot_attr)
            metrics.append(metric(x.labels, pred, prot_attr=prot_attr))
            #if metric == 'average_odds_difference':
            #    metrics.append(average_odds_difference(x.labels, pred, prot_attr=prott_attr))
            #elif metric == 'statistical_parity_difference':
            #    metrics.append(statistical_parity_difference(x.labels, pred, prot_attr=prott_attr))
            #elif metric == 'equal_opportunity_difference':
            #    metrics.append(equal_opportunity_difference(x.labels, pred, prot_attr=prott_attr))    

        pred_yb = pd.DataFrame(metrics,index=left[feature_name].unique())
        index = []
        for i in pred_yb.index:
            i = features[feature_indices[i]+1]
            index.append(float(i))

        pred_yb = pd.DataFrame(metrics,index=index)
        pred_yb = pred_yb.sort_index()
        df_b = fair - pred_yb
        df_b = df_b.dropna()

        delta_df = pd.concat([df_a, df_b])
        delta_df = delta_df.reset_index()
        df = delta_df.groupby(['index']).mean()
        df.loc[features[0]] = 0
        df = df.loc[features].cumsum()

        return pd.DataFrame(
            df.values - np.sum(df.values * fractions.values, axis=0),
            columns=df.columns,
            index=[categorical_names[column][int(i)] for i in df.index.values]
    )


"""

"""
def fale_continuous(data, column, bins,features_columns, predict_fn, test_labels,prot_attr,metric):

    """
    Computes the fairness accumulated local effect of a numeric continuous feature.
    
    This function divides the feature in question into grid_size intervals (bins) 
    and computes the difference in fairness between the first and last value 
    of each interval and then centers the results.
    """
    #Create bins for the selected feature
    data1 = data.copy()
    feat_bins = pd.cut(data[:,column], bins, include_lowest=True)
    #Create dataframe that contains the population size of each bin
    data1[:,column] = [feat_bins.categories[i].right for i in feat_bins.codes] 
    unique, counts = np.unique(data1[:,column], return_counts=True)
    count_df = pd.DataFrame(counts, columns=["size"], index=unique)

    #Change the selected feature's value of each instance to the left value of the corresponding bin
    z = data.copy()
    z[:,column] = [feat_bins.categories[i].left for i in feat_bins.codes]
    #Group instances by bins
    left = pd.DataFrame(z, columns= features_columns)
    left['bins'] = [bins[b + 1] for b in feat_bins.codes]
    left = left.set_index('bins')
    left = left.sort_values(by = 'bins')
    left['labels'] = test_labels.values

    #Compute fairness value in each bin
    metrics = []
    for i in left.index.unique():
        x = left[left.index == i]
        pred = predict_fn.predict(x.loc[:,x.columns!='labels'])
        x = x.set_index(prot_attr)
        metrics.append(metric(x.labels, pred, prot_attr=prot_attr))
        #if metric == 'average_odds_difference':
        #    metrics.append(average_odds_difference(x.labels, pred, prot_attr=prott_attr))
        #elif metric == 'statistical_parity_difference':
        #    metrics.append(statistical_parity_difference(x.labels, pred, prot_attr=prott_attr))
        #elif metric == 'equal_opportunity_difference':
        #    metrics.append(equal_opportunity_difference(x.labels, pred, prot_attr=prott_attr))

    ya = np.asarray(metrics)

    #Change the selected feature's value of each instance to the right value of the corresponding bin
    z[:,column] = [feat_bins.categories[i].right for i in feat_bins.codes]

    #Compute fairness value in each bin
    right = pd.DataFrame(z, columns= features_columns)
    right['bins'] = [bins[b + 1] for b in feat_bins.codes]
    right = right.set_index('bins')
    right = right.sort_values(by = 'bins')
    right['labels'] = test_labels.values
    metrics = []
    for i in right.index.unique():
        x = right[right.index == i]
        pred = predict_fn.predict(x.loc[:,x.columns!='labels'])
        x = x.set_index(prot_attr)
        metrics.append(metric(x.labels, pred, prot_attr=prot_attr))
        #if metric == 'average_odds_difference':
        #    metrics.append(average_odds_difference(x.labels, pred, prot_attr=prott_attr))
        #elif metric == 'statistical_parity_difference':
        #    metrics.append(statistical_parity_difference(x.labels, pred, prot_attr=prott_attr))
        #elif metric == 'equal_opportunity_difference':
        #    metrics.append(equal_opportunity_difference(x.labels, pred, prot_attr=prott_attr))

    yb = np.asarray(metrics)
    
    if ya.ndim == 1:
        ya = np.expand_dims(ya, axis=-1)
        yb = np.expand_dims(yb, axis=-1)

    #[str(bins) for bins in feat_bins.categories]
    #[bins for bins in feat_bins.categories.right]

    #Compute differnce of fairness in each bin
    cols = OrderedDict({column: [bins for bins in feat_bins.categories.right]})
    delta_cols = OrderedDict({f"delta_{i}": yb[:, i] - ya[:, i] for i in range(ya.shape[1])})
    cols.update(delta_cols)
    delta_df = pd.DataFrame(cols)
    delta_df['size'] = count_df['size'].values
    delta_df = delta_df.set_index(column)
    delta_df.loc[min(bins), :] = 0
    delta_df = delta_df.sort_index()

    #df = delta_df.groupby([column])[list(delta_cols.keys())].agg(["mean", "size"])
    #for col in delta_cols.keys():
        #df[(col, "mean")] = df[(col, "mean")].cumsum()
        #df = df.sort_index()

    #Compute cumulative sum
    for col in delta_cols.keys():
        delta_df[col] = delta_df[col].cumsum()

    #for col in delta_cols.keys():
        #z = (df[(col, "mean")] + df[(col, "mean")].shift(1, fill_value=0)) * 0.5
        #avg = (z * df[(col, "size")]).sum() / (df[(col, "size")].sum() + 1e-6)
        #df[(col, "mean")] = df[(col, "mean")] - avg
    #df = df[[(col, "mean") for col in delta_cols.keys()]]
    #df.columns = list(delta_cols.keys())
    
    #Center values
    for col in delta_cols.keys():
        z = (delta_df[(col)] + delta_df[(col)].shift(1, fill_value=0)) * 0.5
        avg = (z * delta_df["size"]).sum() / (delta_df["size"].sum() + 1e-6)
        delta_df[(col)] = delta_df[(col)] - avg
    df = delta_df[[(col) for col in delta_cols.keys()]]
    df.columns = list(delta_cols.keys())
    return df

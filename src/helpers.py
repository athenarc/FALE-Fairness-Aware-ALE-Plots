import pandas as pd
import numpy as np
from aif360.sklearn.datasets import fetch_adult,fetch_compas,fetch_german
import seaborn as sns 
import matplotlib.pyplot as plt
from aif360.sklearn.metrics.metrics import average_odds_difference,equal_opportunity_difference 
from collections import OrderedDict
from FALElib import categorical_order,fale_categorical,fale_continuous
def binarize(X,y):
    X.index = y.index = pd.MultiIndex.from_arrays(X.index.codes, names=X.index.names)
    y = pd.Series(y.factorize(sort=True)[0], index=y.index, name=y.name)
    #X = pd.get_dummies(X, prefix_sep='__', drop_first=True)
    return X,y


def load_dataset(dataset):
    if dataset=='adult':
        X, y, sample_weight = fetch_adult()
        X.race = np.where(X.race != 'White','Non-White', X.race)
        X.race = np.where(X.race != 'White','Non-White', X.race)
        X.sex.replace('Male',1,inplace = True)
        X.sex.replace('Female',0,inplace = True)
        X.race.replace('Non-White',0,inplace=True)
        X.race.replace('White',1,inplace=True)
        X.sex = pd.to_numeric(X.sex)
        X.race = pd.to_numeric(X.race)
        X = X.drop(columns = 'education')

        feature_names = ['race', 'sex', 'age', 'workclass','education-num',
       'marital-status', 'occupation', 'relationship', 'capital-gain',
       'capital-loss', 'hours-per-week', 'native-country']
        
        prot_attr = ['sex','race']
        
        categorical_columns=[feature_names[i] for i in [3,5,6,7,11]]

    elif dataset=='compas':
        X, y = fetch_compas()
        X.sex.replace('Male',0,inplace = True)
        X.sex.replace('Female',1,inplace = True)
        X.race.replace('African-American',0,inplace=True)
        X.race.replace('Caucasian',1,inplace=True)
        X.sex = pd.to_numeric(X.sex)
        X.race = pd.to_numeric(X.race)
        X = X.drop(columns = ['c_charge_desc','age_cat'])
        X['labels']= y.values
        X.labels.replace('Recidivated',0,inplace=True)
        X.labels.replace('Survived',1,inplace=True)
        X.labels = pd.to_numeric(X.labels)
        y = X.labels
        X = X.drop(columns='labels')

        prot_attr = ['race','sex']
        feature_names = ['sex', 'age','race',  'juv_fel_count','juv_misd_count',
       'juv_other_count', 'priors_count	', 'c_charge_degree']
        categorical_columns=[feature_names[i] for i in [7]]

    else: 
        X, y = fetch_german()
        X.sex.replace('male',1,inplace = True)
        X.sex.replace('female',0,inplace = True)
        X['age'] = np.where((X['age'] > 25.0) , 'aged','young')
        X.age.replace('aged',1,inplace=True)
        X.age.replace('young',0,inplace=True)
        X['foreign_worker'].replace('no',0,inplace=True)
        X['foreign_worker'].replace('yes',1,inplace=True)
        X.sex = pd.to_numeric(X.sex)
        X.age = pd.to_numeric(X.age)
        X['foreign_worker'] = pd.to_numeric(X['foreign_worker'])

        prot_attr=['sex','age']
        feature_names = ['sex', 'age','foreign_worker', 'checking_status', 'duration',
       'credit_history', 'purpose', 'credit_amount', 'savings_status',
       'employment', 'installment_commitment', 'other_parties',
       'residence_since', 'property_magnitude', 'other_payment_plans',
       'housing', 'existing_credits', 'job', 'num_dependents', 'own_telephone',
       'marital_status']
        categorical_columns=[feature_names[i] for i in [0,1,2,3,5,6,8,9,11,13,14,15,17,18,19,20]]

    return X,y,feature_names,categorical_columns,prot_attr

def FALE_plots(explanations,metric,prot_attr,data,feature,categorical_columns,test_labels,pred):

    """Plots the FALE plots for each feature for the given fairness metric and protected attribute"""

    df1 = pd.DataFrame(explanations, columns = explanations.keys() )
    #data = transformer.invert(data).to_pd()
    
    if len(explanations)>15:
        a=22
        b=1
        c=1
    else:
        a=15
        b=1
        c=1
    
    plt.rcParams['pdf.fonttype'] = 42 
    plt.rcParams['ps.fonttype'] = 42

    fig = plt.figure(figsize=(20,70))
    value =  round(metric(test_labels, pred, prot_attr=prot_attr),3)
    if metric == average_odds_difference:
        fig.suptitle(f"FALE Plots for protected attribute '{prot_attr}' using fairness metric '{metric.__name__}' \n \
        Formula : [(FPRunpriv - FPRpriv) + (TPRunpriv - TPRpriv)]/2", fontsize=16)
    elif metric == equal_opportunity_difference:
        fig.suptitle(f"FALE Plots for protected attribute '{prot_attr}' using fairness metric '{metric.__name__}'  \n \
        Formula : TPRunpriv - TPRpriv", fontsize=16)     
    else:  
        fig.suptitle(f"FALE Plots for protected attribute '{prot_attr}' using fairness metric '{metric.__name__}' \n \
            Formula : PPRunpriv - PPRpriv (Predicted Positive Rate)", fontsize=16)               
    for i in feature:
        if i not in categorical_columns and len(explanations[i]['values'])<=2:
            continue
        else:
            if i not in categorical_columns:
                perce = np.linspace(0, 100, 10)
                bin = sorted(set(np.percentile(data[i].to_numpy(), perce)))
                feat_bin = pd.cut(data[i].to_numpy(), bin, include_lowest=True)
                data[i] = [feat_bin.categories[i].right for i in feat_bin.codes] 
                data.loc[len(data)] = np.nan
                data.at[len(data)-1,i]= min(bin)

                #x = data[i].value_counts().to_frame()
                #x.loc[min(bin), :] = 0
                #x = x.sort_index()

                #test[i] = [str(feat_bin.categories[i]) for i in feat_bin.codes]  

            plt.subplot(a,b,c)

            plt.title(i)
            plt.ylabel(f'(Test set fairness value : {value}) \n Fairness value ')
            plt.xticks(rotation=90)
            sns.lineplot(x = [str(values) for values in df1._get_value('values',i)], 
                y = [df1._get_value('scores',i)[m][0] for m in range(len(df1._get_value('scores',i)))] ,marker='o',color = 'red')  
            if i in categorical_columns: 
                ax2 = plt.twinx()
                ax = sns.countplot(data=data,x=i,hue=prot_attr,order=df1._get_value('values',i),ax=ax2,alpha=0.5)
                ax.set(ylabel='Population Count')
                for container in ax.containers:
                    ax.bar_label(container)
            else:
                ax2 = plt.twinx()
                ax = sns.countplot(data=data,x=i,hue=prot_attr,order = data[i].value_counts().sort_index().index,ax=ax2,alpha=0.5)
                ax.set(ylabel='Population Count')
                for container in ax.containers:
                    ax.bar_label(container)
            plt.legend(loc='upper right')
            data = data.drop(len(data)-1)
        """             
        if i in categorical_columns:   
            ax2 = plt.twinx() 
            ax2.set_ylabel('Population Count')
            sns.barplot(x = df1._get_value('values',i),y=[len(data[data[i]==x]) for x in df1._get_value('values',i)],ax=ax2,alpha=0.5,color='skyblue')
        else:
            ax2 = plt.twinx()
            ax2.set_ylabel('Population Count')
            #sns.barplot(x=x.index.values,y=x[i].values,ax=ax2,alpha=0.5,color='skyblue')
            data[i].value_counts().sort_index().plot.bar(ax=ax2,alpha=0.5,color='skyblue')
            """
        c=c+1
    #plt.subplots_adjust(top=0.95)
    plt.tight_layout(rect=[0, 0.03, 1, 0.98])
    plt.savefig('plot.pdf', format='pdf', bbox_inches='tight')
    plt.show()

def get_fale_metric(explanations,b,data,feature,categorical_columns):
    #explanations = get_explanation(metric = metric_,prot_attr=prot_attr_)
    #total_fair =  round(metric_(test_labels, pred, prot_attr=prot_attr_), 3)
    df1 = pd.DataFrame(explanations, columns=explanations.keys())
    diffs = []
    feat_sub_metric = []
    max_sub_fair = -10000

    for i in feature:
        if i not in categorical_columns and len(explanations[i]['values'])<=2:
            continue
        else:
            x = [str(values) for values in df1._get_value('values', i)]
            y = [df1._get_value('scores', i)[m][0] for m in range(len(df1._get_value('scores', i)))]
            if x[0].isnumeric() or (x[0].count('.') == 1 and x[0].replace('.','').isnumeric()):
                num_x = sorted([float(val) for val in x])
                for idx in range(0, len(num_x) - 1):
                    val1 = num_x[idx]
                    val2 = num_x[idx + 1]
                    ale_fair = y[idx + 1]
                    cur_data = data[(val1 <= data[i]) & (data[i] <= val2)]
                    feat_sub_metric.append((i, val2, ale_fair, len(cur_data)))
                    max_sub_fair = max(max_sub_fair, abs(ale_fair))
            else:
                for val, ale_fair in zip(x, y):
                    cur_data = data[data[i] == val]

                    feat_sub_metric.append((i, val, ale_fair, len(cur_data)))
                    max_sub_fair = max(max_sub_fair, abs(ale_fair))

    metric_res = []

    for feat, sub, sub_fair, sz in feat_sub_metric:
        m = abs(sub_fair) / max_sub_fair * b + sz / len(data) * (10 - b)
        metric_res.append((feat, sub, m))
    
    metric_res = sorted(metric_res, key=lambda x: x[2], reverse=True)
    metric_df = pd.DataFrame(metric_res, columns=['Feature', 'Subgroup', 'Value'])
    return metric_df

def get_explanation(metric,prot_attr,column_index,feature_columns,categorical_features,categorical_names,model,test,test_labels,feature):
    explanations = OrderedDict()
    for feature_name in feature:
        i = column_index[feature_name]
        if i in categorical_features:
            features = categorical_order(data = test,feature_columns = feature_columns,categorical_features = categorical_features, column = i)
            scores = fale_categorical(data = test, features=features, column=i, feature_name=feature_name, feature_columns = feature_columns, categorical_names=categorical_names, predict_fn=model, test_labels=test_labels,prot_attr = prot_attr ,metric = metric)
        else:
            percentiles = np.linspace(0, 100, 10)
            bins = sorted(set(np.percentile(test[:, i], percentiles)))
            scores = fale_continuous(data=test, column=i, bins=bins,features_columns = feature_columns, predict_fn=model ,test_labels=test_labels,prot_attr = prot_attr ,metric = metric)      
        #sampled_scores = []

        #scores = scores.sort_index()
        explanations[feature_name] = \
                {"values": list(scores.index.values), "scores": scores.values}
    
    return explanations
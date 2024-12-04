from evidently.report import  Report 
from evidently.metric_preset import DataDriftPreset
from evidently import ColumnMapping


X_train = pd.read_csv('../data/application_train.csv', index_col='SK_ID_CURR').drop(columns=['TARGET'])
X_test = pd.read_csv('../data/application_test.csv', index_col='SK_ID_CURR')

mask = (info_df['Dtype'].str.contains('str')) | (info_df['n_unique']==2)
categorical = list(info_df.loc[mask,:].index)
numerical = [idx for idx in info_df.index if idx not in categorical]
column_mapping = {'categorical_features': categorical,
                  'numerical_features': numerical }

ColumnMapping(**column_mapping)


data_drift_report = Report(metrics=[DataDriftPreset()])
data_drift_report.run(reference_data=X_train, 
                      current_data=X_test,
                       column_mapping=ColumnMapping(**column_mapping) )
data_drift_report.save('../data_drift_report.html')
data_drift_report.show()


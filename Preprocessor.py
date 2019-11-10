import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler
import pickle

final_file_name = 'data/test_proc.tsv'
train_file_name = 'data/train.tsv'
test_file_name = 'data/test.tsv'


class Preprocessor:
    def __init__(self):
        self.normalizer_file_name = 'normalizer.save'
        self._normalizer = None
        self.features_names = None

    @property
    def normalizer(self):
        if self._normalizer is None:
            self._normalizer = pickle.load(open(self.normalizer_file_name, 'rb'))
        return self._normalizer

    def features_list_to_columns(self, df):
        df = df.join(pd.DataFrame(df.features.tolist())).drop('features', 1)
        df = df.set_index(['id_job', 'set_code']).unstack()

        self.features_names = df.columns
        return df

    def transform_data(self, df_columns):
        x_transformed = self.normalizer.transform(df_columns)
        return pd.DataFrame(x_transformed, columns=df_columns.columns, index=df_columns.index)

    def get_features(self, df):
        df_ = df.copy()
        df_['max_feature_{}_index'] = df_.features.apply(np.argmax)

        max_ind = df_[['max_feature_{}_index', 'set_code']].values
        max_ind = [tuple(el) for el in max_ind]
        mean_df = pd.Series(self.normalizer.mean_, index=self.features_names)
        mean_of_max = mean_df[max_ind].reset_index(drop=True)

        df_['max_feature_{}_abs_mean_diff'] = df_.features.apply(np.max) - mean_of_max

        df_ = df_.set_index(['id_job', 'set_code']).drop('features', 1).unstack()
        df_.columns = [a.format(b) for a, b in df_.columns.values]
        return df_

    def fit(self, file_name=train_file_name):
        df = pd.read_csv(file_name, sep='\t')
        df = self.prepare_file(df)
        df = self.features_list_to_columns(df)

        st = StandardScaler()
        st.fit(df)
        pickle.dump(st, open(self.normalizer_file_name, 'wb'))

    def run(self, file_name, chunk_size=10**1):
        print('Preprocessing has started')
        i = 0
        for chunk in pd.read_csv(file_name, sep='\t', chunksize=chunk_size):
            print('chunk {} is being precessing..'.format(i))
            df = self.prepare_file(chunk)
            df_final = self.get_features(df)

            df = self.features_list_to_columns(df)

            df_transform = self.transform_data(df)
            df_transform = self.feature_columns_to_str(df_transform)
            df_final = df_final.join(df_transform)

            self.save_chunk(df_final, file_name=final_file_name)
            i += 1
        print('done!')

    @staticmethod
    def prepare_file(df):
        df = df.reset_index(drop=True)
        df.features = df.features.str.split(',')
        df['set_code'] = df.features.str[0].astype(int)
        df.features = df.features.str[1:]
        df.features = df.features.apply(lambda x: np.array(x).astype(int))
        return df

    @staticmethod
    def save_chunk(df, file_name):
        header = not os.path.isfile(file_name)
        with open(final_file_name, 'a') as f:
            df.to_csv(f, sep='\t', header=header)

    @staticmethod
    def feature_columns_to_str(df):
        long_features = pd.DataFrame([], index=df.index)
        for name, group in df.stack().groupby('set_code'):
            group = group.reset_index().set_index('id_job').drop('set_code', 1)
            group['new'] = np.apply_along_axis(lambda x: ",".join(x), axis=1, arr=group.values.astype(str))
            long_features['feature_{}_stand'.format(name)] = group['new']
        return long_features


if __name__ == '__main__':
    normalizer = StandardScaler()
    p = Preprocessor()
    p.fit()
    p.run(test_file_name)

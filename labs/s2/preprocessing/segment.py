import os

import pandas as pd
from sklearn.preprocessing import MinMaxScaler

from utils.dataset import read_dataset


# # Segment
# 
# 

# ## About the features
# 
# From the database:
# 
# `
# % 7. Attribute Information:
# %
# %     1.  region-centroid-col:  the column of the center pixel of the region.
# %     2.  region-centroid-row:  the row of the center pixel of the region.
# %     3.  region-pixel-count:  the number of pixels in a region = 9.
# %     4.  short-line-density-5:  the results of a line extractoin algorithm that
# %          counts how many lines of length 5 (any orientation) with
# %          low contrast, less than or equal to 5, go through the region.
# %     5.  short-line-density-2:  same as short-line-density-5 but counts lines
# %          of high contrast, greater than 5.
# %     6.  vedge-mean:  measure the contrast of horizontally
# %          adjacent pixels in the region.  There are 6, the mean and
# %          standard deviation are given.  This attribute is used as
# %         a vertical edge detector.
# %     7.  vegde-sd:  (see 6)
# %     8.  hedge-mean:  measures the contrast of vertically adjacent
# %           pixels. Used for horizontal line detection.
# %     9.  hedge-sd: (see 8).
# %     10. intensity-mean:  the average over the region of (R + G + B)/3
# %     11. rawred-mean: the average over the region of the R value.
# %     12. rawblue-mean: the average over the region of the B value.
# %     13. rawgreen-mean: the average over the region of the G value.
# %     14. exred-mean: measure the excess red:  (2R - (G + B))
# %     15. exblue-mean: measure the excess blue:  (2B - (G + R))
# %     16. exgreen-mean: measure the excess green:  (2G - (R + B))
# %     17. value-mean:  3-d nonlinear transformation
# %          of RGB. (Algorithm can be found in Foley and VanDam, Fundamentals
# %          of Interactive Computer Graphics)
# %     18. saturatoin-mean:  (see 17)
# %     19. hue-mean:  (see 17)
# `
# 
# - Numerical data has been normalized
# - no need to treat different types fo data (all numerical)
# - no need to treat missing values (no missing values in this dataset according to problem statement)


def preprocess():
    dataset = read_dataset('segment')
    data = dataset['data']

    df = pd.DataFrame(data)

    y = df['class'].copy()

    df = df.drop(columns=['class', 'region-pixel-count'])

    scaler = MinMaxScaler()

    X_num = scaler.fit_transform(df)
    df_X_num = pd.DataFrame(X_num, columns=df.columns)

    df_X_cat = df_X_num.copy()

    for col in list(df.columns):
        df_X_cat[col] = pd.qcut(x=df_X_num[col], q=5, duplicates='drop')

    df_X_num.to_csv(os.path.join('datasets', 'segment_clean_num.csv'), index=False)
    df_X_cat.to_csv(os.path.join('datasets', 'segment_clean_cat.csv'), index=False)

    y.to_csv(os.path.join('datasets', 'segment_clean_y.csv'), index=False, header=False)

    return 'segment_clean_num.csv', 'segment_clean_cat.csv', 'segment_clean_y.csv'

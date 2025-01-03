def clof(df):
    df_date = df.iloc[:,2:]
    df_other = df.iloc[:,0:2]

    df_date.set_index(df_date.columns[0], inplace=True)

    df_date.columns = pd.to_datetime(df_date.columns)
    df_date = df_date.T

    df_date = df_date.sort_index()
    df_date = df_date.T.reset_index()

    df = pd.concat([df_other,df_date], axis=1)

    df_to_calculate = df.iloc[:,2:]

    # 計算每一列的變異數
    row_variances = df_to_calculate.var(axis=1)

    # 計算每一列的平均值和標準差
    row_means = df_to_calculate.mean(axis=1)
    row_std = df_to_calculate.std(axis=1)

    df_to_calculate = df.iloc[:,2:]
    row_means = df_to_calculate.mean(axis=1)
    row_std = df_to_calculate.std(axis=1)

    #換掉行index為數字(方便後面計算時的索引)
    original_columns = df_to_calculate.columns
    df_to_calculate.columns = range(len(df_to_calculate.columns))

    count_values_exceeding_threshold = 0
    row_outlier = []

    for i in range(df_to_calculate.shape[0]):
        row = df_to_calculate.iloc[i]
        threshold = row_means[i] + 3 * row_std[i]
        outliers = row[row > threshold]  # 選超過threshold的值
        row_outlier.append(len(outliers))
        count_values_exceeding_threshold += len(outliers)

        # 轉換超過threshold得值為兩側平均，但保持NaN值不变
        for idx in outliers.index:
            if np.isnan(row[max(0, idx - 1)]) or np.isnan(row[min(len(row) - 1, idx + 1)]):
                continue  # 如果左右值有一个是NaN，则不进行替换
            else:
                mean_lr = (row[max(0, idx - 1)] + row[min(len(row) - 1, idx + 1)]) / 2
                df_to_calculate.at[i, idx] = mean_lr

    df_to_calculate = df_to_calculate.apply(lambda row: row.fillna(row.mean()), axis=1)
    df_to_calculate.dropna(how='all', inplace=True)

    max_values = df_to_calculate.max(axis=1)

    # 将每列除以对应的最大值进行标准化
    max_values[max_values == 0] = 1  # 將最大值為0的元素設置為1，避免除以0的情況
    df_to_calculate = df_to_calculate.div(max_values, axis=0)
    df_to_calculate[max_values == 0] = 0
    
    df_other = df_other
    df = pd.concat([df_other,df_to_calculate], axis=1)
    df.dropna(how='any', inplace=True)
    df.to_csv('df_CLOF.csv', index=False)

import numpy as np
from scipy import stats

def compute_CI_scale(percent,num_samples):
    from scipy import stats
    return stats.t.ppf(percent, df=num_samples-1) 

def compute_central_tendency(samples_arr):
    samples = np.array(samples_arr)
    
    mean = np.mean(samples,axis=0)
    max = np.max(samples,axis=0)
    min = np.min(samples,axis=0)
    mid = (max+min)/2

    std = np.std(samples, axis=0, ddof=1)  # ddof=1 for sample std
    n = samples.shape[0]  # number of repetitions
    # 95% CI using t-distribution
    sem = std / np.sqrt(n)  # standard error of mean
    
    t_crit = compute_CI_scale(0.975,n)
    moe = t_crit * sem  # ✅ Margin of error
    
    
    median = np.median(samples,axis=0)
    q1 = np.percentile(samples, 25, axis=0,method='midpoint')
    q3 = np.percentile(samples, 75, axis=0,method='midpoint')
    iqr = q3 - q1
    mad = np.median(np.abs(samples-median),axis=0) 
    
    return mid,max,min,mean,sem,moe,median,q1,q3,iqr,mad

def remove_outliers_iqr(ori_data, multiplier=1.5):
    """
    Remove outliers using the IQR (Interquartile Range) method.
    
    Parameters:
    -----------
    data : array-like
        Input data
    multiplier : float
        IQR multiplier for outlier detection. Default is 1.5 (standard).
        - 1.5: Standard (removes extreme outliers)
        - 3.0: Conservative (removes only very extreme outliers)
        
    Returns:
    --------
    cleaned_data : array
        Data with outliers removed
    outliers : array
        The outlier values that were removed
    mask : boolean array
        Mask indicating which points are NOT outliers
    """
    if len(ori_data.shape) == 1:
        data = np.expand_dims(ori_data,axis=1)
    else:
        data = ori_data
        
    outliers = []
    masks = []
    for dim in range(data.shape[1]):
        cur_dim_data = data[:,dim]
        
        q1 = np.percentile(cur_dim_data, 25,method='midpoint')
        q3 = np.percentile(cur_dim_data, 75,method='midpoint')
        iqr = q3 - q1
        
        lower_bound = q1 - multiplier * iqr
        upper_bound = q3 + multiplier * iqr
        
        mask = (cur_dim_data >= lower_bound) & (cur_dim_data <= upper_bound)
        masks.append(mask)
        outlier = data[~mask]
        outliers.append(outlier)
    total_mask = np.array(masks).T
    final_mask = np.all(total_mask, axis=1)
    cleaned_data = data[final_mask,:]
    outliers = data[~final_mask,:]

    return cleaned_data, outliers, final_mask

def assign_stars(df,p_column):
    df["stars"] = np.select(
        [
            df[p_column] < 0.001,
            df[p_column] < 0.01,
            df[p_column] < 0.05,
        ],
        ["***", "**", "*"],
        default=" "
    )
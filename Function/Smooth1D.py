import numpy as np
'''moving-average smooths a 1d function'''
'''by default, wraps around to be beginning of the function once the window exceeds the last edge'''
def moving_avg_smooth(arr, window, run_times):
    window_margin = (window - 1)//2
    out_arr = arr.copy()
    for run_time in range(0, run_times):
        temp_arr = np.zeros(out_arr.shape)
        for i in range(0, window_margin):
            arr_snip = out_arr[:i].tolist()
            arr_snip.extend(out_arr[out_arr.shape[0] - (window - len(arr_snip)):].tolist())
            arr_snip = np.asarray(arr_snip)
            snip_avg = np.average(arr_snip)
            temp_arr[i] = snip_avg
        for i in range(window_margin, out_arr.shape[0] - window_margin):
            arr_snip = out_arr[i-window_margin : i + window_margin + 1]
            snip_avg = np.average(arr_snip)
            temp_arr[i] = snip_avg
        for i in range(out_arr.shape[0] - window_margin, out_arr.shape[0]):
            arr_snip = out_arr[i - window_margin:].tolist()
            arr_snip.extend(out_arr[:window - len(arr_snip)].tolist())
            arr_snip = np.asarray(arr_snip)
            snip_avg = np.average(arr_snip)
            temp_arr[i] = snip_avg
        out_arr = temp_arr
    return out_arr

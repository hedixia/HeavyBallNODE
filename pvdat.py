from base import *



def lab_to_dat(lab, data, input_len, forecast_len=30):
    lab_size = len(lab)
    obs_size = lab_size // (input_len + 1)
    obs_lab = lab[:obs_size * (input_len + 1)].reshape(obs_size, input_len + 1)
    obs_lab = obs_lab.transpose()
    t = np.diff(obs_lab, axis=0)
    dat = data[obs_lab]
    x = dat[:input_len]
    y = dat[1:input_len + 1]
    fore_lab = obs_lab[-1]
    fore_lab = repeat(fore_lab, 'b -> t b', t=forecast_len)
    fore_lab = fore_lab + np.arange(forecast_len).reshape(-1, 1)
    z = data[fore_lab]
    out = (x, y, t, z)
    return (torch.Tensor(i) for i in out)


def pv(path='./data/pv.csv', input_len=64, tvt_ratio=(2, 1, 1), throwout_rate=0.1, error_var=0.0, verbose=False,
       forecast_len=30):
    data = np.genfromtxt(path, delimiter=',')[1:, :-2]
    np.random.seed(1)
    data = (data - data.mean(axis=0)) / data.std(axis=0)
    data = data + error_var * np.random.randn(*(data.shape))

    full_len = len(data) - 100
    eff_len = int(full_len * (1 - throwout_rate))
    assert eff_len > 0
    label_set = np.random.choice(range(full_len), eff_len, replace=False)
    label_set.sort()

    tvt_ratio = np.array(tvt_ratio) / np.sum(tvt_ratio)
    trlen, valen, _ = (tvt_ratio * eff_len).astype(int)
    tslen = eff_len - valen - trlen
    trlab, valab, tslab = label_set[:trlen], label_set[trlen:-tslen], label_set[-tslen:]

    output = EmptyClass()
    output.train_x, output.train_y, output.train_times, output.trext = lab_to_dat(trlab, data, input_len, forecast_len)
    output.valid_x, output.valid_y, output.valid_times, output.vaext = lab_to_dat(valab, data, input_len, forecast_len)
    output.test_x, output.test_y, output.test_times, output.tsext = lab_to_dat(tslab, data, input_len, forecast_len)

    if verbose:
        print('Train-validation-test ratio: {}'.format(tvt_ratio))
        print('Input {} tp | forecast {} tp'.format(input_len, forecast_len))
        print('Full {} tp | using {} tp'.format(full_len, eff_len))
        print('Train {} tp | Validation {} tp | Test {} tp'.format(trlen, valen, tslen))

    return output


if __name__ == '__main__':
    pv(verbose=True)

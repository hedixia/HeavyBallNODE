from base import *
from pvdat import pv

seqlen = 64
forelen = 8
data = pv(input_len=seqlen, verbose=True, forecast_len=forelen)


def fcriteria(a, b):
    d = a - b
    d2 = d * d
    d2 = rearrange(d2, 't ... -> t (...)')
    return d2.mean(dim=1)


def trainpv(model, fname, mname):
    lr_dict = {0: 0.001, 50: 0.0001}
    res = True
    cont = True
    torch.manual_seed(0)
    model_dict = model.state_dict()
    for i in model_dict:
        model_dict[i] *= 0.01
    model.load_state_dict(model_dict)
    outfile = open(fname, 'w')
    outfile.write('res: {}, cont: {}\n'.format(res, cont))
    outfile.write(model.__str__())
    outfile.write('\n' * 3)
    outfile.close()
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    print('Number of Parameters: {}'.format(count_parameters(model)))
    timelist = [time.time()]
    for epoch in range(200):
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])
        model.cell.nfe = 0
        batchsize = 64
        losslist = []
        for b_n in range(0, data.train_x.shape[1], batchsize):
            init, predict, forecast = model(data.train_times[:, b_n:b_n + batchsize],
                                            data.train_x[:, b_n:b_n + batchsize],
                                            multiforecast=torch.arange(forelen))
            loss = criteria(predict, data.train_y[:, b_n:b_n + batchsize])
            loss = loss + criteria(init, data.train_x[:, b_n:b_n + batchsize])
            lossf = criteria(forecast, data.trext[:, b_n:b_n + batchsize])
            total_loss = loss * 0.1 + lossf
            total_loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            losslist.append(loss.detach().cpu().numpy())
            optimizer.step()
        tloss = np.mean(losslist)
        timelist.append(time.time())
        nfe = model.cell.nfe / len(range(0, data.train_x.shape[1], batchsize))
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            init, predict, forecast = model(data.valid_times, data.valid_x, multiforecast=torch.arange(forelen))
            vloss = criteria(predict, data.valid_y)
            vloss = vloss + criteria(init, data.valid_x)
            vloss = vloss.detach().cpu().numpy()
            vfloss = fcriteria(forecast, data.vaext).detach().cpu().numpy()
            outfile = open(fname, 'a')
            outstr = str_rec(['epoch', 'tloss', 'nfe', 'vloss', 'vfloss', 'time'],
                             [epoch, tloss, nfe, vloss, vfloss, timelist[-1] - timelist[-2]])
            print(outstr)
            outfile.write(outstr + '\n')
            outfile.close()
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            init, predict, forecast = model(data.test_times, data.test_x, multiforecast=torch.arange(forelen))
            sloss = criteria(predict, data.test_y)
            sloss = sloss + criteria(init, data.test_x)
            sloss = sloss.detach().cpu().numpy()
            sfloss = fcriteria(forecast, data.tsext).detach().cpu().numpy()
            outfile = open(fname, 'a')
            outstr = str_rec(['epoch', 'nfe', 'sloss', 'sfloss', 'time'],
                             [epoch, model.cell.nfe, sloss, sfloss, timelist[-1] - timelist[-2]])
            print(outstr)
            outfile.write(outstr + '\n')
            outfile.close()
            torch.save(model.state_dict(), mname)

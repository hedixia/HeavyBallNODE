from base import *
from pvdat import pv

seqlen = 64
forelen = 8


def fcriteria(a, b):
    d = a - b
    d2 = d * d
    d2 = rearrange(d2, 't ... -> t (...)')
    return d2.mean(dim=1)


def trainpv(model, fname, mname, niter=500, lr_dict=None, gradrec=None, pre_shrink=0.01):
    data = pv(input_len=seqlen, verbose=True, forecast_len=forelen)
    lr_dict = {0: 0.001, 50: 0.0001} if lr_dict is None else lr_dict
    recorder = Recorder()
    torch.manual_seed(0)
    model = shrink_parameters(model, pre_shrink)
    criteria = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[0])
    print('Number of Parameters: {}'.format(count_parameters(model)))

    for epoch in range(niter):

        recorder['epoch'] = epoch

        # Train
        if epoch in lr_dict:
            optimizer = torch.optim.Adam(model.parameters(), lr=lr_dict[epoch])

        batchsize = 64
        for b_n in range(0, data.train_x.shape[1], batchsize):
            model.cell.nfe = 0
            batch_start_time = time.time()
            model.zero_grad()

            # Forward pass
            init, predict, forecast = model(data.train_times[:, b_n:b_n + batchsize],
                                            data.train_x[:, b_n:b_n + batchsize],
                                            multiforecast=torch.arange(forelen))
            loss = criteria(predict, data.train_y[:, b_n:b_n + batchsize])
            loss = loss + criteria(init, data.train_x[:, b_n:b_n + batchsize])
            lossf = criteria(forecast, data.trext[:, b_n:b_n + batchsize])
            total_loss = loss * 0.1 + lossf
            recorder['forward_time'] = time.time() - batch_start_time
            recorder['forward_nfe'] = model.cell.nfe
            # recorder['train_loss'] = loss
            recorder['train_forecast_loss'] = lossf

            # Gradient backprop computation
            if gradrec is not None:
                lossf.backward(retain_graph=True)
                vals = model.ode_rnn.h_ode
                for i in range(len(vals)):
                    recorder['grad_{}'.format(i)] = torch.norm(vals[i].grad)
                model.zero_grad()

            # Backward pass
            model.cell.nfe = 0
            total_loss.backward()
            # recorder['model_gradient_2norm']= gradnorm(model)
            # recorder['cell_gradient_2norm'] = gradnorm(model.cell)
            # recorder['ic_gradient_2norm'] = gradnorm(model.ic)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            recorder['mean_batch_time'] = time.time() - batch_start_time
            recorder['backward_nfe'] = model.cell.nfe

        # Validation
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            validation_start_time = time.time()
            init, predict, forecast = model(data.valid_times, data.valid_x, multiforecast=torch.arange(forelen))
            vloss = criteria(predict, data.valid_y)
            vloss = vloss + criteria(init, data.valid_x)
            vloss = vloss.detach().cpu().numpy()
            vfloss = fcriteria(forecast, data.vaext).detach().cpu().numpy()
            # recorder['validation_loss'] = vloss
            recorder['validation_foreast_loss'] = vfloss
            recorder['validation_nfe'] = model.cell.nfe
            recorder['validation_time'] = time.time() - validation_start_time

        # Test
        if epoch == 0 or (epoch + 1) % 1 == 0:
            model.cell.nfe = 0
            test_start_time = time.time()
            init, predict, forecast = model(data.test_times, data.test_x, multiforecast=torch.arange(forelen))
            sloss = criteria(predict, data.test_y)
            sloss = sloss + criteria(init, data.test_x)
            sloss = sloss.detach().cpu().numpy()
            sfloss = fcriteria(forecast, data.tsext).detach().cpu().numpy()
            # recorder['test_loss'] = sloss
            recorder['test_forecast_loss'] = sfloss
            recorder['test_nfe'] = model.cell.nfe
            recorder['test_time'] = time.time() - test_start_time

        recorder.capture(verbose=True)
        print('Epoch {} complete.'.format(epoch))

        if epoch % 20 == 0 or epoch == niter:
            recorder.writecsv(fname)
            torch.save(model.state_dict(), mname)

from misc import *

rec_names = ["iter", "loss", "acc", "nfe", "time/iter", "time"]
rec_unit = ["", "", "", "", "s", "min"]


# only for training mnist dataset
def train(model, optimizer, trdat, tsdat, args, evalfreq=2, lrscheduler=False, stdout=sys.stdout, **extraprint):
    defaultout = sys.stdout
    sys.stdout = stdout
    print("==> Train model {}, params {}".format(type(model), count_parameters(model)))
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print("==> Use accelerator: ", device)
    epoch = 0
    itrcnt = 0
    loss_func = nn.CrossEntropyLoss()
    itr_arr = np.zeros(args.niters)
    loss_arr = np.zeros(args.niters)
    nfe_arr = np.zeros(args.niters)
    time_arr = np.zeros(args.niters)
    acc_arr = np.zeros(args.niters)

    # training
    start_time = time.time()
    while epoch < args.niters:
        epoch += 1
        iter_start_time = time.time()
        acc = 0
        for x, y in trdat:
            itrcnt += 1
            model[1].df.nfe = 0
            optimizer.zero_grad()
            # forward in time and solve ode
            pred_y = model(x.to(device=args.gpu))
            # compute loss
            loss = loss_func(pred_y, y.to(device=args.gpu))
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            # make arrays
            itr_arr[epoch - 1] = epoch
            loss_arr[epoch - 1] += loss
            nfe_arr[epoch - 1] += model[1].df.nfe
            acc += torch.sum((torch.argmax(pred_y, dim=1) == y.to(device=args.gpu)).float())
        if lrscheduler:
            lrscheduler.step()
        iter_end_time = time.time()
        time_arr[epoch - 1] = iter_end_time - iter_start_time
        loss_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        acc /= 60000
        printouts = [epoch, loss_arr[epoch - 1], acc, nfe_arr[epoch - 1], time_arr[epoch - 1],
                     (time.time() - start_time) / 60]
        print(str_rec(rec_names, printouts, rec_unit, presets="Train|| {}"))
        print('Extra: ', extraprint)
        try:
            print(torch.sigmoid(model[1].df.gamma))
        except Exception:
            pass
        if epoch % evalfreq == 0:
            model[1].df.nfe = 0
            end_time = time.time()
            loss = 0
            acc = 0
            bcnt = 0
            for x, y in tsdat:
                # forward in time and solve ode
                y = y.to(device=args.gpu)
                pred_y = model(x.to(device=args.gpu))
                pred_l = torch.argmax(pred_y, dim=1)
                acc += torch.sum((pred_l == y).float())
                bcnt += 1
                # compute loss
                loss += loss_func(pred_y, y) * y.shape[0]

            loss /= 10000
            acc /= 10000
            printouts = [epoch, loss.detach().cpu(), acc.detach().cpu(), str(model[1].df.nfe / bcnt),
                         str(count_parameters(model))]
            names = ["iter", "loss", "acc", "nfe", "param cnt"]
            print(str_rec(names, printouts, presets="Test|| {}"))
            acc_arr[epoch - 1] = acc.detach().cpu().numpy()
        if time.time() - start_time > 3600 * 4:
            break
    sys.stdout = defaultout
    return itr_arr, loss_arr, nfe_arr, time_arr, acc_arr

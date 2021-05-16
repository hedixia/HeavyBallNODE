from misc import *

rec_names = ["model", "test#", "train/test", "iter", "loss", "acc", "forwardnfe", "backwardnfe", "time/iter",
             "time_elapsed"]
rec_unit = ["", "", "", "", "", "", "", "", "s", "min"]
import csv


# only for training mnist dataset
def train(model, optimizer, trdat, tsdat, args, modelname, testnumber=0, evalfreq=1, lrscheduler=False,
          csvname='outdat.csv', stdout=sys.stdout, **extraprint):
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
    forward_nfe_arr = np.zeros(args.niters)

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
            if isinstance(pred_y, tuple):
                pred_y, rec = pred_y
                # compute loss
                loss = loss_func(pred_y, y.to(device=args.gpu)) + 0.1 * torch.mean(rec)
            else:
                loss = loss_func(pred_y, y.to(device=args.gpu))
            forward_nfe_arr[epoch - 1] += model[1].df.nfe

            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10.0)
            optimizer.step()
            # make arrays
            itr_arr[epoch - 1] = epoch
            loss_arr[epoch - 1] += loss.detach().cpu().numpy()
            nfe_arr[epoch - 1] += model[1].df.nfe
            acc += torch.sum((torch.argmax(pred_y, dim=1) == y.to(device=args.gpu)).float())
        if lrscheduler:
            lrscheduler.step()
        iter_end_time = time.time()
        time_arr[epoch - 1] = iter_end_time - iter_start_time
        loss_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        forward_nfe_arr[epoch - 1] *= 1.0 * epoch / itrcnt
        backwardnfe = nfe_arr[epoch - 1] - forward_nfe_arr[epoch - 1]
        acc = acc.detach().cpu().numpy() / 60000
        printouts = [modelname, testnumber, 'train', epoch,
                     loss_arr[epoch - 1], acc, forward_nfe_arr[epoch - 1],
                     backwardnfe, time_arr[epoch - 1],
                     (time.time() - start_time) / 60]
        csvfile = open(csvname, 'a')
        writer = csv.writer(csvfile)
        writer.writerow(printouts)
        csvfile.close()
        print(str_rec(rec_names, printouts, rec_unit))
        if time_arr[epoch - 1] > 2400:
            break
        if epoch % evalfreq == 0:
            model[1].df.nfe = 0
            test_time = time.time()
            loss = 0
            acc = 0
            bcnt = 0
            for x, y in tsdat:
                # forward in time and solve ode
                y = y.to(device=args.gpu)
                pred_y = model(x.to(device=args.gpu))
                if isinstance(pred_y, tuple):
                    pred_y, rec = pred_y
                pred_l = torch.argmax(pred_y, dim=1)
                acc += torch.sum((pred_l == y).float())
                bcnt += 1
                # compute loss
                loss += loss_func(pred_y, y) * y.shape[0]
            test_time = time.time() - test_time
            loss = loss.detach().cpu().numpy() / 10000
            acc = acc.detach().cpu().numpy() / 10000
            printouts = [modelname, testnumber, 'test', epoch,
                         loss, acc, model[1].df.nfe / len(tsdat),
                         0, test_time,
                         (time.time() - start_time) / 60]
            csvfile = open(csvname, 'a')
            writer = csv.writer(csvfile)
            writer.writerow(printouts)
            csvfile.close()
            print(str_rec(rec_names, printouts, rec_unit))
            acc_arr[epoch - 1] = acc
    sys.stdout = defaultout
    return itr_arr, loss_arr, nfe_arr, time_arr, acc_arr

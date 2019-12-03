import numpy as np
import torch
from tqdm import tqdm
from torch.optim.lr_scheduler import StepLR

def pre_train(model, epoch, criterion, optimizer, scheduler, train_loader, type='only', print_num=100):
    model.train()
    sum_loss = 0

    scheduler.step()

    for ix, tr_data in enumerate(tqdm(train_loader)):
        if type == 'only':
            # Anchor embedding
            a_img = tr_data[0].cuda()
            p_img = tr_data[1].cuda()
            n_img = tr_data[2].cuda()

            # Embedding
            a_emb = model(a_img)
            p_emb = model(p_img)
            n_emb = model(n_img)

        else:
            a_img = tr_data[0].cuda()
            p_img = tr_data[1].cuda()
            n_img = tr_data[2].cuda()

            # Embedding
            a_emb, p_emb, n_emb = model(a_img, p_img, n_img)

        loss = criterion(a_emb, p_emb, n_emb)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        sum_loss += loss.item()

        if ix % print_num == 0:
            print('%d iter/%d Epoch - Batch Loss %.3f' %(ix, epoch, loss.item()))

    total_loss = sum_loss / len(train_loader)
    print('(Train) %d Epoch - Total Loss %.3f' %(epoch, total_loss))
    return model

def pre_test(model, epoch, criterion, test_loader, type='only'):
    model.eval()
    sum_loss = 0

    size = len(test_loader)

    count1, count2 = 0, 0
    num = 0
    with torch.no_grad():
        for ix, te_data in enumerate(tqdm(test_loader)):
            if type == 'only':
                # Anchor embedding
                a_img = te_data[0].cuda()
                p_img = te_data[1].cuda()
                n_img = te_data[2].cuda()

                # Embedding
                a_emb = model(a_img)
                p_emb = model(p_img)
                n_emb = model(n_img)

            else:
                a_img = te_data[0].cuda()
                p_img = te_data[1].cuda()
                n_img = te_data[2].cuda()

                # Embedding
                a_emb, p_emb, n_emb = model(a_img, p_img, n_img)

            loss = criterion(a_emb, p_emb, n_emb)

            a,b = (a_emb - p_emb).pow(2).mean(1), (a_emb - n_emb).pow(2).mean(1)
            count1 += torch.sum((b-a > 0).view(-1)).item()
            count2 += torch.sum((a-b > 0).view(-1)).item()

            num += a_img.size()[0]

            sum_loss += loss.item()

        total_loss = sum_loss / size
        print('(Test) %d Epoch - Total Loss : %.3f' %(epoch, total_loss))
        print('%d/%d' %(count1, count2))

    return model, total_loss
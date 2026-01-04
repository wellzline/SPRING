from autoattack import AutoAttack
import torch,os
from tqdm import tqdm
import torch.nn.functional as F
import torchvision.transforms as T
import torchvision
import numpy as np
import statistics
import torch.distributions as dist
import statistics
import random
from utils import data_prepare
from attack_algorithms import (pgd_loss, fgsm_loss, trades_loss, mart_loss, CVaR_loss)
from tiny_imagenet import TinyImageNet
import torchvision.transforms as transforms
from torchvision.datasets import ImageFolder

seed = 0
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)

device = 'cuda' if torch.cuda.is_available() else 'cpu'


x_min = torch.tensor([0.0, 0.0, 0.0]).cuda()  
x_max = torch.tensor([1.0, 1.0, 1.0]).cuda()




def get_dataloader(args):
    transform_test = T.Compose([
            T.ToTensor()
        ])
    if args.dataset == "CIFAR10":
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10("./dataset/cifar_10", train=False, download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR10("./dataset/cifar_10", train=True, download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)
    
    elif args.dataset == "CIFAR100":
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100("./dataset/cifar_100", train=False, download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)

        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.CIFAR100("./dataset/cifar_100", train=True, download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)
        
    elif args.dataset == "CINIC10":
        train_loader = torch.utils.data.DataLoader(
            ImageFolder(root=os.path.join("./dataset/cinic-10", 'train'), transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)

        test_loader = torch.utils.data.DataLoader(
            ImageFolder(root=os.path.join("./dataset/cinic-10", 'test'), transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)

    elif args.dataset == "svhn":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root="./dataset/svhn", split='train', download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)

        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.SVHN(root="./dataset/svhn", split='test', download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)
    
    elif args.dataset == "TinyImageNet":
        train_loader = torch.utils.data.DataLoader(
            TinyImageNet("./dataset/tiny-imagenet", split='train',  transform=transform_test, download=True),
            batch_size=1000, shuffle=False, num_workers=8)

        test_loader = torch.utils.data.DataLoader(
            TinyImageNet("./dataset/tiny-imagenet", split='val',  transform=transform_test, download=True),
            batch_size=1000, shuffle=False, num_workers=8)
    
    elif args.dataset == "MNIST":
        train_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root="./dataset/MNIST", train=True, download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)
        test_loader = torch.utils.data.DataLoader(
            torchvision.datasets.MNIST(root="./dataset/MNIST", train=False, download=True, transform=transform_test),
            batch_size=1000, shuffle=False, num_workers=8)
    
    elif args.dataset == "ImageNet":
        transform_test = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor()
        ])
        train_dir = os.path.join("./dataset/ImageNet50", 'train')
        val_dir = os.path.join("./dataset/ImageNet50", 'val')

        train_dataset = torchvision.datasets.ImageFolder(train_dir, transform=transform_test)
        val_dataset = torchvision.datasets.ImageFolder(val_dir, transform=transform_test)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=1000, shuffle=False, num_workers=8)
        test_loader = torch.utils.data.DataLoader(val_dataset, batch_size=1000, shuffle=False, num_workers=8)

    return test_loader, train_loader





def evaluate_aa(args, model, testloader, log_file):
    model.eval()
    l = [x for (x, y) in testloader]
    x_test = torch.cat(l, 0)
    l = [y for (x, y) in testloader]
    y_test = torch.cat(l, 0)
    adversary = AutoAttack(model, norm='Linf', eps=args.attack_eps/255, version='standard', log_path=log_file)
    x_adv = adversary.run_standard_evaluation(x_test, y_test, bs=256)


def accuracy(logits, target):
    _, pred = torch.max(logits, dim=1)
    correct = (pred == target).sum()
    total = target.size(0)
    acc = (float(correct) / total) * 100
    return acc


def func_pgd(args, data_loader, model, optimizer):
    model.eval()
    acc, pgd_acc = 0, 0
    tq = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
    for i, (x,y) in tq:
        x, y = x.to(device), y.to(device)
        x_pgd = pgd_loss(model, x, y, optimizer=optimizer,
                            step_size=args.attack_lr/255, 
                            epsilon=args.attack_eps/255, 
                            attack_steps=args.attack_steps, 
                            attack=True)
        with torch.no_grad():
            logits = model(x)
            pgd_logits = model(x_pgd)
            acc += accuracy(logits, y)
            pgd_acc += accuracy(pgd_logits, y)
        tq.set_description('Evaluation: clean/pgd: {:.2f}/{:.2f}'.format(acc/(i+1), pgd_acc/(i+1)))     
    acc, pgd_acc = round(acc/len(tq), 2), round(pgd_acc/len(tq), 2)
    return acc, pgd_acc




    

def evaluate_PGD(args, model, optimizer, log_file):
    trainloader, testloader = data_prepare(args)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    with open(log_file, 'a+') as file:
        train_acc, train_pgd_acc = func_pgd(args, trainloader, model, optimizer)  
        test_acc, test_pgd_acc = func_pgd(args, testloader, model, optimizer)  
        file.write(f"train dataset: PGD-{args.attack_steps}:@Evaluation: clean/pgd {train_acc}/{train_pgd_acc} lr:{optimizer.param_groups[0]['lr']}\n\n")
        file.write(f"test dataset: PGD-{args.attack_steps}:@Evaluation: clean/pgd {test_acc}/{test_pgd_acc} lr:{optimizer.param_groups[0]['lr']}\n\n")
        file.write(f"gap error: PGD-{args.attack_steps}:@Evaluation: clean/pgd {round(train_acc-test_acc, 2)}/{round(train_pgd_acc-test_pgd_acc, 2)} lr:{optimizer.param_groups[0]['lr']}\n\n")
        file.write("*"*100 + "\n")


def cw_loss(args, logits, targets, margin=2, reduce=True):
    onehot_targets = F.one_hot(targets, args.num_class).float().to(logits.device)
    self_loss = torch.sum(onehot_targets * logits, dim=1)
    other_loss = torch.max(
        (1 - onehot_targets) * logits - onehot_targets * 1000, dim=1
    )[0]

    loss = -torch.sum(torch.clamp(self_loss - other_loss + margin, min=0))

    if reduce:
        sample_num = onehot_targets.shape[0]
        loss = loss / sample_num
    return loss

def cw_attack(args, model, optimizer, x, y):
    x_adv = x.detach().clone()
    x_adv = x_adv + torch.empty_like(x_adv).uniform_(-args.attack_eps/255, args.attack_eps/255)
    x_adv = torch.clamp(x_adv, min=0, max=1).detach()

    for _ in range(args.attack_steps):
        x_adv.requires_grad = True
        optimizer.zero_grad()
        logits = model(x_adv)  
        loss = cw_loss(args, logits, y)  # torch.Size([256, 100])   torch.Size([256])
        
        loss.backward()
        grad = x_adv.grad.detach()
        grad = grad.sign()

        # Update adversarial example
        x_adv = x_adv + (args.attack_lr/255) * grad

        # Projection to the epsilon-ball
        x_adv = x + torch.clamp(x_adv - x, min=-args.attack_eps/255, max=args.attack_eps/255)
        x_adv = x_adv.detach()
        x_adv = torch.clamp(x_adv, min=0, max=1).detach()
    return x_adv


def func_cw(args, model, optimizer, data_loader):
    model.eval()
    acc, cw_acc = 0, 0

    tq = tqdm(enumerate(data_loader), total=len(data_loader), leave=True)
    for i, (x,y) in tq:
        x, y = x.to(device), y.to(device)
        x_cw = cw_attack(args, model, optimizer, x, y)
        with torch.no_grad():
            logits = model(x)
            cw_logits = model(x_cw)
            acc += accuracy(logits, y)
            cw_acc += accuracy(cw_logits, y)
        tq.set_description('Evaluation: clean/cw: {:.2f}/{:.2f}'.format(acc/(i+1), cw_acc/(i+1)))
    acc, cw_acc = round(acc/len(tq), 2), round(cw_acc/len(tq), 2) 
    return acc, cw_acc
        

def evaluate_cw(args, model, optimizer, log_file):
    trainloader, testloader = data_prepare(args)
    with open(log_file, 'a+') as file:
        train_acc, train_cw_acc = func_cw(args, model, optimizer, trainloader)
        test_acc, test_cw_acc = func_cw(args, model, optimizer, testloader)
        file.write(f"train dataset CW-{args.attack_steps}:@Evaluation: clean/cw {train_acc}/{train_cw_acc} lr:{optimizer.param_groups[0]['lr']}\n\n")
        file.write(f"test  dataset CW-{args.attack_steps}:@Evaluation: clean/cw {test_acc}/{test_cw_acc} lr:{optimizer.param_groups[0]['lr']}\n\n")
        file.write(f"gap error: CW--{args.attack_steps}:@Evaluation: clean/pgd {round(train_acc-test_acc, 2)}/{round(train_cw_acc - test_cw_acc, 2)} lr:{optimizer.param_groups[0]['lr']}\n\n")
        file.write("*"*100 + "\n")



def run_prob(x, y, model, eps, sample_id, distribute, GE):
    def prop(x):
        model.eval()
        y = model(x)  #  torch.Size([batch, 10])
        y_diff = torch.cat((y[:,:x_class], y[:,(x_class+1):]),dim=1) - y[:,x_class].unsqueeze(-1)
        y_diff, _ = y_diff.max(dim=1)
        return y_diff  # >0 means AE

    def brute_force(prop, distribute, count_iterations=1):
        count_above, count_total, count_particles = int(0), int(0), int(100)

        for i in range(count_iterations):
            prior = distribution[distribute]
            x = prior.sample(torch.Size([count_particles]))  # torch.Size([batch, 3, 32, 32])
            
            x = torch.clamp(x, x_sample - eps, x_sample + eps)
            x = torch.clamp(x, min=x_min.view(3, 1, 1), max=x_max.view(3, 1, 1))
            # x = torch.clamp(x, min=x_min.view(1, 1, 1), max=x_max.view(1, 1, 1))
            
            s_x = prop(x).squeeze(-1)   # print(s_x.shape)  torch.Size([10])
            count_above += int((s_x >= 0).float().sum().item())  # the number of AE
            count_total += count_particles

        return count_above, count_total

    x_sample = x[sample_id]  
    x_class = y[sample_id]
    
    # prior_uni = dist.Uniform(  # MNIST
    #     low=torch.max(x_sample - eps * (x_max - x_min).view(1, 1, 1), x_min.view(1, 1, 1)),
    #     high=torch.min(x_sample + eps * (x_max - x_min).view(1, 1, 1), x_max.view(1, 1, 1))
    # )
    
    # prior_norm = dist.Normal(
    #     loc=x_sample,  
    #     scale=eps * (x_max - x_min).view(1, 1, 1)  
    # )

    # prior_lap = dist.Laplace(
    #     loc=x_sample,  
    #     scale=eps * (x_max - x_min).view(1, 1, 1)  
    # )



    prior_uni = dist.Uniform(   # CIFAR, SVHN, Tiny-ImageNet
        low=torch.max(x_sample - eps * (x_max - x_min).view(3, 1, 1), x_min.view(3, 1, 1)),
        high=torch.min(x_sample + eps * (x_max - x_min).view(3, 1, 1), x_max.view(3, 1, 1))
    )
    
    prior_norm = dist.Normal(
        loc=x_sample,  
        scale=eps * (x_max - x_min).view(3, 1, 1)  
    )

    prior_lap = dist.Laplace(
        loc=x_sample,  
        scale=eps * (x_max - x_min).view(3, 1, 1)  
    )
    
    distribution = {"Uniform":prior_uni, "Normal": prior_norm, "Laplace":prior_lap}

    
    # input = x_sample.view(1, 3, 224, 224)  # imagenet
    input = x_sample.view(1, 3, 32, 32)  # CIFAR, SVHN
    ## input = x_sample.view(1, 1, 28, 28)  # MNIST
    ## input = x_sample.view(1, 3, 64, 64)    # Tiny-ImageNet
    s_x = prop(input).squeeze(-1)  

    if GE:
        with torch.no_grad():
            AEs, total = brute_force(prop, distribute, count_iterations=1)
        return AEs, total
    else:
        if s_x.item() < 0:
            with torch.no_grad():
                AEs, total = brute_force(prop, distribute, count_iterations=1)
            return AEs, total
        else:
            return -1, -1



def func_pr(data_loader, radius, model, log_file, distribute, GE, mode):
    x = torch.zeros(10000, 3, 32, 32).cuda() 
    y = torch.zeros(10000, dtype=torch.long).cuda()

    if GE:
        sample_num = 1000
        for idx, (data, target) in enumerate(data_loader):
            if idx <= 0:
                data, target = data.float().cuda(), target.long().cuda()
                x[(idx*1000):((idx+1)*1000),:,:,:] = data
                y[(idx*1000):((idx+1)*1000)] = target
    else:
        if mode == "train":
            sample_num = 1000
            for idx, (data, target) in enumerate(data_loader):
                if idx <= 0:
                    data, target = data.float().cuda(), target.long().cuda()
                    x[(idx*1000):((idx+1)*1000),:,:,:] = data
                    y[(idx*1000):((idx+1)*1000)] = target
                else:
                    break
        else:
            sample_num = 10000
            for idx, (data, target) in enumerate(data_loader):
                if idx <= 9:
                    data, target = data.float().cuda(), target.long().cuda()
                    x[(idx*1000):((idx+1)*1000),:,:,:] = data
                    y[(idx*1000):((idx+1)*1000)] = target
                else:
                    break

    result_Prob, result_pr = [], {}
    with open(log_file, 'a+') as file:   
        file.write(f"distribution: {distribute}\n")    
        for eps in radius:
            pr, correct , all_count = [], 0, 0
            for idx in tqdm(range(sample_num)):
                AEs, samples = run_prob(x, y, model, eps, idx, distribute, GE)
                if AEs >= 0:
                    correct += samples - AEs
                    all_count += samples
                    pr.append( (samples - AEs) / samples)
            mean_value, std_dev = statistics.mean(pr), statistics.stdev(pr)
            result_pr[f"PR_{str(eps)}"] =  f"{str(round(mean_value * 100, 2))}/{str(round(std_dev * 100, 2))}"
            print(f"eps: {eps} Mean/Std: {round(mean_value * 100, 2)}/{round(std_dev * 100, 2)}, Length: {len(pr)}\n\n")


            Prob_acc = {}
            Prob_acc['Aug.Acc'] = round( (correct / all_count) * 100, 2) 
            q_values = [0.2, 0.1, 0.05, 0.01]
            for quantile in q_values:
                threshold = 1 - quantile
                filtered_pr = [p for p in pr if p > threshold]
                Prob_acc[f"Prob.Acc_{quantile}"] = round( (len(filtered_pr) / len(pr)) * 100, 2)  

            pr_ = np.array(pr)  
            mean_value, std_dev = np.mean(pr_), np.std(pr_, ddof=0)    
            Prob_acc['Mean/Std'] = f"{str(round(mean_value * 100, 2))}/{str(round(std_dev * 100, 2))}"
            result_Prob.append(Prob_acc)
            file.write(f"{mode} eps: {eps} \nProb_acc: {str(Prob_acc)} \n\n")

        file.write(f"Length: {len(pr)}\n\n")

        file.write("--"*100 + "\n")
        return result_pr, result_Prob


def distribution(args, model, log_file, radius, GE, distribute='none'):
    with open(log_file, 'a+') as file:   
        file.write("@"*100 + "\n\n")  
    test_loader, train_loader = get_dataloader(args) 
    result_train_pr, Prob_train = func_pr(train_loader, radius, model, log_file, distribute, GE, mode='train')
    result_test_pr, Prob_test = func_pr(test_loader, radius, model, log_file, distribute, GE, mode='test')
    result = {}
    for key in list(result_train_pr):
        train = result_train_pr[key].split('/')
        test = result_test_pr[key].split('/')
        result[key] = f"{train[0]}/{test[0]}/{ str(round(float(train[0])-float(test[0]), 2)) }"

    with open(log_file, 'a+') as file: 
        file.write(f"train PR {str(result_train_pr)} \n\n") 
        file.write(f"test PR {str(result_test_pr)} \n\n") 
        file.write("gap error: PR:\n")
        file.write(f"{str(result)} \n\n") 
        file.write("--"*100 + "\n")

        for index in range(len(Prob_train)):
            train, test = Prob_train[index], Prob_test[index]
            Prob_gap = {}
            for key in list(train)[:-1]:
                Prob_gap[key] = round(train[key] - test[key], 2)
            file.write(f"gap error of epsilon {radius[index]}:\n")
            file.write(str(Prob_gap) + '\n\n')

        file.write("**"*100 + "\n")
        file.write("**"*100 + "\n") 
        
        

def evaluate_PR(args, model, log_file, GE=False):
    radius = [round(8/255, 2), 0.08, 0.1, 0.12]

    distribution(args, model, log_file, radius, GE, distribute = 'Uniform')
 












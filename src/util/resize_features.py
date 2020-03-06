import sys, os, glob
import torch
import torch.nn.functional as F

if __name__ == '__main__':
    slow_size = (10, 8, 10)
    fast_size = (100, 8, 10)

    # for MSR-VTT
    root = "/groups1/gaa50131/datasets/MSR-VTT/features/sfnl152_k700_16f"
    sem_path = glob.glob(os.path.join(root, "semantic/*"))
    mot_path = glob.glob(os.path.join(root, "motion/*"))
    save_path_sem = os.path.join(root, "sem_int")
    save_path_mot = os.path.join(root, "mot_int")
    for i, path in enumerate(sem_path):
        savepath = os.path.join(save_path_sem, os.path.basename(path))
        if os.path.exists(savepath):
            continue
        ft = torch.load(path)
        if tuple(ft.size()[1:4]) != slow_size:
            ft = F.interpolate(ft.unsqueeze(0), size=slow_size, mode='trilinear', align_corners=False).squeeze(0)
            torch.save(ft, savepath)
            print(savepath)
        else:
            torch.save(ft, savepath)
        if i % 10 == 9:
            print("{} done".format(i+1))
    for i, path in enumerate(mot_path):
        savepath = os.path.join(save_path_mot, os.path.basename(path))
        if os.path.exists(savepath):
            continue
        ft = torch.load(path)
        if tuple(ft.size()[1:4]) != fast_size:
            ft = F.interpolate(ft.unsqueeze(0), size=fast_size, mode='trilinear', align_corners=False).squeeze(0)
            torch.save(ft, savepath)
            print(savepath)
        else:
            torch.save(ft, savepath)
        if i % 10 == 9:
            print("{} done".format(i+1))

    print("done!!!")

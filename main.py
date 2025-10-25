import os
import importlib
import torch
from option import get_option
import faulthandler; faulthandler.enable()

def main():
    opt = get_option()

    if torch.cuda.is_available(): 
        os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"    
        torch.cuda.set_device(opt.gpu_num)
        torch.backends.cudnn.enabled = True
        torch.backends.cudnn.benchmark =True
        print("Current device: idx%s | %s" %(torch.cuda.current_device(), torch.cuda.get_device_name(torch.cuda.current_device())))
        
    torch.manual_seed(opt.seed)

    module = importlib.import_module("models.{}".format(opt.model.lower()))


    from solver import Solver
    solver = Solver(module, opt)
    solver.fit()
    
if __name__ == "__main__":        
    main()

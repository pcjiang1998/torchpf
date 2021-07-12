[![Build Status](https://travis-ci.com/2985578957/torchpf.svg?branch=master)](https://travis-ci.com/2985578957/torchpf)

# torchpf
This is a lightweight neural network analyzer based on PyTorch.
It is designed to make building your networks quick and easy, with the ability to debug them.
**Note**: This repository is currently under development. Therefore, some APIs might be changed.

This tools can show

* Total number of network parameters
* Theoretical amount of floating point arithmetics (FLOPs)
* Theoretical amount of multiply-adds (MAdd)
* Memory usage

## Installing
There're two ways to install torchpf into your environment.
* Install it via pip.
```bash
$ pip install -U git+https://github.com/2985578957/torchpf.git

# or 

pip install torchpf
```

* Install and update using **setup.py** after cloning this repository.
```bash
$ python3 setup.py install
```

## A Simple Example
If you want to run the torchpf asap, you can call it as a CLI tool if your network exists in a script.
Otherwise you need to import torchpf as a module.

### CLI tool
```bash
$> torchpf -f example.py -m Net
      module name  input shape output shape     params memory(MB)           MAdd         Flops  MemRead(B)  MemWrite(B) duration[%]   MemR+W(B)
0           conv1    3 224 224   10 220 220      760.0       1.85   72,600,000.0  36,784,000.0    605152.0    1936000.0      57.49%   2541152.0
1           conv2   10 110 110   20 106 106     5020.0       0.86  112,360,000.0  56,404,720.0    504080.0     898880.0      26.62%   1402960.0
2      conv2_drop   20 106 106   20 106 106        0.0       0.86            0.0           0.0         0.0          0.0       4.09%         0.0
3             fc1        56180           50  2809050.0       0.00    5,617,950.0   2,809,000.0  11460920.0        200.0      11.58%  11461120.0
4             fc2           50           10      510.0       0.00          990.0         500.0      2240.0         40.0       0.22%      2280.0
total                                        2815340.0       3.56  190,578,940.0  95,998,220.0      2240.0         40.0     100.00%  15407512.0
===============================================================================================================================================
Total params: 2,815,340
-----------------------------------------------------------------------------------------------------------------------------------------------
Total memory: 3.56MB
Total MAdd: 190.58MMAdd
Total Flops: 96.0MFlops
Total MemR+W: 14.69MB
```

If you're not sure how to use a specific command, run the command with the -h or –help switches.
You'll see usage information and a list of options you can use with the command.

### Module
```python
from torchpf import show_stat
import torchvision.models as models

model = models.resnet18()
input_size=(3, 224, 224)
show_stat(model, input_size)
```

```python
from torchpf import cal_Flops, cal_MAdd, cal_Memory, cal_params
import torchvision.models as models

model = models.resnet18()
input_size=(3, 224, 224)
print('Flops = ',cal_Flops(model, input_size))
print('MAdd = ',cal_MAdd(model, input_size))
print('Memory = ',cal_Memory(model, input_size))
print('Params = ',cal_params(model, input_size))
```

## Features & TODO
**Note**: These features work only nn.Module. Modules in torch.nn.functional are not supported yet.
- [x] FLOPs
- [x] Number of Parameters
- [x] Total memory
- [x] Madd(FMA)
- [x] MemRead
- [x] MemWrite
- [ ] Model summary(detail, layer-wise)
- [ ] Export score table
- [ ] Arbitrary input shape

For the supported layers, check out [the details](./detail.md).


## Requirements
* Python 3.6+
* Pytorch 0.4.0+
* Pandas 0.23.4+
* NumPy 1.14.3+

## References
Thanks to @Swall0w for the initial version.
* [torchstat](https://github.com/Swall0w/torchstat)
* [flops-counter.pytorch](https://github.com/sovrasov/flops-counter.pytorch)
* [pytorch_model_summary](https://github.com/ceykmc/pytorch_model_summary)
* [chainer_computational_cost](https://github.com/belltailjp/chainer_computational_cost)
* [convnet-burden](https://github.com/albanie/convnet-burden).

import torch.nn as nn
from . import ModelHook
from collections import OrderedDict
from . import StatTree, StatNode, report_format
from .compute_memory import num_params


def get_parent_node(root_node, stat_node_name):
    assert isinstance(root_node, StatNode)

    node = root_node
    names = stat_node_name.split('.')
    for i in range(len(names) - 1):
        node_name = '.'.join(names[0:i+1])
        child_index = node.find_child_index(node_name)
        assert child_index != -1
        node = node.children[child_index]
    return node


def convert_leaf_modules_to_stat_tree(leaf_modules):
    assert isinstance(leaf_modules, OrderedDict)

    create_index = 1
    root_node = StatNode(name='root', parent=None)
    for leaf_module_name, leaf_module in leaf_modules.items():
        names = leaf_module_name.split('.')
        for i in range(len(names)):
            create_index += 1
            stat_node_name = '.'.join(names[0:i+1])
            parent_node = get_parent_node(root_node, stat_node_name)
            node = StatNode(name=stat_node_name, parent=parent_node)
            parent_node.add_child(node)
            if i == len(names) - 1:  # leaf module itself
                input_shape = leaf_module.input_shape.numpy().tolist()
                output_shape = leaf_module.output_shape.numpy().tolist()
                node.input_shape = input_shape
                node.output_shape = output_shape
                node.parameter_quantity = leaf_module.parameter_quantity.numpy()[
                    0]
                node.inference_memory = leaf_module.inference_memory.numpy()[0]
                node.MAdd = leaf_module.MAdd.numpy()[0]
                node.Flops = leaf_module.Flops.numpy()[0]
                node.duration = leaf_module.duration.numpy()[0]
                node.Memory = leaf_module.Memory.numpy().tolist()
    return StatTree(root_node)


class ModelStat(object):
    def __init__(self, model, input_size, query_granularity=1, DEBUG=False):
        assert isinstance(model, nn.Module)
        assert isinstance(input_size, (tuple, list)) and len(input_size) == 3
        self._model = model
        self._input_size = input_size
        self._query_granularity = query_granularity
        self.DEBUG = DEBUG

    def _analyze_model(self):
        model_hook = ModelHook(self._model, self._input_size, self.DEBUG)
        leaf_modules = model_hook.retrieve_leaf_modules()
        stat_tree = convert_leaf_modules_to_stat_tree(leaf_modules)
        collected_nodes = stat_tree.get_collected_stat_nodes(
            self._query_granularity)
        return collected_nodes

    def show_report(self):
        collected_nodes = self._analyze_model()
        report = report_format(collected_nodes)
        print(report)


def get_stat(model, input_size, query_granularity=1, DEBUG=False):
    return ModelStat(model, input_size, query_granularity, DEBUG=DEBUG)._analyze_model()


def show_stat(model, input_size, query_granularity=1, DEBUG=False):
    ms = ModelStat(model, input_size, query_granularity, DEBUG=DEBUG)
    ms.show_report()


def cal_Flops(model, input_size, clever_format=True, query_granularity=1, DEBUG=False):
    ms = ModelStat(model, input_size, query_granularity, DEBUG=DEBUG)
    analyze_data = ms._analyze_model()
    Flops = 0
    for i in range(len(analyze_data)):
        Flops += analyze_data[i].Flops
    if clever_format:
        if Flops > 1E9:
            return f'{Flops/1E9:.2f}G'
        elif Flops > 1E6:
            return f'{Flops/1E6:.2f}M'
        elif Flops > 1E3:
            return f'{Flops/1E3:.2f}K'
        else:
            return Flops
    else:
        return Flops


def cal_MAdd(model, input_size, clever_format=True, query_granularity=1, DEBUG=False):
    ms = ModelStat(model, input_size, query_granularity, DEBUG=DEBUG)
    analyze_data = ms._analyze_model()
    MAdd = 0
    for i in range(len(analyze_data)):
        MAdd += analyze_data[i].MAdd
    if clever_format:
        if MAdd > 1E9:
            return f'{MAdd/1E9:.2f}G'
        elif MAdd > 1E6:
            return f'{MAdd/1E6:.2f}M'
        elif MAdd > 1E3:
            return f'{MAdd/1E3:.2f}K'
        else:
            return MAdd
    else:
        return MAdd


def cal_Memory(model, input_size, clever_format=True, query_granularity=1, DEBUG=False):
    ms = ModelStat(model, input_size, query_granularity, DEBUG=DEBUG)
    analyze_data = ms._analyze_model()
    Memory = [0, 0]
    for i in range(len(analyze_data)):
        Memory[0] += analyze_data[i].Memory[0]
        Memory[1] += analyze_data[i].Memory[1]
    Memory = sum(Memory)
    if clever_format:
        if Memory > 1024**3:
            return f'{Memory/1024**3:.2f}G'
        elif Memory > 1024**2:
            return f'{Memory/1024**2:.2f}M'
        elif Memory > 1024:
            return f'{Memory/1024:.2f}K'
        else:
            return Memory
    else:
        return Memory


def cal_params(model, clever_format=True):
    params = num_params(model)
    if clever_format:
        if params > 1E9:
            return f'{params/1E9:.2f}G'
        elif params > 1E6:
            return f'{params/1E6:.2f}M'
        elif params > 1E3:
            return f'{params/1E3:.2f}K'
        else:
            return params
    else:
        return params

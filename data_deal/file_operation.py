# coding:utf-8
import os


def _operation_single_file(path, func, *args, **kwds):
    try:
        print('operation file path: %s' % path)
        func(path, *args, **kwds)
    except Exception as e:
        print('operation file %s encounter an error:%s' % (path, e))


def operation_file(path, func, filter_func=lambda x: True, *args, **kwds):
    if os.path.isdir(path):
        [_operation_single_file(path + '\\' + f_name, func, *args, **kwds)
         for f_name in os.listdir(path) if filter_func(f_name)]
    else:
        _operation_single_file(path, func, *args, **kwds)


def load_tuple_text_from_file(path):
    tuple_dict = {}
    with open(path, encoding='utf-8', mode='r') as data:
        for line in data.readlines():
            try:
                couple = line.strip().split('\t')
                if len(couple) == 2:
                    tuple_dict[couple[0]] = float(couple[1])
            except Exception as e:
                print('read file: %s encounter an error %s' % (path, e))
    return tuple_dict


def write_tuple_to_text(path, data):
    with open(path, encoding='utf-8', mode='w') as out:
        if isinstance(data, dict):
            [out.write(str(x) + '\t' + str(k) + '\n') for (x, k) in data.items()]
        elif isinstance(data, list):
            [out.write(str(x) + '\t' + str(k) + '\n') for (x, k) in data]


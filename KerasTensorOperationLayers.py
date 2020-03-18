# -*- coding: utf-8 -*-
"""
Created on Sat Sep  7 15:31:29 2019

@author: kINGHAREM
"""
from keras import backend as K
from keras.layers import Lambda


def Slice_tensor(tensor, position, slice_, name=None, fast_index=None,
                 keepdims=False):
    '''Slice dimension is position
       slice_ means the slice number
    '''
    if name is None:
        name = 'Slice'
        if fast_index is not None:
            name = name+str(fast_index)
    elif type(name) is not str:
        raise ValueError('argument <name> should be a string or "None".')
#    print('Example: Lambda(lambda x: x[:, :, 0:3], name=''<name>'')'
#          + '(''<tensor>'')')

    def slice_strided(x, **args):
        assert (len(args) == 2) | (len(args) == 3),\
            'need two argument: position and slice_ and one optional argument'
        if 'position' not in args:
            raise ValueError('need position argument')
        if 'slice_' not in args:
            raise ValueError('need slice_ argument')
        if 'keepdims' in args:
            keepdims = args['keepdims']
        else:
            keepdims = False
        position = args['position']
        slice_ = args['slice_']
        import numpy as np
        output_list = []

        assert np.mod(K.int_shape(x)[position], slice_) == 0,\
            'incorrect slice'\
            + 'number with the length in dimension'+str(position)+' is '\
            + str(x.shape[position])+' and the slice number is '\
            + str(slice_)

        stride = int(K.int_shape(x)[position] / slice_)

        if (keepdims is False) & (K.int_shape(x)[position] == slice_):
            if position == 0:
                for i in range(slice_):
                    output_list.append(x[i])
            if position == 1:
                for i in range(slice_):
                    output_list.append(x[:, i])
            if position == 2:
                for i in range(slice_):
                    output_list.append(x[:, : i])
            if position == 3:
                for i in range(slice_):
                    output_list.append(x[:, :, :, i])
            if position == 4:
                for i in range(slice_):
                    output_list.append(x[:, :, :, :, :, i])
            if position == -4:
                for i in range(slice_):
                    output_list.append(x.T[:, :, :, i]).T
            if position == -3:
                for i in range(slice_):
                    output_list.append(x.T[:, : i]).T
            if position == -2:
                for i in range(slice_):
                    output_list.append(x.T[:, i]).T
            if position == -1:
                for i in range(slice_):
                    output_list.append(x.T[i]).T
        else:
            if position == 0:
                for i in range(slice_):
                    output_list.append(x[i*stride:(i+1)*stride])
            elif position == 1:
                for i in range(slice_):
                    output_list.append(x[:, i*stride:(i+1)*stride])
            elif position == 2:
                for i in range(slice_):
                    output_list.append(x[:, :, i*stride:(i+1)*stride])
            elif position == 3:
                for i in range(slice_):
                    output_list.append(x[:, :, :, i*stride:(i+1)*stride])
            elif position == 4:
                for i in range(slice_):
                    output_list.append(x[:, :, :, :, i*stride:(i+1)*stride])
            elif position == -4:
                for i in range(slice_):
                    output_list.append(x.T[:, :, :, i*stride:(i+1)*stride].T)
            elif position == -3:
                for i in range(slice_):
                    output_list.append(x.T[:, :, i*stride:(i+1)*stride].T)
            elif position == -2:
                for i in range(slice_):
                    output_list.append(x.T[:, i*stride:(i+1)*stride].T)
            elif position == -1:
                for i in range(slice_):
                    output_list.append(x.T[i*stride:(i+1)*stride].T)
            else:
                raise ValueError('Sorry, we do not support the operation '
                                 + 'between'
                                 + 'tensor position abs larger than 4, now.')

        return output_list

    args = {'position': position, 'slice_': slice_, 'keepdims': keepdims}
    return Lambda(
            slice_strided, arguments=args,
            name='stride'+str(slice_)+name
            )(tensor)


def Expand_dims(tensor, position, name=None, fast_index=None):
    '''expand dimenion in <position>
    '''
    if name is None:
        name = 'Expand_dims'
        if fast_index is not None:
            name = name + str(fast_index)
    elif type(name) is not str:
        raise ValueError('argument <name> should be a string or "None".')
    if position > len(tensor.shape):
        raise ValueError('Dimension for expansion exceeded the sup')
    return Lambda(lambda x: K.expand_dims(x, position), name=name)(tensor)


def Concetanate_tensor(tensors, position, name=None, fast_index=None):
    '''concetenate tensor in the axis of <position>.
    '''
    if name is None:
        name = 'Concetanater'
        if fast_index is not None:
            name = name + str(fast_index)
    elif type(name) is not str:
        raise ValueError('argument <name> should be a string or "None".')

    def concetanator(tensors, **args):
        if 'axis' not in args:
            axis = -1
        else:
            axis = args['axis']
        if isinstance(tensors, list):
            tensors = tuple(tensors)
            return K.concatenate(tensors, axis=axis)
        else:
            raise ValueError('Concetanate_tensor should be fed a list of'
                             + 'tensors which are waiting to be concetanated.')
            return -1

    args = {'axis': position}
    return Lambda(
            concetanator, arguments=args,
            name=name)(tensors)


def Expand_concetanate_tensor(tensors, position,
                              name=None, fast_index=None):
    '''concetenate tensor in the axis of 1 (real and image channel axis).
    '''
    if name is None:
        name = 'Conceta_new_dim'
        if fast_index is not None:
            name = name + str(fast_index)
    elif type(name) is not str:
        raise ValueError('argument <name> should be a string or "None".')

    def concetanator_on_dimension(tensors, **args):
        if 'axis' not in args:
            axis = -1
        else:
            axis = args['axis']
        if isinstance(tensors, list):
            for i in range(len(tensors)):
                tensors[i] = K.expand_dims(tensors[i], axis=axis)
            tensors = tuple(tensors)
            return K.concatenate(tensors, axis=axis)
        else:
            raise ValueError('Concetanate_tensor should be fed a list of'
                             + 'tensors which are waiting to be concetanated.')

    args = {'axis': position}
    return Lambda(
            concetanator_on_dimension, arguments=args,
            name=name)(tensors)


def Dot(tensorx, tensory, model='dot',    # 'batch_dot'
        name=None, fast_index=None):
    '''matrices multiply betweeb tensors
    '''
    if name is None:
        if model == 'dot':
            name = 'Dot'
        elif model == 'batch_dot':
            name = 'Batch_Dot'
        if fast_index is not None:
            name = name + str(fast_index)
    elif type(name) is not str:
        raise ValueError('argument <name> should be a string or "None".')

    def dot_tensors(double_tensors):
        if isinstance(double_tensors, list) & (len(double_tensors) == 2):
            x = double_tensors[0]
            y = double_tensors[1]
        else:
            raise ValueError('dot_tensor should be fed a list of TWO tensors.')
        return K.dot(x, y)

    def batch_dot_tensors(double_tensors):
        if isinstance(double_tensors, list) & (len(double_tensors) == 2):
            x = double_tensors[0]
            y = double_tensors[1]
        else:
            raise ValueError('batch_dot_tensor should be fed a list of TWO'
                             + 'tensors.')
        return K.batch_dot(x, y)

    tensors = [tensorx, tensory]

    if model == 'dot':
        return Lambda(dot_tensors, name=name)(tensors)
    elif model == 'batch_dot':
        return Lambda(batch_dot_tensors, name=name)(tensors)
    else:
        raise ValueError('model must be "dot" or "batch_dot". ')


def matrix_wise_transpose(tensors, name=None, fast_index=None):
    '''matrix-wise transpose of a tensor
       e.g. <Tensor> a: (None, 2, 3, 4) --> <Tensor> b: (None, 2, 4, 3)
    '''
    if name is None:
        name = 'matrix_wise_T'
        if fast_index is not None:
            name = name + str(fast_index)
    elif type(name) is not str:
        raise ValueError('argument <name> should be a string or "None".')

    def transpose(tensors):
        original_shape = list(K.int_shape(tensors))
        assert len(original_shape) >= 3, 'shape of target tensor less than 3.'\
            + str(original_shape) + ' found'
        assert None not in original_shape[1:], 'dimension from 1 to last '\
            + 'should not contain <None>. ' + str(original_shape)
        if None is original_shape[0]:
            tmp = original_shape[1:-2]
            unchanged_shape = tuple(tmp)
            tmp.reverse()
            inversed_unchanged_shape = tuple(tmp)
            r = original_shape[-2]
            c = original_shape[-1]

            tensors = K.transpose(
                    K.reshape(
                            K.transpose(
                                    tensors),
                            (r*c, )+inversed_unchanged_shape+(-1, )
                            )
                    )
            return K.reshape(tensors, (-1, )+unchanged_shape+(c, r))
        else:
            tmp = original_shape[0:-2]
            unchanged_shape = tuple(tmp)
            tmp.reverse()
            inversed_unchanged_shape = tuple(tmp)
            r = original_shape[-2]
            c = original_shape[-1]

            tensors = K.transpose(
                    K.reshape(
                            K.transpose(
                                    tensors),
                            (r*c, )+inversed_unchanged_shape
                            )
                    )
            return K.reshape(tensors, unchanged_shape+(c, r))

    return Lambda(
            transpose, name=name)(tensors)

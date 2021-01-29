"""
Functions for converting and plotting training logs.
"""
# -*- coding: utf-8 -*-
import os
import json
from math import ceil
from warnings import warn
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patheffects as path_effects
from util.apples_dataset import get_apples_dataset

try:
    from tensorflow.core.util import event_pb2
    from tensorflow.python.lib.io import tf_record
    from tensorflow.errors import DataLossError
    TF_AVAILABLE = True
except ModuleNotFoundError:
    TF_AVAILABLE = False

def extract_tensorboard_scalars(log_dir=None, save_as_npz=None, tags=None):
    if not TF_AVAILABLE:
        raise RuntimeError('Tensorflow could not be imported, which is '
                           'required by `extract_tensorboard_scalars`')

    log_files = [f for f in os.listdir(log_dir)
                 if os.path.isfile(os.path.join(log_dir, f))]
    if len(log_files) == 0:
        raise FileNotFoundError('no file in log dir "{}"'.format(log_dir))
    elif len(log_files) > 1:
        warn('multiple files in log_dir "{}", choosing the one modified last'
             .format(log_dir))
        log_files.sort(
            key=lambda f: os.path.getmtime(os.path.join(log_dir, f)),
            reverse=True)
    log_file = os.path.join(log_dir, log_files[0])

    def my_summary_iterator(path):
        for r in tf_record.tf_record_iterator(path):
            yield event_pb2.Event.FromString(r)

    if tags is not None:
        tags = [t.replace('/', '_').lower() for t in tags]
    values = {}
    try:
        for event in tqdm(my_summary_iterator(log_file)):
            if event.WhichOneof('what') != 'summary':
                continue
            step = event.step
            for value in event.summary.value:
                if value.HasField('simple_value'):
                    tag = value.tag.replace('/', '_').lower()
                    if tags is None or tag in tags:
                        values.setdefault(tag, []).append((step,
                                                           value.simple_value))
    except DataLossError as e:
        warn('stopping for log_file "{}" due to DataLossError: {}'.format(
            log_file, e))
    scalars = {}
    for k in values.keys():
        v = np.asarray(values[k])
        steps, steps_counts = np.unique(v[:, 0], return_counts=True)
        scalars[k + '_steps'] = steps
        scalars[k + '_scalars'] = v[np.cumsum(steps_counts)-1, 1]  # last of
        #                                                            each step

    if save_as_npz is not None:
        np.savez(save_as_npz, **scalars)

    return scalars

def apples_convert_logs(tensorboard_log_path, npz_log_path, method,
                        noise_setting=None, num_angles=None,
                        convert_running_to_current=False,
                        hyper_params_path=None, batch_sizes=None):
    """
    Convert loss and psnr values from tensorboard log files to npz log files
    for experiments on the apple dataset.
    The tensorboard logs must have the same format as the logs of
    :meth:`StandardLearnedReconstructor.train` (in dival 0.5.7).

    Parameters
    ----------
    tensorboard_log_path : str
        Path containing the tensorboard log folders.
    npz_log_path : str
        Output path in which the npz files are stored.
    method : {``'learnedpd'``, ``'fbpunet'``, ``'fbpmsdnet'``}
        Method.
    noise_setting : str or sequence of str, optional
        Noise setting or list of noise settings.
        Valid noise settings are ``'noisefree'``, ``'gaussian_noise'`` and
        ``'scattering'``.
        If not specified, all noise settings are used.
    num_angles : int or sequence of int, optional
        Number of angles or list of numbers of angles.
        Valid angles are ``50``, ``10``, ``5`` and ``2``.
        The default is ``[50, 10, 5, 2]``.
    convert_running_to_current : bool, optional
        Whether to convert running log values (mean value since start of
        current epoch) to current log values (mean value of current batch).
        Note that this conversion often is inaccurate in the first epoch due to
        the fact that single precision was used for accumulating the log
        values, which is problematic due the strongly varying value range.
        The default is `False`.
    hyper_params_path : str, optional
        Path to hyper params files. Required when using
        ``convert_running_to_current=True`` and `batch_sizes` is not specified.
    batch_sizes : int or 4-sequence of int, optional
        Batch sizes for each element in `num_angles`.
        Required when using ``convert_running_to_current=True`` and
        `hyper_params_path` is not specified.
        If a single integer is specified, it is used for all numbers of angles.
        If batch sizes are specified via this field (as opposed to via
        `hyper_params_path`), the same value is used for all noise settings.
    """
    noise_setting_list = (
        ['gaussian_noise', 'scattering'] if noise_setting is None
        else (
            [noise_setting] if isinstance(noise_setting, str)
            else noise_setting))
    num_angles_list = (
        [50, 10, 5, 2] if num_angles is None
        else (
            [num_angles] if isinstance(num_angles, int)
            else num_angles))
    if batch_sizes is None:
        batch_sizes = (None,) * len(num_angles_list)
    elif isinstance(batch_sizes, int):
        batch_sizes = (batch_sizes,) * len(num_angles_list)
    if convert_running_to_current:
        assert None not in batch_sizes or hyper_params_path is not None, (
            'must specify either `batch_sizes` for all numbers of angles or '
            '`hyper_params_path` when using `convert_running_to_current=True`')
    tags = ('loss_train', 'psnr_train', 'loss_validation', 'psnr_validation')
    for n in noise_setting_list:
        for a, batch_size in zip(num_angles_list, batch_sizes):
            name = 'apples_{}_{:02d}_{}'.format(n, a, method)
            scalars = extract_tensorboard_scalars(
                os.path.join(tensorboard_log_path, name), tags=tags)
            if convert_running_to_current:
                warn('conversion from running loss values may be inaccurate '
                     'due to the floating point precision that was used when '
                     'summing the loss values!')
                if batch_size is None:
                    with open(os.path.join(hyper_params_path,
                                           name + '_hyper_params.json'),
                              'r') as f:
                        hyper_params = json.load(f)
                        batch_size = hyper_params['batch_size']
                dataset = get_apples_dataset(noise_setting=noise_setting,
                                             impl='skimage')
                num_samples_per_epoch = dataset.get_len('train')
                for k in ('loss', 'psnr'):
                    num_epochs = len(scalars['{}_validation_steps'.format(k)])
                    num_steps_per_epoch = ceil(
                        num_samples_per_epoch / batch_size)
                    assert (len(scalars['{}_train_steps'.format(k)])
                            // num_steps_per_epoch) == num_epochs
                    arr = scalars['{}_train_scalars'.format(k)]
                    for i in range(ceil(len(arr) / num_steps_per_epoch)):
                        # i in range(0, num_epochs) or range(0, num_epochs+1)
                        first = i * num_steps_per_epoch
                        n = (num_steps_per_epoch if i < num_epochs
                             else len(arr) - num_epochs * num_steps_per_epoch)
                        running_num_samples = np.arange(1, n+1) * batch_size
                        # last batch of each epoch can be shorter:
                        running_num_samples[-1] = min(
                            num_samples_per_epoch, running_num_samples[-1])
                        denormalized = (arr[first:first+n].astype(np.float64) *
                                        running_num_samples)
                        arr[first:first+n] = (
                            np.diff(denormalized, prepend=0)
                            / np.diff(running_num_samples, prepend=0))
            np.savez(os.path.join(npz_log_path, name), **scalars)

def apples_convert_logs_lightning(tensorboard_log_path, npz_log_path, method,
                                  noise_setting=None, num_angles=None,
                                  convert_current_to_running=False,
                                  hyper_params_path=None, batch_sizes=None,
                                  rename_fields=True):
    """
    Convert loss values from tensorboard log files to npz log files for
    experiments on the apple dataset.
    The tensorboard logs must have the same format as the logs of
    `util.cinn_reconstructor.CINNReconstructor`, which uses PyTorch Lightning.

    Parameters
    ----------
    tensorboard_log_path : str
        Path containing the tensorboard log folders.
    npz_log_path : str
        Output path in which the npz files are stored.
    method : {``'cinn'``}
        Method.
    noise_setting : str or sequence of str, optional
        Noise setting or list of noise settings.
        Valid noise settings are ``'noisefree'``, ``'gaussian_noise'`` and
        ``'scattering'``.
        If not specified, all noise settings are used.
    num_angles : int or sequence of int, optional
        Number of angles or list of numbers of angles.
        Valid angles are ``50``, ``10``, ``5`` and ``2``.
        The default is ``[50, 10, 5, 2]``.
    convert_current_to_running: bool, optional
        Whether to convert current log values (mean value of current batch)
        to running log values (mean value since start of current epoch).
        This option is not yet implemented.
        The default is `False`.
    hyper_params_path : str, optional
        Path to hyper params files. Required when using
        ``convert_current_to_running=True`` and `batch_sizes` is not specified.
    batch_sizes : int or 4-sequence of int, optional
        Batch sizes for each element in `num_angles`.
        Required when using ``convert_current_to_running=True`` and
        `hyper_params_path` is not specified.
        If a single integer is specified, it is used for all numbers of angles.
        If batch sizes are specified via this field (as opposed to via
        `hyper_params_path`), the same value is used for all noise settings.
    rename_fields : bool, optional
        Whether to rename fields ``'train_loss_*'`` and ``'val_loss_*'``
        to ``'loss_train_*'`` and ``'loss_validation_*'`` in order
        to match the names for logs from
        :meth:`StandardLearnedReconstructor.train` (in dival 0.5.7).
        The default is `True`.
    """
    noise_setting_list = (
        ['gaussian_noise', 'scattering'] if noise_setting is None
        else (
            [noise_setting] if isinstance(noise_setting, str)
            else noise_setting))
    num_angles_list = (
        [50, 10, 5, 2] if num_angles is None
        else (
            [num_angles] if isinstance(num_angles, int)
            else num_angles))
    if batch_sizes is None:
        batch_sizes = (None,) * len(num_angles_list)
    elif isinstance(batch_sizes, int):
        batch_sizes = (batch_sizes,) * len(num_angles_list)
    tags = ('train_loss', 'val_loss')
    tags_renamed = ('loss_train', 'loss_validation')
    for n in noise_setting_list:
        for a, batch_size in zip(num_angles_list, batch_sizes):
            name = 'apples_{}_{:02d}_{}'.format(n, a, method)
            scalars = extract_tensorboard_scalars(
                os.path.join(tensorboard_log_path, name),
                tags=tags + ('epoch',))
            if convert_current_to_running:
                print()
                print(n)
                print(a)
                print(scalars['val_loss_steps'][0]*3 / (44647 if noise_setting != 'scattering' else 5280))
                print()
                for val_step_prev, val_step in zip(
                        np.concatenate(
                            ([-1], scalars['val_loss_steps'])),
                        np.concatenate(
                            (scalars['val_loss_steps'], [np.inf]))):
                    if np.isfinite(val_step):
                        # assert that val_step marks the last step in an epoch
                        idx = np.where(scalars['epoch_steps']==val_step)[0][0]
                        assert np.diff(scalars['epoch_scalars'][idx:idx+2])==1
                    epoch_mask = np.logical_and(
                        scalars['train_loss_steps'] > val_step_prev,
                        scalars['train_loss_steps'] <= val_step)
                    scalars['train_loss_scalars'][epoch_mask] = (
                        np.cumsum(scalars['train_loss_scalars'][epoch_mask]) /
                                  np.arange(1, np.count_nonzero(epoch_mask)+1))
                    val_step_prev = val_step
            if rename_fields:
                for tag, tag_renamed in zip(tags, tags_renamed):
                    scalars[tag_renamed+'_steps'] = scalars[tag+'_steps']
                    scalars[tag_renamed+'_scalars'] = scalars[tag+'_scalars']
                    del scalars[tag+'_steps']
                    del scalars[tag+'_scalars']
            np.savez(os.path.join(npz_log_path, name), **scalars)

def plot_logs(npz_log_path, method, noise_setting, plot='loss',
              hyper_params_path=None, batch_sizes=None, loss_scale_log=True,
              ylim=None, **kwargs):
    """
    Plot validation and training loss curves from npz log files.

    Parameters
    ----------
    npz_log_path : str
        Path to the npz log files.
        Cf. :func:`apples_convert_logs` for conversion from tensorboard logs.
    method : str
        Method.
    noise_setting : str
        Noise setting.
    plot : {``'loss'``, ``'psnr'``}
        Plot type. The default is ``'loss'``.
    hyper_params_path : str, optional
        Path to hyper params files. Required if `batch_sizes` is not specified.
        If `batch_sizes` is specified, this parameter will be ignored. 
    batch_sizes : int or 4-sequence of int, optional
        Batch sizes for 50, 10, 5 and 2 angles, respectively.
        Required if `hyper_params_path` is not specified.
        If a single integer is specified, it is used for all numbers of angles.
    loss_scale_log : bool, optional
        Whether to use a logarithmic y-axis in case of ``plot=='loss'``.
        The default is `True`.
    ylim : 2-sequence of float, optional
        y-axis limits passed to :meth:`ax.set_ylim`.
    **kwargs : dict
        Keyword arguments passed to :func:`plt.subplots`.

    Returns
    -------
    ax : :class:`matplotlib.axes.Axes`
        The matplotlib axes.
    """
    num_angles_list = [50, 10, 5, 2]
    dataset = get_apples_dataset(noise_setting=noise_setting, impl='skimage')
    num_samples_per_epoch = dataset.get_len('train')
    fig, ax = plt.subplots(**kwargs)
    xs_train = []
    ys_train = []
    xs_validation = []
    ys_validation = []
    if batch_sizes is None:
        batch_sizes = (None,) * len(num_angles_list)
    elif isinstance(batch_sizes, int):
        batch_sizes = (batch_sizes,) * len(num_angles_list)
    assert None not in batch_sizes or hyper_params_path is not None, (
        'must specify either `batch_sizes` for all numbers of angles or '
        '`hyper_params_path`')
    for a, batch_size in zip(num_angles_list, batch_sizes):
        name = 'apples_{}_{:02d}_{}'.format(noise_setting, a, method)
        if batch_size is None:
            with open(os.path.join(hyper_params_path,
                                   name + '_hyper_params.json'), 'r') as f:
                hyper_params = json.load(f)
                batch_size = hyper_params['batch_size']
        scalars = np.load(os.path.join(npz_log_path, name + '.npz'))
        steps_train = scalars['{}_train_steps'.format(plot)]
        epochs_train = steps_train * batch_size / num_samples_per_epoch
        values_train = scalars['{}_train_scalars'.format(plot)]
        steps_validation = scalars['{}_validation_steps'.format(plot)]
        epochs_validation = (steps_validation * batch_size /
                             num_samples_per_epoch)
        values_validation = scalars['{}_validation_scalars'.format(plot)]
        xs_train.append(epochs_train)
        ys_train.append(values_train)
        xs_validation.append(epochs_validation)
        ys_validation.append(values_validation)
    for a, x_train, y_train, x_validation, y_validation in zip(
            num_angles_list,
            xs_train, ys_train, xs_validation, ys_validation):
        y_train[x_train > np.max(x_validation)] = np.nan  # hide additional
        line_train = ax.plot(x_train, y_train, alpha=0.5)[0]
        ax.plot(x_validation, y_validation,
                color=line_train.get_color(), linestyle='--', zorder=3,
                path_effects=[
                    path_effects.withStroke(offset=(0., -0.25), linewidth=3,
                                            foreground='w', alpha=1.)],
                label='{:02d} angles'.format(a))
    if plot == 'loss' and loss_scale_log:
        ax.set_yscale('log')
    ax.set_xlim(0, None)
    ax.xaxis.set_major_locator(ticker.MaxNLocator(
        integer=True, steps=[1, 2, 4, 5, 10]))
    if plot == 'psnr':
        if ylim is None and ax.get_ylim()[0] < 0.:
            ylim = (0, None)
    ax.set_ylim(ylim)
    ax.set_xlabel('Epochs')
    ax.set_ylabel('PSNR (dB)' if plot == 'psnr' else 'Loss')
    handles, labels = ax.get_legend_handles_labels()
    if plot == 'loss':
        handles = reversed(handles)
        labels = reversed(labels)
    ax.legend(handles=handles, labels=labels,
              loc=('lower right' if plot == 'psnr' else 'upper right'))
    return ax

import csv
import datetime
from collections import defaultdict

import numpy as np
import torch
from termcolor import colored
from torch.utils.tensorboard import SummaryWriter

COMMON_TRAIN_FORMAT = [
    ('frame', 'F', 'int'),
    ('step', 'S', 'int'),
    ('episode', 'E', 'int'),
    ('episode_length', 'L', 'int'),
    ('episode_reward', 'R', 'float'),
    ('buffer_size', 'BS', 'int'),
    ('fps', 'FPS', 'float'),
    ('total_time', 'T', 'time')
]

COMMON_EVAL_FORMAT = [
    ('frame', 'F', 'int'),
    ('step', 'S', 'int'),
    ('episode', 'E', 'int'),
    ('episode_length', 'L', 'int'),
    ('episode_reward', 'R', 'float'),
    ('episode_success', 'R', 'float'),
    ('total_time', 'T', 'time')
]


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MetersGroup(object):
    def __init__(self, csv_file_name, formating):
        self._csv_file_name = csv_file_name
        self._formating = formating
        self._meters = defaultdict(AverageMeter)
        self._csv_file = None
        self._csv_writer = None

    def log(self, key, value, n=1):
        self._meters[key].update(value, n)

    def _prime_meters(self):
        data = {}
        for key, meter in self._meters.items():
            # Strip the prefix (e.g., 'train/' or 'eval/') if present
            if key.startswith('train'):
                new_key = key[len('train') + 1:]
            elif key.startswith('actor'):
                new_key = key[len('actor') + 1:]
            elif key.startswith('critic'):
                new_key = key[len('critic') + 1:]
            elif key.startswith('pretrain'):
                new_key = key[len('pretrain') + 1:]
            else:  # Assuming eval
                new_key = key[len('eval') + 1:]
            new_key = new_key.replace('/', '_')
            data[new_key] = meter.value()
        return data

    def _remove_old_entries(self, data):
        rows = []
        with self._csv_file_name.open('r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                if float(row['episode']) >= data['episode']:
                    break
                rows.append(row)
        with self._csv_file_name.open('w') as f:
            writer = csv.DictWriter(f,
                                    fieldnames=sorted(data.keys()),
                                    restval=0.0)
            writer.writeheader()
            for row in rows:
                writer.writerow(row)

    def _dump_to_csv(self, data):
        if self._csv_writer is None:
            should_write_header = True
            if self._csv_file_name.exists():
                self._remove_old_entries(data)
                should_write_header = False

            self._csv_file = self._csv_file_name.open('a')
            self._csv_writer = csv.DictWriter(self._csv_file,
                                              fieldnames=sorted(data.keys()),
                                              restval=0.0)
            if should_write_header:
                self._csv_writer.writeheader()

        self._csv_writer.writerow(data)
        self._csv_file.flush()

    def _format(self, key, value, ty):
        if ty == 'int':
            value = int(value)
            return f'{key}: {value}'
        elif ty == 'float':
            return f'{key}: {value:.04f}'
        elif ty == 'time':
            value = str(datetime.timedelta(seconds=int(value)))
            return f'{key}: {value}'
        else:
            raise ValueError(f'invalid format type: {ty}')

    def _dump_to_console(self, data, prefix):
        color = 'yellow' if prefix == 'train' else 'green'
        prefix_disp = colored(prefix, color)
        pieces = [f'| {prefix_disp: <14}']
        for key, disp_key, ty in self._formating:
            value = data.get(key, 0)
            pieces.append(self._format(disp_key, value, ty))
        print(' | '.join(pieces))

    def dump(self, step, prefix):
        if len(self._meters) == 0:
            return
        data = self._prime_meters()
        data['frame'] = step
        self._dump_to_csv(data)
        self._dump_to_console(data, prefix)
        self._meters.clear()


class Logger(object):
    def __init__(self, log_dir, use_tb):
        self._log_dir = log_dir

        self._pretrain_mg = MetersGroup(log_dir / 'pretrain.csv',
                                        formating=COMMON_TRAIN_FORMAT)
        self._train_mg    = MetersGroup(log_dir / 'train.csv',
                                        formating=COMMON_TRAIN_FORMAT)
        self._actor_mg    = MetersGroup(log_dir / 'actor.csv',
                                        formating=COMMON_TRAIN_FORMAT)
        self._critic_mg   = MetersGroup(log_dir / 'critic.csv',
                                        formating=COMMON_TRAIN_FORMAT)
        self._eval_mg     = MetersGroup(log_dir / 'eval.csv',
                                        formating=COMMON_EVAL_FORMAT)

        if use_tb:
            self._sw = SummaryWriter(str(log_dir / 'tb'))
        else:
            self._sw = None

    def _try_sw_log(self, key, value, step):
        if self._sw is not None:
            self._sw.add_scalar(key, value, step)

    def log(self, key, value, step):
        # Expect keys like 'train/episode_reward' or 'eval/episode_success', etc.
        assert key.startswith('train') or key.startswith('actor') \
            or key.startswith('critic') or key.startswith('eval') \
            or key.startswith('pretrain')

        if isinstance(value, torch.Tensor):
            value = value.item()
        self._try_sw_log(key, value, step)

        if key.startswith('train'):
            mg = self._train_mg
        elif key.startswith('actor'):
            mg = self._actor_mg
        elif key.startswith('critic'):
            mg = self._critic_mg
        elif key.startswith('eval'):
            mg = self._eval_mg
        else:  # pretrain
            mg = self._pretrain_mg

        mg.log(key, value)

    def log_metrics(self, metrics, step, ty):
        for key, value in metrics.items():
            self.log(f'{ty}/{key}', value, step)

    def dump(self, step, ty=None):
        if ty == 'train':
            self._train_mg.dump(step, 'train')
        if ty is None or ty == 'eval':
            self._eval_mg.dump(step, 'eval')
        if ty == 'critic':
            self._critic_mg.dump(step, 'critic')
        if ty == 'actor':
            self._actor_mg.dump(step, 'actor')
        if ty == 'pretrain':
            self._pretrain_mg.dump(step, 'pretrain')

    def log_and_dump_ctx(self, step, ty):
        return LogAndDumpCtx(self, step, ty)


class LogAndDumpCtx:
    def __init__(self, logger, step, ty):
        self._logger = logger
        self._step = step
        self._ty = ty

    def __enter__(self):
        return self

    def __call__(self, key, value):
        self._logger.log(f'{self._ty}/{key}', value, self._step)

    def __exit__(self, *args):
        self._logger.dump(self._step, self._ty)

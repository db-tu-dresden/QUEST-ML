import logging
from schema import Schema, And, Use, SchemaError, Or
import yaml


def validate_yaml(data: dict, schema: Schema):
    try:
        return schema.validate(data)
    except SchemaError as e:
        logging.exception(e)


class Config:

    default_schema = Schema({
        'on_gpu': Use(bool),
        'epochs': And(Use(int), lambda x: x > 0),
        'learning_rate': And(Use(float), lambda x: x > 0),
        'momentum': And(Use(float), lambda x: x > 0),
        'batch_size': And(Use(int), lambda x: x > 0),

        'job_id': Use(int),

        'num_workers_dataloader': Use(int),
        'pin_memory': Use(bool),
        'set_gradients_none': Use(bool),
        'fp16': Use(bool),
        'allow_tf32': Use(bool),

        'master_addr': Or(None, And(Use(str), lambda x: len(x) > 0)),
        'master_port': Or(None, Use(int)),
        'device': Or(None, Use(int)),

        'min_checkpoint_epoch': Use(int),
        'min_checkpoint_epoch_dist': Use(int),

        'root_dir': Use(str),
        'checkpoint_path': Use(str),
        'model_save_path': Use(str),
    })

    def __init__(self, path: str, schema: Schema = default_schema):
        self.path = path
        self.schema = schema

        with open(self.path) as f:
            self.data = yaml.full_load(f)

        self.validate()

    def validate(self):
        self.data = validate_yaml(self.data, self.schema)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(path={self.path!r}, schema={self.schema!r})'

    def __str__(self):
        return str({
            'data': self.data,
            'schema': self.schema,
        })

    def __getitem__(self, item):
        return self.data[item]

    def __setitem__(self, key, value):
        self.data[key] = value

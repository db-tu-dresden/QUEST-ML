import logging
from schema import Schema, And, Use, Optional, SchemaError
import yaml


def validate_yaml(data: dict, schema: Schema):
    try:
        return schema.validate(data)
    except SchemaError as e:
        logging.exception(e)


class ConfigSchema(Schema):
    def validate(self, data, _is_config_schema=True):
        data = super(ConfigSchema, self).validate(data, _is_config_schema=False)
        if _is_config_schema and data['processCount'] < data['length']:
            raise SchemaError('Process count can not be smaller than the length of the shortest path!')
        return data


default_schema = ConfigSchema({
    'jobCount': And(Use(int), lambda x: x > 0),
    Optional('branchingFactor', default={'min': 1, 'max': 1}): {
        Optional('min', default=1): And(Use(int), lambda x: x > 0),
        Optional('max', default=1): And(Use(int), lambda x: x > 0),
    },
    Optional('length'): And(Use(int), lambda x: x > 0),
    Optional('processCount'): And(Use(int), lambda x: x > 0),
})


class Config:
    def __init__(self, path: str, schema: ConfigSchema = default_schema):
        self.schema = schema

        with open(path) as f:
            self.data = yaml.full_load(f)

        self.validate()

    def validate(self):
        self.data = validate_yaml(self.data, self.schema)

    def __repr__(self):
        return str({
            'data': self.data,
            'schema': self.schema,
        })


def main():
    c = Config(path='config.yaml')
    print(c)


if __name__ == '__main__':
    main()

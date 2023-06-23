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
        Or('jobLimit', 'until', only_one=True): And(Use(int), lambda x: x > 0),
    })

    def __init__(self, path: str, schema: Schema = default_schema):
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

    def __getitem__(self, item):
        return self.data[item]


def main():
    c = Config(path='config.yaml')
    print(c)


if __name__ == '__main__':
    main()
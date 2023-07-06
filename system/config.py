import logging
from schema import Schema, And, Use, SchemaError, Or
import yaml


def validate_yaml(data: dict, schema: Schema):
    try:
        return schema.validate(data)
    except SchemaError as e:
        logging.exception(e)


class JobSchema(Schema):
    def validate(self, data, _is_job_schema=True):
        data = super(JobSchema, self).validate(data, _is_job_schema=False)
        if _is_job_schema and len(data):
            probs = [elem['arrivalProbability'] for elem in data]
            if sum(probs) != 1.0:
                raise SchemaError(f'Job arrival probabilities are not a probability distribution.'
                                  f'Given probabilities are: {probs}')
        return data


class Config:

    default_schema = Schema({
        'until': And(Use(int), lambda x: x > 0),
        'loggingRate': And(Use(float), lambda x: x > 0),
        'randomSeed': object,
        'jobs': JobSchema([{
            'name': And(str, len),
            'arrivalProbability': And(Use(float), lambda x: 0 <= x <= 1),
            'failureRate': And(Use(float), lambda x: 0 <= x <= 1)
        }]),
        'arrivalProcess': {
            'beta': And(Use(float), lambda x: x > 0),
        },
        'processes': {
            'mean': Use(float),
            'std': And(Use(float), lambda x: x > 0),
        }
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

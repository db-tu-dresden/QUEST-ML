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
        'jobs': JobSchema([{
            'name': And(str, len),
            'arrivalProbability': And(Use(float), lambda x: 0 <= x <= 1),
            'failureRate': And(Use(float), lambda x: 0 <= x <= 1)
        }]),
        'loggingRate': And(Use(float), lambda x: x > 0),
    })

    def __init__(self, path: str, schema: Schema = default_schema):
        self.schema = schema

        with open(path) as f:
            self.data = yaml.full_load(f)

        self.validate()

    def validate(self):
        self.data = validate_yaml(self.data, self.schema)

    def __repr__(self):
        cls = self.__class__.__name__
        return f'{cls}(path={self.path!r}, schema={self.schema!r})'

    def __getitem__(self, item):
        return self.data[item]


def main():
    c = Config(path='../config.yaml')
    print(c)


if __name__ == '__main__':
    main()

import os

CERTIFICATES_DIRECTORY = os.path.dirname(__file__)


def get_redis_config():
    return {'username': 'public',
            'password': 'public',
            'host': '185.212.80.70',
            'port': 6381,
            'ssl_ca_certs': os.path.join(CERTIFICATES_DIRECTORY, 'certificates/ca.crt'),
            'ssl': True,
            'decode_responses': True
            }


if __name__ == '__main__':
    print(get_redis_config())

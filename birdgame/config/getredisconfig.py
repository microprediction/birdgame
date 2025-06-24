import importlib.resources


def get_redis_config():
    ca_crt = importlib.resources.read_text(__package__ + ".certificates", 'ca.crt')

    return {
        'username': 'public',
        'password': 'public',
        'host': '185.212.80.70',
        'port': 6381,
        'ssl_ca_data': ca_crt,
        'ssl': True,
        'decode_responses': True,
        "ssl_cert_reqs": "none"
    }


if __name__ == '__main__':
    print(get_redis_config())

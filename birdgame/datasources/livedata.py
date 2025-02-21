import redis
from birdgame.config.getredisconfig import get_redis_config
import orjson
import time
import logging

BIRD_PAYLOAD_NAME = 'bird_payload'

bird_logger = logging.getLogger(__name__)


def safe_decode(value):
    return value.decode() if isinstance(value, bytes) else value


def live_data_generator(start_from_latest=True):
    """
       Generate data from the bird game. by default this will read from the start of the stream and then transition to live play.

        {'time': 96358382, 'falcon_location': 9458.498520155423, 'dove_location': 9458.502255534888, 'falcon_id': 11}
        {'time': 96358382, 'falcon_location': 9458.517620971368, 'dove_location': 9458.502757986242, 'falcon_id': 0}
        {'time': 96358382, 'falcon_location': 9458.45873633578, 'dove_location': 9458.502757986242, 'falcon_id': 2}


    :return:
    """
    config = get_redis_config()
    client = redis.Redis(**config)
    stream_name = 'prod_bird_game_public'

    # Determine the last existing message in the stream
    last_id = "$" if start_from_latest else "0-0"

    retries = 0
    max_retries = 5  # Limit the maximum retries to avoid infinite loops

    while True:
        try:
            # Fetch messages from Redis stream
            messages = client.xread(
                streams={stream_name: last_id},
                count=100,
                block=5000,
            )
            retries = 0  # Reset retry counter on success

            if messages:
                for _stream_name, msgs in messages:
                    stream_name_str = (
                        _stream_name.decode()
                        if isinstance(_stream_name, bytes)
                        else _stream_name
                    )

                    for redis_id, msg_data in msgs:
                        last_id = redis_id  # Update last ID

                        payload_str = msg_data.get(BIRD_PAYLOAD_NAME)
                        if payload_str is None:
                            raise ValueError(f"Payload '{BIRD_PAYLOAD_NAME}' not found in msg_data")

                        # Decode payload if in bytes
                        payload_str = safe_decode(payload_str)

                        try:
                            data = orjson.loads(payload_str)
                        except orjson.JSONDecodeError as e:
                            raise ValueError(f"Failed to decode payload '{BIRD_PAYLOAD_NAME}': {e}")


                        key_order = ['time', 'falcon_location', 'dove_location', 'falcon_id']
                        data = {k: data[k] for k in key_order}

                        yield data

        except (ConnectionError, TimeoutError) as e:
            bird_logger.error(f"Redis connection error: {e}")
            retries += 1
            sleep_time = min(
                2 ** retries, 60
            )  # Exponential backoff up to 60 sec
            bird_logger.info(f"Retrying after {sleep_time} seconds...")
            time.sleep(sleep_time)

            if retries > max_retries:
                bird_logger.error(
                    "Max retries reached, exiting fetch loop."
                )
                break
            client = redis.Redis(**config)
        except Exception as e:
            bird_logger.exception(
                "Unexpected error while reading from Redis"
            )
            time.sleep(1)  # Sleep to prevent excessive retries


if __name__ == '__main__':
    gen = live_data_generator()
    for _ in range(3):
        print(next(gen))

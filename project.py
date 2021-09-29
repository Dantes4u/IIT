import os
import yaml
import shutil
import argparse
import requests

from distutils.dir_util import copy_tree
from train_model import train_model


def telegram_bot_sendtext(bot_message):
   bot_token = '1522593977:AAHzsPx4BD65vxR6qjP9oJ83V-X4uGpuphw'
   bot_chatID = '76724614'
   send_text = 'https://api.telegram.org/bot' + bot_token + '/sendMessage?chat_id=' + bot_chatID + '&parse_mode=Markdown&text=' + bot_message
   response = requests.get(send_text)
   return response.json()


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config-name', type=str, default='train.yaml',
        help='Configuration file name'
    )
    parser.add_argument('--local_rank', default=os.getenv('LOCAL_RANK', 0), type=int,
                        help='Used for multi-process training. Can either be manually set ' +
                             'or automatically set by using "python -m multiproc".')
    return parser.parse_args()


def run_project():
    args = parse_args()
    with open(os.path.join('config', args.config_name), 'r') as input_file:
        settings = yaml.safe_load(input_file)
    gpu = int(os.environ.get('LOCAL_RANK', args.local_rank))

    output_dir = settings['Logs_dir']
    train_config = settings['Configuration']
    output_code_dir = os.path.join(output_dir, 'Code')
    if gpu == 0:
        os.makedirs(output_code_dir, exist_ok=True)

        if os.path.exists(output_dir):
            shutil.rmtree(output_dir)

        for directory in settings['Directories']:
            copy_tree(directory, os.path.join(output_code_dir, directory))

        for filename in settings['Filenames']:
            shutil.copy(filename, output_code_dir)

    kwargs = {'logs_dir': output_dir, 'config_name': train_config, 'gpu': gpu}
    if settings['Type'] == 'train':
        train_model(**kwargs)
    telegram_bot_sendtext(f"Обучение закончилось! Результат ищи тут: {output_dir}")


if __name__ == '__main__':
    run_project()

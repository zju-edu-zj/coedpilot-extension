# Prepare server with just 'one click'.
#
# Author: gongty [at] tongji [dot] edu [dot] cn
# Created on: 2024.11.26 at Minhang, Shanghai

# Usage: python ./init-server.py [lang]
# Example: python3 ./init-server.py typescript


import sys
import os
import git
import requests
import shutil
import tqdm

# os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'

SUPPORTED_LANG = [
    'go', 'java', 'javascript', 'python', 'typescript'
]


HUGGING_FACE_CO_PREFIX_CANDIDATES = {
    'hugging-face': 'https://huggingface.co',
    'hf-mirror': 'https://hf-mirror.com'
}


HUGGING_FACE_CO_PREFIX = HUGGING_FACE_CO_PREFIX_CANDIDATES['hugging-face']


def usage():
    print(f'Usage: {sys.argv[0]} [lang]')
    print('  Supported languages:')
    for it in SUPPORTED_LANG:
        print(f'  + {it}')


def prepare_dir(model_dir: str):
    for it in SUPPORTED_LANG:
        os.makedirs(f'{model_dir}/{it}', exist_ok=True)


def download_file(url: str, dest: str):
    """下载文件并显示进度条，支持断点续传"""
    # 检查文件是否已经存在并获取其大小
    if os.path.exists(dest):
        existing_file_size = os.path.getsize(dest)
    else:
        existing_file_size = 0

    headers = {"Range": f"bytes={existing_file_size}-"}
    response = requests.get(url, headers=headers, stream=True)

    # 获取总大小并调整为剩余大小
    total_size = int(response.headers.get('content-length', 0)) + existing_file_size

    with open(dest, 'ab') as file:  # 使用 'ab' 模式以追加到文件
        with tqdm.tqdm(
            desc=dest,
            total=total_size,
            initial=existing_file_size,
            unit='B',
            unit_scale=True,
            unit_divisor=1024,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                file.write(data)
                bar.update(len(data))

    # 检查文件大小是否与预期一致
    if total_size != 0 and total_size != os.path.getsize(dest):
        print(f"ERROR, something went wrong")
    return 0

# def download_file(url: str, target: str) -> int:
#     """
#     Download a single file from `url`, save to file `target`.
#     """

#     response = requests.get(url)
#     if response.status_code != 200:
#         print(f"Failed to download: {url}")
#         return 1
#     with open(target, 'wb') as f:
#         f.write(response.content)

#     return 0


def clone_dependency_analyzer(model_dir: str):
    dependency_analyzer_dir = f'{model_dir}/dependency-analyzer'
    ready_file = f'{dependency_analyzer_dir}/.__dependency-analyzer-ready'

    if os.path.exists(ready_file):
        print(
            f"Dependency analyzer already cloned. Re-clone by removing \'{dependency_analyzer_dir}\' and run this script again.")
        return

    if os.path.exists(dependency_analyzer_dir):
        shutil.rmtree(dependency_analyzer_dir)

    print('Cloning dependency-analyzer... ')
    git.Repo.clone_from(
        url=f'{HUGGING_FACE_CO_PREFIX}/code-philia/dependency-analyzer',
        to_path=dependency_analyzer_dir,
    )
    print('Finished cloning dependency-analyzer.')
    with open(ready_file, 'w') as f:
        f.write('This file marks that dependency-analyzer is successfully cloned.')


def download(model_dir: str, lang: str) -> int:
    lang_model_dir = f'{model_dir}/{lang}'

    clone_dependency_analyzer(model_dir)

    print(f'Cloning models for \'{lang}\'... ')
    file_locator_model_base_url = f'{HUGGING_FACE_CO_PREFIX}/code-philia/CoEdPilot-file-locator/resolve/main/{lang}'
    line_locator_model_base_url = f'{HUGGING_FACE_CO_PREFIX}/code-philia/CoEdPilot-line-locator/resolve/main/{lang}'
    generator_model_base_url = f'{HUGGING_FACE_CO_PREFIX}/code-philia/CoEdPilot-generator/resolve/main/{lang}'

    download_list = [
        [f'{file_locator_model_base_url}/embedding_model.bin', 'embedding_model.bin'],
        [f'{file_locator_model_base_url}/reg_model.pickle', 'reg_model.pickle'],
        [f'{line_locator_model_base_url}/checkpoint-best-bleu/pytorch_model.bin',
            'locator_model.bin'],
        [f'{generator_model_base_url}/checkpoint-best-bleu/pytorch_model.bin',
            'generator_model.bin'],
    ]
    for it in download_list:
        res = download_file(
            url=it[0],
            dest=f'{lang_model_dir}/{it[1]}'
        )
        if res != 0:
            return 2
        print(f'{it[1]} downloaded.')

    print(f'All models for {lang} is ready.')

    return 0


def main() -> int:
    if (len(sys.argv) < 2):
        usage()
        return 1

    lang = sys.argv[1]
    if lang not in SUPPORTED_LANG:
        print(f'Language \'{lang}\' is not supported.')
        usage()
        return 2

    executable_path = os.path.abspath(sys.argv[0])
    src_dir, _ = os.path.split(executable_path)
    model_dir = f'{src_dir}/models'
    prepare_dir(model_dir)
    res = download(model_dir, lang)
    if res != 0:
        return res

    print("Server ready. Run `python3 src/model_server/server.py` and enjoy! :D")

    return 0


if __name__ == '__main__':
    res = main()
    exit(res)

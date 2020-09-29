import sys
import re
import tqdm

from pathlib import Path


if __name__ == '__main__':
    if len(sys.argv) != 2:
        print('Usage: python tokenize_file.py path_to_text_file')
        exit(1)
    filepath = Path(sys.argv[1])
    output = filepath.parent / (filepath.name.split('.')[0] + '_tokenized.txt')
    print(f'writing to: {output}')
    tokenize_reg = re.compile(r"[\w']+|[^\w ]")
    with filepath.open('r', encoding='utf8') as fin:
        with output.open('w', encoding='utf8') as fout:
            for line in tqdm.tqdm(fin):
                fout.write(' '.join(tokenize_reg.findall(line)).strip() + '\n')
    print('finished')

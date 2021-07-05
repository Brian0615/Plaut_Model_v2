from common.helpers import get_graphemes, get_phonemes

import pandas as pd
from tqdm import tqdm


class LensRepresentation:
    def __init__(self):
        self.df = None

    def create_lens_file(self, filepath):
        self.df = pd.read_csv(filepath, na_filter=False, dtype={'freq': float, 'log_freq': float})
        text = []
        for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
            for col in self.df.columns:
                text.append(f"name: {row[col]}")
                text.append(f"I: {' '.join([str(x) for x in get_graphemes(row[col])])}")
                text.append(';')
        # for i, row in tqdm(self.df.iterrows(), total=len(self.df)):
        #     text.append(f"name: {row['orth']}")
        #     text.append(f"freq: {row['log_freq']}")
        #     text.append(f"I: {' '.join([str(x) for x in get_graphemes(row['orth'])])}")
        #     text.append(f"T: {' '.join([str(x) for x in get_phonemes(row['phon'])])}")
        #     text.append(';')

        with open('lens.txt', 'w') as f:
            f.write('\n'.join(text))

    def parse_lens_results(self, filepath):
        if 'hidden.txt' in filepath:
            return self.__parse_lens_hidden(filepath)
        elif 'output.txt' in filepath:
            return self.__parse_lens_output(filepath)
        else:
            raise ValueError('Unable to parse given data')

    def __parse_lens_hidden(self, filepath):
        print(f"Loading lens hidden activations from {filepath}")

        with open(filepath, 'r') as f:
            lines = f.readlines()

        # parse data from txt file
        data = [x.strip().split('|') for x in lines]
        data = pd.DataFrame(data, columns=['epoch', 'word', 'hidden'])

        print(f"Processing lens hidden activations...")
        # convert hidden (list of str) to list of float
        data['hidden'] = data['hidden'].apply(lambda x: [float(i) for i in x.lstrip('output ').split(' ')])

        # extract word info (e.g. orth, phon, etc.)
        try:
            word_data = pd.DataFrame(data['word'].apply(lambda x: x.split('_')).tolist(),
                                     columns=['word_id', 'orth', 'phon', 'word_type', 'type_detail'])
            word_data['word_type'] = word_data.apply(lambda row: row['word_type'] if row['type_detail'] is None
                                                     else f"{row['word_type']}_{row['type_detail']}", axis=1)
            word_data = word_data.drop(columns='type_detail')

            # merge word info
            data = data.merge(word_data, left_index=True, right_index=True)
            data = data.drop(columns=['word'])
            data['word_id'] = data['word_id'].astype(int)
        except:
            pass
        data['epoch'] = data['epoch'].astype(int)

        # save
        data.to_pickle(filepath.replace('.txt', '.pkl'))
        print(f"Lens hidden activations saved as pickle file.")

    def __parse_lens_output(self, filepath):
        with open(filepath, 'r') as f:
            lines = f.readlines()

        # extract data from text file
        output_lines = [x for x in lines if 'output' in x]
        output_data = [x.strip().split('|') for x in output_lines]
        output_data = pd.DataFrame(output_data, columns=['epoch', 'word', 'output'])

        target_lines = [x for x in lines if 'target' in x]
        target_data = [x.strip().split('|') for x in target_lines]
        target_data = pd.DataFrame(target_data, columns=['epoch', 'word', 'target'])

        output_data['output'] = output_data['output'].apply(lambda x: [float(i)
                                                                       for i in x.lstrip('output ').split(' ')])
        target_data['target'] = target_data['target'].apply(lambda x: [float(i)
                                                                       for i in x.lstrip('target ').split(' ')])

        try:
            output_word_data = pd.DataFrame(output_data['word'].apply(lambda x: x.split('_')).tolist(),
                                            columns=['word_id', 'orth', 'phon', 'word_type', 'type_detail'])
            target_word_data = pd.DataFrame(target_data['word'].apply(lambda x: x.split('_')).tolist(),
                                            columns=['word_id', 'orth', 'phon', 'word_type', 'type_detail'])

            output_data = output_data.merge(output_word_data, left_index=True, right_index=True)
            target_data = target_data.merge(target_word_data, left_index=True, right_index=True)

            target_data = target_data.drop(columns=['word', 'orth', 'phon', 'word_type', 'type_detail'])
            output_data = output_data.merge(target_data, on=['epoch', 'word_id'])

            output_data['word_type'] = output_data.apply(lambda row: row['word_type'] if row['type_detail'] is None
                                                         else f"{row['word_type']}_{row['type_detail']}", axis=1)
            output_data = output_data.drop(columns=['word', 'type_detail'])
            output_data['word_id'] = output_data['word_data'].astype(int)
        except:
            pass
        output_data['epoch'] = output_data['epoch'].astype(int)

        output_data.to_pickle(filepath.replace('.txt', '.pkl'))

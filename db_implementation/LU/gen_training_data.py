import pandas as pd
import sqlite3
import os
import pprint
import argparse

pp = pprint.PrettyPrinter()

def init_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--task", default='uabsa', type=str, required=False,
                        help="The name of the task, selected from: [uabsa, aste, tasd, aope]")
    parser.add_argument("--dataset", type=str, required=True,
                        help="The name of the dataset")
    parser.add_argument("--paradigm", default='extraction', type=str, required=False,
                        help="The way to construct target sentence, selected from: [annotation, extraction, classification]")
    
    return parser.parse_args()

def gen_LU_data_cross_val(args):

    if args.paradigm == 'classification':
        query = '''SELECT * FROM
             (SELECT s.file, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)

    ORDER BY file, file_sentence_id, target_offset_start;
    
    '''
    else:
        query = '''SELECT * FROM
             (SELECT s.file, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)
    
    WHERE label <> 'O'
    ORDER BY file, file_sentence_id, target_offset_start;
    
    '''

    sql_data = cur.execute(query).fetchall()

    data = []
    encountered_sentences = {}

    for row in sql_data:
        file = row[0]
        sentence = row[2]
        target_head = row[3]
        target_offset_start = row[4]
        target_offset_end = row[5]
        label = {'CB': 'true', 'NCB': 'false', 'NA': 'unknown', 'O': 'other'}[row[6]]

        if args.paradigm == 'annotation':
            entry = [target_head, target_offset_start, target_offset_end, label]
        else:
            entry = [target_head, label, target_offset_start]# f'({target_head}, {label})'

        key = (file, sentence)

        if args.paradigm == 'annotation':
            if key not in encountered_sentences:
                encountered_sentences[key] = [sentence, [entry]]
            else:
                current_list = encountered_sentences[key][1]
                current_list.append(entry)
                encountered_sentences[key] = [sentence, current_list]
        else:
            if key not in encountered_sentences:
                encountered_sentences[key] = [entry]
            else:
                current_list = encountered_sentences[key]
                current_list.append(entry)
                encountered_sentences[key] = current_list

    e2e_prompt = 'Generate (target, presentational factuality) tuples. The target denotes the content of the belief, ' \
                 'and the presentational factuality denotes the factuality according to the author: '

    if args.paradigm == 'annotation':
        for key, value in encountered_sentences.items():
            file, file_sentence_id = key
            sentence, tuples = value

            gold_sentence = ''
            start_here = 0
            # print(sentence)

            for t in sorted(tuples, key=lambda x:x[1]):
                # print(t)
                target_head = t[0]
                target_offset_start, target_offset_end = t[1], t[2]
                label = t[3]

                gold_sentence += f'{sentence[start_here:target_offset_start]}[{target_head}|{label}]'
                # print(gold_sentence)
                start_here = target_offset_end

            gold_sentence += sentence[start_here:]

            # print(gold_sentence)
            # print('-'*30)

            formatted_row = [f'{file}', f'{e2e_prompt}{sentence}', f'{gold_sentence}']
            data.append(formatted_row)
    elif args.paradigm == 'extraction':
        for key, value in encountered_sentences.items():
            file, sentence = key
            tuples = sorted(list(value), key=lambda x:x[2])

            target_data = ''
            for t in tuples:
                head = t[0]
                label = t[1]
                target_data += f'({head}, {label}); '
            target_data = target_data[:-2]
            formatted_row = [f'{file}', f'{e2e_prompt}{sentence}', f'{target_data}']
            data.append(formatted_row)
    elif args.paradigm == 'classification':
        for key, value in encountered_sentences.items():
            file, sentence = key
            tuples = sorted(list(value), key=lambda x:x[2])

            for t in tuples:
                head = t[0]
                label = t[1]
                formatted_row = [f'{file}', f'classify:{head}|||{sentence}', f'{label}']
                data.append(formatted_row)


            # target_data = ''
            # for t in tuples:
            #     head = t[0]
            #     label = t[1]
            #     target_data += f'({head}, {label}); '
            # target_data = target_data[:-2]
            # formatted_row = [f'{file}', f'lu presentational factuality: {sentence}', f'{target_data}']
            # data.append(formatted_row)


    all_data = pd.DataFrame(data)
    print(f'Total data length: {len(all_data.index)}')

    all_data.columns = ['file', 'input_text', 'target_text']

    total_size = len(all_data.index)
    num_bins = 5
    max_bin_size = int((1 / num_bins) * total_size)
    bins = []

    unique_files = set(all_data['file'])
    file_to_data = {}
    for file in unique_files:
        file_to_data[file] = all_data[all_data['file'] == file]
    
    sorted_file_lengths = []
    for file, df in file_to_data.items():
        sorted_file_lengths.append((file, len(df.index)))
    sorted_file_lengths.sort(key=lambda x: -x[1])
    
    for i in range(1, num_bins + 1):
        bin = pd.DataFrame(columns=['file', 'input_text', 'target_text'])

        for t in sorted_file_lengths:
            file = t[0]
            file_length = t[1]
            
            if len(bin.index) + file_length <= max_bin_size:
                bin = pd.concat([bin, file_to_data[file]])
                sorted_file_lengths.remove(t)
        bins.append(bin)
    
    if not os.path.exists(f'formatted_data/{args.task}'):
        os.mkdir(f'formatted_data/{args.task}')
    if not os.path.exists(f'formatted_data/{args.task}/{args.dataset}'):
        os.mkdir(f'formatted_data/{args.task}/{args.dataset}')
    if not os.path.exists(f'formatted_data/{args.task}/{args.dataset}/{args.paradigm}'):
        os.mkdir(f'formatted_data/{args.task}/{args.dataset}/{args.paradigm}')

    total_train = 0

    for i, bin in enumerate(bins):
        iter_path = f'formatted_data/{args.task}/{args.dataset}/{args.paradigm}/iter_{i+1}'

        if not os.path.exists(iter_path):
            os.mkdir(iter_path)

        bin.columns = ['file', 'input_text', 'target_text']
        
        bin.to_csv(f'{iter_path}/test.csv')

        train = all_data.drop(bin.index)
        train.columns = ['file', 'input_text', 'target_text']
        
        train.iloc[:len(train.index) // 5].to_csv(f'{iter_path}/dev.csv')
        train.iloc[len(train.index) // 5:].to_csv(f'{iter_path}/train.csv')

        total_train += len(train.iloc[len(train.index) // 5:].index)
        print(f'Bin {i + 1}: Train = {len(train.iloc[len(train.index) // 5:].index)}, Test = {len(bin.index)}, Dev = {len(train.iloc[:len(train.index) // 5].index)}')

        assert len(train.iloc[:len(train.index) // 5].index) + len(train.iloc[len(train.index) // 5:].index) + len(bin.index) == len(all_data.index)
    
    print(f'Total train = {total_train}')



if __name__ == "__main__":
    con = sqlite3.connect("LU_master.db")
    name = 'LU_initial'
    cur = con.cursor()

    args = init_args()

    gen_LU_data_cross_val(args)
    

    con.close()
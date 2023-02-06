from enum import unique
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
                        help="The way to construct target sentence, selected from: [annotation, extraction]")
    
    return parser.parse_args()

# ...sentence...####(head text, label as number)
# 0 CB, 1 NCB, 2 NA
def gen_LU_data_cross_val(args):
    sql_data = cur.execute('''SELECT * FROM
             (SELECT s.file, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)
    
    WHERE label <> 'O'
    ORDER BY file, file_sentence_id;
    
    ''').fetchall()

    data = []
    encountered_sentences = {}

    for row in sql_data:
        file = row[0]
        sentence = row[2]
        target_head = row[3]
        target_offset_start = row[4]
        target_offset_end = row[5]
        label = {'CB': 'true', 'NCB': 'false', 'NA': 'unknown'}[row[6]]

        if args.paradigm == 'annotation':
            entry = (target_head, target_offset_start, target_offset_end, label)
        else:
            entry = f'({target_head}, {label})'

        key = (file, sentence)

        if args.paradigm == 'annotation':
            if key not in encountered_sentences:
                encountered_sentences[key] = [sentence, [entry]]
            else:
                current_list = encountered_sentences[key][1]
                current_list.append(entry)
                encountered_sentences[key] = \
                    [sentence, current_list]
        else:
            if key not in encountered_sentences:
                encountered_sentences[key] = [entry]
            else:
                current_list = encountered_sentences[key]
                current_list.append(entry)
                encountered_sentences[key] = current_list


    if args.paradigm == 'annotation':
        for key, value in encountered_sentences.items():
            file, file_sentence_id = key
            sentence, tuples = value

            gold_sentence = ''
            start_here = 0

            for t in tuples:
                target_head = t[0]
                target_offset_start, target_offset_end = t[1], t[2]
                label = t[3]

                gold_sentence += f'{sentence[start_here:target_offset_start]}[{target_head}|{label}]'
                start_here = target_offset_end

            gold_sentence += sentence[start_here:]

            formatted_row = [f'{file}', f'source target factuality: {sentence}', f'{gold_sentence}']
            data.append(formatted_row)
    else:
        for key, value in encountered_sentences.items():
            file, sentence = key
            tuples = value

            target_data = ''
            for t in tuples:
                target_data += f'{t}; '
            target_data = target_data[:-2]
            formatted_row = [f'{file}', f'LU Factuality: {sentence}', f'{target_data}']
            data.append(formatted_row)


    all_data = pd.DataFrame(data)
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

    for i, bin in enumerate(bins):
        iter_path = f'formatted_data/{args.task}/{args.dataset}/{args.paradigm}/iter_{i+1}'

        if not os.path.exists(iter_path):
            os.mkdir(iter_path)

        bin.columns = ['file', 'input_text', 'target_text']
        bin.iloc[:len(bin.index) // 2].to_csv(f'{iter_path}/test.csv')
        bin.iloc[len(bin.index) // 2:].to_csv(f'{iter_path}/dev.csv')

        train = all_data.drop(bin.index)
        train.columns = ['file', 'input_text', 'target_text']
        train.to_csv(f'{iter_path}/train.csv')

    # boundary = int(0.8 * total_size)
    # train = df.iloc[0:boundary]
    # test = df.iloc[boundary + 1:]

    # before_boundary_file = train.iat[-1, 0]
    # after_boundary_file = test.iat[0, 0]
    # if before_boundary_file == after_boundary_file:
    #     spillover = train[train['file'] == before_boundary_file]
    #     test = pd.concat([test, spillover])
    #     train.drop(spillover.index, inplace=True)

    # dev = test.iloc[len(test.index) // 2:]
    # test.drop(dev.index, inplace=True)

    # print(train)
    # print(test)
    # print(dev)
    #
    # print(boundary, total_size - boundary)
    # print(before_boundary_file, after_boundary_file)

    # return train, test, dev




if __name__ == "__main__":
    con = sqlite3.connect("LU_master.db")
    name = 'LU_initial'
    cur = con.cursor()

    args = init_args()

    gen_LU_data_cross_val(args)
    

    con.close()
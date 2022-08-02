from progress.bar import Bar


class FB_SENTENCE_PROCESSOR:

    FILE = 0
    SENTENCE_ID = 1
    SENTENCE = 2
    RAW_OFFSET_INIT = 2
    REL_SOURCE_TEXT = 2

    def __init__(self, sentences_set, initial_offsets, rel_source_texts,
                 source_offsets, target_offsets, targets, fact_values):

        # loading data from outside object's SQL queries
        self.sentences_set = sentences_set
        self.initial_offsets = initial_offsets
        self.source_offsets = source_offsets
        self.errors = {}
        self.num_errors = 0
        self.rel_source_texts = rel_source_texts
        self.target_offsets = target_offsets
        self.fact_values = fact_values
        self.targets = targets

        # python representation of database for data pre-processing
        self.sentences = []
        self.next_sentence_id = 1

        self.mentions = []
        self.next_mention_id = 1

        self.sources = []
        self.next_source_id = 1

        self.attitudes = []
        self.next_attitude_id = 1

        # self.problem_eid_label_keys = []

    def go(self):
        bar = Bar('Sentences Processed', max=len(self.sentences_set))
        for row in self.sentences_set:
            row = list(row)
            self.process_sentence(row, bar)
        print('\nSentence processing complete.')
        # for row in self.problem_eid_label_keys:
        #     print(row)

        bar.finish()

    def get_errors(self):
        return self.errors, self.num_errors

    # dealing with a single sentence -- go nesting level by nesting level,
    # dealing with each top-level source as it appears in FactBank
    def process_sentence(self, row, bar):
        if row[self.SENTENCE_ID] == 0:
            return

        row[self.SENTENCE] = str(row[self.SENTENCE][1:-2].replace("\\", "").replace("`", ""))

        self.sentences.append(
            (self.next_sentence_id, row[self.FILE][1:-1], row[self.SENTENCE_ID], row[self.SENTENCE]))
        global_sentence_id = self.next_sentence_id
        self.next_sentence_id += 1

        # grabbing the relevant top-level source from the dictionary created earlier
        rel_source_key = (row[self.FILE], row[self.SENTENCE_ID])
        if rel_source_key not in self.rel_source_texts:
            self.rel_source_texts[rel_source_key] = ['AUTHOR']
        sources = self.rel_source_texts[rel_source_key]

        # dealing with each relevant source starting at the lowest nesting level, i.e., AUTHOR
        for current_nesting_level in range(0, 3):
            for rel_source_text in sources:
                nesting_level, relevant_source = self.calc_nesting_level(rel_source_text)

                # only dealing with sources at the relevant nesting level
                if nesting_level != current_nesting_level:
                    continue

                # getting the source offsets
                source_offsets_key = (row[self.FILE], row[self.SENTENCE_ID])
                if source_offsets_key not in self.source_offsets:
                    self.source_offsets[source_offsets_key] = (None, None, relevant_source)
                source_offsets = self.source_offsets[source_offsets_key]

                # tweaking offsets as needed
                offset_start, offset_end, success = self.calc_offsets(row[self.FILE], row[self.SENTENCE_ID],
                                                                      row[self.SENTENCE],
                                                                      source_offsets[0],
                                                                      source_offsets[1],
                                                                      relevant_source, rel_source_text)

                if not success:
                    continue
                # saving the newly-minted mention for later insertion
                self.mentions.append(
                    (self.next_mention_id, global_sentence_id, relevant_source, offset_start, offset_end))

                global_source_token_id = self.next_mention_id
                self.next_mention_id += 1

                # if a parent source is relevant, find it
                if nesting_level == 0:
                    parent_source_id = -1
                else:
                    parent_source_text = self.calc_parent_source(rel_source_text)
                    parent_source_id = None
                    for i in range(len(self.sources)):
                        if self.sources[i][1] == global_sentence_id \
                                and self.sources[i][4] == current_nesting_level \
                                and self.sources[i][5] == parent_source_text:
                            parent_source_id = i + 1
                            break

                self.sources.append(
                    (self.next_source_id, global_sentence_id, global_source_token_id, parent_source_id,
                     current_nesting_level, relevant_source))

                # dealing with targets now
                attitude_source_id = self.next_source_id
                self.next_source_id += 1

                eid_label_key = (row[self.FILE], row[self.SENTENCE_ID],
                                 "'{}'".format(rel_source_text))
                if eid_label_key not in self.fact_values:
                    # self.problem_eid_label_keys.append(eid_label_key) # DEBUGGING
                    continue
                else:
                    eid_label_return = self.fact_values[eid_label_key]

                for example in eid_label_return:

                    eid = example[0]
                    fact_value = example[1][1:-2]
                    target_return = self.targets[(row[self.FILE], row[self.SENTENCE_ID], eid)]

                    tok_loc = target_return[0]
                    target_head = target_return[1][1:-1]


                    target_offsets_return = self.target_offsets[(row[self.FILE], row[self.SENTENCE_ID],
                                                                tok_loc)]

                    target_offset_start = target_offsets_return[0]
                    target_offset_end = target_offsets_return[1]

                    target_offset_start, target_offset_end, success = self.calc_offsets(row[self.FILE],
                                                                                        row[self.SENTENCE_ID],
                                                                                        row[self.SENTENCE],
                                                                                        target_offset_start,
                                                                                        target_offset_end,
                                                                                        target_head,
                                                                                        rel_source_text)

                    self.mentions.append((self.next_mention_id, global_sentence_id,
                                          target_head, target_offset_start, target_offset_end))

                    target_token_id = self.next_mention_id
                    self.next_mention_id += 1

                    self.attitudes.append((self.next_attitude_id, attitude_source_id,
                                           target_token_id, fact_value, 'Belief'))
                    self.next_attitude_id += 1

        bar.next()

    def calc_parent_source(self, source_text):
        if source_text == 'AUTHOR':
            return None
        start_index = source_text.index('_') + 1
        parent_source = source_text[start_index:]
        if parent_source.count('_') > 0:
            parent_source = parent_source[:parent_source.index('_')]
        if '=' in parent_source:
            parent_source = parent_source[:parent_source.index('=')]
        return parent_source

    def calc_nesting_level(self, source_text):
        nesting_level = source_text.count('_')
        if '=' in source_text:
            return nesting_level, source_text[0:source_text.index('=')]
        if source_text == 'AUTHOR':
            return 0, 'AUTHOR'
        return nesting_level, source_text[:source_text.index('_')]

    # calculating the initial offset, since the indices are file-based and not sentence-based in the DB
    def calc_offsets(self, file, sent_id, raw_sentence, offset_start, offset_end, head, rel_source_text):
        if (offset_start is None and offset_end is None) or head is None:
            return -1, -1, True

        file_offset = self.initial_offsets[(file, sent_id)]
        success = True

        # ad hoc logic to adjust offsets
        head_length = offset_end - offset_start

        offset_start -= file_offset

        while (
                0 < offset_start < len(raw_sentence) and
                raw_sentence[offset_start] not in ' `"'
        ):
            offset_start -= 1
        if offset_start > 0:
            offset_start += 1
        offset_end = offset_start + head_length
        pred_head = raw_sentence[offset_start:offset_end]

        # keeping the asterisks just for easier understanding of the error dataset
        result_sentence = raw_sentence[:offset_start] + "* " + head + " *" + raw_sentence[offset_end:]

        if pred_head != head and raw_sentence.count(head) == 1:
            # attempting index method if head exists uniquely in sentence
            offset_start = raw_sentence.index(head)
            offset_end = offset_start + len(head)
            pred_head = raw_sentence[offset_start:offset_end]
            if pred_head != head:
                success = False
            else:
                success = True
        if not success:
            self.num_errors += 1
            error_key = (file, sent_id)
            entry = (file[1:-1], sent_id, offset_start, offset_end, pred_head, head,
                     raw_sentence, result_sentence, rel_source_text)
            if error_key not in self.errors:
                self.errors[error_key] = [entry]
            else:
                self.errors[error_key].append(entry)

        return offset_start, offset_end, success



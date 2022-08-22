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

        for i in range(len(self.sources)):
            self.sources[i] = self.sources[i][:-1]

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

        row[self.SENTENCE] = str(row[self.SENTENCE][1:-2].replace("\\", ""))

        self.sentences.append(
            (self.next_sentence_id, row[self.FILE][1:-1], row[self.SENTENCE_ID], row[self.SENTENCE]))
        global_sentence_id = self.next_sentence_id
        self.next_sentence_id += 1

        # grabbing the relevant top-level source from the dictionary created earlier
        rel_source_key = (row[self.FILE], row[self.SENTENCE_ID])
        if rel_source_key not in self.rel_source_texts:
            self.rel_source_texts[rel_source_key] = [(-1, 'AUTHOR')]
        sources = self.rel_source_texts[rel_source_key]

        # dealing with each relevant source starting at the lowest nesting level, i.e., AUTHOR
        for current_nesting_level in range(0, 4):
            for rel_source_id, rel_source_text in sources:
                nesting_level, relevant_source_id, relevant_source = \
                    self.calc_nesting_level(rel_source_text, rel_source_id)

                # only dealing with sources at the relevant nesting level
                if nesting_level != current_nesting_level:
                    continue

                # getting the source offsets
                source_offsets_key = (row[self.FILE], row[self.SENTENCE_ID])
                if source_offsets_key not in self.source_offsets:
                    self.source_offsets[source_offsets_key] = (None, None, relevant_source)
                source_offsets = self.source_offsets[source_offsets_key]

                # tweaking offsets as needed
                relevant_source = relevant_source.replace("\\", "")
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
                    parent_relevant_source_id = self.calc_parent_source(rel_source_id)
                    parent_source_id = None
                    for i in range(len(self.sources)):
                        if self.sources[i][1] == global_sentence_id \
                                and self.sources[i][4] == current_nesting_level - 1 \
                                and self.sources[i][6] == parent_relevant_source_id:
                            parent_source_id = i + 1
                            break

                self.sources.append(
                    (self.next_source_id, global_sentence_id, global_source_token_id, parent_source_id,
                     current_nesting_level, relevant_source, relevant_source_id))

                # dealing with targets now
                attitude_source_id = self.next_source_id
                self.next_source_id += 1

                eid_label_key = (row[self.FILE], row[self.SENTENCE_ID],
                                 "'{}'".format(rel_source_id))

                if eid_label_key not in self.fact_values:
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

                    target_head = target_head.replace("\\", "")
                    target_offset_start, target_offset_end, success = self.calc_offsets(row[self.FILE],
                                                                                        row[self.SENTENCE_ID],
                                                                                        row[self.SENTENCE],
                                                                                        target_offset_start,
                                                                                        target_offset_end,
                                                                                        target_head,
                                                                                        rel_source_text)
                    if success:

                        self.mentions.append((self.next_mention_id, global_sentence_id,
                                              target_head, target_offset_start, target_offset_end))

                        target_token_id = self.next_mention_id
                        self.next_mention_id += 1

                        self.attitudes.append((self.next_attitude_id, attitude_source_id,
                                               target_token_id, fact_value, 'Belief'))
                        self.next_attitude_id += 1
        bar.next()

    def calc_parent_source(self, source_id):
        if source_id == 's0':
            return None
        start_index = source_id.index('_') + 1
        parent_source = source_id[start_index:]
        if '_' in parent_source:
            parent_source = parent_source[:parent_source.index('_')]
        if '=' in parent_source:
            parent_source = parent_source[:parent_source.index('=')]
        return parent_source

    def calc_nesting_level(self, source_text, rel_source_id):
        nesting_level = source_text.count('_')
        if '=' in source_text:
            source_text = source_text[:source_text.index('=')]
        if source_text == 'AUTHOR':
            return 0, -1, 'AUTHOR'
        if '_' in source_text:
            source_text = source_text[:source_text.index('_')]
        if '=' in rel_source_id:
            rel_source_id = rel_source_id[:rel_source_id.index('=')]
        if '_' in rel_source_id:
            rel_source_id = rel_source_id[:rel_source_id.index('_')]
        return nesting_level, rel_source_id, source_text

    # calculating the initial offset, since the indices are file-based and not sentence-based in the DB
    def calc_offsets(self, file, sent_id, raw_sentence, offset_start, offset_end, head, rel_source_text):

        if (offset_start is None and offset_end is None) or head in [None, 'AUTHOR', 'GEN', 'DUMMY']:
            return -1, -1, True

        success = False

        if raw_sentence.count(head) == 1:
            # attempting index method if head exists uniquely in sentence
            offset_start = raw_sentence.index(head)
            offset_end = offset_start + len(head)
            pred_head = raw_sentence[offset_start:offset_end]
            if pred_head != head:
                success = False
            else:
                success = True

        if not success:
            file_offset = self.initial_offsets[(file, sent_id)]
            offset_start -= file_offset
            offset_end = offset_start + len(head)

            left_side_boundary = offset_start
            right_side_boundary = left_side_boundary + 1
            search_left = True
            search_right = True

            while not success:
                # keeping boundaries in range
                if left_side_boundary < 0:
                    search_left = False
                if right_side_boundary > len(raw_sentence):
                    search_right = False

                # give up if there's nothing left to search
                if not search_left and not search_right:
                    break

                # search both sides at the current boundaries, if there's space left, for the head
                parts = [(search_left, left_side_boundary), (search_right, right_side_boundary)]
                for part in parts:
                    search = part[0]
                    boundary = part[1]
                    if search and raw_sentence[boundary:boundary + len(head)] == head:
                        offset_start = boundary
                        offset_end = boundary + len(head)
                        success = True
                        break

                # if no match, shift the boundaries
                left_side_boundary -= 1
                right_side_boundary += 1

        pred_head = raw_sentence[offset_start:offset_end]
        if not success:
            # keeping the asterisks just for easier understanding of the error dataset
            result_sentence = raw_sentence[:offset_start] + "* " + head + " *" + raw_sentence[offset_end:]
            self.num_errors += 1
            error_key = (file, sent_id)
            entry = (file[1:-1], sent_id, offset_start, offset_end, pred_head, head,
                     raw_sentence, result_sentence, rel_source_text)
            if error_key not in self.errors:
                self.errors[error_key] = [entry]
            else:
                self.errors[error_key].append(entry)

        return offset_start, offset_end, success



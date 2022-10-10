from progress.bar import Bar
import spacy

class FbSentenceProcessor:

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
        self.unique_mentions = {}
        self.next_mention_id = 1

        self.sources = []
        self.next_source_id = 1

        self.attitudes = {}
        self.next_attitude_id = 1

        self.nlp = spacy.load("en_core_web_sm")
        self.current_doc = None
        self.current_sentence = None
        self.current_head = None

    # sending each sentence to the process_sentence function
    def go(self):
        bar = Bar('Examples Processed', max=13506)  # 13506 = number of attitudes
        for row in self.sentences_set:
            row = list(row)
            self.process_sentence(row, bar)
        print('\nSentence processing complete.')

        # removing internal data before SQL insertion
        for i in range(len(self.sources)):
            self.sources[i] = self.sources[i][:-1]
        bar.finish()

        self.uu_to_rob()

    def get_errors(self):
        return self.errors, self.num_errors

    # dealing with a single sentence -- go nesting level by nesting level,
    # dealing with each top-level source as it appears in FactBank
    def process_sentence(self, row, bar):
        if row[self.SENTENCE_ID] == 0:
            return

        row[self.SENTENCE] = str(row[self.SENTENCE][1:-2].replace("\\", ""))
        row[self.SENTENCE] = row[self.SENTENCE].replace("``", '"')
        row[self.SENTENCE] = row[self.SENTENCE].replace("''", "\"")
        self.current_sentence = row[self.SENTENCE]

        self.sentences.append(
            (self.next_sentence_id, row[self.FILE][1:-1], row[self.SENTENCE_ID], self.current_sentence))
        global_sentence_id = self.next_sentence_id
        self.next_sentence_id += 1

        self.current_doc = self.nlp(self.current_sentence)

        self.traverse_nesting_structure(row, global_sentence_id, bar)

    def traverse_nesting_structure(self, row, global_sentence_id, bar):
        # grabbing the relevant top-level source from the dictionary created earlier
        # and filling in values for author-only annotations
        rel_source_key = (row[self.FILE], row[self.SENTENCE_ID])
        if rel_source_key not in self.rel_source_texts:
            self.rel_source_texts[rel_source_key] = [(-1, 'AUTHOR')]
        sources = self.rel_source_texts[rel_source_key]

        # dealing with each relevant source starting at the lowest nesting level, i.e., AUTHOR
        # here, rel_source_id represents the sentence-level ID for that source,
        # i.e., s0, s1, etc. or s2_s1_s0 for nested sources
        for current_nesting_level in range(0, 4):
            for rel_source_id, rel_source_text in sources:
                nesting_level, relevant_source_id, relevant_source = \
                    self.calc_nesting_level(rel_source_text, rel_source_id)

                # only dealing with sources at the relevant nesting level
                if nesting_level != current_nesting_level:
                    continue

                relevant_source, offset_start, offset_end, success = \
                    self.get_source_offsets(row, relevant_source, rel_source_text)

                if not success:
                    continue

                global_source_token_id = self.catalog_mention(global_sentence_id, relevant_source,
                                                              offset_start, offset_end)

                parent_source_id = self.find_parent_source(global_sentence_id, nesting_level,
                                                           current_nesting_level, rel_source_id)

                # now we actually insert the source
                self.sources.append((self.next_source_id, global_sentence_id, global_source_token_id,
                                     parent_source_id, current_nesting_level, relevant_source, relevant_source_id))

                # dealing with targets now
                attitude_source_id = self.next_source_id
                self.next_source_id += 1

                # retrieving all attitudes linked to the source we just inserted, if there are any
                eid_label_key = (row[self.FILE], row[self.SENTENCE_ID],
                                 "'{}'".format(rel_source_id))

                if eid_label_key not in self.fact_values:
                    continue
                else:
                    eid_label_return = self.fact_values[eid_label_key]

                self.parse_attitudes(eid_label_return, row, rel_source_text,
                                     global_sentence_id, attitude_source_id, bar)

    def parse_attitudes(self, eid_label_return, row, rel_source_text, global_sentence_id, attitude_source_id, bar):
        # iterating over each attitude, inserting to the attitudes table
        for example in eid_label_return:

            eid = example[0]
            fact_value = example[1][1:-2]

            target_return = self.targets[(row[self.FILE], row[self.SENTENCE_ID], eid)]

            # we need the tokLoc in order to get at the target offsets
            # (this is an artifact of the original FactBank data)
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
                self.catalog_attitude(global_sentence_id, target_head, target_offset_start,
                                      target_offset_end, attitude_source_id, fact_value)
            bar.next()

    def get_head_span(self, head_token_offset_start, head_token_offset_end):

        pred_head = self.current_sentence[head_token_offset_start:head_token_offset_end]
        head_token = self.current_doc.char_span(head_token_offset_start, head_token_offset_end)
        if head_token is None:
            head_token = self.current_doc.char_span(head_token_offset_start, head_token_offset_end + 1)
        head_token = head_token[0]

        # print('stop here')

        if head_token.text != head_token.head.text:
            head_token = head_token.head

        span_start = min(child.idx for child in head_token.children)

        span_end_token = [child for child in head_token.children if child.idx == \
                          max(child.idx for child in head_token.children if child.dep_ != 'punct')][0]
        span_end = span_end_token.idx + len(span_end_token.text)

        return (span_start, span_end)

    def catalog_attitude(self, global_sentence_id, target_head, target_offset_start,
                         target_offset_end, attitude_source_id, fact_value):
        target_token_id = self.catalog_mention(global_sentence_id, target_head,
                                               target_offset_start, target_offset_end)

        # the attitudes table is represented internally as a dictionary of lists, again to help
        # with the UU -> ROB task
        attitude_key = (attitude_source_id, target_token_id)
        if attitude_key in self.attitudes:
            self.attitudes[attitude_key].append([self.next_attitude_id, attitude_source_id,
                                                 target_token_id, fact_value, 'Belief'])
        else:
            self.attitudes[attitude_key] = [[self.next_attitude_id, attitude_source_id,
                                             target_token_id, fact_value, 'Belief']]
        self.next_attitude_id += 1

    # saving a newly-minted mention for later insertion
    # we maintain a separate dictionary of unique mentions in order to
    # make it easier to perform the UU -> ROB label switch later on
    def catalog_mention(self, global_sentence_id, text, target_offset_start, target_offset_end):
        # adding the target mention to the aforementioned dictionary of unique mentions
        unique_mention_key = (global_sentence_id, text, target_offset_start, target_offset_end)
        if unique_mention_key not in self.unique_mentions:
            self.unique_mentions[unique_mention_key] = self.next_mention_id

            self.current_head = text
            if target_offset_start != -1:
                span_offset_start, span_offset_end = self.get_head_span(target_offset_start, target_offset_end)
                span_text = self.current_sentence[span_offset_start:span_offset_end]
                self.mentions.append((self.next_mention_id, global_sentence_id,
                                      text, target_offset_start, target_offset_end,
                                      span_text, span_offset_start, span_offset_end))
            else:
                self.mentions.append((self.next_mention_id, global_sentence_id,
                                      text, target_offset_start, target_offset_end,
                                      None, None, None))

            global_token_id = self.next_mention_id
            self.next_mention_id += 1
        else:
            global_token_id = self.unique_mentions[unique_mention_key]

        return global_token_id

    def find_parent_source(self, global_sentence_id, nesting_level, current_nesting_level, rel_source_id):
        # if a parent source is relevant, find it and catalog it
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

        return parent_source_id

    def get_source_offsets(self, row, relevant_source, rel_source_text):
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
        return relevant_source, offset_start, offset_end, success

    # finding the source id after the first underscore; for 's2_s1_s0', we retrieve 's1'
    @staticmethod
    def calc_parent_source(source_id): # s1_s0
        if source_id == 's0':
            return None
        start_index = source_id.index('_') + 1
        parent_source = source_id[start_index:]
        if '_' in parent_source:
            parent_source = parent_source[:parent_source.index('_')]
        if '=' in parent_source:
            parent_source = parent_source[:parent_source.index('=')]
        return parent_source

    # determining as source's nesting level based on underscore placement in rel_source_id strings
    @staticmethod
    def calc_nesting_level(source_text, rel_source_id):
        nesting_level = source_text.count('_')
        if '=' in source_text:
            source_text = source_text[:source_text.index('=')]
        if source_text == 'AUTHOR':
            return 0, 's0', 'AUTHOR'
        if '_' in source_text:
            source_text = source_text[:source_text.index('_')]
        if '=' in rel_source_id:
            rel_source_id = rel_source_id[:rel_source_id.index('=')]
        if '_' in rel_source_id:
            rel_source_id = rel_source_id[:rel_source_id.index('_')]
        return nesting_level, rel_source_id, source_text

    # changing relevant Uu labels to ROB (reported belief)
    # in order to more closely match the BEST corpus' annotation style
    # if bottom of source structure is NOT Uu (GEN and DUMMY do trigger this algo), go up the source tree,
    # for each intermediate source including the very top, switch Uu to ROB, for the target gotten from the bottom
    """
    for each attitude:
        if label is not Uu:
            save target_token_id and source_id in variables
            for each parent source until NULL, find the attitude with the corresponding target_token_id and source_id,
            and if the label is Uu, change it to ROB
    :return:
    """
    def uu_to_rob(self):
        num_changes = 0
        # for each attitude
        for key in self.attitudes:

            bottom_attitude_list = self.attitudes[key]
            for bottom_attitude in bottom_attitude_list:

                bottom_label = bottom_attitude[3]
                if bottom_label != 'Uu':

                    # save target_token_id and source_id in variables
                    relevant_target_token_id = bottom_attitude[2]
                    bottom_source = self.sources[bottom_attitude[1] - 1]
                    parent_source_id = bottom_source[3]

                    # for each parent source until NULL
                    while parent_source_id not in (None, -1):

                        current_source = self.sources[parent_source_id - 1]
                        current_source_id = parent_source_id
                        parent_source_id = current_source[3]
                        attitude_key = (current_source_id, relevant_target_token_id)

                        # find the attitude with the corresponding target_token_id and source_id
                        if attitude_key in self.attitudes:

                            current_attitude_list = self.attitudes[attitude_key]

                            for current_attitude in current_attitude_list:

                                # if the label is Uu, change it to ROB
                                if current_attitude[3] == 'Uu':

                                    current_attitude_list.remove(current_attitude)
                                    current_attitude[3] = 'ROB'
                                    current_attitude_list.append(current_attitude)
                                    self.attitudes[attitude_key] = current_attitude_list
                                    num_changes += 1

        print('{} changes from Uu to ROB'.format(num_changes))

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

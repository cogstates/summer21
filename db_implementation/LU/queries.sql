SELECT * FROM
             (SELECT s.file, a.attitude_id, s.sentence_id, s.file_sentence_id,
       s.sentence, m.token_text target_head, m.phrase_text span_text, m.token_offset_start target_offset_start,
       m.token_offset_end target_offset_end, m.phrase_offset_start span_offset_start, m.phrase_offset_end span_offset_end,
       s2.source, a.label, a.label_type
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id)
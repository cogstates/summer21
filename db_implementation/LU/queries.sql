SELECT a.attitude_id, s.sentence_id, s.sentence, s.file, s.file_sentence_id, m.token_text target_head,
       m.token_offset_start target_offset_start, m.token_offset_end target_offset_end, m.token_id target_token,
       s2.source, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id
    JOIN sources s2 on s2.source_id = a.source_id;
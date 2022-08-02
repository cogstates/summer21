SELECT * FROM
             (SELECT a.attitude_id, s.sentence_id, s.sentence, s.file, s.file_sentence_id, m.token_text target_head,
       m.token_offset_start target_offset_start, m.token_offset_end target_offset_end, m.token_id target_token, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id) target_data
JOIN (SELECT a.attitude_id, m.token_text source_text,
       m.token_offset_start source_offset_start, m.token_offset_end source_offset_end, m.token_id source_token_id
FROM attitudes a
    JOIN sources s on a.source_id = s.source_id
    JOIN mentions m on s.token_id = m.token_id) source_data on target_data.attitude_id = source_data.attitude_id;
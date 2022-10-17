--BASIC QUERY
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

-- SPAN QUERY
SELECT distinct target_data.*, source_data.* FROM
             (SELECT a.attitude_id, s.sentence_id, s.sentence, s.file, s.file_sentence_id, m.token_text target_head,
       m.token_offset_start target_offset_start, m.token_offset_end target_offset_end,
       m.phrase_offset_start target_span_start, m.phrase_offset_end target_span_end,
       SUBSTR(s.sentence, m.phrase_offset_start, m.phrase_offset_end - m.phrase_offset_start + 1) target_span, m.token_id target_token, a.label
FROM attitudes a
    JOIN mentions m on m.token_id = a.target_token_id
    JOIN sentences s on m.sentence_id = s.sentence_id) target_data
JOIN (SELECT a.attitude_id, m.token_text source_text,
       s.nesting_level, m.token_offset_start source_offset_start, m.token_offset_end source_offset_end,
       m.phrase_offset_start source_span_start, m.phrase_offset_end source_span_end,
       SUBSTR(s2.sentence, m.phrase_offset_start, m.phrase_offset_end - m.phrase_offset_start + 1) source_span,
       m.token_id source_token_id, s.parent_source_id, s.source_id
FROM attitudes a
    JOIN sources s on a.source_id = s.source_id
    JOIN mentions m on s.token_id = m.token_id
    JOIN sentences s2 on m.sentence_id = s2.sentence_id) source_data on target_data.attitude_id = source_data.attitude_id;
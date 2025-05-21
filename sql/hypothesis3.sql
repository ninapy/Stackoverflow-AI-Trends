WITH labeled_posts AS (
    SELECT 
        CONCAT(j.title, ' ', j.body) AS full_text,
        
        CASE
            WHEN j.date < '2022-11-30' THEN 'Pre-ChatGPT'
            ELSE 'Post-ChatGPT'
        END AS period

    FROM joined j
    WHERE j.tags IS NOT NULL
)

SELECT *
FROM labeled_posts;

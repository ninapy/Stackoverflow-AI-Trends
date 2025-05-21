WITH labeled_posts AS (
    SELECT 
        CONCAT(j.title, ' ', j.body) AS full_text,
        
        CASE
            WHEN LOWER(j.tags) LIKE '%tensorflow%' 
              OR LOWER(j.tags) LIKE '%pytorch%' 
              OR LOWER(j.tags) LIKE '%jax%' 
              OR LOWER(j.tags) LIKE '%machine learning%' 
              OR LOWER(j.tags) LIKE '%deep learning%' 
              OR LOWER(j.tags) LIKE '%ai%' 
              OR LOWER(j.tags) LIKE '%artificial intelligence%' 
            THEN 'Yes'
            ELSE 'No'
        END AS ai_related,
        
        CASE
            WHEN j.date < '2022-11-30' THEN 'Pre-ChatGPT'
            ELSE 'Post-ChatGPT'
        END AS period,
        
        CASE
            WHEN LOWER(j.tags) LIKE '%python%' THEN 'Yes'
            ELSE 'No'
        END AS has_python

    FROM joined j
    WHERE j.tags IS NOT NULL
)

SELECT *
FROM labeled_posts
WHERE ai_related='Yes';

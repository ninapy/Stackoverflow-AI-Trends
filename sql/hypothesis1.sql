SELECT 
    language,
    language_type,
    full_text
FROM (
    SELECT 
        CASE
            WHEN LOWER(j.tags) LIKE '%abap%' THEN 'Abap'
            WHEN LOWER(j.tags) LIKE '%ada%' THEN 'Ada'
            WHEN LOWER(j.tags) LIKE '%c++%' OR LOWER(j.tags) LIKE '%c/%' OR LOWER(j.tags) LIKE '%c%' THEN 'C/C++'
            WHEN LOWER(j.tags) LIKE '%c#%' THEN 'C#'
            WHEN LOWER(j.tags) LIKE '%cobol%' THEN 'Cobol'
            WHEN LOWER(j.tags) LIKE '%dart%' THEN 'Dart'
            WHEN LOWER(j.tags) LIKE '%delphi%' OR LOWER(j.tags) LIKE '%pascal%' THEN 'Delphi/Pascal'
            WHEN LOWER(j.tags) LIKE '%go%' THEN 'Go'
            WHEN LOWER(j.tags) LIKE '%groovy%' THEN 'Groovy'
            WHEN LOWER(j.tags) LIKE '%haskell%' THEN 'Haskell'
            WHEN LOWER(j.tags) LIKE '%java%' THEN 'Java'
            WHEN LOWER(j.tags) LIKE '%javascript%' OR LOWER(j.tags) LIKE '%js%' THEN 'JavaScript'
            WHEN LOWER(j.tags) LIKE '%julia%' THEN 'Julia'
            WHEN LOWER(j.tags) LIKE '%kotlin%' THEN 'Kotlin'
            WHEN LOWER(j.tags) LIKE '%lua%' THEN 'Lua'
            WHEN LOWER(j.tags) LIKE '%matlab%' THEN 'Matlab'
            WHEN LOWER(j.tags) LIKE '%objective-c%' THEN 'Objective-C'
            WHEN LOWER(j.tags) LIKE '%perl%' THEN 'Perl'
            WHEN LOWER(j.tags) LIKE '%php%' THEN 'PHP'
            WHEN LOWER(j.tags) LIKE '%powershell%' THEN 'Powershell'
            WHEN LOWER(j.tags) LIKE '%python%' THEN 'Python'
            WHEN LOWER(j.tags) LIKE '% r %' OR LOWER(j.tags) LIKE '% r%' THEN 'R'
            WHEN LOWER(j.tags) LIKE '%ruby%' THEN 'Ruby'
            WHEN LOWER(j.tags) LIKE '%rust%' THEN 'Rust'
            WHEN LOWER(j.tags) LIKE '%scala%' THEN 'Scala'
            WHEN LOWER(j.tags) LIKE '%swift%' THEN 'Swift'
            WHEN LOWER(j.tags) LIKE '%typescript%' THEN 'TypeScript'
            WHEN LOWER(j.tags) LIKE '%vba%' THEN 'VBA'
            WHEN LOWER(j.tags) LIKE '%visual basic%' THEN 'Visual Basic'
            ELSE 'Other'
        END AS language,

        CASE
            WHEN LOWER(j.tags) LIKE '%c++%' OR LOWER(j.tags) LIKE '%rust%' OR LOWER(j.tags) LIKE '%objective-c%' THEN 'low-level'
            ELSE 'high-level'
        END AS language_type,

        CONCAT(j.title, ' ', j.body) AS full_text,
        j.tags
    FROM joined j
    WHERE j.tags IS NOT NULL
) AS lang_posts
WHERE language != 'Other'
ORDER BY language;

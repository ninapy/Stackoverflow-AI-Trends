import pandas as pd
import sqlite3
import os

# Step 1: Define paths
DIR = os.path.dirname(os.path.abspath(__file__))
TO_DATA = os.path.join(DIR, '..', 'data')

# Step 2: Load CSV into DataFrame
df = pd.read_csv(TO_DATA + '/datasets/joined.csv', encoding='latin1')

# Step 3: Save DataFrame to SQLite
conn = sqlite3.connect(TO_DATA + '/datasets/joined.db')
df.to_sql('joined', conn, if_exists='replace', index=False)

# Step 4: Define SQL query for filtering and transformation
query = """
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

        j.title || ' ' || j.body AS full_text  -- SQLite uses || for concatenation
    FROM joined j
    WHERE j.tags IS NOT NULL
) AS lang_posts
WHERE language != 'Other'
ORDER BY language;
"""

filtered_df = pd.read_sql_query(query, conn)

# export cleaned data to CSV
output_path = os.path.join(TO_DATA, 'cleaned_joined.csv')
filtered_df.to_csv(output_path, index=False)

conn.close()

print(f"Cleaned data saved to {output_path}")
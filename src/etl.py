# batch reads the data source and processes just for pop-english songs
# this data is stored as a single parquet file for fast loading
import re
from pathlib import Path

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.utils import timer


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data"
SOURCE_CSV_PATH = DATA_DIR / "song_lyrics.csv"
TARGET_PARQUET_PATH = DATA_DIR / "pop_songs.parquet"


# filter for pop-english songs, sample random songs to fit into memory
# sample 2.5%
FRAC_SIZE = 0.025
def filter_songs(chunk: pd.DataFrame) -> pd.DataFrame:
  pop_songs = chunk[(chunk['tag'].str.lower().str.contains('pop', na=False)) & (chunk['language'] == 'en')][['tag', 'title', 'lyrics']]
  return pop_songs.sample(frac=FRAC_SIZE, random_state=42)


# here we want to process 2 things:
# add tokens to denote sections in the lyrics (e.g. <VERSE>)
def process_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
  # filter for pop songs
  pop_songs = filter_songs(chunk)

  # lyrics preprocessing: add tokens to denote sections in the lyrics (e.g. <VERSE>, <CHORUS>)
  def preprocess_lyrics(lyrics):
    # genius already sections lyrics with [n], we can search for '[]' and identify keywords in section to replace
    if pd.isna(lyrics):
      return lyrics

    def replace_section(match):
      sections = ['VERSE', 'CHORUS', 'BRIDGE', 'INTRO', 'OUTRO', 'PRE-CHORUS', 'HOOK']
      inner_text = match.group(1).upper()
      for section in sections:
        if section in inner_text:
          return f' <{section}> '
      
      # this usually contains extra info about the song (e.g. additional artists, lyrictist)
      return f'[N/A]'

    # replace all occurrences of [n] with the appropriate section token
    replaced_lyrics = re.sub(r'\[(.*?)\]', replace_section, lyrics, flags=re.IGNORECASE)

    # remove all the [N/A] tokens as they don't add value to the lyrics
    replaced_lyrics = re.sub(r'\[N/A\]', '', replaced_lyrics, flags=re.IGNORECASE)

    # replace all the \n with <NEWLINE> token to preserve line breaks in the lyrics
    replaced_lyrics = re.sub(r'\n+', ' <NEWLINE> ', replaced_lyrics, flags=re.IGNORECASE)

    return replaced_lyrics

  pop_songs['lyrics'] = pop_songs['lyrics'].apply(preprocess_lyrics)

  return pop_songs

def main() -> None:
  batch_size = 100000

  schema = pa.schema([
      ('tag', pa.string()),
      ('title', pa.string()),
      ('lyrics', pa.string())
  ])

  TARGET_PARQUET_PATH.parent.mkdir(parents=True, exist_ok=True)

  with pq.ParquetWriter(TARGET_PARQUET_PATH, schema) as writer, timer("ETL process"):
    print("Processing data in batches...")
    print(f"Reading source CSV: {SOURCE_CSV_PATH}")
    print(f"Writing parquet output: {TARGET_PARQUET_PATH}")
    reader = pd.read_csv(SOURCE_CSV_PATH, chunksize=batch_size)

    for i, chunk in enumerate(reader):
      pop_songs = process_chunk(chunk)
      pop_songs = pa.Table.from_pandas(pop_songs, schema=schema)
      writer.write_table(pop_songs)
    else:
      print("Finished processing all batches.")
    writer.close()


if __name__ == '__main__':
  main()
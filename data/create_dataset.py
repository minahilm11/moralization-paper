from read_data import *
from read_mps import *
from time import time
from pathlib import Path


if __name__ == "__main__":

    start = time()

    PROJECT_ROOT = Path(__file__).resolve().parents[1]

    files = parse_sessions(start_year=1991, end_year=2023, root_dir=os.path.join(PROJECT_ROOT, "data", "records", "data"))
    sessions =[]

    # make sure only xml-files are read
    files= [file for file in files if file[0].endswith(".xml")]
    print(f"Parsing {len(files)} files...")
    # parse xml files and create Session, Protocol and Speech dataclasses
    for session in generate_protocol_MP(sessions=files):
        sessions.append(session)
    print("Protocols have been parsed.")

    # create dataclasses for parties
    parties = read_party_names(os.path.join(PROJECT_ROOT, "data", "persons", "data", "party.csv"))

    party_lookup = {party.id: party for party in parties}

    # create dataclasses for all representatives
    people = read_names(
        os.path.join(PROJECT_ROOT, "data", "persons", "data", "name.csv"),
        os.path.join(PROJECT_ROOT, "data", "persons", "data", "party_affiliation.csv"),
        party_lookup
        )

    rep_lookup = {rep.id: rep for rep in people}

    all_speeches = [
    speech
    for session in sessions
    for protocol in session.protocols
    for speech in protocol.speeches
    ]

    print(f"{len(all_speeches)} speeches")
    add_speeches(people, all_speeches)
    add_party_to_speeches(all_speeches, rep_lookup)


    end = time()
    print(f"Program took {end-start} to run.")


    ####### SAVE AS PARQUET ############
    speeches_df = pd.DataFrame(
        {
            "speaker_id" : [speech.who for speech in all_speeches],
            "speaker_name" : [rep_lookup.get(speech.who).name for speech in all_speeches],
            "text" : [speech.text for speech in all_speeches],
            "date" : [str(speech.speech_date) for speech in all_speeches],
            "party_id" : [speech.party.id for speech in all_speeches],
            "party_name" : [speech.party.name for speech in all_speeches]
        }
    )

    print(speeches_df.head())
    print(len(speeches_df))

    filename = PROJECT_ROOT / "data" / "speeches_1990_2022.parquet"

    chunk_size = 10_000

    # save in chunks
    for i in range(0, len(speeches_df), chunk_size):
        chunk = speeches_df.iloc[i:i+chunk_size]
        if i == 0:
            chunk.to_parquet(filename, compression="snappy", engine='fastparquet', index=True)
        else:
            chunk.to_parquet(filename, compression="snappy", engine='fastparquet', index=False, append=True)
        del chunk

    # check that all data is saved
    df = pd.read_parquet(filename)
    print(len(df))
